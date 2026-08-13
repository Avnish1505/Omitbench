"""
Experiment harness v3.

Fixes three defects in pilot2 that would have been fatal in review:

  1. BASE RATE. pilot2 emitted 3 mutations + 1 clean per instance -> 75% of
     records were OMITTED. At that base rate "flag everything" scores F1 0.82
     and beats every real detector. Fix: score EVERY new symbol in the patch
     as a separate requirement. In a mutated variant exactly one requirement
     is omitted and the rest are implemented, which is what reality looks like.
     Negatives now come for free and precision becomes measurable.

  2. NON-INDEPENDENCE. The variants of one base instance share a repo, a patch
     and a symbol. Bootstrapping over records pretends they are independent and
     produces CIs that are far too narrow. Fix: cluster bootstrap over iid.

  3. F1 ONLY. F1 ignores true negatives and moves with the base rate. Added MCC,
     which uses all four cells and is comparable across base rates.

Detector contract is unchanged and is the load-bearing invariant of the project:

    detect(spec, requirements, before, after, ctx) -> {req: verdict}

No gold patch. No mutation label. See tests/test_leakage.py.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict

from omitbench import corpus as C
from omitbench import mutate as M
from omitbench import detectors as P

MAX_REQS = 8  # cap requirements per instance so one huge patch cannot dominate


# --------------------------------------------------------------------------
# cheap commit pre-filter -- snapshot_repo costs ~1.5s, so never call it on a
# commit that cannot possibly yield an instance
# --------------------------------------------------------------------------

def promising(repo: str, sha: str) -> bool:
    try:
        diff = C._git(repo, "show", "--format=", "--unified=0", sha)
    except Exception:
        return False
    if len(diff) > 200_000:
        return False
    touches_py = added_def = False
    for line in diff.splitlines():
        if line.startswith("+++ b/") and line.endswith(".py"):
            touches_py = True
        elif line.startswith("+") and not line.startswith("+++"):
            s = line[1:].lstrip()
            if s.startswith(("def ", "class ", "async def ")):
                added_def = True
    return touches_py and added_def


def build_base(repo, sha):
    """
    Lean v3 instance. pilot2 reused build_instance(), which still ran v1
    hunk-deletion logic: it required a >=20-char commit message, 3-40 hunks,
    and 1-12 identifiers extractable from the commit text. v3 derives
    requirements from new SYMBOLS, so those filters rejected 73% of otherwise
    usable commits for reasons that no longer apply.
    """
    parent = C._git(repo, "rev-parse", f"{sha}^").strip()
    if not parent:
        return None
    diff = C._git(repo, "diff", "--unified=3", parent, sha, "--", "*.py")
    hunks = C.parse_patch(diff)
    if not (1 <= len(hunks) <= 60):
        return None
    paths = sorted({h.path for h in hunks})
    before = {}
    for p in paths:
        s = C._git(repo, "show", f"{parent}:{p}")
        if not s:
            return None
        before[p] = s.split("\n")
    after = C.apply_hunks(before, hunks)
    for ls in after.values():
        try:
            ast.parse("\n".join(ls))
        except SyntaxError:
            return None
    spec = C._git(repo, "log", "-1", "--format=%B", sha).strip()
    iid = hashlib.sha1(f"{repo}:{sha}".encode()).hexdigest()[:12]
    return iid, os.path.basename(repo), spec, parent, before, after


def build(repos, per_repo, seed, scan_cap=1200):
    rng = random.Random(seed)
    insts = []
    for repo in repos:
        name = os.path.basename(repo)
        commits = C.list_commits(repo, limit=scan_cap)
        rng.shuffle(commits)
        got = 0
        for sha in commits:
            if got >= per_repo:
                break
            if not promising(repo, sha):
                continue
            try:
                b = build_base(repo, sha)
                if b is None:
                    continue
                iid, rname, spec, parent, patch_before, patch_after = b
                before = C.snapshot_repo(repo, parent)
                if not before:
                    continue
                full_after = dict(before)
                full_after.update(patch_after)

                syms = M.new_symbols(patch_before, patch_after)
                if not syms:
                    continue

                # Which mutations are applicable per symbol. UNWIRED needs an
                # internal call site; ABSENT and STUB do not. pilot2 required
                # call sites for ALL mutations, discarding 2/3 of symbols and
                # excluding public API -- exactly where real agents omit work.
                cand = []
                for path, sym in syms[:MAX_REQS]:
                    d = M._defs("\n".join(full_after.get(path, []))).get(sym)
                    excl = (d[0], d[1]) if d else None
                    wired = any(
                        sym in M.call_targets("\n".join(ls))
                        and M.call_lines("\n".join(ls), sym,
                                         excl if p2 == path else None)
                        for p2, ls in full_after.items())
                    cand.append((path, sym, wired))
                if not cand:
                    continue

                reqs = [f"{p}::{s}" for p, s, _ in cand]
                path, target, _ = rng.choice(cand)

                variants = [("CLEAN", full_after)]
                for cls, fn in M.MUTATIONS.items():
                    try:
                        mutated = fn(full_after, path, target)
                    except Exception:
                        mutated = None
                    if mutated is not None:
                        variants.append((cls, mutated))
                if len(variants) < 2:
                    continue

                insts.append((iid, rname, spec, before, f"{path}::{target}",
                              reqs, variants))
                got += 1
            except Exception:
                continue
        print(f"  {name:12s} {got:4d} instances", file=sys.stderr)
    return insts


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def cells(rows):
    tp = fp = fn = tn = 0
    for pred, gold in rows:
        p, g = pred == "OMITTED", gold == "OMITTED"
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif not p and g:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def metrics(rows):
    tp, fp, fn, tn = cells(rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "mcc": mcc,
            "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def cluster_bootstrap(by_iid, key, n=1000, seed=0):
    """Resample INSTANCES, not records. Records within an instance are dependent."""
    rng = random.Random(seed)
    iids = list(by_iid)
    if not iids:
        return (0.0, 0.0)
    vals = []
    for _ in range(n):
        rows = []
        for _ in iids:
            rows.extend(by_iid[rng.choice(iids)])
        vals.append(metrics(rows)[key])
    vals.sort()
    return (vals[int(0.025 * n)], vals[int(0.975 * n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--per-repo", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--scan-cap", type=int, default=1200)
    ap.add_argument("--out", default="results/runs_v3.jsonl")
    a = ap.parse_args()

    recs = []
    for seed in a.seeds:
        for repo in a.repos:
            insts = build([repo], a.per_repo, seed, a.scan_cap)
            for iid, rname, spec, before, target, reqs, variants in insts:
                for cls, after in variants:
                    for dname, fn in P.DETECTORS.items():
                        try:
                            preds = fn(spec, reqs, before, after, {})
                        except Exception:
                            continue
                        for r in reqs:
                            gold = "OMITTED" if (cls != "CLEAN" and r == target) \
                                   else "IMPLEMENTED"
                            recs.append({
                                "iid": iid, "repo": rname, "req": r,
                                "target": target, "mutation": cls,
                                "detector": dname, "pred": preds.get(r, "IMPLEMENTED"),
                                "gold": gold, "seed": seed,
                            })
            # full-repo snapshots and AST memos are the memory hog; a 9-repo
            # run OOMs without this. Instances are dropped once scored.
            del insts
            for memo in (M._TREE_MEMO, M._DEF_MEMO, M._CALL_MEMO, M._TARGETS_MEMO):
                memo.clear()
            gc.collect()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    n_inst = len({r["iid"] for r in recs})
    pos = sum(r["gold"] == "OMITTED" for r in recs) / max(1, len(recs)) / \
          max(1, len(P.DETECTORS)) * len(P.DETECTORS)
    print(f"\n{len(recs)} records | {n_inst} instances | "
          f"{len({r['repo'] for r in recs})} repos | base rate {pos:.1%}\n",
          file=sys.stderr)

    # ---- headline table -------------------------------------------------
    print(f"{'detector':28} {'P':>5} {'R':>5} {'F1':>5} {'MCC':>6} "
          f"{'MCC 95% CI':>16} {'FPR':>6}")
    print("-" * 76)
    for d in P.DETECTORS:
        rows = [(r["pred"], r["gold"]) for r in recs if r["detector"] == d]
        if not rows:
            continue
        m = metrics(rows)
        by_iid = defaultdict(list)
        for r in recs:
            if r["detector"] == d:
                by_iid[r["iid"]].append((r["pred"], r["gold"]))
        lo, hi = cluster_bootstrap(by_iid, "mcc")
        print(f"{d:28} {m['precision']:>5.2f} {m['recall']:>5.2f} "
              f"{m['f1']:>5.2f} {m['mcc']:>6.3f} [{lo:>6.3f},{hi:>6.3f}] "
              f"{m['fpr']:>6.3f}")

    # ---- recall by mutation class: where the method works and where it dies
    print(f"\n{'detector':28}" + "".join(f"{c:>10}" for c in M.MUTATIONS))
    print("-" * 62)
    for d in P.DETECTORS:
        line = f"{d:28}"
        for cls in M.MUTATIONS:
            rows = [(r["pred"], r["gold"]) for r in recs
                    if r["detector"] == d and r["mutation"] == cls
                    and r["gold"] == "OMITTED"]
            rec = metrics(rows)["recall"] if rows else float("nan")
            line += f"{rec:>10.2f}"
        print(line)


if __name__ == "__main__":
    main()
