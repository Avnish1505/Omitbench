"""
Pilot v2. Requirements are SYMBOLS; omission is injected by mutation class.

Detector contract is unchanged: sees (spec, requirements, R_before, R_after).
Never the gold patch, never the mutation label.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import statistics
import sys

from omitbench import corpus as C
from omitbench import evidence as E
from omitbench import mutate as M


# --------------------------------------------------------------------------
# detectors (symbol-level)
# --------------------------------------------------------------------------

def d_grep(spec, reqs, before, after, ctx):
    """B3: does the symbol appear anywhere in the changed lines?"""
    blob = "\n".join(E._changed_lines(before, after))
    return {r: ("IMPLEMENTED" if r in blob else "OMITTED") for r in reqs}


def d_defined(spec, reqs, before, after, ctx):
    """E1: is the symbol defined in R_after? (catches ABSENT only)"""
    defined = set()
    for p, ls in after.items():
        defined |= set(M._defs("\n".join(ls)))
    return {r: ("IMPLEMENTED" if r in defined else "OMITTED") for r in reqs}


def d_reachable(spec, reqs, before, after, ctx):
    """E1+E3: defined AND called from outside its own body. Catches UNWIRED."""
    res = {}
    for r in reqs:
        ok = False
        for p, ls in after.items():
            src = "\n".join(ls)
            d = M._defs(src).get(r)
            if d is None:
                continue
            excl = (d[0], d[1])
            for p2, ls2 in after.items():
                src2 = "\n".join(ls2)
                if r not in M.call_targets(src2):
                    continue
                e = excl if p2 == p else None
                if M.call_lines(src2, r, e):
                    ok = True
                    break
            if ok:
                break
        res[r] = "IMPLEMENTED" if ok else "OMITTED"
    return res


def _body_trivial(src: str, name: str) -> bool | None:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            body = [s for s in n.body if not (isinstance(s, ast.Expr)
                                              and isinstance(s.value, ast.Constant)
                                              and isinstance(s.value.value, str))]
            if not body:
                return True
            if len(body) == 1:
                s = body[0]
                if isinstance(s, ast.Pass):
                    return True
                if isinstance(s, ast.Raise):
                    exc = s.exc
                    nm = getattr(exc, "id", None) or getattr(getattr(exc, "func", None), "id", None)
                    if nm == "NotImplementedError":
                        return True
                if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis:
                    return True
            return False
    return None


def d_full(spec, reqs, before, after, ctx):
    """P: defined AND reachable AND body non-trivial. The proposed method."""
    reach = d_reachable(spec, reqs, before, after, ctx)
    defd = d_defined(spec, reqs, before, after, ctx)
    res = {}
    for r in reqs:
        if defd[r] == "OMITTED" or reach[r] == "OMITTED":
            res[r] = "OMITTED"
            continue
        trivial = None
        for p, ls in after.items():
            t = _body_trivial("\n".join(ls), r)
            if t is not None:
                trivial = t
                break
        res[r] = "OMITTED" if trivial else "IMPLEMENTED"
    return res


def d_none(spec, reqs, before, after, ctx):
    """B0: flag nothing. Its accuracy is the base rate -- that is the point."""
    return {r: "IMPLEMENTED" for r in reqs}


def d_all(spec, reqs, before, after, ctx):
    """B1: flag everything. Recall 1.0, precision = base rate."""
    return {r: "OMITTED" for r in reqs}


DETECTORS = {
    "B0 flag-nothing": d_none,
    "B1 flag-everything": d_all,
    "B3 line-grep (no AST)": d_grep,
    "P1 defined?": d_defined,
    "P2 defined+reachable": d_reachable,
    "P3 defined+reachable+body": d_full,
}


# --------------------------------------------------------------------------
# instance building
# --------------------------------------------------------------------------

def build(repos, target, seed):
    rng = random.Random(seed)
    per = max(1, target // len(repos))
    insts = []
    for repo in repos:
        commits = C.list_commits(repo, limit=2000)
        rng.shuffle(commits)
        got = 0
        for sha in commits:
            if got >= per:
                break
            try:
                base = C.build_instance(repo, sha, rng, drop_frac_range=(0.0, 0.0))
            except Exception:
                continue
            if base is None:
                continue
            # v2a: FULL repo snapshot -- call-graph over a 2-file slice is blind
            parent = C._git(repo, "rev-parse", f"{sha}^").strip()
            before = C.snapshot_repo(repo, parent)
            if not before:
                continue
            full_after = dict(before)
            full_after.update(base.after)                  # apply the WHOLE gold patch
            base.before = before

            syms = M.new_symbols(base.before, base.after)  # new symbols from the patch
            if not syms:
                continue
            # CORPUS ASSUMPTION: only symbols the reference solution actually wires up
            # internally. Excludes public API called only by users. Limits external
            # validity -- stated in PILOT_ASSUMPTIONS.md, not hidden.
            wired = []
            for path, name in syms:
                d = M._defs("\n".join(full_after.get(path, []))).get(name)
                excl = (d[0], d[1]) if d else None
                if any(name in M.call_targets("\n".join(ls))
                       and M.call_lines("\n".join(ls), name, excl if p2 == path else None)
                       for p2, ls in full_after.items()):
                    wired.append((path, name))
            if not wired:
                continue
            path, name = rng.choice(wired)

            variants = []
            for cls, fn in M.MUTATIONS.items():
                try:
                    mutated = fn(full_after, path, name)
                except Exception:
                    mutated = None
                if mutated is not None:
                    variants.append((cls, mutated))
            if len(variants) < 2:
                continue

            # one clean control: the untouched full patch (requirement IS implemented)
            variants.append(("CLEAN", full_after))
            insts.append((base, name, variants))
            got += 1
        print(f"  {os.path.basename(repo):12s} {got:4d} base instances", file=sys.stderr)
    return insts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    print("building v2 corpus...", file=sys.stderr)
    insts = build(a.repos, a.target, a.seed)
    if not insts:
        sys.exit("no instances")

    os.makedirs(a.out, exist_ok=True)
    recs = []
    for base, name, variants in insts:
        for cls, after in variants:
            gold = "IMPLEMENTED" if cls == "CLEAN" else "OMITTED"
            for dname, fn in DETECTORS.items():
                pred = fn(base.spec, [name], base.before, after, {})[name]
                recs.append({
                    "iid": base.iid, "repo": base.repo, "symbol": name,
                    "mutation": cls, "detector": dname,
                    "pred": pred, "gold": gold, "seed": a.seed,
                })

    with open(f"{a.out}/runs_v2.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    n_pos = sum(r["gold"] == "OMITTED" for r in recs if r["detector"] == "B0 flag-nothing")
    n_all = sum(1 for r in recs if r["detector"] == "B0 flag-nothing")

    print()
    print("=" * 78)
    print("OMITBENCH PILOT v2  (frozen before any optimisation)")
    print("=" * 78)
    print(f"base instances     : {len(insts)}")
    print(f"eval points        : {n_all}")
    print(f"omission base rate : {n_pos/n_all:.3f}")
    print(f"mutation classes   : {sorted({r['mutation'] for r in recs})}")
    print()
    print(f"{'detector':28s} {'prec':>6s} {'recall':>7s} {'F1':>7s} {'acc':>7s}")
    print("-" * 78)
    for d in DETECTORS:
        rows = [r for r in recs if r["detector"] == d]
        s = E.score([r["pred"] for r in rows], [r["gold"] for r in rows])
        print(f"{d:28s} {s['precision']:6.3f} {s['recall']:7.3f} {s['f1']:7.3f} {s['accuracy']:7.3f}")

    print()
    print("RECALL BY OMISSION CLASS  <-- the whole point of the benchmark")
    classes = [c for c in ["ABSENT", "UNWIRED", "STUB"]
               if any(r["mutation"] == c for r in recs)]
    print(f"{'detector':28s} " + " ".join(f"{c:>9s}" for c in classes))
    print("-" * 78)
    for d in DETECTORS:
        cells = []
        for c in classes:
            rows = [r for r in recs if r["detector"] == d and r["mutation"] == c]
            s = E.score([r["pred"] for r in rows], [r["gold"] for r in rows])
            cells.append(f"{s['recall']:9.3f}")
        print(f"{d:28s} " + " ".join(cells))

    print()
    print("FALSE POSITIVE RATE on CLEAN (correct) patches -- a noisy CI gate gets uninstalled")
    for d in DETECTORS:
        rows = [r for r in recs if r["detector"] == d and r["mutation"] == "CLEAN"]
        fpr = sum(r["pred"] == "OMITTED" for r in rows) / max(1, len(rows))
        print(f"  {d:28s} FPR={fpr:.3f}")

    print(f"\nwrote {len(recs)} records -> {a.out}/runs_v2.jsonl")


if __name__ == "__main__":
    main()
