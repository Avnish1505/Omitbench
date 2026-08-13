"""
Reads results/shards/*.jsonl and produces every number that goes in the README.
Nothing in the README is hand-typed; it all comes from here.

Three things this reports that pilot2 did not:

  MCC          F1 ignores true negatives and moves with the base rate. At
               pilot2's 75% positive rate "flag everything" scored F1 0.82 and
               beat every real detector. MCC uses all four cells and is 0 for
               any non-discriminating rule, which is what a baseline should get.

  CLUSTER CI   The 4 variants x N requirements of one instance share a repo, a
               patch and a symbol. Bootstrapping over records treats them as
               independent and yields CIs that are far too narrow. We resample
               INSTANCES.

  PAIRED TEST  Detectors are run on identical inputs, so comparing them with
               independent CIs throws away the pairing. We report a paired
               cluster bootstrap of the MCC difference; if that interval
               contains 0, the ordering is not supported.
"""

from __future__ import annotations

import glob
import json
import math
import random
from collections import defaultdict

ORDER = ["B0 flag-nothing", "B1 flag-everything", "B3 line-grep (no AST)",
         "P1 defined", "P2 defined+reachable", "P3 defined+reachable+body"]


def load():
    recs = []
    for p in sorted(glob.glob("results/shards/*.jsonl")):
        with open(p) as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
    return recs


def metrics(rows):
    tp = fp = fn = tn = 0
    for pred, gold in rows:
        p, g = pred == "OMITTED", gold == "OMITTED"
        tp += p and g
        fp += p and not g
        fn += (not p) and g
        tn += (not p) and (not g)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "mcc": mcc,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def boot_indices(iids, n, seed):
    rng = random.Random(seed)
    return [[rng.choice(iids) for _ in iids] for _ in range(n)]


def main():
    recs = load()
    iids = sorted({r["iid"] for r in recs})
    repos = sorted({r["repo"] for r in recs})
    dets = [d for d in ORDER if any(r["detector"] == d for r in recs)]

    per_det = {d: defaultdict(list) for d in dets}
    for r in recs:
        if r["detector"] in per_det:
            per_det[r["detector"]][r["iid"]].append((r["pred"], r["gold"]))

    n_rows = len(recs) // max(1, len(dets))
    pos = sum(1 for r in recs if r["detector"] == dets[0] and r["gold"] == "OMITTED")
    print(f"records {len(recs)} | instances {len(iids)} | repos {len(repos)} "
          f"({', '.join(repos)})")
    print(f"requirements scored per detector {n_rows} | base rate "
          f"{pos / max(1, n_rows):.1%} OMITTED\n")

    B = boot_indices(iids, 1000, 0)

    print(f"{'detector':28}{'P':>6}{'R':>6}{'F1':>6}{'MCC':>7}{'  MCC 95% CI':>18}{'FPR':>7}")
    print("-" * 78)
    summary = {}
    for d in dets:
        rows = [x for i in iids for x in per_det[d][i]]
        m = metrics(rows)
        vals = sorted(metrics([x for i in samp for x in per_det[d][i]])["mcc"]
                      for samp in B)
        lo, hi = vals[25], vals[974]
        summary[d] = m
        print(f"{d:28}{m['precision']:>6.2f}{m['recall']:>6.2f}{m['f1']:>6.2f}"
              f"{m['mcc']:>7.3f}  [{lo:>6.3f},{hi:>6.3f}]{m['fpr']:>7.3f}")

    # ---- recall by mutation class -------------------------------------
    muts = ["ABSENT", "UNWIRED", "STUB"]
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["gold"] == "OMITTED":
            by[r["detector"]][r["mutation"]].append((r["pred"], r["gold"]))
    print(f"\nRECALL BY OMISSION CLASS  (n per cell in parens)")
    print(f"{'detector':28}" + "".join(f"{c:>16}" for c in muts))
    print("-" * 76)
    for d in dets:
        line = f"{d:28}"
        for c in muts:
            rows = by[d][c]
            v = metrics(rows)["recall"] if rows else float("nan")
            line += f"{v:>10.2f} ({len(rows):>3})"
        print(line)

    # ---- paired comparison against the strongest baseline --------------
    print("\nPAIRED cluster bootstrap of MCC difference (95% CI). "
          "Interval containing 0 => ordering unsupported.")
    base = "B3 line-grep (no AST)"
    for d in dets:
        if d.startswith("B"):
            continue
        diffs = []
        for samp in B:
            a = metrics([x for i in samp for x in per_det[d][i]])["mcc"]
            b = metrics([x for i in samp for x in per_det[base][i]])["mcc"]
            diffs.append(a - b)
        diffs.sort()
        lo, hi = diffs[25], diffs[974]
        obs = summary[d]["mcc"] - summary[base]["mcc"]
        sig = "" if lo <= 0 <= hi else "  *"
        print(f"  {d:28} vs {base:24} {obs:>+7.3f}  [{lo:>+6.3f},{hi:>+6.3f}]{sig}")

    # ---- per-repo stability of the best detector ------------------------
    best = max((d for d in dets if d.startswith("P")),
               key=lambda d: summary[d]["mcc"])
    print(f"\nPER-REPO MCC for {best} (generalisation check)")
    for repo in repos:
        rows = [(r["pred"], r["gold"]) for r in recs
                if r["detector"] == best and r["repo"] == repo]
        m = metrics(rows)
        print(f"  {repo:16}{m['mcc']:>7.3f}   P {m['precision']:.2f}  "
              f"R {m['recall']:.2f}   n={len(rows)}")


if __name__ == "__main__":
    main()
