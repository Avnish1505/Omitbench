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

import argparse
import glob
import json
import math
import random
import sys
from collections import defaultdict

sys.path.insert(0, ".")

ORDER = ["B0 flag-nothing", "B1 flag-everything", "B3 line-grep (no AST)",
         "B4 LLM judge (gpt-oss-120b)", "B5 LLM judge (mid-tier)",
         "B6 LLM judge (NVIDIA NIM)",
         "P1 defined", "P2 defined+reachable", "P3 defined+reachable+body"]

# B4/B5/B6 are the three LLM-judge baselines (omitbench/judges.py). A single
# judge cannot support "LLM judges are structurally weak at omission" -- see
# that module's docstring -- so every comparison against P1 below is run for
# all three, not just one.
JUDGE_NAMES = ["B4 LLM judge (gpt-oss-120b)", "B5 LLM judge (mid-tier)",
               "B6 LLM judge (NVIDIA NIM)"]


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


def analyze_synthetic():
    """Everything this module printed before --source existed. Body is
    unchanged from the original main() -- only extracted into a named,
    reusable function so --source real/both can pull its `summary` dict
    for the side-by-side MCC column without re-running this table."""
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

    # ---- P1 vs each LLM judge: the specific comparison this repo exists to
    # make ----------------------------------------------------------------
    # RELATED.md's wedge is "LLM judges are structurally weak at omission
    # specifically" -- reported on its own rather than folded into the
    # generic vs-B3 loop above, so it can't get buried under other rows.
    # Run for EVERY judge present, not just one: a single judge losing to P1
    # would not support the claim (a reviewer's first objection is "you
    # picked a weak model"), so all of B4/B5/B6 are reported here. Same
    # paired resamples (B), so each is directly comparable to the others.
    p1_d = "P1 defined"
    judge_ds = [d for d in JUDGE_NAMES if d in dets]
    if judge_ds and p1_d in dets:
        print()
        for judge_d in judge_ds:
            diffs = []
            for samp in B:
                a = metrics([x for i in samp for x in per_det[p1_d][i]])["mcc"]
                b = metrics([x for i in samp for x in per_det[judge_d][i]])["mcc"]
                diffs.append(a - b)
            diffs.sort()
            lo, hi = diffs[25], diffs[974]
            obs = summary[p1_d]["mcc"] - summary[judge_d]["mcc"]
            sig = "" if lo <= 0 <= hi else "  *"
            print(f"  {p1_d:28} vs {judge_d:28} {obs:>+7.3f}  "
                  f"[{lo:>+6.3f},{hi:>+6.3f}]{sig}")
        print("  (this is the central claim RELATED.md exists to test -- "
              "read it plainly, do not re-run to change it)")

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

    return summary


# --------------------------------------------------------------------------
# real-trajectory analysis (TASKS.md T4)
# --------------------------------------------------------------------------

def _synthetic_mcc_summary() -> dict:
    """Lightweight synthetic MCC lookup for the --source real (not --both)
    path, where analyze_synthetic() never ran and never printed anything.
    Reuses load()/metrics() -- no new logic, no duplicate bootstrap."""
    recs = load()
    dets = [d for d in ORDER if any(r["detector"] == d for r in recs)]
    out = {}
    for d in dets:
        rows = [(r["pred"], r["gold"]) for r in recs if r["detector"] == d]
        out[d] = metrics(rows)
    return out


def analyze_real(synthetic_summary: dict | None = None) -> None:
    from omitbench import real as R

    print("=" * 78)
    print("REAL-TRAJECTORY ANALYSIS  (data/real/*.json, TASKS.md T4)")
    print("=" * 78)

    try:
        instances = R.load_real()
    except R.RealSchemaError as e:
        print(f"\nERROR loading real trajectories:\n{e}", file=sys.stderr)
        raise SystemExit(1)

    # ---- gold-label distribution FIRST, before any score is computed ----
    all_golds = [g for inst in instances for g in inst.gold.values()]
    n = len(all_golds)
    n_omitted = all_golds.count("OMITTED")
    n_implemented = n - n_omitted
    print(f"\n{len(instances)} trajectories | {n} requirements")
    print(f"GOLD LABEL DISTRIBUTION: {n_implemented} IMPLEMENTED, "
          f"{n_omitted} OMITTED "
          f"({(n_omitted / n if n else 0):.1%} positive rate)")
    for inst in instances:
        n_om = list(inst.gold.values()).count("OMITTED")
        print(f"  {inst.iid:10} {inst.repo:12} {len(inst.reqs)} reqs, "
              f"{n_om} OMITTED")

    if n == 0:
        print("\nNo requirements to score.")
        return

    if n_omitted == 0:
        print(f"\nWARNING: 0 of {n} gold labels are OMITTED across all "
              f"{len(instances)} trajectories. The positive class is EMPTY.")
        print("Recall and F1 below are a 0/0 fallback (reported as 0.00 by "
              "convention), NOT evidence of detector skill.")
        print("MCC is degenerate for the same reason: with zero OMITTED "
              "labels, TP+FN=0 forces the MCC denominator to 0 regardless "
              "of what any detector predicts, so every detector's MCC "
              "reads 0.000 here no matter how it behaves.")
        print("Precision is ALSO uninformative here, for a different "
              "reason: TP=0 whenever there are no positive gold labels, so "
              "precision = 0/(0+FP) reads 0.00 for every detector "
              "regardless of how many false positives it raises -- do not "
              "read it as 'perfect precision'. FPR is the ONLY well-defined "
              "signal in the table below (there IS a real negative class, "
              "and detectors do differ in how many of it they flag).")
    elif n_omitted < 5:
        print(f"\nNOTE: only {n_omitted} OMITTED example(s) across "
              f"{len(instances)} trajectories -- recall below is estimated "
              f"from a very small positive class; treat it as indicative, "
              f"not precise.")

    recs = R.score(instances)
    dets = sorted({r["detector"] for r in recs},
                  key=lambda d: (ORDER.index(d) if d in ORDER else len(ORDER)))
    per_det = {d: defaultdict(list) for d in dets}
    for r in recs:
        per_det[r["detector"]][r["iid"]].append((r["pred"], r["gold"]))

    iids = sorted({inst.iid for inst in instances})
    print(f"\nn = {len(iids)} instances -- SMALL SAMPLE; cluster-bootstrap "
          f"CI below will be wide and should be read as indicative, not "
          f"precise.\n")

    if synthetic_summary is None:
        try:
            synthetic_summary = _synthetic_mcc_summary()
        except FileNotFoundError:
            synthetic_summary = {}

    B = boot_indices(iids, 1000, 0)
    header = (f"{'detector':28}{'P':>6}{'R':>6}{'F1':>6}{'MCC':>7}"
              f"{'  MCC 95% CI':>18}{'FPR':>7}")
    if synthetic_summary:
        header += f"{'synthetic MCC':>16}"
    print(header)
    print("-" * len(header))
    for d in dets:
        rows = [x for i in iids for x in per_det[d][i]]
        m = metrics(rows)
        vals = sorted(metrics([x for i in samp for x in per_det[d][i]])["mcc"]
                      for samp in B)
        lo, hi = vals[25], vals[974]
        line = (f"{d:28}{m['precision']:>6.2f}{m['recall']:>6.2f}{m['f1']:>6.2f}"
                f"{m['mcc']:>7.3f}  [{lo:>6.3f},{hi:>6.3f}]{m['fpr']:>7.3f}")
        if synthetic_summary:
            syn = synthetic_summary.get(d)
            line += f"{syn['mcc']:>16.3f}" if syn else f"{'n/a':>16}"
        print(line)

    if n_omitted == 0:
        print("\n(Re-read the WARNING above: MCC/recall/F1/precision in "
              "this table are 0-by-construction, not a demonstrated "
              "result. FPR is the only column with signal.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "real", "both"],
                     default="synthetic")
    a = ap.parse_args()

    synthetic_summary = None
    if a.source in ("synthetic", "both"):
        synthetic_summary = analyze_synthetic()
    if a.source in ("real", "both"):
        if a.source == "both":
            print()
        analyze_real(synthetic_summary)


if __name__ == "__main__":
    main()
