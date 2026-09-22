"""
Calibration of the B7 Jev judges. Reads the `p_implemented` field that
scripts/run_jev.py writes into results/shards/llm_judge_b7_*.jsonl and
reports, per B7 variant:

  Brier score       mean (p_omit - y)^2, y = 1 if gold is OMITTED. Lower is
                    better. Reported with a cluster-bootstrap CI over iid
                    (CLAUDE.md anti-pattern 3: never bootstrap over records).
  Brier skill       1 - Brier / Brier(base-rate forecaster). > 0 means the
                    probabilities beat always predicting the corpus base rate.
  ECE               expected calibration error, 10 equal-width bins.
  AUROC             threshold-free discrimination, cluster-bootstrap CI.
  reliability       per bin: n, mean predicted P(omitted), observed rate.
  per class         the same, split by mutation class.
  abstain band      pre-registered: treat 0.3 <= p_omit <= 0.7 as "ask a
                    human" (the band TypeSafe's Noul docs suggest routing to
                    review). Reports coverage and MCC on the confident rest.

For reference it also scores every HARD-LABEL detector present in the shards
(B5, P1, ...) as a 0/1 forecaster. A hard label's Brier score is just its
error rate, so if B7's probabilities carry information, B7's Brier should
beat its own thresholded verdicts, and the comparison against B5 says
whether probabilities buy anything over a stronger model's bare labels.

Nothing here is hand-typed into the README; cite this script's output.
"""

from __future__ import annotations

import glob
import json
import math
import random
import sys
from collections import defaultdict

sys.path.insert(0, ".")

BINS = 10
N_BOOT = 2000
ABSTAIN = (0.3, 0.7)
OUT = "results/calibration_b7.json"
REFERENCE = ["B5 LLM judge (mid-tier)", "B4 LLM judge (gpt-oss-120b)", "P1 defined"]


def load():
    recs = []
    for p in sorted(glob.glob("results/shards/*.jsonl")):
        with open(p) as f:
            recs.extend(json.loads(line) for line in f if line.strip())
    return recs


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else float("nan")


def auroc(pairs):
    pos = [p for p, y in pairs if y == 1]
    neg = [p for p, y in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    # rank-based (Mann-Whitney), ties count half
    allv = sorted((p, y) for p, y in pairs)
    i = 0
    rank_sum = 0.0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def reliability(pairs):
    bins = [[] for _ in range(BINS)]
    for p, y in pairs:
        bins[min(BINS - 1, int(p * BINS))].append((p, y))
    rows, ece, n = [], 0.0, len(pairs)
    for b, xs in enumerate(bins):
        if not xs:
            rows.append({"bin": f"{b/BINS:.1f}-{(b+1)/BINS:.1f}", "n": 0})
            continue
        mp = sum(p for p, _ in xs) / len(xs)
        obs = sum(y for _, y in xs) / len(xs)
        ece += len(xs) / n * abs(mp - obs)
        rows.append({"bin": f"{b/BINS:.1f}-{(b+1)/BINS:.1f}", "n": len(xs),
                     "mean_pred": round(mp, 4), "observed": round(obs, 4)})
    return rows, ece


def mcc(pairs, thr=0.5):
    tp = fp = fn = tn = 0
    for p, y in pairs:
        pr = p >= thr
        tp += pr and y
        fp += pr and not y
        fn += (not pr) and y
        tn += (not pr) and not y
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0


def cluster_ci(by_iid, stat, seed=0):
    iids = sorted(by_iid)
    rng = random.Random(seed)
    vals = []
    for _ in range(N_BOOT):
        pairs = [x for _ in iids for x in by_iid[rng.choice(iids)]]
        v = stat(pairs)
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    if not vals:
        return [float("nan"), float("nan")]
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def summarize(rows, probabilistic: bool):
    by_iid = defaultdict(list)
    for r in rows:
        y = 1 if r["gold"] == "OMITTED" else 0
        if probabilistic:
            p = 1.0 - r["p_implemented"]
        else:
            p = 1.0 if r["pred"] == "OMITTED" else 0.0
        by_iid[r["iid"]].append((p, y))
    pairs = [x for xs in by_iid.values() for x in xs]
    base = sum(y for _, y in pairs) / len(pairs)
    b = brier(pairs)
    b_ref = brier([(base, y) for _, y in pairs])
    rel, ece = reliability(pairs)
    out = {
        "n": len(pairs), "iids": len(by_iid), "base_rate": round(base, 4),
        "brier": round(b, 4), "brier_ci": [round(x, 4) for x in cluster_ci(by_iid, brier)],
        "brier_skill": round(1 - b / b_ref, 4) if b_ref else None,
        "ece": round(ece, 4),
        "auroc": round(auroc(pairs), 4),
        "auroc_ci": [round(x, 4) for x in cluster_ci(by_iid, auroc)],
        "mcc_at_0.5": round(mcc(pairs), 4),
    }
    if probabilistic:
        out["reliability"] = rel
        lo, hi = ABSTAIN
        conf = [(p, y) for p, y in pairs if p < lo or p > hi]
        out["abstain_band"] = {
            "band": [lo, hi], "coverage": round(len(conf) / len(pairs), 4),
            "mcc_on_confident": round(mcc(conf), 4) if conf else None,
            "error_rate_on_confident": round(
                sum((p >= 0.5) != bool(y) for p, y in conf) / len(conf), 4) if conf else None,
        }
    return out


def main():
    recs = load()
    b7_ids = sorted({r["detector"] for r in recs if r["detector"].startswith("B7")})
    if not b7_ids:
        sys.exit("no B7 shards in results/shards/ -- run scripts/run_jev.py first")
    report = {}
    for d in b7_ids:
        rows = [r for r in recs if r["detector"] == d]
        usable = [r for r in rows if isinstance(r.get("p_implemented"), (int, float))]
        dropped = len(rows) - len(usable)
        rep = {"dropped_missing_probability": dropped, "overall": summarize(usable, True),
               "by_class": {}}
        for cls in ("ABSENT", "UNWIRED", "STUB", "CLEAN"):
            sub = [r for r in usable if r["mutation"] == cls]
            if sub:
                rep["by_class"][cls] = summarize(sub, True)
        # Decision rule 2, exactly as pre-registered -- no other reading.
        o = rep["overall"]
        worst_gap = max((abs(b["mean_pred"] - b["observed"]) for b in o["reliability"]
                         if b["n"] >= 30), default=0.0)
        rep["rule2"] = {"ece_ok": o["ece"] <= 0.05, "worst_bin_gap_n>=30": round(worst_gap, 4),
                        "bins_ok": worst_gap <= 0.10,
                        "skill_ok": (o["brier_skill"] or 0) > 0}
        rep["rule2"]["calibrated_on_this_task"] = all(
            rep["rule2"][k] for k in ("ece_ok", "bins_ok", "skill_ok"))
        report[d] = rep
    b7_iids = {r["iid"] for r in recs if r["detector"] in b7_ids}
    # Decision rule 3 (ASSUMPTIONS.md section 13): each B7 variant's OWN
    # thresholded verdicts, scored as a 0/1 forecaster, come first.
    for d in b7_ids + REFERENCE:
        rows = [r for r in recs if r["detector"] == d and r["iid"] in b7_iids]
        if rows:
            report[f"{d} (as 0/1 forecaster)"] = {"overall": summarize(rows, False)}

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"{'detector':44}{'n':>6}{'Brier':>8}{'  95% CI':>18}{'skill':>7}"
          f"{'ECE':>7}{'AUROC':>7}{'MCC':>7}")
    for d, rep in report.items():
        o = rep["overall"]
        ci = f"[{o['brier_ci'][0]:.3f},{o['brier_ci'][1]:.3f}]"
        print(f"{d:44}{o['n']:>6}{o['brier']:>8.3f}{ci:>18}{o['brier_skill']:>7.3f}"
              f"{o['ece']:>7.3f}{o['auroc']:>7.3f}{o['mcc_at_0.5']:>7.3f}")
    for d in b7_ids:
        rep = report[d]
        print(f"\n{d}: reliability (P(omitted) bins)   "
              f"[{rep['dropped_missing_probability']} records with no probability dropped]")
        for row in rep["overall"]["reliability"]:
            if row["n"]:
                print(f"  {row['bin']:>8}  n={row['n']:>5}  predicted {row['mean_pred']:.3f}"
                      f"  observed {row['observed']:.3f}")
        ab = rep["overall"]["abstain_band"]
        print(f"  abstain band {ab['band']}: coverage {ab['coverage']:.3f}, "
              f"MCC on confident {ab['mcc_on_confident']}, "
              f"error on confident {ab['error_rate_on_confident']}")
        r2 = rep["rule2"]
        print(f"  rule 2 (pre-registered): ECE<=0.05 {r2['ece_ok']}, worst bin gap "
              f"{r2['worst_bin_gap_n>=30']} (<=0.10: {r2['bins_ok']}), skill>0 "
              f"{r2['skill_ok']} => calibrated on this task: "
              f"{'SUPPORTED' if r2['calibrated_on_this_task'] else 'NOT SUPPORTED'}")
        print(f"  by class: " + ", ".join(
            f"{c} Brier {s['brier']:.3f} ECE {s['ece']:.3f}"
            for c, s in rep["by_class"].items()))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
