"""
Reads results/k5_subsample/llm_judge_*_seed*.jsonl (written by
`scripts/run_judge.py --k 5`) and reports the standard deviation of MCC
across seeds, per judge (B4/B5/B6), on the same stratified 100-instance
subsample.

Deliberately separate from scripts/analyze.py: these files are five NOISE
SAMPLES of the same 100 instances per judge, not five independent detectors
or five more instances. Averaging them into the headline table's cluster
bootstrap would misrepresent them as more data; the only honest use of this
run is exactly what this script does -- measure run-to-run variance, one
judge at a time.

Grouped by the `detector` field actually present in each record (which now
holds the full judge_id, e.g. "B4 LLM judge (gpt-oss-120b)"), not by
filename -- the filename encodes the same judge for convenience but the
record is the source of truth.
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")  # analyze.py has no package __init__, import it directly

from analyze import metrics  # noqa: E402


def main():
    paths = sorted(glob.glob("results/k5_subsample/llm_judge_*_seed*.jsonl"))
    if not paths:
        sys.exit("no results/k5_subsample/llm_judge_*_seed*.jsonl files found -- "
                  "run `python3 scripts/run_judge.py --k 5` first")

    # judge -> seed -> [(pred, gold), ...]
    by_judge: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                by_judge[r["detector"]][r["seed"]].append((r["pred"], r["gold"]))

    for judge in sorted(by_judge):
        per_seed_mcc = {seed: metrics(rows)["mcc"]
                         for seed, rows in by_judge[judge].items()}
        print(f"\n{judge}")
        print(f"{'seed':>6}{'MCC':>10}")
        for seed, mcc in sorted(per_seed_mcc.items()):
            print(f"{seed:>6}{mcc:>10.4f}")

        vals = list(per_seed_mcc.values())
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"n_seeds={len(vals)}  mean MCC={mean:.4f}  stdev={stdev:.4f}")

    print("\n(measures run-to-run variance on the SAME stratified ~100-instance "
          "subsample, per judge -- not a substitute for the k=1 full-corpus "
          "headline number)")


if __name__ == "__main__":
    main()
