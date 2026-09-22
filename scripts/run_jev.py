"""
Runs the pre-registered B7 Jev judges (omitbench/jev.py) over the OmitBench
corpus, k=1, and writes one shard per variant:

    results/shards/llm_judge_b7_jev_single.jsonl
    results/shards/llm_judge_b7_jev_split.jsonl

Each record carries the usual (pred, gold) pair that scripts/analyze.py
scores, plus `p_implemented`, the probability Jev gave for that requirement,
which scripts/analyze_calibration.py turns into Brier / ECE / reliability.

Corpus construction and the alignment pre-flight are imported from
scripts/run_judge.py rather than copied, so B7 is guaranteed to be scored
on the same instances as B4/B5 and P1 (analyze.py's paired bootstrap
depends on it).

    export TYPESAFE_API_KEY=sk-...        # never commit, never paste in chat
    python3 scripts/run_jev.py --repos corpus/* --dry-run
    python3 scripts/run_jev.py --repos corpus/* --limit-instances 5   # smoke
    python3 scripts/run_jev.py --repos corpus/*                       # full

`--max-cost` (default $1.00) hard-stops mid-run once estimated spend,
computed from each response's usage.input_tokens at $0.042/1M, crosses it.
The full two-variant sweep is expected to cost well under $0.25.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, ".")

from omitbench import judges as J  # noqa: E402
from omitbench import jev as V  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_judge", os.path.join(os.path.dirname(__file__), "run_judge.py"))
RJ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RJ)

# run_judge.py's default --per-repo/--scan-cap/--seeds -- the values the
# shards were built with. Kept identical so the alignment check means
# something.
PER_REPO, SCAN_CAP, SEED = 40, 600, 0

SHORT = {"single": "B7 Jev (single)", "split": "B7 Jev (split)"}


def _slug(judge_id: str) -> str:
    return J._slugify(judge_id)


def dry_run(insts, variants) -> None:
    for v in variants:
        cfg = V.CONFIGS[SHORT[v]]
        build = V.build_jev_request if cfg.variant == "single" else V.build_jev_request_split
        calls = chars = trunc = 0
        for _iid, _rn, _spec, before, _t, reqs, vs in insts:
            for _cls, after in vs:
                diff, t = J._truncate(J.unified_diff(before, after), J.DIFF_TOKEN_CEILING)
                chars += len(json.dumps(build(reqs, diff)))
                calls += 1
                trunc += t
        tok = chars / 4
        print(f"  {cfg.judge_id:18} calls {calls:>5}  ~{tok/1e6:.2f}M input tokens "
              f"(chars/4 estimate)  est. ${tok * V.PRICE_PER_1M_INPUT / 1e6:.3f}  "
              f"truncated diffs {trunc}")
    print("\nNO API call was made.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True,
                    help="corpus/<repo> dirs (see `make corpus`)")
    ap.add_argument("--variants", nargs="+", default=["single", "split"],
                    choices=["single", "split"])
    ap.add_argument("--max-cost", type=float, default=1.0)
    ap.add_argument("--limit-instances", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="results/shards")
    a = ap.parse_args()

    # black is excluded everywhere else (ASSUMPTIONS.md); exclude it here too
    repos = [r for r in a.repos if os.path.basename(r.rstrip("/")) != "black"]
    insts = RJ.build_instances(repos, PER_REPO, SEED, SCAN_CAP)
    RJ.check_alignment(insts, limit=len(insts))
    if a.limit_instances:
        insts = insts[: a.limit_instances]

    if a.dry_run:
        dry_run(insts, a.variants)
        return
    if not os.environ.get(V.API_KEY_ENV):
        sys.exit(f"${V.API_KEY_ENV} is not set")

    spent = 0.0
    for v in a.variants:
        cfg = V.CONFIGS[SHORT[v]]
        fn = V.DETECTORS[cfg.judge_id]
        recs, errors = [], 0
        for iid, rname, spec, before, target, reqs, variants in insts:
            for cls, after in variants:
                record: dict = {}
                try:
                    preds = fn(spec, reqs, before, after, {"_judge_record": record})
                except Exception as e:
                    # A model-pin mismatch or 4xx is a bug, not noise: stop.
                    if "pinned" in str(e) or "HTTP 4" in str(e):
                        raise
                    errors += 1
                    print(f"  ERROR {cfg.judge_id}/{iid}/{cls}: {e!r}", file=sys.stderr)
                    continue
                if record.get("cost_usd"):
                    spent += record["cost_usd"]
                if spent > a.max_cost:
                    sys.exit(f"HARD STOP: est. spend ${spent:.3f} > --max-cost "
                             f"${a.max_cost:.2f}")
                probs = record.get("p_implemented") or {}
                for r in reqs:
                    gold = "OMITTED" if (cls != "CLEAN" and r == target) else "IMPLEMENTED"
                    recs.append({
                        "iid": iid, "repo": rname, "req": r, "target": target,
                        "mutation": cls, "detector": cfg.judge_id,
                        "pred": preds.get(r, "IMPLEMENTED"), "gold": gold,
                        "p_implemented": probs.get(r), "threshold": V.THRESHOLD,
                        "seed": SEED, "model_slug": V.PINNED_MODEL,
                        "model_returned": record.get("model_returned"),
                        "truncated": record.get("truncated", False),
                        "parse_failed": record.get("parse_failed", False),
                        "cache_hit": record.get("cache_hit", False),
                        "cost_usd": record.get("cost_usd"),
                        "cost_source": record.get("cost_source"),
                    })
        os.makedirs(a.out_dir, exist_ok=True)
        out = os.path.join(a.out_dir, f"llm_judge_{_slug(cfg.judge_id)}.jsonl")
        with open(out, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        s = V._stats_for(cfg.judge_id)
        print(f"{cfg.judge_id}: {len(recs)} records -> {out} | calls {s.calls} | "
              f"cache hits {s.cache_hits} | parse failures {s.parse_failures} | "
              f"truncated {s.truncated} | errors {errors} | "
              f"input tokens {s.input_tokens}", file=sys.stderr)
    print(f"\nretries {V._RETRIES[0]} | est. spend this run ${spent:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
