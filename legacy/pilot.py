"""
Week-2 pilot. Freezes baseline numbers BEFORE any optimisation.

    python -m omitbench.pilot --repos corpus/click corpus/requests corpus/attrs \
        --target 120 --seed 0

Writes results/runs.jsonl (one record per requirement per detector) and
prints the pilot table. Every number in the README is computed from that file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

from omitbench import corpus as C
from omitbench import evidence as E


def build_corpus(repos: list[str], target: int, seed: int) -> list[C.Instance]:
    rng = random.Random(seed)
    per_repo = max(1, target // len(repos))
    out: list[C.Instance] = []

    for repo in repos:
        commits = C.list_commits(repo, limit=700)
        rng.shuffle(commits)
        got = 0
        for sha in commits:
            if got >= per_repo:
                break
            try:
                inst = C.build_instance(repo, sha, rng)
            except Exception:
                continue
            if inst is None:
                continue
            out.append(inst)
            got += 1
        print(f"  {os.path.basename(repo):12s} {got:4d} instances", file=sys.stderr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    print("building corpus...", file=sys.stderr)
    insts = build_corpus(args.repos, args.target, args.seed)
    if not insts:
        sys.exit("no instances built")

    ctx = {
        "median_changed_lines": statistics.median(
            len(E._changed_lines(i.before, i.after)) for i in insts
        )
    }

    os.makedirs(args.out, exist_ok=True)
    records: list[dict] = []

    for inst in insts:
        for dname, fn in E.DETECTORS.items():
            preds = fn(inst.spec, inst.requirements, inst.before, inst.after, ctx)
            for r in inst.requirements:
                records.append({
                    "iid": inst.iid, "repo": inst.repo, "commit": inst.commit,
                    "detector": dname, "requirement": r,
                    "pred": preds[r], "gold": inst.labels[r],
                    "n_hunks_total": inst.n_hunks_total,
                    "n_hunks_kept": inst.n_hunks_kept,
                    "seed": args.seed,
                })

    with open(f"{args.out}/runs.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # ---- corpus stats ----
    n_req = sum(len(i.requirements) for i in insts)
    n_om = sum(sum(v == "OMITTED" for v in i.labels.values()) for i in insts)
    print()
    print("=" * 74)
    print("OMITBENCH PILOT  (frozen before any optimisation)")
    print("=" * 74)
    print(f"instances            : {len(insts)}")
    print(f"repos                : {sorted({i.repo for i in insts})}")
    print(f"requirements (total) : {n_req}")
    print(f"omission base rate   : {n_om / n_req:.3f}   <-- accuracy is meaningless above this")
    print(f"median hunks/instance: {statistics.median(i.n_hunks_total for i in insts):.0f}")
    print(f"median changed lines : {ctx['median_changed_lines']:.0f}")
    print()

    # ---- pooled scores ----
    print(f"{'detector':26s} {'prec':>6s} {'recall':>7s} {'F1':>7s} {'acc':>7s}   {'TP':>4s} {'FP':>4s} {'FN':>4s}")
    print("-" * 74)
    for dname in E.DETECTORS:
        rows = [r for r in records if r["detector"] == dname]
        s = E.score([r["pred"] for r in rows], [r["gold"] for r in rows])
        print(f"{dname:26s} {s['precision']:6.3f} {s['recall']:7.3f} {s['f1']:7.3f} "
              f"{s['accuracy']:7.3f}   {s['tp']:4d} {s['fp']:4d} {s['fn']:4d}")

    # ---- per-repo (leakage / generalisation sanity) ----
    print()
    print("per-repo F1 (proposed method) -- large spread => repo-specific overfit risk")
    for repo in sorted({r["repo"] for r in records}):
        rows = [r for r in records
                if r["detector"] == "P  AST symbol diff" and r["repo"] == repo]
        if rows:
            s = E.score([r["pred"] for r in rows], [r["gold"] for r in rows])
            print(f"  {repo:12s} n={len(rows):4d}  F1={s['f1']:.3f}  "
                  f"prec={s['precision']:.3f} rec={s['recall']:.3f}")

    print()
    print(f"wrote {len(records)} records -> {args.out}/runs.jsonl")


if __name__ == "__main__":
    main()
