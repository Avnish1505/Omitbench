"""
T6 latency benchmark (TASKS.md T6 acceptance: p95 < 10s per instance on a
laptop). Measures the compute-bound part of the gate -- checklist parsing +
git snapshot fetch + P1 detection -- against real files from corpus/*
repos. Does NOT measure `gh pr view`/`gh pr comment` network round-trips;
those are GitHub API latency, not this project's code, and are unbounded by
anything `omitbench` controls.

Run: `make gate-bench` (or `python3 scripts/bench_gate_latency.py`).
Not part of `make test` -- this is a one-off measurement to record in
TASKS.md, not a per-commit regression gate.
"""

from __future__ import annotations

import glob
import subprocess
import sys
import time

sys.path.insert(0, ".")

from omitbench import gate as G

N_SAMPLES = 40


def _sample_pr_bodies(repo_dir: str, n: int) -> list[tuple[str, str]]:
    """(pr_body, head_sha) pairs built from real HEAD-tracked .py files in
    repo_dir -- stand-ins for a PR checklist, not corpus.py's mutation
    pipeline (that's for the offline benchmark; this measures wall time on
    ordinary git operations, which don't care whether the symbol is real)."""
    sha = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    files = subprocess.run(
        ["git", "-C", repo_dir, "ls-files", "*.py"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()[:n]
    out = []
    for path in files:
        body = f"- [ ] some_symbol in {path}\n"
        out.append((body, sha))
    return out


def main() -> None:
    repos = sorted(glob.glob("corpus/*"))
    repos = [r for r in repos if r.split("/")[-1] != "black"]  # excluded, see CLAUDE.md
    if not repos:
        print("No corpus/* repos found -- run scripts/fetch_corpus.sh first.",
              file=sys.stderr)
        raise SystemExit(1)

    samples = _sample_pr_bodies(repos[0], N_SAMPLES)
    times = []
    for body, sha in samples:
        start = time.monotonic()
        G.score_pr(body, repo_dir=repos[0], head_sha=sha)
        times.append(time.monotonic() - start)

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    print(f"n={len(times)} samples from {repos[0]}")
    print(f"p50 = {p50 * 1000:.1f}ms   p95 = {p95 * 1000:.1f}ms   "
          f"(acceptance bar: p95 < 10000ms)")
    if p95 >= 10.0:
        print("FAILS acceptance bar.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
