"""
T6 CI-gate entrypoint. Reads the PR under review from GitHub Actions env
vars, scores it with P1 only (TASKS.md T6, decision 1), and posts one
comment via `gh pr comment` -- unless results/gate_status.json (written by
`make gate-check`, scripts/check_gate_precision.py) says the gate is
suspended, in which case it does nothing and exits 0: a suspended gate is
a known, already-flagged state, not a CI failure in itself. Loads
results/baseline_t6.json and passes it to G.render_comment so every posted
comment carries the live precision CI, not just README.md's status block.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, ".")

from omitbench import gate as G


def _fetch_pr_body(pr_number: str) -> str:
    proc = subprocess.run(
        ["gh", "pr", "view", pr_number, "--json", "body", "-q", ".body"],
        capture_output=True, text=True, check=True,
    )
    return proc.stdout


def _post_comment(pr_number: str, body: str) -> None:
    subprocess.run(["gh", "pr", "comment", pr_number, "--body", body], check=True)


def main() -> int:
    with open("results/gate_status.json") as f:
        status = json.load(f)
    if status["status"] != "active":
        print("Gate is suspended (results/gate_status.json) -- not posting.",
              file=sys.stderr)
        return 0

    with open("results/baseline_t6.json") as f:
        baseline = json.load(f)

    pr_number = os.environ["PR_NUMBER"]
    head_sha = os.environ["PR_HEAD_SHA"]
    pr_body = _fetch_pr_body(pr_number)

    result = G.score_pr(pr_body, repo_dir=".", head_sha=head_sha)
    comment = G.render_comment(result, baseline)
    if comment is None:
        return 0

    _post_comment(pr_number, comment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
