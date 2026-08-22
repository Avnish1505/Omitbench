"""
T6 falsification check (TASKS.md T6, decision 2): if P1's precision on the
CURRENT shards drops below the frozen 0.80 acceptance threshold, the gate
must be marked suspended -- not silently left running. Run via
`make gate-check`; wired into ci.yml on every push to main (not on PRs --
PRs don't regenerate shards, so there's nothing new to check per-PR).

This threshold check is a POINT ESTIMATE against 0.80, exactly as decided.
Know its limits before trusting a single flip: results/baseline_t6.json
records a 95% CI of roughly [0.72, 0.88] around today's 0.8018 point
estimate -- the interval straddles this threshold on both sides. A future
run landing on, say, 0.79 is well within that noise band, not necessarily
a real regression. See ASSUMPTIONS.md's T6 section.
"""

from __future__ import annotations

import datetime
import json
import re
import sys

sys.path.insert(0, ".")

from scripts.analyze import load, metrics
from scripts.build_baseline_t6 import THRESHOLD

STATUS_START = "<!-- GATE_STATUS_START -->"
STATUS_END = "<!-- GATE_STATUS_END -->"


def current_precision() -> float:
    recs = load()
    rows = [(r["pred"], r["gold"]) for r in recs if r["detector"] == "P1 defined"]
    return metrics(rows)["precision"]


def status_block(active: bool, precision: float) -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    if active:
        body = (f"**Gate status: active.** Last checked {ts}, P1 precision "
                 f"{precision:.4f} on the frozen corpus (threshold "
                 f"{THRESHOLD}).")
    else:
        body = (f"**Gate status: suspended — precision regression "
                 f"detected.** Last checked {ts}, P1 precision "
                 f"{precision:.4f} fell below the {THRESHOLD} threshold. Do "
                 f"not deploy `.github/workflows/omission-gate.yml` until "
                 f"this is resolved.")
    return f"{STATUS_START}\n{body}\n{STATUS_END}"


def _update_readme(block: str) -> None:
    with open("README.md") as f:
        text = f.read()
    pattern = re.compile(re.escape(STATUS_START) + r".*?" + re.escape(STATUS_END), re.S)
    if not pattern.search(text):
        raise SystemExit(
            "README.md is missing the GATE_STATUS markers -- add "
            f"{STATUS_START} / {STATUS_END} once by hand before this "
            "script can run (see T6 plan, Task 5, Step 3a)."
        )
    with open("README.md", "w") as f:
        f.write(pattern.sub(block, text))


def main() -> int:
    precision = current_precision()
    active = precision >= THRESHOLD
    status = {
        "status": "active" if active else "suspended",
        "measured_precision": precision,
        "threshold": THRESHOLD,
    }
    with open("results/gate_status.json", "w") as f:
        json.dump(status, f, indent=2)
        f.write("\n")
    _update_readme(status_block(active, precision))
    print(json.dumps(status, indent=2))
    return 0 if active else 1


if __name__ == "__main__":
    raise SystemExit(main())
