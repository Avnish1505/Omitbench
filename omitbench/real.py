"""
Loader for hand-labelled real-agent trajectories (TASKS.md T4).

Reads data/real/real_*.json into the SAME (spec, reqs, before, after, gold)
shape omitbench/experiment.py builds from the synthetic mutation pipeline,
so every detector currently registered in omitbench.detectors.DETECTORS runs
against real trajectories completely unchanged -- same 5-positional-argument
contract, same tests/test_leakage.py guarantee, because nothing here changes
what a detector is called with:

    detect(spec, reqs, before, after, ctx) -> {req: "IMPLEMENTED"|"OMITTED"}

gold is kept OUT of that call, exactly like experiment.py keeps `target`/
`cls` out of it -- see score() below and tests/test_real.py's leakage guard.

Each data/real/real_*.json is one hand-labelled trajectory:
  id, repo, commit              -- provenance
  agent, task_text               -- what was asked (task_text is used as `spec`)
  before, after                  -- {path: [line, ...]}, same shape as
                                     everywhere else in this codebase
  requirements                   -- [{"req": "path.py::Symbol",
                                       "gold": "IMPLEMENTED"|"OMITTED"}, ...]
  labelled_by, labelled_at       -- who/when hand-labelled this
  detector_run_before_labelling  -- must be present; a labeller who saw a
                                     detector's verdict before labelling
                                     risks anchoring on it, which is why this
                                     field exists. load_real() only checks it
                                     is present and boolean -- it does not
                                     reject True, since excluding cases is
                                     its own bias (CLAUDE.md anti-pattern #1).
                                     scripts/analyze.py's analyze_real() is
                                     where that risk should be surfaced to a
                                     reader, not silently filtered here.

Schema validation fails LOUDLY: every malformed file is reported, not just
the first one, and nothing is silently skipped.
"""

from __future__ import annotations

import glob
import json
import os
from typing import NamedTuple

DATA_GLOB = "data/real/real_*.json"

REQUIRED_FIELDS = {
    "id", "repo", "commit", "before", "after", "requirements",
    "labelled_by", "labelled_at", "detector_run_before_labelling",
}
VALID_GOLD = {"IMPLEMENTED", "OMITTED"}


class RealSchemaError(Exception):
    """Raised with every malformed file and every problem in it, not just
    the first -- so a bad batch of labels is fixed in one pass, not one
    file per run."""


class RealInstance(NamedTuple):
    iid: str
    repo: str
    spec: str
    reqs: list[str]
    before: dict[str, list[str]]
    after: dict[str, list[str]]
    gold: dict[str, str]  # req -> "IMPLEMENTED" | "OMITTED"


def _validate(data) -> list[str]:
    problems = []
    if not isinstance(data, dict):
        return [f"top-level JSON must be an object, got {type(data).__name__}"]

    missing = REQUIRED_FIELDS - data.keys()
    if missing:
        problems.append(f"missing required field(s): {sorted(missing)}")

    for key in ("before", "after"):
        val = data.get(key)
        if key in data and not (
            isinstance(val, dict)
            and all(isinstance(p, str) for p in val)
            and all(isinstance(ls, list) and all(isinstance(x, str) for x in ls)
                    for ls in val.values())
        ):
            problems.append(f"{key!r} must be an object of path -> [line, ...]")

    if "detector_run_before_labelling" in data and \
            not isinstance(data["detector_run_before_labelling"], bool):
        problems.append("'detector_run_before_labelling' must be a boolean")

    reqs = data.get("requirements")
    if "requirements" in data:
        if not isinstance(reqs, list) or not reqs:
            problems.append("'requirements' must be a non-empty list")
        else:
            seen = set()
            for i, item in enumerate(reqs):
                if not isinstance(item, dict) or "req" not in item or "gold" not in item:
                    problems.append(
                        f"requirements[{i}] must be an object with 'req' and 'gold'")
                    continue
                if not isinstance(item["req"], str) or not item["req"]:
                    problems.append(f"requirements[{i}]['req'] must be a non-empty string")
                elif item["req"] in seen:
                    problems.append(
                        f"duplicate requirement key {item['req']!r} "
                        f"(gold map would silently collide)")
                else:
                    seen.add(item["req"])
                if item.get("gold") not in VALID_GOLD:
                    problems.append(
                        f"requirements[{i}]['gold'] must be one of {sorted(VALID_GOLD)}, "
                        f"got {item.get('gold')!r}")

    return problems


def _build(data) -> RealInstance:
    reqs = [item["req"] for item in data["requirements"]]
    gold = {item["req"]: item["gold"] for item in data["requirements"]}
    return RealInstance(
        iid=data["id"],
        repo=os.path.basename(data["repo"]),
        spec=data.get("task_text", ""),
        reqs=reqs,
        before=data["before"],
        after=data["after"],
        gold=gold,
    )


def load_real(pattern: str = DATA_GLOB) -> list[RealInstance]:
    """Validate and load every file matching `pattern`. Raises RealSchemaError
    listing ALL malformed files (with all problems in each) if any file is
    invalid -- nothing is silently skipped."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise RealSchemaError(f"no real-trajectory files matched {pattern!r}")

    problems: dict[str, list[str]] = {}
    instances = []
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            problems[path] = [f"invalid JSON: {e}"]
            continue
        file_problems = _validate(data)
        if file_problems:
            problems[path] = file_problems
            continue
        instances.append(_build(data))

    if problems:
        lines = [f"{len(problems)} malformed real-trajectory file(s):"]
        for path in sorted(problems):
            lines.append(f"  {path}:")
            lines.extend(f"    - {p}" for p in problems[path])
        raise RealSchemaError("\n".join(lines))

    instances.sort(key=lambda inst: inst.iid)
    return instances


def score(instances: list[RealInstance], detectors: dict | None = None) -> list[dict]:
    """Run every detector over every real instance. Same call shape as
    experiment.py's scoring loop: fn(spec, reqs, before, after, ctx={}).
    gold is looked up from RealInstance.gold and attached to the OUTPUT
    record only -- it is never passed into the detector call, so this reuses
    the exact leakage guarantee tests/test_leakage.py already proves for the
    synthetic pipeline (see tests/test_real.py's leakage guard for the
    real-data equivalent).

    `detectors` defaults to omitbench.detectors.DETECTORS -- never hardcode a
    detector-name list here, so a future detector (or P2/P3 coming back per
    TASKS.md T3) is picked up for free.
    """
    if detectors is None:
        from omitbench import detectors as P
        detectors = P.DETECTORS

    recs = []
    for inst in instances:
        for dname, fn in detectors.items():
            preds = fn(inst.spec, inst.reqs, inst.before, inst.after, {})
            for r in inst.reqs:
                recs.append({
                    "iid": inst.iid,
                    "repo": inst.repo,
                    "req": r,
                    "detector": dname,
                    "pred": preds.get(r, "IMPLEMENTED"),
                    "gold": inst.gold[r],
                    "source": "real",
                })
    return recs
