# Real-trajectory loader (T4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `omitbench/real.py` so the 8 hand-labelled trajectories in
`data/real/real_*.json` load into the same `(spec, reqs, before, after, gold)`
shape `experiment.py` builds from the synthetic pipeline, validate their
schema loudly, and extend `scripts/analyze.py` with a `--source
{synthetic,real,both}` flag that reports the real-corpus numbers correctly —
including the degenerate case where the positive class is empty.

**Architecture:** One new module (`omitbench/real.py`) mirrors the existing
`omitbench/judges.py` pattern: pure loader/validator functions, a `NamedTuple`
instance shape, and a `score()` function that calls detectors with the exact
same 5-positional-argument contract `experiment.py` and `judges.py` already
use. `scripts/analyze.py` gains two new top-level functions
(`analyze_real`, `_synthetic_mcc_summary`) and an `argparse` flag; the
existing no-arg code path (`make analyze`) is refactored into
`analyze_synthetic()` but must print byte-identical output to today.

**Tech Stack:** stdlib only (`json`, `glob`, `typing.NamedTuple`) — matches
the zero-deps constraint in CLAUDE.md. `pytest` for tests.

**Spec:** the user's task message in this conversation (no separate spec
file). Key facts gathered before planning:
- `data/real/real_NNN.json` shape (all 8 files, verified): top-level keys
  `id, repo, commit, agent, task_text, before, after, requirements,
  labelled_by, labelled_at, detector_run_before_labelling`.
- `before`/`after` are `{path: [line, ...]}` — identical shape to
  `experiment.py`'s `before`/`after`.
- `requirements` is a list of `{"req": "path.py::Symbol", "gold":
  "IMPLEMENTED"|"OMITTED"}`.
- **Verified fact to report before any other output:** across all 8 files,
  16 requirements total, **0 are OMITTED, 16 are IMPLEMENTED**. The positive
  class is empty in this sample.
- Detector contract (`omitbench/detectors.py`, enforced by
  `tests/test_leakage.py`): `detect(spec, reqs, before, after, ctx) ->
  {req: "IMPLEMENTED"|"OMITTED"}`. `P.DETECTORS` currently = `B0
  flag-nothing, B1 flag-everything, B3 line-grep (no AST), P1 defined` (P2/P3
  are retired, per TASKS.md T3 — the loader must iterate whatever is
  registered in `P.DETECTORS`, not a hardcoded name list).
- `scripts/analyze.py` already has `metrics(rows)`, `boot_indices(iids, n,
  seed)`, and `ORDER`. Reuse them; do not duplicate.
- Import convention for scripts pulling in the `omitbench` package (see
  `scripts/run_judge.py:73-77`, `scripts/k5_variance.py:28-29`):
  `sys.path.insert(0, ".")` then `from omitbench import X as Y  # noqa: E402`.

## Global Constraints

- No detector, `results/pilot.json`, or `results/shards/*.jsonl` may be
  modified. (CLAUDE.md do-not-touch list.)
- `tests/test_leakage.py` must not be modified (CLAUDE.md do-not-touch list)
  — any new leakage-style guard for the real-data path goes in the new
  `tests/test_real.py` instead.
- Core stays stdlib-only; no new dependency.
- Schema validation must fail loudly and **list every malformed file**, not
  stop at the first one and not silently skip any.
- Do not tune anything based on what the real-corpus MCC/recall turns out to
  be. Report the 0-OMITTED result as-is.
- `make analyze` (no args) must print exactly what it prints today — the
  refactor into `analyze_synthetic()` is a pure extraction, no behavior
  change to the default path.

---

### Task 1: Schema validator + loader (`omitbench/real.py`)

**Files:**
- Create: `omitbench/real.py`
- Test: `tests/test_real.py`

**Interfaces:**
- Produces:
  - `class RealInstance(NamedTuple)`: `iid: str`, `repo: str`, `spec: str`,
    `reqs: list[str]`, `before: dict[str, list[str]]`, `after: dict[str,
    list[str]]`, `gold: dict[str, str]` (maps each `req` string to
    `"IMPLEMENTED"`/`"OMITTED"`).
  - `class RealSchemaError(Exception)` — raised with a single message
    listing every malformed file and every problem found in each.
  - `REQUIRED_FIELDS: set[str]` = `{"id", "repo", "commit", "before",
    "after", "requirements", "labelled_by", "labelled_at",
    "detector_run_before_labelling"}`.
  - `DATA_GLOB = "data/real/real_*.json"` (module-level constant, default
    for `load_real`).
  - `def load_real(pattern: str = DATA_GLOB) -> list[RealInstance]` —
    validates every matched file, raises one `RealSchemaError` listing all
    problems if any file is malformed, otherwise returns one
    `RealInstance` per file, sorted by `iid`.

- [ ] **Step 1: Write the failing tests for the validator**

```python
# tests/test_real.py
from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, ".")

from omitbench import real as R  # noqa: E402

VALID = {
    "id": "real_999",
    "repo": "tqdm/tqdm",
    "commit": "a" * 40,
    "agent": "claude-code",
    "task_text": "do the thing",
    "before": {"m.py": ["def f():", "    return 1"]},
    "after": {"m.py": ["def f():", "    return 1"], "n.py": ["def g():", "    return 2"]},
    "requirements": [{"req": "n.py::g", "gold": "IMPLEMENTED"}],
    "labelled_by": "avnish",
    "labelled_at": "2026-08-19",
    "detector_run_before_labelling": False,
}


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def test_missing_required_field_is_rejected(tmp_path):
    bad = dict(VALID)
    del bad["commit"]
    write(tmp_path, "real_001.json", bad)
    with pytest.raises(R.RealSchemaError) as exc:
        R.load_real(str(tmp_path / "real_*.json"))
    assert "real_001.json" in str(exc.value)
    assert "commit" in str(exc.value)


def test_bad_gold_value_is_rejected(tmp_path):
    bad = dict(VALID)
    bad["requirements"] = [{"req": "n.py::g", "gold": "MAYBE"}]
    write(tmp_path, "real_001.json", bad)
    with pytest.raises(R.RealSchemaError) as exc:
        R.load_real(str(tmp_path / "real_*.json"))
    assert "real_001.json" in str(exc.value)
    assert "gold" in str(exc.value).lower()


def test_all_malformed_files_are_listed_not_just_the_first(tmp_path):
    bad1 = dict(VALID)
    del bad1["commit"]
    bad2 = dict(VALID)
    del bad2["labelled_by"]
    write(tmp_path, "real_001.json", bad1)
    write(tmp_path, "real_002.json", bad2)
    with pytest.raises(R.RealSchemaError) as exc:
        R.load_real(str(tmp_path / "real_*.json"))
    msg = str(exc.value)
    assert "real_001.json" in msg and "real_002.json" in msg


def test_valid_file_loads_into_real_instance(tmp_path):
    write(tmp_path, "real_001.json", VALID)
    [inst] = R.load_real(str(tmp_path / "real_*.json"))
    assert inst.iid == "real_999"
    assert inst.repo == "tqdm"  # basename, matches shard repo naming
    assert inst.spec == "do the thing"
    assert inst.reqs == ["n.py::g"]
    assert inst.before == VALID["before"]
    assert inst.after == VALID["after"]
    assert inst.gold == {"n.py::g": "IMPLEMENTED"}


def test_no_files_match_raises():
    with pytest.raises(R.RealSchemaError):
        R.load_real("data/real/does_not_exist_*.json")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_real.py -v`
Expected: `ModuleNotFoundError: No module named 'omitbench.real'` (or
`ImportError`) for every test.

- [ ] **Step 3: Implement `omitbench/real.py`**

```python
"""
Loader for hand-labelled real-agent trajectories (TASKS.md T4).

Reads data/real/real_*.json into the SAME (spec, reqs, before, after, gold)
shape omitbench/experiment.py builds from the synthetic mutation pipeline,
so every detector currently registered in omitbench.detectors.DETECTORS
runs against real trajectories completely unchanged -- same 5-positional-
argument contract, same tests/test_leakage.py guards apply because nothing
here changes what a detector is called with.

    detect(spec, reqs, before, after, ctx) -> {req: "IMPLEMENTED"|"OMITTED"}

gold is kept OUT of that call, exactly like experiment.py keeps `target`/
`cls` out of it -- see score() below and tests/test_real.py's leakage guard.

Each data/real/real_*.json is one hand-labelled trajectory:
  id, repo, commit           -- provenance
  agent, task_text           -- what was asked (task_text is used as `spec`)
  before, after              -- {path: [line, ...]}, same shape as everywhere
                                 else in this codebase
  requirements                -- [{"req": "path.py::Symbol",
                                    "gold": "IMPLEMENTED"|"OMITTED"}, ...]
  labelled_by, labelled_at    -- who/when hand-labelled this
  detector_run_before_labelling -- must be present; a labeller who saw a
                                 detector's verdict before labelling risks
                                 anchoring on it, which is why this field
                                 exists, but load_real() only checks it is
                                 present -- see docstring warning in
                                 scripts/analyze.py's analyze_real().

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_real.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add omitbench/real.py tests/test_real.py
git commit -m "feat: add schema-validating loader for real trajectories (T4)"
```

---

### Task 2: `score()` + real-data leakage guard + gold-distribution check against the actual 8 files

**Files:**
- Modify: `omitbench/real.py`
- Modify: `tests/test_real.py`

**Interfaces:**
- Consumes: `RealInstance` from Task 1; `omitbench.detectors.DETECTORS`
  (dict `name -> detect(spec, reqs, before, after, ctx)`).
- Produces: `def score(instances: list[RealInstance], detectors: dict |
  None = None) -> list[dict]`. Each dict has keys `iid, repo, req,
  detector, pred, gold, source` (`source` is always the literal string
  `"real"`, so a caller merging real + synthetic records can tell them
  apart). `detectors` defaults to `omitbench.detectors.DETECTORS` — never
  hardcode a detector-name list, so a future detector is picked up for
  free.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_real.py

def test_score_calls_detector_with_exact_contract_and_no_gold_leak():
    inst = R.RealInstance(
        iid="x1", repo="tqdm", spec="add a thing",
        reqs=["m.py::helper"],
        before={"m.py": ["def existing():", "    return 1"]},
        after={"m.py": ["def helper():", "    return 2"],
               "m2.py": ["def existing():", "    return helper()"]},
        gold={"m.py::helper": "OMITTED"},
    )

    seen_args = {}

    def fake_detector(spec, reqs, before, after, ctx):
        seen_args["spec"] = spec
        seen_args["reqs"] = reqs
        seen_args["before"] = before
        seen_args["after"] = after
        seen_args["ctx"] = ctx
        return {"m.py::helper": "IMPLEMENTED"}

    recs = R.score([inst], {"fake": fake_detector})
    assert seen_args["spec"] == "add a thing"
    assert seen_args["reqs"] == ["m.py::helper"]
    assert seen_args["before"] == inst.before
    assert seen_args["after"] == inst.after
    assert seen_args["ctx"] == {}
    assert "gold" not in seen_args and "OMITTED" not in seen_args.values()

    assert recs == [{
        "iid": "x1", "repo": "tqdm", "req": "m.py::helper",
        "detector": "fake", "pred": "IMPLEMENTED", "gold": "OMITTED",
        "source": "real",
    }]


def test_score_missing_prediction_defaults_to_implemented():
    """Mirrors experiment.py's `preds.get(r, "IMPLEMENTED")` convention."""
    inst = R.RealInstance(
        iid="x1", repo="tqdm", spec="", reqs=["m.py::helper"],
        before={}, after={}, gold={"m.py::helper": "IMPLEMENTED"},
    )
    recs = R.score([inst], {"empty": lambda spec, reqs, before, after, ctx: {}})
    assert recs[0]["pred"] == "IMPLEMENTED"


def test_score_defaults_to_registered_detectors():
    """score() with no detectors= arg must use every currently-registered
    detector, not a hardcoded name list -- so P2/P3 coming back (TASKS.md
    T3) or a new detector being added needs no change here."""
    from omitbench import detectors as P
    inst = R.RealInstance(
        iid="x1", repo="tqdm", spec="", reqs=["m.py::f"],
        before={"m.py": ["def f():", "    return 1"]},
        after={"m.py": ["def f():", "    return 1"]},
        gold={"m.py::f": "IMPLEMENTED"},
    )
    recs = R.score([inst])
    assert {r["detector"] for r in recs} == set(P.DETECTORS)


def test_actual_real_corpus_gold_distribution():
    """Locks in the finding that must be reported before any recall/MCC
    number: as of the 8 committed trajectories, the positive class
    (OMITTED) is EMPTY. This is a fact about data/real/*.json, not a
    tuned assertion -- if it changes because trajectories were added or
    relabelled, this test SHOULD fail so the change is visible, not a
    reason to weaken the assertion.
    """
    instances = R.load_real()
    assert len(instances) == 8
    all_golds = [g for inst in instances for g in inst.gold.values()]
    assert len(all_golds) == 16
    assert all_golds.count("OMITTED") == 0
    assert all_golds.count("IMPLEMENTED") == 16
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_real.py -v`
Expected: the four new tests FAIL with `AttributeError: module 'omitbench.real'
has no attribute 'score'`.

- [ ] **Step 3: Implement `score()`**

Append to `omitbench/real.py`:

```python
def score(instances: list[RealInstance], detectors: dict | None = None) -> list[dict]:
    """Run every detector over every real instance. Same call shape as
    experiment.py's scoring loop: fn(spec, reqs, before, after, ctx={}).
    gold is looked up from RealInstance.gold and attached to the OUTPUT
    record only -- it is never passed into the detector call, so this
    reuses the exact leakage guarantee tests/test_leakage.py already
    proves for the synthetic pipeline.
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_real.py -v`
Expected: all 10 tests PASS.

- [ ] **Step 5: Report the gold-label distribution to the user now, before
  touching analyze.py**

Run:
```bash
python3 -c "
from omitbench import real as R
insts = R.load_real()
all_g = [g for i in insts for g in i.gold.values()]
print(f'{len(insts)} trajectories, {len(all_g)} requirements')
print(f'IMPLEMENTED: {all_g.count(\"IMPLEMENTED\")}  OMITTED: {all_g.count(\"OMITTED\")}')
for i in insts:
    print(f'  {i.iid:10} {i.repo:12} {len(i.reqs)} reqs, ' +
          f'{list(i.gold.values()).count(\"OMITTED\")} OMITTED')
"
```
Paste this output into the conversation verbatim before proceeding to Task 3.

- [ ] **Step 6: Commit**

```bash
git add omitbench/real.py tests/test_real.py
git commit -m "feat: score real trajectories against registered detectors"
```

---

### Task 3: `scripts/analyze.py --source {synthetic,real,both}`

**Files:**
- Modify: `scripts/analyze.py`

**Interfaces:**
- Consumes: `omitbench.real.load_real`, `omitbench.real.score`,
  `omitbench.real.RealSchemaError`, `omitbench.detectors.DETECTORS`
  (Task 1/2). Existing `metrics(rows)`, `boot_indices(iids, n, seed)`,
  `ORDER`, `load()` in `analyze.py` itself.
- Produces: `def analyze_synthetic() -> dict[str, dict]` (detector name ->
  its `metrics()` dict, for the "both" side-by-side column) and `def
  analyze_real(synthetic_summary: dict | None = None) -> None`.

- [ ] **Step 1: Extract the existing `main()` body into `analyze_synthetic()`**

At the top of `scripts/analyze.py`, add the import block (matches
`scripts/run_judge.py`'s convention):

```python
import argparse
import sys

sys.path.insert(0, ".")
```

Rename the body of the current `main()` to `analyze_synthetic()`, keep
every line of its logic and print statements **exactly as they are today**,
and add `return summary` as its last line (the `summary` dict already exists
at that point in the function — see `scripts/analyze.py:99-108`).

Add a thin new `main()`:

```python
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
```

- [ ] **Step 2: Verify the refactor changed nothing**

Run: `python3 scripts/analyze.py > /tmp/after.txt 2>&1`
Compare against a copy of the output taken *before* this task's edits
(`git stash`, run, save as `/tmp/before.txt`, `git stash pop`, run, diff):
```bash
diff /tmp/before.txt /tmp/after.txt
```
Expected: no output (identical). This is not optional — it is the proof the
extraction didn't change the frozen synthetic path.

- [ ] **Step 3: Write `_synthetic_mcc_summary()` and `analyze_real()`**

```python
def _synthetic_mcc_summary() -> dict[str, dict]:
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

    if any(inst.gold.get(r) is None for inst in instances for r in inst.reqs):
        pass  # unreachable: load_real() already guarantees every req has a gold

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
        print("Precision and FPR ARE well-defined (there is a negative "
              "class) and are the only real signal in the table below.")
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
    header = f"{'detector':28}{'P':>6}{'R':>6}{'F1':>6}{'MCC':>7}{'  MCC 95% CI':>18}{'FPR':>7}"
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
        print("\n(Re-read the WARNING above: MCC/recall/F1 in this table "
              "are 0-by-convention, not a demonstrated result.)")
```

- [ ] **Step 4: Manual smoke test — real-only**

Run: `python3 scripts/analyze.py --source real`
Expected: gold-label distribution block prints FIRST, then the WARNING
(0 OMITTED across 16 requirements), then a table with `P`, `FPR` populated
and `R`, `MCC` at 0.000 for every detector, with a `synthetic MCC` column
alongside.

- [ ] **Step 5: Manual smoke test — both**

Run: `python3 scripts/analyze.py --source both`
Expected: the untouched synthetic table first, then the real section from
Step 4, in one run.

- [ ] **Step 6: Manual smoke test — default unchanged**

Run: `python3 scripts/analyze.py` and re-diff against `/tmp/before.txt`
from Step 2. Expected: still identical.

- [ ] **Step 7: Commit**

```bash
git add scripts/analyze.py
git commit -m "feat: add --source {synthetic,real,both} to analyze.py (T4)"
```

---

### Task 4: Full verification protocol

**Files:** none (verification only, per CLAUDE.md's required protocol).

- [ ] **Step 1:** `make test` — run, paste the actual output (test count,
  pass/fail), not a summary.
- [ ] **Step 2:** `make analyze` — run, paste the actual headline table.
- [ ] **Step 3:** Compare against `results/pilot.json`. State explicitly:
  "no metric moved" (expected, since no detector or shard changed) or name
  which one did and by how much, if any.
- [ ] **Step 4:** `python3 scripts/analyze.py --source real` and `--source
  both` — run both, paste actual output, including the gold-distribution
  block and WARNING.
- [ ] **Step 5:** `grep -rn "load_real\|RealInstance\|def score" omitbench/
  scripts/` to show the loader is actually wired into `analyze.py`, not just
  present as a dead file.
- [ ] **Step 6:** Report to the user: gold-label distribution (again, in
  the final summary), the fact that recall/MCC are degenerate on this
  sample, and precision/FPR as the only defined real-corpus signal, exactly
  as computed — no rounding toward a nicer story.

---

## Self-Review Notes

- **Spec coverage:** (1) loader → Task 1. (2) validator, fails loudly,
  lists every malformed file → Task 1. (3) `--source` flag → Task 3;
  gold distribution first → Task 3 Step 3 (printed before scoring);
  small-n MCC CI stated explicitly → Task 3 Step 3 `n = 8 instances` line;
  precision/FPR reported separately from recall → same table, WARNING
  text calls out which columns are meaningful; side-by-side synthetic MCC
  → `synthetic MCC` column. "Report gold distribution before running
  anything else" → Task 2 Step 5, done standalone before Task 3 exists.
- **No detector/pilot.json/shard edits** anywhere in this plan — confirmed
  by file list (only `omitbench/real.py`, `tests/test_real.py`,
  `scripts/analyze.py` touched).
- **Placeholder scan:** none found — every step has complete code.
- **Type consistency:** `RealInstance` fields (`iid, repo, spec, reqs,
  before, after, gold`) used identically in Task 1's tests, Task 2's
  `score()`, and Task 3's `analyze_real()`. `score()`'s return-dict keys
  (`iid, repo, req, detector, pred, gold, source`) match what
  `analyze_real()` reads (`r["detector"]`, `r["iid"]`, `(r["pred"],
  r["gold"])`).
