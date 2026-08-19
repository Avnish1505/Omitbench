"""
Tests for omitbench/real.py -- the loader for hand-labelled real-agent
trajectories (TASKS.md T4).

Pins two things: the schema validator fails loudly and lists every malformed
file (not just the first), and score() calls detectors with the exact same
5-positional-argument contract as experiment.py, with gold kept out of the
call -- same leakage guarantee tests/test_leakage.py proves for the
synthetic pipeline, exercised here for the real-data path instead.
"""

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


# --------------------------------------------------------------------------
# schema validation
# --------------------------------------------------------------------------

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


def test_duplicate_requirement_key_is_rejected(tmp_path):
    bad = dict(VALID)
    bad["requirements"] = [
        {"req": "n.py::g", "gold": "IMPLEMENTED"},
        {"req": "n.py::g", "gold": "OMITTED"},
    ]
    write(tmp_path, "real_001.json", bad)
    with pytest.raises(R.RealSchemaError) as exc:
        R.load_real(str(tmp_path / "real_*.json"))
    assert "duplicate" in str(exc.value).lower()


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


# --------------------------------------------------------------------------
# score()
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# the actual committed corpus
# --------------------------------------------------------------------------

def test_actual_real_corpus_gold_distribution():
    """Locks in the finding that must be reported before any recall/MCC
    number: as of the 8 committed trajectories, the positive class
    (OMITTED) is EMPTY. This is a fact about data/real/*.json, not a tuned
    assertion -- if it changes because trajectories were added or
    relabelled, this test SHOULD fail so the change is visible, not a
    reason to weaken the assertion.
    """
    instances = R.load_real()
    assert len(instances) == 8
    all_golds = [g for inst in instances for g in inst.gold.values()]
    assert len(all_golds) == 16
    assert all_golds.count("OMITTED") == 0
    assert all_golds.count("IMPLEMENTED") == 16
