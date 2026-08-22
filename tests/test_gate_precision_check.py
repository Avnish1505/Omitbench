from __future__ import annotations

import sys

sys.path.insert(0, ".")

from scripts.build_baseline_t6 import build


def test_build_baseline_t6_has_required_fields_and_sane_values():
    out = build()
    assert out["detector"] == "P1 defined"
    assert out["acceptance_threshold"] == 0.80
    assert 0.0 <= out["precision"] <= 1.0
    lo, hi = out["precision_ci95"]
    assert 0.0 <= lo <= out["precision"] <= hi <= 1.0
    assert out["n_instances"] > 0


# --------------------------------------------------------------------------
# Task 5: precision regression check
# --------------------------------------------------------------------------

import json
import re

from scripts.check_gate_precision import STATUS_END, STATUS_START, status_block


def test_status_block_active_mentions_precision_and_threshold():
    text = status_block(active=True, precision=0.8123)
    assert STATUS_START in text and STATUS_END in text
    assert "active" in text
    assert "0.8123" in text


def test_status_block_suspended_says_suspended_and_why():
    text = status_block(active=False, precision=0.71)
    assert "suspended" in text.lower()
    assert "precision" in text.lower()
    assert "0.7100" in text


def test_main_exit_code_reflects_active_status(monkeypatch, tmp_path):
    import scripts.check_gate_precision as C

    monkeypatch.setattr(C, "current_precision", lambda: 0.85)
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{STATUS_START}\nold\n{STATUS_END}\nafter\n")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()

    rc = C.main()

    assert rc == 0
    status = json.loads((tmp_path / "results" / "gate_status.json").read_text())
    assert status["status"] == "active"
    assert STATUS_START in readme.read_text()
    # word-boundary check, not substring -- "old" is a substring of
    # "threshold", which the new active-status block legitimately contains
    assert not re.search(r"\bold\b", readme.read_text())


def test_main_exit_code_reflects_suspended_status(monkeypatch, tmp_path):
    import scripts.check_gate_precision as C

    monkeypatch.setattr(C, "current_precision", lambda: 0.5)
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{STATUS_START}\nold\n{STATUS_END}\nafter\n")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()

    rc = C.main()

    assert rc == 1
    status = json.loads((tmp_path / "results" / "gate_status.json").read_text())
    assert status["status"] == "suspended"
    assert "suspended" in readme.read_text().lower()
