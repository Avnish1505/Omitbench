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
