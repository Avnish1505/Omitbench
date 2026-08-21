"""
T5 (TASKS.md): score the LLM extractor (omitbench/extract.py) against the
oracle requirement list (mutate.py::new_symbols, via experiment.py).

MATCHING METHOD (frozen before any prompt iteration -- see extract.py and
ASSUMPTIONS.md's entry on this module):

  - Two match types, BOTH computed and BOTH reported, always:
      1. symbol-only    -- extracted "symbol" equals an oracle symbol,
                            exactly, case-sensitive. Ignores path entirely.
      2. path-qualified -- extracted (path, symbol) equals an oracle
                            (path, symbol) pair, exactly. Strictly harder.
  - Multiset (Counter) comparison, not set -- caps credit at
    min(count_oracle, count_extracted) per name, so two same-named symbols
    in different files can't be double-credited by one extracted guess.
  - Aggregation is MICRO-averaged: pool TP/FP/FN across every instance,
    THEN compute one P/R/F1 -- the same convention as
    omitbench/experiment.py::cells()/metrics(). Never per-instance-averaged.
  - Cluster bootstrap CI over INSTANCES (iid), not records -- CLAUDE.md
    rule 4; mirrors experiment.py::cluster_bootstrap.
  - Pure Python set/multiset arithmetic. NO LLM judge anywhere in this
    path -- judging LLM-extracted text with another LLM would reintroduce
    the exact noise source T5 exists to isolate from. The oracle is
    structured (path, symbol) pairs, so this is sufficient.

KNOWN LIMITATION of symbol-only matching, stated up front (also in
ASSUMPTIONS.md): it over-credits an extractor that names the right function
but attaches it to the wrong file (or no file) -- real signal about what the
LLM understood, but blind to WHERE. That is exactly why path-qualified is
reported as a separate, stricter number and never folded into symbol-only.
"""

from __future__ import annotations

import random
from collections import Counter


def split_req(req: str) -> tuple[str, str]:
    path, _, sym = req.rpartition("::")
    return path, sym


def _multiset_cells(oracle: Counter, extracted: Counter) -> tuple[int, int, int]:
    """-> (tp, fp, fn) for one instance, one match type, via Counter
    intersection/difference (Python's Counter already implements
    min()/max()-per-key semantics for &/-)."""
    tp = sum((oracle & extracted).values())
    fp = sum((extracted - oracle).values())
    fn = sum((oracle - extracted).values())
    return tp, fp, fn


def per_instance_cells(oracle_reqs: list[str], extracted_items: list[dict]) -> dict:
    """oracle_reqs: ["path::symbol", ...] (experiment.py's `reqs` for one
    instance). extracted_items: [{"symbol": str, "path": str|None}, ...]
    (extract.py::extract_one's return for that instance's task_text).

    -> {"symbol_only": (tp, fp, fn), "path_qualified": (tp, fp, fn)}
    """
    oracle_pairs = [split_req(r) for r in oracle_reqs]

    oracle_sym = Counter(sym for _, sym in oracle_pairs)
    extracted_sym = Counter(it["symbol"] for it in extracted_items)

    oracle_pq = Counter(oracle_pairs)
    extracted_pq = Counter((it.get("path"), it["symbol"]) for it in extracted_items)
    # NOTE: Counter key order above is (path, symbol) to match oracle_pairs'
    # (path, symbol) order from split_req. A `path: null` extraction becomes
    # key (None, symbol) -- it can never equal a real oracle pair, since
    # every oracle path is a real non-null string (verbatim from the diff).
    # That is the point: null is an honest abstention, scored as a miss on
    # path-qualified but still eligible for symbol-only credit.

    return {
        "symbol_only": _multiset_cells(oracle_sym, extracted_sym),
        "path_qualified": _multiset_cells(oracle_pq, extracted_pq),
    }


def micro_prf1(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def score(by_iid: dict[str, dict]) -> dict:
    """by_iid: {iid: {"symbol_only": (tp,fp,fn), "path_qualified": (tp,fp,fn)}}
    (i.e. per_instance_cells()'s output, keyed by instance).

    -> {"symbol_only": {...micro P/R/F1..., "n_instances": N},
        "path_qualified": {...}}
    No CI here -- see cluster_bootstrap_ci below, kept separate so a caller
    that only wants the point estimate doesn't pay the bootstrap cost.
    """
    out = {}
    for match_type in ("symbol_only", "path_qualified"):
        tp = sum(c[match_type][0] for c in by_iid.values())
        fp = sum(c[match_type][1] for c in by_iid.values())
        fn = sum(c[match_type][2] for c in by_iid.values())
        m = micro_prf1(tp, fp, fn)
        m["n_instances"] = len(by_iid)
        out[match_type] = m
    return out


def cluster_bootstrap_ci(by_iid: dict[str, dict], match_type: str, metric: str,
                          n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Resample INSTANCES with replacement (CLAUDE.md rule 4 / ASSUMPTIONS.md
    #5 -- an instance's oracle requirements and extracted items are not
    independent draws; a record-level bootstrap here isn't even meaningful
    since there IS no sub-instance record, but resampling anything other
    than instances would still misrepresent the correlation structure).
    Mirrors experiment.py::cluster_bootstrap.
    """
    rng = random.Random(seed)
    iids = list(by_iid)
    if not iids:
        return (0.0, 0.0)
    vals = []
    for _ in range(n):
        tp = fp = fn = 0
        for _ in iids:
            c = by_iid[rng.choice(iids)][match_type]
            tp += c[0]; fp += c[1]; fn += c[2]
        vals.append(micro_prf1(tp, fp, fn)[metric])
    vals.sort()
    return (vals[int(0.025 * n)], vals[int(0.975 * n)])


def report(by_iid: dict[str, dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """Point estimates + 95% cluster-bootstrap CI for precision/recall/f1,
    both match types. This is the full Step 3 result."""
    point = score(by_iid)
    out = {}
    for match_type, m in point.items():
        ci = {}
        for metric in ("precision", "recall", "f1"):
            lo, hi = cluster_bootstrap_ci(by_iid, match_type, metric, n_boot, seed)
            ci[metric] = (lo, hi)
        out[match_type] = {**m, "ci": ci}
    return out
