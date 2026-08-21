"""
T5 (TASKS.md) Step 4: re-run P1 detection with EXTRACTED requirements
instead of oracle requirements, paired on the same 310 instances already
scored in results/shards/*.jsonl, and report the F1/MCC drop ("extraction
tax").

Three conditions, all on the SAME 310 instances / SAME variants as the
existing shards:

  ORACLE     -- already computed. Read straight from results/shards/*.jsonl,
                detector == "P1 defined". Not re-run.
  EXTRACTED-A (literal pipeline) -- the LLM's extracted (symbol, path) pairs
                fed to P1 exactly as extracted, wrong/null paths included.
                PRIMARY / headline extraction-tax number.
  EXTRACTED-B (symbol-identification-only) -- same extracted symbols, but for
                any item whose symbol matches ANY oracle requirement's symbol
                for that instance, the oracle's true path is substituted in
                before scoring. Non-matching items are left as extracted.
                SECONDARY diagnostic: isolates whether the tax is driven by
                requirement-UNDERSTANDING (wrong/hallucinated symbol) or
                PATH-ATTRIBUTION (right symbol, wrong file).

GROUND TRUTH for an extracted item is SYMBOL-based, not string-based: an
item is gold OMITTED iff the active variant is a real mutation (cls !=
CLEAN) AND item["symbol"] == the mutated target's symbol -- mirrors
experiment.py's own rule (`gold = "OMITTED" if (cls != "CLEAN" and r ==
target) else "IMPLEMENTED"`), just comparing on the symbol alone since an
extracted item's PATH may be wrong -- ground truth about whether a symbol
was actually omitted from the diff does not depend on whether the extractor
guessed its file correctly. This is also exactly why condition A can look
much worse than the extractor's own precision suggests: P1 (d_defined)
looks the symbol up at the (possibly wrong) extracted path, so a
symbol-right/path-wrong item is looked up in the WRONG place and is
therefore reported OMITTED almost by construction, regardless of the
variant's true content -- this is the real, structural cost of the
extraction tax, not a scoring artifact.

SYNTHETIC MISS ROW. If a mutated variant's target symbol was never named by
ANY extracted item at all, the detector never got asked about it -- a real
deployment would silently miss it. One synthetic (pred=IMPLEMENTED,
gold=OMITTED) row is added for that variant so recall reflects this. Without
it, recall would be computed only over targets the extractor happened to
mention, silently hiding the worst failure mode (never proposing the
omitted requirement as a requirement at all).

DIAGNOSTIC SUBSET (secondary, per user instruction -- NOT the headline).
Mirrors Step 3's "non-test-path oracle reqs only" split: restricted to the
same 76 instances that have >=1 non-test-path oracle requirement, and within
those, a mutated variant only counts as an in-scope omission event if the
mutated TARGET itself is a non-test-path symbol (a mutation that hollowed a
test function is not a "production omission" under this diagnostic's
definition). CLEAN variants are scored in the diagnostic exactly as in the
primary table.
"""

from __future__ import annotations

import glob
import gc
import json
import random
import sys
from collections import defaultdict

sys.path.insert(0, ".")

from omitbench import experiment as E
from omitbench import mutate as M
from omitbench import detectors as P

REPOS = ["corpus/click", "corpus/flask", "corpus/requests", "corpus/attrs",
         "corpus/jinja", "corpus/werkzeug", "corpus/httpx", "corpus/itsdangerous"]
PER_REPO = 40
SCAN_CAP = 600
SEED = 0

P1 = P.d_defined


def is_test_path(path: str) -> bool:
    return "test" in path.lower()


def split(req: str) -> tuple[str, str]:
    path, _, sym = req.rpartition("::")
    return path, sym


# --------------------------------------------------------------------------
# 1. oracle P1 rows -- read straight from the frozen shards, never re-run.
# --------------------------------------------------------------------------

def load_oracle_p1() -> dict[str, list[tuple[str, str]]]:
    by_iid = defaultdict(list)
    for fp in sorted(glob.glob("results/shards/*.jsonl")):
        if "llm_judge" in fp:
            continue
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                if r["detector"] == "P1 defined":
                    by_iid[r["iid"]].append((r["pred"], r["gold"]))
    return dict(by_iid)


# --------------------------------------------------------------------------
# 2. extracted items, loaded from results/extraction/extracted.jsonl
#    (Step 3's output -- already spent, never re-called here).
# --------------------------------------------------------------------------

def load_extracted() -> dict[str, dict]:
    out = {}
    with open("results/extraction/extracted.jsonl") as f:
        for line in f:
            r = json.loads(line)
            oracle_pairs = [split(q) for q in r["reqs"]]
            sym_to_path = {}
            for pth, sym in oracle_pairs:
                sym_to_path.setdefault(sym, pth)  # first occurrence wins
            nontest_reqs = [q for q in r["reqs"] if not is_test_path(split(q)[0])]
            out[r["iid"]] = {
                "items": r["items"],
                "oracle_reqs": r["reqs"],
                "sym_to_path": sym_to_path,
                "has_nontest": bool(nontest_reqs),
            }
    return out


# --------------------------------------------------------------------------
# 3. score conditions A/B against every rebuilt instance/variant.
# --------------------------------------------------------------------------

def score_instance(items, sym_to_path, target_path, target_symbol, cls, before, after,
                    diagnostic: bool):
    """-> (rows_a, rows_b) for ONE (instance, variant). diagnostic=True
    restricts what counts as an in-scope omission to non-test-path targets
    (see module docstring); it does not filter which extracted items are
    scored, only whether a mutated variant contributes an OMITTED-gold row.
    """
    in_scope_omission = (cls != "CLEAN") and not (diagnostic and is_test_path(target_path))

    rows_a, rows_b = [], []
    matched_target_a = False
    matched_target_b = False

    for it in items:
        sym = it["symbol"]
        gold = "OMITTED" if (in_scope_omission and sym == target_symbol) else "IMPLEMENTED"
        if in_scope_omission and sym == target_symbol:
            matched_target_a = True

        # A: literal -- extracted path exactly as given, wrong/null included.
        path_a = it.get("path") or ""
        req_a = f"{path_a}::{sym}"
        pred_a = P1("", [req_a], before, after, {})[req_a]
        rows_a.append((pred_a, gold))

        # B: symbol-identification-only -- ONLY items whose symbol matches
        # some oracle requirement's symbol are scored at all, with the
        # oracle's true path substituted. A non-matching item (the extractor
        # named something with no basis in the real requirement list) is
        # NOT a path-attribution failure -- it's a requirement-understanding
        # failure, which is exactly what B is supposed to hold constant and
        # exclude, not re-score with a broken path.
        oracle_path = sym_to_path.get(sym)
        if oracle_path is not None:
            req_b = f"{oracle_path}::{sym}"
            pred_b = P1("", [req_b], before, after, {})[req_b]
            rows_b.append((pred_b, gold))
            if in_scope_omission and sym == target_symbol:
                matched_target_b = True

    if in_scope_omission and not matched_target_a:
        # extractor never named the true omitted symbol at all -- a real
        # deployment would silently miss it; count it as a miss, not a
        # scoring gap.
        rows_a.append(("IMPLEMENTED", "OMITTED"))
    if in_scope_omission and not matched_target_b:
        rows_b.append(("IMPLEMENTED", "OMITTED"))

    return rows_a, rows_b


def build_extraction_rows(extracted: dict) -> tuple[dict, dict, dict, dict]:
    """Rebuilds all 310 instances (same repos/per-repo/scan-cap/seed as
    run_shards.sh -- verified iid-identical to results/shards/*.jsonl before
    this script exists to rely on it) and scores conditions A/B, primary and
    diagnostic. Returns (by_iid_a, by_iid_b, by_iid_a_diag, by_iid_b_diag).
    """
    by_iid_a = defaultdict(list)
    by_iid_b = defaultdict(list)
    by_iid_a_diag = defaultdict(list)
    by_iid_b_diag = defaultdict(list)

    rng = random.Random(SEED)
    for repo in REPOS:
        insts = E.build([repo], PER_REPO, SEED, SCAN_CAP)
        for iid, rname, spec, before, target, oracle_reqs, variants in insts:
            ex = extracted.get(iid)
            if ex is None:
                continue  # should not happen -- iid sets verified identical
            items = ex["items"]
            sym_to_path = ex["sym_to_path"]
            target_path, target_symbol = split(target)
            diag_eligible = ex["has_nontest"]

            for cls, after in variants:
                ra, rb = score_instance(items, sym_to_path, target_path, target_symbol,
                                         cls, before, after, diagnostic=False)
                by_iid_a[iid].extend(ra)
                by_iid_b[iid].extend(rb)

                if diag_eligible:
                    rad, rbd = score_instance(items, sym_to_path, target_path, target_symbol,
                                               cls, before, after, diagnostic=True)
                    by_iid_a_diag[iid].extend(rad)
                    by_iid_b_diag[iid].extend(rbd)
        del insts
        for memo in (M._TREE_MEMO, M._DEF_MEMO, M._CALL_MEMO, M._TARGETS_MEMO):
            memo.clear()
        gc.collect()
        print(f"  {repo} done", file=sys.stderr)

    return dict(by_iid_a), dict(by_iid_b), dict(by_iid_a_diag), dict(by_iid_b_diag)


# --------------------------------------------------------------------------
# 4. metrics + paired cluster bootstrap (same convention as
#    experiment.py::metrics/cluster_bootstrap and analyze.py::boot_indices).
# --------------------------------------------------------------------------

def metrics(rows):
    tp = fp = fn = tn = 0
    for pred, gold in rows:
        p, g = pred == "OMITTED", gold == "OMITTED"
        tp += p and g; fp += p and not g
        fn += (not p) and g; tn += (not p) and (not g)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    import math
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "mcc": mcc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def boot_indices(iids, n=1000, seed=0):
    rng = random.Random(seed)
    return [[rng.choice(iids) for _ in iids] for _ in range(n)]


def ci_for(by_iid, iids, B, metric):
    vals = sorted(metrics([x for i in samp for x in by_iid.get(i, [])])[metric]
                   for samp in B)
    n = len(vals)
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def paired_diff_ci(by_iid_x, by_iid_y, iids, B, metric="mcc"):
    diffs = []
    for samp in B:
        mx = metrics([r for i in samp for r in by_iid_x.get(i, [])])[metric]
        my = metrics([r for i in samp for r in by_iid_y.get(i, [])])[metric]
        diffs.append(mx - my)
    diffs.sort()
    n = len(diffs)
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def report_condition(name, by_iid, iids, B):
    rows = [x for i in iids for x in by_iid.get(i, [])]
    m = metrics(rows)
    mcc_lo, mcc_hi = ci_for(by_iid, iids, B, "mcc")
    f1_lo, f1_hi = ci_for(by_iid, iids, B, "f1")
    n_rows = len(rows)
    n_covered = sum(1 for i in iids if by_iid.get(i))
    print(f"{name:34} n_rows={n_rows:5} n_instances_with_rows={n_covered:4}")
    print(f"  P={m['precision']:.3f} R={m['recall']:.3f} "
          f"F1={m['f1']:.3f} [{f1_lo:.3f},{f1_hi:.3f}]  "
          f"MCC={m['mcc']:.3f} [{mcc_lo:.3f},{mcc_hi:.3f}]  "
          f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']})")
    m = dict(m)
    m["mcc_ci"] = [mcc_lo, mcc_hi]
    m["f1_ci"] = [f1_lo, f1_hi]
    m["n_rows"] = n_rows
    m["n_instances_with_rows"] = n_covered
    return m


def main():
    print("loading oracle P1 rows from results/shards/*.jsonl ...", file=sys.stderr)
    oracle_by_iid = load_oracle_p1()
    print(f"  {len(oracle_by_iid)} instances", file=sys.stderr)

    print("loading extracted items from results/extraction/extracted.jsonl ...",
          file=sys.stderr)
    extracted = load_extracted()
    print(f"  {len(extracted)} instances", file=sys.stderr)

    print("rebuilding instances + scoring conditions A/B (no API calls) ...",
          file=sys.stderr)
    by_iid_a, by_iid_b, by_iid_a_diag, by_iid_b_diag = build_extraction_rows(extracted)

    all_iids = sorted(oracle_by_iid)
    diag_iids = sorted(i for i in all_iids if extracted.get(i, {}).get("has_nontest"))
    print(f"\nprimary n_instances={len(all_iids)}  "
          f"diagnostic (non-test-path subset) n_instances={len(diag_iids)}\n")

    B_all = boot_indices(all_iids, 1000, 0)
    B_diag = boot_indices(diag_iids, 1000, 0)

    print("=" * 100)
    print("PRIMARY TABLE -- all 310 instances (same corpus as results/pilot.json's P1 row)")
    print("=" * 100)
    m_oracle = report_condition("ORACLE (existing, unchanged)", oracle_by_iid, all_iids, B_all)
    m_a = report_condition("EXTRACTED-A literal (headline tax)", by_iid_a, all_iids, B_all)
    m_b = report_condition("EXTRACTED-B symbol-only (diagnostic)", by_iid_b, all_iids, B_all)

    print("\nPAIRED cluster bootstrap of MCC difference (95% CI), same B draws:")
    lo, hi = paired_diff_ci(oracle_by_iid, by_iid_a, all_iids, B_all, "mcc")
    obs = m_oracle["mcc"] - m_a["mcc"]
    sig = "" if lo <= 0 <= hi else "  *"
    print(f"  ORACLE - EXTRACTED-A(literal)      {obs:>+7.3f}  [{lo:>+6.3f},{hi:>+6.3f}]{sig}")
    paired_a = {"obs": obs, "ci": [lo, hi], "excludes_zero": not (lo <= 0 <= hi)}
    lo, hi = paired_diff_ci(oracle_by_iid, by_iid_b, all_iids, B_all, "mcc")
    obs = m_oracle["mcc"] - m_b["mcc"]
    sig = "" if lo <= 0 <= hi else "  *"
    print(f"  ORACLE - EXTRACTED-B(symbol-only)  {obs:>+7.3f}  [{lo:>+6.3f},{hi:>+6.3f}]{sig}")
    paired_b = {"obs": obs, "ci": [lo, hi], "excludes_zero": not (lo <= 0 <= hi)}

    print("\n" + "=" * 100)
    print(f"SECONDARY / DIAGNOSTIC TABLE -- non-test-path oracle-req subset only "
          f"(n={len(diag_iids)}) -- NOT the headline")
    print("=" * 100)
    oracle_diag = {i: oracle_by_iid[i] for i in diag_iids}
    m_oracle_d = report_condition("ORACLE (same subset)", oracle_diag, diag_iids, B_diag)
    m_a_d = report_condition("EXTRACTED-A literal", by_iid_a_diag, diag_iids, B_diag)
    m_b_d = report_condition("EXTRACTED-B symbol-only", by_iid_b_diag, diag_iids, B_diag)

    print("\nPAIRED cluster bootstrap of MCC difference (95% CI), diagnostic subset:")
    lo, hi = paired_diff_ci(oracle_diag, by_iid_a_diag, diag_iids, B_diag, "mcc")
    obs = m_oracle_d["mcc"] - m_a_d["mcc"]
    sig = "" if lo <= 0 <= hi else "  *"
    print(f"  ORACLE - EXTRACTED-A(literal)      {obs:>+7.3f}  [{lo:>+6.3f},{hi:>+6.3f}]{sig}")
    paired_a_diag = {"obs": obs, "ci": [lo, hi], "excludes_zero": not (lo <= 0 <= hi)}
    lo, hi = paired_diff_ci(oracle_diag, by_iid_b_diag, diag_iids, B_diag, "mcc")
    obs = m_oracle_d["mcc"] - m_b_d["mcc"]
    sig = "" if lo <= 0 <= hi else "  *"
    print(f"  ORACLE - EXTRACTED-B(symbol-only)  {obs:>+7.3f}  [{lo:>+6.3f},{hi:>+6.3f}]{sig}")
    paired_b_diag = {"obs": obs, "ci": [lo, hi], "excludes_zero": not (lo <= 0 <= hi)}

    # dump raw numbers for the README table, so nothing gets hand-typed
    out = {
        "primary": {"n_instances": len(all_iids),
                     "oracle": m_oracle, "extracted_a": m_a, "extracted_b": m_b,
                     "paired_mcc_diff": {"oracle_minus_a": paired_a, "oracle_minus_b": paired_b}},
        "diagnostic_nontest": {"n_instances": len(diag_iids),
                     "oracle": m_oracle_d, "extracted_a": m_a_d, "extracted_b": m_b_d,
                     "paired_mcc_diff": {"oracle_minus_a": paired_a_diag, "oracle_minus_b": paired_b_diag}},
    }
    with open("results/extraction/detection_tax.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/extraction/detection_tax.json")


if __name__ == "__main__":
    main()
