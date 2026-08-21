"""
T5 (TASKS.md): formats the README's extraction-tax tables from the
committed outputs of the T5 pipeline. Prints, never hand-types -- mirrors
scripts/analyze.py's role for the headline table.

Reads (never recomputes):
  results/extraction/extraction_prf1.json   -- Step 3: extractor P/R/F1
                                                 vs oracle (score_extraction.py)
  results/extraction/detection_tax.json     -- Step 4: P1 under oracle vs
                                                 extracted requirements
                                                 (run_extraction_tax.py)

To regenerate those two files from scratch (re-spends nothing -- the LLM
calls are content-addressed and cached under cache/extract_*.json, committed
to git the same way B4/B5/B6's cache is):

    python3 -m omitbench.extract          # not a CLI; see extract_one()
    python3 scripts/run_extraction_tax.py # rebuilds the 310 instances
                                           # locally, no API calls, and
                                           # writes detection_tax.json
"""

from __future__ import annotations

import json


def load():
    with open("results/extraction/extraction_prf1.json") as f:
        prf1 = json.load(f)
    with open("results/extraction/detection_tax.json") as f:
        tax = json.load(f)
    return prf1, tax


def fmt_ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def print_prf1_table(prf1):
    print("STEP 3 -- extractor precision/recall vs oracle requirements "
          f"(n={prf1['n_instances']} instances, micro-averaged, "
          "cluster bootstrap 95% CI)")
    print(f"  {prf1['n_nonempty_extraction']}/{prf1['n_instances']} instances "
          f"got >=1 extracted item; {prf1['n_empty_extraction']} got an "
          "honest empty extraction.\n")
    header = f"{'match type':16}{'P':>8}{'R':>8}{'F1':>8}   95% CI (F1)"
    print(header)
    print("-" * len(header))
    for label, key in [("symbol-only", "symbol_only"), ("path-qualified", "path_qualified")]:
        m = prf1["all_oracle_reqs"][key]
        print(f"{label:16}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}   "
              f"{fmt_ci(*m['ci']['f1'])}")
    print()

    nt = prf1["nontest_oracle_reqs_only"]
    print(f"  diagnostic -- oracle reqs restricted to non-test-path symbols "
          f"only (n={nt['n_instances']} instances):")
    for label, key in [("symbol-only", "symbol_only"), ("path-qualified", "path_qualified")]:
        m = nt[key]
        print(f"  {label:14}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}   "
              f"{fmt_ci(*m['ci']['f1'])}")


def print_tax_table(tax):
    for section, label in [("primary", "PRIMARY -- all 310 instances"),
                            ("diagnostic_nontest", "SECONDARY / DIAGNOSTIC -- "
                             "non-test-path oracle-req subset only")]:
        d = tax[section]
        print(f"\nSTEP 4 -- {label} (n={d['n_instances']} instances)")
        header = f"{'condition':38}{'P':>7}{'R':>7}{'F1':>7}{'MCC':>8}"
        print(header)
        print("-" * len(header))
        for cond_key, cond_label in [
            ("oracle", "ORACLE requirements (existing)"),
            ("extracted_a", "EXTRACTED-A literal (headline tax)"),
            ("extracted_b", "EXTRACTED-B symbol-only (diagnostic)"),
        ]:
            m = d[cond_key]
            print(f"{cond_label:38}{m['precision']:>7.3f}{m['recall']:>7.3f}"
                  f"{m['f1']:>7.3f}{m['mcc']:>8.3f}")


def main():
    prf1, tax = load()
    print_prf1_table(prf1)
    print_tax_table(tax)


if __name__ == "__main__":
    main()
