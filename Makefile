.PHONY: help corpus pilot analyze test reproduce clean judge-dry-run judge-check-alignment gate-baseline gate-check gate-bench jev-dry-run jev-smoke jev calibration

help:
	@echo "corpus     clone the 9 evaluation repos (~93MB, needs network)"
	@echo "pilot      run all shards -> results/shards/*.jsonl  (~25 min)"
	@echo "analyze    regenerate every number in the README"
	@echo "test       run the test suite (leakage guards must pass)"
	@echo "reproduce  test + analyze from committed shards (no network, <1 min)"
	@echo "judge-dry-run        estimate B4/B5/B6 call count and spend, no API call"
	@echo "judge-check-alignment  verify judge corpus iids are a subset of the"
	@echo "                       shards' -- run after pilot, before a real sweep"
	@echo "gate-baseline  regenerate results/baseline_t6.json (T6, run after pilot)"
	@echo "gate-check     T6 falsification check -- suspend the gate if P1"
	@echo "               precision drops below 0.80 on the current shards"
	@echo "gate-bench     T6 latency benchmark (p50/p95) against corpus/* repos"
	@echo "jev-dry-run    B7 (TypeSafe Jev): exact call count and cost, no API call"
	@echo "jev-smoke      B7 on 5 instances (needs TYPESAFE_API_KEY) -- run first"
	@echo "jev            B7 full sweep, both pre-registered variants"
	@echo "calibration    B7 Brier / ECE / reliability -> results/calibration_b7.json"

corpus:
	bash scripts/fetch_corpus.sh

pilot:
	bash scripts/run_shards.sh

analyze:
	python3 scripts/analyze.py

# The claim of this repo is that its numbers regenerate. This target is that
# claim, executable. It needs no network and no corpus: shards are committed.
reproduce: test
	python3 scripts/analyze.py | tee results/REPRODUCED.txt
	@echo "\nCompare against results/pilot.json (frozen before any tuning)."

test:
	python3 -m pytest tests/ -q

judge-dry-run:
	python3 scripts/run_judge.py --dry-run

# black is excluded here as it is from results/shards/* -- see ASSUMPTIONS.md.
# Including it would make this fail spuriously (0 black entries in the
# shards to ever be a superset of) for a reason that has nothing to do with
# the seed/scan-cap/per-repo mismatch this check exists to catch.
judge-check-alignment:
	python3 scripts/run_judge.py --check-alignment --repos \
	    corpus/click corpus/flask corpus/jinja corpus/werkzeug \
	    corpus/itsdangerous corpus/requests corpus/attrs corpus/httpx

gate-baseline:
	python3 scripts/build_baseline_t6.py

gate-check:
	python3 scripts/check_gate_precision.py

gate-bench:
	python3 scripts/bench_gate_latency.py

# B7 -- see ASSUMPTIONS.md section 13. Same repo list as judge-check-alignment.
JEV_REPOS = corpus/click corpus/flask corpus/jinja corpus/werkzeug \
	    corpus/itsdangerous corpus/requests corpus/attrs corpus/httpx

jev-dry-run:
	python3 scripts/run_jev.py --dry-run --repos $(JEV_REPOS)

jev-smoke:
	python3 scripts/run_jev.py --limit-instances 5 --out-dir results/jev_smoke --repos $(JEV_REPOS)

jev:
	python3 scripts/run_jev.py --repos $(JEV_REPOS)

calibration:
	python3 scripts/analyze_calibration.py

clean:
	rm -rf __pycache__ omitbench/__pycache__ .pytest_cache results/smoke.jsonl
