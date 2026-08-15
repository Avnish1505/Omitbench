.PHONY: help corpus pilot analyze test reproduce clean judge-dry-run judge-check-alignment

help:
	@echo "corpus     clone the 9 evaluation repos (~93MB, needs network)"
	@echo "pilot      run all shards -> results/shards/*.jsonl  (~25 min)"
	@echo "analyze    regenerate every number in the README"
	@echo "test       run the test suite (leakage guards must pass)"
	@echo "reproduce  test + analyze from committed shards (no network, <1 min)"
	@echo "judge-dry-run        estimate B4/B5/B6 call count and spend, no API call"
	@echo "judge-check-alignment  verify judge corpus iids are a subset of the"
	@echo "                       shards' -- run after pilot, before a real sweep"

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

clean:
	rm -rf __pycache__ omitbench/__pycache__ .pytest_cache results/smoke.jsonl
