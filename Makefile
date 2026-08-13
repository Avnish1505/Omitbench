.PHONY: help corpus pilot analyze test reproduce clean

help:
	@echo "corpus     clone the 9 evaluation repos (~93MB, needs network)"
	@echo "pilot      run all shards -> results/shards/*.jsonl  (~25 min)"
	@echo "analyze    regenerate every number in the README"
	@echo "test       run the test suite (leakage guards must pass)"
	@echo "reproduce  test + analyze from committed shards (no network, <1 min)"

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

clean:
	rm -rf __pycache__ omitbench/__pycache__ .pytest_cache results/smoke.jsonl
