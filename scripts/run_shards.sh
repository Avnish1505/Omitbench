#!/usr/bin/env bash
# One process per repo: bounds peak memory and makes the run restartable.
set -u
mkdir -p results/shards
for r in click flask requests attrs jinja werkzeug httpx itsdangerous black; do
  out="results/shards/${r}.jsonl"
  [ -s "$out" ] && { echo "skip $r (done)"; continue; }
  timeout 600 python3 -m omitbench.experiment --repos "corpus/$r" \
      --per-repo 40 --scan-cap 600 --out "$out" >>results/shards/${r}.log 2>&1
  echo "$r -> $(wc -l < "$out" 2>/dev/null || echo 0) records"
done
