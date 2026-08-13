FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY pyproject.toml Makefile ./
COPY omitbench/ ./omitbench/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY results/shards/ ./results/shards/

RUN pip install --no-cache-dir pytest

# Default target needs no network: it reproduces the paper numbers from the
# committed shards. `make pilot` (regenerating shards) needs the corpus.
CMD ["make", "reproduce"]
