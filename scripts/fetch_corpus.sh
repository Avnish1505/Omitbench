#!/usr/bin/env bash
# Nine mid-sized pure-Python libraries with long histories and real review
# culture. Chosen because they are Python-only (the AST engine is Python-only),
# actively maintained, and none is a code generator or formatter whose commits
# are atypical. `black` is fetched but currently EXCLUDED from results --
# see ASSUMPTIONS.md.
set -euo pipefail
mkdir -p corpus && cd corpus
repos=(
  "https://github.com/pallets/click"
  "https://github.com/pallets/flask"
  "https://github.com/pallets/jinja"
  "https://github.com/pallets/werkzeug"
  "https://github.com/pallets/itsdangerous"
  "https://github.com/psf/requests"
  "https://github.com/python-attrs/attrs"
  "https://github.com/encode/httpx"
  "https://github.com/psf/black"
)
for url in "${repos[@]}"; do
  name=$(basename "$url")
  [ -d "$name" ] && { echo "have $name"; continue; }
  git clone --quiet --filter=blob:none "$url" "$name"
  echo "cloned $name"
done
