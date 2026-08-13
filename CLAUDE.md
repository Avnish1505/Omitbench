# CLAUDE.md

Context for Claude Code working in this repository. Read this before editing.

---

## What this project is

OmitBench measures whether a **deterministic** program-analysis detector can
find *omitted requirements* in coding-agent patches better than a no-AST
baseline, using ground-truth labels produced by mutating real merged commits.

It is a **measurement project**, not a product. The output is numbers with
confidence intervals, plus an honest boundary where the method stops working.
Code quality matters only insofar as it makes the numbers trustworthy.

Read `README.md` for results, `ASSUMPTIONS.md` for limits, `RELATED.md` for
positioning against prior work.

---

## THE ONE INVARIANT — do not break this

```
detect(spec, requirements, before, after, ctx) -> {req: "IMPLEMENTED"|"OMITTED"}
```

**No detector may ever receive the gold patch, the mutation class, or the
ground-truth label.** If a detector can see the reference solution, "find the
deleted hunks" is trivial and every number in `results/` becomes meaningless.

This is enforced by `tests/test_leakage.py` three ways: a signature contract, a
runtime canary planted in the gold patch, and a determinism check.

If you are tempted to pass extra context into a detector "just for debugging" —
don't. Add it to `ctx` only if it is derivable from `(before, after)` alone, and
add a test proving it.

---

## Do NOT touch without explicit instruction

| path | why |
|---|---|
| `results/pilot.json` | **Frozen Week-2 baseline.** Targets are deltas from these numbers. Editing it is moving goalposts. |
| `tests/test_leakage.py` | The credibility of the whole repo. Never weaken an assertion to make a change pass. |
| `results/shards/*.jsonl` | Raw scored records. Regenerate with `make pilot`, never hand-edit. |
| `legacy/` | Provenance of failed approaches. Kept deliberately. |
| Numbers in `README.md` | Regenerate via `make analyze`. Never hand-type a metric. |

---

## Verification protocol for every change

Run these, in order, and **report the actual output** — do not summarise as
"tests pass":

```bash
make test                 # 14 unit + 3 leakage guards. All must pass.
make analyze              # regenerates the headline table
```

Then compare against `results/pilot.json`. If a metric moved, say which one and
by how much, **before** proposing the change is good.

If you changed anything in `mutate.py`, the labels changed, so the shards are
stale:

```bash
make pilot                # ~25 min, one process per repo
make analyze
```

Grep-verification before claiming a change is wired:

```bash
grep -rn "your_new_function" omitbench/ tests/
```

---

## Project-specific anti-patterns

These are mistakes this codebase has already made once. Do not repeat them.

**1. Do not filter the corpus in a way that removes cases the detector fails on.**
An earlier version required every candidate symbol to be called internally by the
reference solution. That silently excluded public API — exactly where the
reachability detectors produce false positives — and made P2/P3 look strong.
Relaxing the filter exposed FPR 0.84. Any new filter needs a written
justification in `ASSUMPTIONS.md`.

**2. Do not use F1 as a headline metric.** F1 ignores true negatives and moves
with the base rate. At an earlier 75% positive rate, `flag-everything` scored
F1 0.82 and beat every real detector. **MCC is the headline.** Report F1 only
alongside it.

**3. Do not bootstrap over records.** Variants and requirements from one commit
share a repo, a patch and a symbol. Cluster bootstrap over `iid`. A record-level
bootstrap yields intervals several times too narrow.

**4. Do not compare detectors with independent intervals.** All detectors see
identical inputs. Use the **paired** bootstrap of the MCC difference already in
`scripts/analyze.py`. If that interval contains zero, there is no ordering claim
— say so rather than reaching for a different metric.

**5. Do not add a mutation class or a detector to "improve results."** The
bottleneck is not coverage. Adding variants until something wins is p-hacking
with extra steps.

**6. Do not delete a failing case.** If a mutation or instance breaks the
pipeline, fix the pipeline or record the exclusion in `ASSUMPTIONS.md` with the
reason. `black` is excluded this way and it is stated in three places.

---

## Architecture

```
omitbench/
├── corpus.py       git plumbing, unified-diff parsing, hunk application
├── mutate.py       ABSENT / UNWIRED / STUB injection. Owns ground truth.
├── evidence.py     AST symbol tables, changed-line extraction
├── detectors.py    baselines (B*) + proposed detectors (P*). Contract lives here.
└── experiment.py   corpus build, scoring loop, cluster bootstrap
scripts/analyze.py  produces every number in README.md
```

Data flow: `commit -> build_base() -> full_after -> mutate one symbol ->
variants -> detectors score each requirement -> results/shards/*.jsonl ->
analyze.py`.

Core is **stdlib only** (`ast`, `difflib`, `re`). Keep it that way — zero deps
is part of the reproducibility claim. `pytest` is dev-only.

**Memory:** `mutate.py` memoises AST parses in module-level dicts. A multi-repo
run OOMs without clearing them between repos. `experiment.py` does this; if you
add a new entry point, do the same. This is why `run_shards.sh` uses one process
per repo.

---

## Known open items, in priority order

See `TASKS.md` for full acceptance criteria.

1. **LLM-judge baseline is missing.** The entire motivating argument is that
   judges are structurally weak at omission. Right now that is borrowed from
   other people's papers, not demonstrated here. Highest-value work.
2. **`UNWIRED` numbers in the README are stale.** `mut_unwired` was fixed to
   handle calls inside `return` statements (previously it only matched
   standalone calls and assignments, starving that class to n=16). Shards need
   a re-run before the `UNWIRED` column is cited.
3. **No real agent trajectories.** All omissions are injected. Nothing
   generalises to deployed agents until 40–60 hand-labelled real traces exist.
4. **P2/P3 are worse than grep** (FPR 0.84). Either fix reachability to exempt
   public API, or delete them and say why.
5. `black` shards fail to build.

---

## Writing style for this repo

Comments explain **why**, especially why an approach was rejected. Several
comments in `mutate.py` and `experiment.py` document defects that were found and
fixed; they are load-bearing documentation, not clutter. Preserve them.

Docstrings on detectors state what the detector is **blind to**, not just what it
catches. That blindness is the contribution.
