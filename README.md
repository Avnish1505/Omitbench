# OmitBench

**Detecting silent omissions in coding-agent patches, with labels that cost nothing.**

When an autonomous coding agent reports "done", some fraction of what it said it
would build is simply not in the diff. Snowflake AI Research measured the shape
of this: across 1,750 trajectories on 50 SWE-bench Verified tasks, GPT-5 submits
a patch on 100% of runs but resolves 44%, and *silent* failures — confidently and
consistently wrong — account for the majority of the gap. Completion-based and
consistency-based monitoring both look healthy exactly when the agent should not
be trusted.

OmitBench attacks one tractable slice of that problem: deciding whether a named
requirement is **absent** from a produced diff. It ships

1. a **labelled omission corpus** built by mutating real merged commits, so
   ground truth costs zero rupees and zero GPU-hours,
2. a **deterministic detector** using only program-analysis evidence, with no
   access to any reference solution,
3. a fair **no-AST baseline**, and
4. a reproduction pipeline: `make reproduce` regenerates every number below.

It does **not** detect incorrect behaviour, and it does not yet generalise to
real agents. See *Scope and limits* — that section is the point, not a
disclaimer.

---

## Headline result

`make analyze`. Positive class is `OMITTED`. Intervals are **cluster bootstrap
over instances** (n=310), not over records — the variants and requirements of a
single commit are not independent.

| detector | P | R | F1 | **MCC** | MCC 95% CI | FPR |
|---|---|---|---|---|---|---|
| B0 flag-nothing | 0.00 | 0.00 | 0.00 | **0.000** | [0.000, 0.000] | 0.000 |
| B1 flag-everything | 0.28 | 1.00 | 0.44 | **0.000** | [0.000, 0.000] | 1.000 |
| B3 line-grep (no AST) | 0.76 | 0.30 | 0.43 | **0.374** | [0.320, 0.433] | 0.038 |
| **P1 defined** | **0.81** | 0.45 | **0.58** | **0.508** | **[0.459, 0.557]** | 0.042 |
| P2 defined+reachable | 0.30 | 0.90 | 0.45 | 0.087 | [0.044, 0.129] | 0.835 |
| P3 defined+reachable+body | 0.31 | 0.99 | 0.48 | 0.201 | [0.165, 0.237] | 0.843 |

Paired cluster bootstrap of the MCC **difference** against the strongest
baseline (detectors see identical inputs, so comparing independent intervals by
eye would discard the pairing):

| comparison | ΔMCC | 95% CI | supported? |
|---|---|---|---|
| P1 − B3 | **+0.134** | [+0.107, +0.162] | **yes** |
| P2 − B3 | −0.287 | [−0.355, −0.226] | yes, and *against* P2 |
| P3 − B3 | −0.173 | [−0.240, −0.109] | yes, and *against* P3 |

**Read that honestly: only the simplest detector beats the baseline. The two
more elaborate ones are significantly worse.** Their false-positive rate is
0.84 — they flag most correctly-implemented requirements as omitted, because
reachability analysis condemns any public API symbol that nothing calls
internally. A CI gate with FPR 0.84 gets uninstalled in a week.

An earlier version of this corpus required every candidate symbol to be called
internally by the reference solution. That filter had been **hiding this
defect** by never generating the cases P2 and P3 fail on. Relaxing it tripled
yield and exposed the problem. This is the most transferable lesson in the repo:
*a corpus filter that removes the cases your method fails on will make your
method look good.* (`ASSUMPTIONS.md` §6.)

### Why MCC and not F1

F1 ignores true negatives and moves with the base rate. At an earlier pilot's
75% positive rate, `flag-everything` scored **F1 0.82 and beat every real
detector**. MCC uses all four cells and returns 0.000 for any non-discriminating
rule, which is what B0 and B1 correctly receive above.

### Where the method works, and where it dies

Recall by omission class:

| detector | ABSENT | UNWIRED | STUB |
|---|---|---|---|
| B3 line-grep | 0.65 | 0.00 | 0.00 |
| P1 defined | **0.98** | 0.00 | 0.00 |
| P2 defined+reachable | 0.99 | 0.62 | 0.83 |
| P3 defined+reachable+body | 0.99 | 0.62 | **1.00** |

P1 is near-perfect on `ABSENT` and structurally blind to the other two — "is it
defined" cannot see a hollow implementation. P2 and P3 recover recall on
`UNWIRED` and `STUB` but pay for it with the FPR above, so their recall is not
usable as-is. `STUB` is in the corpus *specifically* because deterministic
analysis should struggle with it. A benchmark you always win on measures nothing.

---

## Method

**Corpus.** Eight mid-sized pure-Python libraries (`click`, `flask`, `jinja`,
`werkzeug`, `itsdangerous`, `requests`, `attrs`, `httpx`). For each sampled
merged commit: apply the whole patch to get a correct `R_after`, then mutate one
newly-defined symbol to simulate an agent that omitted work.

**Mutations.** v1 injected omissions by deleting hunks. That was *vacuous* —
the omitted code was literally absent from the diff and `grep` scored F1 1.00.
Real agent omissions are not absent, they are **hollow**:

| class | what it does |
|---|---|
| `ABSENT` | delete the definition; call sites survive |
| `UNWIRED` | keep the definition, neutralise every call site — dead code that reads as finished |
| `STUB` | keep definition and call sites, hollow the body |

**Requirements** are path-qualified: `src/click/core.py::ParamType`. Bare names
collide constantly in Python (`main`, `run`, `parse`), and a same-named symbol
elsewhere used to mask real deletions — that capped recall on `ABSENT` at 0.43.

**Base rate.** Every new symbol in a patch is scored as a separate requirement.
In a mutated variant exactly one is omitted and the rest are not, so negatives
come for free and precision is measurable. Base rate is 28%, not 75%.

### The load-bearing invariant

```
detect(spec, requirements, before, after, ctx) -> {req: "IMPLEMENTED"|"OMITTED"}
```

**No detector ever receives the gold patch.** If it could, "find the deleted
hunks" is trivial and every number here is meaningless. `tests/test_leakage.py`
enforces this three ways: a signature contract, a **runtime canary** planted in
the gold patch that must never reach a detector, and a determinism check. These
run in CI and are not optional.

---

## Reproduce

```bash
make test        # 14 unit tests + 3 leakage guards, ~2s
make reproduce   # regenerates every number above from committed shards
```

`make reproduce` needs **no network, no corpus, no GPU and no API key** — the
scored shards are committed. To rebuild the corpus from scratch:

```bash
make corpus      # clones 9 repos, ~93MB
make pilot       # ~25 min, one process per repo
make analyze
```

Core has **zero dependencies** — stdlib `ast`, `difflib`, `re`. `pytest` for
tests only.

```
omitbench/
├── corpus.py       git plumbing, patch parsing, hunk application
├── mutate.py       ABSENT / UNWIRED / STUB injection + labels
├── evidence.py     AST symbol tables, changed-line extraction
├── detectors.py    baselines + proposed detectors (the contract lives here)
└── experiment.py   corpus build, scoring, cluster bootstrap
scripts/analyze.py  every number in this README
tests/              leakage guards + micro-repo semantics
```

---

## Scope and limits

Short version; full accounting in [`ASSUMPTIONS.md`](ASSUMPTIONS.md), positioning
against prior work in [`RELATED.md`](RELATED.md).

- **Omissions are injected, not observed.** No real agent trajectories yet.
  Until 40–60 hand-labelled real traces exist, nothing here generalises to
  deployed agents. This is the single largest threat to validity.
- **No LLM-judge baseline yet.** The motivating argument is that LLM judges are
  structurally weak at omission — reported elsewhere at roughly 6–7× worse on
  planted omissions than planted over-inclusions. That comparison is
  **unmeasured here.** Largest open item.
- **Requirements are derived from the gold patch**, which assumes a perfect
  extractor. All numbers are therefore an **upper bound**.
- **Python only.** Functions and classes only — no config keys, dependencies, or
  behaviour-only requirements.
- **`UNWIRED` is underpowered** (n=16 in the run above; the mutation missed calls
  inside `return` statements, now fixed — that column needs a re-run before it
  is cited).
- **`black` is excluded**: its shards fail to build, and a code formatter's
  commits are atypical. Stated rather than silently dropped.

## What would falsify the main claim

If the paired P1 − B3 interval included zero, there would be no ordering claim
and this would be a null result. It does not: [+0.107, +0.162]. If real agent
omissions turn out to be dominated by `STUB`-like semantic hollowing rather than
`ABSENT`, P1's advantage largely evaporates and the honest headline becomes
"deterministic analysis handles the easy third of this problem."

## Prior work this does not claim to precede

The observation that coding agents claim completion they have not achieved is
**not novel**. See `RELATED.md` for Liu et al. (UIUC + IBM) on plan compliance
across 16,991 trajectories, Snowflake on silent semantic failure, and the MAST →
AgentDebugX line on failure taxonomy and attribution. This repo builds a
*detector with injected ground truth and a reported failure boundary*, which is
a different object of study, and a much narrower one.

MIT licensed.
