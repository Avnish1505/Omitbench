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

**Read this plainly, before anything else on the page: a mid-tier LLM judge
(B5) beats the deterministic detector (P1) overall, with a paired interval
excluding zero.** The motivating thesis of this repo — that judges are
structurally weak at omission — is **not supported** for B5. P1 wins narrowly
against a strong reasoning judge (B4) and comfortably against the no-AST
baseline (B3), but P1 loses to B5, and that is the honest headline, not a
footnote.

| detector | P | R | F1 | **MCC** | MCC 95% CI | FPR |
|---|---|---|---|---|---|---|
| B0 flag-nothing | 0.00 | 0.00 | 0.00 | **0.000** | [0.000, 0.000] | 0.000 |
| B1 flag-everything | 0.28 | 1.00 | 0.44 | **0.000** | [0.000, 0.000] | 1.000 |
| B3 line-grep (no AST) | 0.75 | 0.30 | 0.43 | **0.371** | [0.317, 0.421] | 0.039 |
| B4 LLM judge (gpt-oss-120b) | 0.40 | 0.97 | 0.56 | **0.388** | [0.345, 0.434] | 0.569 |
| **B5 LLM judge (mid-tier)** | **0.74** | 0.84 | **0.79** | **0.701** | **[0.650, 0.756]** | 0.113 |
| P1 defined | 0.80 | 0.44 | 0.57 | **0.499** | [0.449, 0.545] | 0.043 |

Paired cluster bootstrap of the MCC **difference**, P1 against each other
detector (detectors see identical inputs, so comparing independent intervals
by eye would discard the pairing):

| comparison | ΔMCC | 95% CI | supported? |
|---|---|---|---|
| P1 − B3 | +0.128 | [+0.103, +0.156] | yes, P1 wins |
| P1 − B4 | +0.111 | [+0.055, +0.164] | yes, P1 wins |
| **P1 − B5** | **−0.202** | **[−0.252, −0.150]** | **yes, and *against* P1** |

**Why B5 wins despite P1 having higher precision:** P1 is a pure "is it
defined" check. It is near-perfect on `ABSENT` (a deleted definition) but
**structurally blind** to `UNWIRED` (defined, never called) and `STUB`
(defined, called, hollow body) — recall 0.00 on both, by construction, not by
bad luck. B5 recovers real recall on both of those classes (0.21 `UNWIRED`,
0.87 `STUB`) at some precision cost. Averaged over the corpus's actual mix of
mutation classes, that trade wins. See the recall table below.

**A related result is more encouraging for the deterministic side but does not
change this conclusion:** the diff sent to every judge is token-truncated for
12/310 instances (32 variants) — P1 sees the full `after` snapshot regardless,
so this is a fairness asymmetry against the judges, not the detector. Excluding
those 12 instances:

| detector | MCC (full corpus, n=310) | MCC (truncation excluded, n=298) |
|---|---|---|
| P1 defined | 0.499 [0.449, 0.545] | 0.491 [0.439, 0.539] |
| B4 LLM judge (gpt-oss-120b) | 0.388 [0.345, 0.434] | 0.392 [0.348, 0.434] |
| B5 LLM judge (mid-tier) | 0.701 [0.650, 0.756] | **0.752 [0.704, 0.799]** |

P1 and B4 barely move. **B5 moves meaningfully** — the fairer, truncation-
excluded comparison makes B5's win over P1 *larger*, not smaller: paired
P1 − B5 on this subset is **−0.261 [−0.293, −0.226]**, versus −0.202 on the
full corpus. Removing the one asymmetry that favoured the deterministic side
strengthens the judge's case, not P1's.

### Why MCC and not F1

F1 ignores true negatives and moves with the base rate. At an earlier pilot's
75% positive rate, `flag-everything` scored **F1 0.82 and beat every real
detector**. MCC uses all four cells and returns 0.000 for any non-discriminating
rule, which is what B0 and B1 correctly receive above.

### Where the method works, and where it dies

Recall by omission class:

| detector | ABSENT (n=281) | UNWIRED (n=19) | STUB (n=309) |
|---|---|---|---|
| B3 line-grep | 0.65 | 0.00 | 0.00 |
| B4 LLM judge (gpt-oss-120b) | 0.98 | 0.68 | 0.99 |
| B5 LLM judge (mid-tier) | 0.84 | 0.21 | 0.87 |
| **P1 defined** | **0.96** | 0.00 | 0.00 |

P1 is near-perfect on `ABSENT` and structurally blind to the other two — "is it
defined" cannot see a hollow implementation or a dead call site. `STUB` is in
the corpus *specifically* because deterministic analysis should struggle with
it; a benchmark you always win on measures nothing. This is exactly the gap
B5 fills, and exactly why it wins overall despite losing on `ABSENT`.

### P2/P3 were removed

Two more elaborate detectors, `P2 defined+reachable` and
`P3 defined+reachable+body`, used to appear here. They scored FPR 0.83–0.85 —
reachability analysis condemns any public API symbol that nothing calls
internally, so they flagged most correctly-implemented public symbols as
omitted, and lost to `B3 line-grep` with a paired interval excluding zero
(ΔMCC −0.281 [−0.343, −0.225] for P2, −0.178 [−0.237, −0.114] for P3).

**T3 tried one fix, decided from Python packaging semantics and measured
exactly once:** exempt a symbol from the reachability check if it is listed in
`__all__`, re-exported in its package's `__init__.py`, or carries a decorator.
It narrowed the gap (FPR down to ~0.59–0.61; P3 moved from *losing significantly*
to *statistically tied* with B3) but did not clear the pre-agreed bar of P2/P3
beating B3 outright. Per that pre-agreed protocol, the rule was not iterated —
**P2 and P3 are removed** from the scored detector set. Full before/after
numbers and the reasoning are in `ASSUMPTIONS.md` §9; the code is kept in
`omitbench/detectors.py`, clearly marked as retired, not deleted.

**The most transferable finding from that exercise:** P2's apparent 0.84 recall
on `STUB` before the fix was **mostly an artifact of the FPR bug, not real stub
detection.** `d_reachable` has no mechanism to see a hollow body — only P3's
extra check does that. P2 was simply over-flagging ~83% of everything as
omitted, which incidentally caught genuine `STUB` cases too, for the wrong
reason. Once the exemption removed the over-flagging, P2's `STUB` recall
collapsed to 0.58 — much closer to what a detector blind to body content
should actually score. **A metric that looks good for a reason unrelated to
the thing it claims to measure is the single most transferable lesson in this
repo**, alongside the corpus-filter lesson in `ASSUMPTIONS.md` §6.

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
make test        # 91 tests (86 unit + 5 leakage guards), <0.1s
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
- **Requirements are derived from the gold patch**, which assumes a perfect
  extractor. All numbers are therefore an **upper bound**.
- **Python only.** Functions and classes only — no config keys, dependencies, or
  behaviour-only requirements.
- **`UNWIRED` is underpowered** (n=19; the mutation used to miss calls inside
  `return` statements, fixed in `mutate.py`, which raised n from 16 to 19 — still
  far short of the ~100 needed for that column to be more than noise. Read the
  `UNWIRED` recall numbers above as directional, not conclusive.
- **`black` is excluded**: its shards fail to build, and a code formatter's
  commits are atypical. Stated rather than silently dropped.

## What would falsify the main claim

If the paired P1 − B3 interval included zero, there would be no ordering claim
and this would be a null result. It does not: [+0.103, +0.156]. That test was
never the whole story, though — the real falsification event already
happened: the motivating thesis of this repo is that LLM judges are
structurally weak at omission, and the paired P1 − B5 interval is
**[−0.252, −0.150]**, excluding zero *against* P1. Read plainly, that thesis
is not supported for a mid-tier judge on this corpus. See `RELATED.md` for
what the literature this repo is testing against actually claims, and the
Headline result section above for why P1 still wins on `ABSENT` while losing
overall. If real agent omissions turn out to be dominated by `STUB`-like
semantic hollowing rather than `ABSENT`, P1's narrow win over B3 and B4 would
also be expected to erode further.

## Prior work this does not claim to precede

The observation that coding agents claim completion they have not achieved is
**not novel**. See `RELATED.md` for Liu et al. (UIUC + IBM) on plan compliance
across 16,991 trajectories, Snowflake on silent semantic failure, and the MAST →
AgentDebugX line on failure taxonomy and attribution. This repo builds a
*detector with injected ground truth and a reported failure boundary*, which is
a different object of study, and a much narrower one.

MIT licensed.
