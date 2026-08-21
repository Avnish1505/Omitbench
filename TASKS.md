# TASKS

Ordered backlog. Each task has an acceptance criterion that is checkable, and a
falsification condition — what result would mean the task failed rather than
succeeded. Work top to bottom. Do not start T3 before T1 lands.

Every task ends the same way: `make test`, `make analyze`, and a stated diff
against `results/pilot.json`.

---

## T1 — LLM-judge baseline  ← **start here**

**Why first.** The whole argument for this project is that LLM judges are
structurally weak at detecting omission — roughly 6–7× worse on planted
omissions than planted over-inclusions, per prior work. Right now the repo beats
`grep` and *cites* someone else's paper for the judge claim. Until a judge runs
on these same 310 instances, the central claim is borrowed, not demonstrated.

**Build.** `omitbench/judges.py`, same contract as every other detector:

```python
def d_llm_judge(spec, reqs, before, after, ctx): ...
```

- Input to the model: the requirement list and the unified diff of
  `before -> after`. **Never the gold patch.** Add a canary case to
  `tests/test_leakage.py` covering the judge.
- **Content-addressed cache** keyed by `sha256(prompt + model + params)`, stored
  in `cache/`. Commit the cache. Re-running the paper must then cost ₹0 and
  work offline, which `make reproduce` already promises.
- Make it a *fair* baseline. Same requirement list, a genuinely good prompt, one
  call per instance. A strawman judge destroys the result's credibility faster
  than a negative finding does.
- Budget cap: hard-fail if projected spend exceeds a `--max-cost` flag.

**Acceptance.** `d_llm_judge` appears in the headline table with MCC and a
cluster-bootstrap CI, plus a paired `P1 − judge` interval.

**Falsification.** If judge recall on `ABSENT` exceeds ~0.85 and its MCC beats
P1 with a paired interval excluding zero, the deterministic wedge is gone.
Report that. Do not re-prompt until the judge loses.

---

## T2 — Re-run shards after the UNWIRED fix

`mut_unwired` previously matched only standalone calls and simple assignments,
so `return helper(3)` was skipped. Most real call sites are inside returns —
that is why `UNWIRED` had n=16. Fixed in `mutate.py`, but the committed shards
predate the fix.

**Do.** `make pilot && make analyze`. Update the README tables from the new
output. State the n for `UNWIRED` before and after.

**Acceptance.** `UNWIRED` n ≥ 100. If it is still under ~50, the mutation is
still missing common call forms — instrument which lines matched and which did
not before changing anything else.

---

## T3 — Fix or remove P2/P3

They score FPR 0.84 and are **significantly worse than grep** (−0.287, −0.173
MCC, both intervals excluding zero). Reachability condemns any public API symbol
nothing calls internally.

**Two honest options, pick one and justify it in `ASSUMPTIONS.md`:**

- **Fix.** Exempt symbols exported via `__all__`, re-exported in `__init__.py`,
  or decorated as public entry points, then re-measure.
- **Remove.** Delete them, and keep a section in the README explaining that
  reachability is the wrong signal for library code and why.

**Acceptance.** Either P2/P3 beat `B3 line-grep` with a paired interval
excluding zero, or they are gone and the reason is written down.

**Do not** tune the exemption rule until the numbers improve. Decide the rule
from Python packaging semantics first, then measure once.

---

## T4 — Real agent trajectories (the largest validity threat)

Everything so far is injected omissions. Nothing generalises to deployed agents
until this exists.

**Do.** Collect 40–60 trajectories from a real agent (Claude Code, mini-SWE-agent,
or OpenHands with a local model) on repos **not** in `corpus/`. Hand-label which
stated requirements were omitted. Commit to `data/real/`.

**Acceptance.** A `real vs synthetic` MCC comparison in the README, with the gap
stated numerically.

**Falsification.** A gap greater than ~0.25 MCC means injected omissions are
unrepresentative and the synthetic corpus is a weak proxy. **Publish that anyway.**
A paper saying "our proxy is worse than we hoped, here is by how much" is more
credible than one that never checked.

---

## T5 — Requirement-extraction error term

Requirements are currently derived from the gold patch, which assumes a perfect
extractor, so all numbers are an upper bound.

**Do.** Build an LLM extractor that produces requirements from the *commit
message / issue text only*. Measure its precision/recall against the
patch-derived requirements. Then re-run detection using extracted requirements
and report the drop.

**Acceptance.** README reports detection F1 under both oracle and extracted
requirements, so the extraction tax is visible as a separate error term.

**Falsification.** If symbol-only extraction F1 exceeds ~0.9 against the
oracle, requirement extraction is not the bottleneck, and the resulting
detection-F1 tax is expected to be small — report that as the finding, not
as grounds to keep refining the prompt. Do not re-prompt after seeing this
number to manufacture a bigger gap.

**Status: measured, synthetic corpus only (n=310).** Falsification bar not
cleared — symbol-only extraction F1 is 0.066 [0.044, 0.091], nowhere near
0.9. Extraction is the dominant bottleneck: P1's MCC falls from 0.499
(oracle requirements) to −0.935 (extracted requirements, literal pipeline) —
worse than flagging nothing. See README's *Requirement-extraction error
term (T5)* section for the full table and `ASSUMPTIONS.md` §11 for the full
methodology, diagnostics, and sample output. Not measured on the real T4
corpus (`data/real/`) — n=8 is too small; future work.

---

## T6 — Package as a CI gate

Only after T1–T3. A GitHub Action that comments on a PR:

> Plan item 3 of 7 (`retry_with_backoff` in `client.py`) is not present in this diff.

**Acceptance.** p95 latency under 10s per instance on a laptop, precision ≥ 0.80
on the frozen corpus. A gate below that precision gets uninstalled in a week and
should not ship.

**Do not build a web frontend.** It adds no technical signal and costs two weeks.

---

## Explicitly NOT on this list

- A fourth mutation class
- A fifth detector
- A dashboard, web UI, or visualisation layer
- Multi-language (tree-sitter) support
- Any refactor that does not change a number

If a task is not on this list and does not move T1–T6 forward, it is scope
creep. The bottleneck is evidence, not features.
