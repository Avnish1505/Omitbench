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

Only after T1–T3 (done — see T1/T3 status; T2 remains open, see below, and
is NOT a blocker for T6: T6's acceptance criterion targets P1's overall
precision, not UNWIRED-specific recall). A GitHub Action that comments on a
PR:

> Plan item 3 of 7 (`retry_with_backoff` in `client.py`) is not present in this diff.

**Decisions locked 2026-08-22** (see
`docs/superpowers/plans/2026-08-22-t6-ci-gate.md` for the full plan):

1. Detector: **P1 only.** B5 fails its own 0.80 precision bar (0.74), and a
   paid API call per PR is an operational liability for a CI gate.
2. Requirements come from a **structured checklist in the PR/issue body**
   (`- [ ] symbol in path`), never from free-text extraction — T5 measured
   that pipeline at MCC −0.935.
3. Frozen baseline: `results/baseline_t6.json`, built fresh from the
   current post-T3 corpus. `results/pilot.json` is NOT reused (stale,
   pre-P2/P3-removal) and stays untouched.
4. PR comments carry a disclaimer that this gate is structurally blind to
   `UNWIRED`/`STUB` omissions (recall 0.00 on both, by design) — it only
   catches missing/never-added definitions.

**Acceptance.** p95 latency under 10s per instance on a laptop, precision ≥
0.80 on `results/baseline_t6.json`.

**Falsification.** If P1's precision on a re-run of the frozen corpus
drops below 0.80, the gate must be marked **suspended** (`make gate-check`,
`results/gate_status.json`, `README.md`'s `GATE_STATUS` block) — not
silently left running. Note the acceptance bar is thin: this session
measured P1 precision at 0.8018 with a cluster-bootstrap 95% CI of
[0.723, 0.885] — the interval straddles 0.80 on both sides, so a future
run landing just under it may be noise, not a real regression; read
`ASSUMPTIONS.md`'s T6 section before treating any single suspension as
proof the detector regressed.

**Do not build a web frontend.** It adds no technical signal and costs two
weeks.

**Status: implemented and confirmed live.** p50/p95 latency
(`make gate-bench`, 40 samples from `corpus/attrs`): **12.0ms / 26.5ms** —
well under the 10s bar. Manual GitHub Actions smoke test (2026-08-22):
**passed** — a throwaway PR against this branch, with checklist item
`totally_fake_smoke_test_symbol in omitbench/gate.py` (a real path, a
symbol that doesn't exist there), triggered `omission-gate.yml`
([run 32553438223](https://github.com/Avnish1505/Omitbench/actions/runs/32553438223)),
which posted exactly one comment matching `render_comment`'s expected
output byte-for-byte — plan-item line, structural-blindness disclaimer,
and a precision caveat sourced live from the committed
`results/baseline_t6.json` (0.88, not a hardcoded value). Throwaway PR
closed and branch deleted after confirming.

---

## T7 — B7: TypeSafe Jev as a probabilistic judge, and its calibration

**Status:** harness built, tests pass, pre-registered (ASSUMPTIONS.md §13).
Real sweep not yet run.

**Why this is not the "fifth detector" the list below rules out.** That line
is about adding *proposed* detectors until one beats the baselines. B7 is a
*baseline* (a judge, like B4/B5/B6). It never replaces P1's numbers, and
its main output is a new kind of measurement: whether a model sold as
producing calibrated probabilities is calibrated on a labelled task outside
its training domain. Both variants and every decision rule were fixed
before the first call.

**Acceptance:**
- `make jev-dry-run` prints call count and cost, and the figure is recorded
  in ASSUMPTIONS.md §13 before the sweep.
- `make jev-smoke` (5 instances) finishes with 0 parse failures and 0
  model-pin errors, or the harness is fixed and the fix is recorded.
- `make jev` writes `results/shards/llm_judge_b7_jev_{single,split}.jsonl`
  covering every shard iid.
- `make analyze` prints B7 − B5 paired intervals. `make calibration` prints
  Brier/ECE/AUROC, the reliability table and the rule-2 verdict.
- README gets a B7 section written from those outputs only: no hand-typed
  numbers, and no re-run to change a result.

---

## Explicitly NOT on this list

- A fourth mutation class
- A fifth detector
- A dashboard, web UI, or visualisation layer
- Multi-language (tree-sitter) support
- Any refactor that does not change a number

If a task is not on this list and does not move T1–T6 forward, it is scope
creep. The bottleneck is evidence, not features.
