# Related work, and what is left

Week 1 deliverable. Written before the experiments so the positioning could not
be retrofitted to whatever the numbers turned out to be.

The short version: **the observation that coding agents claim completion they
have not achieved is not novel and this repo does not claim it is.** Three
strong groups published it in 2025–26. What is not yet done is a *cheap,
labelled, deterministic* detection benchmark, and that is the only gap this
repo tries to fill.

---

## What already exists

**Plan compliance — Liu, Dehghan, Ganhotra, Hirzel, Jabbarvand (UIUC + IBM),
`arXiv:2604.12147`, April 2026.**
The first systematic analysis of whether programming agents follow instructed
plans: 16,991 SWE-agent trajectories, four LLMs, SWE-bench Verified and Pro,
eight plan variations. Finds that agents fall back on internalised workflows
without an explicit plan, that periodic plan reminders reduce violations, and
that a subpar plan is worse than no plan.

*Relation to this repo:* they **measure compliance descriptively, offline, on a
fixed benchmark, using LLM-based trajectory analysis**. This repo builds a
**detector** with injected ground truth and reports precision/recall against a
baseline. Different object of study. Any claim here of the form "first to notice
agents deviate from plans" would be false, and is not made.

**Silent semantic failures — Snowflake AI Research, `arXiv:2603.25764`.**
1,750 trajectories on 50 SWE-bench Verified tasks. GPT-5 submits a patch on 100%
of runs but resolves 44%; Llama 4 submits on 99% and resolves 18%. Silent
semantic failure accounts for roughly 68–80% of failing runs, and the paper
notes that completion-based *and* consistency-based monitoring both look healthy
exactly when the agent should not be trusted.

*Relation:* this is the **motivation**, and it is a far better one than a
personal anecdote. It establishes that the failure mode is systemic and
measurable. It does not build a detector.

**Failure taxonomy and attribution — MAST (Cemri et al., NeurIPS 2025 D&B),
Who&When, TRAIL, AgentDebug, AgenTracer, GraphTracer, AEGIS, AgentRx, and
AgentDebugX (an open-source failure observability/attribution/recovery toolkit,
July 2026).**

*Relation:* this subfield is **crowded and well resourced**. An "agent failure
observability tool" is not a viable contribution for one person in 2026. This
repo deliberately does *not* attempt attribution, taxonomy, or recovery. It does
one narrow thing: decide whether a named requirement is present in a diff.

**Omission is the hard direction — `Judging Is Not Enumerating` (Aug 2026) and
related work on omission in agent pipelines.**
Reports that models detect *planted over-inclusions* roughly 6–7× more often
than *planted omissions*, and describes a production deployment failing
omission-first at about 10:1. Work on clinical summarisation makes the same
point: omission detection is far less developed than hallucination detection,
because omission is silent.

*Relation:* **this is the entire wedge.** LLM-as-judge is structurally weak at
precisely the failure class targeted here. The bet is not "beat a strong
baseline everywhere" — it is "beat a known-weak baseline on the specific class
where it is known to be weak, and publish where that stops working."

**Evaluation statistics — `On Randomness in Agentic Evals` (`arXiv:2602.07150`),
`Stochasticity in Agentic Evaluations` (`arXiv:2512.06710`), Miller,
*Adding Error Bars to Evals*.**
Single-run pass@1 varies by 2.2–6.0 percentage points depending on which run is
selected, with standard deviations above 1.5pp even at temperature 0; reported
2–3 point improvements may be noise. Independent audits report LLM-judge error
rates above 50%, with position, length and agreeableness bias.

*Relation:* this is why no single-run number appears anywhere in this repo, why
intervals are cluster-bootstrapped, and why detector comparisons are paired.

---

## The gap this repo targets

Everything above either (a) measures the problem descriptively, or (b) detects
it with an LLM judge. Nobody has published a **labelled omission-detection
benchmark whose labels cost nothing to produce**, evaluated against a fair
no-AST baseline, with the boundary of the deterministic approach stated.

Three properties make that gap addressable by one person with no GPU:

1. **Labels are free.** Mutating a known-good patch yields ground truth without
   annotation. 310 instances cost zero rupees and zero GPU-hours.
2. **The baseline is known-weak in a specific direction**, so the claim is
   falsifiable and narrow rather than "our system is better."
3. **The negative space is publishable.** `STUB` mutations are included
   *expecting* deterministic analysis to fail on them. Reporting where the
   method stops working is the contribution, not an admission.

---

## What this repo has NOT done

Stated plainly so the gap between claim and evidence stays visible.

- **No real agent trajectories.** All omissions are injected. Until 40–60
  hand-labelled real traces exist (design doc E7), nothing here generalises to
  deployed agents.
- **No LLM-judge baseline yet.** The strongest baseline currently implemented is
  `grep` over changed lines. The judge comparison — the entire motivating
  argument above — is **unmeasured**. This is the largest open item.
- **No requirement-extraction error term.** Requirements are derived from the
  gold patch, so results are an upper bound (see `ASSUMPTIONS.md` §2).
- **No multi-language support.** Python only.
- **`UNWIRED` is underpowered** at n=16.

## Current honest status

One detector (`P1 defined`) beats the no-AST baseline with a paired interval
excluding zero. The two more elaborate detectors are **significantly worse than
grep**, and an earlier corpus filter had been concealing that. Details and
numbers in `README.md`; the methodological lesson is in `ASSUMPTIONS.md` §6.
