# Assumptions and limits

Every shortcut this benchmark takes, written down. A reviewer will find these
anyway; the only question is whether they find them in this file or in the
results. Ordered roughly by how much damage each one does to the claims.

---

## 1. Omissions are injected, not observed

**What we do.** We take a real merged commit, apply the whole patch to get a
correct `R_after`, then mutate one new symbol to simulate an agent that omitted
work. Ground-truth labels come from knowing which symbol we mutated.

**Why this is a real threat.** Nothing guarantees that a mutation resembles what
a coding agent actually does. If injected omissions are structurally simpler
than real ones, every number here is optimistic.

**What we did about it.** v1 of this benchmark injected omissions by deleting
hunks from the gold patch. That was *vacuous*: the omitted code was literally
absent from the diff, so a `grep` baseline scored F1 1.00. Real agent omissions
are not absent, they are **hollow** — the agent writes something that looks
finished. v2 replaced hunk deletion with three mutation classes chosen to span
that spectrum:

| class | what it does | why it is here |
|---|---|---|
| `ABSENT` | delete the definition | control; the easy case, grep should do reasonably |
| `UNWIRED` | keep the definition, delete every call site | dead code that reads as complete |
| `STUB` | keep definition and call sites, hollow the body | the semantic case; static analysis *should* struggle |

`STUB` is included specifically because deterministic analysis is expected to do
badly on it. A benchmark you always win on measures nothing.

**Residual risk: still high.** This is a proxy. The only fix is E7 in the design
doc — hand-label 40–60 real agent trajectories and measure the gap. **Not done
yet.** Until it is, no claim in this repo generalises to real agents, and the
README says so.

---

## 2. Requirements are derived from the gold patch, not from a plan

**What we do.** A requirement is a symbol the reference commit newly defines,
written path-qualified: `src/click/core.py::ParamType`.

**Why this is a threat.** A real plan is prose written *before* the code, is
noisier, is sometimes wrong, and does not decompose neatly into symbols. Deriving
requirements from the solution assumes a perfect requirement extractor.

**Consequence.** Every number here is an **upper bound**. A real pipeline pays an
extraction-error tax on top. That tax is a separate measurement (design doc
§7.10, "requirement extractor quality" ablation) and is not measured yet.

**Not a leak.** The detector never sees the gold patch — only
`(spec, requirements, before, after)`. Requirements name a symbol and a file;
checking whether that symbol exists in `after` is a pure function of `after`.
Enforced by `tests/test_leakage.py`, which plants a canary in the gold patch and
fails if it reaches any detector.

---

## 3. Only symbols, only Python, only top-level and nested defs

- Requirements are functions and classes. Config keys, dependencies, docs
  changes, and behaviour-only requirements are out of scope entirely.
- The evidence engine is Python-only (stdlib `ast`). No multi-language claim.
- Symbols starting with `_` and shorter than 4 characters are skipped, to avoid
  matching on `id`, `fn`, `_x`. This biases toward public-ish API.

---

## 4. Corpus composition

Eight repositories: `click`, `flask`, `jinja`, `werkzeug`, `itsdangerous`,
`requests`, `attrs`, `httpx`. All are mid-sized, pure-Python, well-reviewed
libraries.

**`black` is fetched but excluded from results.** Its shards fail to build in
the current harness (large files, OOM). It is also a code formatter, whose
commits are atypical of application code. Both reasons are stated rather than
silently dropping it.

**Bias.** Library code, not application code. No web frameworks with heavy
metaprogramming, no ML codebases, no async-heavy services beyond `httpx`. Five
of eight are Pallets projects, so review culture and code style are correlated —
the per-repo MCC table in the README exists so this correlation is visible
rather than hidden inside an average.

---

## 5. Statistical assumptions

- **Records within an instance are dependent.** The variants and requirements of
  one commit share a repo, a patch and a symbol. All confidence intervals are
  **cluster bootstrap over instances** (n=310), never over records (n=2166).
  A record-level bootstrap would report intervals several times too narrow.
- **Detector comparisons are paired.** All detectors see identical inputs, so
  the README reports a paired bootstrap of the MCC *difference*. Comparing two
  independent CIs by eye discards the pairing and is not a test.
- **F1 is reported but is not the headline.** F1 ignores true negatives and moves
  with the base rate. At the earlier pilot's 75% positive rate, `flag-everything`
  scored F1 0.82 and beat every real detector. MCC uses all four cells and is
  0.000 for any non-discriminating rule. That is why MCC is the headline.
- **`UNWIRED` has n=19** (raised from n=16 after the `mutate.py` fix for calls
  inside `return` statements — see §6). That is far too few for any claim.
  Recall numbers in that column are reported for completeness and should be
  read as noise. **TASKS.md T2 (re-run targeting n≥100) remains open** due to
  this low mutation-eligible sample size; this is a documented known
  limitation, not a blocker for T6, since T6's CI-gate acceptance criterion
  targets P1's overall precision (currently 0.80), not `UNWIRED`-specific
  recall.

---

## 6. What changed between pilots, and why it matters

The earlier pilot required every candidate symbol to be **called internally by
the reference solution**. That filter excluded public API — symbols defined for
downstream users and never called inside the library.

Relaxing it (only `UNWIRED` genuinely needs a call site) roughly tripled yield.
It also **exposed a defect the filter had been hiding**: the reachability-based
detectors `P2` and `P3` flag any symbol nothing calls, so on public API they
produce false positives at FPR ≈ 0.84. Under the old filter those cases were
never generated, and `P2`/`P3` looked strong.

The lesson is recorded here because it is the most transferable thing in this
repo: **a corpus filter that removes the cases your method fails on will make
your method look good.**

---

## 7. B6 no longer tests an independent inference stack

B6 was originally planned as direct NVIDIA NIM (a separate inference stack, not
brokered by OpenRouter) to give judge diversity across infrastructure, not just
model weights. It now runs as `nvidia/nemotron-3-super-120b-a12b` via
OpenRouter, pinned to `deepinfra/bf16`, because this environment holds only
`OPENROUTER_API_KEY` — there is no `NVIDIA_API_KEY` / direct NIM access
credential available to call `https://integrate.api.nvidia.com/v1` at all.
**Consequence:** all three judges (B4, B5, B6) now share the OpenRouter
routing layer. B6 still differs from B4/B5 in model weights and training
stack, so it is not a redundant data point, but it no longer tests a fully
independent inference stack — a bug or reporting quirk specific to OpenRouter
could in principle confound all three judges simultaneously in a way an
independent NIM path would have ruled out. This is a limit on what B6 can
claim, not something to route around quietly.

---

## 8. B4's strict JSON parser understated its recall — fixed with a prose-repair pass

`omitbench/judges.py::_parse_message` requires the model to return a JSON
object mapping each requirement to `IMPLEMENTED`/`OMITTED`; on any parse
problem it defaults the affected requirement(s) to `IMPLEMENTED` (never
`OMITTED` — see that function's docstring for why the default is one-sided).

Auditing B4's 7 call-level parse failures (of 919 total calls) against the
raw cached responses found this was too strict for `openai/gpt-oss-120b`
specifically: gpt-oss-120b sometimes reasons to a clearly stated conclusion
("`app.py::helper` -> OMITTED", "So OMITTED.") but never emits the requested
JSON object at all — the strict parser then silently discarded a conclusion
the model had already reached, defaulting to `IMPLEMENTED` and costing B4 a
real recall point. Of the 7 failed calls: 3 were genuine losses of this
shape (`8355a3f4c946`/ABSENT and /STUB, `f32ecb269f90`/ABSENT — all had the
target requirement's gold label OMITTED and the model's own reasoning
already said OMITTED); 2 were confounded with the truncation issue below
(diff cut off before the model could reach a conclusion at all —
`93d0a8240780`/ABSENT, `7e47f3b00fa1`/STUB); 2 had no recall impact (a CLEAN
variant with nothing to omit, and a missing key whose gold was IMPLEMENTED
anyway).

**Fix:** `_repair_from_prose` (`omitbench/judges.py`) runs only over
requirements the strict JSON parse already defaulted — a value the JSON
confidently gave is never reconsidered — and only recovers a verdict when a
bare `OMITTED`/`IMPLEMENTED` token appears in a bounded window after the
*last* mention of that exact requirement string in the raw response,
stopping at the next requirement's name so one window can't bleed into the
next. No fuzzy/synonym matching. `parse_failed` still reports `True` when
this fires (the JSON genuinely was never emitted — that fact stays visible
in `stats.parse_failures`), it just no longer forces the verdict itself to
the default. This is a pure re-parse of already-cached raw responses: **no
new API calls**, and it is a no-op for B5/B6 (0 parse failures observed for
either in this corpus, so their cached responses never reach the repaired
code path even though it lives in the shared `_parse_message`).

**Effect on the numbers:** B4 MCC 0.383 → 0.388 [0.345, 0.434] (barely
moves — 3 cells out of 2184). Paired P1-vs-B4: +0.116 → +0.111
[+0.055,+0.164], still excludes zero — P1 still beats B4. The fix was
correct and worth keeping (it removes a real, one-sided harness bug that
was specifically punishing B4, not a coincidence of numbers), but it did
not change which detector wins.

---

## 9. T3 — P2/P3 reachability: public-API exemption tried, did not clear the
bar, removed from the scored detector set

**The problem (TASKS.md T3).** P2 (`defined+reachable`) and P3
(`defined+reachable+body`) scored FPR 0.834 / 0.846 and lost to `B3
line-grep` with a paired interval excluding zero in both directions
(−0.281, −0.178). Reachability — "is this symbol called from outside its
own body?" — condemns any public API symbol, because public API is by
definition called from *outside the repo*, not from within it.

**The rule tried (exemption, `omitbench/detectors.py`).** A symbol that
fails the internal-call-site check is treated as reachable anyway if the
`after` snapshot shows any of: (1) listed in a module-level `__all__`, (2)
re-exported by name or via `from .mod import *` in its package's
`__init__.py`, (3) carries a decorator and isn't underscore-prefixed.
Decided from Python packaging semantics before measuring, not tuned to a
named list of framework decorators — see the docstrings on
`_has_all_export` / `_reexported_in_init` / `_decorated_public` /
`_is_exempt_public_api` for exactly what each checks and why. Applied only
to the reachability leg; the stub check (`_hollow`, P3-only) is unchanged,
so an exempted symbol with a hollow body is still caught by P3
(`test_p3_still_catches_stub_on_an_exempted_public_symbol`).

**Measured once**, after `make pilot` (ground truth unchanged — only
`detectors.py` changed, `mutate.py` did not):

| metric | P2 before | P2 after | P3 before | P3 after |
|---|---|---|---|---|
| MCC [95% CI] | 0.089 [0.048,0.129] | 0.164 [0.118,0.209] | 0.193 [0.160,0.227] | 0.365 [0.323,0.411] |
| FPR | 0.834 | 0.587 | 0.846 | 0.608 |
| vs B3, ΔMCC [95% CI] | −0.281 [−0.343,−0.225] * | −0.207 [−0.268,−0.143] * | −0.178 [−0.237,−0.114] * | −0.006 [−0.067,+0.060] |
| recall ABSENT / UNWIRED / STUB | .99 / .58 / .84 | .98 / .58 / .58 | .99 / .58 / 1.00 | .98 / .58 / 1.00 |

(`*` = interval excludes zero.)

**Verdict, against the criterion fixed before measuring** ("P2/P3 beat B3
with a paired interval excluding zero, on the positive side, or they are
removed"): **not cleared.** P2 still loses to B3 significantly. P3 moved
from *significantly worse* to *statistically indistinguishable* — its
interval now straddles zero, mean slightly negative — which is real
progress but is a tie, not a win. Per the pre-agreed protocol (**not**
re-tuned after seeing this): **P2 and P3 are removed from `DETECTORS`**
(`omitbench/detectors.py`) and no longer appear in the headline table or
any paired comparison. The functions themselves (`d_reachable`, `d_full`,
`_hollow`, and the three exemption helpers) are left in the file, clearly
marked as retired and unregistered, rather than deleted — this was a
real, principled attempt, not a strawman, and the code plus these numbers
are the record of why it fell short.

**Two findings worth keeping, beyond "didn't clear the bar":**

1. **P2's old STUB recall (0.84) was mostly an artifact of the FPR bug, not
   real stub detection.** `d_reachable` has no `_hollow` check — it cannot
   see a hollow body by construction, only P3 can. Before the exemption,
   P2 was over-flagging ~83% of everything as OMITTED (the FPR bug), which
   *incidentally* also flagged genuine STUB cases as OMITTED for the wrong
   reason. Once the exemption removed the over-flagging, P2's STUB recall
   collapsed to 0.58 — much closer to what "defined+reachable, blind to
   body" should structurally produce. A detector's apparent strength on a
   mutation class it has no mechanism to detect is a sign to check *why*,
   not a result to report at face value.
2. **The predicted UNWIRED-recall cost did not show up.** The exemption
   should cost recall on any UNWIRED target that is *both* internally
   called (a precondition for `mut_unwired` to fire at all) and
   independently public (`__all__`/decorator/re-export) — a real mechanism,
   pinned by `test_p2_exemption_does_not_mask_a_real_unwired_mutation`. At
   n=19 for `UNWIRED` (ASSUMPTIONS.md §5), recall stayed exactly 0.58
   before and after. Absence of a visible cost here does not disprove the
   mechanism; it says the overlap between "internally called" and
   "independently public by one of these three signals" is small or zero
   in this corpus at this n, which is itself a small-n statement, not a
   clean one.

**The wider lesson, stated as TASKS.md asked:** reachability is the wrong
signal for library code, and a principled, non-corpus-tuned attempt to
patch it with packaging semantics *narrows* the gap (FPR down ~25 points,
P3 goes from losing to tied) without closing it. That's evidence the
problem is structural, not a matter of a better exemption list: a symbol's
"public-ness" is a fuzzy, context-dependent property (public by convention
without `__all__`, used only from tests, wired through a plugin registry,
called only via monkey-patching) that static analysis of `(before, after)`
alone cannot fully recover. `P1 defined` remains the only proposed
detector in the headline table; it does not claim to catch `UNWIRED` or
`STUB` and never has.

---

## 10. T4 — 8 real trajectories collected, 0 contain an omission: recall on
real agent output is not yet measurable

**What we did.** Collected 8 hand-labelled trajectories from a real coding
agent (`claude-code`) working real GitHub issues on repos already in this
project's orbit but distinct commits from the synthetic corpus: `tqdm` (5
trajectories), `bandit`, `marshmallow`, `mkdocs` (1 each). Each trajectory
is the agent's actual before/after repo snapshot plus a hand-labelled
verdict — `IMPLEMENTED` or `OMITTED` — per newly-required symbol, loaded via
`omitbench/real.py::load_real()` into the same `(spec, reqs, before, after,
gold)` shape the synthetic pipeline uses, so every registered detector runs
against it completely unchanged (`omitbench/real.py::score()`;
`scripts/analyze.py --source real`).

| iid | repo | requirements | OMITTED |
|---|---|---|---|
| real_001 | tqdm | 5 | 0 |
| real_002 | tqdm | 3 | 0 |
| real_003 | tqdm | 3 | 0 |
| real_004 | bandit | 1 | 0 |
| real_005 | marshmallow | 1 | 0 |
| real_006 | tqdm | 1 | 0 |
| real_007 | mkdocs | 1 | 0 |
| real_008 | tqdm | 1 | 0 |
| **total** | | **16** | **0** |

**The finding, stated plainly.** All 16 requirements across all 8
trajectories were labelled `IMPLEMENTED`. This coding agent, on these 8
tasks, did not omit anything a human labeller could find. **This sample
cannot measure real-corpus recall** — recall is `TP / (TP + FN)`, and with
zero real `OMITTED` examples there is no positive class to compute it over.
The same zero-positives problem degrades further than it first appears:
**precision is not usable either**, for a different reason than recall —
`TP = 0` whenever there are no positive gold labels, so precision =
`0 / (0 + FP)` reads `0.00` for *every* detector regardless of how many
false positives it actually raises. It is not "perfect precision"; it is an
artifact of the denominator, and reading it as a real number was a mistake
caught and corrected in `scripts/analyze.py::analyze_real()` before this
note was written (its printed WARNING block says the same thing at run
time, not just here). MCC is degenerate for the identical reason — `TP + FN
= 0` forces its denominator to 0, so every detector's MCC reads `0.000`
here independent of behavior.

**What IS measurable on this sample: FPR on the `IMPLEMENTED` class.**
There are 16 real negatives and detectors do differ in how many they flag:

| detector | real FPR (n=16) | synthetic FPR (n=1559) | delta |
|---|---|---|---|
| B0 flag-nothing | 0.000 | 0.000 | +0.000 |
| B1 flag-everything | 1.000 | 1.000 | +0.000 |
| B3 line-grep (no AST) | 0.250 | 0.039 | +0.211 |
| P1 defined | 0.062 | 0.043 | +0.020 |

`P1`'s false-positive rate barely moves from synthetic to real (+0.020 on a
16-negative sample — well within noise at this n, but directionally
consistent rather than reversed). `B3` drifts more (+0.211): real diffs
appear to contain more grep-confusable name reuse (same identifier
appearing elsewhere in the file/diff without being the definition in
question) than the injected mutations do. Neither number says anything
about recall — a detector could have this exact FPR profile and still miss
every real omission there ever was, or catch all of them; this sample
cannot distinguish those cases.

**Why this is not surprising, and not evidence the synthetic corpus is
representative.** 8 trajectories on real, presumably-competent-agent runs
against well-scoped GitHub issues is a small sample from a distribution
that plausibly has a low omission rate to begin with — real coding agents
on clearly-specified, single-issue tasks may simply omit less often than a
forced single-symbol mutation does. Zero-in-16 is consistent with a low but
nonzero true omission rate; it is also consistent with the true rate being
near zero on tasks shaped like these 8. This sample has no power to tell
those apart, and it is one order of magnitude short of the 40–60
trajectories TASKS.md T4 calls for.

**Recommendation for future work (not done here — collecting more data
would be tuning the result, see CLAUDE.md anti-pattern #5's spirit applied
to data collection, not just detectors):**

1. **Collect from a weaker or less-guided agent.** A smaller/older model, or
   the same agent given less specification (vaguer issue text, no explicit
   acceptance criteria), is more likely to produce genuine omissions to
   label — the synthetic corpus's own justification (`ASSUMPTIONS.md` §1)
   is that STUB/UNWIRED-shaped failures are what agents that don't fully
   finish a task look like.
2. **Deliberately include harder tasks.** Multi-file, multi-requirement
   issues (the kind more likely to have a plan item silently dropped) are
   under-represented here — 5 of 8 trajectories have exactly 1 requirement,
   where "omit part of the plan" isn't structurally possible.

Either direction should be pursued **before** drawing any conclusion from
this table beyond "FPR looks stable"; growing the `OMITTED` count by
picking easier-to-satisfy criteria after the fact would be p-hacking the
corpus, not measuring it.

---

## 11. T5 — requirement-extraction error term: extraction is the dominant
bottleneck, not a small tax (synthetic corpus only)

**What this measures.** Every requirement scored elsewhere in this repo
comes from `mutate.py::new_symbols(patch_before, patch_after)` — the gold
patch itself. That is a perfect, structured "extractor" that assumes a real
agent's plan decomposes exactly onto the symbols the reference solution
happens to define (§2, above). T5 replaces it with a genuinely imperfect
source: an LLM (`omitbench/extract.py`, `mistralai/mistral-medium-3` via
OpenRouter, same model/provider pin as B5) reads **only** the commit
message / issue text for a synthetic instance and predicts which symbols it
expects the eventual patch to define — **never the diff**. Blindness is
enforced the same way as every detector/judge: `tests/test_leakage.py` locks
`extract_one`'s signature to `task_text` alone, locks the pure
`build_extract_prompt(task_text)` prompt-builder's signature, and plants a
canary. Real T4 trajectories (`data/real/`) are **out of scope for T5** —
n=8 is far too small to extract requirements from meaningfully; this is
flagged as future work, not measured here.

**Matching method, frozen before the extraction prompt was written.** Output
schema: a JSON array of `{"symbol": "<name>", "path": "<path-or-null>"}`.
`path: null` is an allowed, honest abstention — the model is never forced to
fabricate a file path it has no textual basis for. Two match types are
computed and reported, **always both, never folded into one number**:

1. **symbol-only** — extracted `symbol` equals an oracle symbol, exactly,
   case-sensitive. Ignores path entirely.
2. **path-qualified** — extracted `(path, symbol)` equals an oracle
   `(path, symbol)` pair, exactly. Strictly harder.

Both use **multiset (Counter) comparison**, not set — caps credit at
`min(count_oracle, count_extracted)` per name, so two same-named symbols in
different files can't be double-credited by one guess. Aggregation is
**micro-averaged** (pool TP/FP/FN across every instance, then compute one
P/R/F1 — same convention as `experiment.py::cells()`/`metrics()`), with a
**cluster bootstrap over instances** (`iid`), never records (CLAUDE.md rule
4). All of this is pure Python set/multiset arithmetic
(`scripts/score_extraction.py`) — **no LLM judge anywhere in the matching
path**: judging LLM-extracted text with another LLM would reintroduce
exactly the noise source T5 exists to isolate from.

**Known limitation of symbol-only matching, stated up front (this is a
design tradeoff, not a bug):** it over-credits an extractor that names the
right function but attaches it to the wrong file, or no file at all — real
signal about what the LLM understood, but blind to *where*. That is exactly
why path-qualified is reported as a separate, stricter number and never
folded into symbol-only.

**Falsification condition (added to `TASKS.md` T5, which had none before
this task).** If symbol-only extraction F1 came out above ~0.9, that would
mean extraction is not the bottleneck and the resulting detection-F1 tax
would be expected to be small — a valid, honest result to report as such,
not grounds to keep refining the prompt.

**Result: the falsification bar was not cleared, by a wide margin.**

| match type | Precision | Recall | F1 | 95% CI |
|---|---|---|---|---|
| symbol-only (n=310) | 0.116 | 0.046 | **0.066** | [0.044, 0.091] |
| path-qualified (n=310) | 0.007 | 0.003 | **0.004** | [0.000, 0.010] |

**A diagnostic checked before reporting this, because the number looked
suspiciously low, NOT a corpus filter (CLAUDE.md anti-pattern #1 — this
subset is reported alongside the full number, never substituted for it):**
79.9% of oracle requirements (590/738) live in a test-file path
(`tests/...` or a `test_*` module) — `new_symbols()` has no
production-vs-test filter, so a commit that adds
`test_hook_new_field_without_alias` counts as a "requirement" exactly like a
production symbol does, and no commit message states a test function's
literal name before it is written. Restricting to the 76 instances with
≥1 non-test-path oracle requirement (still not the headline; a diagnostic
subset) raises symbol-only F1 to 0.220 [0.143, 0.300] — better, but still an
order of magnitude short of the 0.9 bar.

**Sanity-checked before trusting these numbers (not degenerate):** 150/310
instances got a non-empty extraction, 160/310 got an honest empty array
(correct behaviour when the commit message gives no textual basis to
guess). Raw samples: a vague message (`"ParamType typing improvements"`)
correctly abstained; an informative one
(`"Add tests/test_validators.py::TestOr test cases"` verbatim in the commit
body) was correctly extracted as `tests/test_validators.py::TestOr`
(one of only 2 path-qualified hits in the whole corpus — the model used a
path stated explicitly in the text, not inferred it); another instance's
commit body literally said `"Update httpx/_auth.py"`, and the model
correctly attached `NetRCAuth` to that path. Zero JSON parse failures across
303 real calls (matches B5's historically clean behaviour, §8 above). Real
cost: **$0.0567** for the full 310-instance sweep (303 calls + 7 cache hits
from a smoke test), cached under `cache/extract_*.json` and committed, same
"re-running costs $0" promise as the judges.

**Re-running P1 detection with extracted requirements (`scripts/run_extraction_tax.py`),
paired on the identical 310 instances P1's headline row already covers.**
Two conditions, both derived from the SAME extracted items, never
re-extracted or re-tuned after seeing either result:

- **(a) EXTRACTED-A, literal pipeline** — the extractor's `(symbol, path)`
  fed to P1 exactly as produced, wrong or null paths included. **Primary /
  headline** extraction-tax number: a real deployment faces exactly this.
- **(b) EXTRACTED-B, symbol-identification-only** — restricted to items
  whose *symbol* matches some oracle requirement for that instance, with the
  oracle's true path substituted in; non-matching items are dropped
  entirely (kept, they would just reintroduce the same path-noise as (a),
  defeating the point of isolating path-attribution from
  requirement-understanding). **Secondary diagnostic.**

Ground truth for an extracted item is **symbol-based**, mirroring
`experiment.py`'s own rule (`gold = "OMITTED" if (cls != "CLEAN" and r ==
target) else "IMPLEMENTED"`) but compared on the symbol alone: whether a
symbol was truly omitted from the diff doesn't depend on whether the
extractor guessed its file correctly. If a mutated variant's true target was
never named by *any* extracted item, one synthetic
`(pred=IMPLEMENTED, gold=OMITTED)` row is added — a real deployment would
silently miss that requirement entirely, and recall must reflect that, not
just recall computed over the subset the extractor happened to mention.

| condition | P | R | F1 | MCC | n=310 |
|---|---|---|---|---|---|
| ORACLE (existing headline row) | 0.802 | 0.445 | 0.572 | **0.499** | |
| EXTRACTED-A literal (headline tax) | 0.046 | 0.062 | 0.053 | **−0.935** | |
| EXTRACTED-B symbol-only (diagnostic) | 0.923 | 0.020 | 0.039 | **0.006** | |

Paired ΔMCC, 95% CI, both exclude zero: ORACLE − A **+1.434
[+1.378, +1.488]**; ORACLE − B **+0.493 [+0.424, +0.592]**.

**Reading this plainly.** Under the literal pipeline, P1's MCC does not just
drop, it goes **negative** — worse than flagging nothing at all
(B0's MCC is 0.000 by construction). This is a real, structural consequence
of P1 being a **path-qualified** detector (§3: qualification was added
specifically because bare-name matching let a same-named symbol elsewhere
mask a real deletion, capping `ABSENT` recall at 0.43). Given a wrong or
`null` path, `d_defined` cannot find the symbol at all and defaults to
OMITTED; with most extracted paths wrong or absent this fires on nearly
everything, collapsing precision (0.046) by far more than it helps recall.
Condition B isolates the two failure modes cleanly: once the oracle's true
path is substituted for every item whose symbol was named correctly (and
noise from non-matching items removed), precision recovers to 0.923 — P1
does its job once it knows where to look — but recall stays at 0.020,
because the extractor so rarely names the true omitted symbol as one of its
guesses in the first place. **The tax is dominated by
requirement-understanding (the extractor doesn't guess the right symbols
often enough), and catastrophically compounded by path-attribution the
moment a pipeline is deployed literally.** The non-test-path secondary
table (n=76, `has_nontest` instances, in-scope omissions restricted to
non-test-path targets) tells the same story at smaller n: ORACLE MCC 0.578
→ EXTRACTED-A −0.789 → EXTRACTED-B 0.160.

**Per the pre-agreed protocol, this result is reported as measured.** No
extraction prompt, matching threshold, or corpus change was made after
seeing either Step 3's or Step 4's numbers — the large gap is the finding,
not a defect to paper over. One implementation bug in the Step 4 *scoring
harness itself* (condition B) was found and fixed after seeing a result,
and both the wrong and corrected numbers are disclosed below rather than
only the corrected one — see "Audit trail: the condition-B scoring fix"
immediately below. `scripts/run_extraction_tax.py` and
`scripts/analyze_extraction.py` regenerate every number in this section and
the README's T5 section from committed artifacts
(`results/extraction/*.json*`, `cache/extract_*.json`) with **zero new API
calls**.

### Audit trail: the condition-B scoring fix (disclosed, not hidden)

**What was originally implemented, and its result.** The first version of
`scripts/run_extraction_tax.py::score_instance` scored *every* extracted
item under condition B, substituting the oracle's true path only for items
whose symbol matched an oracle requirement — but for a **non-matching**
item, it fell back to scoring that item with its own (extracted, possibly
wrong or `null`) path, exactly as condition A does. Run against the real
310-instance corpus, this produced:

| condition | P | R | F1 | MCC |
|---|---|---|---|---|
| EXTRACTED-B, original (uncorrected) | 0.016 | 0.020 | 0.018 | **−0.898** [−0.936, −0.856] |

**What was wrong with it, independent of the number.** Condition B's own
definition — written before any code existed, in the task instructions
this work was done against — is *"oracle path substituted in for extracted
items that symbol-matched... isolating whether the tax comes from
requirement-understanding or path-attribution."* A non-matching item is a
**requirement-understanding failure** (the extractor named something with
no basis in the real requirement list) — it was never supposed to be part
of a metric that holds requirement-understanding fixed and varies only path
handling. Scoring it anyway with a broken path reintroduces exactly the
noise source B exists to exclude, which is why the original B (−0.898) came
out nearly identical to A (−0.935): it wasn't isolating anything.

**The fix.** `score_instance` was changed to include a symbol in condition
B's rows *only* when `sym_to_path.get(sym) is not None` (a real oracle
match exists for it); non-matching items are skipped for B entirely rather
than scored with a fallback path. Condition A's code path is untouched.
Re-run against the same corpus, same cached extraction, same rebuilt
instances:

| condition | P | R | F1 | MCC |
|---|---|---|---|---|
| EXTRACTED-B, corrected | 0.923 | 0.020 | 0.039 | **+0.006** [−0.084, +0.053] |

**Full before/after diff** (`score_instance`, `scripts/run_extraction_tax.py`):

```diff
     rows_a, rows_b = [], []
-    matched_target = False
+    matched_target_a = False
+    matched_target_b = False

     for it in items:
         sym = it["symbol"]
-        path_a = it.get("path") or ""
         gold = "OMITTED" if (in_scope_omission and sym == target_symbol) else "IMPLEMENTED"
         if in_scope_omission and sym == target_symbol:
-            matched_target = True
+            matched_target_a = True

+        # A: literal -- extracted path exactly as given, wrong/null included.
+        path_a = it.get("path") or ""
         req_a = f"{path_a}::{sym}"
         pred_a = P1("", [req_a], before, after, {})[req_a]
         rows_a.append((pred_a, gold))

-        oracle_path = sym_to_path.get(sym)
-        path_b = oracle_path if oracle_path is not None else (it.get("path") or "")
-        req_b = f"{path_b}::{sym}"
-        pred_b = P1("", [req_b], before, after, {})[req_b]
-        rows_b.append((pred_b, gold))
+        # B: symbol-identification-only -- ONLY items whose symbol matches
+        # some oracle requirement's symbol are scored at all, with the
+        # oracle's true path substituted. A non-matching item is NOT a
+        # path-attribution failure -- it's a requirement-understanding
+        # failure, which is exactly what B is supposed to exclude, not
+        # re-score with a broken path.
+        oracle_path = sym_to_path.get(sym)
+        if oracle_path is not None:
+            req_b = f"{oracle_path}::{sym}"
+            pred_b = P1("", [req_b], before, after, {})[req_b]
+            rows_b.append((pred_b, gold))
+            if in_scope_omission and sym == target_symbol:
+                matched_target_b = True

-    if in_scope_omission and not matched_target:
+    if in_scope_omission and not matched_target_a:
         rows_a.append(("IMPLEMENTED", "OMITTED"))
+    if in_scope_omission and not matched_target_b:
         rows_b.append(("IMPLEMENTED", "OMITTED"))
```

**Order of operations, stated plainly.** The uncorrected number (−0.898)
was seen *before* the fix was made — the fix was not made blind to it. What
triggered the fix was noticing that B's result was suspiciously close to
A's (both catastrophically negative), which meant B wasn't isolating
anything, which prompted re-reading condition B's own definition (already
fixed in the task instructions before any code was run) against the code
and finding a genuine mismatch between the two. This is disclosed as a
scoring-harness correction against a pre-existing specification, not as a
result-driven tweak to the extractor, the matching thresholds, or the
corpus — none of which were touched — but the number came first,
chronologically, and that is stated here rather than implied otherwise.

**Condition A is unchanged by this fix.** Every EXTRACTED-A row in this
document (P=0.046, R=0.062, F1=0.053, MCC=−0.935, tp=38 fp=791 fn=571 tn=7
on the primary 310-instance table) is byte-identical whether computed from
the original or the corrected `score_instance` — verified by re-running
both versions against the same cached extraction and rebuilt instances
immediately before this entry was written.

Both numbers (−0.898 and +0.006) are kept in this record permanently, not
just the corrected one, so this correction is auditable rather than
asserted.

---

## 12. T6 — CI gate: precision margin, detector choice, requirement source

**Precision is not comfortably above the 0.80 acceptance bar.** Point
estimate 0.8018 (`results/baseline_t6.json`), cluster-bootstrap 95% CI
[0.723, 0.885] (same method as every other CI in this document — resampled
over instances, not records, per §5). The interval straddles 0.80 on both
sides. `scripts/check_gate_precision.py` implements the falsification
condition exactly as decided (point estimate vs. 0.80) because that is
what TASKS.md T6 specifies, but a single future measurement crossing 0.80
in either direction should be read against this CI, not treated alone as
proof of a real change.

**Why P1 only, not a judge.** B5 (the strongest judge, T1) has higher MCC
(0.701 vs 0.499) but lower precision (0.74) than P1 — it fails T6's own
0.80 precision bar outright. It also requires a paid API call per PR,
which is an operational liability (cost, latency variance, an external
dependency a CI gate now depends on) that P1's determinism avoids
entirely.

**Why requirements are a structured checklist, not extracted from prose.**
T5 (§11) measured what an LLM extractor produces from commit-message-style
text alone: MCC −0.935 when fed straight into P1 — worse than flagging
nothing, dominated by wrong-or-null path attribution. A live PR's
description is exactly this kind of free text; reusing T5's extractor (or
building a similar one) for T6 would reproduce that same collapse. The
gate instead requires the PR/issue body to name requirements explicitly,
as a markdown checklist (`omitbench/gate.py::parse_requirements`) — no
LLM, no network, and no guessing.

**Why the PR-comment disclaimer leads with "structural," not "n=19."**
P1's blindness to `UNWIRED`/`STUB` is 0.00 recall by construction — it is
an "is this symbol defined" check and neither of those omission shapes
touches the not-defined question at all (`omitbench/detectors.py`
docstrings). The offline benchmark's `UNWIRED` sample being small (n=19,
§5, TASKS.md T2) is a real but separate limitation: it means the *0.00
recall number itself* is not independently re-verified at scale, not that
recall would be higher with more data.
`omitbench/gate.py`'s `STRUCTURAL_BLINDNESS_DISCLAIMER` states the
structural fact first and cites n=19 as supporting context, so
the comment doesn't imply "we're not sure how good we are" when the
correct statement is "we are not designed to catch this at all."

---

## 13. B7 — TypeSafe Jev as a probabilistic omission judge (PRE-REGISTERED)

**Written 2026-09-23, before any real Jev call.** Everything in this section
was fixed before the first response came back, and is not edited after it.
If a harness bug turns up in the smoke run (`--limit-instances 5`), the
harness gets fixed (parsing, pinning, plumbing) and the fix is recorded
below. The question text in `omitbench/jev.py` is **frozen** at the commit
that adds this section. Any change to the question text makes a new
detector id that is reported next to B7, never in place of it.

**Why.** B5 beats P1 (§ T1 results) but gives a bare label, so there is
nothing to threshold. Jev returns P(yes) per typed question at roughly
100 ms and $0.042/1M input tokens. TypeSafe describes the model as
"calibrated". Its public docs (`docs.typesafe.ai/confidence.md`, checked
2026-09-23) give no calibration numbers. This corpus has a ground-truth
label for every requirement, so the claim can be tested on a hard,
out-of-domain task. As far as we found, nobody has tested it independently.

**Inputs held equal to B5.** Same `judges.unified_diff`, same
`DIFF_TOKEN_CEILING` truncation, and the same three-part test (defined,
wired, not a stub) in the question text. Model pinned to `jev-1.13.0`, not
the `jev-latest` alias. A response reporting any other model raises.

**Two variants, fixed now, no others:**
- `B7 Jev (single)`: one Noul per requirement. P(implemented) = Noul value.
- `B7 Jev (split)`: three Nouls (defined / wired / real body).
  P(implemented) = **min** of the three. Min is the Fréchet upper bound on
  "all three hold". It was chosen over the product, which assumes
  independence these questions do not have, because it errs toward
  IMPLEMENTED. That is the same direction as judges.py's "never default to
  OMITTED" rule.
- Verdict for both: OMITTED iff P(implemented) < **0.5**. Not tuned.
- A missing or malformed answer defaults to IMPLEMENTED and is counted as a
  parse failure (same rule as B4/B5/B6).

**Known handicaps, stated up front.** The single variant asks a compound
question, which TypeSafe's Noul docs advise against. Jev's own "jaggedness"
page lists multi-hop reasoning and large irrelevant state as weak spots,
and a 6000-token diff is large state. If B7 loses, these are the first
explanations to check, and they are listed here so they cannot be
discovered after the fact and used as excuses.

**Decision rules (read the output against these, nothing else):**
1. *Discrimination.* Paired cluster bootstrap of MCC, B7 − B5
   (`scripts/analyze.py`). If the interval contains 0, B7 "matches" B5. If
   the interval lies entirely below 0, B7 "loses". If it lies entirely above
   0, B7 "wins". No directional prediction is made.
2. *Calibration claim.* "Calibrated on this task" is supported only if all
   three hold for the variant: overall ECE ≤ 0.05 (10 equal-width bins),
   every reliability bin with n ≥ 30 has |predicted − observed| ≤ 0.10,
   and Brier skill vs. the base-rate forecaster > 0. Otherwise the report
   says "not supported on this task", which is a finding, not a failure.
3. *Do probabilities add value?* B7's Brier score vs. the Brier score of
   B7's own thresholded verdicts scored as a 0/1 forecaster. If the
   probabilistic Brier is not lower, the probabilities add nothing over the
   label on this task.
4. *Abstain band.* 0.3 ≤ P(omitted) ≤ 0.7 is routed to "ask a human". This
   is TypeSafe's suggested review band, not one fitted here. Report
   coverage and MCC on the confident remainder. This is the number a Stop
   hook would actually run on.

**Budget.** `scripts/run_jev.py --max-cost 1.00` (default) hard-stops.
Expected total for both variants is well under $0.25 at list price. The
exact figure comes from `--dry-run` over the real corpus and is recorded
here after the dry run, before the sweep.

*Dry run, 2026-09-23, Python 3.12.14, before any real call* (`make
jev-dry-run`, 310 instances, alignment check passed 310/310):

| variant | calls | input tokens (chars/4 estimate) | est. cost | truncated diffs |
|---|---|---|---|---|
| B7 Jev (single) | 919 | ~1.73M | $0.073 | 32 |
| B7 Jev (split) | 919 | ~2.06M | $0.087 | 32 |

Total: 1838 calls, est. $0.160, under the $1.00 hard stop.

**Not claimed.** B7 results say nothing about real agent trajectories
(§1, §10). They also say nothing about Jev on tasks it was built for.
They are one model on one hard, out-of-domain classification task.

**Harness notes (fixes to plumbing, not to decision rules).**
- *2026-09-23, interpreter pin.* Shards and B7 are built under Python
  <=3.13 (3.11 and 3.12.14 both verified at 30 itsdangerous instances).
  Cause: PEP 758 in Python 3.14 makes Python 2 `except X, e:` parse, so
  `build_base`'s `ast.parse` gate admits 5 extra Python 2 commits
  (itsdangerous 30 -> 35, total 315 vs the shards' 310), which the
  alignment check caught on the first dry run. `scripts/run_jev.py` now
  exits on Python >= 3.14 and prints the interpreter version in its stderr
  output (`tests/test_jev.py::test_run_jev_refuses_python_314_plus`).
  `build_base` itself is unchanged in this branch.
