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
- **`UNWIRED` has n=16.** That is far too few for any claim. Recall numbers in
  that column are reported for completeness and should be read as noise.

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
