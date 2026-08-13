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
