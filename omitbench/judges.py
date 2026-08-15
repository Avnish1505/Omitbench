"""
LLM-as-judge baselines (B4/B5/B6). Same contract as every detector in
detectors.py:

    d_xxx(spec, reqs, before, after, ctx) -> {req: "IMPLEMENTED"|"OMITTED"}

Blind to: the gold patch, the mutation class, the ground-truth label, and the
full repository -- exactly the ONE INVARIANT in CLAUDE.md, same as every
other detector. Additionally blind to anything not visible inside the
unified diff it is shown: a hollow STUB body sitting outside the diff's
context window, or a call site the diff doesn't happen to include, is as
invisible to a judge as it would be to a human reviewer skimming only the
diff.

NEW LEAK SURFACE. Every other detector's only leak surface is its function
signature (enforced by tests/test_leakage.py's three guards). Judges add a
second one: the rendered PROMPT STRING. Prompt construction is therefore
isolated in one pure function, build_prompt(reqs, diff), which takes no
other arguments and touches no global state -- see
tests/test_leakage.py::test_judge_prompt_never_contains_canary. All three
judges below share this one function; if it ever grows a third argument,
that argument is where the next leak will ride in.

THREE JUDGES, NOT ONE. A single judge cannot support the claim that LLM
judges are structurally weak at omission specifically -- a reviewer's first
objection would be "you picked a weak model." B4/B5/B6 span three different
vendors/model families (an open-weight reasoning MoE, a proprietary mid-tier
model, and NVIDIA's own Nemotron line) so the finding, if it holds, holds
across vendors rather than being an artifact of one of them.

UPDATE 2026-08-13: B6 ROUTES THROUGH OPENROUTER, NOT DIRECT NVIDIA NIM. B6
originally called NVIDIA NIM (`https://integrate.api.nvidia.com/v1`) directly
with its own `NVIDIA_API_KEY`, serving `meta/llama-3.3-70b-instruct` -- a
dense model, and genuinely a second broker independent of OpenRouter, which
was the point of the "NIM entirely separate from OpenRouter" framing this
docstring used to make. We do not hold a direct NVIDIA API key; only
OPENROUTER_API_KEY is available. B6 was therefore re-pointed to go through
OpenRouter like B4/B5, to `nvidia/nemotron-3-super-120b-a12b` (NVIDIA's own
model, so the "different vendor/training stack" claim above still holds) --
see the B6 entry in JUDGES for the exact provider pin and the reasoning this
was picked. Two consequences worth stating plainly, not burying:
  (a) B6 is no longer a broker-independent data point -- all three judges now
      depend on OpenRouter's routing/reporting being trustworthy. Any RESULTS
      language claiming B6 as a check against an OpenRouter-specific bug is
      now false and must be corrected before citing it.
  (b) meta/llama-3.3-70b-instruct is dense; nemotron-3-super-120b-a12b is MoE
      (12B active / 120B total). Every dense NVIDIA-authored Nemotron model
      (49B/70B/253B/340B) had zero active OpenRouter endpoints when checked
      on 2026-08-13 -- not a preference, the only NVIDIA models actually
      being served. If a dense NVIDIA model comes back online later, prefer
      it over this MoE substitute for the "architecture diversity" plank of
      the argument above.

PROVIDER PINNING (OpenRouter only). OpenRouter load-balances one model slug
across multiple backend providers that serve measurably different
(often differently-quantized) variants of "the same" model. Left unpinned,
the k=5 variance run (scripts/run_judge.py --k 5) would measure provider
routing noise, not model output variance, and the two are not the same
finding. Every OpenRouter request therefore pins `provider.order` to one
slug with `allow_fallbacks: false`, and _judge() asserts the provider
OpenRouter says it actually used matches the pin -- FAILING LOUDLY (raising,
not logging) on a mismatch, because a silently-substituted backend is
exactly the confound pinning exists to prevent. This now applies to all
three judges, B6 included, since the update above put it behind OpenRouter
too -- there is no judge left with "nothing to pin".

The core (`omitbench.detectors`, `omitbench.mutate`, `omitbench.evidence`,
`omitbench.corpus`) is zero-dependency by design (ASSUMPTIONS.md / CLAUDE.md)
and this module does not change that: `openai` is imported lazily, only
inside the functions that make network calls, so `import omitbench.judges`
succeeds even with the package absent and the rest of the test suite is
unaffected. Install the extra with `pip install -e .[judge]` before running a
real sweep, and set OPENROUTER_API_KEY (all three judges route through
OpenRouter as of 2026-08-13; NVIDIA_API_KEY is no longer read by anything in
this module).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass

CACHE_DIR = "cache"
MAX_RETRIES = 4
VALID = {"IMPLEMENTED", "OMITTED"}

DIFF_TOKEN_CEILING = int(os.environ.get("OMITBENCH_JUDGE_DIFF_TOKEN_CEILING", "6000"))


# --------------------------------------------------------------------------
# 0. judge registry. Each entry is one fully-specified backend: which API,
#    which model, which provider (if the API brokers across providers), and
#    which reasoning setting. Every judge shares build_prompt and the same
#    (before, after) inputs -- only the backend differs.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeConfig:
    judge_id: str                       # matches the DETECTORS / analyze.py ORDER key
    base_url: str
    api_key_env: str
    model_slug: str                     # value sent in the request's "model" field
    provider_slug: str | None           # OpenRouter provider.order pin; None = not brokered
    reasoning_effort: str | None        # None = model has no reasoning-depth knob we set
    dry_run_price_per_1m: tuple[float, float]   # (in, out) USD/1M -- ESTIMATE, --dry-run only
    nim_price_per_1m: tuple[float, float] | None = None  # real-run cost fallback; see B6 below


JUDGES: dict[str, JudgeConfig] = {
    # B4: openai/gpt-oss-120b via OpenRouter, pinned to Cerebras's FP16
    # endpoint. gpt-oss-120b ships natively MXFP4-quantized and most
    # OpenRouter providers re-quantize further (fp4/fp8) for cost; Cerebras
    # is the one endpoint on this model's provider page
    # (openrouter.ai/api/v1/models/openai/gpt-oss-120b/endpoints, checked
    # 2026-08-13) explicitly labelled fp16 rather than "Unknown" or a lower
    # precision. Given the entire reason we pin providers is to control for
    # quantization variance, picking the most precisely documented one is
    # the point, not a tie-breaker. reasoning_effort="medium": gpt-oss-120b
    # is a genuine reasoning model (see RELATED.md on judge quality) and
    # "low" would under-use exactly the capability we are trying to give
    # this baseline a fair shot with; "high"/"max" roughly doubles spend for
    # a task (2-requirement classification) that does not obviously need it.
    # Override with OMITBENCH_JUDGE_B4_REASONING_EFFORT.
    "B4 LLM judge (gpt-oss-120b)": JudgeConfig(
        judge_id="B4 LLM judge (gpt-oss-120b)",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_slug="openai/gpt-oss-120b",
        provider_slug="cerebras/fp16",
        reasoning_effort=os.environ.get("OMITBENCH_JUDGE_B4_REASONING_EFFORT", "medium"),
        dry_run_price_per_1m=(0.35, 0.75),
    ),
    # B5: mistralai/mistral-medium-3 via OpenRouter. Justification for the
    # pick: (a) different vendor and training stack from both B4 (OpenAI
    # open-weights) and B6 (Meta via NVIDIA), so a finding that holds across
    # all three isn't a single-lab artifact; (b) genuinely mid-tier on both
    # price ($0.40/$2.00 per 1M vs gpt-oss-120b's $0.35/$0.75 and frontier
    # models several times higher) and general capability, sitting between
    # the other two rather than duplicating either; (c) no reasoning/thinking
    # mode to configure, which keeps this judge's comparison to B4 isolated
    # to "different model" rather than conflated with "different reasoning
    # setting". Mistral is the only provider OpenRouter lists for this model
    # (checked 2026-08-13) -- pinning it is defense-in-depth against
    # OpenRouter adding a second, lower-fidelity provider later, not a
    # response to an existing quantization spread.
    "B5 LLM judge (mid-tier)": JudgeConfig(
        judge_id="B5 LLM judge (mid-tier)",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_slug="mistralai/mistral-medium-3",
        provider_slug="mistral",
        reasoning_effort=None,
        dry_run_price_per_1m=(0.40, 2.00),
    ),
    # B6: nvidia/nemotron-3-super-120b-a12b via OpenRouter, pinned to
    # DeepInfra's bf16 endpoint. RE-POINTED 2026-08-13 (see the module
    # docstring's "UPDATE 2026-08-13" note for the full reasoning) -- this
    # judge used to call NVIDIA NIM directly with its own NVIDIA_API_KEY,
    # serving meta/llama-3.3-70b-instruct (dense). We only hold
    # OPENROUTER_API_KEY, not a direct NVIDIA key, so B6 now goes through
    # OpenRouter like B4/B5, to an NVIDIA-*authored* model instead (keeping
    # the "different vendor" plank of THREE JUDGES, NOT ONE) rather than a
    # Meta model NVIDIA merely used to host.
    #
    # Model choice: nvidia/nemotron-3-super-120b-a12b (120B total, 12B
    # active -- MoE, not dense; every dense NVIDIA Nemotron model checked on
    # OpenRouter on 2026-08-13 -- the 49B/70B/253B/340B Llama-Nemotron line
    # -- returned an EMPTY endpoints array, i.e. no active provider serving
    # it at all, not a preference against them). 120B total is the closest
    # available scale to the retired 70B target, and its $0.085/$0.40 per 1M
    # pricing sits in the same tier as B4 ($0.35/$0.75) and B5
    # ($0.40/$2.00) rather than being a cut-rate substitute.
    #
    # Provider pin: deepinfra/bf16. Checked live endpoints for this model on
    # 2026-08-13: DeepInfra (bf16, 99.31% 30m uptime), DigitalOcean
    # (unlabelled precision), Nebius (fp4, only an 8K context window -- too
    # small a margin above DIFF_TOKEN_CEILING=6000 to risk). bf16 is the
    # highest-precision option on offer -- same "pick the most precisely
    # documented endpoint" logic as B4's cerebras/fp16 pin, for the same
    # reason: the whole point of pinning is controlling for quantization
    # variance, so picking the most precise one is the point, not a
    # tie-break.
    #
    # reasoning_effort=None, deliberately, even though this model DOES
    # expose a reasoning/thinking toggle (unlike the old NIM Llama 3.3 70B
    # target, which had none): turning it on would be a second, separate
    # change to what B6 measures, not just a transport swap. Leaving it
    # unset keeps this change scoped to "how B6 reaches its model", matching
    # the constraint it was made under.
    #
    # NOTE the JUDGES/DETECTORS key string below still reads "NVIDIA NIM" --
    # left unchanged deliberately (detector registration keys are not part
    # of this change). It is now a legacy label, not a description of the
    # transport: this judge no longer talks to NIM at all.
    #
    # NOTE cost: unlike the old NIM path, OpenRouter DOES return a `cost`
    # field in `usage` for this judge now (same `usage: {include: true}`
    # request _request_kwargs() already sends whenever provider_slug is
    # set) -- _extract_cost() will report cost_source="reported" for B6 same
    # as B4/B5, not "estimated", purely as a side effect of now being
    # OpenRouter-brokered; _extract_cost() itself is unchanged.
    # nim_price_per_1m is kept populated as a fallback of last resort should
    # OpenRouter ever omit usage.cost for this specific model.
    "B6 LLM judge (NVIDIA NIM)": JudgeConfig(
        judge_id="B6 LLM judge (NVIDIA NIM)",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_slug="nvidia/nemotron-3-super-120b-a12b",
        provider_slug="deepinfra/bf16",
        reasoning_effort=None,
        dry_run_price_per_1m=(0.085, 0.40),
        nim_price_per_1m=(0.085, 0.40),
    ),
}


# --------------------------------------------------------------------------
# stats -- module-level ledger the runner reads after a sweep, one per judge
# (a detector's return type is fixed at {req: verdict} by the contract, so
# cross-cutting numbers -- parse-failure rate, truncation rate, cache-hit
# rate, retries -- have to be reported out of band, same as cost).
# --------------------------------------------------------------------------

class _Stats:
    def __init__(self):
        self.calls = 0
        self.cache_hits = 0
        self.parse_failures = 0   # counted per API RESPONSE, not per requirement
        self.reqs_scored = 0
        self.truncated = 0
        self.retries = 0

    def reset(self):
        self.__init__()


STATS: dict[str, _Stats] = {}


def _stats_for(judge_id: str) -> _Stats:
    if judge_id not in STATS:
        STATS[judge_id] = _Stats()
    return STATS[judge_id]


# --------------------------------------------------------------------------
# 1. diff construction -- the only input besides the requirement list, and
#    built from the same (before, after) dicts every other detector gets.
#    Shared by all three judges. Unchanged from the single-judge version.
# --------------------------------------------------------------------------

def unified_diff(before: dict[str, list[str]], after: dict[str, list[str]],
                  context: int = 3) -> str:
    """before -> after, generated with difflib, across every file that
    changed. Same two dicts every other detector receives -- nothing else
    touches this path. `context` lines of surrounding code are kept per hunk
    so a judge can see nearby call sites, not just the changed line.
    """
    parts: list[str] = []
    for path in sorted(set(before) | set(after)):
        b, a = before.get(path, []), after.get(path, [])
        if b == a:
            continue
        lines = list(difflib.unified_diff(
            b, a, fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=context))
        if lines:
            parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _estimate_tokens(text: str) -> int:
    """chars/4 heuristic. Deliberately not a real tokenizer -- three
    different backends here (OpenAI-family, Mistral, Llama) would need three
    different tokenizers to be exact, and a real `count_tokens()` call would
    double the API traffic this truncation gate exists to bound. Good enough
    to gate truncation; the numbers that go in the README come from real
    `usage` on actual responses, not this estimate.
    """
    return max(1, len(text) // 4)


def _truncate(diff: str, ceiling: int) -> tuple[str, bool]:
    """Truncate on a line boundary once the diff's estimated token count
    exceeds `ceiling`. Truncation is a confound (it can only make OMITTED
    look like IMPLEMENTED, never the reverse, since evidence gets cut) --
    the caller MUST record and report the returned flag, never swallow it.
    """
    if _estimate_tokens(diff) <= ceiling:
        return diff, False
    keep_chars = ceiling * 4
    out: list[str] = []
    total = 0
    for line in diff.splitlines():
        if total + len(line) + 1 > keep_chars:
            break
        out.append(line)
        total += len(line) + 1
    out.append(f"... [diff truncated at ~{ceiling} estimated tokens] ...")
    return "\n".join(out), True


# --------------------------------------------------------------------------
# 2. prompt construction -- the second leak surface. PURE. No other inputs.
#    UNCHANGED by the OpenRouter/NIM rework: identical template, identical
#    signature, shared by all three judges. tests/test_leakage.py locks this
#    signature and plants a canary through it; neither needed edits here.
# --------------------------------------------------------------------------

_PROMPT_TEMPLATE = """You are reviewing a code change to judge whether specific requirements were actually implemented, not just mentioned.

Each requirement below names a symbol (a function or class) and the file it must live in, written as `path/to/file.py::SymbolName`. Requirements to judge:

{req_list}

You are given the unified diff (before -> after) for this change. This diff is the entire change; nothing outside it exists for this judgment.

{diff}

For each requirement, decide IMPLEMENTED or OMITTED using this test:

A requirement is IMPLEMENTED only if the diff shows the symbol is:
  1. DEFINED -- its `def`/`class` statement exists in the target file.
  2. WIRED -- something else in the diff actually calls, instantiates, or otherwise uses it. A symbol that only appears in its own definition, a docstring, a comment, or an import line is not wired.
  3. NOT A STUB -- its body does real work. A body consisting only of `pass`, `...`, a bare docstring, or `raise NotImplementedError` is a stub, not an implementation.

If any of the three fails, the requirement is OMITTED. Be strict: a symbol whose name merely appears somewhere in the diff (for example only inside a call site, with the definition missing) is OMITTED, not IMPLEMENTED -- that is exactly the case a naive text search gets wrong.

If a requirement is not visibly present anywhere in the diff, it is OMITTED: the diff is the whole change, and a requirement with no trace in it was not implemented as part of it.

Respond with a single JSON object and nothing else -- no markdown code fences, no prose before or after. The object must map each requirement string EXACTLY as written above (byte-for-byte) to either "IMPLEMENTED" or "OMITTED". Every requirement listed above must appear as a key, and no other keys may appear. Example shape:

{{"path/to/file.py::Foo": "IMPLEMENTED", "path/to/file.py::Bar": "OMITTED"}}"""


def build_prompt(reqs, diff) -> str:
    """Pure: (requirement list, unified diff) -> prompt string. No other
    parameters -- tests/test_leakage.py asserts this signature directly via
    inspect.signature. This is the entire surface a leak could ride in on;
    holding it to exactly these two arguments is what makes that surface
    auditable by inspection rather than by trust. Shared verbatim by all
    three judges -- see FAIRNESS in the module docstring.
    """
    req_list = "\n".join(f"- {r}" for r in sorted(reqs))
    return _PROMPT_TEMPLATE.format(req_list=req_list, diff=diff)


# --------------------------------------------------------------------------
# 3. cache -- content-addressed, committed to git, checked before every
#    call. Cache SUCCESSES only. A cached error/timeout would turn a
#    transient failure permanent, so _call_model's exceptions are never
#    written here.
#
#    Key now folds in model_slug + provider_slug + reasoning_effort +
#    base_url (not just model + prompt): a provider or reasoning-depth
#    switch is a different engine and must never silently return a stale
#    verdict from the old one.
#
#    VARIANCE SEED. scripts/run_judge.py --k 5 calls the SAME (spec, reqs,
#    before, after) five times with different seeds to measure run-to-run
#    output variance at temperature=0 (RELATED.md: prior work finds this is
#    nonzero even where a real sampling seed exists, and none of these three
#    APIs expose one -- see the module docstring). But build_prompt is a
#    PURE function of (reqs, diff): the prompt is byte-identical across all
#    5 seeds for one instance. Without a seed in the key, seeds 2-5 would
#    all cache-hit seed 1's response, and scripts/k5_variance.py would
#    report stdev=0.0000 -- a caching artifact reported as a finding, not a
#    measurement. `variance_seed`, threaded in via ctx["_variance_seed"],
#    fixes that by making each seed's call address a different cache slot.
#    It is None (omitted from the key entirely, not just set to None) on
#    every OTHER path -- the k=1 headline sweep, and anything that doesn't
#    set it -- so the k=1 cache key is BYTE-IDENTICAL to before this fix.
#    That matters twice: it means k=1's "second run makes zero API calls"
#    property is unaffected, and it means this fix cannot invalidate any
#    k=1 cache entries already committed to git.
# --------------------------------------------------------------------------

def _cache_key(cfg: JudgeConfig, prompt: str, variance_seed: int | None = None) -> str:
    params = {
        "base_url": cfg.base_url,
        "model_slug": cfg.model_slug,
        "provider_slug": cfg.provider_slug,
        "reasoning_effort": cfg.reasoning_effort,
    }
    if variance_seed is not None:
        params["variance_seed"] = variance_seed
    blob = cfg.model_slug + json.dumps(params, sort_keys=True) + prompt
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def _cache_get(key: str):
    p = _cache_path(key)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _cache_put(key: str, record: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = _cache_path(key) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, _cache_path(key))  # atomic: a crash mid-write can't leave
                                        # a corrupt cache entry that looks cached


# --------------------------------------------------------------------------
# 4. parsing -- strip markdown fences, then default any failure to
#    IMPLEMENTED. Never OMITTED: defaulting to OMITTED would inflate the
#    baseline's recall and bias the comparison in this project's favour.
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strict_parse(text: str, reqs: list[str]) -> tuple[dict[str, str], set[str]]:
    """Internal to this module -- _parse_response is the public 2-tuple
    contract every existing test asserts against, and stays that shape.
    This is the same strict-JSON logic, refactored to additionally return
    WHICH requirements got the IMPLEMENTED default (as opposed to a
    genuinely-parsed value), so a repair pass (_repair_from_prose, called
    from _parse_message) knows exactly which keys it is allowed to touch
    and never overwrites a value the JSON already confidently gave.
    """
    stripped = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return {r: "IMPLEMENTED" for r in reqs}, set(reqs)
    if not isinstance(obj, dict):
        return {r: "IMPLEMENTED" for r in reqs}, set(reqs)

    out: dict[str, str] = {}
    defaulted: set[str] = set()
    for r in reqs:
        v = obj.get(r)
        if v not in VALID:
            v = "IMPLEMENTED"
            defaulted.add(r)
        out[r] = v
    return out, defaulted


def _parse_response(text: str, reqs: list[str]) -> tuple[dict[str, str], bool]:
    """-> ({req: verdict}, parse_failed). On ANY problem -- non-JSON, wrong
    top-level type, a missing key, a malformed value -- every requirement in
    `reqs` defaults to IMPLEMENTED and parse_failed is True. Extra keys in
    the model's response (a requirement it hallucinated) are silently
    ignored; only the requirements we asked about are ever returned.
    """
    out, defaulted = _strict_parse(text, reqs)
    return out, bool(defaulted)


_VERDICT_TOKEN_RE = re.compile(r"\b(OMITTED|IMPLEMENTED)\b")


def _repair_from_prose(text: str, req: str, reqs: list[str]) -> str | None:
    """Recover a verdict for `req` from unstructured prose when the model
    reasoned to a stated conclusion but never emitted valid JSON for it --
    see ASSUMPTIONS.md's harness-limitation entry on this. Observed failure
    mode (found while investigating B4's parse-failure records): gpt-oss-120b
    sometimes writes out its answer as prose ("... So OMITTED.") or a
    "req -> VERDICT" summary line instead of the requested JSON object, and
    the strict parser in _strict_parse then defaults that requirement to
    IMPLEMENTED even though the model's own reasoning already reached
    OMITTED.

    Deliberately conservative, to preserve _parse_response's "never default
    TO OMITTED" bias (a false positive here would do exactly the harm that
    rule exists to prevent):
      - only looks at the LAST occurrence of `req`'s exact text -- a
        reasoning trace's closing summary is more likely to be its final
        answer than an earlier, tentative mention of the same symbol;
      - the search window after that occurrence stops at the next OTHER
        requirement's name (if any appears first) or 300 chars, whichever
        is nearer, so one requirement's window can never bleed into the
        next requirement's verdict;
      - only a bare, exact `OMITTED` or `IMPLEMENTED` token counts. No
        synonym matching ("not implemented", "missing", "looks done") --
        those are exactly the ambiguous cases this function must refuse to
        guess on.

    Returns None -- never a guess -- when no such token is found in that
    window; the caller's existing IMPLEMENTED default is then left alone.
    """
    idx = text.rfind(req)
    if idx == -1:
        return None
    window_end = len(text)
    search_from = idx + len(req)
    for other in reqs:
        if other == req:
            continue
        pos = text.find(other, search_from)
        if pos != -1:
            window_end = min(window_end, pos)
    window_end = min(window_end, search_from + 300)
    m = _VERDICT_TOKEN_RE.search(text[search_from:window_end])
    return m.group(1) if m else None


def _parse_message(message: dict, reqs: list[str]) -> tuple[dict[str, str], bool, str]:
    """Defensive extraction across whichever field the JSON actually landed
    in. Do NOT assume it is message["content"]: gpt-oss-120b returns chain-
    of-thought, and depending on the provider that reasoning has been
    observed to arrive in a sibling field (`reasoning` / `reasoning_content`)
    rather than -- or alongside -- `content`, sometimes leaving `content`
    empty or truncated. Try `content` first (the common case for the other
    two judges), then the reasoning fields, in order; use the first
    candidate that parses cleanly. If none parse, fall back to `content` (or
    the first non-empty candidate) so the parse-failure path still resolves
    to one stable, inspectable string rather than silently picking whichever
    field happened to be tried last.

    Returns (verdicts, parse_failed, field_used) -- field_used is recorded in
    every shard record so "the judge's answer was in `reasoning`, not
    `content`" is a visible, countable fact rather than a one-off surprise
    found by reading a raw log by hand.

    If every candidate field fails strict JSON parsing, a prose-repair pass
    (_repair_from_prose) runs over ALL candidate text before giving up --
    see that function's docstring. `parse_failed` still reports True in
    that case (the JSON genuinely was never emitted; that fact stays
    visible and countable in stats.parse_failures / the shard record)
    even when the repair pass recovers some or all of the requirements
    correctly. Only requirements the strict parse actually defaulted are
    ever touched -- a value the JSON confidently gave is never
    reconsidered.
    """
    candidates: list[tuple[str, str]] = []
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        candidates.append(("content", content))
    for alt in ("reasoning", "reasoning_content"):
        val = message.get(alt)
        if isinstance(val, str) and val.strip():
            candidates.append((alt, val))
    if not candidates:
        candidates.append(("content", content or ""))

    for field_name, text in candidates:
        verdicts, failed = _parse_response(text, reqs)
        if not failed:
            return verdicts, False, field_name

    field_name, text = candidates[0]
    verdicts, defaulted = _strict_parse(text, reqs)
    if defaulted:
        all_text = "\n".join(t for _, t in candidates)
        for r in defaulted:
            guess = _repair_from_prose(all_text, r, reqs)
            if guess is not None:
                verdicts[r] = guess
    return verdicts, True, field_name


# --------------------------------------------------------------------------
# 5. the API call -- lazy openai import (OpenAI-compatible client pointed at
#    OpenRouter or NVIDIA NIM depending on cfg.base_url), exponential
#    backoff, no caching of failures.
# --------------------------------------------------------------------------

def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _log_raw_response_once(cfg: JudgeConfig, raw: dict) -> None:
    """Writes ONE full raw response per judge to
    cache/_raw_response_<judge>.json the first time that judge is called in
    a process (overwritten thereafter, not appended -- this is a diagnostic
    sample, not a log). Exists so where reasoning content lands
    (message.content vs message.reasoning), where OpenRouter reports the
    served provider, and whether usage.cost is actually present can all be
    confirmed EMPIRICALLY against a real response before any number derived
    from _extract_provider / _extract_cost / _parse_message is trusted --
    see the docstrings on those functions for the specific uncertainty each
    one is guarding against.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"_raw_response_{_slugify(cfg.judge_id)}.json")
    if os.path.exists(path):
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(raw, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _extract_provider(raw: dict) -> str | None:
    """Where OpenRouter reports which provider actually served a routed
    request has not been consistent across documentation revisions seen
    while building this: sometimes a bare top-level `provider` string,
    sometimes nested under `openrouter_metadata.endpoints.available[]` with
    a `selected: true` flag. Check both. If this guesses wrong for the live
    API, `_log_raw_response_once` above exists precisely so that's
    discoverable from a real response and fixable here -- NOT a reason to
    weaken the "fail loudly on mismatch" check in _judge().
    """
    if not isinstance(raw, dict):
        return None
    top = raw.get("provider")
    if isinstance(top, str) and top:
        return top
    meta = raw.get("openrouter_metadata") or {}
    for ep in (meta.get("endpoints") or {}).get("available") or []:
        if ep.get("selected"):
            return ep.get("provider")
    return None


def _provider_matches(returned: str | None, pinned: str) -> bool:
    """Compare loosely (case/punctuation-insensitive substring match on the
    provider's base name, ignoring any '/quant' suffix in the pin) because
    OpenRouter has been observed to return a display name ("Cerebras")
    where the pin is a slug ("cerebras/fp16"). A returned value of None, or
    one that doesn't match at all, is a real mismatch -- not normalised
    away.
    """
    if not returned:
        return False
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    base = norm(pinned.split("/")[0])
    got = norm(returned)
    return bool(base) and (base in got or got in base)


def _extract_cost(raw: dict, cfg: JudgeConfig) -> tuple[float | None, str]:
    """-> (cost_usd, source). OpenRouter reports real spend directly in
    usage.cost (requested via extra_body {"usage": {"include": true}}) --
    trust that number rather than multiplying token counts by a price table
    that would drift out of sync. NIM's OpenAI-compatible endpoint does not
    return a cost field at all, so B6's cost is a labelled ESTIMATE from
    cfg.nim_price_per_1m -- source is always reported alongside the number
    so an estimate is never mistaken for a measurement downstream.
    """
    usage = raw.get("usage") or {}
    cost = usage.get("cost")
    if cost is not None:
        return float(cost), "reported"
    if cfg.nim_price_per_1m is not None:
        pin, pout = cfg.nim_price_per_1m
        est = (usage.get("prompt_tokens", 0) * pin +
               usage.get("completion_tokens", 0) * pout) / 1_000_000
        return est, "estimated"
    return None, "unavailable"


def _client_for(cfg: JudgeConfig):
    import openai  # lazy: keeps the core importable with the package absent

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{cfg.judge_id}: ${cfg.api_key_env} is not set in the "
            f"environment -- required for any call beyond --dry-run")
    return openai.OpenAI(base_url=cfg.base_url, api_key=api_key)


def _request_kwargs(cfg: JudgeConfig, prompt: str) -> dict:
    kwargs: dict = {
        "model": cfg.model_slug,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2048,
    }
    extra_body: dict = {}
    if cfg.provider_slug is not None:
        # PIN THE PROVIDER (see module docstring). allow_fallbacks=False
        # means the call FAILS rather than silently landing on a different,
        # unpinned backend if ours is unavailable -- a hard failure here is
        # more honest than an invisible substitution.
        extra_body["provider"] = {"order": [cfg.provider_slug], "allow_fallbacks": False}
        extra_body["usage"] = {"include": True}  # ask for real usage.cost
    if cfg.reasoning_effort is not None:
        extra_body["reasoning"] = {"effort": cfg.reasoning_effort}
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def _call_model(cfg: JudgeConfig, prompt: str) -> tuple[dict, dict]:
    """-> (message_dict, meta_dict). meta_dict has usage / provider_returned
    / cost / cost_source. Raises after MAX_RETRIES exhausted. Never called
    on a cache hit.

    Uses `.with_raw_response.create(...)` and parses the JSON body by hand
    (json.loads(resp.text)) rather than the SDK's typed response object --
    OpenRouter attaches fields (`provider`, `openrouter_metadata`,
    `usage.cost`) that are not part of the OpenAI response schema the SDK's
    pydantic models are built from, and a typed parse would silently drop
    exactly the fields this module needs to verify.
    """
    import openai

    client = _client_for(cfg)
    kwargs = _request_kwargs(cfg, prompt)
    stats = _stats_for(cfg.judge_id)

    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            raw_resp = client.chat.completions.with_raw_response.create(**kwargs)
            data = json.loads(raw_resp.text)
            _log_raw_response_once(cfg, data)

            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            usage = data.get("usage") or {}
            provider_returned = _extract_provider(data)
            cost, cost_source = _extract_cost(data, cfg)

            meta = {
                "usage": usage,
                "provider_returned": provider_returned,
                "cost": cost,
                "cost_source": cost_source,
            }
            return message, meta
        except openai.RateLimitError as e:
            last_exc = e
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                last_exc = e
            else:
                raise  # non-retryable: a 4xx other than 429 is a bug, not a blip
        except openai.APIConnectionError as e:
            last_exc = e
        if attempt < MAX_RETRIES:
            stats.retries += 1
            time.sleep(delay)
            delay *= 2
    raise last_exc  # noqa: B904 -- re-raising the last retryable failure


# --------------------------------------------------------------------------
# 6. the detector core, shared by all three judges, plus the three thin
#    per-judge wrappers DETECTORS registers.
# --------------------------------------------------------------------------

def _judge(cfg: JudgeConfig, spec, reqs, before, after, ctx):
    diff = unified_diff(before, after)
    diff, truncated = _truncate(diff, DIFF_TOKEN_CEILING)
    prompt = build_prompt(reqs, diff)

    # See the VARIANCE SEED comment above _cache_key: absent (None) on every
    # path except scripts/run_judge.py --k 5, which is the only caller that
    # puts anything under this ctx key. ctx is caller-supplied and, per the
    # ONE INVARIANT, the caller never has the gold patch or mutation label
    # to put there either -- this is a run index, not new information about
    # the instance.
    variance_seed = ctx.get("_variance_seed") if isinstance(ctx, dict) else None

    key = _cache_key(cfg, prompt, variance_seed)
    cached = _cache_get(key)
    stats = _stats_for(cfg.judge_id)

    if cached is not None:
        stats.cache_hits += 1
        message = cached["message"]
        meta = {
            "usage": None,  # no new spend to price on a cache hit
            "provider_returned": cached.get("provider_returned"),
            "cost": None,
            "cost_source": "cache_hit",
        }
    else:
        message, meta = _call_model(cfg, prompt)
        stats.calls += 1
        _cache_put(key, {
            "judge_id": cfg.judge_id, "model_slug": cfg.model_slug,
            "provider_slug": cfg.provider_slug,
            "reasoning_effort": cfg.reasoning_effort, "base_url": cfg.base_url,
            "variance_seed": variance_seed,
            "prompt": prompt, "message": message,
            "provider_returned": meta["provider_returned"],
            "usage": meta["usage"], "cost": meta["cost"],
            "cost_source": meta["cost_source"],
        })

    # PIN VERIFICATION -- fail loudly, not a warning. See module docstring.
    if cfg.provider_slug is not None:
        returned = meta.get("provider_returned")
        if not _provider_matches(returned, cfg.provider_slug):
            raise RuntimeError(
                f"{cfg.judge_id}: pinned provider {cfg.provider_slug!r} but "
                f"OpenRouter reports {returned!r} actually served this "
                f"request. Refusing to silently accept a possibly "
                f"differently-quantized backend. If this looks like "
                f"_extract_provider() misreading an unfamiliar response "
                f"shape, check cache/_raw_response_"
                f"{_slugify(cfg.judge_id)}.json and fix that function -- "
                f"do not relax this check.")

    verdicts, parse_failed, used_field = _parse_message(message, reqs)
    stats.reqs_scored += len(reqs)
    if parse_failed:
        stats.parse_failures += 1
    if truncated:
        stats.truncated += 1

    record = ctx.get("_judge_record") if isinstance(ctx, dict) else None
    if record is not None:
        record.update({
            "judge_id": cfg.judge_id,
            "model_slug": cfg.model_slug,
            "provider_pinned": cfg.provider_slug,
            "provider_returned": meta.get("provider_returned"),
            "reasoning_effort": cfg.reasoning_effort,
            "truncated": truncated,
            "parse_failed": parse_failed,
            "parsed_from_field": used_field,
            "cache_hit": cached is not None,
            "usage": meta.get("usage"),
            "cost_usd": meta.get("cost"),
            "cost_source": meta.get("cost_source"),
        })

    return verdicts


def d_gpt_oss_120b(spec, reqs, before, after, ctx):
    """B4: gpt-oss-120b via OpenRouter, pinned to the Cerebras fp16 endpoint
    (see JUDGES). ONE call per variant (median 2 requirements per variant).
    Blind to everything listed in the module docstring, plus: the FP16 pin
    means this number does not generalise to the (cheaper, more common)
    fp4/fp8 OpenRouter endpoints for the same model -- see the k=5 run for
    whether that matters more than run-to-run sampling noise does.
    """
    return _judge(JUDGES["B4 LLM judge (gpt-oss-120b)"], spec, reqs, before, after, ctx)


def d_mid_tier(spec, reqs, before, after, ctx):
    """B5: mistralai/mistral-medium-3 via OpenRouter (first-party, single
    provider). See JUDGES for the model-choice justification. Blind to
    everything listed in the module docstring.
    """
    return _judge(JUDGES["B5 LLM judge (mid-tier)"], spec, reqs, before, after, ctx)


def d_nim(spec, reqs, before, after, ctx):
    """B6: nvidia/nemotron-3-super-120b-a12b via OpenRouter, pinned to the
    DeepInfra bf16 endpoint (see JUDGES for the full 2026-08-13 re-pointing
    rationale -- this used to be meta/llama-3.3-70b-instruct via direct
    NVIDIA NIM, before OPENROUTER_API_KEY became the only key available).
    Blind to everything listed in the module docstring, plus: unlike the
    retired NIM path, this number CAN be confounded by an OpenRouter-
    specific routing or reporting bug, the same as B4/B5 -- B6 is no longer
    an independent-broker check on that. And the model itself is now MoE
    (12B active / 120B total), not dense, since no dense NVIDIA Nemotron
    model had a live OpenRouter endpoint when this was picked.
    """
    return _judge(JUDGES["B6 LLM judge (NVIDIA NIM)"], spec, reqs, before, after, ctx)


DETECTORS = {
    "B4 LLM judge (gpt-oss-120b)": d_gpt_oss_120b,
    "B5 LLM judge (mid-tier)": d_mid_tier,
    "B6 LLM judge (NVIDIA NIM)": d_nim,
}
