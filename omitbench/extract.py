"""
T5 (TASKS.md): LLM-based requirement extractor.

Every requirement fed to a detector today comes from
mutate.py::new_symbols(patch_before, patch_after) -- the reference solution.
That is a perfect, structured "extractor" that assumes a real agent's plan
decomposes exactly onto the symbols the gold patch happens to define. Every
number in results/pilot.json is therefore an UPPER BOUND (ASSUMPTIONS.md #2).

This module measures what happens when requirements instead come from a
genuinely imperfect source: an LLM reading ONLY the text a real agent would
actually start from -- the commit message / issue text (`spec` in
corpus.py's Instance, `task_text` here) -- and NEVER the diff. It predicts
which new symbols (functions/classes) it expects the eventual patch to
define, before having seen that patch.

BLIND TO, by construction: patch_before, patch_after, before, after, any
diff, the oracle requirement list, the mutation class, the gold label.
Its only real input is task_text. tests/test_leakage.py enforces this the
same way it enforces the detector/judge contract: a signature lock on
extract_one, a signature lock on the pure prompt-builder, and a canary test.

SCHEMA, FROZEN BEFORE ANY PROMPT ITERATION (do not change after seeing Step
3/4 results -- see TASKS.md T5 and ASSUMPTIONS.md's entry on this module):

    [{"symbol": "<name>", "path": "<repo-relative-path-or-null>"}, ...]

`path: null` is an allowed, honest abstention -- the model is never forced
to fabricate a file path it has no textual basis for guessing. Downstream
scoring (scripts/score_extraction.py) computes two SEPARATE match types --
symbol-only (ignores path) and path-qualified (both fields must match an
oracle pair exactly) -- and never folds them into one number. See that
module and ASSUMPTIONS.md for why, and for the known limitation of
symbol-only matching.

BACKEND CHOICE. Reuses judges.py's JudgeConfig / OpenRouter client / retry /
cost-extraction plumbing verbatim ("same client setup as the B4/B5 judges",
per TASKS.md T5) rather than reimplementing it. Pinned to the SAME model and
provider as B5 (mistralai/mistral-medium-3 via OpenRouter, provider
"mistral", no reasoning knob): B5 is the only one of the three judges with
zero observed JSON parse failures on this corpus (ASSUMPTIONS.md #8) and the
cheapest, most compact output of the three (avg 51.6 completion tokens/call
against B4's 636.7 -- see cache/*.json), which is exactly the profile wanted
for a short structured-JSON extraction task. This is reuse of an
already-vetted, already-cheap backend, not a new untested pick.

CACHE. Content-addressed, same content-addressing scheme as judges.py
(sha256 of model/provider/prompt), but filed under a distinct
`extract_<hash>.json` name in the same `cache/` directory so it is
trivially greppable apart from judge cache entries and committing it keeps
the same "re-running costs $0" promise CLAUDE.md makes for judges.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

from omitbench.judges import (
    JudgeConfig,
    _call_model,
    _extract_provider,
    _provider_matches,
)

CACHE_DIR = "cache"

# Reuses B5's exact model/provider pin -- see module docstring for why.
EXTRACT_CONFIG = JudgeConfig(
    judge_id="T5 extractor (mistral-medium-3)",
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY",
    model_slug="mistralai/mistral-medium-3",
    provider_slug="mistral",
    reasoning_effort=None,
    dry_run_price_per_1m=(0.40, 2.00),
)

MAX_ITEMS = 8  # matches MAX_REQS in experiment.py -- no point extracting more
               # than the oracle itself ever exposes as requirements


class _Stats:
    def __init__(self):
        self.calls = 0
        self.cache_hits = 0
        self.parse_failures = 0
        self.items_extracted = 0

    def reset(self):
        self.__init__()


STATS = _Stats()


# --------------------------------------------------------------------------
# 1. prompt construction -- the only leak surface besides extract_one's
#    signature. PURE. task_text only. Locked by
#    tests/test_leakage.py::test_extract_prompt_is_pure_and_signature_locked.
# --------------------------------------------------------------------------

_EXTRACT_PROMPT_TEMPLATE = """You are reading an issue description or commit message for a software change, written BEFORE the change was implemented. You have NOT seen the code change itself.

Issue / commit text:
{task_text}

Based ONLY on this text, predict which new functions or classes (symbols) the implementation will need to define. For each one, give:

  - "symbol": the exact name you expect the `def` or `class` statement to use (case-sensitive). Use your best guess at the identifier a Python developer would actually write, not a paraphrase of the text.
  - "path": the exact repo-relative file path you expect it to be defined in (for example "src/pkg/core.py"), or `null` if the text gives you no real basis for guessing a specific file. Do not fabricate a plausible-looking path -- `null` is the honest and expected answer when the text does not name or clearly imply a file.

Only name symbols you have genuine textual evidence for -- a name mentioned, described, or strongly implied by the text (for example, "add a `retry_with_backoff` helper" or "the Foo class needs a validate method"). Do not invent symbols the text gives no evidence for. If the text gives no evidence for any specific new symbol, return an empty array.

List at most {max_items} symbols, most-confident first.

Respond with a single JSON array and nothing else -- no markdown code fences, no prose before or after. Example shape:

[{{"symbol": "retry_with_backoff", "path": "src/pkg/client.py"}}, {{"symbol": "validate", "path": null}}]"""


def build_extract_prompt(task_text: str) -> str:
    """Pure: task_text -> prompt string. No other parameters --
    tests/test_leakage.py asserts this signature directly via
    inspect.signature. This is the entire surface a leak could ride in on.
    """
    return _EXTRACT_PROMPT_TEMPLATE.format(task_text=task_text, max_items=MAX_ITEMS)


# --------------------------------------------------------------------------
# 2. cache -- content-addressed, committed to git, checked before every
#    call. Cache SUCCESSES only, same rule as judges.py: a cached
#    error/timeout would turn a transient failure permanent.
# --------------------------------------------------------------------------

def _cache_key(cfg: JudgeConfig, prompt: str) -> str:
    params = {
        "purpose": "extract",  # disambiguates from judges.py's key namespace,
                                # even though a collision is already
                                # astronomically unlikely on sha256 content
        "base_url": cfg.base_url,
        "model_slug": cfg.model_slug,
        "provider_slug": cfg.provider_slug,
        "reasoning_effort": cfg.reasoning_effort,
    }
    blob = cfg.model_slug + json.dumps(params, sort_keys=True) + prompt
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"extract_{key}.json")


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
    os.replace(tmp, _cache_path(key))


# --------------------------------------------------------------------------
# 3. parsing -- strict JSON array of {"symbol": str, "path": str|null}.
#    On ANY problem, return an EMPTY list, never a guess. This is the
#    extraction analogue of judges.py's "never default to OMITTED": the
#    conservative failure direction here is to abstain (extract nothing)
#    rather than fabricate a plausible-looking symbol/path pair, which
#    would inflate neither precision nor recall honestly.
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_extraction(text: str) -> tuple[list[dict], bool]:
    """-> (items, parse_failed). items are {"symbol": str, "path": str|None},
    deduplicated by (symbol, path) in first-seen order, capped at MAX_ITEMS.
    Malformed individual entries (missing/non-string symbol, non-string
    non-null path) are dropped rather than defaulted -- this counts toward
    parse_failed only if it empties an otherwise-nonempty response, since a
    single malformed entry among several valid ones is not a total parse
    failure. Symbol/path strings are kept EXACTLY as returned -- no
    normalization, no case-folding, matching the oracle's own
    (path, symbol) pairs, which are also verbatim (see mutate.py::new_symbols
    and experiment.py's `f"{path}::{symbol}"` construction).
    """
    stripped = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return [], True
    if not isinstance(obj, list):
        return [], True

    out: list[dict] = []
    seen: set[tuple[str, str | None]] = set()
    dropped_any = False
    for entry in obj:
        if not isinstance(entry, dict):
            dropped_any = True
            continue
        sym = entry.get("symbol")
        path = entry.get("path", None)
        if not isinstance(sym, str) or not sym.strip():
            dropped_any = True
            continue
        if path is not None and not isinstance(path, str):
            dropped_any = True
            continue
        key = (sym, path)
        if key in seen:
            continue
        seen.add(key)
        out.append({"symbol": sym, "path": path})
        if len(out) >= MAX_ITEMS:
            break

    parse_failed = dropped_any and not out
    return out, parse_failed


def _message_text(message: dict) -> str:
    """Same defensive field probing as judges.py::_parse_message, minus the
    prose-repair pass -- extraction output is a short JSON array, not a
    multi-requirement verdict object, so there is no per-requirement window
    to repair; a response that isn't valid JSON is simply an abstention
    (see _parse_extraction's docstring).
    """
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    for alt in ("reasoning", "reasoning_content"):
        val = message.get(alt)
        if isinstance(val, str) and val.strip():
            return val
    return content or ""


# --------------------------------------------------------------------------
# 4. public entry point. Signature locked by
#    tests/test_leakage.py::test_extractor_never_accepts_diff_content --
#    task_text is the only real input; ctx exists only as an optional
#    caller-supplied record sink (mirrors judges.py's ctx["_judge_record"])
#    and this module never reads it for anything else.
# --------------------------------------------------------------------------

def extract_one(task_text: str, ctx: dict | None = None) -> list[dict]:
    """-> [{"symbol": str, "path": str|None}, ...]. ONE call per synthetic
    instance (extraction does not depend on the mutation variant -- the
    issue text is identical across CLEAN/ABSENT/UNWIRED/STUB for a given
    instance, so this is called once per iid, not once per variant, unlike
    the detectors/judges).
    """
    prompt = build_extract_prompt(task_text)
    key = _cache_key(EXTRACT_CONFIG, prompt)
    cached = _cache_get(key)

    if cached is not None:
        STATS.cache_hits += 1
        message = cached["message"]
        meta = {"provider_returned": cached.get("provider_returned"),
                 "cost": None, "cost_source": "cache_hit"}
    else:
        message, meta = _call_model(EXTRACT_CONFIG, prompt)
        STATS.calls += 1
        _cache_put(key, {
            "judge_id": EXTRACT_CONFIG.judge_id,
            "model_slug": EXTRACT_CONFIG.model_slug,
            "provider_slug": EXTRACT_CONFIG.provider_slug,
            "reasoning_effort": EXTRACT_CONFIG.reasoning_effort,
            "base_url": EXTRACT_CONFIG.base_url,
            "prompt": prompt, "message": message,
            "provider_returned": meta["provider_returned"],
            "usage": meta["usage"], "cost": meta["cost"],
            "cost_source": meta["cost_source"],
        })

    # PIN VERIFICATION -- fail loudly, not a warning. Same rule as judges.py.
    returned = meta.get("provider_returned")
    if not _provider_matches(returned, EXTRACT_CONFIG.provider_slug):
        raise RuntimeError(
            f"{EXTRACT_CONFIG.judge_id}: pinned provider "
            f"{EXTRACT_CONFIG.provider_slug!r} but OpenRouter reports "
            f"{returned!r} actually served this request. Refusing to "
            f"silently accept a possibly differently-quantized backend.")

    items, parse_failed = _parse_extraction(_message_text(message))
    if parse_failed:
        STATS.parse_failures += 1
    STATS.items_extracted += len(items)

    record = ctx.get("_extract_record") if isinstance(ctx, dict) else None
    if record is not None:
        record.update({
            "model_slug": EXTRACT_CONFIG.model_slug,
            "provider_pinned": EXTRACT_CONFIG.provider_slug,
            "provider_returned": returned,
            "parse_failed": parse_failed,
            "cache_hit": cached is not None,
            "usage": meta.get("usage"),
            "cost_usd": meta.get("cost"),
            "cost_source": meta.get("cost_source"),
            "items": items,
        })

    return items
