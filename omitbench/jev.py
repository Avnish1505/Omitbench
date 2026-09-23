"""
B7: TypeSafe Jev as an omission judge. Same contract as every detector:

    d_xxx(spec, reqs, before, after, ctx) -> {req: "IMPLEMENTED"|"OMITTED"}

WHY THIS EXISTS. B5 (a mid-tier LLM judge) beats P1 overall (paired MCC
-0.202, CI excluding zero). The two practical objections to deploying B5 as
a gate are that it is slow and that it hands back a bare label with no
probability, so there is nothing to threshold on. Jev is a "System One"
model from TypeSafe AI (public since 2026-09-15) that answers typed yes/no
questions ("Nouls") with a probability instead of generating text, at
roughly 100 ms per call and $0.042 per 1M input tokens. TypeSafe markets
those probabilities as calibrated; its own docs contain no calibration
numbers. That claim is testable here because every requirement in this
corpus has a ground-truth label, so B7 gets TWO evaluations:

  1. the usual one: thresholded verdicts through scripts/analyze.py, paired
     against P1 and B5 like every other detector;
  2. the new one: scripts/analyze_calibration.py reads the per-requirement
     probability this module records and reports Brier score, ECE and a
     reliability table, overall and per mutation class.

WHAT B7 IS BLIND TO. Everything B5 is blind to, by construction: it sees the
SAME unified diff (judges.unified_diff) under the SAME truncation ceiling
(judges.DIFF_TOKEN_CEILING, via judges._truncate), so the B5/B7 comparison is
a comparison of models, not inputs. A hollow STUB body outside the diff's
context window, or a call site the diff does not include, is invisible to
both. Additionally blind to: anything that needs counting or multi-hop
reasoning, which TypeSafe's own "Jev 1.13 jaggedness" page lists as weak
spots. The single-question variant asks one compound question per
requirement, which TypeSafe's docs advise against; that is deliberate (it
mirrors B5's three-part test exactly) and is the reason B7-split exists.

TWO VARIANTS, BOTH PRE-REGISTERED (ASSUMPTIONS.md section on B7, written
before any real call was made). Adding detectors until one wins is the
anti-pattern CLAUDE.md names; the defence is fixing the set in advance:
  B7 Jev (single)  one Noul per requirement: "is it defined AND wired AND
                   not a stub?" P(implemented) is the Noul value.
  B7 Jev (split)   three Nouls per requirement (defined / wired / real
                   body). P(implemented) = min of the three: the Frechet
                   upper bound on "all three hold", chosen over the product
                   (which assumes independence the three questions do not
                   have) because it errs toward IMPLEMENTED, same direction
                   as judges.py's "never default TO OMITTED" rule.
Verdict for both: OMITTED iff P(implemented) < 0.5. The threshold is fixed
here and is not tuned on this corpus.

LEAK SURFACE. The request payload is built by two pure functions,
build_jev_request(reqs, diff) and build_jev_request_split(reqs, diff), which
take nothing else. tests/test_jev.py locks both signatures and plants the
gold canary through them, mirroring test_leakage.py's judge tests.

MODEL PINNING. Requests pin "jev-1.13.0", not the "jev-latest" alias
(TypeSafe: pin the version ID if you have tuned thresholds against it). The
response's `model` field is checked and a mismatch RAISES, same logic as
judges.py's provider pin: a silently swapped model is the confound pinning
exists to prevent.

ZERO DEPENDENCIES. Plain urllib, no SDK, so the core's stdlib-only claim
holds and there is no SDK retry layer hiding failures from this module's
own counters. Set TYPESAFE_API_KEY for any real call.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from omitbench import judges as J

API_URL = os.environ.get("TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone")
API_KEY_ENV = "TYPESAFE_API_KEY"
PINNED_MODEL = "jev-1.13.0"
PRICE_PER_1M_INPUT = 0.042      # USD; output tokens are free (TypeSafe pricing)
THRESHOLD = 0.5                 # pre-registered, never tuned on this corpus
MAX_RETRIES = 4
TIMEOUT_S = 60
CACHE_DIR = os.environ.get("OMITBENCH_JEV_CACHE_DIR", J.CACHE_DIR)   # override for mock runs
RETRYABLE = {429, 500, 502, 503, 504, 529}
# urllib's default "Python-urllib/3.x" User-Agent is blocked by Cloudflare in
# front of the API (HTTP 403, "error code: 1010") before the key is checked.
# Transport fix, found in the first smoke call; ASSUMPTIONS.md section 13.
USER_AGENT = "omitbench/0.3 (+https://github.com/Avnish1505/Omitbench)"


@dataclass(frozen=True)
class JevConfig:
    judge_id: str
    variant: str          # "single" | "split"


CONFIGS = {
    "B7 Jev (single)": JevConfig("B7 Jev (single)", "single"),
    "B7 Jev (split)": JevConfig("B7 Jev (split)", "split"),
}


# --------------------------------------------------------------------------
# 1. request construction -- the leak surface. PURE: (reqs, diff) only.
# --------------------------------------------------------------------------

_NOT_PRESENT = ("If the symbol does not visibly appear anywhere in the diff, "
                "the answer is no: the diff is the entire change.")

_SINGLE_INSTR = (
    "Requirement: `{req}` (a function or class, written as "
    "path/to/file.py::SymbolName). Is this requirement fully implemented by "
    "the code change in the state? Fully implemented means all three hold: "
    "(1) DEFINED: its def/class statement exists in that file; "
    "(2) WIRED: something else in the diff calls, instantiates or otherwise "
    "uses it (its own definition, a docstring, a comment or an import line "
    "do not count); (3) NOT A STUB: its body does real work, not only "
    "`pass`, `...`, a bare docstring or `raise NotImplementedError`. "
    + _NOT_PRESENT)

_SINGLE_CRITERIA = {
    "true": "The symbol is defined in the named file, used elsewhere in the "
            "diff, and has a real body.",
    "false": "The definition is missing, or nothing else in the diff uses it, "
             "or its body is a stub, or it does not appear in the diff.",
}

_SPLIT = {
    "defined": (
        "Requirement: `{req}`. Does the diff show a def or class statement "
        "for this exact symbol name inside that exact file? A call site, "
        "import, comment or docstring mentioning the name is not a "
        "definition. " + _NOT_PRESENT,
        {"true": "A def/class statement for this symbol exists in that file.",
         "false": "No def/class statement for this symbol in that file."}),
    "wired": (
        "Requirement: `{req}`. Does something in the diff other than the "
        "symbol's own definition call, instantiate or otherwise use it? Its "
        "own def line, docstrings, comments and import lines do not count. "
        + _NOT_PRESENT,
        {"true": "Other code in the diff calls or uses this symbol.",
         "false": "Nothing else in the diff uses this symbol."}),
    "real_body": (
        "Requirement: `{req}`. Does this symbol's body do real work? A body "
        "that is only `pass`, `...`, a bare docstring or "
        "`raise NotImplementedError` is a stub, so the answer is no. "
        + _NOT_PRESENT,
        {"true": "The body contains real logic.",
         "false": "The body is a stub, or the symbol is not in the diff."}),
}


def _qkey(i: int, part: str | None = None) -> str:
    return f"r{i}" if part is None else f"r{i}_{part}"


def build_jev_request(reqs, diff) -> dict:
    """Pure: (requirements, unified diff) -> request payload for B7 (single).
    One Noul per requirement, keyed r0..rN over sorted(reqs). No other
    parameters; tests/test_jev.py asserts the signature.
    """
    qs = {}
    for i, r in enumerate(sorted(reqs)):
        qs[_qkey(i)] = {"type": "noul",
                        "instructions": _SINGLE_INSTR.format(req=r),
                        "criteria": dict(_SINGLE_CRITERIA)}
    return {"model": PINNED_MODEL, "state": diff, "questions": qs}


def build_jev_request_split(reqs, diff) -> dict:
    """Pure: (requirements, unified diff) -> request payload for B7 (split).
    Three Nouls per requirement: r{i}_defined, r{i}_wired, r{i}_real_body.
    """
    qs = {}
    for i, r in enumerate(sorted(reqs)):
        for part, (instr, crit) in _SPLIT.items():
            qs[_qkey(i, part)] = {"type": "noul",
                                  "instructions": instr.format(req=r),
                                  "criteria": dict(crit)}
    return {"model": PINNED_MODEL, "state": diff, "questions": qs}


# --------------------------------------------------------------------------
# 2. answers -> probabilities -> verdicts
# --------------------------------------------------------------------------

def _noul(answers: dict, key: str) -> float | None:
    a = answers.get(key)
    if not isinstance(a, dict):
        return None
    v = a.get("noul")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if 0.0 <= v <= 1.0 else None


def p_implemented(variant: str, reqs, answers: dict) -> dict[str, float | None]:
    """{req: P(implemented) or None when the answer is missing/malformed}."""
    out: dict[str, float | None] = {}
    for i, r in enumerate(sorted(reqs)):
        if variant == "single":
            out[r] = _noul(answers, _qkey(i))
        else:
            parts = [_noul(answers, _qkey(i, p)) for p in _SPLIT]
            out[r] = None if any(p is None for p in parts) else min(parts)
    return out


def verdicts_from(probs: dict[str, float | None]) -> tuple[dict[str, str], bool]:
    """Missing probability -> IMPLEMENTED, never OMITTED (judges.py's rule:
    defaulting to OMITTED would inflate this baseline's recall)."""
    out, failed = {}, False
    for r, p in probs.items():
        if p is None:
            out[r], failed = "IMPLEMENTED", True
        else:
            out[r] = "OMITTED" if p < THRESHOLD else "IMPLEMENTED"
    return out, failed


# --------------------------------------------------------------------------
# 3. cache (successes only) and the HTTP call
# --------------------------------------------------------------------------

def _cache_key(payload: dict) -> str:
    blob = "jev|" + API_URL + "|" + json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_get(key: str):
    p = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _cache_put(key: str, record: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, f"{key}.json")
    with open(p + ".tmp", "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(p + ".tmp", p)


class _Stats:
    def __init__(self):
        self.calls = self.cache_hits = self.parse_failures = 0
        self.truncated = 0
        self.input_tokens = 0


STATS: dict[str, _Stats] = {}


def _stats_for(judge_id: str) -> _Stats:
    return STATS.setdefault(judge_id, _Stats())


def _post(payload: dict) -> dict:
    """Swappable in tests (monkeypatch jev._post). Raises on non-retryable
    HTTP errors; retries 429/529/5xx and connection errors with backoff."""
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"${API_KEY_ENV} is not set -- required for any "
                           f"call beyond --dry-run")
    body = json.dumps(payload).encode("utf-8")
    delay, last = 1.0, None
    for attempt in range(MAX_RETRIES + 1):
        req = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE:
                detail = e.read().decode("utf-8", "replace")[:500]
                raise RuntimeError(f"TypeSafe HTTP {e.code}: {detail}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        if attempt < MAX_RETRIES:
            _RETRIES[0] += 1
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"TypeSafe: retries exhausted: {last!r}")


_RETRIES = [0]   # process-wide retry counter; the runner prints it


def _cost(usage: dict | None) -> float | None:
    if not isinstance(usage, dict) or usage.get("input_tokens") is None:
        return None
    return usage["input_tokens"] * PRICE_PER_1M_INPUT / 1_000_000


# --------------------------------------------------------------------------
# 4. the detector core + registered wrappers
# --------------------------------------------------------------------------

def _judge(cfg: JevConfig, spec, reqs, before, after, ctx):
    diff = J.unified_diff(before, after)
    diff, truncated = J._truncate(diff, J.DIFF_TOKEN_CEILING)
    build = build_jev_request if cfg.variant == "single" else build_jev_request_split
    payload = build(reqs, diff)

    key = _cache_key(payload)
    cached = _cache_get(key)
    stats = _stats_for(cfg.judge_id)
    if cached is not None:
        stats.cache_hits += 1
        resp = cached["response"]
        cost, cost_source = None, "cache_hit"
    else:
        resp = _post(payload)
        stats.calls += 1
        cost = _cost(resp.get("usage"))
        cost_source = "estimated_from_usage" if cost is not None else "unavailable"
        if isinstance(resp.get("usage"), dict):
            stats.input_tokens += int(resp["usage"].get("input_tokens") or 0)
        # Cache only after the model pin passes: a response from the wrong
        # model must not become a permanent cache entry.
        if resp.get("model") == PINNED_MODEL:
            _cache_put(key, {"judge_id": cfg.judge_id, "variant": cfg.variant,
                             "payload": payload, "response": resp})

    model_returned = resp.get("model")
    if model_returned != PINNED_MODEL:
        raise RuntimeError(
            f"{cfg.judge_id}: pinned {PINNED_MODEL!r} but the response says "
            f"{model_returned!r}. Refusing to mix model versions in one "
            f"sweep -- re-pin deliberately, do not relax this check.")

    probs = p_implemented(cfg.variant, reqs, resp.get("answers") or {})
    verdicts, parse_failed = verdicts_from(probs)
    if parse_failed:
        stats.parse_failures += 1
    if truncated:
        stats.truncated += 1

    record = ctx.get("_judge_record") if isinstance(ctx, dict) else None
    if record is not None:
        record.update({
            "judge_id": cfg.judge_id, "model_slug": PINNED_MODEL,
            "model_returned": model_returned, "variant": cfg.variant,
            "p_implemented": probs, "threshold": THRESHOLD,
            "truncated": truncated, "parse_failed": parse_failed,
            "cache_hit": cached is not None, "usage": resp.get("usage"),
            "cost_usd": cost, "cost_source": cost_source,
        })
    return verdicts


def d_jev_single(spec, reqs, before, after, ctx):
    """B7 (single): one compound Noul per requirement. See module docstring
    for what it is blind to."""
    return _judge(CONFIGS["B7 Jev (single)"], spec, reqs, before, after, ctx)


def d_jev_split(spec, reqs, before, after, ctx):
    """B7 (split): defined / wired / real-body Nouls, min-combined."""
    return _judge(CONFIGS["B7 Jev (split)"], spec, reqs, before, after, ctx)


DETECTORS = {
    "B7 Jev (single)": d_jev_single,
    "B7 Jev (split)": d_jev_split,
}
