"""
Unit tests for the LLM-judge baselines (omitbench/judges.py): B4
(gpt-oss-120b via OpenRouter), B5 (mistral-medium-3 via OpenRouter), B6
(nemotron-3-super-120b-a12b via OpenRouter, pinned to DeepInfra bf16 --
re-pointed 2026-08-13 from a direct NVIDIA NIM call now that only
OPENROUTER_API_KEY is available; see JUDGES in judges.py for why).

None of these make a real network call -- _call_model is monkeypatched
wherever a full d_xxx() invocation is needed. Leakage-specific guards
(prompt purity, canary) live in tests/test_leakage.py, not here.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from omitbench import judges as J  # noqa: E402


def repo(**files):
    return {p: src.strip("\n").split("\n") for p, src in files.items()}


BEFORE = repo(**{"app.py": """
def run():
    return 1
"""})

AFTER = repo(**{"app.py": """
def helper(x):
    total = x * 2
    return total

def run():
    return helper(3)
"""})

B4, B5, B6 = (J.JUDGES["B4 LLM judge (gpt-oss-120b)"],
              J.JUDGES["B5 LLM judge (mid-tier)"],
              J.JUDGES["B6 LLM judge (NVIDIA NIM)"])


# --------------------------------------------------------------------------
# registry -- three judges, three distinct backends
# --------------------------------------------------------------------------

def test_three_judges_registered():
    assert set(J.DETECTORS) == {
        "B4 LLM judge (gpt-oss-120b)",
        "B5 LLM judge (mid-tier)",
        "B6 LLM judge (NVIDIA NIM)",
    }
    assert set(J.JUDGES) == set(J.DETECTORS)


def test_openrouter_judges_are_pinned_to_a_provider():
    """Provider pinning is not optional for anything routed through
    OpenRouter -- see the module docstring. All three judges as of
    2026-08-13, B6 included since its NIM re-point."""
    assert B4.provider_slug is not None
    assert B5.provider_slug is not None
    assert B6.provider_slug is not None


def test_b6_now_shares_openrouter_transport_with_b4_b5():
    """B6 was re-pointed 2026-08-13 from direct NVIDIA NIM to OpenRouter
    (we only hold OPENROUTER_API_KEY, not a direct NVIDIA key) -- it must
    now share base_url and api_key_env with B4/B5, not be a separate
    broker."""
    assert B6.base_url == B4.base_url == B5.base_url
    assert "openrouter" in B6.base_url
    assert B6.api_key_env == "OPENROUTER_API_KEY"
    assert B6.model_slug.startswith("nvidia/")  # still an NVIDIA model


def test_judges_share_build_prompt_not_a_per_judge_copy():
    """All three call the same pure build_prompt -- there is exactly one
    prompt-construction path in this module."""
    diff = J.unified_diff(BEFORE, AFTER)
    p = J.build_prompt(["app.py::helper"], diff)
    assert p  # smoke: just confirms the shared function is reachable/pure


# --------------------------------------------------------------------------
# build_prompt / unified_diff (unchanged by the OpenRouter/NIM rework)
# --------------------------------------------------------------------------

def test_build_prompt_is_deterministic():
    diff = J.unified_diff(BEFORE, AFTER)
    p1 = J.build_prompt(["app.py::helper"], diff)
    p2 = J.build_prompt(["app.py::helper"], diff)
    assert p1 == p2


def test_build_prompt_lists_every_requirement():
    diff = J.unified_diff(BEFORE, AFTER)
    prompt = J.build_prompt(["app.py::helper", "app.py::other"], diff)
    assert "app.py::helper" in prompt
    assert "app.py::other" in prompt


def test_build_prompt_states_the_p1_definition_of_implemented():
    """Fairness requirement: the judge is told the same thing P1/P3 check --
    defined AND wired AND not a stub -- not left to guess a weaker bar."""
    prompt = J.build_prompt(["app.py::helper"], "some diff")
    for word in ("DEFINED", "WIRED", "STUB"):
        assert word in prompt


def test_unified_diff_empty_when_nothing_changed():
    assert J.unified_diff(BEFORE, BEFORE) == ""


def test_unified_diff_uses_real_difflib_hunks():
    diff = J.unified_diff(BEFORE, AFTER)
    assert "@@" in diff
    assert "+def helper(x):" in diff


# --------------------------------------------------------------------------
# parsing -- default to IMPLEMENTED on any failure, never OMITTED
# --------------------------------------------------------------------------

REQS = ["app.py::helper", "app.py::other"]


def test_parse_clean_json():
    text = json.dumps({"app.py::helper": "OMITTED", "app.py::other": "IMPLEMENTED"})
    out, failed = J._parse_response(text, REQS)
    assert not failed
    assert out == {"app.py::helper": "OMITTED", "app.py::other": "IMPLEMENTED"}


def test_parse_strips_markdown_fences():
    text = "```json\n" + json.dumps({r: "OMITTED" for r in REQS}) + "\n```"
    out, failed = J._parse_response(text, REQS)
    assert not failed
    assert out == {r: "OMITTED" for r in REQS}


def test_parse_failure_defaults_to_implemented_not_omitted():
    """Defaulting to OMITTED would inflate the baseline's recall and bias
    the comparison in this project's favour -- must default the other way."""
    out, failed = J._parse_response("not json at all", REQS)
    assert failed
    assert out == {r: "IMPLEMENTED" for r in REQS}


def test_parse_missing_key_defaults_only_that_key():
    text = json.dumps({"app.py::helper": "OMITTED"})  # app.py::other missing
    out, failed = J._parse_response(text, REQS)
    assert failed
    assert out["app.py::helper"] == "OMITTED"
    assert out["app.py::other"] == "IMPLEMENTED"


def test_parse_malformed_value_defaults_to_implemented():
    text = json.dumps({"app.py::helper": "MAYBE", "app.py::other": "OMITTED"})
    out, failed = J._parse_response(text, REQS)
    assert failed
    assert out["app.py::helper"] == "IMPLEMENTED"
    assert out["app.py::other"] == "OMITTED"


def test_parse_non_object_top_level_fails():
    out, failed = J._parse_response(json.dumps(["OMITTED", "IMPLEMENTED"]), REQS)
    assert failed
    assert out == {r: "IMPLEMENTED" for r in REQS}


def test_parse_ignores_extra_hallucinated_keys():
    text = json.dumps({**{r: "IMPLEMENTED" for r in REQS}, "app.py::ghost": "OMITTED"})
    out, failed = J._parse_response(text, REQS)
    assert not failed
    assert set(out) == set(REQS)


# --------------------------------------------------------------------------
# _repair_from_prose -- salvages a verdict from unstructured reasoning when
# the model never emitted JSON at all (found live in B4's cached responses;
# see ASSUMPTIONS.md). Deliberately conservative: only a bare OMITTED /
# IMPLEMENTED token counts, only in a bounded window after the LAST mention
# of the requirement, and it must never invent OMITTED out of nothing.
# --------------------------------------------------------------------------

def test_repair_from_prose_finds_arrow_summary_line():
    text = ("Now compile final JSON mapping.\n"
            "- app.py::helper -> OMITTED (not defined)\n"
            "- app.py::other -> IMPLEMENTED\n")
    assert J._repair_from_prose(text, "app.py::helper", REQS) == "OMITTED"
    assert J._repair_from_prose(text, "app.py::other", REQS) == "IMPLEMENTED"


def test_repair_from_prose_finds_sentence_conclusion():
    text = "So I think we should mark app.py::helper as OMITTED."
    assert J._repair_from_prose(text, "app.py::helper", REQS) == "OMITTED"


def test_repair_from_prose_prefers_last_occurrence_over_tentative_earlier_one():
    text = ("app.py::helper might be OMITTED but let's keep reading.\n"
            "Final answer: app.py::helper -> IMPLEMENTED\n")
    assert J._repair_from_prose(text, "app.py::helper", REQS) == "IMPLEMENTED"


def test_repair_from_prose_window_does_not_bleed_into_next_requirement():
    text = "app.py::helper is defined. app.py::other -> OMITTED"
    # helper's own window stops at "app.py::other"; no verdict token in it.
    assert J._repair_from_prose(text, "app.py::helper", REQS) is None


def test_repair_from_prose_returns_none_when_requirement_never_mentioned():
    assert J._repair_from_prose("not json at all", "app.py::helper", REQS) is None


def test_repair_from_prose_returns_none_on_no_unambiguous_token():
    text = "app.py::helper looks done and probably fine."
    assert J._repair_from_prose(text, "app.py::helper", REQS) is None


def test_parse_message_repairs_prose_only_response():
    """End-to-end: content is empty, reasoning is pure prose (no JSON at
    all) but states a clear conclusion -- parse_failed stays True (JSON was
    genuinely never emitted) but the repaired verdict is recovered."""
    msg = {"content": "",
           "reasoning": "Let's check app.py::helper. So app.py::helper -> OMITTED. "
                        "app.py::other is defined and wired -> IMPLEMENTED."}
    out, failed, field = J._parse_message(msg, REQS)
    assert failed  # still true: no valid JSON was ever produced
    assert field == "reasoning"
    assert out == {"app.py::helper": "OMITTED", "app.py::other": "IMPLEMENTED"}


def test_parse_message_repair_never_overwrites_a_confidently_parsed_key():
    """Partial JSON success (one key present, one missing) must NOT let the
    repair pass touch the key the JSON already answered, even if prose in
    another candidate field would suggest a different value for it -- the
    repair loop only ever iterates over `defaulted`, which never includes a
    key the strict JSON parse already resolved."""
    msg = {"content": json.dumps({"app.py::helper": "IMPLEMENTED"}),
           "reasoning": "app.py::helper -> OMITTED (contradicting stray "
                        "prose in a different field, must be ignored)"}
    out, failed, field = J._parse_message(msg, REQS)
    assert failed  # app.py::other is still missing -> still a parse failure
    assert out["app.py::helper"] == "IMPLEMENTED"  # confidently parsed by JSON, untouched
    assert out["app.py::other"] == "IMPLEMENTED"  # missing key, no prose mentions it -> default


def test_parse_message_no_repair_when_nothing_recoverable_still_defaults_implemented():
    msg = {"content": "garbage", "reasoning": "also garbage, no requirement names here"}
    out, failed, field = J._parse_message(msg, REQS)
    assert failed
    assert out == {r: "IMPLEMENTED" for r in REQS}


# --------------------------------------------------------------------------
# _parse_message -- defensive extraction across content / reasoning fields
# --------------------------------------------------------------------------

def test_parse_message_prefers_content_when_it_parses():
    msg = {"content": json.dumps({r: "IMPLEMENTED" for r in REQS}),
           "reasoning": "some chain of thought, not JSON"}
    out, failed, field = J._parse_message(msg, REQS)
    assert not failed
    assert field == "content"


def test_parse_message_falls_back_to_reasoning_when_content_is_empty():
    """Observed failure mode this guards against: a provider returns empty
    `content` and puts the actual JSON answer in `reasoning` instead."""
    msg = {"content": "", "reasoning": json.dumps({r: "OMITTED" for r in REQS})}
    out, failed, field = J._parse_message(msg, REQS)
    assert not failed
    assert field == "reasoning"
    assert out == {r: "OMITTED" for r in REQS}


def test_parse_message_tries_reasoning_content_field_too():
    msg = {"content": "not json", "reasoning_content": json.dumps({r: "IMPLEMENTED" for r in REQS})}
    out, failed, field = J._parse_message(msg, REQS)
    assert not failed
    assert field == "reasoning_content"


def test_parse_message_all_fields_fail_defaults_to_implemented():
    msg = {"content": "garbage", "reasoning": "also garbage"}
    out, failed, field = J._parse_message(msg, REQS)
    assert failed
    assert out == {r: "IMPLEMENTED" for r in REQS}


def test_parse_message_missing_all_fields_does_not_crash():
    out, failed, field = J._parse_message({}, REQS)
    assert failed
    assert out == {r: "IMPLEMENTED" for r in REQS}


# --------------------------------------------------------------------------
# truncation
# --------------------------------------------------------------------------

def test_truncate_noop_under_ceiling():
    diff, truncated = J._truncate("short diff", ceiling=1000)
    assert not truncated
    assert diff == "short diff"


def test_truncate_flags_and_shortens_over_ceiling():
    big = "\n".join(f"+line {i}" for i in range(2000))
    diff, truncated = J._truncate(big, ceiling=50)
    assert truncated
    assert len(diff) < len(big)
    assert "truncated" in diff


# --------------------------------------------------------------------------
# cache -- content-addressed, keyed by sha256(base_url + model + provider +
# reasoning_effort + prompt), not just model + prompt
# --------------------------------------------------------------------------

def test_cache_key_stable_for_identical_inputs():
    k1 = J._cache_key(B4, "prompt text")
    k2 = J._cache_key(B4, "prompt text")
    assert k1 == k2


def test_cache_key_changes_with_prompt():
    assert J._cache_key(B4, "prompt text") != J._cache_key(B4, "different prompt")


def test_cache_key_changes_with_judge():
    """The whole point of item 5: a provider or reasoning-depth switch is a
    different engine and must never hit the same cache entry."""
    keys = {J._cache_key(cfg, "same prompt") for cfg in (B4, B5, B6)}
    assert len(keys) == 3


def test_cache_key_changes_with_provider_pin():
    import dataclasses
    b4_other_provider = dataclasses.replace(B4, provider_slug="groq")
    assert J._cache_key(B4, "p") != J._cache_key(b4_other_provider, "p")


def test_cache_key_changes_with_reasoning_effort():
    import dataclasses
    b4_low = dataclasses.replace(B4, reasoning_effort="low")
    assert J._cache_key(B4, "p") != J._cache_key(b4_low, "p")


def test_cache_key_changes_with_variance_seed():
    """The k=5 variance run calls the SAME prompt 5 times with different
    seeds. Without a seed in the key, seeds 2-5 would cache-hit seed 1's
    response and scripts/k5_variance.py would report stdev=0.0000 as a
    measurement when it is actually a caching artifact."""
    keys = {J._cache_key(B4, "same prompt", variance_seed=s) for s in range(5)}
    assert len(keys) == 5


def test_cache_key_omitting_variance_seed_matches_explicit_none():
    """The k=1 headline sweep never passes variance_seed. Its cache key must
    be byte-identical whether the argument is omitted or passed as None --
    this fix must not invalidate any k=1 cache entry already committed."""
    assert J._cache_key(B4, "p") == J._cache_key(B4, "p", variance_seed=None)


def test_cache_key_variance_seed_zero_differs_from_no_seed():
    """seed=0 is falsy but must still be a DIFFERENT key from "no variance
    seed at all" -- a `if variance_seed:` check instead of `is not None`
    would silently collapse seed 0 back into the k=1 cache key."""
    assert J._cache_key(B4, "p") != J._cache_key(B4, "p", variance_seed=0)


# --------------------------------------------------------------------------
# provider pin verification -- fail loudly on mismatch
# --------------------------------------------------------------------------

def test_provider_matches_accepts_display_name_for_slug():
    assert J._provider_matches("Cerebras", "cerebras/fp16")
    assert J._provider_matches("cerebras", "cerebras/fp16")


def test_provider_matches_rejects_a_different_provider():
    assert not J._provider_matches("DeepInfra", "cerebras/fp16")


def test_provider_matches_rejects_missing_provider():
    assert not J._provider_matches(None, "cerebras/fp16")
    assert not J._provider_matches("", "cerebras/fp16")


def test_extract_provider_reads_top_level_field():
    assert J._extract_provider({"provider": "Cerebras"}) == "Cerebras"


def test_extract_provider_reads_nested_metadata_field():
    raw = {"openrouter_metadata": {"endpoints": {"available": [
        {"provider": "DeepInfra", "selected": False},
        {"provider": "Cerebras", "selected": True},
    ]}}}
    assert J._extract_provider(raw) == "Cerebras"


def test_extract_provider_returns_none_when_absent():
    assert J._extract_provider({}) is None


# --------------------------------------------------------------------------
# cost extraction -- trust OpenRouter's reported cost; NIM falls back to a
# labelled estimate
# --------------------------------------------------------------------------

def test_extract_cost_prefers_reported_value():
    raw = {"usage": {"cost": 0.0042, "prompt_tokens": 100, "completion_tokens": 20}}
    cost, source = J._extract_cost(raw, B4)
    assert cost == 0.0042
    assert source == "reported"


def test_extract_cost_estimates_for_nim_when_absent():
    raw = {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}
    cost, source = J._extract_cost(raw, B6)
    assert source == "estimated"
    assert cost == B6.nim_price_per_1m[0]


def test_extract_cost_unavailable_when_no_fallback_and_no_reported_cost():
    raw = {"usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    cost, source = J._extract_cost(raw, B5)
    assert cost is None
    assert source == "unavailable"


# --------------------------------------------------------------------------
# d_xxx -- full call, with _call_model monkeypatched (no network)
# --------------------------------------------------------------------------

def _fake_meta(provider="Cerebras", cost=0.001):
    return {"usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "provider_returned": provider, "cost": cost, "cost_source": "reported"}


def test_d_gpt_oss_120b_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(J, "CACHE_DIR", str(tmp_path / "cache"))

    calls = {"n": 0}

    def fake_call_model(cfg, prompt):
        calls["n"] += 1
        return ({"content": json.dumps({"app.py::helper": "IMPLEMENTED"})},
                _fake_meta())

    monkeypatch.setattr(J, "_call_model", fake_call_model)

    reqs = ["app.py::helper"]
    v1 = J.d_gpt_oss_120b("spec", reqs, BEFORE, AFTER, {})
    v2 = J.d_gpt_oss_120b("spec", reqs, BEFORE, AFTER, {})

    assert v1 == v2 == {"app.py::helper": "IMPLEMENTED"}
    assert calls["n"] == 1, "second call should be served from cache, not the API"


def test_judge_never_caches_a_raised_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(J, "CACHE_DIR", str(tmp_path / "cache"))

    def flaky(cfg, prompt):
        raise RuntimeError("simulated transient failure")

    monkeypatch.setattr(J, "_call_model", flaky)

    reqs = ["app.py::helper"]
    try:
        J.d_gpt_oss_120b("spec", reqs, BEFORE, AFTER, {})
        assert False, "expected the simulated failure to propagate"
    except RuntimeError:
        pass

    cache_dir = tmp_path / "cache"
    real_entries = [p for p in cache_dir.glob("*.json")] if cache_dir.exists() else []
    assert not real_entries


def test_judge_populates_judge_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(J, "CACHE_DIR", str(tmp_path / "cache"))

    def fake_call_model(cfg, prompt):
        return ({"content": json.dumps({"app.py::helper": "OMITTED"})},
                _fake_meta(provider="Cerebras", cost=0.0055))

    monkeypatch.setattr(J, "_call_model", fake_call_model)

    record = {}
    out = J.d_gpt_oss_120b("spec", ["app.py::helper"], BEFORE, AFTER,
                            {"_judge_record": record})
    assert out == {"app.py::helper": "OMITTED"}
    assert record["judge_id"] == "B4 LLM judge (gpt-oss-120b)"
    assert record["model_slug"] == "openai/gpt-oss-120b"
    assert record["provider_pinned"] == "cerebras/fp16"
    assert record["provider_returned"] == "Cerebras"
    assert record["reasoning_effort"] == B4.reasoning_effort
    assert record["parse_failed"] is False
    assert record["truncated"] is False
    assert record["cache_hit"] is False
    assert record["parsed_from_field"] == "content"
    assert record["cost_usd"] == 0.0055


def test_judge_raises_loudly_on_provider_mismatch(tmp_path, monkeypatch):
    """The core enforcement this rework exists for: a provider switch is
    caught and stops the run, not silently absorbed into the results."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(J, "CACHE_DIR", str(tmp_path / "cache"))

    def fake_call_model(cfg, prompt):
        return ({"content": json.dumps({"app.py::helper": "IMPLEMENTED"})},
                _fake_meta(provider="DeepInfra"))  # NOT the pinned Cerebras

    monkeypatch.setattr(J, "_call_model", fake_call_model)

    try:
        J.d_gpt_oss_120b("spec", ["app.py::helper"], BEFORE, AFTER, {})
        assert False, "expected a provider-mismatch RuntimeError"
    except RuntimeError as e:
        assert "DeepInfra" in str(e)
        assert "cerebras/fp16" in str(e)


def test_d_llm_judge_variance_seeds_are_not_cache_hits_of_each_other(tmp_path, monkeypatch):
    """End-to-end regression for the k=5 caching bug: calling the SAME
    (spec, reqs, before, after) 5 times with 5 different
    ctx["_variance_seed"] values must make 5 real calls, not 1 call + 4
    cache hits."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(J, "CACHE_DIR", str(tmp_path / "cache"))

    calls = {"n": 0}

    def fake_call_model(cfg, prompt):
        calls["n"] += 1
        return ({"content": json.dumps({"app.py::helper": "IMPLEMENTED"})},
                _fake_meta())

    monkeypatch.setattr(J, "_call_model", fake_call_model)

    reqs = ["app.py::helper"]
    for seed in range(5):
        J.d_gpt_oss_120b("spec", reqs, BEFORE, AFTER, {"_variance_seed": seed})
    assert calls["n"] == 5, "each variance seed must be an independent call"

    # re-running seed 0 a second time SHOULD now hit the cache -- the fix
    # must not have disabled caching altogether for the k=5 path.
    J.d_gpt_oss_120b("spec", reqs, BEFORE, AFTER, {"_variance_seed": 0})
    assert calls["n"] == 5, "re-running an already-seen seed should cache-hit"


def test_d_llm_judge_k1_path_unaffected_by_variance_seed_fix(tmp_path, monkeypatch):
    """The k=1 headline sweep (no ctx["_variance_seed"]) must still get a
    single cache hit on repeat, exactly as before this fix."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(J, "CACHE_DIR", str(tmp_path / "cache"))

    calls = {"n": 0}

    def fake_call_model(cfg, prompt):
        calls["n"] += 1
        return ({"content": json.dumps({"app.py::helper": "IMPLEMENTED"})},
                _fake_meta())

    monkeypatch.setattr(J, "_call_model", fake_call_model)

    reqs = ["app.py::helper"]
    J.d_gpt_oss_120b("spec", reqs, BEFORE, AFTER, {})
    J.d_gpt_oss_120b("spec", reqs, BEFORE, AFTER, {})
    assert calls["n"] == 1


def test_judge_enforces_provider_check_for_b6_since_openrouter_repoint(tmp_path, monkeypatch):
    """B6 was re-pointed 2026-08-13 from direct NVIDIA NIM (no provider pin)
    to OpenRouter (pinned to deepinfra/bf16) -- it must now fail loudly on a
    provider mismatch exactly like B4/B5, not silently accept whatever
    OpenRouter reports."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(J, "CACHE_DIR", str(tmp_path / "cache"))

    def fake_call_model(cfg, prompt):
        return ({"content": json.dumps({"app.py::helper": "IMPLEMENTED"})},
                {"usage": {"prompt_tokens": 5, "completion_tokens": 2},
                 "provider_returned": None, "cost": 0.0001, "cost_source": "estimated"})

    monkeypatch.setattr(J, "_call_model", fake_call_model)

    try:
        J.d_nim("spec", ["app.py::helper"], BEFORE, AFTER, {})
        assert False, "expected a provider-mismatch RuntimeError for B6 now"
    except RuntimeError as e:
        assert "deepinfra/bf16" in str(e)


def test_judge_passes_provider_check_for_b6_when_pin_matches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(J, "CACHE_DIR", str(tmp_path / "cache"))

    def fake_call_model(cfg, prompt):
        return ({"content": json.dumps({"app.py::helper": "IMPLEMENTED"})},
                {"usage": {"prompt_tokens": 5, "completion_tokens": 2},
                 "provider_returned": "DeepInfra", "cost": 0.0001, "cost_source": "reported"})

    monkeypatch.setattr(J, "_call_model", fake_call_model)

    out = J.d_nim("spec", ["app.py::helper"], BEFORE, AFTER, {})
    assert out == {"app.py::helper": "IMPLEMENTED"}
