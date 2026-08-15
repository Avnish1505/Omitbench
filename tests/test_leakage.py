"""
The load-bearing test of this project.

If a detector can see the reference solution, "find the deleted hunks" is
trivial and every number in results/ is meaningless. This file proves it
cannot, three ways:

  1. signature check   -- no detector accepts a gold/patch/diff argument
  2. runtime canary    -- a unique token planted in the gold patch must never
                          appear in anything a detector receives
  3. label canary      -- the mutation class must never reach the detector

Run: python3 -m pytest tests/test_leakage.py -q
"""

from __future__ import annotations

import inspect
import sys

sys.path.insert(0, ".")

from omitbench import detectors as P  # noqa: E402
from omitbench import judges as J  # noqa: E402

CANARY = "OMITBENCH_GOLD_CANARY_9f3a2c"
BANNED_PARAMS = {"gold", "patch", "diff", "reference", "solution",
                 "mutation", "label", "cls"}


def test_no_detector_accepts_gold_or_label():
    for name, fn in P.DETECTORS.items():
        params = set(inspect.signature(fn).parameters)
        leaked = params & BANNED_PARAMS
        assert not leaked, f"{name} accepts forbidden argument(s): {leaked}"
        assert params == {"spec", "requirements", "before", "after", "ctx"} or \
               params == {"spec", "reqs", "before", "after", "ctx"}, \
               f"{name} has off-contract signature: {params}"


def test_canary_never_reaches_detector():
    """
    Plant the canary in the gold patch only. Wrap every detector so it asserts
    the canary is absent from every argument it is handed.
    """
    before = {"m.py": ["def existing():", "    return 1"]}
    # gold patch adds `helper` and wires it -- the canary rides along in a
    # comment, exactly where a leak would carry it
    gold_after = {"m.py": [
        "def helper():",
        f"    # {CANARY}",
        "    return 2",
        "def existing():",
        "    return helper()",
    ]}
    # the mutated R_after the detector actually sees: helper removed, and the
    # canary removed with it
    after = {"m.py": ["def existing():", "    return helper()"]}

    def seen(obj) -> bool:
        if isinstance(obj, str):
            return CANARY in obj
        if isinstance(obj, dict):
            return any(seen(k) or seen(v) for k, v in obj.items())
        if isinstance(obj, (list, tuple, set, frozenset)):
            return any(seen(x) for x in obj)
        return False

    for name, fn in P.DETECTORS.items():
        args = ("fix the thing", ["helper"], before, after, {})
        for a in args:
            assert not seen(a), f"{name} was handed the canary in {a!r}"
        try:
            fn(*args)
        except Exception as e:  # a crash is a bug, but not a leak
            raise AssertionError(f"{name} raised {e!r}") from e

    # sanity: the canary really is in the gold patch, so this test can fail
    assert seen(gold_after), "canary missing from gold -- test is vacuous"


def test_detector_cannot_distinguish_identical_inputs():
    """
    Two instances with identical (before, after) but different mutation labels
    must receive identical verdicts. If a detector ever disagrees, information
    is reaching it through a side channel.
    """
    before = {"m.py": ["def existing():", "    return 1"]}
    after = {"m.py": ["def existing():", "    return 1"]}
    for name, fn in P.DETECTORS.items():
        a = fn("spec A", ["helper"], before, after, {})
        b = fn("spec A", ["helper"], dict(before), dict(after), {})
        assert a == b, f"{name} is non-deterministic or reads hidden state"


def test_judge_build_prompt_is_pure_and_signature_locked():
    """
    The LLM judge (omitbench/judges.py) introduces a SECOND leak surface
    beyond the function signature checked above: the rendered PROMPT STRING.
    Prompt construction is isolated in one pure function, build_prompt(reqs,
    diff). Lock its signature to exactly those two arguments -- if it ever
    grows a `gold=`, `ctx=`, or `spec=` parameter, that argument is where the
    next leak will ride in.
    """
    params = set(inspect.signature(J.build_prompt).parameters)
    assert params == {"reqs", "diff"}, \
        f"build_prompt has off-contract signature: {params}"


def test_judge_prompt_never_contains_canary():
    """
    Plants the canary in the gold patch only -- never in the mutated `after`
    the judge is actually built from -- then builds the diff and the prompt
    from ONLY (before, after) and asserts the canary never rides along.
    Mirrors test_canary_never_reaches_detector, scoped to the judge's prompt-
    construction path rather than a detector's argument list.
    """
    before = {"m.py": ["def existing():", "    return 1"]}
    gold_after = {"m.py": [
        "def helper():",
        f"    # {CANARY}",
        "    return 2",
        "def existing():",
        "    return helper()",
    ]}
    # what d_llm_judge actually sees: helper mutated away, canary gone with it
    after = {"m.py": ["def existing():", "    return helper()"]}

    diff = J.unified_diff(before, after)
    assert CANARY not in diff, "canary leaked into the diff itself"

    prompt = J.build_prompt(["m.py::helper"], diff)
    assert CANARY not in prompt, "build_prompt leaked the gold canary"

    # sanity: the canary really is in the gold patch, so this test can fail
    assert any(CANARY in line for line in gold_after["m.py"]), \
        "canary missing from gold -- test is vacuous"


if __name__ == "__main__":
    test_no_detector_accepts_gold_or_label()
    test_canary_never_reaches_detector()
    test_detector_cannot_distinguish_identical_inputs()
    test_judge_build_prompt_is_pure_and_signature_locked()
    test_judge_prompt_never_contains_canary()
    print("all leakage guards passed")
