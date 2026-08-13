"""
Micro-repo unit tests for the mutation engine and the detectors.

These pin the semantics that every number in results/ depends on. If a mutation
does not actually produce the failure mode it claims, the ground-truth labels
are wrong and nothing downstream means anything.
"""

from __future__ import annotations

import ast
import sys

sys.path.insert(0, ".")

from omitbench import detectors as D  # noqa: E402
from omitbench import mutate as M  # noqa: E402


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


# --------------------------------------------------------------------------
# mutation engine
# --------------------------------------------------------------------------

def test_new_symbols_finds_added_definition():
    syms = M.new_symbols(BEFORE, AFTER)
    assert ("app.py", "helper") in syms
    assert ("app.py", "run") not in syms, "run existed before; not a new symbol"


def test_absent_removes_definition_but_leaves_call_site():
    out = M.mut_absent(AFTER, "app.py", "helper")
    assert out is not None
    src = "\n".join(out["app.py"])
    assert "def helper" not in src
    # the call survives -- this is exactly why grep cannot catch ABSENT
    assert "helper(3)" in src


def test_unwired_keeps_definition_and_kills_call_site():
    out = M.mut_unwired(AFTER, "app.py", "helper")
    assert out is not None
    src = "\n".join(out["app.py"])
    assert "def helper" in src, "UNWIRED must keep the definition"
    assert not M.call_lines(src, "helper", M._defs(src)["helper"][:2]), \
        "UNWIRED must leave no call site"


def test_stub_keeps_signature_and_hollows_body():
    out = M.mut_stub(AFTER, "app.py", "helper")
    assert out is not None
    src = "\n".join(out["app.py"])
    assert "def helper(x)" in src, "STUB must keep the signature"
    assert "total = x * 2" not in src, "STUB must remove the real body"
    assert "helper(3)" in src, "STUB must keep call sites"


def test_every_mutation_produces_parseable_python():
    """A mutation that breaks syntax is caught by any linter and is not silent."""
    for name, fn in M.MUTATIONS.items():
        out = fn(AFTER, "app.py", "helper")
        assert out is not None, f"{name} failed to apply"
        for path, lines in out.items():
            ast.parse("\n".join(lines))  # raises SyntaxError on failure


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------

REQ = "app.py::helper"


def verdicts(after):
    return {name: fn("spec", [REQ], BEFORE, after, {})[REQ]
            for name, fn in D.DETECTORS.items()}


def test_clean_patch_is_not_flagged_by_p1():
    v = verdicts(AFTER)
    assert v["P1 defined"] == "IMPLEMENTED"


def test_p1_catches_absent():
    v = verdicts(M.mut_absent(AFTER, "app.py", "helper"))
    assert v["P1 defined"] == "OMITTED"


def test_p1_cannot_catch_unwired_or_stub():
    """
    Documented limitation, asserted so it cannot silently change.
    'Is it defined' is blind to hollow implementations by construction.
    """
    for mut in (M.mut_unwired, M.mut_stub):
        v = verdicts(mut(AFTER, "app.py", "helper"))
        assert v["P1 defined"] == "IMPLEMENTED"


def test_p2_catches_unwired():
    v = verdicts(M.mut_unwired(AFTER, "app.py", "helper"))
    assert v["P2 defined+reachable"] == "OMITTED"


def test_p3_catches_stub():
    v = verdicts(M.mut_stub(AFTER, "app.py", "helper"))
    assert v["P3 defined+reachable+body"] == "OMITTED"


def test_grep_is_fooled_by_absent():
    """
    The no-AST baseline sees `helper(3)` still in the diff and concludes the
    requirement is implemented. This is the argument for AST, as a test.
    """
    v = verdicts(M.mut_absent(AFTER, "app.py", "helper"))
    assert v["B3 line-grep (no AST)"] == "IMPLEMENTED"


def test_p2_false_positive_on_uncalled_public_api():
    """
    Pins the defect that the pilot corpus filter was hiding: a correctly
    implemented public symbol that nothing calls internally is flagged OMITTED
    by the reachability detectors. This is why P2/P3 score FPR ~0.84 at scale.
    """
    before = repo(**{"api.py": "VERSION = '1'"})
    after = repo(**{"api.py": """
VERSION = '1'

def public_entrypoint(x):
    return x + 1
"""})
    req = "api.py::public_entrypoint"
    p1 = D.d_defined("spec", [req], before, after, {})[req]
    p2 = D.d_reachable("spec", [req], before, after, {})[req]
    assert p1 == "IMPLEMENTED", "the symbol IS correctly implemented"
    assert p2 == "OMITTED", "known false positive -- documented, not fixed"


def test_path_qualification_prevents_name_collision():
    """
    Bare-name matching let a same-named symbol elsewhere mask a real deletion,
    capping recall on ABSENT at 0.43. Qualified names fix it.
    """
    before = repo(**{"a.py": "X = 1", "b.py": "def helper():\n    return 0"})
    after = repo(**{"a.py": "X = 1", "b.py": "def helper():\n    return 0"})
    # a.py::helper was never defined in a.py, even though b.py defines `helper`
    req = "a.py::helper"
    assert D.d_defined("spec", [req], before, after, {})[req] == "OMITTED"


def test_all_detectors_return_a_verdict_for_every_requirement():
    reqs = [REQ, "app.py::run", "nowhere.py::ghost"]
    for name, fn in D.DETECTORS.items():
        out = fn("spec", reqs, BEFORE, AFTER, {})
        assert set(out) == set(reqs), f"{name} dropped a requirement"
        assert all(v in ("IMPLEMENTED", "OMITTED") for v in out.values())


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            bad += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
