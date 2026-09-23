"""
B7 (omitbench/jev.py): leakage guards mirroring tests/test_leakage.py's judge
tests, plus behaviour tests against a mocked TypeSafe endpoint. No network,
no API key: jev._post is monkeypatched in every test that would call out.
"""

from __future__ import annotations

import inspect
import sys

import pytest

sys.path.insert(0, ".")

from omitbench import jev as V  # noqa: E402
from omitbench import judges as J  # noqa: E402

CANARY = "OMITBENCH_GOLD_CANARY_9f3a2c"
BANNED = {"gold", "patch", "reference", "solution", "mutation", "label", "cls"}


# ---------------- leakage ------------------------------------------------

def test_detectors_are_on_contract():
    for name, fn in V.DETECTORS.items():
        params = set(inspect.signature(fn).parameters)
        assert not params & BANNED, f"{name} accepts {params & BANNED}"
        assert params == {"spec", "reqs", "before", "after", "ctx"}, name


@pytest.mark.parametrize("builder", [V.build_jev_request, V.build_jev_request_split])
def test_request_builders_are_signature_locked(builder):
    assert set(inspect.signature(builder).parameters) == {"reqs", "diff"}


@pytest.mark.parametrize("builder", [V.build_jev_request, V.build_jev_request_split])
def test_request_never_contains_gold_canary(builder):
    before = {"m.py": ["def existing():", "    return 1"]}
    gold_after = {"m.py": ["def helper():", f"    # {CANARY}", "    return 2",
                           "def existing():", "    return helper()"]}
    after = {"m.py": ["def existing():", "    return helper()"]}
    diff = J.unified_diff(before, after)
    payload = builder(["m.py::helper"], diff)
    assert CANARY not in repr(payload)
    assert any(CANARY in line for line in gold_after["m.py"]), "vacuous test"


@pytest.mark.parametrize("builder", [V.build_jev_request, V.build_jev_request_split])
def test_request_is_pure(builder):
    a = builder(["a.py::X", "a.py::Y"], "diff text")
    b = builder(["a.py::Y", "a.py::X"], "diff text")   # order-insensitive
    assert a == b
    assert a["model"] == V.PINNED_MODEL and a["state"] == "diff text"


def test_question_text_is_frozen():
    """ASSUMPTIONS.md section 13 freezes the question text. If this fails,
    you changed a pre-registered detector: make a NEW detector id instead."""
    import hashlib
    blob = V._SINGLE_INSTR + repr(V._SINGLE_CRITERIA) + repr(V._SPLIT)
    assert hashlib.sha256(blob.encode()).hexdigest()[:16] == FROZEN_HASH


# ---------------- behaviour against a fake endpoint ----------------------

def _fake_post(values: dict, model=V.PINNED_MODEL, calls=None):
    def post(payload):
        if calls is not None:
            calls.append(payload)
        answers = {k: {"type": "noul", "noul": values.get(k, 0.9)}
                   for k in payload["questions"]}
        return {"model": model, "answers": answers,
                "usage": {"input_tokens": 1000, "output_tokens": 0}}
    return post


BEFORE = {"m.py": ["def existing():", "    return 1"]}
AFTER = {"m.py": ["def helper():", "    return 2", "", "def existing():",
                  "    return helper()"]}
REQS = ["m.py::helper", "m.py::other"]   # sorted: helper -> r0, other -> r1


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "CACHE_DIR", str(tmp_path))
    V.STATS.clear()


def test_single_thresholds_at_half_and_records_probability(monkeypatch):
    monkeypatch.setattr(V, "_post", _fake_post({"r0": 0.8, "r1": 0.2}))
    rec = {}
    out = V.d_jev_single("", REQS, BEFORE, AFTER, {"_judge_record": rec})
    assert out == {"m.py::helper": "IMPLEMENTED", "m.py::other": "OMITTED"}
    assert rec["p_implemented"] == {"m.py::helper": 0.8, "m.py::other": 0.2}
    assert rec["cost_usd"] == pytest.approx(1000 * 0.042 / 1e6)


def test_split_uses_min_of_three(monkeypatch):
    vals = {"r0_defined": 0.95, "r0_wired": 0.9, "r0_real_body": 0.3,
            "r1_defined": 0.9, "r1_wired": 0.8, "r1_real_body": 0.7}
    monkeypatch.setattr(V, "_post", _fake_post(vals))
    rec = {}
    out = V.d_jev_split("", REQS, BEFORE, AFTER, {"_judge_record": rec})
    assert rec["p_implemented"] == {"m.py::helper": 0.3, "m.py::other": 0.7}
    assert out["m.py::helper"] == "OMITTED" and out["m.py::other"] == "IMPLEMENTED"


def test_missing_answer_defaults_to_implemented_never_omitted(monkeypatch):
    def post(payload):
        return {"model": V.PINNED_MODEL, "answers": {"r0": {"noul": "junk"}}}
    monkeypatch.setattr(V, "_post", post)
    rec = {}
    out = V.d_jev_single("", REQS, BEFORE, AFTER, {"_judge_record": rec})
    assert set(out.values()) == {"IMPLEMENTED"}
    assert rec["parse_failed"] is True
    assert rec["p_implemented"] == {"m.py::helper": None, "m.py::other": None}


def test_model_pin_mismatch_raises_and_is_not_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(V, "_post", _fake_post({}, model="jev-2.0.0", calls=calls))
    with pytest.raises(RuntimeError, match="pinned"):
        V.d_jev_single("", REQS, BEFORE, AFTER, {})
    with pytest.raises(RuntimeError, match="pinned"):
        V.d_jev_single("", REQS, BEFORE, AFTER, {})
    assert len(calls) == 2, "a wrong-model response must never be served from cache"


def test_second_run_is_a_cache_hit(monkeypatch):
    calls = []
    monkeypatch.setattr(V, "_post", _fake_post({"r0": 0.8}, calls=calls))
    V.d_jev_single("", REQS, BEFORE, AFTER, {})
    rec = {}
    V.d_jev_single("", REQS, BEFORE, AFTER, {"_judge_record": rec})
    assert len(calls) == 1 and rec["cache_hit"] is True and rec["cost_usd"] is None


def test_b7_sees_exactly_the_diff_b5_sees(monkeypatch):
    calls = []
    monkeypatch.setattr(V, "_post", _fake_post({}, calls=calls))
    V.d_jev_single("", REQS, BEFORE, AFTER, {})
    b5_diff, _ = J._truncate(J.unified_diff(BEFORE, AFTER), J.DIFF_TOKEN_CEILING)
    assert calls[0]["state"] == b5_diff


def test_no_key_means_no_call(monkeypatch):
    monkeypatch.delenv(V.API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        V._post({"model": V.PINNED_MODEL, "state": "", "questions": {}})


FROZEN_HASH = "b1eab0befa8aec0f"


# ---------------- the real HTTP path, against a local server -------------

def test_post_sends_bearer_retries_429_and_parses(monkeypatch):
    import http.server
    import json
    import threading

    seen = {"n": 0, "auth": None, "body": None}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["n"] += 1
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if seen["n"] == 1:
                self.send_response(429)
                self.end_headers()
                return
            out = json.dumps({"model": V.PINNED_MODEL,
                              "answers": {"r0": {"type": "noul", "noul": 0.7}},
                              "usage": {"input_tokens": 5}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(V, "API_URL", f"http://127.0.0.1:{srv.server_port}/v1/systemone")
        monkeypatch.setenv(V.API_KEY_ENV, "sk-test")
        monkeypatch.setattr(V.time, "sleep", lambda s: None)
        payload = V.build_jev_request(["m.py::x"], "d")
        resp = V._post(payload)
    finally:
        srv.shutdown()
    assert seen["n"] == 2, "429 must be retried"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"] == payload
    assert resp["answers"]["r0"]["noul"] == 0.7


def test_post_does_not_retry_422(monkeypatch):
    import io
    import urllib.error

    n = {"c": 0}

    def boom(*a, **k):
        n["c"] += 1
        raise urllib.error.HTTPError("u", 422, "bad", {}, io.BytesIO(b'{"detail":"x"}'))
    monkeypatch.setenv(V.API_KEY_ENV, "sk-test")
    monkeypatch.setattr(V.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="HTTP 422"):
        V._post({"model": V.PINNED_MODEL, "state": "", "questions": {}})
    assert n["c"] == 1


# ---------------- interpreter guard --------------------------------------

def _load_run_jev():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_jev", "scripts/run_jev.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("ver", [(3, 14, 0), (3, 15, 1)])
def test_run_jev_refuses_python_314_plus(ver):
    # PEP 758: on 3.14+ Python 2 `except X, e:` parses, itsdangerous grows
    # from 30 to 35 instances and B7 no longer aligns with the shards.
    RJEV = _load_run_jev()
    with pytest.raises(SystemExit, match="PEP 758"):
        RJEV.check_interpreter(ver)


@pytest.mark.parametrize("ver", [(3, 11, 9), (3, 12, 14), (3, 13, 5)])
def test_run_jev_accepts_python_up_to_313(ver):
    _load_run_jev().check_interpreter(ver)
