from __future__ import annotations

from omitbench import gate as G


def test_parses_symbol_in_path_form():
    body = "Plan:\n- [ ] retry_with_backoff in src/client.py\n"
    assert G.parse_requirements(body) == ["src/client.py::retry_with_backoff"]


def test_parses_backticked_symbol_with_parens_and_path():
    body = "- [ ] `validate()` in `src/schema.py`\n"
    assert G.parse_requirements(body) == ["src/schema.py::validate"]


def test_parses_qualified_form_directly():
    body = "- [x] src/client.py::retry_with_backoff\n"
    assert G.parse_requirements(body) == ["src/client.py::retry_with_backoff"]


def test_checkbox_state_does_not_affect_parsing():
    checked = G.parse_requirements("- [x] foo in bar.py\n")
    unchecked = G.parse_requirements("- [ ] foo in bar.py\n")
    assert checked == unchecked == ["bar.py::foo"]


def test_multiple_items_in_order():
    body = (
        "- [ ] retry_with_backoff in src/client.py\n"
        "- [ ] validate in src/schema.py\n"
    )
    assert G.parse_requirements(body) == [
        "src/client.py::retry_with_backoff",
        "src/schema.py::validate",
    ]


def test_non_checklist_lines_are_ignored():
    body = "Closes #42\n\nThis PR adds retry_with_backoff to src/client.py.\n"
    assert G.parse_requirements(body) == []


def test_unrecognized_checklist_items_are_reported_not_dropped_silently():
    body = "- [ ] update the docs\n- [ ] retry_with_backoff in src/client.py\n"
    assert G.parse_requirements(body) == ["src/client.py::retry_with_backoff"]
    assert G.unrecognized_items(body) == ["update the docs"]


def test_empty_body_yields_no_requirements_and_no_unrecognized_items():
    assert G.parse_requirements("") == []
    assert G.unrecognized_items("") == []


# --------------------------------------------------------------------------
# Task 2: snapshot fetch + P1 scoring
# --------------------------------------------------------------------------

import os
import subprocess
import tempfile


def _init_repo_with_file(path: str, content: str) -> tuple[str, str]:
    """Real git repo in a tmp dir with one commit. Returns (repo_dir, sha)."""
    d = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q", d], check=True)
    subprocess.run(["git", "-C", d, "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", d, "config", "user.name", "t"], check=True)
    full = f"{d}/{path}"
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    subprocess.run(["git", "-C", d, "add", "."], check=True)
    subprocess.run(["git", "-C", d, "commit", "-q", "-m", "init"], check=True)
    sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return d, sha


def test_fetch_after_snapshot_reads_requested_paths_at_sha():
    d, sha = _init_repo_with_file(
        "src/client.py", "def retry_with_backoff():\n    pass\n"
    )
    snap = G.fetch_after_snapshot(d, sha, ["src/client.py"])
    assert snap["src/client.py"] == ["def retry_with_backoff():", "    pass", ""]


def test_fetch_after_snapshot_missing_path_maps_to_empty_list():
    d, sha = _init_repo_with_file("src/client.py", "x = 1\n")
    snap = G.fetch_after_snapshot(d, sha, ["src/nope.py"])
    assert snap["src/nope.py"] == []


def test_score_pr_flags_missing_symbol_as_omitted():
    d, sha = _init_repo_with_file("src/client.py", "def other():\n    pass\n")
    body = "- [ ] retry_with_backoff in src/client.py\n"
    result = G.score_pr(body, repo_dir=d, head_sha=sha)
    assert result["omitted"] == ["src/client.py::retry_with_backoff"]
    assert result["implemented"] == []
    assert result["unrecognized"] == []


def test_score_pr_marks_present_symbol_as_implemented():
    d, sha = _init_repo_with_file(
        "src/client.py", "def retry_with_backoff():\n    pass\n"
    )
    body = "- [ ] retry_with_backoff in src/client.py\n"
    result = G.score_pr(body, repo_dir=d, head_sha=sha)
    assert result["implemented"] == ["src/client.py::retry_with_backoff"]
    assert result["omitted"] == []


def test_score_pr_passes_through_unrecognized_items():
    d, sha = _init_repo_with_file("src/client.py", "x = 1\n")
    body = "- [ ] update the docs\n"
    result = G.score_pr(body, repo_dir=d, head_sha=sha)
    assert result["unrecognized"] == ["update the docs"]
    assert result["omitted"] == result["implemented"] == []


def test_score_pr_never_calls_detector_with_extra_arguments(monkeypatch):
    """Signature-lock guard: score_pr must call d_defined with exactly
    (spec, reqs, before, after, ctx) -- mirrors the discipline
    tests/test_leakage.py enforces for every other detector entry point,
    without touching that file (CLAUDE.md: do not edit test_leakage.py)."""
    calls = []

    def fake_d_defined(spec, reqs, before, after, ctx):
        calls.append((spec, reqs, before, after, ctx))
        return {r: "IMPLEMENTED" for r in reqs}

    monkeypatch.setattr(G.D, "d_defined", fake_d_defined)
    d, sha = _init_repo_with_file("src/client.py", "def f():\n    pass\n")
    G.score_pr("- [ ] f in src/client.py\n", repo_dir=d, head_sha=sha)
    assert len(calls) == 1
    spec, reqs, before, after, ctx = calls[0]
    assert before == {}
    assert reqs == ["src/client.py::f"]
    assert isinstance(after, dict) and isinstance(ctx, dict)
