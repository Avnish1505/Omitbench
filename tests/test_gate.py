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
