"""
T6 CI gate: comment on a PR when a symbol named in the PR's own structured
requirements checklist is missing from the diff.

Uses ONLY P1 (`omitbench.detectors.d_defined`) -- see TASKS.md T6 decision 1.
No LLM, no network call to source requirements -- see decision 5 and
`ASSUMPTIONS.md`'s T6 section: T5 measured that free-text extraction
collapses P1 to MCC -0.935. Requirements must be spelled out explicitly in
the PR/issue body as a markdown checklist; this module never guesses one.

Contract note: there is no gold patch anywhere in this flow (a live PR has
no reference solution to leak), so the leakage discipline every other
detector file follows is automatically satisfied here. `score_pr` below
still only ever calls `d_defined(spec, reqs, before, after, ctx)` -- see
tests/test_gate.py's signature-lock test.
"""

from __future__ import annotations

import re
import subprocess

from omitbench import detectors as D

_CHECKLIST_LINE = re.compile(r"^\s*-\s*\[[ xX]\]\s*(.+?)\s*$", re.MULTILINE)
_QUALIFIED = re.compile(r"^`?([\w./\-]+\.py)::([A-Za-z_]\w*)`?$")
_SYMBOL_IN_PATH = re.compile(
    r"^`?([A-Za-z_]\w*)(?:\(\))?`?\s+in\s+`?([\w./\-]+\.py)`?$"
)


def _match_item(text: str) -> str | None:
    """Returns a path::symbol requirement string, or None if `text` isn't
    one of the two recognized checklist-item shapes."""
    m = _QUALIFIED.match(text)
    if m:
        path, sym = m.group(1), m.group(2)
        return f"{path}::{sym}"
    m = _SYMBOL_IN_PATH.match(text)
    if m:
        sym, path = m.group(1), m.group(2)
        return f"{path}::{sym}"
    return None


def parse_requirements(pr_body: str) -> list[str]:
    """path::symbol requirements from markdown checklist items in a PR/issue
    body. Checkbox state ([ ] vs [x]) is ignored by design -- a plan item is
    a plan item whether or not the author ticked it; the gate re-derives
    real status from the diff, it does not trust the checkbox. Two
    supported item shapes:

        - [ ] retry_with_backoff in src/client.py
        - [ ] src/client.py::retry_with_backoff

    Both may be backtick-quoted. Order-preserving; duplicates are not
    deduplicated (a plan listing the same item twice is the author's
    business, not this parser's)."""
    out = []
    for text in _CHECKLIST_LINE.findall(pr_body):
        req = _match_item(text)
        if req is not None:
            out.append(req)
    return out


def unrecognized_items(pr_body: str) -> list[str]:
    """Checklist lines that did not match either requirement shape, in
    order. Lets the gate's comment say '(N items were not in a recognized
    format)' instead of silently doing nothing when a plan uses different
    wording than either supported shape."""
    return [text for text in _CHECKLIST_LINE.findall(pr_body)
            if _match_item(text) is None]


def fetch_after_snapshot(repo_dir: str, head_sha: str,
                          paths: list[str]) -> dict[str, list[str]]:
    """`git show head_sha:path` for exactly the paths named in the parsed
    requirements -- not the whole repo (corpus.py's snapshot_repo pulls
    everything because P2/P3's reachability check needed the full tree;
    P1 only ever looks at one path per requirement, see
    detectors.py:_defined_in). A path missing at head_sha (deleted, or
    never existed -- e.g. a typo in the checklist) maps to [] so
    d_defined's lookup returns None and the requirement reads OMITTED,
    same as every other 'not defined' case."""
    out: dict[str, list[str]] = {}
    for path in paths:
        proc = subprocess.run(
            ["git", "-C", repo_dir, "show", f"{head_sha}:{path}"],
            capture_output=True, text=True,
        )
        out[path] = proc.stdout.split("\n") if proc.returncode == 0 else []
    return out


def score_pr(pr_body: str, repo_dir: str, head_sha: str) -> dict:
    """End-to-end: parse the checklist, fetch only the referenced files at
    head_sha, run P1. `before` is deliberately {} -- d_defined never reads
    it (omitbench/detectors.py:89-95), and there is no base-branch snapshot
    to build for a 'symbol present in this diff?' check. If a second
    detector is ever added to this gate, it will need real `before` data;
    this stays {} only because P1 is the sole detector per decision 1."""
    reqs = parse_requirements(pr_body)
    paths = sorted({r.split("::", 1)[0] for r in reqs})
    after = fetch_after_snapshot(repo_dir, head_sha, paths)
    verdicts = D.d_defined(pr_body, reqs, {}, after, {})
    return {
        "omitted": [r for r in reqs if verdicts[r] == "OMITTED"],
        "implemented": [r for r in reqs if verdicts[r] == "IMPLEMENTED"],
        "unrecognized": unrecognized_items(pr_body),
    }
