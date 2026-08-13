"""
OmitBench corpus construction.

Builds labeled omission instances from real git history.

An instance is:
    spec_text   : commit message  (the "plan" the agent was given)
    R_before    : {path: source}  (repo state at parent commit)
    R_after     : {path: source}  (R_before + a SUBSET of the gold patch hunks)
    labels      : {requirement -> IMPLEMENTED | OMITTED}

CRITICAL INVARIANT
------------------
The gold patch is used ONLY to (a) construct R_after and (b) derive labels.
It is NEVER passed to a detector. See tests/test_no_leak.py.
"""

from __future__ import annotations

import ast
import hashlib
import random
import re
import subprocess
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# git plumbing
# --------------------------------------------------------------------------

def _git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True, text=True, errors="replace",
    ).stdout


def snapshot_repo(repo: str, commit: str) -> dict[str, list[str]]:
    """
    ALL .py files at `commit`. Call-graph analysis over a 2-file slice is blind:
    a symbol looks unreachable simply because its caller lives elsewhere.
    Pilot v2a fix -- see results/v2_pilot.txt (89% FPR before this).
    """
    import tarfile, io
    blob = subprocess.run(["git", "-C", repo, "archive", "--format=tar", commit],
                          capture_output=True).stdout
    files: dict[str, list[str]] = {}
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith(".py") and m.size < 400_000:
                try:
                    files[m.name] = tf.extractfile(m).read().decode("utf-8", "replace").split("\n")
                except Exception:
                    pass
    return files


def list_commits(repo: str, limit: int = 800) -> list[str]:
    out = _git(repo, "log", "--no-merges", "--format=%H", f"-{limit}")
    return [l for l in out.splitlines() if l]


# --------------------------------------------------------------------------
# hunk model
# --------------------------------------------------------------------------

@dataclass
class Hunk:
    path: str
    old_start: int          # 1-indexed line in R_before
    old_count: int
    new_lines: list[str]    # replacement lines (no trailing newline)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.added) + len(self.removed)


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(diff_text: str) -> list[Hunk]:
    """Parse a unified diff into hunks. Python files only. Skips renames/binary."""
    hunks: list[Hunk] = []
    path = None
    cur: Hunk | None = None

    for line in diff_text.split("\n"):
        if line.startswith("diff --git "):
            cur = None
            path = None
            m = re.match(r"diff --git a/(.+) b/(.+)$", line)
            if m and m.group(1) == m.group(2) and m.group(1).endswith(".py"):
                path = m.group(1)
            continue
        if path is None:
            continue
        if line.startswith("@@"):
            m = _HUNK_RE.match(line)
            if not m:
                cur = None
                continue
            cur = Hunk(
                path=path,
                old_start=int(m.group(1)),
                old_count=int(m.group(2)) if m.group(2) is not None else 1,
                new_lines=[],
            )
            hunks.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("\\"):          # "\ No newline at end of file"
            continue
        if line.startswith(" "):
            cur.new_lines.append(line[1:])
        elif line.startswith("+"):
            cur.new_lines.append(line[1:])
            cur.added.append(line[1:])
        elif line.startswith("-"):
            cur.removed.append(line[1:])
        else:
            cur = None                      # left the hunk body
    return [h for h in hunks if h.added or h.removed]


def apply_hunks(before: dict[str, list[str]], hunks: list[Hunk]) -> dict[str, list[str]]:
    """
    Apply a SUBSET of hunks. Bottom-up per file so earlier line numbers stay valid.
    Exact splice -- no fuzz, no `git apply`, no offset guessing.
    """
    after = {p: list(ls) for p, ls in before.items()}
    by_file: dict[str, list[Hunk]] = {}
    for h in hunks:
        by_file.setdefault(h.path, []).append(h)

    for path, hs in by_file.items():
        lines = after.setdefault(path, [])
        for h in sorted(hs, key=lambda x: x.old_start, reverse=True):
            i = h.old_start - 1
            lines[i:i + h.old_count] = h.new_lines
    return after


# --------------------------------------------------------------------------
# requirement extraction (v0: identifier-level, LLM-free)
# --------------------------------------------------------------------------

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BACKTICK = re.compile(r"`([^`]+)`")

# words that look like identifiers but never are requirements
_STOP = {
    "the", "and", "for", "with", "this", "that", "from", "when", "not", "add",
    "fix", "use", "now", "all", "but", "can", "has", "was", "are", "you", "its",
    "python", "test", "tests", "docs", "doc", "changelog", "release", "version",
    "http", "https", "github", "com", "issue", "pr", "closes", "fixes", "ref",
    "signed", "off", "by", "co", "authored", "merge", "branch", "master", "main",
    "self", "def", "class", "return", "import", "true", "false", "none",
}


def extract_requirements(spec: str) -> set[str]:
    """
    Candidate requirements = identifier-like tokens named in the spec text.
    Backticked tokens are strongest; snake_case / CamelCase also count.
    """
    cands: set[str] = set()

    for m in _BACKTICK.finditer(spec):
        for ident in _IDENT.findall(m.group(1)):
            cands.add(ident)

    for tok in _IDENT.findall(spec):
        if "_" in tok and not tok.startswith("_"):
            cands.add(tok)
        elif re.match(r"^[A-Z][a-z]+[A-Z]", tok):     # CamelCase
            cands.add(tok)

    return {
        c for c in cands
        if len(c) >= 4 and c.lower() not in _STOP and not c.isdigit()
    }


def hunk_mentions(h: Hunk, req: str) -> bool:
    body = "\n".join(h.added + h.removed)
    return re.search(rf"\b{re.escape(req)}\b", body) is not None


# --------------------------------------------------------------------------
# instance construction
# --------------------------------------------------------------------------

@dataclass
class Instance:
    iid: str
    repo: str
    commit: str
    spec: str
    before: dict[str, list[str]]
    after: dict[str, list[str]]
    requirements: list[str]
    labels: dict[str, str]              # req -> "IMPLEMENTED" | "OMITTED"
    n_hunks_total: int
    n_hunks_kept: int


def build_instance(repo: str, commit: str, rng: random.Random,
                   drop_frac_range=(0.2, 0.6)) -> Instance | None:
    parent = _git(repo, "rev-parse", f"{commit}^").strip()
    if not parent:
        return None

    spec = _git(repo, "log", "-1", "--format=%B", commit).strip()
    if len(spec) < 20:
        return None

    diff = _git(repo, "diff", "--unified=3", parent, commit, "--", "*.py")
    hunks = parse_patch(diff)
    if not (3 <= len(hunks) <= 40):
        return None

    # ORACLE REQUIREMENTS: spec-derived identifiers that the true solution
    # actually touched. Assumes a perfect extractor -- see PILOT_ASSUMPTIONS.md.
    raw_reqs = extract_requirements(spec)
    reqs = sorted(r for r in raw_reqs if any(hunk_mentions(h, r) for h in hunks))
    if not (1 <= len(reqs) <= 12):
        return None

    # R_before: only files the gold patch touches can ever differ
    paths = sorted({h.path for h in hunks})
    before: dict[str, list[str]] = {}
    for p in paths:
        src = _git(repo, "show", f"{parent}:{p}")
        if not src:
            return None
        before[p] = src.split("\n")

    # sanity: full patch must apply cleanly and parse as valid Python
    full_after = apply_hunks(before, hunks)
    for p, ls in full_after.items():
        try:
            ast.parse("\n".join(ls))
        except SyntaxError:
            return None

    # inject omission: drop a random subset of hunks
    n_drop = max(1, int(round(len(hunks) * rng.uniform(*drop_frac_range))))
    n_drop = min(n_drop, len(hunks) - 1)
    dropped = set(rng.sample(range(len(hunks)), n_drop))
    kept = [h for i, h in enumerate(hunks) if i not in dropped]

    after = apply_hunks(before, kept)
    for p, ls in after.items():
        try:
            ast.parse("\n".join(ls))
        except SyntaxError:
            return None                      # partial patch broke syntax -> unusable

    # labels: a requirement is OMITTED iff NO kept hunk implements it
    labels = {
        r: ("IMPLEMENTED" if any(hunk_mentions(h, r) for h in kept) else "OMITTED")
        for r in reqs
    }

    iid = hashlib.sha1(f"{repo}:{commit}".encode()).hexdigest()[:12]
    return Instance(
        iid=iid, repo=repo.rsplit("/", 1)[-1], commit=commit, spec=spec,
        before=before, after=after, requirements=reqs, labels=labels,
        n_hunks_total=len(hunks), n_hunks_kept=len(kept),
    )
