"""
Omission mutation engine (v2).

v1 (hunk deletion) was VACUOUS: it made omitted requirements literally absent
from the diff, so `grep` scored F1=1.000. See results/v1_pilot.txt.

Real coding-agent omissions are not absent -- they are *hollow*. Three classes,
ordered by how hard they are to catch with static analysis:

  ABSENT   the symbol was never written              (control; grep should win)
  UNWIRED  the symbol exists but nothing calls it    (Avnish's AegisOps bug)
  STUB     the symbol exists, is called, body hollow (semantic; expect to fail)

STUB is included precisely because deterministic analysis should do BADLY on it.
A benchmark you always win on is a benchmark that measures nothing.
"""

from __future__ import annotations

import ast
import re

TRIVIAL_BODY = ("pass", "...", "raise NotImplementedError")

_TREE_MEMO: dict[int, object] = {}


def parse_cached(src: str):
    k = hash(src)
    if k not in _TREE_MEMO:
        try:
            _TREE_MEMO[k] = ast.parse(src)
        except SyntaxError:
            _TREE_MEMO[k] = None
    return _TREE_MEMO[k]


_DEF_MEMO: dict[int, dict] = {}
_CALL_MEMO: dict[tuple, list] = {}


# --------------------------------------------------------------------------
# locating new symbols and their call sites
# --------------------------------------------------------------------------

def _defs(src: str) -> dict[str, tuple[int, int, int]]:
    """name -> (lineno, end_lineno, col_offset) for top-level & nested defs."""
    key = hash(src)
    if key in _DEF_MEMO:
        return _DEF_MEMO[key]
    out: dict[str, tuple[int, int, int]] = {}
    tree = parse_cached(src)
    if tree is None:
        _DEF_MEMO[key] = out
        return out
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.setdefault(n.name, (n.lineno, n.end_lineno or n.lineno, n.col_offset))
    _DEF_MEMO[key] = out
    return out


def call_lines(src: str, name: str, exclude: tuple[int, int] | None) -> list[int]:
    """1-indexed lines holding a call to `name` outside its own definition."""
    key = (hash(src), name, exclude)
    if key in _CALL_MEMO:
        return _CALL_MEMO[key]
    hits: list[int] = []
    tree = parse_cached(src)
    if tree is None:
        _CALL_MEMO[key] = hits
        return hits
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
        if fname != name:
            continue
        if exclude and exclude[0] <= n.lineno <= exclude[1]:
            continue
        hits.append(n.lineno)
    hits = sorted(set(hits))
    _CALL_MEMO[key] = hits
    return hits


_TARGETS_MEMO: dict[int, frozenset] = {}


def call_targets(src: str) -> frozenset[str]:
    """All names called anywhere in the file. Cheap pre-filter before call_lines."""
    k = hash(src)
    if k in _TARGETS_MEMO:
        return _TARGETS_MEMO[k]
    tree = parse_cached(src)
    out = set()
    if tree is not None:
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                nm = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
                if nm:
                    out.add(nm)
    _TARGETS_MEMO[k] = frozenset(out)
    return _TARGETS_MEMO[k]


def new_symbols(before: dict[str, list[str]], after: dict[str, list[str]]) -> list[tuple[str, str]]:
    """[(path, symbol)] for symbols defined in `after` but not in `before`."""
    out = []
    for path in after:
        db = _defs("\n".join(before.get(path, [])))
        da = _defs("\n".join(after[path]))
        for name in da:
            if name not in db and not name.startswith("_") and len(name) >= 4:
                out.append((path, name))
    return out


# --------------------------------------------------------------------------
# mutations -- each returns a new R_after or None if it cannot apply cleanly
# --------------------------------------------------------------------------

def _valid(files: dict[str, list[str]]) -> bool:
    for ls in files.values():
        try:
            ast.parse("\n".join(ls))
        except SyntaxError:
            return False
    return True


def mut_absent(after, path, name):
    """Delete the definition entirely."""
    src = "\n".join(after[path])
    d = _defs(src).get(name)
    if not d:
        return None
    lo, hi, _ = d
    out = {p: list(ls) for p, ls in after.items()}
    del out[path][lo - 1:hi]
    return out if _valid(out) else None


def mut_unwired(after, path, name):
    """Keep the definition. Delete every call site. Dead code, silently."""
    out = {p: list(ls) for p, ls in after.items()}
    hit_any = False
    for p in list(out):
        src = "\n".join(out[p])
        excl = _defs(src).get(name)
        excl = (excl[0], excl[1]) if excl else None
        hits = call_lines(src, name, excl)
        if not hits:
            continue
        for ln in sorted(hits, reverse=True):
            line = out[p][ln - 1]
            indent = len(line) - len(line.lstrip())
            # Only rewrite forms where removing the call is unambiguous and
            # leaves valid, natural-looking Python. Anything else is skipped:
            # a mutation that mangles the file is not a *silent* omission.
            #
            # v3: `return f(...)` was previously missed. Since most real call
            # sites are inside returns rather than standalone statements, that
            # omission is why UNWIRED was starved (n=16) in the v2 corpus.
            if re.match(rf"^\s*return\s+{re.escape(name)}\s*\(", line) or \
               re.match(rf"^\s*return\s+\w+\.{re.escape(name)}\s*\(", line):
                out[p][ln - 1] = " " * indent + "return None"
                hit_any = True
            elif re.match(rf"^\s*(?:\w+\s*=\s*)?{re.escape(name)}\s*\(", line) or \
                 re.match(rf"^\s*\w+\.{re.escape(name)}\s*\(", line):
                out[p][ln - 1] = " " * indent + "pass"
                hit_any = True
    if not hit_any:
        return None
    return out if _valid(out) else None


def mut_stub(after, path, name):
    """Keep the definition and its call sites. Hollow out the body."""
    src = "\n".join(after[path])
    d = _defs(src).get(name)
    if not d:
        return None
    lo, hi, col = d
    lines = list(after[path])
    # find end of the signature (line ending in ':')
    sig_end = lo - 1
    while sig_end < hi and not lines[sig_end].rstrip().endswith(":"):
        sig_end += 1
    if sig_end >= hi - 1:
        return None                                  # one-line body, nothing to hollow
    out = {p: list(ls) for p, ls in after.items()}
    out[path][sig_end + 1:hi] = [" " * (col + 4) + "raise NotImplementedError"]
    return out if _valid(out) else None


MUTATIONS = {
    "ABSENT": mut_absent,
    "UNWIRED": mut_unwired,
    "STUB": mut_stub,
}
