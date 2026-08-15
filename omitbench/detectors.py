"""
Detectors v3. Requirements are PATH-QUALIFIED: "src/click/core.py::ParamType".

Why qualification matters: v2 asked "is any symbol named X defined anywhere in
the repo?". Python codebases reuse names constantly (`main`, `run`, `parse`,
every `__init__`), so a same-named symbol in an unrelated file masked a real
deletion. That capped P1 recall on ABSENT at 0.43 when it should be ~1.0.
Qualification is also the more realistic spec: a plan says "add retry() to
client.py", not "add retry() somewhere".

Contract, enforced by tests/test_leakage.py:

    detect(spec, reqs, before, after, ctx) -> {req: "IMPLEMENTED"|"OMITTED"}

`before`/`after` are {path: [lines]}. The gold patch is never an argument.
"""

from __future__ import annotations

import ast

from omitbench import evidence as E
from omitbench import mutate as M

TRIVIAL = ("Pass", "Ellipsis", "NotImplementedError")


def split(req: str) -> tuple[str, str]:
    path, _, sym = req.rpartition("::")
    return path, sym


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def d_none(spec, reqs, before, after, ctx):
    """B0: flag nothing. Its accuracy is the base rate -- that is the point."""
    return {r: "IMPLEMENTED" for r in reqs}


def d_all(spec, reqs, before, after, ctx):
    """B1: flag everything. Recall 1.0, precision = base rate, MCC 0."""
    return {r: "OMITTED" for r in reqs}


def d_grep(spec, reqs, before, after, ctx):
    """
    B3: does the symbol name appear anywhere in the changed lines?
    The honest no-AST baseline. Fails on ABSENT because deleting a definition
    leaves its call sites behind, so the name is still in the diff -- grep
    cannot tell 'defined' from 'merely mentioned'.
    """
    blob = "\n".join(E._changed_lines(before, after))
    out = {}
    for r in reqs:
        _, sym = split(r)
        out[r] = "IMPLEMENTED" if sym in blob else "OMITTED"
    return out


def d_diffsize(spec, reqs, before, after, ctx):
    """B2: vacuity check. If diff size predicts omission, we measure artifacts."""
    n = len(E._changed_lines(before, after))
    small = n < ctx.get("median_changed_lines", 0)
    return {r: ("OMITTED" if small else "IMPLEMENTED") for r in reqs}


# --------------------------------------------------------------------------
# proposed: deterministic evidence, cumulative
# --------------------------------------------------------------------------
#
# RETIRED (TASKS.md T3 / ASSUMPTIONS.md #9): d_reachable (P2) and d_full (P3),
# and the public-API exemption below, are NOT in DETECTORS and are not
# scored or reported. They lost to B3 line-grep with a paired interval
# excluding zero (FPR ~0.84); a principled __all__/re-export/decorator
# exemption, decided from packaging semantics and measured exactly once,
# narrowed the gap (FPR down to ~0.60, P3 went from losing to statistically
# tied) but did not clear the pre-agreed bar of beating B3 outright. Per
# that pre-agreed protocol the rule was not iterated a second time. Kept
# here, unregistered, as the record of a real attempt and why it fell
# short -- see ASSUMPTIONS.md #9 for the full before/after numbers.
# --------------------------------------------------------------------------

def _defined_in(after, path, sym):
    return M._defs("\n".join(after.get(path, []))).get(sym)


def d_defined(spec, reqs, before, after, ctx):
    """E1: is the symbol defined in the file the requirement names?"""
    out = {}
    for r in reqs:
        path, sym = split(r)
        out[r] = "IMPLEMENTED" if _defined_in(after, path, sym) else "OMITTED"
    return out


# --------------------------------------------------------------------------
# T3 (TASKS.md): public-API exemption for the reachability check.
#
# A symbol nothing calls internally is not necessarily omitted -- it may be
# the library's public surface, which by definition is called from OUTSIDE
# the repo, not from within it. Reachability alone condemns every public
# API symbol, which is why P2/P3 measured FPR ~0.84-0.85 (ASSUMPTIONS.md
# #6). Decided from Python packaging semantics, once, before measuring --
# not tuned afterward (ASSUMPTIONS.md #9 has the before/after numbers and
# the UNWIRED-recall trade-off this predictably costs).
#
# All three checks are pure functions of `after` -- no gold patch, no
# mutation label, same as every other detector input.
# --------------------------------------------------------------------------

def _has_all_export(src: str, sym: str) -> bool:
    """Module-level `__all__ = [...]` (list/tuple/set of string constants,
    direct assignment or augmented assignment) in this module itself,
    containing `sym`. Does not follow `.append()`/`.extend()` calls -- those
    are rare enough next to the direct-assignment form that chasing them
    would be tuning the rule to specific corpus files, not packaging
    semantics in general.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in tree.body:  # __all__ is a module-level convention only
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AugAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            names = {e.value for e in value.elts
                     if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            if sym in names:
                return True
    return False


def _module_basename(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".py")


def _reexported_in_init(after, path: str, sym: str) -> bool:
    """`dirname(path)/__init__.py` imports `sym` from `path` by name, or
    star-imports from it (Python semantics: `import *` re-exports every
    non-underscore top-level name, and `sym` not starting with `_` is
    exactly what makes it eligible here). Same package directory only --
    no ancestor-package walk, so the rule stays auditable against a single
    file rather than a resolved import graph.
    """
    d = path.rsplit("/", 1)[0] if "/" in path else ""
    init_path = f"{d}/__init__.py" if d else "__init__.py"
    if init_path == path:
        return False
    init_src = "\n".join(after.get(init_path, []))
    if not init_src.strip():
        return False
    try:
        tree = ast.parse(init_src)
    except SyntaxError:
        return False
    basename = _module_basename(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.module.split(".")[-1] != basename:
            continue
        for alias in node.names:
            if alias.name in ("*", sym):
                return True
    return False


def _decorated_public(src: str, sym: str) -> bool:
    """`sym`'s own definition carries >=1 decorator and its name isn't
    underscore-prefixed. Deliberately NOT a named allowlist (@app.route,
    @click.command, ...) -- enumerating specific decorators would tune this
    rule to this corpus's frameworks. The general, corpus-agnostic
    justification: applying a decorator is itself a reference to the
    symbol from outside its own body, independent of which decorator it is.
    """
    if sym.startswith("_"):
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and n.name == sym:
            return bool(n.decorator_list)
    return False


def _is_exempt_public_api(after, path: str, sym: str) -> bool:
    src = "\n".join(after.get(path, []))
    return (_has_all_export(src, sym)
            or _reexported_in_init(after, path, sym)
            or _decorated_public(src, sym))


def d_reachable(spec, reqs, before, after, ctx):
    """E1+E3: defined AND (called from outside its own body OR exempt as
    public API, see _is_exempt_public_api above). Catches UNWIRED, but no
    longer condemns every symbol a library exports for external callers --
    see ASSUMPTIONS.md #9 for the FPR fix and the UNWIRED-recall cost it
    predictably trades against.
    """
    out = {}
    for r in reqs:
        path, sym = split(r)
        d = _defined_in(after, path, sym)
        if d is None:
            out[r] = "OMITTED"
            continue
        excl = (d[0], d[1])
        hit = False
        for p2, ls in after.items():
            src = "\n".join(ls)
            if sym not in M.call_targets(src):
                continue
            if M.call_lines(src, sym, excl if p2 == path else None):
                hit = True
                break
        if not hit:
            hit = _is_exempt_public_api(after, path, sym)
        out[r] = "IMPLEMENTED" if hit else "OMITTED"
    return out


def _hollow(after, path, sym) -> bool:
    """Body is only pass / ... / raise NotImplementedError / docstring."""
    src = "\n".join(after.get(path, []))
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if n.name != sym:
            continue
        body = [b for b in n.body
                if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                        and isinstance(b.value.value, str))]
        if not body:
            return True
        for b in body:
            if isinstance(b, ast.Pass):
                continue
            if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) \
                    and b.value.value is Ellipsis:
                continue
            if isinstance(b, ast.Raise):
                exc = b.exc
                nm = None
                if isinstance(exc, ast.Name):
                    nm = exc.id
                elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                    nm = exc.func.id
                if nm == "NotImplementedError":
                    continue
            return False
        return True
    return False


def d_full(spec, reqs, before, after, ctx):
    """E1+E3+E4: defined, reachable, and body is not a stub."""
    base = d_reachable(spec, reqs, before, after, ctx)
    out = {}
    for r in reqs:
        if base[r] == "OMITTED":
            out[r] = "OMITTED"
            continue
        path, sym = split(r)
        out[r] = "OMITTED" if _hollow(after, path, sym) else "IMPLEMENTED"
    return out


DETECTORS = {
    "B0 flag-nothing": d_none,
    "B1 flag-everything": d_all,
    "B3 line-grep (no AST)": d_grep,
    "P1 defined": d_defined,
    # P2 "defined+reachable" and P3 "defined+reachable+body" are retired --
    # see the RETIRED banner above d_reachable and ASSUMPTIONS.md #9.
}
