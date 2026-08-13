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

def _defined_in(after, path, sym):
    return M._defs("\n".join(after.get(path, []))).get(sym)


def d_defined(spec, reqs, before, after, ctx):
    """E1: is the symbol defined in the file the requirement names?"""
    out = {}
    for r in reqs:
        path, sym = split(r)
        out[r] = "IMPLEMENTED" if _defined_in(after, path, sym) else "OMITTED"
    return out


def d_reachable(spec, reqs, before, after, ctx):
    """E1+E3: defined AND called from outside its own body. Catches UNWIRED."""
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
    "P2 defined+reachable": d_reachable,
    "P3 defined+reachable+body": d_full,
}
