"""
Deterministic evidence engine + baselines.

Every detector here has the signature:

    detect(spec, requirements, before, after, ctx) -> {req: "IMPLEMENTED"|"OMITTED"}

`before` / `after` are {path: [lines]}. The gold patch is NOT an argument.
That is the whole point -- if a detector could see the reference diff,
"find the deleted hunks" is trivial and every number below is meaningless.
"""

from __future__ import annotations

import ast
import difflib
import re


# --------------------------------------------------------------------------
# AST symbol table
# --------------------------------------------------------------------------

def _idents(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.keyword) and n.arg:
            out.add(n.arg)
        elif isinstance(n, ast.alias):
            out.add((n.asname or n.name).split(".")[0])
    return out


def symbol_table(src: str) -> dict[str, tuple[str, frozenset[str]]]:
    """qualified name -> (structural hash of body, identifiers used in body)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}

    table: dict[str, tuple[str, frozenset[str]]] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qname = f"{prefix}{child.name}"
                table[qname] = (ast.dump(child), frozenset(_idents(child)))
                visit(child, qname + ".")

    visit(tree, "")

    top = [n for n in tree.body
           if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    mod = ast.Module(body=top, type_ignores=[])
    table["<module>"] = (ast.dump(mod), frozenset(_idents(mod)))
    return table


def _changed_symbols(before: dict[str, list[str]], after: dict[str, list[str]]):
    """Symbols whose structure changed, and the identifiers implicated."""
    changed_names: set[str] = set()
    implicated: set[str] = set()

    for path in set(before) | set(after):
        tb = symbol_table("\n".join(before.get(path, [])))
        ta = symbol_table("\n".join(after.get(path, [])))
        for q in set(tb) | set(ta):
            hb, ib = tb.get(q, (None, frozenset()))
            ha, ia = ta.get(q, (None, frozenset()))
            if hb == ha:
                continue
            changed_names.add(q.split(".")[-1])
            implicated |= (ia ^ ib)          # identifiers that appeared or vanished
            implicated |= ia                 # identifiers present in the new body
    return changed_names, implicated


def _changed_lines(before: dict[str, list[str]], after: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for path in set(before) | set(after):
        b, a = before.get(path, []), after.get(path, [])
        for line in difflib.unified_diff(b, a, n=0, lineterm=""):
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                out.append(line[1:])
    return out


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------

def d_ast(spec, requirements, before, after, ctx):
    """E1+E2: AST symbol structural diff. The proposed method (v0)."""
    changed_names, implicated = _changed_symbols(before, after)
    evidence = changed_names | implicated
    return {r: ("IMPLEMENTED" if r in evidence else "OMITTED") for r in requirements}


def d_token(spec, requirements, before, after, ctx):
    """B3: line-level grep. No AST. Tests whether AST buys anything at all."""
    blob = "\n".join(_changed_lines(before, after))
    return {
        r: ("IMPLEMENTED" if re.search(rf"\b{re.escape(r)}\b", blob) else "OMITTED")
        for r in requirements
    }


def d_all_implemented(spec, requirements, before, after, ctx):
    """B0: flag nothing. Establishes why accuracy is a banned headline metric."""
    return {r: "IMPLEMENTED" for r in requirements}


def d_all_omitted(spec, requirements, before, after, ctx):
    """B1: flag everything. Recall 1.0, precision = base rate."""
    return {r: "OMITTED" for r in requirements}


def d_diffsize(spec, requirements, before, after, ctx):
    """
    B2: THE VACUITY CHECK.
    If a small diff means 'stuff was omitted', the benchmark is measuring
    injection artifacts, not omission detection. This baseline must be weak.
    """
    n = len(_changed_lines(before, after))
    small = n < ctx["median_changed_lines"]
    return {r: ("OMITTED" if small else "IMPLEMENTED") for r in requirements}


DETECTORS = {
    "B0 flag-nothing": d_all_implemented,
    "B1 flag-everything": d_all_omitted,
    "B2 diff-size heuristic": d_diffsize,
    "B3 line-grep (no AST)": d_token,
    "P  AST symbol diff": d_ast,
}


# --------------------------------------------------------------------------
# scoring -- OMITTED is the positive class
# --------------------------------------------------------------------------

def score(preds: list[str], golds: list[str]) -> dict[str, float]:
    tp = sum(p == "OMITTED" and g == "OMITTED" for p, g in zip(preds, golds))
    fp = sum(p == "OMITTED" and g == "IMPLEMENTED" for p, g in zip(preds, golds))
    fn = sum(p == "IMPLEMENTED" and g == "OMITTED" for p, g in zip(preds, golds))
    tn = sum(p == "IMPLEMENTED" and g == "IMPLEMENTED" for p, g in zip(preds, golds))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "accuracy": (tp + tn) / max(1, len(golds)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
