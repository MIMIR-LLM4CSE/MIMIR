"""LaTeX display copy of a calculation, for the client to render under the tool row.

A math tool puts a ``latex`` string in its success payload: the calculation and its
result, as one display-math body (no ``$`` delimiters). The client detects the field
by shape and the chat renders it with KaTeX. Building it must never fail a call, so
every entry point here returns ``None`` rather than raise.

:func:`numeric_latex` converts the restricted Python AST the math server accepts.
It is not SymPy: SymPy canonicalises while it parses — ``a / b`` becomes
``a · b⁻¹`` and ``floor(3.7)`` becomes ``3`` — and the point is to show the
calculation as it was asked.
"""

from __future__ import annotations

import ast
import math
from typing import Callable

# Precedence, loosest first. A child binding looser than its parent is parenthesised.
_ADD, _MUL, _UNARY, _POW, _ATOM = 1, 2, 3, 4, 5

_CONSTANTS = {"pi": r"\pi", "e": "e", "inf": r"\infty", "nan": r"\mathrm{NaN}"}

# One-argument functions LaTeX has a command for, as `\cmd\left(x\right)`.
_NAMED = {
    "sin": r"\sin", "cos": r"\cos", "tan": r"\tan",
    "arcsin": r"\arcsin", "arccos": r"\arccos", "arctan": r"\arctan",
    "sinh": r"\sinh", "cosh": r"\cosh", "tanh": r"\tanh",
    "arcsinh": r"\operatorname{arsinh}", "arccosh": r"\operatorname{arcosh}",
    "arctanh": r"\operatorname{artanh}",
    "log": r"\ln", "log10": r"\log_{10}", "log2": r"\log_{2}",
}

# Functions with a notation of their own, built from their rendered arguments.
_SPECIAL: dict[str, Callable[[list[str]], str]] = {
    "sqrt": lambda a: rf"\sqrt{{{a[0]}}}",
    "cbrt": lambda a: rf"\sqrt[3]{{{a[0]}}}",
    "exp": lambda a: rf"e^{{{a[0]}}}",
    "abs": lambda a: rf"\left|{a[0]}\right|",
    "absolute": lambda a: rf"\left|{a[0]}\right|",
    "floor": lambda a: rf"\left\lfloor {a[0]} \right\rfloor",
    "ceil": lambda a: rf"\left\lceil {a[0]} \right\rceil",
    "square": lambda a: rf"\left({a[0]}\right)^{{2}}",
    "power": lambda a: rf"\left({a[0]}\right)^{{{a[1]}}}",
    "pow": lambda a: rf"\left({a[0]}\right)^{{{a[1]}}}",
    "degrees": lambda a: rf"\left({a[0]}\right)\cdot\frac{{180}}{{\pi}}",
    "radians": lambda a: rf"\left({a[0]}\right)\cdot\frac{{\pi}}{{180}}",
}


def number_latex(value: float, digits: int = 12) -> str:
    """A number as a reader writes it: integers bare, large or tiny ones in ×10ⁿ."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return r"\mathrm{NaN}"
        if math.isinf(value):
            return r"\infty" if value > 0 else r"-\infty"
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
    text = format(value, f".{digits}g")
    if "e" in text:
        mantissa, exponent = text.split("e")
        return rf"{mantissa} \times 10^{{{int(exponent)}}}"
    return text


def _render(node: ast.AST) -> tuple[str, int]:
    """(LaTeX, precedence) of one node of the restricted expression grammar."""
    if isinstance(node, ast.Expression):
        return _render(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        text = number_latex(node.value)
        # `1 \times 10^{20}` is itself a product, and binds like one.
        return text, _MUL if r"\times" in text else _ATOM

    if isinstance(node, ast.Name):
        return _CONSTANTS.get(node.id, rf"\mathrm{{{node.id}}}"), _ATOM

    if isinstance(node, ast.UnaryOp):
        body, prec = _render(node.operand)
        if prec < _POW:
            body = rf"\left({body}\right)"
        sign = "-" if isinstance(node.op, ast.USub) else "+"
        return f"{sign}{body}", _UNARY

    if isinstance(node, ast.BinOp):
        op = type(node.op)
        left, lp = _render(node.left)
        right, rp = _render(node.right)
        # A fraction bar and an exponent group their operands visually.
        if op is ast.Div:
            return rf"\frac{{{left}}}{{{right}}}", _ATOM
        if op is ast.FloorDiv:
            return rf"\left\lfloor \frac{{{left}}}{{{right}}} \right\rfloor", _ATOM
        if op is ast.Pow:
            if lp <= _POW:
                left = rf"\left({left}\right)"
            return f"{{{left}}}^{{{right}}}", _POW
        if op in (ast.Add, ast.Sub):
            if op is ast.Sub and rp <= _ADD:
                right = rf"\left({right}\right)"
            # `a + -b` reads as a typo; the unary minus is kept but bracketed.
            elif rp == _UNARY:
                right = rf"\left({right}\right)"
            sym = "+" if op is ast.Add else "-"
            return f"{left} {sym} {right}", _ADD
        # Mult / Mod
        if lp < _MUL:
            left = rf"\left({left}\right)"
        if rp <= _MUL or rp == _UNARY:
            right = rf"\left({right}\right)"
        sym = r"\cdot" if op is ast.Mult else r"\bmod"
        return f"{left} {sym} {right}", _MUL

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        args = [_render(a)[0] for a in node.args]
        special = _SPECIAL.get(name)
        if special is not None:
            try:
                return special(args), _ATOM
            except IndexError:
                pass
        cmd = _NAMED.get(name, rf"\operatorname{{{name}}}")
        return rf"{cmd}\left({', '.join(args)}\right)", _ATOM

    raise ValueError(f"no LaTeX form for {type(node).__name__}")


def numeric_latex(expression: str, result: float) -> str | None:
    """``<expression> = <result>`` for an expression the math server accepted."""
    try:
        body, _ = _render(ast.parse(expression, mode="eval"))
    except Exception:
        return None
    shown = number_latex(result)
    # A result cut to 12 digits is an approximation, and the sign says so.
    exact = not isinstance(result, float) or not math.isfinite(result) or (
        float(format(result, ".12g")) == result)
    relation = "=" if exact else r"\approx"
    return f"{body} {relation} {shown}"


def sympy_latex(build: Callable[[], str]) -> str | None:
    """Run a LaTeX builder over SymPy objects; a printer failure yields no display."""
    try:
        return build()
    except Exception:
        return None
