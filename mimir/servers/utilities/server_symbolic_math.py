"""MCP Symbolic Math Server

Symbolic mathematics (also called symbolic computation or symbolic algebra) server.
This server provides tools for symbolic mathematics using SymPy, allowing users to
perform algebraic manipulations, calculus, equation solving, and more.

Features:
- Symbolic expression creation and manipulation
- Algebraic simplification and expansion
- Calculus operations (derivatives, integrals, limits)
- Equation solving (algebraic, differential)
- Matrix operations with symbolic elements
- Sequence and series operations
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from responses import err, ok
from capabilities import tool_caps

mcp = FastMCP(
    "SymbolicMathServer",
    debug=False,
    log_level="ERROR",
)


_SYMBOLIC_OPS = (
    "simplify", "expand", "factor", "differentiate", "integrate",
    "solve_equation", "compute_limit", "series_expansion",
    "create_matrix", "matrix_determinant", "solve_system",
)


def _equation_to_expr(sp, equation: str):
    """Parse 'expr1 = expr2' (or a bare expression) into a single SymPy expr = 0."""
    if "=" in equation:
        parts = equation.split("=")
        if len(parts) == 2:
            return sp.sympify(parts[0]) - sp.sympify(parts[1])
    return sp.sympify(equation)


def _parse_matrix(sp, matrix_str: str):
    """Parse '[[a, b], [c, d]]' into a SymPy Matrix and its (rows, cols).

    Each cell is run through ``sympify`` so numeric ('1', '2.5') and symbolic ('x')
    entries are both supported, returning a proper ``Expr``-valued matrix.
    """
    s = matrix_str.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()  # drop the outer brackets, leaving '[a, b], [c, d]'
    rows = []
    for part in s.split("]"):
        part = part.strip().lstrip(",").strip()
        if part.startswith("["):
            part = part[1:].strip()
        if not part:
            continue
        entries = [e.strip() for e in part.split(",") if e.strip()]
        rows.append([sp.sympify(e) for e in entries])
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 0
    return sp.Matrix(rows), n_rows, n_cols


@mcp.tool(**tool_caps(label="Symbolic {op}"))
def symbolic(
    op: str,
    expression: str = "",
    variable: str = "x",
    point: str = "0",
    n: int = 6,
    equation: str = "",
    matrix_str: str = "",
    equations: list = None,
    variables: list = None,
) -> dict:
    """Apply a single symbolic-mathematics (SymPy) operation, selected by ``op``.

    Operations (set ``op`` to one of these):
      simplify          -> simplify `expression`               -> {result, input}
      expand            -> expand products/powers in `expression`
      factor            -> factor `expression` into irreducibles
      differentiate     -> d/d`variable` of `expression`       -> {result, variable}
      integrate         -> indefinite integral of `expression` wrt `variable`
      solve_equation    -> solve `equation` (=0) for `variable` -> {result: [solutions]}
      compute_limit     -> limit of `expression` as `variable` -> `point` ('inf'/'-inf' allowed)
      series_expansion  -> series of `expression` around `point`, `n` terms
      create_matrix     -> build matrix from `matrix_str` '[[x,1],[0,x]]' -> {result, size}
      matrix_determinant-> determinant of `matrix_str`
      solve_system      -> solve `equations` (list) for `variables` (list)

    Args:
        op:         The operation to perform (see list above).
        expression: The symbolic expression for most ops, e.g. 'sin(x)**2 + cos(x)**2'.
        variable:   Variable to act on (default 'x').
        point:      Point for compute_limit / series_expansion ('inf', '-inf', or a number).
        n:          Number of series terms (series_expansion, default 6).
        equation:   Single equation for solve_equation, e.g. 'x**2 - 4' or 'x**2 = 4'.
        matrix_str: Matrix literal for create_matrix / matrix_determinant.
        equations:  List of equation strings for solve_system.
        variables:  List of variable names for solve_system.
    """
    try:
        import sympy as sp

        if op == "simplify":
            r = str(sp.simplify(sp.sympify(expression)))
            return ok({"result": r, "input": expression, "simplified": r})
        if op == "expand":
            r = str(sp.expand(sp.sympify(expression)))
            return ok({"result": r, "input": expression, "expanded": r})
        if op == "factor":
            r = str(sp.factor(sp.sympify(expression)))
            return ok({"result": r, "input": expression, "factored": r})
        if op == "differentiate":
            r = str(sp.diff(sp.sympify(expression), sp.Symbol(variable)))
            return ok({"result": r, "input": expression, "variable": variable, "derivative": r})
        if op == "integrate":
            r = str(sp.integrate(sp.sympify(expression), sp.Symbol(variable)))
            return ok({"result": r, "input": expression, "variable": variable, "integral": r})
        if op == "solve_equation":
            sols = sp.solve(_equation_to_expr(sp, equation), sp.Symbol(variable))
            r = [str(s) for s in sols]
            return ok({"result": r, "equation": equation, "variable": variable, "solutions": r})
        if op == "compute_limit":
            x = sp.Symbol(variable)
            expr = sp.sympify(expression)
            if point in ("inf", "infinity"):
                limit_val = sp.limit(expr, x, sp.oo)
            elif point in ("-inf", "-infinity"):
                limit_val = sp.limit(expr, x, -sp.oo)
            else:
                limit_val = sp.limit(expr, x, float(point))
            r = str(limit_val)
            return ok({"result": r, "expression": expression, "variable": variable,
                       "point": str(point), "limit": r})
        if op == "series_expansion":
            x = sp.Symbol(variable)
            pt = sp.oo if point in ("inf", "infinity") else (
                -sp.oo if point in ("-inf", "-infinity") else float(point))
            r = str(sp.series(sp.sympify(expression), x, pt, n))
            return ok({"result": r, "expression": expression, "variable": variable,
                       "point": str(point), "n": n, "series": r})
        if op == "create_matrix":
            sym_matrix, n_rows, n_cols = _parse_matrix(sp, matrix_str)
            r = str(sym_matrix)
            return ok({"result": r, "matrix_str": matrix_str,
                       "size": f"{n_rows}x{n_cols}", "matrix": r})
        if op == "matrix_determinant":
            sym_matrix, _, _ = _parse_matrix(sp, matrix_str)
            det = str(sym_matrix.det())
            return ok({"result": det, "matrix_str": matrix_str, "determinant": det})
        if op == "solve_system":
            sym_vars = [sp.Symbol(v) for v in (variables or [])]
            exprs = [_equation_to_expr(sp, eq) for eq in (equations or [])]
            r = str(sp.solve(exprs, sym_vars))
            return ok({"result": r, "equations": equations, "variables": variables, "solutions": r})
        return err(
            f"Unknown symbolic op '{op}'.",
            hint=f"Use one of: {', '.join(_SYMBOLIC_OPS)}.",
        )
    except Exception as e:
        return err(
            f"Could not perform symbolic op '{op}': {e}",
            hint="Ensure the expression/matrix uses valid SymPy syntax.",
        )


@mcp.prompt()
def explain_symbolic_operation(operation: str) -> str:
    """Explain a symbolic mathematical operation."""
    return f"Explain the symbolic math operation: {operation}"


if __name__ == "__main__":
    mcp.run()
