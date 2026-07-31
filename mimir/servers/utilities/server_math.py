import ast
import os
import signal
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

import numpy as np
import operator
from mcp.server.fastmcp import FastMCP
from responses import err, ok
from capabilities import tool_caps

mcp = FastMCP(
    "MathServer",
    debug=False,
    log_level="ERROR",
)


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_expr(expression: str, safe_ns: dict) -> float:
    tree = ast.parse(expression, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed.")

        if isinstance(node, ast.Num):  # pragma: no cover
            return node.n

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_BINOPS:
                raise ValueError(f"Operator '{op_type.__name__}' is not allowed.")
            return _ALLOWED_BINOPS[op_type](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARYOPS:
                raise ValueError(f"Unary operator '{op_type.__name__}' is not allowed.")
            return _ALLOWED_UNARYOPS[op_type](_eval(node.operand))

        if isinstance(node, ast.Name):
            if node.id in safe_ns and isinstance(safe_ns[node.id], (int, float)):
                return safe_ns[node.id]
            raise ValueError(f"Unknown constant '{node.id}'.")

        if isinstance(node, ast.Call):
            if node.keywords:
                raise ValueError("Keyword arguments are not allowed.")
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed.")
            fn_name = node.func.id
            fn = safe_ns.get(fn_name)
            if fn is None or not callable(fn):
                raise ValueError(f"Function '{fn_name}' is not allowed.")
            args = [_eval(arg) for arg in node.args]
            return fn(*args)

        raise ValueError(f"Unsupported expression element: {type(node).__name__}.")

    return float(_eval(tree))


_EVAL_TIMEOUT = 5  # seconds

_SAFE_NP_NAMES = {
    # Trigonometric
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
    "degrees", "radians", "deg2rad", "rad2deg",
    # Exponential & logarithmic
    "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
    # Power & roots
    "sqrt", "cbrt", "power", "square",
    # Rounding
    "floor", "ceil", "trunc", "rint", "around",
    # Aggregation (scalar-safe)
    "sum", "prod", "mean", "std", "var", "min", "max",
    "cumsum", "cumprod",
    # Linear algebra basics
    "dot", "cross", "inner", "outer",
    # Misc math
    "abs", "absolute", "sign", "clip", "remainder", "mod", "fmod",
    "gcd", "lcm", "hypot",
    # Array creation (lightweight)
    "array", "linspace", "arange", "zeros", "ones",
    # Constants
    "pi", "e", "inf", "nan",
}


def _build_safe_namespace() -> dict:
    ns: dict = {}
    for name in _SAFE_NP_NAMES:
        val = getattr(np, name, None)
        if val is not None:
            ns[name] = val
    # Python built-in overrides
    ns.update({"abs": abs, "round": round, "int": int, "float": float, "pow": np.power})
    return ns


_SAFE_NS = _build_safe_namespace()


def _timeout_handler(signum, frame):
    raise TimeoutError("Expression evaluation timed out.")


@mcp.tool(**tool_caps(label="Evaluating {expression}"))
def evaluate(expression: str) -> dict:
    """Safely evaluate a mathematical expression string and return the result.

    PREFER THIS TOOL when you have a formula or multi-step expression, e.g.:
      "(3 + 4) * 2 / sqrt(49)"  or  "log(100, 10)"  or  "floor(3.7)"

    Supported operators: +  -  *  /  **  %  //
    Supported functions (curated subset of NumPy):
      sqrt, log, log10, log2, exp, sin, cos, tan, arcsin, arccos, arctan,
      floor, ceil, degrees, radians, gcd, hypot, ...
    Supported constants: pi, e, inf

    Returns {"status": "ok", "result": <number>, "expression": "<input> = result"}
         or {"status": "error", "error": "...", "hint": "..."}

    Args:
        expression: A mathematical expression string, e.g. "pi * 5**2".

    Examples:
        evaluate("2 ** 10")              -> 1024.0
        evaluate("sqrt(144)")            -> 12.0
        evaluate("log(1000, 10)")        -> 3.0
        evaluate("floor(3.7)")           -> 3.0
        evaluate("(100 - 32) * 5 / 9")  -> 37.77...
        evaluate("pi * 5 ** 2")          -> 78.53...
    """
    safe_ns = _SAFE_NS
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        signal.alarm(_EVAL_TIMEOUT)
        result = _safe_eval_expr(expression, safe_ns)
        signal.alarm(0)
        return ok({"result": float(result), "expression": f"{expression} = {result}"})
    except ZeroDivisionError:
        return err(
            "Division by zero in expression.",
            hint="Check for a / 0 or % 0 inside your expression.",
        )
    except TimeoutError:
        return err(
            f"Expression evaluation timed out after {_EVAL_TIMEOUT}s.",
            hint="Simplify the expression or break it into smaller steps.",
        )
    except Exception as e:
        return err(
            f"Could not evaluate '{expression}': {e}",
            hint="Only numeric operators and math module functions are supported.",
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@mcp.prompt()
def explain_operation(operation: str) -> str:
    return f"Explain the math operation: {operation}"


if __name__ == "__main__":
    mcp.run()
