"""LaTeX display copy of a math tool's result.

A result is "math-shaped" when its success payload carries a ``latex`` string: the
calculation and its result, as one display-math body. Detection is by payload shape,
never by tool name, so any tool that computes something can have its work typeset
in the UI by adding the field. The copy rides on the ``tool_result`` event as
``math: {latex}``; the model still reads the full payload through history.
"""

from __future__ import annotations

from typing import Any

from .formatter import parse_tool_payload

# A formula past this length is no longer something to read at a glance, and KaTeX
# on a huge matrix is slow; the row then keeps its plain summary.
_MAX_LATEX_CHARS = 4_000


def extract_math_preview(result_text: str) -> dict[str, Any] | None:
    """Build the ``math`` display object, or ``None`` when the result has none."""
    payload = parse_tool_payload(result_text)
    if payload is None or payload.get("status") == "error":
        return None
    latex = payload.get("latex")
    if not isinstance(latex, str) or not latex.strip():
        return None
    if len(latex) > _MAX_LATEX_CHARS:
        return None
    return {"latex": latex.strip()}
