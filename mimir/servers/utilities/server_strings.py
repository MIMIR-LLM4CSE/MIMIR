import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps
from responses import err, ok

mcp = FastMCP(
    "StringServer",
    debug=False,
    log_level="ERROR",
)


_STRING_OPS = (
    "reverse", "uppercase", "lowercase", "length", "strip", "replace",
    "split", "contains", "count_occurrences", "starts_with", "ends_with",
    "title_case",
)


@mcp.tool(**tool_caps(label="String {op}"))
def string_op(
    op: str,
    text: str,
    old: str = "",
    new: str = "",
    count: int = -1,
    sep: str = None,
    maxsplit: int = -1,
    chars: str = None,
    substring: str = "",
    case_sensitive: bool = True,
    prefix: str = "",
    suffix: str = "",
) -> dict:
    """Apply a single string operation, selected by ``op``.

    Operations (set ``op`` to one of these):
      reverse           -> {"result": <reversed string>}
      uppercase         -> {"result": <UPPERCASED>}
      lowercase         -> {"result": <lowercased>}
      length            -> {"result": <int length>}
      strip             -> {"result": <stripped>}   (uses `chars`, else whitespace)
      replace           -> {"result": <modified>, "replacements": <n>}  (uses `old`,`new`,`count`)
      split             -> {"result": [<parts>], "count": <n>}  (uses `sep`,`maxsplit`)
      contains          -> {"result": bool, "positions": [<start indices>]}  (uses `substring`,`case_sensitive`)
      count_occurrences -> {"result": <count>}  (uses `substring`,`case_sensitive`)
      starts_with       -> {"result": bool}  (uses `prefix`)
      ends_with         -> {"result": bool}  (uses `suffix`)
      title_case        -> {"result": <Title Cased>}

    Args:
        op:             The operation to perform (see list above).
        text:           The input string every operation acts on.
        old, new, count: For ``replace`` (count=-1 means all).
        sep, maxsplit:  For ``split`` (sep=None splits on whitespace, maxsplit=-1 unlimited).
        chars:          For ``strip`` (None strips whitespace).
        substring, case_sensitive: For ``contains`` / ``count_occurrences``.
        prefix:         For ``starts_with``.
        suffix:         For ``ends_with``.
    """
    if op == "reverse":
        return ok({"result": text[::-1]})
    if op == "uppercase":
        return ok({"result": text.upper()})
    if op == "lowercase":
        return ok({"result": text.lower()})
    if op == "length":
        return ok({"result": len(text)})
    if op == "strip":
        return ok({"result": text.strip(chars)})
    if op == "replace":
        result = text.replace(old, new) if count == -1 else text.replace(old, new, count)
        replacements = text.count(old) if count == -1 else min(text.count(old), count)
        return ok({"result": result, "replacements": replacements})
    if op == "split":
        parts = text.split(sep, maxsplit) if maxsplit != -1 else text.split(sep)
        return ok({"result": parts, "count": len(parts)})
    if op in ("contains", "count_occurrences"):
        haystack = text if case_sensitive else text.lower()
        needle = substring if case_sensitive else substring.lower()
        if op == "count_occurrences":
            return ok({"result": haystack.count(needle)})
        positions = []
        start = 0
        while True:
            idx = haystack.find(needle, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1
        return ok({"result": bool(positions), "positions": positions})
    if op == "starts_with":
        return ok({"result": text.startswith(prefix)})
    if op == "ends_with":
        return ok({"result": text.endswith(suffix)})
    if op == "title_case":
        return ok({"result": text.title()})
    return err(
        f"Unknown string op '{op}'.",
        hint=f"Use one of: {', '.join(_STRING_OPS)}.",
    )


@mcp.prompt()
def rewrite(text: str) -> str:
    return f"Rewrite the following sentence in a clearer way:\n{text}"


if __name__ == "__main__":
    mcp.run()
