"""Pre-write diff previews for file-mutating tool calls.

A tool that mutates a file declares a ``preview`` spec in its capability
descriptor — ``{"kind": "<edit shape>", "args": [...]}`` — and the builders here
reconstruct the proposed post-write content from the call arguments so a UI can
render a red/green diff *before* the write executes.

Like approval scope kinds (``policy/approval.py``), the
``kind`` names a generic edit shape reused across tools, never a tool identity:
a new tool (first-party or from a user extension server) gets previews by
declaring a shape, with zero client edits. No tool names appear here.

Kept free of UI/transport imports so it is unit-testable hermetically.
"""

from __future__ import annotations

import difflib
import os
from typing import Callable

from ...context.capabilities import ToolCaps, path_args, preview_spec


def _first(arguments: dict, names: tuple[str, ...]) -> str:
    """First non-empty value among the named args (stringified)."""
    for n in names:
        val = arguments.get(n)
        if val not in (None, ""):
            return str(val)
    return ""


# --- Generic post-write content builders, keyed by the declared edit shape -----
# Each builder maps (old_content, arguments, spec_args) -> proposed new content,
# or None when the arguments don't determine a result (no diff is shown then).
# Builders read ONLY the args the tool declared in its preview spec — no default
# argument names here; a spec whose args don't match its kind's arity yields None.

def _preview_content(old: str, arguments: dict, args: tuple[str, ...]) -> str | None:
    """Whole-file content write: the new content is the declared arg, verbatim."""
    if len(args) < 1:
        return None
    return str(arguments.get(args[0], ""))


def _preview_append(old: str, arguments: dict, args: tuple[str, ...]) -> str | None:
    """Append: old content plus the declared content arg."""
    if len(args) < 1:
        return None
    return old + str(arguments.get(args[0], ""))


def _preview_replace(old: str, arguments: dict, args: tuple[str, ...]) -> str | None:
    """Single occurrence replace: declared args are (old_text, new_text)."""
    if len(args) < 2:
        return None
    old_text = str(arguments.get(args[0], ""))
    new_text = str(arguments.get(args[1], ""))
    if not old_text:
        return None
    return old.replace(old_text, new_text, 1)


def _preview_replace_all(old: str, arguments: dict, args: tuple[str, ...]) -> str | None:
    """All-occurrences replace. Approximate: the server may bound matches at
    identifier boundaries; a plain replace is close enough for a preview card."""
    if len(args) < 2:
        return None
    old_text = str(arguments.get(args[0], ""))
    new_text = str(arguments.get(args[1], ""))
    if not old_text:
        return None
    return old.replace(old_text, new_text)


def _preview_line_splice(old: str, arguments: dict, args: tuple[str, ...]) -> str | None:
    """Replace a 1-based inclusive line range: declared args are (start, end, new_content)."""
    if len(args) < 3:
        return None
    try:
        start = int(arguments.get(args[0], 1))
        end = int(arguments.get(args[1], start))
    except (TypeError, ValueError):
        return None
    replacement = str(arguments.get(args[2], ""))
    lines = old.splitlines(keepends=True)
    # Ensure the replacement ends with a newline so surrounding lines join cleanly.
    repl_lines = replacement.splitlines(keepends=True)
    if repl_lines and not repl_lines[-1].endswith("\n"):
        repl_lines[-1] += "\n"
    return "".join(lines[:start - 1] + repl_lines + lines[end:])


def _preview_delete(old: str, arguments: dict, args: tuple[str, ...]) -> str | None:
    """File deletion: the proposed content is empty (renders as an all-red diff)."""
    return ""


#: edit shape -> builder. Generic strategies, reused across tools.
_PREVIEW_KINDS: dict[str, Callable[[str, dict, tuple[str, ...]], str | None]] = {
    "content": _preview_content,
    "append": _preview_append,
    "replace": _preview_replace,
    "replace_all": _preview_replace_all,
    "line_splice": _preview_line_splice,
    "delete": _preview_delete,
}


def build_preview_diffs(
    tool_name: str,
    arguments: dict,
    registry: dict[str, ToolCaps],
    root: str | None = None,
) -> list[dict]:
    """Pre-write unified diff entries for a previewable file-mutation call.

    Returns a list of diff entries (usually one) shaped for the UI's
    ``file_progress`` card: ``{"file": <rel path>, "patch": <unified diff>}``
    plus ``new_content`` for whole-content writes (so an editor can open a real
    diff view), ``is_new`` for file creations and ``is_delete`` for deletions.
    Empty list when the tool declares no preview spec, the target path is
    missing/unreadable, or the arguments produce no visible change.
    """
    spec = preview_spec(tool_name, registry)
    if not spec:
        return []
    builder = _PREVIEW_KINDS.get(str(spec.get("kind") or ""))
    if builder is None:
        return []

    # Declared path arg-role first; generic arg-name fallback otherwise (the same
    # convention as guardrails.observations validator targeting — file tools don't all
    # declare a path role, and these are argument names, never tool identities).
    declared_paths = path_args(tool_name, registry)
    path = _first(arguments, declared_paths or ("path", "filepath")).strip()
    if not path:
        return []

    if not os.path.isabs(path):
        path = os.path.join(root or os.getcwd(), path)
    rel = os.path.relpath(path, root or os.getcwd())

    is_new = False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            old_content = fh.read()
    except FileNotFoundError:
        old_content = ""
        is_new = True
    except OSError:
        return []

    new_content = builder(old_content, arguments, tuple(spec.get("args") or ()))
    if new_content is None:
        return []

    diff_lines = list(difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{rel}", tofile=f"b/{rel}",
        lineterm="",
    ))
    if not diff_lines:
        return []

    entry: dict = {"file": rel, "patch": "\n".join(diff_lines)}
    # For whole-content writes, include the proposed content so the editor can
    # open a proper diff view (current file <-> proposed content).
    if spec.get("kind") == "content":
        entry["new_content"] = new_content
    if is_new:
        entry["is_new"] = True
    if spec.get("kind") == "delete":
        entry["is_delete"] = True
    return [entry]
