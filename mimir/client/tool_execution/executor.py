from __future__ import annotations

import json
import os
from typing import Any


from ..guardrails.observations import record_tool_observation
from ..guardrails.policy.engine import evaluate_tool_preconditions
from ..context.capabilities import CACHEABLE, EDIT, SEARCH_WITH_PATH, has_cap
from .validation import absolute_workspace_path


# Import _make_hashable from normalizer
from .normalizer import _make_hashable

_READ_HINT_MAX_MATCHES = 3
_READ_HINT_FALLBACK_WINDOW = 80
_PYTHON_EXTS = {".py", ".pyx", ".pyi"}
_CONTINUATION_CHUNK = 120


def _fallback_range(hit_line: int) -> tuple[int, int]:
    start = max(1, hit_line - 10)
    return start, start + _READ_HINT_FALLBACK_WINDOW - 1


def _indent_of(line: str) -> int:
    """Return indentation depth; treat blank/whitespace-only lines as infinite."""
    stripped = line.lstrip()
    if not stripped.strip():
        return 9999
    return len(line) - len(stripped)


def _is_def_line(line: str) -> bool:
    return line.lstrip().startswith(("def ", "async def ", "class "))


def _is_decorator_line(line: str) -> bool:
    return line.lstrip().startswith("@")


def _find_smart_range(abs_path: str, hit_line: int) -> tuple[int, int]:
    """Return a (start, end) 1-based line range that covers the full
    function/class block containing hit_line for Python files.
    Falls back to a fixed window for non-Python files or unreadable files."""
    _, ext = os.path.splitext(abs_path)
    if ext.lower() not in _PYTHON_EXTS:
        return _fallback_range(hit_line)

    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
            raw = fh.readlines()
    except OSError:
        return _fallback_range(hit_line)

    n = len(raw)
    if n == 0:
        return _fallback_range(hit_line)

    hit_idx = max(0, min(hit_line - 1, n - 1))  # 0-based, clamped

    # --- locate the innermost function/class that owns hit_line ---
    # Walk backwards to find the def/class line whose indentation is strictly
    # less than the hit line's indentation (or is the hit line itself).
    hit_ind = _indent_of(raw[hit_idx])
    if hit_ind == 9999:
        # Hit is on a blank line — find the nearest non-blank indent
        for i in range(hit_idx, -1, -1):
            if raw[i].strip():
                hit_ind = _indent_of(raw[i])
                break
        else:
            return _fallback_range(hit_line)

    owner_idx: int | None = None
    owner_ind: int | None = None

    if _is_def_line(raw[hit_idx]):
        owner_idx = hit_idx
        owner_ind = _indent_of(raw[hit_idx])
    else:
        for i in range(hit_idx - 1, -1, -1):
            if not raw[i].strip():
                continue
            ind = _indent_of(raw[i])
            if ind < hit_ind and _is_def_line(raw[i]):
                owner_idx = i
                owner_ind = ind
                break
            # If we reach a line with less indentation that is NOT a def/class
            # (e.g. module-level assignment), the hit is at module level.
            if ind < hit_ind and not _is_def_line(raw[i]) and not _is_decorator_line(raw[i]):
                break

    if owner_idx is None:
        return _fallback_range(hit_line)

    # --- walk back further to include leading decorators ---
    decorator_start = owner_idx
    for i in range(owner_idx - 1, -1, -1):
        line = raw[i]
        if not line.strip():
            break
        if _is_decorator_line(line) and _indent_of(line) == owner_ind:
            decorator_start = i
        else:
            break

    # --- find end of the owning block ---
    # The block ends just before the next def/class/decorator at the same or
    # lower indentation level, skipping blank lines.
    end_idx = n - 1
    for i in range(owner_idx + 1, n):
        if not raw[i].strip():
            continue
        ind = _indent_of(raw[i])
        if ind <= owner_ind and (_is_def_line(raw[i]) or _is_decorator_line(raw[i])):
            end_idx = i - 1
            # Trim trailing blank lines
            while end_idx > owner_idx and not raw[end_idx].strip():
                end_idx -= 1
            break

    # Clamp to valid range
    start_1 = decorator_start + 1          # convert to 1-based
    end_1 = min(end_idx + 1, n)            # convert to 1-based, clamped to file length
    return start_1, end_1


def _build_read_hint(tool_name: str, result: str, resolve_abs_path=None, registry=None) -> str:
    """Return a READ_HINT block when a search result contains actionable file paths.
    Uses smart function-boundary detection for Python files when resolve_abs_path
    is provided; falls back to a fixed window otherwise."""
    if not has_cap(tool_name, SEARCH_WITH_PATH, registry):
        return ""
    try:
        payload = json.loads(result)
    except Exception:
        return ""
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return ""

    matches = payload.get("matches", [])
    if not matches:
        return ""

    seen_paths: set[str] = set()
    suggestions: list[str] = []

    for match in matches:
        if len(suggestions) >= _READ_HINT_MAX_MATCHES:
            break

        path = match.get("path", "")
        snippets = match.get("snippets") or []
        hit_line = int(snippets[0]["line"]) if snippets else 1

        if not path:
            continue

        # Deduplicate: one read suggestion per file
        if path in seen_paths:
            continue
        seen_paths.add(path)

        # Determine smart range
        if resolve_abs_path is not None:
            try:
                abs_path = resolve_abs_path(path)
                start, end = _find_smart_range(abs_path, hit_line)
            except Exception:
                start, end = _fallback_range(hit_line)
        else:
            start, end = _fallback_range(hit_line)

        _, ext = os.path.splitext(path)
        if ext.lower() in _PYTHON_EXTS and resolve_abs_path is not None:
            note = f"  # covering function/class at line {hit_line}"
        else:
            note = f"  # hit at line {hit_line}" if hit_line > 1 else ""

        suggestions.append(
            f'  read_file_lines("{path}", start_line={start}, end_line={end}){note}'
        )

    if not suggestions:
        return ""

    return (
        "\n\nREAD_HINT: Use read_file_lines to inspect the top matches:\n"
        + "\n".join(suggestions)
    )


def _build_continuation_hint(tool_name: str, arguments: dict) -> str:
    """Return a MORE_CONTENT hint when a read_file_lines call did not reach EOF.

    Tells the model the file has more lines and suggests the exact next call,
    enabling the same iterative read strategy a human analyst uses: read a
    chunk, decide if more context is needed, continue from where you left off.
    """
    if tool_name != "read_file_lines":
        return ""

    path = arguments.get("path") or arguments.get("filepath") or ""
    end_line = arguments.get("end_line")
    start_line = arguments.get("start_line", 1)
    if not path or end_line is None:
        return ""

    try:
        end_line = int(end_line)
        start_line = int(start_line)
        abs_path = absolute_workspace_path(path)
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
            total = sum(1 for _ in fh)
        if end_line >= total:
            return ""
        next_start = end_line + 1
        next_end = min(end_line + _CONTINUATION_CHUNK, total)
        return (
            f"\n\nMORE_CONTENT: File has {total} lines total;"
            f" you received lines {start_line}–{end_line}."
            f' To read more: read_file_lines("{path}",'
            f" start_line={next_start}, end_line={next_end})"
        )
    except (OSError, ValueError):
        return ""


async def run_post_write_validation(
    agent: Any,
    tool_name: str,
    arguments: dict[str, Any],
    execution_context: dict[str, Any] | None,
) -> str:
    """Return the advisory AUTO_VALIDATION annotation for a write, or "".

    Runs the post-write validation ladder for an EDIT-capable tool with a target
    path; no-ops (returns "") for anything else. Best-effort: any error yields "".

    Split out of ``execute_tool_call`` so a caller can run it OUTSIDE the write's
    timeout budget. Validation runs after the file is already on disk, so a slow or
    crashing validator must never be able to turn a successful write into a failed
    tool row — the dispatch layer gives this its own independent budget.
    """
    target_path = agent._normalize_workspace_path(arguments.get("path"))
    if not (has_cap(tool_name, EDIT, agent.tool_caps) and target_path):
        return ""
    try:
        auto_validation = await agent._auto_validate_written_file(target_path, execution_context)
    except Exception:
        return ""
    if auto_validation:
        return "\n\nAUTO_VALIDATION\n" + auto_validation
    return ""


async def execute_tool_call(
    *,
    agent: Any,
    tool_name: str,
    arguments: dict[str, Any],
    execution_context: dict[str, Any] | None = None,
    run_auto_validation: bool = True,
) -> str:
    """Run the full policy-and-execution pipeline for one tool call.

    ``run_auto_validation`` controls whether the post-write validation ladder is
    appended inline. The agent loop's dispatcher passes ``False`` and runs
    validation itself under a separate budget (so a slow validator cannot fail an
    already-written file); every other caller keeps the inline default.
    """
    evaluation = evaluate_tool_preconditions(
        agent=agent,
        tool_name=tool_name,
        arguments=arguments,
        execution_context=execution_context,
    )
    if evaluation.violation is not None:
        return evaluation.violation

    tool_name = evaluation.tool_name
    arguments = evaluation.arguments
    execution_context = evaluation.execution_context

    # Serve read-only tool calls from the per-query cache.
    cache: dict = getattr(agent, "_tool_cache", {})
    if has_cap(tool_name, CACHEABLE, agent.tool_caps):
        cache_key = (tool_name, _make_hashable(arguments))
        if cache_key in cache:
            return cache[cache_key]

    # In batch mode, snapshot the target file before the first write so we can
    # produce a diff and revert if the user rejects the batch at the end of turn.
    if agent.approvals.batch_mode:
        path = agent._normalize_workspace_path(
            arguments.get("path") or arguments.get("filepath") or ""
        )
        if path:
            agent.approvals.record_snapshot(path)

    if tool_name not in agent.tool_owner:
        available = sorted(agent.tool_owner.keys())
        return json.dumps({
            "status": "error",
            "error": f"Unknown tool '{tool_name}'. This tool does not exist.",
            "available_tools": available,
        })
    session = agent.sessions[agent.tool_owner[tool_name]]
    result = await session.call_tool(tool_name, arguments)
    normalized = agent._normalize_tool_content(result)

    record_tool_observation(agent, tool_name, arguments, normalized, execution_context)

    # Invalidate cached reads for a path when it has just been written.
    if has_cap(tool_name, EDIT, agent.tool_caps):
        written_path = agent._normalize_workspace_path(arguments.get("path") or "")
        if written_path:
            stale = [k for k in cache if has_cap(k[0], CACHEABLE, agent.tool_caps) and written_path in str(k)]
            for k in stale:
                del cache[k]

    # Store the result in cache if this was a cacheable read.
    if has_cap(tool_name, CACHEABLE, agent.tool_caps):
        cache_key = (tool_name, _make_hashable(arguments))
        cache[cache_key] = normalized

    read_hint = _build_read_hint(
        tool_name,
        normalized,
        resolve_abs_path=lambda p: absolute_workspace_path(p),
        registry=agent.tool_caps,
    )
    if read_hint:
        normalized += read_hint

    continuation_hint = _build_continuation_hint(tool_name, arguments)
    if continuation_hint:
        normalized += continuation_hint

    # Auto-validation is advisory and runs AFTER the write has already succeeded
    # (the file is on disk). When the dispatcher opts out (run_auto_validation=
    # False) it runs the same ladder itself under a separate timeout budget, so a
    # slow validator can never cancel the completed write; every other caller keeps
    # it inline. Either way it is best-effort — see run_post_write_validation.
    if run_auto_validation:
        normalized += await run_post_write_validation(
            agent, tool_name, arguments, execution_context
        )

    return normalized
