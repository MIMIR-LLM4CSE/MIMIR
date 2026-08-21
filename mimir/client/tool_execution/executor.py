from __future__ import annotations

import inspect
import json
import os
from typing import Any


from ..guardrails.observations import record_tool_observation
from ..guardrails.policy.engine import evaluate_tool_preconditions
from ..context.execution_context import unsettled_runs
from ..context.capabilities import (
    CACHEABLE, EDIT, JUDGE, SEARCH_WITH_PATH, has_cap, names_with_cap,
)
from .plugins import PostToolRegistry
from .validation import absolute_workspace_path, is_scratch_path
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


def _build_verdict_due_hint(
    agent: Any, execution_context: dict[str, Any] | None, before: set[str],
) -> str:
    """Return a VERDICT_DUE line when this call left a run nobody has judged yet.

    Said where and when it applies — attached to the run's own result, not carried in
    the system prompt for the whole session. The judging tool is named from the
    registry, so the name lives in the server that declares it and nowhere else; with
    no such tool connected there is nothing to ask for and the hint stays silent.
    """
    if execution_context is None:
        return ""
    opened = sorted(set(unsettled_runs(execution_context)) - before)
    if not opened:
        return ""
    tool = sorted(names_with_cap(JUDGE, agent.tool_caps))
    if not tool:
        return ""
    return (
        f"\n\nVERDICT_DUE: exit 0 says this reached its end, not that its answer is"
        f" right. Nothing downstream can read this output, so read it yourself and report"
        f" what it showed with {tool[0]} (run=\"{opened[0]}\"), naming the number, message"
        f" or behaviour you read it from."
    )


def _path_stamp(agent: Any, arguments: dict) -> tuple | None:
    """``(mtime_ns, size)`` of the file this call names, or ``None`` if it names none.

    What makes the read cache safe to answer from. The cache is invalidated on every
    call to an EDIT-capable tool, which covers the edit tools and nothing else: a
    ``sed -i``, a script the model ran, or the user editing in their IDE all change the
    file with no tool call to notice it. Stamping each entry closes that hole — an entry
    whose file moved is dropped and read again.
    """
    raw = arguments.get("path") or arguments.get("filepath") or ""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        st = os.stat(absolute_workspace_path(agent._normalize_workspace_path(raw) or raw))
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _repeat_read_acknowledgement(
    agent: Any, tool_name: str, arguments: dict, execution_context: dict[str, Any] | None,
) -> str:
    """Short stand-in for a read the model already has, or "" to serve the full copy.

    The cache spares the round trip but not the context: re-serving the same file puts
    every line of it back in the history a second time. This answers the repeat with a
    pointer instead, which costs tokens only on the turn the repetition happens.

    Three cases still get the real content. A path the discovery sets no longer hold is
    one the repair ladder deliberately evicted to force a fresh read
    (``_observe_edit_outcome`` does this after two failed edits) — answering "you
    already have it" there would break exactly the mechanism asking for it. Once the
    context backstop has dropped older content, "above in your context" may simply be
    false. And a pathless cacheable call has no file to reason about, so it is left
    alone.
    """
    if execution_context is None or execution_context.get("history_truncated"):
        return ""
    path = agent._normalize_workspace_path(arguments.get("path") or arguments.get("filepath") or "")
    if not path:
        return ""
    if path not in (execution_context.get("read_files") or set()):
        return ""
    return json.dumps({
        "status": "ok",
        "note": (
            f"Identical read of '{path}' already returned earlier in this query, and the "
            f"file has not changed since. Its content is above in your context — it is not "
            f"repeated here. To read it again anyway, ask for a different line range."
        ),
    }, indent=2, ensure_ascii=False)


def _fork_base_stem(stem: str) -> str:
    """*stem* minus a trailing ``_<word>``, or "" when it carries none.

    Generic on purpose: ``convergence_test_fixed`` → ``convergence_test`` and
    ``solver_ghost`` → ``solver`` without a list of suffixes anyone has to keep up to
    date, which the next model would coin its way around anyway. A remainder under
    three characters is discarded — it matches too much to mean anything.
    """
    base, sep, _suffix = stem.rpartition("_")
    return base if sep and len(base) >= 3 else ""


def _build_fork_hint(
    agent: Any, tool_name: str, arguments: dict, result: str,
    execution_context: dict[str, Any] | None,
) -> str:
    """Flag a file that duplicates one already written, or a probe in the wrong place.

    Said at the write, not at the end of the turn: the model is still holding the file
    it just forked, so the cheap repair (edit the original, drop the copy) is one call
    away. It is an annotation rather than a refusal because nothing is lost by the
    extra file — the write policy deliberately blocks only losses, so a working-order
    rule has no business there.

    Both triggers point at the scratchpad, which exists for exactly the file being
    misplaced here and costs no approval. Capped at one per query: the observation is
    worth making once, and a model that ignored it will ignore the second.
    """
    if execution_context is None or not has_cap(tool_name, EDIT, agent.tool_caps):
        return ""
    counts = execution_context.get("nudge_counts")
    if not isinstance(counts, dict) or counts.get("fork_hint"):
        return ""
    path = agent._normalize_workspace_path(arguments.get("path") or "")
    if not path or is_scratch_path(path):
        return ""
    try:
        payload = json.loads(result)
    except Exception:
        return ""
    # Only a *new* file can be a fork; rewriting one in place is the behaviour we want.
    if not isinstance(payload, dict) or payload.get("operation") != "created":
        return ""

    stem, _ext = os.path.splitext(os.path.basename(path))
    parent = os.path.dirname(path)
    from ...servers._shared.state_paths import scratch_dir
    from ..config.constants import STATE_DIR
    scratch = scratch_dir(STATE_DIR)

    base = _fork_base_stem(stem)
    if base:
        for other in sorted(execution_context.get("dirty_written_files") or ()):
            if other == path or os.path.dirname(other) != parent:
                continue
            if os.path.splitext(os.path.basename(other))[0] == base:
                counts["fork_hint"] = 1
                return (
                    f"\n\nFORK_SUSPECTED: '{os.path.basename(path)}' reads as a second copy"
                    f" of '{os.path.basename(other)}', written earlier in this query. If it"
                    f" supersedes that file, edit it in place and delete what it replaces —"
                    f" leaving both makes the working copy ambiguous. If it is a check you"
                    f" only need while working, it belongs in your scratchpad ({scratch})."
                )

    is_probe_name = stem.startswith("test_") or stem.endswith("_test")
    in_tests_dir = "tests" in parent.split(os.sep)
    if is_probe_name and not in_tests_dir:
        counts["fork_hint"] = 1
        return (
            f"\n\nPROBE_PLACEMENT: '{os.path.basename(path)}' is named like a check but"
            f" sits outside a tests/ directory. A check worth keeping goes in the"
            f" project's test directory; one you only need while working belongs in your"
            f" scratchpad ({scratch}), where it is not reported as produced work."
        )
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


async def run_post_tool_annotations(
    agent: Any,
    tool_name: str,
    arguments: dict[str, Any],
    result: str,
    execution_context: dict[str, Any] | None,
) -> str:
    """Everything appended to a successful tool result: built-in ladder, then hooks.

    The registered hooks (:mod:`tool_execution.plugins`) are the user's own checks, so
    they run last and see the built-in annotation already in *result*. One misbehaving
    hook is skipped rather than allowed to cost the others their turn — same contract as
    a nudge pack, for the same reason.
    """
    text = await run_post_write_validation(agent, tool_name, arguments, execution_context)
    for rule in PostToolRegistry.rules():
        try:
            produced = rule.run(agent, tool_name, arguments, result + text, execution_context)
            if inspect.isawaitable(produced):
                produced = await produced
        except Exception:
            continue
        if produced:
            text += str(produced)
    return text


async def execute_tool_call(
    *,
    agent: Any,
    tool_name: str,
    arguments: dict[str, Any],
    execution_context: dict[str, Any] | None = None,
    run_auto_validation: bool = True,
    call_id: str = "",
) -> str:
    """Run the full policy-and-execution pipeline for one tool call.

    ``run_auto_validation`` controls whether the post-write validation ladder is
    appended inline. The agent loop's dispatcher passes ``False`` and runs
    validation itself under a separate budget (so a slow validator cannot fail an
    already-written file); every other caller keeps the inline default.

    ``call_id`` is the id the UI put this call's row under; it travels with a run that
    ends up awaiting a verdict so the verdict can be shown on that row.
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
    cache_key: tuple | None = None
    cache_stamp: tuple | None = None
    if has_cap(tool_name, CACHEABLE, agent.tool_caps):
        cache_key = (tool_name, _make_hashable(arguments))
        cache_stamp = _path_stamp(agent, arguments)
        hit = cache.get(cache_key)
        if hit is not None:
            cached_text, cached_stamp = hit
            if cached_stamp == cache_stamp:
                short = _repeat_read_acknowledgement(
                    agent, tool_name, arguments, execution_context
                )
                return short if short else cached_text
            # The file moved under us — a write that went around the edit tools (a
            # shell redirect, a script, the user's own editor). Fall through and read
            # it again rather than answer from a copy that is now wrong.
            del cache[cache_key]

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

    runs_before = set(unsettled_runs(execution_context or {}))
    record_tool_observation(agent, tool_name, arguments, normalized, execution_context, call_id)

    # Invalidate cached reads for a path when it has just been written.
    if has_cap(tool_name, EDIT, agent.tool_caps):
        written_path = agent._normalize_workspace_path(arguments.get("path") or "")
        if written_path:
            stale = [k for k in cache if has_cap(k[0], CACHEABLE, agent.tool_caps) and written_path in str(k)]
            for k in stale:
                del cache[k]

    # Store the result in cache if this was a cacheable read, stamped with the state
    # of the file it read so a later hit can tell "unchanged" from "stale".
    if cache_key is not None:
        cache[cache_key] = (normalized, cache_stamp)

    # The tool's own payload, before any annotation is appended to it: the hints below
    # that read it as JSON must not be handed a blob with earlier hints already on it.
    payload_text = normalized

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

    normalized += _build_verdict_due_hint(agent, execution_context, runs_before)

    normalized += _build_fork_hint(agent, tool_name, arguments, payload_text, execution_context)

    # Auto-validation is advisory and runs AFTER the write has already succeeded
    # (the file is on disk). When the dispatcher opts out (run_auto_validation=
    # False) it runs the same ladder itself under a separate timeout budget, so a
    # slow validator can never cancel the completed write; every other caller keeps
    # it inline. Either way it is best-effort — see run_post_write_validation.
    if run_auto_validation:
        normalized += await run_post_tool_annotations(
            agent, tool_name, arguments, normalized, execution_context
        )

    return normalized
