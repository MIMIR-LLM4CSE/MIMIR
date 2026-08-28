from __future__ import annotations

import inspect
import json
import os
from typing import Any


from ..event_sink import captured_emitter
from ..guardrails.observations import record_tool_observation
from ..guardrails.policy.engine import evaluate_tool_preconditions
from ..context.execution_context import failed_runs, unsettled_runs
from ..context.capabilities import (
    CACHEABLE, CODE_NAV, EDIT, JUDGE,
    has_cap, names_with_cap,
)
from . import bash_effect
from .plugins import PostToolRegistry
from .validation import absolute_workspace_path, is_scratch_path
from .normalizer import _make_hashable

_CONTINUATION_CHUNK = 120
_OUTLINE_HINT_MAX_SYMBOLS = 40
# Symbol kinds that have a body worth reading. The budget is spent in document order, so
# without a filter a Python file's imports and a TypedDict's fields fill all forty slots
# and every function in the file goes unlisted. Spans both backends' vocabularies (LSP
# kind names and ctags kind names).
_OUTLINE_STRUCTURAL_KINDS = frozenset({
    "class", "function", "method", "constructor", "struct", "interface", "enum",
    "type", "typedef", "union", "namespace", "module", "subroutine", "prototype",
    "macro",
})


def _build_continuation_hint(agent: Any, tool_name: str, payload: dict) -> str:
    """Return a MORE_CONTENT hint when a read stopped short of the end of the file.

    Read off what the server reported, not off the arguments: the window that comes
    back is clamped by a default and by a per-call cap, so the range asked for is not
    the range served — and a caller who is not told the difference cannot tell "this is
    the file" from "this is its first page", which is how re-reading a header becomes a
    loop.
    """
    if not payload.get("truncated"):
        return ""
    path = payload.get("path") or ""
    total = payload.get("total_lines")
    next_start = payload.get("next_start_line")
    if not path or not total or not next_start:
        return ""
    next_end = min(int(next_start) + _CONTINUATION_CHUNK - 1, int(total))
    capped = (
        f" At most {payload['line_cap']} lines are returned per call."
        if payload.get("line_cap") else ""
    )
    return (
        f"\n\nMORE_CONTENT: '{os.path.basename(path)}' has {total} lines; you received"
        f" lines {payload.get('start_line')}–{payload.get('end_line')}.{capped}"
        f" Continue with start_line={next_start}, end_line={next_end} — or search for the"
        f" symbol you need instead of walking the file."
    )


async def _build_outline_hint(
    agent: Any, tool_name: str, payload: dict, execution_context: dict[str, Any] | None,
) -> str:
    """Return a symbol map of the file a truncated read only showed a slice of.

    A page of a large file is the wrong unit of orientation: it answers "what is at the
    top?" when the question is "where is the thing I need?". The outline answers the
    second in a few dozen tokens, which is the difference between one targeted read and
    a walk through the file.

    Advisory and best-effort — no outline backend, no annotation. Once per file per
    query: a map the model ignored twice is noise.
    """
    if execution_context is None or not payload.get("truncated"):
        return ""
    path = payload.get("path") or ""
    if not path or not agent._is_code_filepath(path):
        return ""
    outline_tools = sorted(names_with_cap(CODE_NAV, agent.tool_caps))
    if not outline_tools:
        return ""

    counts = execution_context.get("nudge_counts")
    key = f"outline_hint:{agent._normalize_workspace_path(path)}"
    if not isinstance(counts, dict) or counts.get(key):
        return ""
    counts[key] = 1

    try:
        # No execution_context: this call is the machine's, not the model's, and
        # read_files/checked_paths are discovery *evidence* the gates count. An
        # annotation must not clear a gate on the model's behalf.
        result = await agent._run_tool(outline_tools[0], {"path": path}, execution_context=None)
        symbols = (agent._parse_tool_payload(result) or {}).get("symbols") or []
    except Exception:
        return ""
    if not symbols:
        return ""

    named = [s for s in symbols if s.get("name") and s.get("line")]
    structural = [s for s in named if s.get("kind") in _OUTLINE_STRUCTURAL_KINDS]
    # Kind alone does not separate a definition from an import: a language server reports
    # `from typing import Iterable` as a class, because that is what Iterable is. What
    # separates them is a body — an import spans its own single line.
    with_body = [s for s in structural if int(s.get("end_line") or 0) > int(s["line"])]
    # A backend that reports neither ends nor usable kinds still gets a map.
    listable = with_body or structural or named
    listed = ", ".join(
        f"{s.get('name')}:{s.get('line')}"
        + (f"-{s['end_line']}" if int(s.get("end_line") or 0) > int(s.get("line") or 0) else "")
        for s in listable[:_OUTLINE_HINT_MAX_SYMBOLS]
    )
    if not listed:
        return ""
    hidden = len(listable) - _OUTLINE_HINT_MAX_SYMBOLS
    more = f" (+{hidden} more)" if hidden > 0 else ""
    return (
        f"\n\nOUTLINE: symbols defined in '{os.path.basename(path)}', as name:start-end"
        f" (end omitted where the backend does not report one) — read the whole span of"
        f" the one you need in a single call, instead of paging toward its end a few"
        f" lines at a time."
        f"\n{listed}{more}"
    )


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
        f" right, and nothing downstream can read this output for you. If it showed"
        f" something worth recording, report it with {tool[0]} (run=\"{opened[0]}\"),"
        f" naming the number, message or behaviour you read it from. If it settles"
        f" nothing — a trivial command, a check whose exit code was the whole finding —"
        f" moving on without one is fine."
    )


def _build_imputation_hint(
    agent: Any, execution_context: dict[str, Any] | None, before: set[str],
) -> str:
    """Return an IMPUTATION_DUE line when this call left a run red for the first time.

    The mirror of :func:`_build_verdict_due_hint`, for the other half of what an exit code
    cannot say. A green exit does not say the answer is right; a red one does not say whose
    fault it was — and only the second question has a wrong default, because a wall the
    environment put there was being charged to the change and reported as an unfinished
    task. Asked here, on the failing result itself, at the moment the model is deciding
    what to do about it; the alternative it needs is precisely the one it never had.

    Costs no extra step: the model was going to react to the red exit anyway.
    """
    if execution_context is None:
        return ""
    opened = sorted(set(failed_runs(execution_context)) - before)
    if not opened:
        return ""
    tool = sorted(names_with_cap(JUDGE, agent.tool_caps))
    if not tool:
        return ""
    return (
        f"\n\nIMPUTATION_DUE: this failed, which says nothing about whose fault it is."
        f" If the cause is in the change, fix it and re-run. If it is a prerequisite this"
        f" environment does not have — a build to configure, a package, a dataset, an"
        f" allocation — say so with {tool[0]} (verdict=\"blocked\", run=\"{opened[0]}\")"
        f" and move on: that is a complete ending, not a failure to report, and it costs"
        f" you no retry budget."
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
    require_edit_cap: bool = True,
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

    ``require_edit_cap=False`` is for the shell path, where the caller has already
    established that the file was created — by observing the disk, since a shell tool
    carries neither the EDIT capability nor a ``path`` argument.
    """
    if execution_context is None:
        return ""
    if require_edit_cap and not has_cap(tool_name, EDIT, agent.tool_caps):
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


def _make_subagent_progress_cb(call_id: str):
    """Turn a tool's progress notifications into sub-agent activity events, or None.

    A tool that delegates has a whole run happening inside one call; it reports what
    its child is doing as progress notifications, which the MCP session correlates to
    this call (and only this one), so concurrent delegations never cross. We forward
    each one as an event carrying the parent's row id, and the frontend renders it as
    an ordinary tool row under that parent.

    Returns None when there is nothing to feed: no row to hang the activity on, or no
    frontend bound — the CLI already prints raw event lines, and multiplying those by
    every child tool call would be noise, not a view.
    """
    # Captured here, not read inside the callback: the callback runs on the session's
    # receive loop, a task started at connect time whose context predates the sink.
    emit_event = captured_emitter()
    if not call_id or emit_event is None:
        return None

    async def _on_progress(progress: float, total: float | None, message: str | None) -> None:
        # Runs on the session's receive loop: parse, emit, return. Anything slow here
        # stalls every response on that session.
        try:
            payload = json.loads(message or "")
        except Exception:
            return
        if not isinstance(payload, dict) or payload.get("v") != 1:
            return
        kind = payload.get("t")
        # Child call ids come from the child's own model ("c1", "c2"), so siblings
        # collide — namespace them under the parent's row id.
        child_id = f"{call_id}:{payload.get('i', '')}"
        if kind == "tc":
            emit_event({
                "type": "subagent_event", "kind": "tool_call", "parent_id": call_id,
                "id": child_id, "name": payload.get("n", ""),
                "label": payload.get("l", ""), "detail": payload.get("d", ""),
            })
        elif kind == "tr":
            emit_event({
                "type": "subagent_event", "kind": "tool_result", "parent_id": call_id,
                "id": child_id, "ok": bool(payload.get("ok")),
                "summary": payload.get("s", ""), "duration_ms": payload.get("ms"),
            })
        elif kind == "end":
            emit_event({
                "type": "subagent_event", "kind": "end", "parent_id": call_id,
                "dropped": payload.get("dropped", 0),
            })

    return _on_progress


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
                return cached_text
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
    # Taken before the call and read after it: a shell write is the one mutation in the
    # system that returns no diff, so what it did has to be observed rather than
    # reported. Returns None for a read-only command, and never raises.
    effect_probe = bash_effect.capture(agent, tool_name, arguments, execution_context)
    # The progress callback is passed unconditionally: it only adds a progress token
    # to the request, and a server that never reports progress pays nothing for it.
    # Which tools are worth listening to is decided by what comes back, not by a list
    # of names here.
    result = await session.call_tool(
        tool_name, arguments, progress_callback=_make_subagent_progress_cb(call_id))
    normalized = agent._normalize_tool_content(result)

    runs_before = set(unsettled_runs(execution_context or {}))
    failed_before = set(failed_runs(execution_context or {}))
    record_tool_observation(agent, tool_name, arguments, normalized, execution_context, call_id)

    # Invalidate cached reads for a path when it has just been written.
    if has_cap(tool_name, EDIT, agent.tool_caps):
        written_path = agent._normalize_workspace_path(arguments.get("path") or "")
        if written_path:
            stale = [k for k in cache if has_cap(k[0], CACHEABLE, agent.tool_caps) and written_path in str(k)]
            for k in stale:
                del cache[k]

    # The tool's own payload, before any annotation is appended to it: the hints below
    # that read it as JSON must not be handed a blob with earlier hints already on it.
    payload_text = normalized

    payload_dict = agent._parse_tool_payload(payload_text) or {}
    normalized += _build_continuation_hint(agent, tool_name, payload_dict)
    normalized += await _build_outline_hint(agent, tool_name, payload_dict, execution_context)

    normalized += _build_verdict_due_hint(agent, execution_context, runs_before)
    normalized += _build_imputation_hint(agent, execution_context, failed_before)

    normalized += _build_fork_hint(agent, tool_name, arguments, payload_text, execution_context)
    normalized += bash_effect.report(effect_probe)
    # A file created by the shell is a fork candidate exactly like one created by the
    # edit tool — `cp solver.py solver.py.bak` was the observed case. The probe already
    # knows what appeared, so this reuses the existing rule instead of adding one.
    for created in bash_effect.created_paths(effect_probe):
        normalized += _build_fork_hint(
            agent, tool_name, {"path": created},
            json.dumps({"operation": "created"}), execution_context,
            require_edit_cap=False,
        )

    # Cached with its annotations, stamped with the state of the file it read. The
    # annotations are part of the answer: cached without them, a repeat of the same
    # read came back missing the very lines telling it where to resume.
    if cache_key is not None:
        cache[cache_key] = (normalized, cache_stamp)

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
