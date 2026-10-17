"""What edits, deletions and reads tell the guardrails about files."""
from __future__ import annotations

from __future__ import annotations
import json
import os
from typing import Any
from ...context import (
    FILE_PATH,
    declared_edit_set_complete,
    fields_with,
)
from ..workflow import (
    has_pending_validation,
    set_workflow_state,
)
from ...context.capabilities import (
    DELEGATE,
    EDIT,
    READ,
    REMOVE,
    REPLACEMENT_TRACK,
    arg_role,
    clears_edit_loop,
    has_cap,
    target_path_arg,
)
from ...context.paths import is_scratch_path
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort
from .call import Call


def _should_enter_validate(execution_context: ExecutionContext) -> bool:
    """Return True when the agent should transition from edit -> validate.

    If the model declared a multi-file edit plan (a task-checklist declaration), do not force
    validation after the first successful edit. Only enter validate when all declared targets have
    been dirtied, or when no explicit edit set exists.
    """
    return declared_edit_set_complete(execution_context)


def _edit_signature(tool_name: str, path: str, arguments: dict, registry: Any = None) -> str:
    # Dedup signature from the args the *server* declares as identifying an edit (the
    # ``edit_sig`` arg-role), each part trimmed to a 64-char preview. Tools declaring no
    # edit_sig role fall back to a full-arguments signature.
    sig_args = arg_role(tool_name, "edit_sig", registry)
    if sig_args:
        parts = "|".join(str(arguments.get(a, "")).strip()[:64] for a in sig_args)
        return f"{tool_name}|{path}|{parts}"
    return f"{tool_name}|{path}|{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"


def register_known_path(execution_context: ExecutionContext, candidate_path: str) -> None:
    if not candidate_path:
        return
    execution_context.existing_paths.add(candidate_path)


def _target_path(agent: AgentPort, tool_name: str, arguments: dict) -> str | None:
    """The normalized workspace path this call targets."""
    return agent.normalize_workspace_path(
        target_path_arg(tool_name, arguments, agent.tool_caps))


def _clear_edit_fail_streak(execution_context: ExecutionContext, path: str) -> None:
    """Drop *path*'s edit-failure streak and re-arm the error_recovery reminder budget.

    The error_recovery nudge is capped per query so two ignored reminders don't turn
    into per-step spam. But the cap must not mute it *forever*: once the streak that
    earned those reminders is resolved (a successful edit or a re-read of the file)
    and no other file is still failing, reset the counter so a later, distinct spate
    of failures earns fresh reminders. While any file remains stuck the budget stays
    spent, preserving the anti-spam guarantee.
    """
    streaks = execution_context.edit_fail_streak_by_file
    streaks.pop(path, None)
    if not streaks:
        counts = execution_context.nudge_counts
        if isinstance(counts, dict):
            counts["error_recovery"] = 0


def record_code_edit(execution_context: ExecutionContext, edited_path: str) -> None:
    """Mark one successfully-written code file dirty.

    The workflow transition (:func:`enter_post_edit_state`) is left to the caller.
    """
    # Scratchpad files are working material, not produced work: recording them
    # would put throwaway probe scripts in the change ledger and make them demand
    # validation before the run could conclude.
    if is_scratch_path(edited_path):
        return
    register_known_path(execution_context, edited_path)
    # The parent directory is implicitly proven to exist once a file inside
    # it is written — register it so the carry context knows the dir exists.
    parent_dir = os.path.dirname(edited_path)
    if parent_dir and parent_dir != ".":
        register_known_path(execution_context, parent_dir)
    execution_context.code_mutation_started = True
    execution_context.planned_edit_targets.add(edited_path)
    execution_context.dirty_written_files.add(edited_path)
    execution_context.validated_files.discard(edited_path)
    # Evidence is about a revision, and so is its absence: a file rewritten after the
    # floor rejected it — or after the floor could not read it — is owed a fresh answer,
    # not the previous one.
    execution_context.unverifiable_files.discard(edited_path)
    execution_context.builtin_check_findings.pop(edited_path, None)
    execution_context.builtin_check_stamp.pop(edited_path, None)
    # Evidence is about a specific revision: rewriting the file retracts it. Runs are
    # not retracted — a run is a past event, and its verdict is a statement about what
    # that event showed, which re-editing does not undo.
    execution_context.validation_tier_by_file.pop(edited_path, None)
    execution_context.edit_loop_state.pop(edited_path, None)
    _clear_edit_fail_streak(execution_context, edited_path)


def enter_post_edit_state(execution_context: ExecutionContext) -> None:
    """Reset the idle counter and move to validate (if the declared set is done) or edit.

    Straight to ``conclude`` when the declared set is done and nothing owes a check:
    every file written is one nothing here can check, so there is no validate step to
    enter and telling the model to check them is asking for a command that does not
    exist on this machine.
    """
    execution_context.steps_since_last_edit = 0
    if not _should_enter_validate(execution_context):
        set_workflow_state(execution_context, "edit")
    elif has_pending_validation(execution_context):
        set_workflow_state(execution_context, "validate")
    else:
        set_workflow_state(execution_context, "conclude")


def edit_outcome(call: Call) -> None:
    """Record the outcome of a single-file code edit.

    Success: mark the file dirty and advance the workflow state.
    Failure: track repeated identical patches, and the per-file streak of wrong anchors.
    These are the two status branches of one event, so they share a single guard.
    """
    execution_context = call.execution_context
    if not has_cap(call.tool_name, EDIT, call.agent.tool_caps):
        return
    edited_path = _target_path(call.agent, call.tool_name, call.arguments)
    if not (edited_path and call.agent.is_code_filepath(edited_path)):
        return

    if call.status == "ok":
        record_code_edit(execution_context, edited_path)
        # Signal the post-dispatch injector that a successful code edit occurred.
        execution_context.last_edit_success_path = edited_path
        enter_post_edit_state(execution_context)
        return

    current_signature = _edit_signature(
        call.tool_name, edited_path, call.arguments, call.agent.tool_caps)
    last_sig, _fail_count = execution_context.edit_loop_state.get(edited_path, (None, 0))
    if current_signature == last_sig:
        new_fail_count = _fail_count + 1
        execution_context.edit_loop_state[edited_path] = (last_sig, new_fail_count)
    else:
        new_fail_count = 1
        execution_context.edit_loop_state[edited_path] = (current_signature, new_fail_count)
    # Per-file streak, incremented whether or not the patch changed — this is what
    # escalates a model trying *different* wrong anchors on the same file. The
    # signature-based count above resets per patch and only guards identical-patch spin.
    # The file is deliberately NOT evicted from the read sets here: a wrong anchor is not
    # missing content, and forging "you have not read this" to provoke a re-read buys a
    # blind reread where the failure itself already carries the current text (see
    # server_files._anchor_retry_hint).
    streak = int(execution_context.edit_fail_streak_by_file.get(edited_path, 0)) + 1
    execution_context.edit_fail_streak_by_file[edited_path] = streak


def replacement_tracking(call: Call) -> None:
    """Track old_text/new_text from replacement operations for completeness checking."""
    execution_context = call.execution_context
    if call.status != "ok" or not has_cap(call.tool_name, REPLACEMENT_TRACK, call.agent.tool_caps):
        return
    old_text = call.arguments.get("old_text")
    if not old_text:
        return
    edited_path = call.agent.normalize_workspace_path(call.arguments.get("path"))
    last_file = execution_context.last_replace_file
    # If this edit is on a different file than the previous tracked edit,
    # clear the stale last_replace_* fields first so completeness checks
    # don't fire against the wrong file.
    if last_file and edited_path and edited_path != last_file:
        execution_context.last_replace_old_text = ""
        execution_context.last_replace_new_text = ""
        execution_context.last_replace_file = ""
    execution_context.last_replace_old_text = old_text
    execution_context.last_replace_new_text = call.arguments.get("new_text") or ""
    execution_context.last_replace_file = edited_path or ""
    # A confirmed replacement (only the workspace-wide replacer takes confirm=True among
    # replacement-tracking tools) seeds the cross-file completeness check.
    if call.arguments.get("confirm"):
        execution_context.cross_file_grep_old_text = old_text
        execution_context.cross_file_grep_source = edited_path or ""


def edit_loop_clear(call: Call) -> None:
    """Re-reading or re-checking a file resets its edit-loop failure tracking."""
    execution_context = call.execution_context
    if not (clears_edit_loop(call.tool_name, call.agent.tool_caps) and call.status == "ok"):
        return
    read_path = call.agent.normalize_workspace_path(
        call.arguments.get("path") or call.arguments.get("filepath"))
    if read_path:
        execution_context.edit_loop_state.pop(read_path, None)
        _clear_edit_fail_streak(execution_context, read_path)


def delete(call: Call) -> None:
    execution_context = call.execution_context
    if not (has_cap(call.tool_name, REMOVE, call.agent.tool_caps) and call.status == "ok"):
        return
    deleted_path = call.agent.normalize_workspace_path(call.arguments.get("path"))
    if not deleted_path:
        return
    # Every field that holds file paths, derived from the FILE_PATH trait: a deleted
    # file must not survive anywhere as evidence, and hand-listing the fields meant a
    # new path-valued field would silently keep stale entries.
    for field in fields_with(FILE_PATH):
        getattr(execution_context, field).discard(deleted_path)
    call.agent.carry.discard_path(deleted_path)


def read(call: Call) -> None:
    execution_context = call.execution_context
    if not (has_cap(call.tool_name, READ, call.agent.tool_caps) and call.status == "ok"):
        return
    path = call.agent.normalize_workspace_path(call.arguments.get("path"))
    if not path:
        return

    execution_context.read_files.add(path)
    register_known_path(execution_context, path)
    execution_context.checked_paths.add(path)


def delegated_exploration(call: Call) -> None:
    """Credit what a sub-agent read back to the caller that sent it.

    These gates ask whether the model has facts about the code or is working off file
    names. A file a child opened, and whose finding came back into this conversation,
    is such a fact — so it is discovery evidence here. It stays out of ``read_files``,
    which also answers "does this agent hold the lines it is about to edit": that one
    only a read of its own can settle.
    """
    execution_context = call.execution_context
    if not (has_cap(call.tool_name, DELEGATE, call.agent.tool_caps) and call.status == "ok"):
        return
    for raw in call.payload.get("files_read") or ():
        path = call.agent.normalize_workspace_path(raw)
        if path:
            execution_context.delegated_read_files.add(path)
            register_known_path(execution_context, path)
