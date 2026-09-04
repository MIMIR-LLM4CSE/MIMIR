"""Tool-observation layer: the writer of the execution_context blackboard.

``record_tool_observation`` runs after every tool call and dispatches the ordered
``_observe_*`` handlers that populate discovery/edit/validation state + workflow
transitions. Both guardrail subsystems (policy gates, nudges) read what this
writes. Split out of the former ``policy/runtime.py`` and hoisted to the
``guardrails`` root because it is shared machinery, not a policy gate.
"""
from __future__ import annotations

import json
import os
import re as _re
from typing import Any

from ..context import (
    FILE_PATH,
    SOURCE_FILE_EXTENSIONS,
    VALIDATION_TIERS,
    bootstrap_runtime_context,
    declared_edit_set_complete,
    ensure_execution_context,
    failed_runs,
    fields_with,
    raise_validation_tier,
    record_run,
)
from ...servers._shared.numerics import observed_failure_verdict
from ..event_sink import emit
from .workflow import (
    VALIDATION_RETRY_BUDGET,
    has_pending_validation,
    pending_validation_paths,
    set_workflow_state,
)
from ..context.capabilities import (
    CANDIDATE_SEARCH,
    CHECK_EXISTENCE,
    CODE_EXEC,
    DELEGATE,
    EDIT,
    ENV_DISCOVERY,
    ENV_MUTATE,
    INSPECT_DIR,
    JUDGE,
    PLAN_BLOCKED,
    READ,
    REMOVE,
    REPLACEMENT_TRACK,
    SEARCH,
    TASK_PLANNING,
    VALIDATE,
    arg_role,
    clears_edit_loop,
    has_cap,
    names_with_cap,
    path_args,
    run_outcome_spec,
    scope_spec,
)
from ..tool_execution.validation import (
    is_python_test_filepath,
    is_scratch_path,
)
from ...servers._shared.shell_paths import (
    EFFECT_BUILD,
    EFFECT_RUN,
    EFFECT_VALIDATE,
    any_command_on_path as _any_command_on_path,
    is_path_like_command,
)
from .policy.bash_classify import (
    Kind, classify_bash_command, opaque_command_executes,
)


# Source-file path regex derived from the canonical extension set so it stays in sync
# (and automatically covers CUDA/Fortran/… as SOURCE_FILE_EXTENSIONS evolves). Longest
# extensions first so e.g. ".f90" is preferred over ".f".
_SOURCE_EXT_ALTERNATION = "|".join(
    _re.escape(ext.lstrip("."))
    for ext in sorted(SOURCE_FILE_EXTENSIONS, key=len, reverse=True)
)
_SOURCE_FILE_PATH_RE = _re.compile(rf"[\w./\-]+\.(?:{_SOURCE_EXT_ALTERNATION})\b")

# Signals that a check/run failed for lack of an importable module rather than a
# code defect. Matches both Python's runtime "ModuleNotFoundError: No module named
# 'x'" and the static import-resolver's "unresolved import: x" phrasing.
_MODULE_NOT_FOUND_RE = _re.compile(
    r"ModuleNotFoundError|No module named|unresolved import|cannot import name|"
    r"could not be resolved",
    _re.IGNORECASE,
)
# Extracts the offending module name (when the message names it) so the
# env-resolution nudge can quote it. Runs only after detection matched.
_MODULE_NAME_RE = _re.compile(
    r"(?:No module named|unresolved import|cannot import name)[:\s]*['\"]?([\w.]+)",
    _re.IGNORECASE,
)


def _should_enter_validate(execution_context: dict[str, Any]) -> bool:
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


def _register_known_path(execution_context: dict[str, Any], candidate_path: str) -> None:
    if not candidate_path:
        return
    execution_context["existing_paths"].add(candidate_path)


def _target_path(agent: Any, arguments: dict) -> str | None:
    """Normalized workspace path from the conventional file-target args (path/filepath)."""
    return agent._normalize_workspace_path(arguments.get("path") or arguments.get("filepath"))


def _clear_edit_fail_streak(execution_context: dict[str, Any], path: str) -> None:
    """Drop *path*'s edit-failure streak and re-arm the error_recovery reminder budget.

    The error_recovery nudge is capped per query so two ignored reminders don't turn
    into per-step spam. But the cap must not mute it *forever*: once the streak that
    earned those reminders is resolved (a successful edit or a re-read of the file)
    and no other file is still failing, reset the counter so a later, distinct spate
    of failures earns fresh reminders. While any file remains stuck the budget stays
    spent, preserving the anti-spam guarantee.
    """
    streaks = execution_context["edit_fail_streak_by_file"]
    streaks.pop(path, None)
    if not streaks:
        counts = execution_context.get("nudge_counts")
        if isinstance(counts, dict):
            counts["error_recovery"] = 0


def _record_code_edit(execution_context: dict[str, Any], edited_path: str) -> None:
    """Mark one successfully-written code file dirty (the per-path edit slice).

    Split out of :func:`_observe_edit_outcome` so the post-edit transition
    (:func:`_enter_post_edit_state`) stays the caller's to make.
    """
    # Scratchpad files are working material, not produced work: recording them
    # would put throwaway probe scripts in the change ledger and make them demand
    # validation before the run could conclude.
    if is_scratch_path(edited_path):
        return
    _register_known_path(execution_context, edited_path)
    # The parent directory is implicitly proven to exist once a file inside
    # it is written — register it so the carry context knows the dir exists.
    parent_dir = os.path.dirname(edited_path)
    if parent_dir and parent_dir != ".":
        _register_known_path(execution_context, parent_dir)
    execution_context["code_mutation_started"] = True
    execution_context["planned_edit_targets"].add(edited_path)
    execution_context["dirty_written_files"].add(edited_path)
    execution_context["validated_files"].discard(edited_path)
    # Evidence is about a revision, and so is its absence: a file rewritten after the
    # floor rejected it — or after the floor could not read it — is owed a fresh answer,
    # not the previous one.
    execution_context["unverifiable_files"].discard(edited_path)
    execution_context.get("builtin_check_findings", {}).pop(edited_path, None)
    execution_context.get("builtin_check_stamp", {}).pop(edited_path, None)
    # Evidence is about a specific revision: rewriting the file retracts it. Runs are
    # not retracted — a run is a past event, and its verdict is a statement about what
    # that event showed, which re-editing does not undo.
    execution_context.get("validation_tier_by_file", {}).pop(edited_path, None)
    execution_context["edit_loop_state"].pop(edited_path, None)
    _clear_edit_fail_streak(execution_context, edited_path)


def _enter_post_edit_state(execution_context: dict[str, Any]) -> None:
    """Reset the idle counter and move to validate (if the declared set is done) or edit.

    Straight to ``conclude`` when the declared set is done and nothing owes a check:
    every file written is one nothing here can check, so there is no validate step to
    enter and telling the model to check them is asking for a command that does not
    exist on this machine.
    """
    execution_context["steps_since_last_edit"] = 0
    if not _should_enter_validate(execution_context):
        set_workflow_state(execution_context, "edit")
    elif has_pending_validation(execution_context):
        set_workflow_state(execution_context, "validate")
    else:
        set_workflow_state(execution_context, "conclude")


def _observe_edit_outcome(
    agent: Any, tool_name: str, arguments: dict, status: Any, execution_context: dict[str, Any],
) -> None:
    """Record the outcome of a single-file code edit.

    Success: mark the file dirty and advance the workflow state.
    Failure: track repeated identical patches, and the per-file streak of wrong anchors.
    These are the two status branches of one event, so they share a single guard.
    """
    if not has_cap(tool_name, EDIT, agent.tool_caps):
        return
    edited_path = _target_path(agent, arguments)
    if not (edited_path and agent._is_code_filepath(edited_path)):
        return

    if status == "ok":
        _record_code_edit(execution_context, edited_path)
        # Signal the post-dispatch injector that a successful code edit occurred.
        execution_context["last_edit_success_path"] = edited_path
        _enter_post_edit_state(execution_context)
        return

    current_signature = _edit_signature(tool_name, edited_path, arguments, agent.tool_caps)
    last_sig, _fail_count = execution_context["edit_loop_state"].get(edited_path, (None, 0))
    if current_signature == last_sig:
        new_fail_count = _fail_count + 1
        execution_context["edit_loop_state"][edited_path] = (last_sig, new_fail_count)
    else:
        new_fail_count = 1
        execution_context["edit_loop_state"][edited_path] = (current_signature, new_fail_count)
    # Per-file streak, incremented whether or not the patch changed — this is what
    # escalates a model trying *different* wrong anchors on the same file. The
    # signature-based count above resets per patch and only guards identical-patch spin.
    # The file is deliberately NOT evicted from the read sets here: a wrong anchor is not
    # missing content, and forging "you have not read this" to provoke a re-read buys a
    # blind reread where the failure itself already carries the current text (see
    # server_files._anchor_retry_hint).
    streak = int(execution_context["edit_fail_streak_by_file"].get(edited_path, 0)) + 1
    execution_context["edit_fail_streak_by_file"][edited_path] = streak


def _observe_todo_flags(agent: Any, tool_name: str, status: Any, execution_context: dict[str, Any]) -> None:
    """Flag that a task plan was recorded. The checklist (ordered steps) and the prose
    rationale are the same TASK_PLANNING capability, told apart by the `plan_steps`
    arg-role — the checklist carries the steps, the rationale does not."""
    if status != "ok" or tool_name not in names_with_cap(TASK_PLANNING, agent.tool_caps):
        return
    if arg_role(tool_name, "plan_steps", agent.tool_caps):
        execution_context["todo_written"] = True
    else:
        execution_context["plan_written"] = True


def _observe_replacement_tracking(
    agent: Any, tool_name: str, arguments: dict, status: Any, execution_context: dict[str, Any],
) -> None:
    """Track old_text/new_text from replacement operations for completeness checking."""
    if not (has_cap(tool_name, REPLACEMENT_TRACK, agent.tool_caps) and status == "ok"):
        return
    old_text = arguments.get("old_text")
    if not old_text:
        return
    edited_path = agent._normalize_workspace_path(arguments.get("path"))
    last_file = execution_context.get("last_replace_file", "")
    # If this edit is on a different file than the previous tracked edit,
    # clear the stale last_replace_* fields first so completeness checks
    # don't fire against the wrong file.
    if last_file and edited_path and edited_path != last_file:
        execution_context["last_replace_old_text"] = ""
        execution_context["last_replace_new_text"] = ""
        execution_context["last_replace_file"] = ""
    execution_context["last_replace_old_text"] = old_text
    execution_context["last_replace_new_text"] = arguments.get("new_text") or ""
    execution_context["last_replace_file"] = edited_path or ""
    # A confirmed replacement (only the workspace-wide replacer takes confirm=True among
    # replacement-tracking tools) seeds the cross-file completeness check.
    if arguments.get("confirm"):
        execution_context["cross_file_grep_old_text"] = old_text
        execution_context["cross_file_grep_source"] = edited_path or ""


def _mark_file_validated(
    execution_context: dict[str, Any], target: str, tier: str = "static",
) -> None:
    """Credit *target* as checked and advance the workflow when nothing is pending.

    Only a **checker** reaches here — ``py_compile``, ``ruff``, ``mypy``, a compiler:
    a tool whose output *is* a list of problems, so exit 0 means there were none and
    there is nothing left for anyone to read. Running the code never lands here, however
    green it exits: "it reached the end" is a fact about the process, not about the
    answer, and it is recorded on the run instead (see ``context.record_run``).

    *tier* records which kind of check it was (see ``VALIDATION_TIERS``). It has no
    effect on the conclude-gate — every tier counts as validated — and is read only by
    the completion ledger.
    """
    execution_context["validation_fail_count_by_file"].pop(target, None)
    execution_context["validated_files"].add(target)
    raise_validation_tier(execution_context, target, tier)
    if execution_context["code_mutation_started"] and not has_pending_validation(execution_context):
        set_workflow_state(execution_context, "conclude")
        execution_context["nudge_counts"]["validation"] = 0
        execution_context["nudge_counts"]["state"] = 0


def _register_validation_failure(
    execution_context: dict[str, Any], target: str, tier: str = "static",
) -> None:
    """Record a failed check of *target* and drive the retry-budget escape.

    The mirror of :func:`_mark_file_validated`: a checker that ran but exited non-zero
    means the file did NOT pass. Increment its per-file failure count, drop it from
    ``validated_files``, and return the workflow to ``edit`` so the model repairs it.
    When every dirty file is either validated or has exhausted
    ``VALIDATION_RETRY_BUDGET`` attempts, escape to ``conclude`` so a file that cannot be
    made to pass does not wedge the workflow — the model concludes with the residual risk
    stated. This is what re-feeds the (otherwise inert) fail-count that the completion
    gate, the anti-thrashing state guard, and the nudge all read.
    """
    fail_counts = execution_context["validation_fail_count_by_file"]
    fail_counts[target] = int(fail_counts.get(target, 0)) + 1
    execution_context["validated_files"].discard(target)
    execution_context.get("validation_tier_by_file", {}).pop(target, None)
    if not execution_context["code_mutation_started"]:
        return
    set_workflow_state(execution_context, "edit")
    dirty = execution_context.get("dirty_written_files", set())
    validated = execution_context.get("validated_files", set())
    all_stuck = dirty and all(
        f in validated or int(fail_counts.get(f, 0)) >= VALIDATION_RETRY_BUDGET
        for f in dirty
    )
    if all_stuck:
        set_workflow_state(execution_context, "conclude")


def _observe_validation_tool(
    agent: Any, tool_name: str, arguments: dict, status: Any, execution_context: dict[str, Any],
) -> None:
    """A dedicated VALIDATE tool marks its target file validated on success.

    The first-party stack validates through bash (credited in ``_observe_command``),
    but the VALIDATE capability stays declarable so an extension-pack server can ship
    its own validator. When such a tool succeeds on a dirty file, it makes the same
    conclude-gate contribution a successful bash validation command does — no removed
    ladder required. Capability-driven: any tool carrying VALIDATE qualifies, keyed by
    its declared ``path_args`` (falling back to filepath/path).
    """
    if status != "ok" or not has_cap(tool_name, VALIDATE, getattr(agent, "tool_caps", None)):
        return
    arg_names = path_args(tool_name, getattr(agent, "tool_caps", None)) or ("filepath", "path")
    raw = next((arguments.get(a) for a in arg_names if arguments.get(a)), None)
    target = agent._normalize_workspace_path(raw)
    if target and target in execution_context.get("dirty_written_files", set()):
        _mark_file_validated(execution_context, target)


def _observe_missing_module(
    agent: Any, tool_name: str, payload: dict, status: Any, execution_context: dict[str, Any],
) -> None:
    """Flag when a check/run failed for a missing importable module (not a code defect).

    Feeds the env-resolution nudge and the cascade in the system prompt: a missing
    module is recoverable by pointing the check at a different interpreter, so it
    must be distinguished from a real failure. Gated on a failing call so a
    successful read of a file that merely *mentions* the text does not trip it:
    any failing tool result whose text matches the module-not-found pattern (e.g. a
    bash ``python``/``pytest`` that raised ``ModuleNotFoundError``).

    A *successful* execution retracts the flag. Without that, one transient
    ModuleNotFoundError stuck for the rest of the query: ``_exercise_looks_feasible``
    reads this set and would keep the exercise nudges silent long after the model had
    found the right interpreter — the one run that proves the environment resolved is
    exactly the one that used to leave it marked unresolvable.
    """
    if status == "ok":
        if has_cap(tool_name, CODE_EXEC, getattr(agent, "tool_caps", None)):
            execution_context.get("unresolved_modules", set()).clear()
        return
    unresolved = execution_context.setdefault("unresolved_modules", set())
    text_fields = (
        payload.get("stderr"),
        payload.get("stdout"),
        payload.get("error"),
        payload.get("hint"),
        payload.get("message"),
    )
    blob = "\n".join(str(f) for f in text_fields if f)
    if _MODULE_NOT_FOUND_RE.search(blob) is None:
        return
    name_match = _MODULE_NAME_RE.search(blob)
    if name_match:
        unresolved.add(name_match.group(1))
    else:
        # Failure is module-shaped but we could not name the module; record a marker
        # so the nudge still fires.
        unresolved.add("")


def _observe_env_probe(
    agent: Any, tool_name: str, execution_context: dict[str, Any],
) -> None:
    """Record that the agent enumerated the available runtime environments.

    Capability-driven (no tool name): any tool carrying ENV_DISCOVERY counts. Once set,
    the env-resolution nudge stops firing — the agent has the env list it was steered to.
    """
    if has_cap(tool_name, ENV_DISCOVERY, getattr(agent, "tool_caps", None)):
        execution_context["env_probed"] = True


def _observe_env_mutation(
    agent: Any, tool_name: str, payload: dict, status: Any, execution_context: dict[str, Any],
) -> None:
    """Record a successful environment mutation as a cleanup obligation.

    Capability-driven (no tool name): a successful ENV_MUTATE call (package install or
    env creation) appends a record of what was created, which the conclude-phase
    cleanup nudge surfaces so the agent offers to undo it. Pure bookkeeping — the undo
    itself is a separate, user-approved action.
    """
    if status != "ok":
        return
    if not has_cap(tool_name, ENV_MUTATE, getattr(agent, "tool_caps", None)):
        return
    mutations = execution_context.setdefault("env_mutations", [])
    record = {
        "tool": tool_name,
        "installed": payload.get("installed") or [],
        "python": payload.get("python") or "",
        "path": payload.get("path") or "",
        "name": payload.get("name") or "",
    }
    if record not in mutations:
        mutations.append(record)


def _observe_denial_clearing(
    agent: Any, tool_name: str, arguments: dict, status: Any, execution_context: dict[str, Any],
) -> None:
    """Clear a recorded denial once the same tool+path later succeeds."""
    if status != "ok":
        return
    resolved_path = agent._normalize_workspace_path(
        arguments.get("path")
        or arguments.get("filepath")
        or arguments.get("subdir")
        or arguments.get("directory")
    )
    denied_calls = execution_context["denied_tool_calls"]
    execution_context["denied_tool_calls"] = [
        item for item in denied_calls
        if not (
            item.get("tool") == tool_name
            and item.get("path") == resolved_path
        )
    ]
    if not execution_context["denied_tool_calls"]:
        execution_context["nudge_counts"]["denial"] = 0


def _observe_edit_loop_clear(
    agent: Any, tool_name: str, arguments: dict, status: Any, execution_context: dict[str, Any],
) -> None:
    """Re-reading or re-checking a file resets its edit-loop failure tracking."""
    if not (clears_edit_loop(tool_name, agent.tool_caps) and status == "ok"):
        return
    read_path = agent._normalize_workspace_path(arguments.get("path") or arguments.get("filepath"))
    if read_path:
        execution_context["edit_loop_state"].pop(read_path, None)
        _clear_edit_fail_streak(execution_context, read_path)


def _observe_delete(
    agent: Any, tool_name: str, arguments: dict, status: Any, execution_context: dict[str, Any],
) -> None:
    if not (has_cap(tool_name, REMOVE, agent.tool_caps) and status == "ok"):
        return
    deleted_path = agent._normalize_workspace_path(arguments.get("path"))
    if not deleted_path:
        return
    # Every field that holds file paths, derived from the FILE_PATH trait: a deleted
    # file must not survive anywhere as evidence, and hand-listing the fields meant a
    # new path-valued field would silently keep stale entries.
    for field in fields_with(FILE_PATH):
        execution_context[field].discard(deleted_path)
    agent._discard_carry_path(deleted_path)


def _observe_search_flags(
    agent: Any, tool_name: str, arguments: dict, status: Any, execution_context: dict[str, Any],
) -> None:
    if not (has_cap(tool_name, SEARCH, agent.tool_caps) and status == "ok"):
        return
    execution_context["searched"] = True
    execution_context["search_tool_calls"] += 1


def _observe_discover_transition(payload: dict, execution_context: dict[str, Any]) -> None:
    """Leave discover state once any tool returns concrete results."""
    if execution_context["workflow_state"] != "discover":
        return
    search_has_results = bool(
        payload.get("matches")
        or payload.get("files")
        or payload.get("results")
        or payload.get("lines_returned")
        or payload.get("tree")
        or payload.get("entries")
    )
    if search_has_results:
        set_workflow_state(execution_context, "edit")


def _observe_candidates(
    agent: Any, tool_name: str, payload: dict, execution_context: dict[str, Any],
) -> None:
    if not has_cap(tool_name, CANDIDATE_SEARCH, agent.tool_caps):
        return
    for item in payload.get("matches", []):
        if not isinstance(item, dict):
            continue
        candidate = agent._normalize_workspace_path(item.get("path") or item.get("file"))
        if not candidate:
            continue
        _register_known_path(execution_context, candidate)
        candidate_parent = agent._parent_path(candidate)
        execution_context["similar_candidates_by_dir"].setdefault(candidate_parent, set()).add(candidate)


def _observe_dir_inspect(
    agent: Any, tool_name: str, status: Any, path: str, payload: dict,
    execution_context: dict[str, Any],
) -> None:
    if not (has_cap(tool_name, INSPECT_DIR, agent.tool_caps) and status == "ok"):
        return
    if path:
        execution_context["inspected_dirs"].add(path)
    listed_dir = agent._normalize_workspace_path(payload.get("path") or payload.get("parent"))
    if listed_dir:
        execution_context["inspected_dirs"].add(listed_dir)


def _observe_existence_check(
    agent: Any, tool_name: str, arguments: dict, status: Any, payload: dict,
    execution_context: dict[str, Any],
) -> None:
    if not (has_cap(tool_name, CHECK_EXISTENCE, agent.tool_caps) and arguments.get("path")):
        return
    checked = agent._normalize_workspace_path(arguments.get("path"))
    if not checked:
        return
    if status == "ok" or payload.get("exists") is not None:
        execution_context["checked_paths"].add(checked)
    if payload.get("exists") is True:
        _register_known_path(execution_context, checked)


def _observe_read(
    agent: Any, tool_name: str, arguments: dict, status: Any,
    execution_context: dict[str, Any],
) -> None:
    if not (has_cap(tool_name, READ, agent.tool_caps) and status == "ok"):
        return
    path = agent._normalize_workspace_path(arguments.get("path"))
    if not path:
        return

    execution_context["read_files"].add(path)
    _register_known_path(execution_context, path)
    execution_context["checked_paths"].add(path)


def _observe_delegated_exploration(
    agent: Any, tool_name: str, payload: dict, status: Any,
    execution_context: dict[str, Any],
) -> None:
    """Credit what a sub-agent read back to the caller that sent it.

    These gates ask whether the model has facts about the code or is working off file
    names. A file a child opened, and whose finding came back into this conversation,
    is such a fact — so it is discovery evidence here. It stays out of ``read_files``,
    which also answers "does this agent hold the lines it is about to edit": that one
    only a read of its own can settle.
    """
    if not (has_cap(tool_name, DELEGATE, agent.tool_caps) and status == "ok"):
        return
    for raw in payload.get("files_read") or ():
        path = agent._normalize_workspace_path(raw)
        if path:
            execution_context["delegated_read_files"].add(path)
            _register_known_path(execution_context, path)


def _observe_declared_edit_set(
    agent: Any, tool_name: str, arguments: dict, status: Any, execution_context: dict[str, Any],
) -> None:
    """Extract source-file paths named in a task-checklist declaration into the declared edit set.

    The checklist tool is the planning tool that declares a ``plan_steps`` arg-role (the
    ordered steps); the prose-rationale tool carries no steps and is skipped.
    """
    if status != "ok":
        return
    step_args = arg_role(tool_name, "plan_steps", agent.tool_caps)
    if not step_args:
        return
    raw_steps = next((arguments.get(a) for a in step_args if arguments.get(a)), "")
    if isinstance(raw_steps, list):
        raw_steps = " ".join(str(s) for s in raw_steps)
    for match in _SOURCE_FILE_PATH_RE.finditer(str(raw_steps)):
        candidate = agent._normalize_workspace_path(match.group(0))
        if candidate:
            execution_context["declared_edit_set"].add(candidate)


def _observe_action_op(
    agent: Any, tool_name: str, status: Any, execution_context: dict[str, Any],
) -> None:
    """Count each successful substantive action (write / execution / mutation).

    PLAN_BLOCKED is the universal marker for a non-discovery, side-effecting tool:
    file writers derive it, and exec/db/web/cluster/proxy/hpc tools declare it. The
    todo nudge reads this count so a task made of many such operations — e.g. an
    optimisation loop that edits a single file but runs many benchmarks/evals — is
    still recognised as multi-step and prompted to record a checklist.
    """
    if status == "ok" and has_cap(tool_name, PLAN_BLOCKED, agent.tool_caps):
        execution_context["action_op_count"] = int(execution_context.get("action_op_count", 0)) + 1


def _carries_shell_command(agent: Any, tool_name: str) -> tuple[str, ...] | None:
    """Command-arg names if *tool_name* is a shell-command tool, else None.

    Registry-driven (no tool name in code): a tool that takes a raw shell command
    declares a ``command_prefix`` scope kind (see server_bash's ``bash_run``); the
    scope's ``args`` name the command argument. Returns those names so the observer
    can classify the command a call carried.
    """
    spec = scope_spec(tool_name, agent.tool_caps)
    if not spec or spec.get("kind") != "command_prefix":
        return None
    return tuple(spec.get("args") or ("command",))


def _run_key(tool_name: str, payload: dict, spec: dict[str, Any] | None) -> str:
    """The ledger key for a run this tool performed.

    A non-shell execution tool has no command line to name its runs, so every call
    used to collapse onto the tool name — twenty optimisation iterations became one
    entry, the last overwriting the rest. When the tool declares which payload field
    identifies the run, that identifier *is* the key.

    Deliberately not prefixed with the tool name: the tool that launches a run and the
    tool that later reports how it went are different ones, and a per-tool key gave them
    separate entries — the machine outcome landed on a second row while the row from the
    launch stayed green and settleable by a ``pass``. The subject is the run.
    """
    field = (spec or {}).get("id")
    raw = payload.get(field) if field else None
    if not isinstance(raw, str) or not raw.strip():
        return tool_name
    return raw.rstrip(os.sep)


def _matches(payload: dict, spec: dict[str, Any], kind: str) -> bool:
    """Whether *payload* meets the ``<kind>_when`` conditions declared in *spec*.

    Two forms: a field/values map, and a ``_present`` list of fields whose mere
    presence and truthiness is the signal — a per-case error string has no
    enumerable value set to match against.
    """
    for field, values in (spec.get(f"{kind}_when") or {}).items():
        if field in payload and payload[field] in values:
            return True
    return any(payload.get(field) for field in (spec.get(f"{kind}_when_present") or ()))


def _observe_tool_run(
    agent: Any, tool_name: str, arguments: dict, status: Any,
    execution_context: dict[str, Any], call_id: str = "", payload: dict | None = None,
) -> None:
    """A non-shell execution tool ran: its output owes a verdict too.

    The counterpart of :func:`_observe_bash_validation`, split by *surface* rather than
    by purpose: a shell tool's calls differ in kind call by call (``cat`` reads,
    ``python`` executes), so only the command text can decide, and that function already
    has the parse. A tool like ``proxy_exec`` or ``ft_run`` has no such variation — the
    call *is* the execution — so the declared capability decides. The two are mutually
    exclusive, which is what keeps a bash run from being registered twice and spares a
    third shlex parse per call.

    Recorded against the tool name, plus the run identifier the tool declares in its
    ``run_outcome`` spec so repeated calls do not collapse onto one entry. It validates
    no file: an execution's exit code says the program reached its end, and which file
    it exercised is not a question anyone here can answer.
    """
    if status != "ok" or _carries_shell_command(agent, tool_name) is not None:
        return
    if not has_cap(tool_name, CODE_EXEC, agent.tool_caps):
        return
    spec = run_outcome_spec(tool_name, getattr(agent, "tool_caps", None))
    _record_run_outcome(
        execution_context, _run_key(tool_name, payload or {}, spec), True, call_id)


def _observe_run_outcome(
    agent: Any, tool_name: str, payload: dict,
    execution_context: dict[str, Any], call_id: str = "",
) -> None:
    """Record what the *server* saw of a run it performed itself.

    The counterpart of ``observed_failure_verdict`` for a non-shell execution tool.
    A shell run is floored by its exit code; a tool like ``proxy_eval`` answers ``ok``
    for the *call* even when the run it launched came back red, so without this the
    ledger held no machine reading at all and a stated ``pass`` was uncontradicted.

    One-way, like every other floor here: it can move a run to crashed or failing,
    never settle one as passing. That asymmetry is what makes it safe to trust — and
    it needs no enforcement of its own, since ``unsettled_runs`` already excludes a run
    that is not ``completed`` or already carries a verdict, putting both beyond the
    reach of a model's ``pass``.

    Takes no call status by design: a read that reports a crashed run says the same
    thing whether or not the call carrying it succeeded.
    """
    spec = run_outcome_spec(tool_name, getattr(agent, "tool_caps", None))
    if not spec or not isinstance(payload, dict):
        return
    _settle_from_machine(execution_context, tool_name, payload, spec, call_id)
    _credit_measured_source(execution_context, payload, spec)
    rows_spec = spec.get("rows")
    if not isinstance(rows_spec, dict):
        return
    rows = payload.get(rows_spec.get("field", ""))
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            _settle_from_machine(execution_context, tool_name, row, rows_spec, call_id)


def _credit_measured_source(
    execution_context: dict[str, Any], payload: dict, spec: dict[str, Any],
) -> None:
    """Raise the optimisation session's source file to the ``measured`` tier.

    The exception to "running the code validates no file": that rule exists because
    attribution is a guess (``python main.py`` exercises ``mesh.py`` without naming it),
    and here it is not one — the session config names exactly one source and the run is
    by construction a run of it. Nothing else may take this route.

    Credits evidence, not correctness: it records that the file was exercised and
    measured. Whether the measurement was any good is a verdict on the run, settled
    elsewhere and never by the server itself.
    """
    if _matches(payload, spec, "failed") or _matches(payload, spec, "crashed"):
        return
    if not _matches(payload, spec, "measured"):
        return
    from .policy.gates import active_proxy_source

    source = active_proxy_source()
    if source and source in execution_context.get("dirty_written_files", set()):
        _mark_file_validated(execution_context, source, "measured")


def _settle_from_machine(
    execution_context: dict[str, Any], tool_name: str, payload: dict,
    spec: dict[str, Any], call_id: str,
) -> None:
    """Apply one declared run outcome to the ledger.

    A crash did not complete and drives the repair ladder. A failing result *did*
    complete — the program ran, its answer is unsatisfactory — so it is recorded as a
    machine ``fail`` verdict, the same shape a model's own ``fail`` takes.

    Anything else is left exactly as it was: notably, an optimisation run that measured
    correctly without improving on the incumbent is the ordinary outcome of an
    experiment, and charging it would punish most of them.
    """
    crashed = _matches(payload, spec, "crashed")
    failed = _matches(payload, spec, "failed")
    if not crashed and not failed:
        return
    key = _run_key(tool_name, payload, spec)
    if crashed:
        _record_run_outcome(
            execution_context, key, False, call_id, _MACHINE_SAW_CRASH)
        return
    _record_run_outcome(execution_context, key, True, call_id)
    run = (execution_context.get("runs") or {}).get(key)
    if run is not None:
        run["verdict"] = "fail"
        run["reason"] = _MACHINE_SAW_FAILURE
        _register_run_failure(execution_context, key, _MACHINE_SAW_FAILURE)


# Checkers that cover a whole tree in one call. A *successful* run of one naming no
# specific source file (`ruff check .`, `mypy src/`) clears every pending file at once;
# per-file crediting is already language-agnostic, this set only adds the shortcut.
#
# INVARIANT (test-enforced): every entry is in `shell_paths.EXEC_EFFECTS` with effect
# `validate`, and in `_VALIDATOR_TIER` below. That is also why `pytest` and `ctest` are
# not here: they are declared `run`, and running a suite is judged on its own output —
# it credits no file however green it exits.
_PROJECT_VALIDATORS = frozenset({
    "ruff", "mypy", "pyflakes", "black", "py_compile",
})
# A positional that names a specific file (has a dotted extension), vs a directory/`.`.
_FILE_EXT_RE = _re.compile(r"\.[A-Za-z0-9]+$")

# The checkers, keyed by command head, and what kind of check each one is. Reached only
# for a segment `shell_paths.EXEC_EFFECTS` already declared a *validator*: a head absent
# from this table credits nothing, it does not thereby become an execution.
_VALIDATOR_TIER = {
    "py_compile": "syntax",
    "ruff": "static", "mypy": "static", "pyflakes": "static", "black": "static",
    # A compiler is a checker, not a run: it emits diagnostics and exit 0 means there
    # were none. Like the linters, its output *is* the verdict — nothing was executed,
    # so there is no result for anyone to judge. It sits on its own tier because it is
    # the one check that needs a toolchain to exist: demanded of nobody, credited when
    # it happens.
    "gcc": "compiled", "g++": "compiled", "gfortran": "compiled", "nvcc": "compiled",
    "javac": "compiled",
}
# Reformatting is not checking. `ruff format` and a bare `black` rewrite the file and
# exit 0 whether or not the code is correct — so crediting them was a pass awarded for
# moving whitespace, and awarded in the very files being moved: `ruff format .` matched
# the project-wide shortcut above and cleared every pending file at tier `static`.
# `--check`/`--diff` are the exception in both tools: they report instead of writing,
# which is what a checker does.
_REPORT_ONLY_FLAGS = ("--check", "--diff")


def _rewrites_instead_of_checking(head: str, argv: tuple[str, ...]) -> bool:
    """True when this validator invocation formats the file rather than judging it."""
    if head not in ("ruff", "black"):
        return False
    if any(flag in argv for flag in _REPORT_ONLY_FLAGS):
        return False
    if head == "black":
        return True
    # ruff formats only under its `format` sub-command; bare `ruff` and `ruff check`
    # are checks. The head is not argv[0] for `python -m ruff`, so read the sub-command
    # after the head rather than at a fixed index.
    rest = argv[argv.index(head) + 1:] if head in argv else argv
    return next((a for a in rest if not a.startswith("-")), "") == "format"


# What turns a compiler invocation back into the cheap, always-affordable check: it
# parses and type-checks the translation unit without emitting an object, so it is the
# mandatory floor for a compiled language the way ``py_compile`` is for Python.
_SYNTAX_ONLY_FLAGS = ("-fsyntax-only", "--fsyntax-only")

# Reason recorded when the run itself declared the failure, so the model is not asked to
# judge output that already carries its own verdict.
_SELF_DECLARED_FAILURE = "the run's own output declared check=fail"
_BUILD_EXIT_IS_THE_VERDICT = "the build exited 0; a build reports diagnostics, not results"
# A server that ran the code itself reported the run red. Its own judgement, not a
# reading of what the program printed — the floor a stated verdict cannot rise above.
_MACHINE_SAW_CRASH = "the run server reported this run did not complete"
_MACHINE_SAW_FAILURE = "the run server judged this run's result failing"


def _outside_workspace(path: str) -> bool:
    """True for a path ``_normalize_workspace_path`` left outside the user's tree.

    Normalization rebases anything under the workspace to a relative path, so what
    stays absolute — or climbs out with ``..`` — is by construction not workspace code.
    The scratchpad above all, which is where the prompt sends every throwaway probe.
    """
    return os.path.isabs(path) or path == ".." or path.startswith(".." + os.sep)


def _bash_validation_scan(
    agent: Any, command: str,
) -> tuple[list[str], bool, str, str, str]:
    """Scan a bash command, telling checks from builds from executions.

    Returns ``(explicit_targets, whole_project, tier, execution, missing_head)``:
    - ``explicit_targets`` — normalized workspace paths of the files named by EXEC
      segments, with a leading ``cd`` rebasing later relative operands (``cd sub &&
      ruff check t.py`` → ``sub/t.py``).
    - ``whole_project`` — True when a recognised project **checker** ran over no specific
      file (``ruff check .``, ``mypy src/``): a green run then validates every pending
      file, not just a named one.
    - ``tier`` — the strongest check any segment performed, or ``""`` when none did.
    - ``execution`` — ``EFFECT_RUN`` when a segment ran the project's code, ``EFFECT_BUILD``
      when the strongest thing it did was drive a build, ``""`` when nothing was executed.
      A run outranks a build in the same chain (``make && ./solver``): the build's exit
      code settles it, the run's output still owes a reading. Which of the two a segment
      is comes from :data:`shell_paths.EXEC_EFFECTS`, declared beside the command
      groups — never from a head's absence from the tier table below, which is how
      ``source env.sh`` came to be recorded as a run owing a verdict.
    - ``missing_head`` — the first command in the chain that is not installed here, or
      ``""``. The one imputation the machine can settle alone: a command the shell could
      not find said nothing about the change, so its red exit is a wall, not a defect.
      Three conditions, each guarding a real trap. It reads ``argv[0]``, never ``head``,
      which is the *module* for ``python -m X`` and so never on PATH (``py_compile`` is a
      checker with no command of its own). It skips a path-like argv0, because a missing
      ``./solver`` is a build that did not happen, which is attributable. And it only looks
      at segments that ran, built or checked something — ``source`` is a shell builtin and
      would otherwise fail this test on every chain that sets up an environment.
    For a command the classifier cannot read at all, the first three are empty — nothing
    is attributable — but ``execution`` still comes from its command-position heads
    (:func:`opaque_command_executes`). Opacity is a reason not to credit; it is not
    a reason to believe nothing ran, and inline payloads (which the base prompt asks for)
    are opaque by construction.
    """
    segments = classify_bash_command(command)
    if not segments:
        return [], False, "", EFFECT_RUN if opaque_command_executes(command) else "", ""
    rel_base = ""
    targets: list[str] = []
    whole_project = False
    tier_rank = -1
    execution = ""
    missing_head = ""
    for seg in segments:
        if seg.kind == Kind.CHDIR:
            new_dir = seg.operands[0] if seg.operands else ""
            if new_dir:
                rel_base = new_dir if os.path.isabs(new_dir) else os.path.normpath(
                    os.path.join(rel_base, new_dir) if rel_base else new_dir
                )
            continue
        # UNKNOWN is an execution too: a head the taxonomy does not place ran
        # something, and a run owes a verdict whether or not we can name it.
        if seg.kind not in (Kind.EXEC, Kind.UNKNOWN):
            continue
        if (
            not missing_head
            and seg.effect in (EFFECT_RUN, EFFECT_BUILD, EFFECT_VALIDATE)
            and seg.argv
            and not is_path_like_command(seg.argv[0])
            and not _any_command_on_path((seg.argv[0],))
        ):
            missing_head = seg.argv[0]
        for op in seg.operands:
            based = op if (os.path.isabs(op) or not rel_base) else os.path.join(rel_base, op)
            path = agent._normalize_workspace_path(based)
            if path:
                targets.append(path)
        if seg.effect == EFFECT_RUN:
            execution = EFFECT_RUN
            continue
        if seg.effect == EFFECT_BUILD:
            execution = execution or EFFECT_BUILD
            continue
        if seg.effect != EFFECT_VALIDATE:
            continue  # env setup: preparation, not evidence
        seg_tier = _VALIDATOR_TIER.get(seg.head)
        if seg_tier is None:
            continue
        if _rewrites_instead_of_checking(seg.head, seg.argv):
            continue
        if seg_tier == "compiled" and any(f in seg.argv for f in _SYNTAX_ONLY_FLAGS):
            seg_tier = "syntax"
        tier_rank = max(tier_rank, VALIDATION_TIERS.index(seg_tier))
        if seg.head in _PROJECT_VALIDATORS and not any(_FILE_EXT_RE.search(op) for op in seg.operands):
            whole_project = True
    tier = VALIDATION_TIERS[tier_rank] if tier_rank >= 0 else ""
    return targets, whole_project, tier, execution, missing_head


def _record_run_outcome(
    execution_context: dict[str, Any], command: str, completed: bool,
    call_id: str = "", reason: str = "", effect: str = EFFECT_RUN,
    missing_head: str = "",
) -> None:
    """Register an execution and, when it did not complete, charge the repair budget.

    A run that exited non-zero is already judged — by the machine, in the only direction
    an exit code is trustworthy — so it owes no verdict and goes straight onto the
    repair ladder. A green one owes a reading of what it printed.

    A **build** owes nothing in either direction: like the compilers it drives, what it
    reports is diagnostics, and a green exit means the artefacts were produced. It is
    still recorded — a failed build is a wall the user must see — but settled by the
    machine, so it can never surface as "ran but never judged".

    ``missing_head`` is the one wall the machine can name without being told: a command
    that is not installed said nothing about the patch, so it charges no repair budget and
    steers nothing. Every other wall needs the model to claim it (``guardrails/verdict.py``);
    unclaimed, a red exit drives the repair ladder exactly as it always did.
    """
    run = record_run(execution_context, command, completed=completed, call_id=call_id, effect=effect)
    if completed and effect == EFFECT_BUILD:
        run["verdict"] = "pass"
        run["reason"] = _BUILD_EXIT_IS_THE_VERDICT
    if not completed:
        if missing_head:
            run["blocked"] = f"{missing_head} is not installed here"
            execution_context["exercise_blocked_reason"] = run["blocked"]
            execution_context["exercise_advice_closed"] = True
        else:
            _register_run_failure(execution_context, command, reason)


def _register_run_failure(
    execution_context: dict[str, Any], command: str, reason: str = "",
) -> None:
    """Charge a run's failure against the repair budget and steer back to ``edit``.

    Same shape as the per-file budget: try, try again, then stop trying. Past
    ``VALIDATION_RETRY_BUDGET`` attempts at the same command the workflow is released to
    ``conclude`` rather than wedged — the run is reported unresolved with what was
    tried, and the answer carries the residual risk instead of looping on it.
    """
    run = (execution_context.get("runs") or {}).get(command)
    if run is None:
        return
    run["failures"] = int(run.get("failures", 0)) + 1
    if reason:
        run.setdefault("attempts", []).append(reason)
    if not execution_context.get("code_mutation_started"):
        return
    set_workflow_state(execution_context, "edit")
    if all(
        int(r.get("failures", 0)) >= VALIDATION_RETRY_BUDGET
        for r in failed_runs(execution_context).values()
    ) and not has_pending_validation(execution_context):
        set_workflow_state(execution_context, "conclude")


def _observe_bash_validation(
    agent: Any, tool_name: str, arguments: dict, status: Any, payload: dict,
    execution_context: dict[str, Any], call_id: str = "",
) -> None:
    """Drive validation and run state from a bash command — success AND failure.

    Two axes, never mixed, because they answer different questions:

    - a **checker** (``py_compile`` → syntax; ``ruff``/``mypy``/a compiler → static)
      validates the *files it names*. Its output is a list of problems and an empty one
      is the finding, so exit 0 settles it and nobody has to read anything. Non-zero
      charges the file's retry budget.
    - an **execution** (``pytest``, ``python solver.py``, ``./solver``, ``python -c …``)
      validates no file at all. It is recorded as a *run*: a non-zero exit is a failure
      the machine already judged, and a green one owes the model a reading of what it
      printed (``guardrails/verdict.py``). Which file it exercised is not asked — the
      run is the subject, and guessing an attribution is what used to let one green
      benchmark credit every file edited beforehand.

    A green project-wide **checker** (``ruff check .``) covers every pending file at
    once. Test files are recorded in ``tests_run`` either way (pass or fail) so the
    regression nudge sees them. Only shell-command tools qualify.
    """
    command_args = _carries_shell_command(agent, tool_name)
    if command_args is None:
        return
    command = str(arguments.get(command_args[0], "") or "")
    explicit, whole_project, tier, execution, missing_head = _bash_validation_scan(
        agent, command,
    )
    # `execution` (not just a named target) is what catches `python -c …` and a bare
    # `./solver`: they validate nothing, but they produced output somebody must judge.
    if not explicit and not whole_project and not execution:
        return
    stdout = str(payload.get("stdout") or "")
    # A run that computes its own criteria, reports them unmet and still returns 0 is a
    # green exit over a red result — the shape of the failure this guards: a self-written
    # boundary test printed "significant reflection may be present", exited 0, and was
    # recorded as validated. A declared verdict therefore outranks the exit code, one way
    # only: `check=fail` demotes a pass, `check=pass` never rescues a failure.
    declared_failure = status == "ok" and observed_failure_verdict(stdout)
    if declared_failure:
        status = "error"
    dirty = execution_context.get("dirty_written_files", set())
    for target in explicit:
        if is_python_test_filepath(target):
            execution_context["tests_run"].add(target)
        if not tier or target not in dirty:
            continue
        if status != "ok":
            # A specific-file failure is attributable; a whole-project failure is not.
            _register_validation_failure(execution_context, target, tier)
        else:
            _mark_file_validated(execution_context, target, tier)
    # A green project-wide check covers every still-pending file. A failing one is not
    # attributable to a single file, so it leaves pending untouched (the model narrows
    # down or re-runs per file).
    if tier and whole_project and status == "ok":
        for target in list(pending_validation_paths(execution_context)):
            _mark_file_validated(execution_context, target, tier)
    if execution:
        _record_run_outcome(
            execution_context, command, status == "ok", call_id,
            _SELF_DECLARED_FAILURE if declared_failure else "",
            effect=execution, missing_head=missing_head,
        )


def _observe_command(
    agent: Any, tool_name: str, arguments: dict, status: Any, payload: dict,
    execution_context: dict[str, Any],
) -> None:
    """Credit the blackboard from a shell command classified into capability kinds.

    A dual-use bash tool hides its effect inside a ``command`` string, so the other
    ``_observe_*`` handlers (which key off static caps + a named path arg) miss it.
    ``classify_bash_command`` maps each ``; && || |`` segment to a kind + operands;
    this mirrors the dedicated observers so a bash ``cat``/``grep``/``sed -i`` feeds
    the same discovery/edit/action state the file/search tools would.
    """
    if status != "ok":
        return
    command_args = _carries_shell_command(agent, tool_name)
    if command_args is None:
        return
    command = arguments.get(command_args[0], "")
    segments = classify_bash_command(command)
    if not segments:
        return

    # Track the shell's cwd across the chain: a ``cd sub`` rebases the relative path
    # operands of every later segment, so they normalize to the same workspace path the
    # dedicated file/search observers record. Applied to every path-bearing kind.
    rel_base = ""

    def _resolve(op: str) -> str | None:
        if not op:
            return None
        based = op if (os.path.isabs(op) or not rel_base) else os.path.join(rel_base, op)
        return agent._normalize_workspace_path(based)

    for seg in segments:
        kind, operands = seg.kind, seg.operands
        if kind == Kind.CHDIR:
            new_dir = operands[0] if operands else ""
            if new_dir:
                rel_base = new_dir if os.path.isabs(new_dir) else os.path.normpath(
                    os.path.join(rel_base, new_dir) if rel_base else new_dir
                )
            continue
        if kind == Kind.READ:
            for op in operands:
                path = _resolve(op)
                if path:
                    execution_context["read_files"].add(path)
                    execution_context["checked_paths"].add(path)
                    _register_known_path(execution_context, path)
        elif kind == Kind.SEARCH:
            execution_context["searched"] = True
            execution_context["search_tool_calls"] += 1
        elif kind == Kind.INSPECT:
            for op in operands:
                path = _resolve(op)
                if path:
                    execution_context["inspected_dirs"].add(path)
        elif kind == Kind.WRITE:
            execution_context["action_op_count"] = int(execution_context.get("action_op_count", 0)) + 1
            for op in operands:
                path = _resolve(op)
                if path and agent._is_code_filepath(path):
                    _record_code_edit(execution_context, path)
                    execution_context["last_edit_success_path"] = path
                    _enter_post_edit_state(execution_context)
        elif kind in (Kind.EXEC, Kind.UNKNOWN):
            # Marking a dirty file validated (or failed) is handled status-agnostically
            # by _observe_bash_validation; here we only count the substantive action.
            execution_context["action_op_count"] = int(execution_context.get("action_op_count", 0)) + 1
        elif kind == Kind.ENV_DISCOVERY:
            execution_context["env_probed"] = True
        elif kind == Kind.ENV_MUTATE:
            mutations = execution_context.setdefault("env_mutations", [])
            record = {
                "tool": tool_name, "installed": list(operands),
                "python": "", "path": "", "name": "",
            }
            if record not in mutations:
                mutations.append(record)

    # Mirror _observe_discover_transition (which matches result payloads the bash
    # tool doesn't emit): a read/search/inspect that produced output leaves discover.
    if execution_context["workflow_state"] == "discover" and payload.get("stdout"):
        if any(seg.kind in (Kind.READ, Kind.SEARCH, Kind.INSPECT) for seg in segments):
            set_workflow_state(execution_context, "edit")


def _observe_verdict_tool(
    agent: Any, tool_name: str, arguments: dict, status: Any,
    execution_context: dict[str, Any],
) -> None:
    """The model judged a run's output: settle the run it speaks for.

    The judging tool is found by capability and read through its declared arg-roles, so
    neither the tool's name nor its parameter names appear here — the server owns both.
    Its own result carries nothing: what the call *means* is state on the blackboard,
    which is what this writes.

    Settled runs are announced per call id so the UI can badge the run's own row; a
    statement that settled nothing (an unqualified ``pass`` over runs bearing on
    different files) announces nothing, and the nudge asks for the missing scope.
    """
    if status != "ok" or not has_cap(tool_name, JUDGE, agent.tool_caps):
        return
    # Local import: verdict.py builds on this module's ladder entry points, so the
    # dependency only runs the other way at call time.
    from .verdict import apply_verdict

    def _arg(role: str) -> str:
        names = arg_role(tool_name, role, agent.tool_caps)
        return str(arguments.get(names[0], "") or "").strip() if names else ""

    verdict = _arg("verdict").lower()
    settled = apply_verdict(verdict, _arg("verdict_reason"), _arg("verdict_scope"), execution_context)
    for run in settled:
        call_id = run.get("call_id")
        if call_id:
            emit({"type": "verdict", "id": call_id, "verdict": verdict})


def record_tool_observation(
    agent: Any,
    tool_name: str,
    arguments: dict,
    result: str,
    execution_context: dict[str, Any] | None,
    call_id: str = "",
) -> None:
    execution_context = ensure_execution_context(execution_context)
    execution_context = (
        bootstrap_runtime_context(execution_context) if execution_context is not None else None
    )
    if execution_context is None:
        return

    payload = agent._parse_tool_payload(result) or {}
    path = agent._normalize_workspace_path(
        arguments.get("path") or arguments.get("subdir") or arguments.get("directory")
    )
    status = payload.get("status")

    # Order is load-bearing — see the handler-section note above.
    _observe_edit_outcome(agent, tool_name, arguments, status, execution_context)
    _observe_todo_flags(agent, tool_name, status, execution_context)
    _observe_replacement_tracking(agent, tool_name, arguments, status, execution_context)
    _observe_validation_tool(agent, tool_name, arguments, status, execution_context)
    _observe_missing_module(agent, tool_name, payload, status, execution_context)
    _observe_env_probe(agent, tool_name, execution_context)
    _observe_env_mutation(agent, tool_name, payload, status, execution_context)
    _observe_denial_clearing(agent, tool_name, arguments, status, execution_context)
    _observe_edit_loop_clear(agent, tool_name, arguments, status, execution_context)
    _observe_delete(agent, tool_name, arguments, status, execution_context)
    _observe_search_flags(agent, tool_name, arguments, status, execution_context)
    _observe_discover_transition(payload, execution_context)
    _observe_candidates(agent, tool_name, payload, execution_context)
    _observe_dir_inspect(agent, tool_name, status, path, payload, execution_context)
    _observe_existence_check(agent, tool_name, arguments, status, payload, execution_context)
    _observe_read(agent, tool_name, arguments, status, execution_context)
    _observe_delegated_exploration(agent, tool_name, payload, status, execution_context)
    _observe_declared_edit_set(agent, tool_name, arguments, status, execution_context)
    _observe_bash_validation(agent, tool_name, arguments, status, payload, execution_context, call_id)
    _observe_command(agent, tool_name, arguments, status, payload, execution_context)
    _observe_action_op(agent, tool_name, status, execution_context)
    _observe_tool_run(agent, tool_name, arguments, status, execution_context, call_id, payload)
    _observe_run_outcome(agent, tool_name, payload, execution_context, call_id)
    _observe_verdict_tool(agent, tool_name, arguments, status, execution_context)
