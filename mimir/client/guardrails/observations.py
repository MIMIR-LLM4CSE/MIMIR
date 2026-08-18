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
    fields_with,
    raise_validation_tier,
)
from ...servers._shared.numerics import observed_failure_verdict, observed_invariant_metrics
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
    EDIT,
    ENV_DISCOVERY,
    ENV_MUTATE,
    INSPECT_DIR,
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
    scope_spec,
)
from ..tool_execution.validation import is_python_test_filepath, is_scratch_path
from .policy.bash_classify import Kind, classify_bash_command


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

    Shared by the single-file edit path (:func:`_observe_edit_outcome`) and the
    batch-edit per-sub-path loop (:func:`_observe_apply_edits`), which recorded an
    identical slice. The caller owns the post-edit transition (see
    :func:`_enter_post_edit_state`).
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
    # Evidence is about a specific revision: rewriting the file retracts it. The attempt
    # log is deliberately not retracted — it is the record of what was already tried.
    execution_context.get("validation_tier_by_file", {}).pop(edited_path, None)
    execution_context.get("verdict_by_file", {}).pop(edited_path, None)
    execution_context["edit_loop_state"].pop(edited_path, None)
    _clear_edit_fail_streak(execution_context, edited_path)


def _enter_post_edit_state(execution_context: dict[str, Any]) -> None:
    """Reset the idle counter and move to validate (if the declared set is done) or edit."""
    execution_context["steps_since_last_edit"] = 0
    if _should_enter_validate(execution_context):
        set_workflow_state(execution_context, "validate")
    else:
        set_workflow_state(execution_context, "edit")


def _observe_edit_outcome(
    agent: Any, tool_name: str, arguments: dict, status: Any, execution_context: dict[str, Any],
) -> None:
    """Record the outcome of a single-file code edit.

    Success: mark the file dirty and advance the workflow state.
    Failure: track repeated identical patches; after two, drop the file from read
    context so the model is forced to re-read (stale context is the usual cause).
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
    streak = int(execution_context["edit_fail_streak_by_file"].get(edited_path, 0)) + 1
    execution_context["edit_fail_streak_by_file"][edited_path] = streak
    # After 2 consecutive failures on the file (any patch) force the model to re-read it:
    # drop it from the discovery sets so the discovery gate demands a fresh read.
    if streak >= 2:
        execution_context["read_files"].discard(edited_path)
        execution_context["snippet_read_files"].discard(edited_path)


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


def _observe_apply_edits(
    agent: Any, tool_name: str, arguments: dict, status: Any, execution_context: dict[str, Any],
) -> bool:
    """Register the confirmed sub-paths of a batch-edit call.

    A batch edit is any edit tool that declares an ``edit_batch`` arg-role — the arg
    holding its list of sub-edits — rather than a path arg. Returns True to halt all
    further observation: preview-only mode (confirm=False) must not mark files dirty or
    trigger any downstream transition.
    """
    batch_args = arg_role(tool_name, "edit_batch", agent.tool_caps)
    if not (batch_args and status == "ok"):
        return False
    # Preview-only mode must not mark files as dirty
    if not arguments.get("confirm", False):
        return True
    edits_raw = arguments.get(batch_args[0], "[]")
    try:
        sub_edits = json.loads(edits_raw) if isinstance(edits_raw, str) else edits_raw
    except (json.JSONDecodeError, TypeError):
        sub_edits = []

    has_code_edit = False
    for sub in (sub_edits if isinstance(sub_edits, list) else []):
        sub_path = agent._normalize_workspace_path(sub.get("path"))
        if sub_path and agent._is_code_filepath(sub_path):
            has_code_edit = True
            _record_code_edit(execution_context, sub_path)

    if has_code_edit:
        _enter_post_edit_state(execution_context)
    return False


def _mark_file_validated(
    execution_context: dict[str, Any], target: str, tier: str = "executed",
) -> None:
    """Credit *target* as validated and advance the workflow when nothing is pending.

    Validation no longer runs through dedicated validator tools: the model exercises a
    written file via bash (``python -m py_compile`` / ``pytest`` / ``ruff`` / ``mypy``),
    and a *successful* run (the bash server only returns ``status == "ok"`` on exit 0)
    is the signal that the file passed. A failing run returns ``status == "error"`` and
    never reaches this path, so the file simply stays pending and the validation nudge
    keeps steering — the same conclude-gate as before, now fed from bash.

    *tier* records how much that green run actually proved (see ``VALIDATION_TIERS``).
    It has no effect on the conclude-gate — every tier still counts as validated,
    exactly as before — and is read only by the completion ledger. The default suits
    the extension-pack VALIDATE path, whose tool ran *something* but tells us no more.

    **Red→green promotion.** A check that was observed failing on this file and now
    passes has *discriminated*: it told the broken state apart from the fixed one, so
    it is not vacuous. That is the domain-agnostic form of "compared against something
    the code does not itself define" — it needs no numerical invariant, so a parser, a
    CLI or a refactor can reach ``oracle`` where previously only numerical code could.
    Deliberately not gated on ``syntax``/``static``: a ``py_compile`` going red→green
    proves the file parses, which is not evidence about behaviour.

    A model cannot award itself this by writing a test after the fix — such a test
    never ran red. Provoking it on purpose means writing a failing check first, which
    is TDD; a signal whose only exploit is the desired behaviour needs no defending.
    """
    execution_context["validation_fail_count_by_file"].pop(target, None)
    execution_context["validated_files"].add(target)
    if tier == "executed" and target in execution_context.get("executed_failures", set()):
        tier = "oracle"
    raise_validation_tier(execution_context, target, tier)
    if execution_context["code_mutation_started"] and not has_pending_validation(execution_context):
        set_workflow_state(execution_context, "conclude")
        execution_context["nudge_counts"]["validation"] = 0
        execution_context["nudge_counts"]["state"] = 0


def _record_executed_failure(execution_context: dict[str, Any], target: str) -> None:
    """Remember that an ``executed``-tier check was seen failing on *target*.

    The red half of red→green (see :func:`_mark_file_validated`). Kept in its own set
    rather than read off ``validation_fail_count_by_file``, which
    :func:`_mark_file_validated` *pops* on success — destroying the record at the exact
    moment it becomes evidence.

    Never cleared within the query, and in particular **not** on re-edit, unlike
    ``validated_files`` and the tier map. Those record what is true of the current
    revision; this records that a check discriminated at some point, and the edit is
    precisely what happens between the red run and the green one. Clearing it on
    re-edit would erase the signal on its way to being earned.
    """
    execution_context.setdefault("executed_failures", set()).add(target)


def _register_validation_failure(
    execution_context: dict[str, Any], target: str, tier: str = "executed",
    *, arms_red_green: bool = True,
) -> None:
    """Record a failed validation of *target* and drive the retry-budget escape.

    The mirror of :func:`_mark_file_validated`: a validation command that ran but
    exited non-zero (``status == "error"``) means the file did NOT pass. Increment its
    per-file failure count, drop it from ``validated_files``, and return the workflow to
    ``edit`` so the model repairs it. When every dirty file is either validated or has
    exhausted ``VALIDATION_RETRY_BUDGET`` attempts, escape to ``conclude`` so a file that
    cannot be made to pass does not wedge the workflow — the model concludes with the
    residual risk stated. This is what re-feeds the (otherwise inert) fail-count that the
    completion gate, the anti-thrashing state guard, and the nudge all read.

    *tier* says how strong the failing check was; an ``executed`` one also arms the
    red→green promotion — unless *arms_red_green* is False, which the model's own
    ``verdict: fail`` passes. A self-declared failure followed by a self-declared pass
    would otherwise forge discrimination, the one signal an exit code cannot fake: a
    model may lower its own credit, never raise it.
    """
    if tier == "executed" and arms_red_green:
        _record_executed_failure(execution_context, target)
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
    """
    if status == "ok":
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
    for key in ("query", "pattern", "filename", "filename_hint", "name"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            execution_context["search_queries_used"].add(value.strip().lower())


def _observe_discover_transition(payload: dict, execution_context: dict[str, Any]) -> None:
    """Leave discover state once any tool returns concrete results."""
    if execution_context["workflow_state"] != "discover":
        return
    search_has_results = bool(
        payload.get("matches")
        or payload.get("files")
        or payload.get("results")
        or payload.get("content")
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
    agent: Any, tool_name: str, arguments: dict, status: Any, execution_context: dict[str, Any],
) -> None:
    if not (has_cap(tool_name, READ, agent.tool_caps) and status == "ok"):
        return
    checked = agent._normalize_workspace_path(arguments.get("path"))
    if not checked:
        return
    execution_context["read_files"].add(checked)
    _register_known_path(execution_context, checked)
    execution_context["checked_paths"].add(checked)

    # A ranged read declares a line-range arg-role (start/end); a whole-file read does
    # not. Drive the line accounting off that declared role, not the tool name.
    range_args = arg_role(tool_name, "line_range", agent.tool_caps)
    is_ranged = bool(range_args)
    if is_ranged:
        start_arg, end_arg = (range_args + ("start_line", "end_line"))[:2]
        start = int(arguments.get(start_arg, 1))
        end = int(arguments.get(end_arg, start))
        lines_read = max(0, end - start + 1)
        prev = execution_context["read_file_line_counts"].get(checked, 0)
        execution_context["read_file_line_counts"][checked] = min(10_000, prev + lines_read)
    else:
        execution_context["read_file_line_counts"][checked] = 10_000


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


def _observe_tool_run(
    agent: Any, tool_name: str, arguments: dict, status: Any,
    execution_context: dict[str, Any],
) -> None:
    """A non-shell execution tool ran: its output owes a verdict too.

    The counterpart of :func:`_observe_bash_validation`, split by *surface* rather than
    by purpose: a shell tool's calls differ in kind call by call (``cat`` reads,
    ``python`` executes), so only the command text can decide, and that function already
    has the parse. A tool like ``proxy_exec`` or ``ft_run`` has no such variation — the
    call *is* the execution — so the declared capability decides. The two are mutually
    exclusive, which is what keeps a bash run from being registered twice and spares a
    third shlex parse per call.

    Attribution comes from the tool's declared path args, through the same
    :func:`_attributed_run_paths` rule bash goes through; with none, the run is recorded
    against the command alone, which the ledger and the nudge both handle.
    """
    if status != "ok" or _carries_shell_command(agent, tool_name) is not None:
        return
    if not has_cap(tool_name, CODE_EXEC, agent.tool_caps):
        return
    targets = [
        p for p in (
            agent._normalize_workspace_path(arguments.get(name))
            for name in path_args(tool_name, agent.tool_caps)
        ) if p
    ]
    dirty = execution_context.get("dirty_written_files", set())
    _register_unjudged_run(
        execution_context, tool_name,
        _attributed_run_paths(execution_context, targets, dirty), "executed",
    )


# Leading commands (or `python -m` modules) that validate code project-wide. A
# *successful* run of one naming no specific source file (bare `pytest`, `ruff check .`,
# `mypy src/`) validates the whole tree → clears every pending file. Per-file validation
# is already language-agnostic; this set only adds the whole-project shortcut.
#
# INVARIANT: every entry must also be in the bash server's `_ALLOWED_COMMANDS` AND
# `bash_classify._EXEC_COMMANDS`, or it is dead — rejected by the server or untaggable
# by the classifier. Declare a new project runner in those two places first.
# Excluded on purpose: `python`/`node`/`./binary` (running a program is not validation)
# and `make`/`cmake` (a green `make clean` validates nothing).
_PROJECT_VALIDATORS = frozenset({
    # Python
    "pytest", "ruff", "mypy", "pyflakes", "black", "py_compile",
    # C / C++ / CUDA / Fortran — CMake test runner (green run validates the whole build)
    "ctest",
})
# A positional that names a specific file (has a dotted extension), vs a directory/`.`.
_FILE_EXT_RE = _re.compile(r"\.[A-Za-z0-9]+$")

# How much a green run of each validator actually proves, keyed by command head;
# unlisted heads default to "executed". Deliberately a *lower* bound: `pytest` is
# "executed" however good the test is, since from outside the process a rigorous suite
# and a vacuous one are the same exit code. Only printed invariants (see
# numerics.observed_invariant_metrics) lift a run to "oracle", in _observe_bash_validation.
_VALIDATOR_TIER = {
    "py_compile": "syntax",
    "ruff": "static", "mypy": "static", "pyflakes": "static", "black": "static",
    # A compiler is a checker, not a run: it emits diagnostics and exit 0 means there
    # were none. Like the linters, its output *is* the verdict — nothing was executed,
    # so there is no result for anyone to judge.
    "gcc": "static", "g++": "static", "gfortran": "static", "nvcc": "static",
    "javac": "static",
    "pytest": "executed", "ctest": "executed",
}
_DEFAULT_VALIDATION_TIER = "executed"

# Reason recorded when the run itself declared the failure, so the model is not asked to
# judge output that already carries its own verdict.
_SELF_DECLARED_FAILURE = "the run's own output declared check=fail"


def _outside_workspace(path: str) -> bool:
    """True for a path ``_normalize_workspace_path`` left outside the user's tree.

    Normalization rebases anything under the workspace to a relative path, so what
    stays absolute — or climbs out with ``..`` — is by construction not workspace code.
    The scratchpad above all, which is where the prompt sends every throwaway probe.
    """
    return os.path.isabs(path) or path == ".." or path.startswith(".." + os.sep)


def _bash_validation_scan(agent: Any, command: str) -> tuple[list[str], bool, str, bool]:
    """Scan a bash command for validation intent.

    Returns ``(explicit_targets, whole_project, tier, ran_code)``:
    - ``explicit_targets`` — normalized workspace paths of the files named by EXEC
      segments, with a leading ``cd`` rebasing later relative operands (``cd sub &&
      pytest t.py`` → ``sub/t.py``).
    - ``whole_project`` — True when a recognised project validator ran over no specific
      file (bare ``pytest``, ``ruff check .``, ``mypy src/``): a green run then validates
      every pending file, not just a named one.
    - ``tier`` — the *strongest* evidence any EXEC segment in the command offers
      (``py_compile a.py && pytest a.py`` is an execution, not a syntax check).
    - ``ran_code`` — True when any segment executed at all, named file or not. The two
      above are about *crediting* a file; this is about a run having produced output
      somebody must judge, which ``python -c …`` and a bare ``./solver`` also do.
    Empty/False/default for an opaque command or one with no exec segment.
    """
    segments = classify_bash_command(command)
    if not segments:
        return [], False, _DEFAULT_VALIDATION_TIER, False
    rel_base = ""
    targets: list[str] = []
    whole_project = False
    tier_rank = -1
    for seg in segments:
        if seg.kind == Kind.CHDIR:
            new_dir = seg.operands[0] if seg.operands else ""
            if new_dir:
                rel_base = new_dir if os.path.isabs(new_dir) else os.path.normpath(
                    os.path.join(rel_base, new_dir) if rel_base else new_dir
                )
            continue
        if seg.kind != Kind.EXEC:
            continue
        for op in seg.operands:
            based = op if (os.path.isabs(op) or not rel_base) else os.path.join(rel_base, op)
            path = agent._normalize_workspace_path(based)
            if path:
                targets.append(path)
        # A bare `/tmp/<scratch>/probe` names its artifact in the head, not the operands.
        # Kept only when it resolves outside the workspace — the sole thing
        # `_attributed_run_paths` reads it for, so no other attribution shifts.
        head_path = agent._normalize_workspace_path(
            seg.head if (os.path.isabs(seg.head) or not rel_base)
            else os.path.join(rel_base, seg.head)
        )
        if os.sep in seg.head and head_path and _outside_workspace(head_path):
            targets.append(head_path)
        seg_tier = _VALIDATOR_TIER.get(seg.head, _DEFAULT_VALIDATION_TIER)
        tier_rank = max(tier_rank, VALIDATION_TIERS.index(seg_tier))
        if seg.head in _PROJECT_VALIDATORS and not any(_FILE_EXT_RE.search(op) for op in seg.operands):
            whole_project = True
    tier = VALIDATION_TIERS[tier_rank] if tier_rank >= 0 else _DEFAULT_VALIDATION_TIER
    return targets, whole_project, tier, tier_rank >= 0


def _register_unjudged_run(
    execution_context: dict[str, Any], command: str, paths: list[str], tier: str,
) -> None:
    """Record a run whose output nobody has judged yet.

    Exit 0 means the program reached its end, never that its answer is right, so an
    execution no longer credits the file on its own: it parks here until the model says
    what the output showed. ``paths`` may be empty — an analysis-only session running
    ``pytest`` with nothing written still owes a judgement, it just has no file to
    attach it to.
    """
    execution_context.setdefault("unjudged_runs", {})[command] = {
        "paths": sorted(set(paths)), "tier": tier,
    }


def _attributed_run_paths(
    execution_context: dict[str, Any], named: list[str], dirty: set,
) -> list[str]:
    """The pending files a run's output speaks for.

    A run naming dirty files speaks for exactly those. Naming none, it falls back to
    the whole pending set, because a run routinely exercises code it does not name:
    `pytest`, `./solver` and `python tests/test_solver.py` all validate the source
    behind them, and that fallback is what lets one suite settle a refactor.

    The exception is a run whose every named file lives *outside* the workspace: it
    did name what it ran, and it ran something that is not the user's code. Letting it
    inherit the pending set would mean one `verdict: pass` on a scratchpad probe
    validating source the probe never touched — and the prompt sends every throwaway
    probe to the scratchpad, so this is the common path, not a corner case. The run is
    still registered with no paths: a verdict is owed on it, it just credits nothing.
    """
    attributed = [t for t in named if t in dirty]
    if attributed:
        return attributed
    if named and all(_outside_workspace(t) for t in named):
        return []
    return pending_validation_paths(execution_context)


def record_output_verdict(
    execution_context: dict[str, Any], path: str, verdict: str, reason: str, command: str,
) -> None:
    """Store the model's verdict on *path*, and log it when it is a failure.

    ``verdict_by_file`` holds the current claim (retracted when the file is re-edited,
    like the tier); ``verdict_attempts_by_file`` is append-only and survives re-edits,
    because "what was tried and why it did not work" is exactly the history a re-edit
    would otherwise erase.
    """
    execution_context.setdefault("verdict_by_file", {})[path] = {
        "verdict": verdict, "reason": reason, "command": command,
    }
    if verdict == "fail":
        attempts = execution_context.setdefault("verdict_attempts_by_file", {}).setdefault(path, [])
        attempts.append(f"{command} → {reason}" if reason else command)


def _observe_bash_validation(
    agent: Any, tool_name: str, arguments: dict, status: Any, payload: dict,
    execution_context: dict[str, Any],
) -> None:
    """Drive validation state from a bash command — success AND failure.

    The bash-driven replacement for the removed validator ladder, run *regardless of
    status* (unlike :func:`_observe_command`, which only credits the blackboard on
    success). Three outcomes rather than two, because exit 0 does not mean the same
    thing for every check:

    - ``syntax``/``static`` (``py_compile``, ``ruff``, ``mypy``) — exit 0 **validates**
      the file. These tools *are* the verdict: their output is a list of problems, and
      an empty one is the finding.
    - ``executed`` (``pytest``, ``python solver.py``, ``./solver``) — exit 0 proves only
      that the program reached its end, so the file stays pending and the run is parked
      in ``unjudged_runs`` until the model states what the output showed.
    - non-zero exit — ``_register_validation_failure``, unchanged.

    A green *project-wide* validator (bare ``pytest``, ``ruff check .``) covers every
    pending file at once; at ``executed`` tier that now means every pending file awaits
    the same verdict. Test files are recorded in ``tests_run`` either way (pass or fail)
    so the regression nudge sees them. Only shell-command tools qualify.
    """
    command_args = _carries_shell_command(agent, tool_name)
    if command_args is None:
        return
    command = str(arguments.get(command_args[0], "") or "")
    explicit, whole_project, tier, ran_code = _bash_validation_scan(agent, command)
    # `ran_code` (not just a named target) is what catches `python -c …` and a bare
    # `./solver`: they credit nothing, but they produced output somebody must judge.
    if not explicit and not whole_project and not ran_code:
        return
    stdout = str(payload.get("stdout") or "")
    # A check that computes its own criteria, reports them unmet and still returns 0 is a
    # green exit over a red result — the shape of the failure this guards: a self-written
    # boundary test printed "significant reflection may be present", exited 0, and was
    # recorded as validated. A declared verdict therefore outranks the exit code, one way
    # only: `check=fail` demotes a pass, `check=pass` never rescues a failure.
    declared_failure = status == "ok" and observed_failure_verdict(stdout)
    if declared_failure:
        status = "error"
    # A green run that *reported* a numerical invariant compared the artifact against
    # something it does not define (analytic solution, reference field, conserved
    # quantity, refinement sweep) — the only signal, short of reading the test's
    # assertions, separating "the code ran" from "the code was checked". Presence of the
    # key is what counts; the value is never interpreted (a forged number is
    # unfalsifiable from out here, which is why the proxy seals references server-side).
    if status == "ok" and observed_invariant_metrics(stdout):
        tier = "oracle"
    ran_executable = tier not in ("syntax", "static")
    dirty = execution_context.get("dirty_written_files", set())
    for target in explicit:
        if is_python_test_filepath(target):
            execution_context["tests_run"].add(target)
        if target not in dirty:
            continue
        if status != "ok":
            # A specific-file failure is attributable; a whole-project failure is not.
            _register_validation_failure(execution_context, target, tier)
            if declared_failure:
                record_output_verdict(
                    execution_context, target, "fail", _SELF_DECLARED_FAILURE, command,
                )
        elif not ran_executable:
            _mark_file_validated(execution_context, target, tier)
    # A green project-wide validation run covers every still-pending file. A failing one
    # is not attributable to a single file, so it leaves pending untouched (the model
    # narrows down or re-runs per file).
    if status == "ok" and whole_project and not ran_executable:
        for target in list(pending_validation_paths(execution_context)):
            _mark_file_validated(execution_context, target, tier)
    # …but a failing project-wide run is still *evidence*, if not attribution: the suite
    # was red while exactly these files were pending. That is what lets the repair loop
    # (run suite, fix, run again) earn the red→green promotion, which per-file
    # attribution never sees — the command names the test, the pending file is the
    # source. Separate from `_register_validation_failure` on purpose: no retry budget
    # charged, nothing un-validated, no workflow transition. Spelled out rather than
    # left to an `elif`, which a green `executed` run now also falls through to.
    elif status != "ok" and whole_project and tier == "executed":
        for target in pending_validation_paths(execution_context):
            _record_executed_failure(execution_context, target)
    if status == "ok" and ran_executable:
        _register_unjudged_run(
            execution_context, command,
            _attributed_run_paths(execution_context, explicit, dirty), tier,
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
            for pattern in operands:
                if isinstance(pattern, str) and pattern.strip():
                    execution_context["search_queries_used"].add(pattern.strip().lower())
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
        elif kind == Kind.EXEC:
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


def record_tool_observation(
    agent: Any,
    tool_name: str,
    arguments: dict,
    result: str,
    execution_context: dict[str, Any] | None,
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
    if _observe_apply_edits(agent, tool_name, arguments, status, execution_context):
        return
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
    _observe_declared_edit_set(agent, tool_name, arguments, status, execution_context)
    _observe_bash_validation(agent, tool_name, arguments, status, payload, execution_context)
    _observe_command(agent, tool_name, arguments, status, payload, execution_context)
    _observe_action_op(agent, tool_name, status, execution_context)
    _observe_tool_run(agent, tool_name, arguments, status, execution_context)
