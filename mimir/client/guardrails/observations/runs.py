"""Runs of the project: their ledger, their outcome, and the runs left unwatched."""
from __future__ import annotations

from __future__ import annotations
import os
from typing import Any
from ..ladder import (
    mark_file_validated,
    record_run_outcome,
    register_run_failure,
)
from ..proxy_state import active_proxy_source
from ....servers._shared.jobs import handed_over, launched
from ...context.capabilities import (
    CODE_EXEC,
    PLAN_BLOCKED,
    VALIDATE,
    has_cap,
    path_args,
    run_outcome_spec,
    shell_command_args,
)
from ...context.execution_context import ExecutionContext
from .call import Call


# A server that ran the code itself reported the run red. Its own judgement, not a
# reading of what the program printed — the floor a stated verdict cannot rise above.
_MACHINE_SAW_CRASH = "the run server reported this run did not complete"


_MACHINE_SAW_FAILURE = "the run server judged this run's result failing"


def validation_tool(call: Call) -> None:
    """A dedicated VALIDATE tool marks its target file validated on success.

    The first-party stack validates through bash (``shell.bash_validation``); this is
    for an extension server that ships its own validator. Any tool carrying VALIDATE
    qualifies, its target read from its declared ``path_args`` (else filepath/path).
    """
    execution_context = call.execution_context
    caps = getattr(call.agent, "tool_caps", None)
    if call.status != "ok" or not has_cap(call.tool_name, VALIDATE, caps):
        return
    arg_names = path_args(call.tool_name, caps) or ("filepath", "path")
    raw = next((call.arguments.get(a) for a in arg_names if call.arguments.get(a)), None)
    target = call.agent.normalize_workspace_path(raw)
    if target and target in execution_context.edits.dirty_written_files:
        mark_file_validated(execution_context, target)


def action_op(call: Call) -> None:
    """Count each successful ``PLAN_BLOCKED`` action, so many-operation tasks read as multi-step."""
    execution_context = call.execution_context
    if call.status == "ok" and has_cap(call.tool_name, PLAN_BLOCKED, call.agent.tool_caps):
        execution_context.action_op_count = int(execution_context.action_op_count) + 1


def _run_key(tool_name: str, payload: dict, spec: dict[str, Any] | None) -> str:
    """The ledger key of a run a tool performed: the payload field it declares, else its name.

    Not prefixed with the tool name, so the tool that launches and the one that reports land
    on one entry.
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


def tool_run(call: Call) -> None:
    """A non-shell execution tool ran: record it as a run owing a verdict.

    The call is the execution, so the capability decides (a shell call needs its command
    parsed: :func:`bash_validation`). It validates no file.
    """
    execution_context = call.execution_context
    if call.status != "ok" or shell_command_args(call.tool_name, call.agent.tool_caps) is not None:
        return
    if not has_cap(call.tool_name, CODE_EXEC, call.agent.tool_caps):
        return
    spec = run_outcome_spec(call.tool_name, getattr(call.agent, "tool_caps", None))
    record_run_outcome(
        execution_context, _run_key(call.tool_name, call.payload or {}, spec), True, call.call_id)


def run_outcome(call: Call) -> None:
    """Record what a server saw of a run it performed itself.

    One way only: it can mark a run crashed or failing, never passing. The call's own
    status is not read.
    """
    execution_context = call.execution_context
    spec = run_outcome_spec(call.tool_name, getattr(call.agent, "tool_caps", None))
    if not spec or not isinstance(call.payload, dict):
        return
    _settle_from_machine(execution_context, call.tool_name, call.payload, spec, call.call_id)
    _credit_measured_source(execution_context, call.payload, spec)
    rows_spec = spec.get("rows")
    if not isinstance(rows_spec, dict):
        return
    rows = call.payload.get(rows_spec.get("field", ""))
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            _settle_from_machine(execution_context, call.tool_name, row, rows_spec, call.call_id)


def _credit_measured_source(
    execution_context: ExecutionContext, payload: dict, spec: dict[str, Any],
) -> None:
    """Raise the optimisation session's source to the ``measured`` tier.

    The one exception to "a run validates no file": the session names exactly one source.
    It records that the file was measured, not that the measurement was good.
    """
    if _matches(payload, spec, "failed") or _matches(payload, spec, "crashed"):
        return
    if not _matches(payload, spec, "measured"):
        return

    source = active_proxy_source()
    if source and source in execution_context.edits.dirty_written_files:
        mark_file_validated(execution_context, source, "measured")


def _settle_from_machine(
    execution_context: ExecutionContext, tool_name: str, payload: dict,
    spec: dict[str, Any], call_id: str,
) -> None:
    """Apply one declared run outcome: a crash drives the ladder, a failing result is a ``fail``.

    A run that measured correctly without improving is left alone.
    """
    crashed = _matches(payload, spec, "crashed")
    failed = _matches(payload, spec, "failed")
    if not crashed and not failed:
        return
    key = _run_key(tool_name, payload, spec)
    if crashed:
        record_run_outcome(
            execution_context, key, False, call_id, _MACHINE_SAW_CRASH)
        return
    record_run_outcome(execution_context, key, True, call_id)
    run = (execution_context.execution.runs or {}).get(key)
    if run is not None:
        run["verdict"] = "fail"
        run["reason"] = _MACHINE_SAW_FAILURE
        register_run_failure(execution_context, key, _MACHINE_SAW_FAILURE)


def unwatched_runs(call: Call) -> None:
    """Record scheduler jobs that no descriptor covers; clear the ones a watcher now holds.

    Both facts come from the payload (ids the scheduler printed, no record for them),
    never from a command string. An array's base id clears its tasks.
    """
    execution_context = call.execution_context
    # Shape first, and this is the hot path: every tool call of every turn comes
    # through here, and all but a handful carry none of these keys.
    handles = handed_over(call.payload)
    started = launched(call.payload)
    if started:
        handles = [*handles, started]
    listed = call.payload.get("unwatched") or ()
    if call.status != "ok" or not (listed or handles):
        return
    unwatched = execution_context.execution.unwatched_runs
    unwatched.update(job_id for job_id in listed
                     if isinstance(job_id, str) and job_id)
    # Adopted or freshly submitted, which are the same fact here: something now polls
    # it. Every producer's ``job_key`` *is* the id, so the handles are the whole answer.
    watched = {str(h.get("job_key")) for h in handles if isinstance(h, dict)}
    unwatched.difference_update(
        {key for key in unwatched
         if key in watched or key.split("_", 1)[0] in watched})
