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
    if target and target in execution_context.dirty_written_files:
        mark_file_validated(execution_context, target)


def action_op(call: Call) -> None:
    """Count each successful substantive action (write / execution / mutation).

    PLAN_BLOCKED is the universal marker for a non-discovery, side-effecting tool:
    file writers derive it, and exec/db/web/cluster/proxy/hpc tools declare it. The
    todo nudge reads this count so a task made of many such operations — e.g. an
    optimisation loop that edits a single file but runs many benchmarks/evals — is
    still recognised as multi-step and prompted to record a checklist.
    """
    execution_context = call.execution_context
    if call.status == "ok" and has_cap(call.tool_name, PLAN_BLOCKED, call.agent.tool_caps):
        execution_context.action_op_count = int(execution_context.action_op_count) + 1


def _run_key(tool_name: str, payload: dict, spec: dict[str, Any] | None) -> str:
    """The ledger key for a run this tool performed.

    A non-shell execution tool has no command line to name its runs. When it declares
    which payload field identifies the run, that identifier is the key, so twenty
    iterations are twenty entries.

    Not prefixed with the tool name: the tool that launches a run and the one that
    later reports its outcome must land on the same entry.
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
    """A non-shell execution tool ran: its output owes a verdict too.

    The counterpart of :func:`bash_validation`, split by *surface* rather than
    by purpose: a shell tool's calls differ in kind call by call (``cat`` reads,
    ``python`` executes), so only the command text can decide, and that function already
    has the parse. A tool like ``proxy_exec`` has no such variation — the
    call *is* the execution — so the declared capability decides. The two are mutually
    exclusive, which is what keeps a bash run from being registered twice and spares a
    third shlex parse per call.

    Recorded against the tool name, plus the run identifier the tool declares in its
    ``run_outcome`` spec so repeated calls do not collapse onto one entry. It validates
    no file: an execution's exit code says the program reached its end, and which file
    it exercised is not a question anyone here can answer.
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

    source = active_proxy_source()
    if source and source in execution_context.dirty_written_files:
        mark_file_validated(execution_context, source, "measured")


def _settle_from_machine(
    execution_context: ExecutionContext, tool_name: str, payload: dict,
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
        record_run_outcome(
            execution_context, key, False, call_id, _MACHINE_SAW_CRASH)
        return
    record_run_outcome(execution_context, key, True, call_id)
    run = (execution_context.runs or {}).get(key)
    if run is not None:
        run["verdict"] = "fail"
        run["reason"] = _MACHINE_SAW_FAILURE
        register_run_failure(execution_context, key, _MACHINE_SAW_FAILURE)


def unwatched_runs(call: Call) -> None:
    """Record long work a scheduler says exists and no descriptor covers.

    Payload-driven and tool-agnostic, deliberately — which is why it is not handed the
    tool name: the fact is established by whoever can establish it (the HPC server runs
    the queue *and* owns the submission directories) and arrives as two machine facts,
    the ids the scheduler printed and the absence of a record for them. Nothing here
    reads a command string. A nudge may rest only on something checkable, and "a
    command whose text contained sbatch" is a guess that fires on a ``grep sbatch`` and
    misses every wrapper ever written.

    The twin half is the clearing: a call that returns a watcher's handle for an id has
    made that id watched, so it leaves the set in the same turn. An array is adopted
    under its base id and listed by task, so the base clears its tasks too.
    """
    execution_context = call.execution_context
    # Shape first, and this is the hot path: every tool call of every turn comes
    # through here, and all but a handful carry none of these keys.
    handles = call.payload.get("background_jobs") or []
    if call.payload.get("background_job"):
        handles = [*handles, call.payload["background_job"]]
    listed = call.payload.get("unwatched") or ()
    if call.status != "ok" or not (listed or handles):
        return
    unwatched = execution_context.unwatched_runs
    unwatched.update(job_id for job_id in listed
                     if isinstance(job_id, str) and job_id)
    # Adopted or freshly submitted, which are the same fact here: something now polls
    # it. Every producer's ``job_key`` *is* the id, so the handles are the whole answer.
    watched = {str(h.get("job_key")) for h in handles if isinstance(h, dict)}
    unwatched.difference_update(
        {key for key in unwatched
         if key in watched or key.split("_", 1)[0] in watched})
