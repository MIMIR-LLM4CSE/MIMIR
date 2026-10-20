"""What a tool call tells the guardrails: the writer of the execution_context blackboard.

``record_tool_observation`` runs after every tool call and hands it to each observer
of ``ORDER`` in turn. The policy gates and the nudges read what they write.
"""
from __future__ import annotations

from ...context import ensure_execution_context
from ...tools.capabilities import target_path_arg
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort
from ...protocol.tool_result import parse_tool_payload
from . import discovery, env, evidence, files, plan, runs, shell
from .evidence import record_denial
from .call import Call, Observer

# Every observer runs on every call, in this order. Where order matters:
# - files.edit_outcome marks a file dirty before anything credits its validation;
# - discovery.discover_transition leaves "discover" only after the edit observers ran;
# - shell.bash_validation credits checks before shell.command records the command's
#   edits and reads;
# - runs.tool_run opens the run that runs.run_outcome then settles.
ORDER: tuple[Observer, ...] = (
    files.edit_outcome,
    plan.todo_flags,
    files.replacement_tracking,
    runs.validation_tool,
    env.missing_module,
    env.env_probe,
    env.env_mutation,
    runs.unwatched_runs,
    evidence.denial_clearing,
    files.edit_loop_clear,
    files.delete,
    discovery.search_flags,
    discovery.discover_transition,
    discovery.candidates,
    discovery.dir_inspect,
    discovery.existence_check,
    files.read,
    files.delegated_exploration,
    plan.declared_edit_set,
    shell.bash_validation,
    shell.command,
    runs.action_op,
    runs.tool_run,
    runs.run_outcome,
    evidence.verdict_tool,
)


def record_tool_observation(
    agent: AgentPort,
    tool_name: str,
    arguments: dict,
    result: str,
    execution_context: ExecutionContext | None,
    call_id: str = "",
) -> None:
    execution_context = ensure_execution_context(execution_context)
    if execution_context is None:
        return
    payload = parse_tool_payload(result) or {}
    call = Call(
        agent=agent,
        tool_name=tool_name,
        execution_context=execution_context,
        arguments=arguments,
        payload=payload,
        status=payload.get("status"),
        path=agent.normalize_workspace_path(
            target_path_arg(tool_name, arguments, agent.tool_caps, directories=True)),
        call_id=call_id,
    )
    for observe in ORDER:
        observe(call)


__all__ = ["Call", "ORDER", "Observer", "record_denial", "record_tool_observation"]
