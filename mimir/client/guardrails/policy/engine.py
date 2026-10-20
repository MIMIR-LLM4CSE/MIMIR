"""The call-time policy pipeline: the first gate that blocks a call decides."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from .write import UNCONFIRMED_PATH, check_write_policy
from ...context import ensure_execution_context, is_known_to_exist
from .state_machine import check_state_machine_guard
from .gates import (
    mode_denied_payload,
    check_cluster_submit,
    check_out_of_workspace_access,
    check_sealed_exec,
    check_skill_pull,
    out_of_workspace_targets,
)
# Waives the approval prompt for side-effect-free discovery commands, in any mode.
from .readonly_exempt import readonly_bash_exempt
from ..workflow import approval_is_settled
from .approval import MODE_DENIED_NOTE
from ...tools.capabilities import target_path_arg
from ...protocol.tool_result import json_error_payload
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort
from .violations import clarify_path, enrich_violation_payload, violation_kind


logger = logging.getLogger(__name__)


@dataclass
class PolicyEvaluation:
    tool_name: str
    arguments: dict[str, Any]
    execution_context: ExecutionContext | None
    violation: str | None


@dataclass
class _PolicyCall:
    """The call under evaluation, as every gate of the pipeline sees and may refine it."""

    agent: AgentPort
    tool_name: str
    arguments: dict[str, Any]
    execution_context: ExecutionContext | None
    # Out-of-workspace targets, set by the workspace gate for the approval gate.
    oow_targets: list = field(default_factory=list)

    def refuse(self, violation: str, stage: str) -> PolicyEvaluation:
        """The evaluation that blocks this call, its violation enriched for *stage*."""
        return PolicyEvaluation(
            tool_name=self.tool_name,
            arguments=self.arguments,
            execution_context=self.execution_context,
            violation=enrich_violation_payload(
                violation=violation,
                policy_stage=stage,
                execution_context=self.execution_context,
                tool_name=self.tool_name,
            ),
        )

    def allow(self) -> PolicyEvaluation:
        return PolicyEvaluation(tool_name=self.tool_name, arguments=self.arguments,
                                execution_context=self.execution_context, violation=None)

    def record_denial(self, note: str) -> None:
        self.agent.record_denied_tool_call(
            self.tool_name, self.arguments, self.execution_context, note)


def _run_extra_checks(extra_checks: list | None, stage: str,
                      call: _PolicyCall) -> PolicyEvaluation | None:
    """Run the extension set's ``PolicyCheck``s for *stage*; return the first violation.

    They can only block. A check that raises counts as no opinion, and is logged.
    """
    for pc in extra_checks or ():
        if pc.stage != stage:
            continue
        try:
            violation = pc.check(call.agent, call.tool_name, call.arguments,
                                 call.execution_context)
        except Exception:
            logger.warning("policy check %r raised; treated as no opinion", pc.name,
                           exc_info=True)
            continue
        if violation:
            return call.refuse(violation, pc.name)
    return None


# ── The built-in gates, in pipeline order ──────────────────────────────────────
# Each takes the call and returns the evaluation that blocks it, or None to let it on.
# The guards they consult are read off this module at call time.

def _registry_gate(call: _PolicyCall) -> PolicyEvaluation | None:
    if call.tool_name in call.agent.tool_owner:
        return None
    return call.refuse(json_error_payload(
        f"Unknown tool '{call.tool_name}'.",
        hint="Refresh the server registry or choose one of the registered tool names.",
        tool=call.tool_name,
    ), "registry")


def _cluster_gate(call: _PolicyCall) -> PolicyEvaluation | None:
    violation = check_cluster_submit(call.agent, call.tool_name, call.execution_context,
                                      call.arguments)
    return call.refuse(violation, "cluster_submit") if violation is not None else None


def _sealed_gate(call: _PolicyCall) -> PolicyEvaluation | None:
    violation = check_sealed_exec(call.agent, call.tool_name, call.arguments,
                                   call.execution_context)
    return call.refuse(violation, "sealed_exec") if violation is not None else None


def _skill_gate(call: _PolicyCall) -> PolicyEvaluation | None:
    violation = check_skill_pull(call.agent, call.tool_name, call.arguments,
                                  call.execution_context)
    return call.refuse(violation, "skill_pull") if violation is not None else None


def _state_gate(call: _PolicyCall) -> PolicyEvaluation | None:
    violation = check_state_machine_guard(call.agent, call.tool_name, call.arguments,
                                          call.execution_context)
    return call.refuse(violation, "state_guard") if violation is not None else None


def _write_gate(call: _PolicyCall) -> PolicyEvaluation | None:
    violation = check_write_policy(call.agent, call.tool_name, call.arguments,
                                   call.execution_context)
    if violation is None:
        return None
    # A target nobody confirmed may be a mistyped path rather than a policy matter:
    # offer to clarify it before blocking — only then, and only when the path is not
    # already known from a prior read or check.
    write_path = call.agent.normalize_workspace_path(
        target_path_arg(call.tool_name, call.arguments, call.agent.tool_caps))
    path_already_confirmed = (bool(write_path) and call.execution_context is not None
                              and is_known_to_exist(call.execution_context, write_path))
    if violation_kind(violation) == UNCONFIRMED_PATH and not path_already_confirmed:
        clarified_path = clarify_path(
            agent=call.agent, tool_name=call.tool_name,
            path=call.arguments.get("path", ""), execution_context=call.execution_context,
        )
        if clarified_path:
            call.arguments = {**call.arguments, "path": clarified_path}
            # The clarified path may change what the state guard decides.
            blocked = _state_gate(call)
            if blocked is not None:
                return blocked
            violation = check_write_policy(call.agent, call.tool_name, call.arguments,
                                           call.execution_context)
    return call.refuse(violation, "write_policy") if violation is not None else None


def _workspace_gate(call: _PolicyCall) -> PolicyEvaluation | None:
    """Ask about targets outside the workspace first, and keep them on the call.

    The approval card after it would ask the same decision twice; the targets are gone once
    the grants are recorded.
    """
    call.oow_targets = out_of_workspace_targets(call.agent, call.tool_name, call.arguments)
    violation = check_out_of_workspace_access(
        call.agent, call.tool_name, call.arguments, call.execution_context, call.oow_targets)
    if violation is None:
        return None
    call.record_denial(MODE_DENIED_NOTE if call.agent.approvals.unattended
                       else "path outside the workspace was not approved")
    return call.refuse(violation, "approval")


def _approval_gate(call: _PolicyCall) -> PolicyEvaluation | None:
    agent, approvals = call.agent, call.agent.approvals
    if (not approvals.is_sensitive(call.tool_name, call.arguments)
            or call.oow_targets
            or readonly_bash_exempt(agent, call.tool_name, call.arguments)):
        return None
    if approvals.unattended and not approvals.auto_tools():
        # No one can answer a card here (a sub-agent): refuse without asking. First,
        # so a repeat reads as the mode's refusal too — no user was ever asked.
        call.record_denial(MODE_DENIED_NOTE)
        return call.refuse(
            mode_denied_payload(agent, call.tool_name, call.arguments, "auto"), "approval")
    scope = agent.approval_scope(call.tool_name, call.arguments)
    if call.execution_context is not None and approval_is_settled(call.execution_context, scope):
        # Already refused twice on this scope: refuse it ourselves rather than putting
        # the same card in front of the user again. From `drop_or_stop` on, the ladder
        # has already ruled out another attempt at the same goal.
        approved, note = False, "already refused; not asked again"
    elif approvals.auto_tools():
        # Auto mode: the user has taken the decision for the session. Deliberately
        # *after* the settled check — a refusal already given stands.
        approved, note = True, "auto-approved (auto mode)"
    else:
        approved, note = agent.request_tool_approval(call.tool_name, call.arguments)
    if approved:
        return None
    call.record_denial(note)
    return call.refuse(agent.denied_tool_result(
        call.tool_name, call.arguments, note, call.execution_context), "approval")


# The order is the policy. Stages named by a string are where the registered
# PolicyChecks of that stage run.
_PIPELINE: tuple = (
    _registry_gate,
    "pre_mutation",
    _cluster_gate,
    _sealed_gate,
    _skill_gate,
    _state_gate,
    _write_gate,
    "pre_approval",
    _workspace_gate,
    _approval_gate,
)


def evaluate_tool_preconditions(
    *,
    agent: AgentPort,
    tool_name: str,
    arguments: dict[str, Any],
    execution_context: ExecutionContext | None = None,
) -> PolicyEvaluation:
    """Run the call through :data:`_PIPELINE`; the first gate that blocks decides.

    Registered ``PolicyCheck``s run at their stage and can only add constraints.
    """
    extra_checks = agent.extensions.policy_checks()
    normalized_context = ensure_execution_context(execution_context)

    # Normalize + rewrite BEFORE the registry check. Aliases such as the
    # model's habitual `read_file` are intentionally unregistered and exist only
    # to be healed here (→ read_file_lines). Checking the registry on the raw
    # name first would reject the alias before the rewrite that resolves it can
    # run. Both helpers no-op safely on an unregistered name, so genuinely unknown
    # tools still reach the registry gate and report an "Unknown tool" violation.
    normalized_arguments = agent.normalize_tool_arguments(tool_name, arguments)
    normalized_tool_name, rewritten_arguments = agent.rewrite_tool_for_context(
        tool_name, normalized_arguments)
    call = _PolicyCall(agent, normalized_tool_name, rewritten_arguments, normalized_context)
    for step in _PIPELINE:
        blocked = (_run_extra_checks(extra_checks, step, call) if isinstance(step, str)
                   else step(call))
        if blocked is not None:
            return blocked
    return call.allow()
