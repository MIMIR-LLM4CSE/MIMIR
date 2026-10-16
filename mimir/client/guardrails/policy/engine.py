from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from ... import human_pause
from .write import UNCONFIRMED_PATH, check_write_policy
from .plugins import PolicyRegistry
from ...context import ensure_execution_context, is_known_to_exist, known_existing_files
from .state_machine import check_state_machine_guard
from .gates import (
    mode_denied_payload,
    _check_cluster_submit,
    _check_out_of_workspace_access,
    _check_proxy_exec,
    _check_skill_pull,
    _out_of_workspace_targets,
)
# The exempt helper waives the approval prompt for side-effect-free discovery
# commands in any mode (the read-only classifier itself lives in bash_classify.py).
from .readonly_exempt import _readonly_bash_exempt
from ..workflow import approval_is_settled
from .approval import MODE_DENIED_NOTE
from ...config.limits import DISCOVERY_EVIDENCE_MIN_DISTINCT
from ...context.capabilities import EDIT, REMOVE, has_cap, target_path_arg
from ...context.execution_context import (
    has_discovery_evidence,
    unwritten_declared_files,
)
from ...context.execution_context import ExecutionContext
from ...protocol.tool_result import json_error_payload
from ...core.ports import AgentPort

logger = logging.getLogger(__name__)


@dataclass
class PolicyEvaluation:
    tool_name: str
    arguments: dict[str, Any]
    execution_context: ExecutionContext | None
    violation: str | None


def _is_interactive_session(agent: AgentPort) -> bool:
    """Return True if it is reasonable to prompt the human mid-run."""
    if getattr(agent, "batch_mode", False):
        return False
    if getattr(agent, "non_interactive", False):
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _collect_candidate_paths(path: str, execution_context: ExecutionContext | None) -> list[str]:
    """Collect likely candidate paths related to `path` from execution context."""
    if execution_context is None:
        return []

    target_base = os.path.basename(path).lower()
    known: set[str] = known_existing_files(execution_context)

    similar_by_dir = execution_context.similar_candidates_by_dir
    if isinstance(similar_by_dir, dict):
        for paths in similar_by_dir.values():
            if isinstance(paths, set):
                known.update(str(x) for x in paths if x)

    candidates = sorted({
        p for p in known
        if os.path.basename(p).lower() == target_base or target_base in p.lower()
    })
    return candidates


def _interactive_clarify_path(
    *,
    agent: AgentPort,
    tool_name: str,
    path: str,
    execution_context: ExecutionContext | None,
) -> str | None:
    """Ask the user to confirm or correct an ambiguous target path.

    Called when a write/delete policy block is caused by an unrecognised path.
    Returns the confirmed path if the user replies, or None to let the violation
    propagate unchanged.
    """
    if not _is_interactive_session(agent):
        return None

    # Limit prompts to path-sensitive file mutation tools. Driven off the live
    # capability registry (EDIT or REMOVE) rather than a hardcoded tool list, so
    # adding/merging/renaming file tools needs no change here. Batch-edit tools
    # (no single `path` arg) never reach this branch — see the caller's path gate.
    registry = getattr(agent, "tool_caps", None)
    if not (has_cap(tool_name, EDIT, registry) or has_cap(tool_name, REMOVE, registry)):
        return None

    candidates = _collect_candidate_paths(path, execution_context)

    prompt_parts = [f"\n⚠️  Policy check: {tool_name!r} on unconfirmed path '{path}'."]
    if candidates:
        prompt_parts.append("   Similar known paths:")
        for i, c in enumerate(candidates[:5], 1):
            prompt_parts.append(f"     {i}. {c}")
        prompt_parts.append(
            "   Enter a number to select, paste the correct path, or press Enter to cancel: "
        )
    else:
        prompt_parts.append("   Enter the correct path or press Enter to cancel: ")

    try:
        # Waiting on the user, not on the tool (see human_pause): this prompt is
        # raised from inside the tool call it guards.
        with human_pause.human_pause():
            reply = input("\n".join(prompt_parts)).strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not reply:
        return None

    if reply.isdigit() and candidates:
        idx = int(reply) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]

    return reply


def _missing_evidence(execution_context: ExecutionContext | None) -> list[str]:
    """Return concrete missing evidence items to guide the agent toward policy compliance."""
    if execution_context is None:
        return []


    evidence: list[str] = []
    state = execution_context.workflow_state

    searched = execution_context.searched
    read_files = execution_context.read_files or set()
    dirty_files = execution_context.dirty_written_files or set()
    validated_files = execution_context.validated_files or set()

    if state == "discover":
        # Coherent with the discovery gates: only guide toward discovery while the
        # model has no real (model-initiated) evidence yet. The signal set lives once
        # in context.execution_context (has_discovery_evidence), at the same bar as
        # the discovery nudge — two consumers of one definition must not disagree on
        # how much evidence clears it.
        if not has_discovery_evidence(
            execution_context, min_distinct=DISCOVERY_EVIDENCE_MIN_DISTINCT
        ):
            if not searched:
                evidence.append("Run a targeted local search first.")
            if not read_files:
                evidence.append("Read at least one concrete file before proposing code changes.")

    elif state == "edit":
        if not read_files:
            evidence.append("Read the target file explicitly before editing it.")
        remaining_declared = unwritten_declared_files(execution_context)
        if remaining_declared:
            evidence.append(
                "Finish the remaining declared edit targets before switching to validation: "
                + ", ".join(remaining_declared[:5])
            )

    elif state == "validate":
        pending = sorted(dirty_files - validated_files)
        if pending:
            evidence.append(
                "These modified files are still waiting on the check that runs before "
                "this run may conclude: " + ", ".join(pending[:5])
            )

    if dirty_files and not validated_files:
        evidence.append("At least one modified file still lacks successful validation.")

    # Deduplicate while preserving order.
    deduped: list[str] = []
    seen = set()
    for item in evidence:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped


def _next_tool_class(policy_stage: str, execution_context: ExecutionContext | None) -> str:
    if policy_stage == "state_guard" and execution_context is not None:
        state = execution_context.workflow_state
        if state == "discover":
            return "discovery"
        if state == "edit":
            return "targeted_read_or_fix"
        if state == "validate":
            return "validation"
        return "conclusion"

    if policy_stage == "read_policy":
        return "targeted_read"
    if policy_stage == "write_policy":
        return "write_preconditions"
    if policy_stage == "external_fetch":
        return "local_discovery"
    if policy_stage == "cluster_submit":
        return "local_validation"
    if policy_stage == "proxy_exec":
        return "proxy_eval_run"
    if policy_stage == "approval":
        # Not "…_or_approval": asking again is the one move a refusal rules out.
        return "safe_alternative_or_skip_or_hand_back"
    if policy_stage == "registry":
        return "available_tool_selection"
    return "diagnostic"


def _violation_kind(violation: str) -> str | None:
    """The ``violation_kind`` a JSON violation declares, or None."""
    try:
        payload = json.loads(violation)
    except ValueError:
        return None
    return payload.get("violation_kind") if isinstance(payload, dict) else None


def _enrich_violation_payload(
    *,
    violation: str,
    policy_stage: str,
    execution_context: ExecutionContext | None,
    tool_name: str,
) -> str:
    """Attach metadata to JSON policy violations; keep non-JSON violations untouched."""
    try:
        payload = json.loads(violation)
    except Exception:
        return violation

    if not isinstance(payload, dict):
        return violation

    payload.setdefault("policy_stage", policy_stage)
    payload.setdefault("tool", tool_name)
    payload.setdefault("suggested_next_tool_class", _next_tool_class(policy_stage, execution_context))
    state = execution_context.workflow_state if execution_context is not None else "discover"
    # Attach missing_evidence only when it is actionable — never during discovery,
    # where everything is still missing and saying so is nagging.
    if policy_stage in {"write_policy", "approval"} and state != "discover":
        missing = _missing_evidence(execution_context)
        if missing:
            payload["missing_evidence"] = missing

    if execution_context is not None:
        payload.setdefault("state", state)

    if policy_stage == "state_guard":
        payload.setdefault("status", "blocked")
    else:
        payload.setdefault("status", "error")

    
    if execution_context is not None:
        nudges = execution_context.nudge_counts.get("denial", 0)
        if nudges >= 2:
            payload.pop("missing_evidence", None)

    return json.dumps(payload, indent=2, ensure_ascii=False)


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
            violation=_enrich_violation_payload(
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
    """Run application-registered PolicyChecks for *stage*; return the first violation.

    Custom checks are additive constraints layered on the built-in core — they run at
    a fixed slot ("pre_mutation" or "pre_approval") and can only BLOCK (return a
    violation string); returning None never relaxes a core gate. A check that raises
    is "no opinion", so a bad pack cannot break the tool pipeline — but it is logged:
    swallowed silently, a pack that never ran looked exactly like one that approved.
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
    violation = _check_cluster_submit(call.agent, call.tool_name, call.execution_context,
                                      call.arguments)
    return call.refuse(violation, "cluster_submit") if violation is not None else None


def _proxy_gate(call: _PolicyCall) -> PolicyEvaluation | None:
    violation = _check_proxy_exec(call.agent, call.tool_name, call.arguments,
                                  call.execution_context)
    return call.refuse(violation, "proxy_exec") if violation is not None else None


def _skill_gate(call: _PolicyCall) -> PolicyEvaluation | None:
    violation = _check_skill_pull(call.agent, call.tool_name, call.arguments,
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
    path_already_confirmed = bool(write_path) and is_known_to_exist(
        call.execution_context, write_path)
    if _violation_kind(violation) == UNCONFIRMED_PATH and not path_already_confirmed:
        clarified_path = _interactive_clarify_path(
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
    """A target outside the workspace is asked about before anything else is.

    Its targets are kept on the call: the out-of-workspace prompt already shows this
    call's label and arguments, so the approval card after it would be the same
    decision asked twice. Once the gate has recorded its grants they no longer exist,
    which is why they are computed here rather than inside the approval gate.
    """
    call.oow_targets = _out_of_workspace_targets(call.agent, call.tool_name, call.arguments)
    violation = _check_out_of_workspace_access(
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
            or _readonly_bash_exempt(agent, call.tool_name, call.arguments)):
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
    _proxy_gate,
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
    """Run registry/state/write/approval checks and return normalized execution inputs.

    The single call-time entry point: the call goes through :data:`_PIPELINE` and the
    first gate that blocks it decides. Application-registered :class:`PolicyCheck`s
    from :data:`PolicyRegistry` run at their declared stage between the built-in gates
    and can only add constraints.
    """
    extra_checks = PolicyRegistry.active_checks()
    normalized_context = ensure_execution_context(execution_context)

    # Normalize + rewrite BEFORE the registry check. Legacy aliases such as the
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
