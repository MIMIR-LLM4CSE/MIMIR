from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from ... import human_pause
from .write import check_write_policy
from .plugins import PolicyRegistry
from ...context import ensure_execution_context, is_known_to_exist, known_existing_files
from .state_machine import check_state_machine_guard
from .gates import (
    _check_cluster_submit,
    _check_external_fetch,
    _check_out_of_workspace_access,
    _check_proxy_exec,
    _out_of_workspace_targets,
)
# The exempt helper waives the approval prompt for side-effect-free discovery
# commands in any mode (the read-only classifier itself lives in bash_classify.py).
from .readonly_exempt import _readonly_bash_exempt
from ..workflow import STAGE_HANDBACK, denial_stage
from ...context.capabilities import EDIT, REMOVE, has_cap
from ...context.execution_context import (
    has_discovery_evidence,
    unwritten_declared_files,
    bootstrap_engine_context as _bootstrap_engine_context,
)


@dataclass
class PolicyEvaluation:
    tool_name: str
    arguments: dict[str, Any]
    execution_context: dict[str, Any] | None
    violation: str | None


def _is_interactive_session(agent: Any) -> bool:
    """Return True if it is reasonable to prompt the human mid-run."""
    if getattr(agent, "batch_mode", False):
        return False
    if getattr(agent, "non_interactive", False):
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _collect_candidate_paths(path: str, execution_context: dict[str, Any] | None) -> list[str]:
    """Collect likely candidate paths related to `path` from execution context."""
    if execution_context is None:
        return []

    target_base = os.path.basename(path).lower()
    known: set[str] = known_existing_files(execution_context)

    similar_by_dir = execution_context.get("similar_candidates_by_dir", {})
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
    agent: Any,
    tool_name: str,
    path: str,
    execution_context: dict[str, Any] | None,
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


def _missing_evidence(execution_context: dict[str, Any] | None) -> list[str]:
    """Return concrete missing evidence items to guide the agent toward policy compliance."""
    if execution_context is None:
        return []

    execution_context = _bootstrap_engine_context(execution_context)

    evidence: list[str] = []
    state = execution_context.get("workflow_state", "discover")

    searched = execution_context.get("searched", False)
    read_files = execution_context.get("read_files", set()) or set()
    dirty_files = execution_context.get("dirty_written_files", set()) or set()
    validated_files = execution_context.get("validated_files", set()) or set()

    if state == "discover":
        # Coherent with the discovery gates: only guide toward discovery while the
        # model has no real (model-initiated) evidence yet. The signal set lives once
        # in context.execution_context (has_discovery_evidence).
        if not has_discovery_evidence(execution_context):
            if not searched:
                evidence.append("Run a targeted local search first.")
            if not read_files:
                evidence.append("Read at least one concrete file before proposing code changes.")

    elif state == "edit":
        if not read_files:
            evidence.append("Read the target file explicitly with read_file or read_file_lines before editing.")
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
                "Validate modified files through the shell before concluding "
                "(python -m py_compile / pytest -q / ruff check / python -m mypy): "
                + ", ".join(pending[:5])
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


def _next_tool_class(policy_stage: str, execution_context: dict[str, Any] | None) -> str:
    if policy_stage == "state_guard" and execution_context is not None:
        state = execution_context.get("workflow_state", "discover")
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


def _enrich_violation_payload(
    *,
    violation: str,
    policy_stage: str,
    execution_context: dict[str, Any] | None,
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
    # Attach missing_evidence only when it is actiona
    if policy_stage in {"write_policy", "approval"}:
        missing = _missing_evidence(execution_context)
        if missing:
            payload["missing_evidence"] = missing
    
    # Never nag during discovery phase
    state = execution_context.get("workflow_state", "discover")
    if state == "discover":
        payload.pop("missing_evidence", None)

    if execution_context is not None:
        payload.setdefault("state", execution_context.get("workflow_state", "discover"))

    if policy_stage == "state_guard":
        payload.setdefault("status", "blocked")
    else:
        payload.setdefault("status", "error")

    
    if execution_context is not None:
        nudges = execution_context.get("nudge_counts", {}).get("denial", 0)
        if nudges >= 2:
            payload.pop("missing_evidence", None)

    return json.dumps(payload, indent=2, ensure_ascii=False)


def _run_extra_checks(
    extra_checks: list | None,
    stage: str,
    agent: Any,
    tool_name: str,
    arguments: dict[str, Any],
    execution_context: dict[str, Any] | None,
) -> PolicyEvaluation | None:
    """Run application-registered PolicyChecks for *stage*; return the first violation.

    Custom checks are additive constraints layered on the built-in core — they run at
    a fixed slot ("pre_mutation" or "pre_approval") and can only BLOCK (return a
    violation string); returning None never relaxes a core gate. A misbehaving check
    is caught so a bad pack cannot break the tool pipeline. Returns a ready
    ``PolicyEvaluation`` (with the violation enriched) so the caller just returns it.
    """
    if not extra_checks:
        return None
    for pc in extra_checks:
        if pc.stage != stage:
            continue
        try:
            violation = pc.check(agent, tool_name, arguments, execution_context)
        except Exception:
            # A pack check that raises is treated as "no opinion", not a hard failure.
            continue
        if violation:
            return PolicyEvaluation(
                tool_name=tool_name,
                arguments=arguments,
                execution_context=execution_context,
                violation=_enrich_violation_payload(
                    violation=violation,
                    policy_stage=pc.name,
                    execution_context=execution_context,
                    tool_name=tool_name,
                ),
            )
    return None


def evaluate_tool_preconditions(
    *,
    agent: Any,
    tool_name: str,
    arguments: dict[str, Any],
    execution_context: dict[str, Any] | None = None,
) -> PolicyEvaluation:
    """Run registry/state/write/approval checks and return normalized execution inputs.

    The single call-time entry point. Built-in gates run in a fixed order and call
    the module-level guards directly (``ensure_execution_context`` /
    ``check_state_machine_guard`` / ``check_write_policy``); application-registered
    :class:`PolicyCheck`s from :data:`PolicyRegistry` run at their declared stage
    between the built-in gates and can only add constraints.
    """
    extra_checks = PolicyRegistry.active_checks()

    normalized_context = ensure_execution_context(execution_context)
    normalized_context = _bootstrap_engine_context(normalized_context)

    # Normalize + rewrite BEFORE the registry check. Legacy aliases such as the
    # model's habitual `read_file` are intentionally unregistered and exist only
    # to be healed here (→ read_file_lines). Checking the registry on the raw
    # name first would reject the alias before the rewrite that resolves it can
    # run — the very false negative this ordering prevents. Both helpers no-op
    # safely on an unregistered name, so genuinely unknown tools fall through to
    # the single check below and still report an "Unknown tool" violation.
    normalized_arguments = agent._normalize_tool_arguments(tool_name, arguments)
    normalized_tool_name, rewritten_arguments = agent._rewrite_tool_for_context(
        tool_name,
        normalized_arguments,
    )

    if normalized_tool_name not in agent.tool_owner:
        violation = agent._json_error_payload(
            f"Unknown tool '{normalized_tool_name}'.",
            hint="Refresh the server registry or choose one of the registered tool names.",
            tool=normalized_tool_name,
        )
        return PolicyEvaluation(
            tool_name=normalized_tool_name,
            arguments=rewritten_arguments,
            execution_context=normalized_context,
            violation=_enrich_violation_payload(
                violation=violation,
                policy_stage="registry",
                execution_context=normalized_context,
                tool_name=normalized_tool_name,
            ),
        )

    pre_mutation_eval = _run_extra_checks(
        extra_checks, "pre_mutation", agent, normalized_tool_name, rewritten_arguments, normalized_context,
    )
    if pre_mutation_eval is not None:
        return pre_mutation_eval

    external_violation = _check_external_fetch(agent, normalized_tool_name, normalized_context)
    if external_violation is not None:
        return PolicyEvaluation(
            tool_name=normalized_tool_name,
            arguments=rewritten_arguments,
            execution_context=normalized_context,
            violation=_enrich_violation_payload(
                violation=external_violation,
                policy_stage="external_fetch",
                execution_context=normalized_context,
                tool_name=normalized_tool_name,
            ),
        )

    cluster_violation = _check_cluster_submit(agent, normalized_tool_name, normalized_context)
    if cluster_violation is not None:
        return PolicyEvaluation(
            tool_name=normalized_tool_name,
            arguments=rewritten_arguments,
            execution_context=normalized_context,
            violation=_enrich_violation_payload(
                violation=cluster_violation,
                policy_stage="cluster_submit",
                execution_context=normalized_context,
                tool_name=normalized_tool_name,
            ),
        )

    proxy_exec_violation = _check_proxy_exec(
        agent, normalized_tool_name, rewritten_arguments, normalized_context)
    if proxy_exec_violation is not None:
        return PolicyEvaluation(
            tool_name=normalized_tool_name,
            arguments=rewritten_arguments,
            execution_context=normalized_context,
            violation=_enrich_violation_payload(
                violation=proxy_exec_violation,
                policy_stage="proxy_exec",
                execution_context=normalized_context,
                tool_name=normalized_tool_name,
            ),
        )

    state_violation = check_state_machine_guard(
        agent,
        normalized_tool_name,
        rewritten_arguments,
        normalized_context,
    )
    if state_violation is not None:
        return PolicyEvaluation(
            tool_name=normalized_tool_name,
            arguments=rewritten_arguments,
            execution_context=normalized_context,
            violation=_enrich_violation_payload(
                violation=state_violation,
                policy_stage="state_guard",
                execution_context=normalized_context,
                tool_name=normalized_tool_name,
            ),
        )

    write_policy_violation = check_write_policy(
        agent,
        normalized_tool_name,
        rewritten_arguments,
        normalized_context,
    )

    if write_policy_violation is not None:
        # For path-related violations, offer interactive clarification before blocking —
        # but only when the path itself is unconfirmed (path confusion). Skip the prompt
        # when the path is already known from a prior read/check: in that case the
        # violation is about policy semantics (e.g. missing overwrite=True), not the path.
        write_path = agent._normalize_workspace_path(rewritten_arguments.get("path", ""))
        path_already_confirmed = bool(write_path) and is_known_to_exist(
            normalized_context, write_path
        )
        if "path" in write_policy_violation.lower() and not path_already_confirmed:
            clarified_path = _interactive_clarify_path(
                agent=agent,
                tool_name=normalized_tool_name,
                path=rewritten_arguments.get("path", ""),
                execution_context=normalized_context,
            )

            if clarified_path:
                rewritten_arguments = {**rewritten_arguments, "path": clarified_path}

                # Re-run state guard because the clarified path may change validity.
                state_violation = check_state_machine_guard(
                    agent,
                    normalized_tool_name,
                    rewritten_arguments,
                    normalized_context,
                )
                if state_violation is not None:
                    return PolicyEvaluation(
                        tool_name=normalized_tool_name,
                        arguments=rewritten_arguments,
                        execution_context=normalized_context,
                        violation=_enrich_violation_payload(
                            violation=state_violation,
                            policy_stage="state_guard",
                            execution_context=normalized_context,
                            tool_name=normalized_tool_name,
                        ),
                    )

                # Re-run write policy on the corrected path.
                write_policy_violation = check_write_policy(
                    agent,
                    normalized_tool_name,
                    rewritten_arguments,
                    normalized_context,
                )

        if write_policy_violation is not None:
            return PolicyEvaluation(
                tool_name=normalized_tool_name,
                arguments=rewritten_arguments,
                execution_context=normalized_context,
                violation=_enrich_violation_payload(
                    violation=write_policy_violation,
                    policy_stage="write_policy",
                    execution_context=normalized_context,
                    tool_name=normalized_tool_name,
                ),
            )

    pre_approval_eval = _run_extra_checks(
        extra_checks, "pre_approval", agent, normalized_tool_name, rewritten_arguments, normalized_context,
    )
    if pre_approval_eval is not None:
        return pre_approval_eval

    # Computed here, not just inside the gate: the out-of-workspace prompt already
    # shows this call's label and arguments, so when it runs the sensitive-tool card
    # below would be the same decision asked twice. The targets are what tells the two
    # apart, and once the gate has recorded its grants they no longer exist.
    oow_targets = _out_of_workspace_targets(agent, normalized_tool_name, rewritten_arguments)
    oow_violation = _check_out_of_workspace_access(
        agent, normalized_tool_name, rewritten_arguments, normalized_context, oow_targets)
    if oow_violation is not None:
        agent._record_denied_tool_call(
            normalized_tool_name, rewritten_arguments, normalized_context,
            "path outside the workspace was not approved")
        return PolicyEvaluation(
            tool_name=normalized_tool_name,
            arguments=rewritten_arguments,
            execution_context=normalized_context,
            violation=_enrich_violation_payload(
                violation=oow_violation,
                policy_stage="approval",
                execution_context=normalized_context,
                tool_name=normalized_tool_name,
            ),
        )

    if (agent.approvals.is_sensitive(normalized_tool_name, rewritten_arguments)
            and not oow_targets
            and not _readonly_bash_exempt(agent, normalized_tool_name, rewritten_arguments)):
        # Already refused to the end of the ladder: refuse it ourselves rather than
        # putting the same card in front of the user a fourth time. Being asked again
        # after saying no is the friction the ladder exists to remove.
        scope = agent.approval_scope(normalized_tool_name, rewritten_arguments)
        if denial_stage(normalized_context or {}, scope) == STAGE_HANDBACK:
            note = "already refused; not asked again"
            approved = False
        else:
            approved, note = agent._request_tool_approval(normalized_tool_name, rewritten_arguments)
        if not approved:
            agent._record_denied_tool_call(
                normalized_tool_name, rewritten_arguments, normalized_context, note)
            return PolicyEvaluation(
                tool_name=normalized_tool_name,
                arguments=rewritten_arguments,
                execution_context=normalized_context,
                violation=_enrich_violation_payload(
                    violation=agent._denied_tool_result(
                        normalized_tool_name, rewritten_arguments, note, normalized_context),
                    policy_stage="approval",
                    execution_context=normalized_context,
                    tool_name=normalized_tool_name,
                ),
            )

    return PolicyEvaluation(
        tool_name=normalized_tool_name,
        arguments=rewritten_arguments,
        execution_context=normalized_context,
        violation=None,
    )