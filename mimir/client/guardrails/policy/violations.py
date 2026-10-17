"""What a refusal tells the model: the violation payload, the evidence it lacks, a path to clarify."""
from __future__ import annotations

import json
import logging
import os
import sys
from ... import human_pause
from ...context import known_existing_files
from ...config.limits import DISCOVERY_EVIDENCE_MIN_DISTINCT
from ...context.capabilities import EDIT, REMOVE, has_cap
from ...context.execution_context import (
    has_discovery_evidence,
    unwritten_declared_files,
)
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort


logger = logging.getLogger(__name__)


def _is_interactive_session(agent: AgentPort) -> bool:
    """Return True if it is reasonable to prompt the human mid-run."""
    if getattr(agent, "batch_mode", False):
        return False
    if getattr(agent, "non_interactive", False):
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        logger.debug("_is_interactive_session: a best-effort step failed and was skipped", exc_info=True)
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


def clarify_path(
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
    registry = agent.tool_caps
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


def missing_evidence(execution_context: ExecutionContext | None) -> list[str]:
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


def violation_kind(violation: str) -> str | None:
    """The ``violation_kind`` a JSON violation declares, or None."""
    try:
        payload = json.loads(violation)
    except ValueError:
        return None
    return payload.get("violation_kind") if isinstance(payload, dict) else None


def enrich_violation_payload(
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
        logger.debug("enrich_violation_payload: a best-effort step failed and was skipped", exc_info=True)
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
        missing = missing_evidence(execution_context)
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
