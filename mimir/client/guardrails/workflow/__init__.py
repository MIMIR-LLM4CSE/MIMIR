"""The workflow every guardrail reads: its state, refused approvals, commitments, and
the completion report written when the loop ends."""

from .commitments import turn_made_commitments, unchecked_checklist_items, unhonoured_commitments
from .completion import (
    COMPLETION_FRAMING,
    COMPLETION_MARKER,
    HEADLINE_HANDBACK,
    HEADLINE_INCOMPLETE,
    HEADLINE_REFUSED_ONLY,
    TERMINATION_ANSWERED,
    TERMINATION_STEP_LIMIT,
    blocked_run_lines,
    collect_completion_issues,
    finalize_incomplete_answer,
    is_incomplete_answer,
    parse_completion_block,
    refused_action_lines,
    render_completion_report,
    split_answer_completion,
    unjudged_run_lines,
    unmeasured_source_lines,
)
from .denials import (
    STAGE_DROP_OR_STOP,
    STAGE_HANDBACK,
    STAGE_RECONSIDER,
    approval_is_settled,
    denial_scope_count,
    denial_stage,
    handback_required,
    handback_scopes,
    worst_denial_stage,
)
from .state import (
    WORKFLOW_STATES,
    has_blocking_denials,
    has_pending_validation,
    pending_validation_paths,
    set_workflow_state,
)

__all__ = [
    "COMPLETION_FRAMING", "COMPLETION_MARKER", "HEADLINE_HANDBACK", "HEADLINE_INCOMPLETE",
    "HEADLINE_REFUSED_ONLY", "STAGE_DROP_OR_STOP", "STAGE_HANDBACK", "STAGE_RECONSIDER",
    "TERMINATION_ANSWERED", "TERMINATION_STEP_LIMIT", "WORKFLOW_STATES",
    "approval_is_settled", "blocked_run_lines", "collect_completion_issues",
    "denial_scope_count", "denial_stage", "finalize_incomplete_answer", "handback_required",
    "handback_scopes", "has_blocking_denials", "has_pending_validation",
    "is_incomplete_answer", "parse_completion_block", "pending_validation_paths",
    "refused_action_lines", "render_completion_report", "set_workflow_state",
    "split_answer_completion", "turn_made_commitments", "unchecked_checklist_items",
    "unhonoured_commitments", "unjudged_run_lines", "unmeasured_source_lines",
    "worst_denial_stage",
]
