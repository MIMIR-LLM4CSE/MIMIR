"""Approval exemption for read-only invocations of the dual-use bash tool.

The dual-use bash tool is ``sensitive`` so build/execution commands prompt for
approval. Side-effect-free discovery (``rg``/``ls``/``cat``/``sed`` without
``-i`` …) is safe to run unattended, so re-prompting for it is pure friction — in
**any** mode. ``_readonly_bash_exempt`` waives the prompt for exactly those calls,
delegating the read-only decision to :func:`bash_classify.bash_command_is_readonly`.
An exec command or an in-place write (``sed -i``) still hits the normal gate.

Formerly ``plan_mode.py``: what began as a plan-mode-only gate now applies to every
mode, so neither the module nor its helper carries the ``plan`` name any longer.
"""

from __future__ import annotations

from typing import Any

from ...context.capabilities import (
    PLAN_READONLY, has_cap, readonly_invocation_spec, shell_command_of,
)
from .bash_classify import bash_command_is_readonly


def is_readonly_invocation(tool_name: str, arguments: dict[str, Any], caps: Any) -> bool:
    """Whether this call of a dual-use (PLAN_READONLY) tool reads and changes nothing.

    The one answer to that question: the read-only modes run a call only when it is
    true, and every mode waives the approval prompt when it is. Two shapes of dual-use
    tool. One declares which argument value makes the call read-only — an omitted
    argument is the tool's own default, which such a tool makes its read-only one; the
    other carries a shell command, which only the classifier can judge.
    """
    if not has_cap(tool_name, PLAN_READONLY, caps):
        return False
    spec = readonly_invocation_spec(tool_name, caps)
    if spec is not None:
        value = str(arguments.get(spec["arg"], "")).strip().lower()
        return not value or value in {v.lower() for v in spec["values"]}
    return bash_command_is_readonly(shell_command_of(tool_name, arguments, caps))


def _readonly_bash_exempt(agent: Any, tool_name: str, arguments: dict[str, Any]) -> bool:
    """True when a sensitive dual-use call is read-only and needs no approval.

    Applies in every mode. Before it shared :func:`is_readonly_invocation`, a
    spec-declared read-only call (``op=list``) ran unasked in plan mode and still
    prompted in agent mode.
    """
    return is_readonly_invocation(tool_name, arguments, agent.tool_caps)
