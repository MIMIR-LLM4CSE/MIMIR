"""The approval exemption for read-only calls of the dual-use bash tool, in every mode."""

from __future__ import annotations

from typing import Any

from ...tools.capabilities import (
    PLAN_READONLY,
    has_cap,
    readonly_invocation_spec,
    shell_command_of,
)
from ...tools.bash_classify import bash_command_is_readonly
from ...core.ports import PolicyAgent


def is_readonly_invocation(tool_name: str, arguments: dict[str, Any], caps: Any) -> bool:
    """Whether this call of a dual-use (``PLAN_READONLY``) tool reads and changes nothing.

    By its declared read-only argument value (omitted means its default), or by the
    classifier for a shell command.
    """
    if not has_cap(tool_name, PLAN_READONLY, caps):
        return False
    spec = readonly_invocation_spec(tool_name, caps)
    if spec is not None:
        value = str(arguments.get(spec["arg"], "")).strip().lower()
        return not value or value in {v.lower() for v in spec["values"]}
    return bash_command_is_readonly(shell_command_of(tool_name, arguments, caps))


def readonly_bash_exempt(agent: PolicyAgent, tool_name: str, arguments: dict[str, Any]) -> bool:
    """True when a sensitive dual-use call is read-only and needs no approval.

    Applies in every mode. Before it shared :func:`is_readonly_invocation`, a
    spec-declared read-only call (``op=list``) ran unasked in plan mode and still
    prompted in agent mode.
    """
    return is_readonly_invocation(tool_name, arguments, agent.tool_caps)
