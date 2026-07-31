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

from ...context.capabilities import PLAN_READONLY, has_cap
from .bash_classify import bash_command_is_readonly


def _readonly_bash_exempt(agent: Any, tool_name: str, arguments: dict[str, Any]) -> bool:
    """True when a sensitive dual-use bash call is read-only and needs no approval.

    Gated on the tool carrying the PLAN_READONLY capability (the dual-use bash tool)
    and the command classifying as read-only. Applies in every mode.
    """
    if not has_cap(tool_name, PLAN_READONLY, agent.tool_caps):
        return False
    return bash_command_is_readonly(arguments.get("command", ""))
