"""A tool call the user made themselves (``/memory clear``, ``/proxy clean``), around the
guardrails: straight to the owning MCP session, with no approval, gate or tool card.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..protocol.tool_result import normalize_tool_content
from ...servers._shared.responses import err, ok

logger = logging.getLogger(__name__)

NO_AGENT = "No agent session is running."


async def call_tool_directly(agent: Any, tool: str, arguments: dict) -> dict:
    """Call *tool* on *agent*'s session that owns it; the decoded payload.

    Never raises: a missing agent, a server that is not connected, a failing call and an
    answer that is not JSON all come back as an error payload. Must run on the loop the
    agent's MCP sessions are bound to.
    """
    if agent is None:
        return err(NO_AGENT)
    owner = (getattr(agent, "tool_owner", None) or {}).get(tool)
    if owner is None:
        return err(f"The server owning '{tool}' is not connected.")
    try:
        raw = await agent.sessions[owner].call_tool(tool, dict(arguments))
    except Exception as exc:
        return err(str(exc) or type(exc).__name__)
    text = normalize_tool_content(raw)
    if not isinstance(text, str):
        return text or {}
    try:
        payload = json.loads(text)
    except ValueError:
        return err(f"'{tool}' did not answer in JSON.")
    return payload if isinstance(payload, dict) else ok(result=payload)
