from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..context.capabilities import infer_tool_caps
from ..config.constants import STATE_DIR


def _make_elicitation_callback(agent: Any):
    """Bridge MCP elicitation requests to the active frontend.

    A server tool (e.g. ``ask_user_question``) calls ``ctx.session.elicit_form``;
    that arrives here as an ``elicitation/create`` request. We parse the rich
    question spec carried in the schema's ``x_mimir`` extension and hand it to
    ``agent._request_user_question`` — a per-frontend handler (CLI ``input``, a
    WebSocket prompt card, or the default no-op) that asks the (possibly several)
    questions sequentially and returns the user's answers.

    The frontend handlers are blocking/queue-based, so we run them off the event
    loop via ``run_in_executor`` to keep the agent loop responsive.
    """

    async def _callback(
        _context: Any,
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData:
        schema = getattr(params, "requestedSchema", None) or {}
        spec = schema.get("x_mimir") if isinstance(schema, dict) else None
        if not isinstance(spec, dict) or spec.get("kind") != "user_question":
            # Not a question we know how to render — decline rather than guess.
            return types.ElicitResult(action="decline")

        questions = spec.get("questions") or []
        if not isinstance(questions, list) or not questions:
            return types.ElicitResult(action="decline")

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                agent._request_user_question,
                questions,
            )
        except Exception:
            return types.ElicitResult(action="cancel")

        answers = list((result or {}).get("answers") or [])
        if not answers:
            return types.ElicitResult(action="decline")

        return types.ElicitResult(action="accept", content={"answers": answers})

    return _callback


async def connect_server(*, agent: Any, name: str, script: str) -> None:
    """Spawn an MCP server process and register its tools on the agent."""
    if not (script.endswith(".py") or script.endswith(".js")):
        raise ValueError("Server script must be a .py or .js file")

    command = sys.executable if script.endswith(".py") else "node"
    # Propagate the client's cwd as the file-server root so paths are predictable,
    # and the central per-workspace state dir so state-owning servers (memory, todo,
    # platform profile) persist to the same place the client resolves. The state dir
    # is deliberately distinct from the file/search sandbox — see config.constants.
    server_env = {
        **os.environ,
        "MCP_FILES_ROOT": os.getcwd(),
        "SEARCH_ROOT": os.getcwd(),
        "MIMIR_STATE_DIR": STATE_DIR,
    }
    params = StdioServerParameters(command=command, args=[script], env=server_env)

    transport = await agent.exit_stack.enter_async_context(stdio_client(params))
    stdio, write = transport
    session = await agent.exit_stack.enter_async_context(
        ClientSession(stdio, write, elicitation_callback=_make_elicitation_callback(agent))
    )
    await session.initialize()

    agent.sessions[name] = session

    listing = await session.list_tools()
    for tool in listing.tools:
        agent.tool_owner[tool.name] = name
        # Derive the tool's semantics (capabilities, arg roles, fallbacks, label)
        # from what the server declares (meta/annotations), falling back to the
        # relocated legacy table.  This registry replaces the old scattered
        # hardcoded tool-name lists; see context/capabilities.py.
        agent.tool_caps[tool.name] = infer_tool_caps(tool)
        # Convert MCP tool -> Ollama function-calling format.
        agent.tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        })

    tool_names = [t.name for t in listing.tools]

    n_resources = await register_resources(agent=agent, name=name, session=session)

    suffix = f", {n_resources} resources" if n_resources else ""
    print(f"✅ [{name}]  {len(tool_names)} tools{suffix}: {tool_names}")


async def register_resources(*, agent: Any, name: str, session: Any) -> int:
    """Discover a server's MCP resources into ``agent.resources``; return the count.

    Resources are user-attached context (Claude/Copilot-style ``@``-mention), not
    model-invokable tools — so they go into a separate registry, never into
    ``agent.tools``. Servers that don't implement the capability raise or return an
    empty list; either is treated as "no resources" (fail-open). URIs are stringified
    (they arrive as pydantic ``AnyUrl``).
    """
    n_resources = 0
    try:
        res_listing = await session.list_resources()
    except Exception:
        return 0
    for resource in res_listing.resources:
        uri = str(resource.uri)
        agent.resources[uri] = {
            "name": resource.name or uri,
            "description": resource.description or "",
            "mimeType": getattr(resource, "mimeType", None),
            "session": name,
        }
        n_resources += 1
    return n_resources
