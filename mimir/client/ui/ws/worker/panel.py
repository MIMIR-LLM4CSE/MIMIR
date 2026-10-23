"""What the side panel shows of an agent: the resources it can attach, and the sections
its servers fill."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import Any, Callable

from .....shared.code_map import FULL_ARGS
from ....tools.capabilities import CODE_MAP, CODE_MAP_STEPS, code_map_tools, panel_sections
from ....protocol.tool_result import normalize_tool_content
from ..runtime import (
    augment_query_with_resources,
)

logger = logging.getLogger(__name__)


class Panel:
    """What the side panel reads of one worker's agent, on that agent's loop.

    *agent* and *loop* answer None until the worker has built them.
    """

    def __init__(self, agent: Callable[[], Any],
                 loop: Callable[[], asyncio.AbstractEventLoop | None]) -> None:
        self._agent = agent
        self._loop = loop

    def resources(self) -> list[dict]:
        """List attachable MCP resources for the frontend picker.

        The registry (``agent.resources``) is populated once at connect and never
        mutated afterward, so this read-only snapshot is safe to take directly from
        the WS event loop without hopping to the worker loop.
        """
        agent = self._agent()
        if agent is None:
            return []
        out: list[dict] = []
        for uri, info in sorted((agent.resources or {}).items()):
            out.append({
                "uri": uri,
                "name": info.get("name", uri),
                "description": info.get("description", ""),
                "mimeType": info.get("mimeType"),
            })
        return out

    def resolve_resources(self, text: str) -> Any:
        """Resolve @-mentions by reading resources on the worker's OWN loop.

        The MCP ``ClientSession`` objects are bound to this worker's event loop, so the
        read must run there — we schedule it via ``run_coroutine_threadsafe`` and return
        a ``concurrent.futures.Future`` the caller awaits with ``asyncio.wrap_future``.
        """
        agent, loop = self._agent(), self._loop()
        if agent is None or loop is None:
            fut: Any = concurrent.futures.Future()
            fut.set_result((text, []))
            return fut
        return asyncio.run_coroutine_threadsafe(
            augment_query_with_resources(agent, text), loop)

    def sections(self, *, force: bool = False) -> Any:
        """Every section the connected servers fill (``PANEL_REPORT``), as a future of a list.

        *force* passes the server's declared refresh argument, for a user's ⟳. Called straight
        on the owning sessions, outside the guardrails.
        """
        agent, loop = self._agent(), self._loop()
        if agent is None or loop is None:
            fut: Any = concurrent.futures.Future()
            fut.set_result([])
            return fut
        declared = panel_sections(agent.tool_caps)

        async def _ask(tool: str, spec: dict) -> dict | None:
            owner = agent.tool_owner.get(tool)
            if owner is None:
                return None
            args = dict(spec.get("args") or {})
            if force and spec.get("refresh_arg"):
                args[str(spec["refresh_arg"])] = True
            try:
                raw = await agent.sessions[owner].call_tool(tool, args)
                text = normalize_tool_content(raw)
                return json.loads(text) if isinstance(text, str) else (text or {})
            except Exception as exc:
                # One server that cannot answer costs its own section, never the
                # panel: the others have already said something worth showing.
                logger.debug("panel section %r failed: %s", tool, exc)
                return None

        async def _run() -> list[dict]:
            # All at once: the sections are independent, they are owned by different
            # processes, and the slow one is slow because it is waiting on a scheduler.
            # Asked in turn, the drawer cost the sum of every server's wait; asked
            # together it costs the longest. The order below is still the declared one.
            payloads = await asyncio.gather(
                *(_ask(tool, spec) for tool, spec in declared))
            out: list[dict] = []
            for (tool, spec), payload in zip(declared, payloads):
                if payload is None:
                    continue
                lines = payload.get("lines") or []
                detail = payload.get("detail") or ""
                # A nested half, for a server whose answer is a tree rather than a
                # list: a scheduler, the kinds of node on it, what one of them is.
                # Passed through unread, exactly as the lines are — this end renders
                # what a server chose to say and never decides what a field means.
                groups = payload.get("groups") or []
                if not lines and not detail and not groups:
                    # A server with nothing to say gets no heading. A section that has
                    # something to report about having nothing — "no optimisation
                    # session here" — says it in `detail` and still appears; one that
                    # returns neither is a facility this machine does not have, and a
                    # bare title is a heading the user reads to learn there is nothing
                    # to read.
                    continue
                out.append({
                    "section": spec.get("section", tool),
                    "title": payload.get("title") or spec.get("section", tool),
                    "lines": lines,
                    "groups": groups,
                    "detail": detail,
                    # A view the section offers to open (the code map), by name.
                    "open": payload.get("open") or "",
                })
            return out

        return asyncio.run_coroutine_threadsafe(_run(), loop)

    def code_map(self) -> Any:
        """The workspace's code map from the first server that answers ``CODE_MAP``, and the
        session's steps from the first that answers ``CODE_MAP_STEPS``, as a future of
        ``{"map": …, "session": …}``; ``{}`` while there is no agent or no map.
        """
        agent, loop = self._agent(), self._loop()
        if agent is None or loop is None:
            fut: Any = concurrent.futures.Future()
            fut.set_result({})
            return fut

        async def _first(cap: str, args: dict, key: str) -> dict | None:
            for tool in code_map_tools(agent.tool_caps, cap):
                owner = agent.tool_owner.get(tool)
                if owner is None:
                    continue
                try:
                    raw = await agent.sessions[owner].call_tool(tool, args)
                    text = normalize_tool_content(raw)
                    payload = json.loads(text) if isinstance(text, str) else (text or {})
                except Exception as exc:
                    logger.debug("code map %r failed: %s", tool, exc)
                    continue
                if isinstance(payload, dict) and payload.get(key) is not None:
                    return payload
            return None

        async def _run() -> dict:
            # Asked together: the map is a walk of the tree, the steps a read of snapshots.
            found, steps = await asyncio.gather(
                _first(CODE_MAP, dict(FULL_ARGS), "files"), _first(CODE_MAP_STEPS, {}, "steps"))
            if found is None:
                return {}
            return {"map": found, **({"session": steps} if steps else {})}

        return asyncio.run_coroutine_threadsafe(_run(), loop)
