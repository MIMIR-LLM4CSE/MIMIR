"""What the side panel shows of an agent: the resources it can attach, and the sections
its servers fill."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import Any, Callable

from ....context.capabilities import panel_sections
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
        """Collect every section the connected servers can fill. Future of a list.

        Asked by capability, never by name: a server declares ``PANEL_REPORT`` and the
        section it fills, and a plugin server's section appears with no client edit.
        Each answer is whatever that server chose to show — the panel renders lines, it
        does not interpret them.

        *force* is the reader asking for a re-read rather than a timer asking for the
        current picture. A section that costs an ``sinfo`` or a reach across ``ssh`` is
        served from the owning server's cache, which is what makes a panel on a timer
        affordable and what made pressing ⟳ hand back the page just read. A server that
        caches names the argument meaning "read it again" in its ``panel`` descriptor;
        this sets it, and only for that press.

        Straight to the owning session, like the slash commands above: this is the
        user opening a drawer, not the model taking a step, and routing it through the
        guardrail pipeline would scatter tool cards through their transcript.
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
                })
            return out

        return asyncio.run_coroutine_threadsafe(_run(), loop)
