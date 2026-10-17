"""Commands, the toggles, the scientific panel, sub-agents, resources, the model.

A part of ``Connection``: a mixin over the connection's own state, so a
handler reads and writes the session it serves. ``MESSAGES`` names the inbound message
types it answers.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .....config import paths
from .....session.commands import CommandResult
from .....session.store import list_subagents as _list_subagents
from .....session.store import read_subagent as _read_subagent
from ..panel_runs import attach_runs, recorded_hosts_sections
from ..commands import WS_COMMANDS, WsCommandHost

logger = logging.getLogger(__name__)


class PanelHandlers:
    MESSAGES: dict[str, str] = {
        "command": "_handle_command_msg",
        "list_toggles": "_handle_list_toggles",
        "list_resources": "_handle_list_resources",
        "toggle_server": "_handle_toggle_server",
        "toggle_skill": "_handle_toggle_skill",
        "toggle_nudge": "_handle_toggle_nudge",
        "set_model": "_handle_set_model",
        "list_subagents": "_handle_list_subagents",
        "list_panel": "_handle_list_panel",
        "read_subagent": "_handle_read_subagent",
    }

    async def _handle_command_msg(self, msg: dict) -> None:
        await self._handle_command((msg.get("text") or "").strip())

    async def _command_reply(
        self,
        command: str,
        title: str,
        *,
        items: list[dict] | None = None,
        note: str = "",
        tone: str = "ok",
        transient: bool = False,
    ) -> None:
        """Send one structured answer for a session command.

        Only two kinds of command reach here: a LISTING, and something IRREVERSIBLE
        that just happened. Every setting — mode, thinking, streaming, batch,
        context, approvals, enforcement — reports state instead and writes nothing to
        the transcript: each already has a control that shows its value, so a line
        about it repeats the chrome. The webview also replays its stored settings on
        every connect, so those lines were not even reports of anything the user had
        just done; they greeted each session with changes nobody had made.

        Structured, not pre-formatted. A listing of twenty memories and a one-line
        result want different shapes on screen, and a frontend cannot lay out what
        reaches it as an already-indented blob of text: the best any frontend can do with
        a blob is print it.

        ``tone`` says how the result should read: ``ok`` for routine, ``warn`` for
        something irreversible that just happened, ``empty`` for a listing with
        nothing in it, ``quiet`` for a state the user has walked back into rather than
        caused — drawn as a rule across the thread, with no badge and no border, since
        there is nothing to deal with. ``items`` are ``{label, detail}`` rows.

        ``transient`` keeps the card out of the stored transcript. For one that
        describes this moment rather than the conversation: stored, it is replayed on
        every load, so a conversation reconnected to twenty times reopens on twenty
        copies of it, each above whatever answer it happened to follow.
        """
        await self.ws.send(json.dumps({
            "type":    "command_output",
            "command": command,
            "title":   title,
            "items":   items or [],
            "note":    note,
            "tone":    tone,
            "transient": transient,
        }))

    async def _handle_command(self, text: str) -> None:
        """Answer a session command the user typed (never seen by the model)."""
        result = await WS_COMMANDS.run(WsCommandHost(self), text)
        if result is None:
            result = CommandResult.error(f"Unknown command: {text}")
        await self._send_command_result(result)

    async def _send_command_result(self, result: CommandResult) -> None:
        """A command's result, the way this front-end shows each shape.

        An error as ``error``. A setting as its STATE message — the webview draws it on
        the control that owns the setting, and a line in the transcript would only
        repeat that control. A listing or an irreversible change as ``command_output``,
        not ``output``: ``output`` is the transient tool-activity channel the webview
        drops on purpose, which silently swallowed every answer — ``/memory list``
        printed nothing, and ``/memory clear`` wiped the store while looking like it
        had done nothing at all.
        """
        if result.failed:
            await self.ws.send(json.dumps({"type": "error", "text": result.title}))
        elif result.state is not None:
            await self.ws.send(json.dumps(result.state))
        elif result.card:
            await self._command_reply(
                result.label, result.title,
                items=[{"label": i.label, "detail": i.detail} for i in result.items],
                note=result.note, tone=result.tone)

    async def _send_toggles(self) -> None:
        """Push the current server/skill enabled state for the toggle panel."""
        state = self.worker.toggles_state()
        await self.ws.send(json.dumps({
            "type": "toggles_list",
            "servers": state.get("servers", []),
            "skills": state.get("skills", []),
            "nudges": state.get("nudges", []),
        }))

    async def _handle_list_toggles(self, msg: dict) -> None:
        await self._send_toggles()

    async def _handle_list_panel(self, msg: dict) -> None:
        """Serve the scientific-computing panel: the servers' sections, plus ours.

        One section is the client's own because its state lives here — the runs the
        watcher is holding. The rest is whatever the connected servers declared they
        could fill.

        Two flags, and both are the difference between the reader asking and a timer
        asking. ``build`` starts this conversation's agent if it has none: the sections
        are filled by MCP servers, a conversation holds none until its first query, and
        a drawer opened before that query showed nothing — which is why the clusters
        only ever appeared once something had been asked of the model. ``refresh`` goes
        on to the servers as "read it again", past the caches they answer a timer from.

        The build is started, never waited on. It is the LLM backend wait plus ~19
        server spawns, and this handler runs on the socket's own read loop — awaiting it
        here would stop the user from typing for as long as it takes, for a drawer. So
        the report below goes out now, saying no agent answered for it; the worker
        announces itself with a ``ready`` once it is up, and that is the drawer's cue to
        ask again — and this end sends the report itself as soon as the build lands, so
        the drawer fills whether or not that cue makes it back.

        The report is not awaited here either. A cold cluster section is seconds of
        ``squeue``/``sinfo``, and the read loop held for it kept the user's own messages
        waiting behind a drawer. Asks that arrive while one is being served are folded
        into a single further pass, so a timer and a ⟳ never queue up reports.
        """
        if msg.get("build") and self.pool.get(self._active_session_id) is None:
            self._schedule(self._report_panel_once_built(self._ensure_worker()))
        self._request_panel(refresh=bool(msg.get("refresh")))

    def _request_panel(self, *, refresh: bool = False) -> None:
        """Have a panel report sent, without waiting for it."""
        wanted = self._panel_again or {}
        self._panel_again = {"refresh": wanted.get("refresh", False) or refresh}
        if self._panel_task is None or self._panel_task.done():
            self._panel_task = asyncio.get_running_loop().create_task(
                self._serve_panel())

    async def _report_panel_once_built(self, build: Any) -> None:
        if await build is not None:
            self._request_panel()

    async def _serve_panel(self) -> None:
        while self._panel_again is not None:
            msg, self._panel_again = self._panel_again, None
            try:
                await self._send_panel_report(refresh=msg["refresh"])
            except Exception:
                logger.debug("panel report failed", exc_info=True)
                return

    async def _send_panel_report(self, *, refresh: bool) -> None:
        # One worker for the whole report: the agent may land during the await, and a
        # report read off the stand-in must not then claim to be the agent's.
        worker = self.worker
        sections = await asyncio.wrap_future(worker.panel_sections(force=refresh))
        agent_ready = worker.agent_ready()
        if not agent_ready:
            sections = recorded_hosts_sections(paths.STATE_DIR)
        # The one place both halves exist: what a cluster is made of came from a
        # server, what is running on it is this client's own watcher state. A run that
        # no group claimed still travels, under `runs`, for the flat list — a run the
        # user cannot see is worse than one shown out of place.
        sections, unplaced = attach_runs(sections, self.worker.watched_runs())
        cards = _list_subagents(self._active_session_id or "")
        await self.ws.send(json.dumps({
            "type": "panel_report",
            # Whether there was anything to ask. An empty section list from a
            # conversation with no agent is not "no clusters here".
            "agent_ready": agent_ready,
            "subagent_level": self.worker.setting("subagent_level"),
            "subagents": cards,
            "running": sum(1 for c in cards if c.get("state") == "running"),
            "runs": unplaced,
            "sections": sections,
        }))

    async def _handle_list_subagents(self, msg: dict) -> None:
        """Serve the sub-agents of the session in view.

        Read-only, and read off the files the children left: a panel that could relaunch
        one would be a second way to delegate, out of the orchestrator's sight.
        """
        cards = _list_subagents(self._active_session_id or "")
        await self.ws.send(json.dumps({
            "type": "subagents_list",
            "session_id": self._active_session_id or "",
            "subagents": cards,
            # For the badge on the button: how many are working right now.
            "running": sum(1 for c in cards if c.get("state") == "running"),
        }))

    async def _handle_read_subagent(self, msg: dict) -> None:
        await self.ws.send(json.dumps({
            "type": "subagent",
            "subagent": _read_subagent(
                self._active_session_id or "", str(msg.get("id") or "")),
        }))

    async def _handle_list_resources(self, msg: dict) -> None:
        """Serve the attachable-resource list for the webview picker/autocomplete."""
        await self.ws.send(json.dumps({
            "type": "resources",
            "resources": self.worker.resources_snapshot(),
        }))

    async def _handle_toggle_server(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self.pool.apply_toggle("server", name, bool(msg.get("enabled", True)))
        await self._send_toggles()

    async def _handle_toggle_skill(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self.pool.apply_toggle("skill", name, bool(msg.get("enabled", True)))
        await self._send_toggles()

    async def _handle_toggle_nudge(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self.pool.apply_toggle("nudge", name, bool(msg.get("enabled", True)))
        await self._send_toggles()

    async def _send_served_models(self) -> None:
        """Tell the client what the endpoint serves, so it can offer a choice.

        The agent process is the one authority on this: it holds the address and the
        API key, and it reaches the cluster the way the rest of the client does, so the
        list does not depend on the extension host's own probe getting through a proxy.

        Sent after the greeting, not inside it: asking the endpoint is a network
        round trip, and the greeting is what takes the webview out of "connecting".
        Off-thread for the same reason — a slow endpoint must not hold the event loop
        that serves every other message. An endpoint that cannot enumerate itself
        sends nothing; the client keeps naming the active model.
        """
        try:
            models = await asyncio.get_event_loop().run_in_executor(
                None, self.worker.served_models)
        except Exception:
            logger.debug("_send_served_models: a best-effort step failed and was skipped", exc_info=True)
            return
        if not models:
            return
        try:
            await self.ws.send(json.dumps({"type": "served_models", "models": models}))
        except Exception:
            logger.debug("_send_served_models: the client went away mid-send", exc_info=True)

    async def _handle_set_model(self, msg: dict) -> None:
        """Switch the served model mid-session.

        On success, report the new model plus the model-derived settings
        (thinking profile, enforcement) so the webview's controls stay coherent —
        the same state-not-narration pattern as ``/mode``. On failure, surface an
        error rather than silently keeping the old model.
        """
        model = (msg.get("model") or "").strip()
        if not model:
            await self.ws.send(json.dumps({
                "type": "error", "text": "  ✗ No model name given.\n",
            }))
            return
        error = next(iter(self.pool.set_model(model)), "")
        if error:
            await self.ws.send(json.dumps({"type": "error", "text": f"  ✗ {error}\n"}))
            return
        await self.ws.send(json.dumps({
            "type": "model_changed",
            "model": self.pool.model,
            "thinking": self.worker.get_thinking_profile(),
            "enforcement": self.worker.setting("enforcement"),
            "temperature": self.worker.get_temperature_state(),
        }))
