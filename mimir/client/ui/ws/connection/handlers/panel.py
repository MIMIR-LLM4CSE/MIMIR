"""Commands, the toggles, the scientific panel, sub-agents, resources, the model.

A handler group of ``Connection``, built with its view state, its socket, the pool
and the shared ``Screen``. ``MESSAGES`` names the inbound message types it answers.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from .....config import paths
from .....session.commands import CommandResult
from .....session.store import list_subagents as _list_subagents
from .....session.store import read_subagent as _read_subagent
from ..panel_runs import attach_runs, recorded_hosts_sections
from ..commands import WS_COMMANDS, WsCommandHost
from ...pool.pool import AgentPool
from ..screen import Screen
from ..view import Sender, ViewState
from .base import Handlers

logger = logging.getLogger(__name__)


class PanelHandlers(Handlers):
    MESSAGES: dict[str, str] = {
        "command": "on_command_msg",
        "list_toggles": "on_list_toggles",
        "list_resources": "on_list_resources",
        "toggle_server": "on_toggle_server",
        "toggle_skill": "on_toggle_skill",
        "toggle_nudge": "on_toggle_nudge",
        "set_model": "on_set_model",
        "list_subagents": "on_list_subagents",
        "list_panel": "on_list_panel",
        "read_subagent": "on_read_subagent",
        "cancel_run": "on_cancel_run",
    }

    def __init__(self, view: ViewState, out: Sender, pool: AgentPool, screen: Screen) -> None:
        super().__init__(view, out, pool, screen)
        # The panel report in flight, and whether another ask (and a re-read) came in
        # behind it: asks arriving while one is served fold into one more pass.
        self._panel_task: asyncio.Task | None = None
        self._panel_again: dict | None = None

    async def on_command_msg(self, msg: dict) -> None:
        await self.run_command((msg.get("text") or "").strip())

    async def run_command(self, text: str) -> None:
        """Answer a session command the user typed (never seen by the model)."""
        result = await WS_COMMANDS.run(WsCommandHost(self.screen), text)
        if result is None:
            result = CommandResult.error(f"Unknown command: {text}")
        await self._send_command_result(result)

    async def _send_command_result(self, result: CommandResult) -> None:
        """Send a command's result: ``error``, its state message, or a ``command_output`` card."""
        if result.failed:
            await self.out.send({"type": "error", "text": result.title})
        elif result.state is not None:
            await self.out.send(result.state)
        elif result.card:
            await self.screen.command_reply(
                result.label, result.title,
                items=[{"label": i.label, "detail": i.detail} for i in result.items],
                note=result.note, tone=result.tone)

    async def send_toggles(self) -> None:
        """Push the current server/skill enabled state for the toggle panel."""
        state = self.screen.worker.toggles_state()
        await self.out.send({
            "type": "toggles_list",
            "servers": state.get("servers", []),
            "skills": state.get("skills", []),
            "nudges": state.get("nudges", []),
        })

    async def on_list_toggles(self, msg: dict) -> None:
        await self.send_toggles()

    async def on_list_panel(self, msg: dict) -> None:
        """Serve the scientific panel: the servers' sections plus the runs the watchers hold.

        ``build`` starts the agent without waiting (the report follows once it lands);
        ``refresh`` re-reads past the servers' caches. Reports are served in the background,
        concurrent asks folded into one pass.
        """
        if msg.get("build") and self.pool.get(self.view.active_session_id) is None:
            self.out.schedule(self._report_panel_once_built(self.screen.ensure_worker()))
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
        worker = self.screen.worker
        sections = await asyncio.wrap_future(worker.panel.sections(force=refresh))
        agent_ready = worker.agent_ready()
        if not agent_ready:
            sections = recorded_hosts_sections(paths.STATE_DIR)
        # The one place both halves exist: what a cluster is made of came from a
        # server, what is running on it is this client's own watcher state. A run that
        # no group claimed still travels, under `runs`, for the flat list — a run the
        # user cannot see is worse than one shown out of place.
        sections, unplaced = attach_runs(sections, self.screen.worker.watched_runs())
        cards = _list_subagents(self.view.active_session_id or "")
        await self.out.send({
            "type": "panel_report",
            # Whether there was anything to ask. An empty section list from a
            # conversation with no agent is not "no clusters here".
            "agent_ready": agent_ready,
            "subagent_level": self.screen.worker.setting("subagent_level"),
            "subagents": cards,
            "running": sum(1 for c in cards if c.get("state") == "running"),
            "runs": unplaced,
            "sections": sections,
        })

    async def on_cancel_run(self, msg: dict) -> None:
        """Stop a run the panel shows, by the cancel op its own server named.

        Only what the conversation's watcher holds can be stopped, so a run is never
        named by the panel alone. The conversation hears of it from the watcher, which
        polls at once and reports the ending like any other.
        """
        job_key = str(msg.get("job_key") or "")
        worker = self.pool.get(self.view.active_session_id)
        if not job_key or worker is None:
            await self.screen.command_reply("cancel", "No run of this conversation to stop.",
                                            tone="warn")
            return
        payload = await asyncio.wrap_future(worker.cancel_run(job_key))
        # A reply card, not an error: an error ends the turn on screen, and the agent
        # may well be working beside the run.
        if payload.get("status") == "error":
            await self.screen.command_reply("cancel", f"Could not stop {job_key}",
                                            note=str(payload.get("error") or ""), tone="warn")
        self._request_panel()

    async def on_list_subagents(self, msg: dict) -> None:
        """Serve the sub-agents of the session in view.

        Read-only, and read off the files the children left: a panel that could relaunch
        one would be a second way to delegate, out of the orchestrator's sight.
        """
        cards = _list_subagents(self.view.active_session_id or "")
        await self.out.send({
            "type": "subagents_list",
            "session_id": self.view.active_session_id or "",
            "subagents": cards,
            # For the badge on the button: how many are working right now.
            "running": sum(1 for c in cards if c.get("state") == "running"),
        })

    async def on_read_subagent(self, msg: dict) -> None:
        await self.out.send({
            "type": "subagent",
            "subagent": _read_subagent(
                self.view.active_session_id or "", str(msg.get("id") or "")),
        })

    async def on_list_resources(self, msg: dict) -> None:
        """Serve the attachable-resource list for the webview picker/autocomplete."""
        await self.out.send({
            "type": "resources",
            "resources": self.screen.worker.panel.resources(),
        })

    async def on_toggle_server(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self.pool.apply_toggle("server", name, bool(msg.get("enabled", True)))
        await self.send_toggles()

    async def on_toggle_skill(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self.pool.apply_toggle("skill", name, bool(msg.get("enabled", True)))
        await self.send_toggles()

    async def on_toggle_nudge(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self.pool.apply_toggle("nudge", name, bool(msg.get("enabled", True)))
        await self.send_toggles()

    async def send_served_models(self) -> None:
        """Tell the client what the endpoint serves (asked off-thread by the agent process)."""
        try:
            models = await asyncio.get_event_loop().run_in_executor(
                None, self.screen.worker.served_models)
        except Exception:
            logger.debug("_send_served_models: a best-effort step failed and was skipped", exc_info=True)
            return
        if not models:
            return
        try:
            await self.out.send({"type": "served_models", "models": models})
        except Exception:
            logger.debug("_send_served_models: the client went away mid-send", exc_info=True)

    async def on_set_model(self, msg: dict) -> None:
        """Switch the served model; report it with its thinking profile, enforcement and temperature."""
        model = (msg.get("model") or "").strip()
        if not model:
            await self.out.send({
                "type": "error", "text": "  ✗ No model name given.\n",
            })
            return
        error = next(iter(self.pool.set_model(model)), "")
        if error:
            await self.out.send({"type": "error", "text": f"  ✗ {error}\n"})
            return
        await self.out.send({
            "type": "model_changed",
            "model": self.pool.model,
            "thinking": self.screen.worker.get_thinking_profile(),
            "enforcement": self.screen.worker.setting("enforcement"),
            "temperature": self.screen.worker.get_temperature_state(),
        })
