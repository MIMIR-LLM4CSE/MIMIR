"""The WebSocket's side of the session commands: its host, its own commands, its replies.

The grammar is shared (``session.commands``). What is the socket's: a setting reaches
the conversation's agent through the pool (recorded for an agent built later, applied
to the live ones), a tool call is scheduled on the worker's own loop, and two commands
exist only here — ``/cancel``, since the terminal's run blocks its prompt, and
``/diag``, which reads the event chain the terminal does not have.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ....session.commands import (
    COMMANDS,
    Command,
    CommandHost,
    CommandRegistry,
    CommandResult,
    Item,
)

if TYPE_CHECKING:
    from .screen import Screen


class WsCommandHost(CommandHost):
    """The conversation on screen, as one connection sees it."""

    surface = "ws"

    def __init__(self, screen: "Screen") -> None:
        self.screen = screen

    def apply(self, name: str, value: Any) -> str:
        error = self.screen.apply_setting(name, value)
        conv = self.screen.view.conv
        if not error and name == "context_mode" and conv is not None:
            # Also what the budget assumes until an agent answers for itself, and what
            # the session is saved with: a mode chosen before the first query must not
            # be forgotten by the one piece of code that sizes the window.
            conv.record.context_mode = value
        return error

    def read(self, name: str) -> Any:
        return self.screen.pool.setting(name, self.screen.view.active_session_id)

    async def call_tool(self, tool: str, arguments: dict) -> dict:
        # The MCP sessions are bound to the worker's loop, so the call is scheduled
        # there rather than awaited on the socket's.
        return await asyncio.wrap_future(self.screen.worker.call_session_tool(tool, arguments))

    def temperature_state(self) -> dict:
        # Before the agent exists this is the stored preference — exactly what the
        # agent will load — where the setting's own default would say "none".
        return self.screen.worker.get_temperature_state()


class CancelCommand(Command[WsCommandHost]):
    name = "cancel"
    surfaces = frozenset({"ws"})
    help = (("", "stop the run in flight"),)

    async def run(self, host: WsCommandHost, args: list[str]) -> CommandResult:
        if host.screen.worker.cancel():
            # The run ending is what the user sees; a card would only say it twice.
            return CommandResult(card=False)
        return CommandResult("Nothing to cancel", tone="empty")


class DiagCommand(Command[WsCommandHost]):
    """``/diag``: the event chain's state, telling a dead pump from a swallowing watermark from silence."""
    name = "diag"
    surfaces = frozenset({"ws"})
    help = (("", "what the event chain has done — pump, journal, watermark"),)

    async def run(self, host: WsCommandHost, args: list[str]) -> CommandResult:
        screen, view = host.screen, host.screen.view
        rows = [Item(r["label"], r.get("detail", "")) for r in screen.pool.bus.diagnostics()]
        rows.append(Item("active session", str(view.active_session_id)))
        rows.append(Item("rendered watermark", str(view.rendered_seq)))
        for sid, count in sorted(view.foreign_withheld.items()):
            rows.append(Item("withheld as foreign",
                             f"{count} event(s) stamped {sid} — not this conversation"))
        for sid, worker in screen.pool.items():
            rows.append(Item(
                f"worker {sid[:8]}",
                f"queue {worker.out_q.qsize()}, "
                f"running {getattr(worker, '_query_session_id', None)}, "
                f"parked {worker.human.awaiting_user()}, "
                f"deferral {worker.human.has_deferral}"))
        return CommandResult("Event chain", rows)


WS_COMMANDS: CommandRegistry = COMMANDS.for_surface("ws", CancelCommand(), DiagCommand())
