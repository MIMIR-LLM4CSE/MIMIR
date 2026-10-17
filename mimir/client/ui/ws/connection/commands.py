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
    from .connection import Connection


class WsCommandHost(CommandHost):
    """The conversation on screen, as one connection sees it."""

    surface = "ws"

    def __init__(self, session: "Connection") -> None:
        self.session = session

    def apply(self, name: str, value: Any) -> str:
        error = self.session._apply_setting(name, value)
        if not error and name == "context_mode" and self.session.conv is not None:
            # Also what the budget assumes until an agent answers for itself, and what
            # the session is saved with: a mode chosen before the first query must not
            # be forgotten by the one piece of code that sizes the window.
            self.session.conv.record.context_mode = value
        return error

    def read(self, name: str) -> Any:
        return self.session.pool.setting(name, self.session._active_session_id)

    async def call_tool(self, tool: str, arguments: dict) -> dict:
        # The MCP sessions are bound to the worker's loop, so the call is scheduled
        # there rather than awaited on the socket's.
        return await asyncio.wrap_future(self.session.worker.call_session_tool(tool, arguments))

    def temperature_state(self) -> dict:
        # Before the agent exists this is the stored preference — exactly what the
        # agent will load — where the setting's own default would say "none".
        return self.session.worker.get_temperature_state()


class CancelCommand(Command[WsCommandHost]):
    name = "cancel"
    surfaces = frozenset({"ws"})
    help = (("", "stop the run in flight"),)

    async def run(self, host: WsCommandHost, args: list[str]) -> CommandResult:
        if host.session.worker.cancel():
            # The run ending is what the user sees; a card would only say it twice.
            return CommandResult(card=False)
        return CommandResult("Nothing to cancel", tone="empty")


class DiagCommand(Command[WsCommandHost]):
    """The chain the events travel, asked rather than inferred.

    The pump is the only consumer of every worker's queue, so a chat that goes quiet has
    several causes that look identical from a chat window: the pump never started, it
    died, a watermark is swallowing what it delivers, or the agent emitted nothing.
    These rows tell them apart.
    """
    name = "diag"
    surfaces = frozenset({"ws"})
    help = (("", "what the event chain has done — pump, journal, watermark"),)

    async def run(self, host: WsCommandHost, args: list[str]) -> CommandResult:
        session = host.session
        rows = [Item(r["label"], r.get("detail", "")) for r in session.pool.bus.diagnostics()]
        rows.append(Item("active session", str(session._active_session_id)))
        rows.append(Item("rendered watermark", str(session._rendered_seq)))
        for sid, count in sorted(session._foreign_withheld.items()):
            rows.append(Item("withheld as foreign",
                             f"{count} event(s) stamped {sid} — not this conversation"))
        for sid, worker in session.pool.items():
            rows.append(Item(
                f"worker {sid[:8]}",
                f"queue {worker.out_q.qsize()}, "
                f"running {getattr(worker, '_query_session_id', None)}, "
                f"parked {worker.human.awaiting_user()}, "
                f"deferral {worker.human.has_deferral}"))
        return CommandResult("Event chain", rows)


WS_COMMANDS: CommandRegistry = COMMANDS.for_surface("ws", CancelCommand(), DiagCommand())
