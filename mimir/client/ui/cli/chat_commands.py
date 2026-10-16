"""The terminal's side of the session commands: its host, its own commands, its text.

The grammar is shared (``session.commands``); what is the terminal's is how a result
is printed, and the two commands that act on the REPL's own state — ``/compact`` on
the history it holds, ``/ledger`` on the last answer it printed.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from ...core.settings import read_setting
from ...session.commands import (
    COMMANDS, Command, CommandHost, CommandRegistry, CommandResult,
)
from ...tool_execution.direct import call_tool_directly


class CliHost(CommandHost):
    """The agent this terminal runs in-process, and the REPL state around it."""

    surface = "cli"

    def __init__(
        self,
        agent: Any,
        *,
        compact: Callable[[], Awaitable[None]] | None = None,
        set_compact_threshold: Callable[[int], None] | None = None,
        ledger: Callable[[], str | None] | None = None,
    ) -> None:
        self.agent = agent
        self.compact = compact
        self.set_compact_threshold = set_compact_threshold
        self.ledger = ledger

    def apply(self, name: str, value: Any) -> str:
        try:
            self.agent.apply_setting(name, value)
        except ValueError as exc:
            return str(exc)
        return ""

    def read(self, name: str) -> Any:
        return read_setting(self.agent, name)

    async def call_tool(self, tool: str, arguments: dict) -> dict:
        return await call_tool_directly(self.agent, tool, arguments)


class CompactCommand(Command):
    name = "compact"
    surfaces = frozenset({"cli"})
    help = (("", "summarize and compress conversation history"),
            ("threshold <N>", "set auto-compaction threshold (default 10)"))

    async def run(self, host: CliHost, args: list[str]) -> CommandResult:
        if args and args[0].lower() == "threshold":
            n = int(args[1]) if len(args) >= 2 and args[1].isdigit() else 0
            if n < 1:
                return CommandResult.error(
                    "Usage: /compact threshold <N> (N must be a positive integer)")
            if host.set_compact_threshold is None:
                return CommandResult.error("Threshold setter not available.")
            host.set_compact_threshold(n)
            return CommandResult(f"⚡ Auto-compact threshold set to {n} exchanges.")
        if host.compact is None:
            return CommandResult.error("Compact not available.")
        await host.compact()   # reports for itself
        return CommandResult(card=False)


class LedgerCommand(Command):
    name = "ledger"
    surfaces = frozenset({"cli"})
    help = (("", "expand the verification ledger of the last answer"),)

    async def run(self, host: CliHost, args: list[str]) -> CommandResult:
        block = host.ledger() if host.ledger is not None else None
        if not block:
            return CommandResult(
                "No verification ledger yet — nothing was written or left open.",
                tone="empty")
        return CommandResult(block, card=False)


CLI_COMMANDS: CommandRegistry = COMMANDS.for_surface(
    "cli", CompactCommand(), LedgerCommand())

_EXTRA_HELP = (
    "  @<path>[:a-b]              -> attach a workspace file (or lines a-b) to a query, "
    "e.g. @src/foo.py:10-20\n"
    "  quit                       -> exit"
)


def render_text(result: CommandResult) -> str:
    """A result as the terminal prints it."""
    if result.failed:
        return f"\n❌ {result.title}\n"
    lines = [result.title] if result.title else []
    for item in result.items:
        lines.append(f"  {item.label}" + (f" — {item.detail}" if item.detail else ""))
    if result.note:
        lines.append(result.note)
    if result.label == "/help":
        lines.append(_EXTRA_HELP)
    return "\n" + "\n".join(lines) + "\n" if lines else ""


async def run_cli_command(host: CliHost, text: str) -> str | None:
    """Run *text* as a command: the text to print, or None if it is not a command."""
    result = await CLI_COMMANDS.run(host, text)
    return None if result is None else render_text(result)
