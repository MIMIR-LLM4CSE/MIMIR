"""Session commands: what the user types after a ``/`` and the model never sees.

One grammar for every front-end. A :class:`Command` parses its arguments, acts through
the :class:`CommandHost` it is given, and returns a :class:`CommandResult` — never text
already laid out for a terminal, nor a payload already shaped for a socket. Each
front-end renders the result its own way: the CLI prints it, the WebSocket sends a
state message, a result card or an error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ...config import TEMPERATURE_BACKENDS
from ...core.settings import parse_setting


@dataclass(frozen=True)
class Item:
    label: str
    detail: str = ""


@dataclass
class CommandResult:
    """What a command has to say, for a front-end to lay out.

    Three shapes:

    - a SETTING changed or reported — ``state`` is the message carrying its value, for
      a front-end that shows settings as controls; ``card`` is False, since a line in
      the transcript would only repeat the control (and the webview replays its stored
      settings on connect, so such lines greeted each session with changes nobody had
      made);
    - a LISTING, or something IRREVERSIBLE that just happened — a card: ``title``,
      ``items``, ``note``;
    - an ERROR — ``tone == "error"`` and the reason in ``title``.

    ``tone``: ``ok`` routine, ``warn`` irreversible, ``empty`` a listing with nothing in
    it, ``error``. ``label`` is the command as the card names it ("/memory list").
    """
    title: str = ""
    items: list[Item] = field(default_factory=list)
    note: str = ""
    tone: str = "ok"
    state: dict | None = None
    card: bool = True
    label: str = ""

    @classmethod
    def error(cls, text: str) -> "CommandResult":
        return cls(title=text, tone="error", card=False)

    @classmethod
    def setting(cls, title: str, state: dict | None = None) -> "CommandResult":
        return cls(title=title, state=state, card=False)

    @property
    def failed(self) -> bool:
        return self.tone == "error"


class CommandHost:
    """What a command may reach on the front-end that ran it.

    ``agent`` is None where the front-end holds no agent in this thread (the WebSocket
    reaches its agents through workers); a command that needs one is offered on the
    surfaces that have it.
    """

    surface: str = ""
    agent: Any = None
    # The registry running the command, for one that lists the others (/help).
    registry: "CommandRegistry | None" = None

    def apply(self, name: str, value: Any) -> str:
        """Apply a setting; "" on success, else why it was refused."""
        raise NotImplementedError

    def read(self, name: str) -> Any:
        """The value of a setting now in force."""
        raise NotImplementedError

    async def call_tool(self, tool: str, arguments: dict) -> dict:
        """Call one tool directly, around the guardrails — see ``call_tool_directly``."""
        raise NotImplementedError

    def temperature_state(self) -> dict:
        backend = str(self.read("backend")).lower()
        return {"supported": backend in TEMPERATURE_BACKENDS,
                "value": self.read("temperature")}


class Command:
    """One session command. Subclasses set the class attributes and :meth:`run`."""

    name: str = ""
    aliases: tuple[str, ...] = ()
    # The front-ends that offer it: "cli", "ws".
    surfaces: frozenset[str] = frozenset({"cli", "ws"})
    # ``/help`` lines: (usage after the name, what it does).
    help: tuple[tuple[str, str], ...] = ()

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        raise NotImplementedError

    def usage(self) -> CommandResult:
        forms = " | ".join(f"/{self.name} {u}".rstrip() for u, _ in self.help)
        return CommandResult.error(f"Usage: {forms or '/' + self.name}")


def apply_setting_or_error(host: CommandHost, name: str, value: Any
                           ) -> tuple[Any, CommandResult | None]:
    """Check and apply one setting: ``(value in force, None)`` or ``(None, error)``."""
    try:
        value = parse_setting(name, value)
    except ValueError as exc:
        return None, CommandResult.error(str(exc))
    refusal = host.apply(name, value)
    if refusal:
        return None, CommandResult.error(refusal)
    return value, None


class CommandRegistry:
    """The commands one front-end offers, by name and alias."""

    def __init__(self, commands: Iterable[Command]) -> None:
        self.commands: list[Command] = list(commands)
        self._by_name: dict[str, Command] = {}
        for command in self.commands:
            for name in (command.name, *command.aliases):
                if name in self._by_name:
                    raise ValueError(f"two commands answer to /{name}")
                self._by_name[name] = command

    def for_surface(self, surface: str, *extra: Command) -> "CommandRegistry":
        return CommandRegistry(
            [c for c in self.commands if surface in c.surfaces] + list(extra))

    def find(self, text: str) -> tuple[Command, list[str]] | None:
        """The command *text* invokes and its arguments, or None if it names none."""
        parts = text.strip().split()
        if not parts or not parts[0].startswith("/"):
            return None
        command = self._by_name.get(parts[0][1:].lower())
        return None if command is None else (command, parts[1:])

    def names(self) -> list[str]:
        return [c.name for c in self.commands]

    async def run(self, host: CommandHost, text: str) -> CommandResult | None:
        """Run the command *text* names. None when *text* is not a command at all."""
        if not text.strip().startswith("/"):
            return None
        found = self.find(text)
        if found is None:
            return CommandResult.error(f"Unknown command: {text.split()[0]}. Type /help.")
        command, args = found
        host.registry = self
        result = await command.run(host, args)
        if not result.label:
            result.label = f"/{command.name}"
        return result

    def help_text(self) -> str:
        lines = ["Commands:"]
        for command in self.commands:
            for usage, what in command.help:
                head = f"/{command.name} {usage}".rstrip()
                if len(head) > 26:
                    lines.append(f"  {head}")
                    head = ""
                lines.append(f"  {head:<26} -> {what}")
        return "\n".join(lines)
