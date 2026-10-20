"""Commands that read or change the agent itself, offered where one lives in-thread.

The WebSocket front-end shows the same things as controls — the status bar, the
toggle panel, the approval cards — so these are the terminal's.
"""
from __future__ import annotations

from ...config import THINKING_DEPTH_LABELS
from .base import Command, CommandHost, CommandResult, Item
from .settings import APPROVAL_LABELS, on_off

_CLI = frozenset({"cli"})


def _no_agent(name: str) -> CommandResult:
    return CommandResult.error(f"/{name} is not available here.")


def _trusted(agent) -> str:
    items = sorted(set(agent.approvals.trusted_tools_view()))
    return ", ".join(items) if items else "none"


class HelpCommand(Command):
    name = "help"
    surfaces = _CLI

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        assert host.registry is not None, "a registry runs every command"
        return CommandResult(host.registry.help_text(), card=False)


class StatusCommand(Command):
    name = "status"
    surfaces = _CLI
    help = (("", "show the current settings"),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        depth = host.read("thinking_depth")
        temperature = host.read("temperature")
        approvals = host.read("approval_mode")
        fields = [
            ("Current mode", host.read("mode")),
            ("thinking", THINKING_DEPTH_LABELS[depth]
             if 0 <= depth < len(THINKING_DEPTH_LABELS) else on_off(host.read("thinking"))),
            ("streaming", on_off(host.read("streaming"))),
            ("batch", on_off(host.read("batch"))),
            ("context", host.read("context_mode")),
            ("enforcement", host.read("enforcement")),
            ("temperature", "default" if temperature is None else f"{temperature:g}"),
            ("approvals", APPROVAL_LABELS.get(approvals, approvals)),
        ]
        if host.agent is not None:
            fields.append(("trusted", _trusted(host.agent)))
        return CommandResult(" | ".join(f"{k}: {v}" for k, v in fields), card=False)


class ResourcesCommand(Command):
    name = "resources"
    surfaces = _CLI
    help = (("", "list attachable MCP resources (use @<uri> to attach one to a query)"),)

    _FILE_HINT = ("You can also attach a workspace file with @<path>[:a-b], "
                  "e.g. @src/foo.py:10-20.")

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        resources = host.agent.resources or {}
        if not resources:
            return CommandResult("No MCP resources are declared by connected servers.",
                                 note=self._FILE_HINT, tone="empty")
        items = []
        for uri, info in sorted(resources.items()):
            name, desc = info.get("name", ""), info.get("description", "")
            label = uri + (f"  ({name})" if name and name != uri else "")
            items.append(Item(label, desc))
        return CommandResult(
            "Attachable MCP resources (reference with @<uri> in your message):",
            items, note=self._FILE_HINT)


class TrustCommand(Command):
    name = "trust"
    surfaces = _CLI
    help = (("<tool>", "auto-approve a sensitive tool for this session"),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if host.agent is None:
            return _no_agent(self.name)
        if not args:
            return CommandResult(f"Session-trusted tools: {_trusted(host.agent)}",
                                 note="Usage: /trust <tool_name>", card=False)
        host.agent.approvals.trust_tool(args[0])
        return CommandResult(f"✅ '{args[0]}' auto-approved for this session.")


class UntrustCommand(Command):
    name = "untrust"
    surfaces = _CLI
    help = (("<tool>", "remove session auto-approval for a tool"),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if host.agent is None:
            return _no_agent(self.name)
        if not args:
            return self.usage()
        host.agent.approvals.untrust_tool(args[0])
        return CommandResult(f"🔒 '{args[0]}' removed from session trust list.")


class ToggleCommand(Command):
    """``/servers``, ``/skills``, ``/nudges``: list them, or switch one on or off."""
    surfaces = _CLI

    def __init__(self, kind: str, what: str) -> None:
        self.name = kind
        self.kind = kind
        self.help = (("[on|off <name>]", what),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        agent = host.agent
        if agent is None:
            return _no_agent(self.name)
        rows = agent.toggles_state().get(self.kind, [])
        singular = self.kind[:-1]
        if not args:
            return CommandResult(
                f"{self.kind.capitalize()} (✓ = active, ✗ = hidden):",
                [Item(f"[{'✓' if r['enabled'] else '✗'}] {r['name']}",
                      r.get("description") or "") for r in rows],
                note=f"Usage: /{self.name} on|off <name>")
        if len(args) >= 2 and args[0].lower() in ("on", "off"):
            enabled, name = args[0].lower() == "on", args[1]
            if name not in {r["name"] for r in rows}:
                return CommandResult.error(f"Unknown {singular}: {name}")
            agent.set_enabled(singular, name, enabled)
            return CommandResult(
                f"{'✓ enabled' if enabled else '✗ disabled'} {singular}: {name}")
        return CommandResult.error(f"Usage: /{self.name} on|off <name>")


class UndoCommand(Command):
    name = "undo"
    surfaces = _CLI
    help = (("", "revert all file changes made in the last agent turn"),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if host.agent is None:
            return CommandResult.error("Undo is not available (no approval manager attached).")
        messages: list[str] = []
        reverted = host.agent.approvals.batch.revert(output_fn=messages.append)
        if reverted:
            return CommandResult(f"↩  Reverted {len(reverted)} file(s).",
                                 [Item(m) for m in messages], tone="warn")
        return CommandResult(messages[0] if messages else "Nothing to undo.", tone="empty")


AGENT_COMMANDS: tuple[Command, ...] = (
    HelpCommand(), StatusCommand(), ResourcesCommand(), TrustCommand(), UntrustCommand(),
    ToggleCommand("servers", "list/toggle MCP servers (hide their tools from the LLM)"),
    ToggleCommand("skills", "list/toggle skills (a switched-off skill is hidden from "
                            "/<name> and from what the model can load)"),
    ToggleCommand("nudges", "list/toggle application nudges (extension packs)"),
    UndoCommand(),
)
