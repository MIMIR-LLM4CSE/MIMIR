"""The commands that change one of the user's settings, or report it when bare.

Each reports the value the agent now HOLDS, read back after the change, not the one
asked for: the request is a wish, the attribute is the fact, and echoing the request is
what let a front-end announce a depth the agent was not on.
"""
from __future__ import annotations

from ...config import (
    APPROVAL_MODE_ALIASES,
    BACKENDS,
    SUBAGENT_LEVEL_ALIASES,
    SUBAGENT_LEVELS,
    TEMPERATURE_BACKENDS,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    THINKING_DEPTH_LABELS,
    parse_temperature,
    thinking_depth_from_label,
)
from ...config.preferences import save_subagent_level
from ...query_engine.backends.factory import switch_backend
from .base import Command, CommandHost, CommandResult, Item, apply_setting_or_error

APPROVAL_LABELS: dict[str, str] = {
    "manual": "manual", "auto": "auto (tools)", "auto_all": "auto (everything)",
}

_ON = ("on", "true", "1", "yes")
_OFF = ("off", "false", "0", "no")


def _flag(args: list[str]) -> bool | None:
    word = args[0].lower()
    return True if word in _ON else False if word in _OFF else None


def on_off(value: bool) -> str:
    return "on" if value else "off"


class ModeCommand(Command):
    name = "mode"
    help = (("agent", "enable autonomous tool mode"),
            ("plan", "read-only exploration, then a plan you approve before any work"),
            ("ask", "read-only Q&A about the codebase; no edits, no plan"))

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if args:
            _, error = apply_setting_or_error(host, "mode", args[0])
            if error:
                return error
        mode = host.read("mode")
        title = f"Switched mode to: {mode}" if args else f"Current mode: {mode}"
        return CommandResult.setting(title, {"type": "mode", "mode": mode})


class ThinkingCommand(Command):
    """``/thinking``, ``/think`` and ``/thinking-depth`` are one setting: the rung.

    ``on`` is ``auto`` (the model calibrates its own depth), a digit is the rung by
    index — what the webview's slider sends.
    """
    name = "thinking"
    aliases = ("think", "thinking-depth")
    help = (("|".join(THINKING_DEPTH_LABELS),
             "reasoning depth; auto (default) lets the model calibrate per turn, "
             "the rest impose a fixed budget"),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if args:
            word = args[0]
            level = (int(word) if word.isdigit() else thinking_depth_from_label(word))
            if level is None or not 0 <= level < len(THINKING_DEPTH_LABELS):
                return self.usage()
            _, error = apply_setting_or_error(host, "thinking_depth", level)
            if error:
                return error
        depth = host.read("thinking_depth")
        label = THINKING_DEPTH_LABELS[depth] if 0 <= depth < len(THINKING_DEPTH_LABELS) else ""
        title = (f"Thinking depth set to {label}." if args
                 else f"Thinking is currently: {label}")
        return CommandResult.setting(
            title, {"type": "thinking_depth", "depth": depth, "label": label})


class StreamingCommand(Command):
    name = "streaming"
    aliases = ("stream",)
    help = (("on|off", "stream tokens as they arrive"),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if args:
            on = _flag(args)
            if on is None:
                return self.usage()
            _, error = apply_setting_or_error(host, "streaming", on)
            if error:
                return error
        enabled = bool(host.read("streaming"))
        title = (f"Streaming mode turned {on_off(enabled)}." if args
                 else f"Streaming is currently: {on_off(enabled)}")
        return CommandResult.setting(title, {"type": "streaming", "enabled": enabled})


class BatchCommand(Command):
    name = "batch"
    help = (("on|off", "batch all write approvals until end of turn"),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if args:
            on = _flag(args)
            if on is None:
                return self.usage()
            _, error = apply_setting_or_error(host, "batch", on)
            if error:
                return error
        enabled = bool(host.read("batch"))
        return CommandResult.setting(
            f"Batch approval mode turned {on_off(enabled)}." if args
            else f"Batch mode is currently: {on_off(enabled)}")


class ApprovalsCommand(Command):
    name = "approvals"
    help = (("manual|auto|all",
             "who answers the approval cards; manual: you do; auto: sensitive tools "
             "pass, leaving the workspace still asks; all: nothing asks (guardrails "
             "still apply)"),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if args:
            word = args[0].lower()
            _, error = apply_setting_or_error(
                host, "approval_mode", APPROVAL_MODE_ALIASES.get(word, word))
            if error:
                return error
        mode = host.read("approval_mode")
        label = APPROVAL_LABELS.get(mode, mode)
        state = {"type": "approval_mode", "mode": mode}
        if not args:
            return CommandResult.setting(
                f"Approvals are currently: {label}  (manual=you answer every card; "
                "auto=sensitive tools pass, leaving the workspace still asks; "
                "all=nothing asks)", state)
        if mode == "manual":
            return CommandResult.setting(
                "Approvals switched to: manual — every card comes back to you.", state)
        return CommandResult.setting(
            f"⚠ Approvals switched to: {label} — "
            f"{'nothing' if mode == 'auto_all' else 'no sensitive tool'} will ask you "
            "again this session. Guardrails (denylist, sandbox) still apply.", state)


class ContextCommand(Command):
    name = "context"
    help = (("compact|full",
             "compact: aggressive compaction (small models); full: keep all tool "
             "messages in history (200K+ models)"),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if args:
            _, error = apply_setting_or_error(host, "context_mode", args[0])
            if error:
                return error
        mode = host.read("context_mode")
        title = (f"Context mode switched to: {mode}." if args
                 else f"Context mode is currently: {mode}  "
                      "(compact=aggressive compaction, full=keep all tool messages)")
        return CommandResult.setting(title, {"type": "context_mode", "mode": mode})


class EnforcementCommand(Command):
    name = "enforcement"
    help = (("strict|light|off",
             "guidance-nudge level (verification/safety always on); strict: all "
             "guidance; light: drop discovery nudge; off: no guidance"),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if args:
            _, error = apply_setting_or_error(host, "enforcement", args[0])
            if error:
                return error
        level = host.read("enforcement")
        title = (f"Enforcement level switched to: {level}." if args
                 else f"Enforcement is currently: {level}  (strict=all guidance nudges, "
                      "light=drop discovery nudge, off=no guidance; verification/safety "
                      "always on)")
        return CommandResult.setting(title, {"type": "enforcement", "mode": level})


class TemperatureCommand(Command):
    name = "temperature"
    help = ((f"<{TEMPERATURE_MIN:g}-{TEMPERATURE_MAX:g}>|default",
             "sampling temperature for this model, kept across restarts; default "
             "sends none, the model's own applies"),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if args:
            ok, value = parse_temperature(args[0])
            if not ok:
                return self.usage()
            _, error = apply_setting_or_error(host, "temperature", value)
            if error:
                return error
        state = host.temperature_state()
        value = state.get("value")
        backend = host.read("backend")
        ignored = ("" if str(backend).lower() in TEMPERATURE_BACKENDS
                   else f" The {backend} backend ignores it.")
        if args:
            shown = "default (the model's own)" if value is None else f"{value:g}"
            title = f"Temperature set to: {shown}.{ignored}"
        else:
            shown = "default" if value is None else f"{value:g}"
            meaning = ("the model's own, none sent" if value is None
                       else "set for this model")
            title = f"Temperature is currently: {shown} ({meaning}).{ignored}"
        return CommandResult.setting(title, {"type": "temperature", **state})


class SubagentsCommand(Command):
    """How far a sub-agent may go is the user's call, not the model's."""
    name = "subagents"
    help = (("|".join(SUBAGENT_LEVELS),
             "explore=read only; build=+edit and run commands, each in a copy of the "
             "repository on its own branch"),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if args:
            word = args[0].lower()
            # The rung's former name is still accepted: it is in everyone's
            # preferences who set it, and refusing it on one front-end only was a gap.
            if word not in SUBAGENT_LEVELS and word not in SUBAGENT_LEVEL_ALIASES:
                return self.usage()
            level, error = apply_setting_or_error(host, "subagent_level", word)
            if error:
                return error
            # Persisted here as well as by the agent: a new agent reads it from there,
            # and there may be no live one to do it.
            save_subagent_level(level)
        level = host.read("subagent_level")
        title = (f"Sub-agents may now: {level}." if args
                 else f"Sub-agents may currently: {level}  (explore=read only; "
                      "build=+edit and run commands, each in a copy of the repository "
                      "on its own branch)")
        return CommandResult.setting(title, {"type": "subagent_level", "level": level})


class BackendCommand(Command):
    """Process-wide, so switched even with no agent up yet."""
    name = "backend"
    help = (("|".join(BACKENDS), "switch the LLM backend"),)

    async def run(self, host: CommandHost, args: list[str]) -> CommandResult:
        if not args:
            return CommandResult("Backend", [Item(str(host.read("backend")))])
        name = args[0].lower()
        if name not in BACKENDS:
            return CommandResult.error(
                f"Unknown backend: {name}. Use {', '.join(BACKENDS)}.")
        switch_backend(name)
        _, error = apply_setting_or_error(host, "backend", name)
        if error:
            return error
        # The one setting that still speaks: it owns no control anywhere, and
        # switching the LLM in silence would be the worse trade.
        return CommandResult("Backend", [Item(name)])


SETTING_COMMANDS: tuple[Command, ...] = (
    ModeCommand(), ContextCommand(), EnforcementCommand(), TemperatureCommand(),
    ThinkingCommand(), StreamingCommand(), BatchCommand(), ApprovalsCommand(),
    SubagentsCommand(), BackendCommand(),
)
