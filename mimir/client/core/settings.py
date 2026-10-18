"""The user's settings as one table: what each may be, and how it reaches an agent.

``parse_setting`` checks a value without an agent, so a front-end can record one to
replay on an agent built later; ``apply_setting`` sets the attribute the loop reads at
every step. An invalid value raises ``ValueError`` with the message to show.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from ..config.models import VALID_MODES, enforcement_level
from ..config.preferences import load_temperature, save_subagent_level, save_temperature
from ..config.settings import (
    APPROVAL_MODES,
    CONTEXT_MODES,
    DEFAULT_SUBAGENT_LEVEL,
    DEFAULT_THINKING_DEPTH,
    ENFORCEMENT_LEVELS,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    THINKING_DEPTH_BUDGETS,
    clamp_subagent_level,
    clamp_thinking_depth,
)


def normalize_mode(mode: str) -> str:
    normalized = (mode or "").strip().lower()
    if normalized in VALID_MODES:
        return normalized
    raise ValueError("Invalid mode. Use " + " or ".join(f"'{m}'" for m in VALID_MODES) + ".")


def _one_of(allowed: tuple[str, ...], what: str) -> Callable[[str], str]:
    def parse(value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in allowed:
            raise ValueError(f"Invalid {what}. Use " + ", ".join(f"'{a}'" for a in allowed) + ".")
        return normalized
    return parse


def _parse_temperature(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not TEMPERATURE_MIN <= value <= TEMPERATURE_MAX:
        raise ValueError(
            f"Temperature must be between {TEMPERATURE_MIN:g} and {TEMPERATURE_MAX:g}.")
    return value


def _parse_name(what: str) -> Callable[[str], str]:
    def parse(value: str) -> str:
        name = (value or "").strip()
        if not name:
            raise ValueError(f"{what} must not be empty.")
        return name
    return parse


def _attr(name: str) -> Callable[[Any, Any], None]:
    return lambda agent, value: setattr(agent, name, value)


def _store_thinking_depth(agent: Any, level: int) -> None:
    """A rung of the thinking ladder, keeping the derived flags coherent."""
    agent.thinking_depth = level
    agent.thinking = level > 0
    agent.thinking_budget = THINKING_DEPTH_BUDGETS[level]


def _store_thinking(agent: Any, enabled: bool) -> None:
    """The on/off switch: "on" means "auto" — the model calibrates its own depth."""
    _store_thinking_depth(agent, DEFAULT_THINKING_DEPTH if enabled else 0)


def _store_batch(agent: Any, enabled: bool) -> None:
    agent.approvals.batch_mode = enabled


def _store_approval_mode(agent: Any, mode: str) -> None:
    """Session state — see ``ApprovalManager.approval_mode`` for what an auto mode
    does and does not lift."""
    agent.approvals.approval_mode = mode


def _store_temperature(agent: Any, value: float | None) -> None:
    """This model's sampling temperature, persisted; None restores the model's own."""
    agent.temperature = value
    save_temperature(agent.model, value)


def _store_subagent_level(agent: Any, level: str) -> None:
    agent.subagent_level = level
    save_subagent_level(level)


def _store_backend(agent: Any, name: str) -> None:
    # At call time: the query engine reads the mode through this module, so importing
    # its backend factory here at load would be a cycle.
    from ..query_engine.backends.factory import switch_backend
    switch_backend(name)
    agent.backend = name


def _store_model(agent: Any, model: str) -> None:
    """Switch the served model mid-session; enforcement and temperature follow it.

    Nothing goes to ``os.environ``: a server's environment is frozen at spawn, and in one
    process it would change what another session's servers were told.
    """
    agent.model = model
    agent.enforcement = enforcement_level(model)
    agent.temperature = load_temperature(model)


def _read(name: str) -> Callable[[Any], Any]:
    return lambda agent: getattr(agent, name)


@dataclass(frozen=True)
class Setting:
    name: str
    # Checks and normalises a value; raises ValueError with the message to show.
    parse: Callable[[Any], Any]
    store: Callable[[Any, Any], None]
    read: Callable[[Any], Any]
    # What is in force with no agent yet: the value the greeting reports.
    default: Callable[[], Any]


SETTINGS: dict[str, Setting] = {s.name: s for s in (
    Setting("mode", normalize_mode, _attr("mode"), _read("mode"), lambda: "agent"),
    Setting("model", _parse_name("Model name"), _store_model, _read("model"), lambda: ""),
    Setting("thinking", bool, _store_thinking, _read("thinking"),
            lambda: DEFAULT_THINKING_DEPTH > 0),
    Setting("thinking_depth", clamp_thinking_depth, _store_thinking_depth,
            _read("thinking_depth"), lambda: DEFAULT_THINKING_DEPTH),
    # The token budget without moving the rung (manual/advanced use).
    Setting("thinking_budget", int, _attr("thinking_budget"), _read("thinking_budget"),
            lambda: THINKING_DEPTH_BUDGETS[DEFAULT_THINKING_DEPTH]),
    Setting("streaming", bool, _attr("streaming"), _read("streaming"), lambda: True),
    Setting("batch", bool, _store_batch, lambda agent: agent.approvals.batch_mode,
            lambda: True),
    Setting("context_mode", _one_of(CONTEXT_MODES, "context mode"), _attr("context_mode"),
            _read("context_mode"), lambda: "full"),
    Setting("enforcement", _one_of(ENFORCEMENT_LEVELS, "enforcement level"),
            _attr("enforcement"), _read("enforcement"), lambda: "light"),
    Setting("approval_mode", _one_of(APPROVAL_MODES, "approval mode"), _store_approval_mode,
            lambda agent: agent.approvals.approval_mode, lambda: "manual"),
    Setting("temperature", _parse_temperature, _store_temperature, _read("temperature"),
            lambda: None),
    # An unknown rung lands on the weakest one rather than raising: "here is what is
    # in force" is more use than an error.
    Setting("subagent_level", clamp_subagent_level, _store_subagent_level,
            _read("subagent_level"), lambda: DEFAULT_SUBAGENT_LEVEL),
    Setting("backend", _parse_name("Backend name"), _store_backend, _read("backend"),
            lambda: os.environ.get("LLM_BACKEND", "vllm")),
)}


def parse_setting(name: str, value: Any) -> Any:
    """*value* as setting *name* holds it. Raises ``ValueError`` for a value the
    setting refuses, ``KeyError`` for an unknown setting."""
    return SETTINGS[name].parse(value)


def apply_setting(agent: Any, name: str, value: Any) -> Any:
    """Apply *value* to *agent*'s setting *name*; returns the value now in force."""
    setting = SETTINGS[name]
    value = setting.parse(value)
    setting.store(agent, value)
    return value


def read_setting(agent: Any, name: str) -> Any:
    """*agent*'s current value of *name* — or what is in force with no agent yet."""
    setting = SETTINGS[name]
    return setting.default() if agent is None else setting.read(agent)
