"""The user's settings, as one table: what each one may be, and how it reaches an agent.

Every front-end changes an agent the same way — ``apply_setting(agent, name, value)``
— and reads it the same way. A setting is validated here, once: an invalid value raises
``ValueError`` with the message to show, and a setting that cannot fail (a flag) does
not. What is applied is the agent's own attribute, read live by the loop at every step,
so a change lands on the next model call of a query already running.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from ..config.models import VALID_MODES, enforcement_level
from ..config.preferences import load_temperature, save_subagent_level, save_temperature
from ..config.settings import (
    DEFAULT_SUBAGENT_LEVEL,
    DEFAULT_THINKING_DEPTH,
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


def _one_of(value: str, allowed: tuple[str, ...], what: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized not in allowed:
        raise ValueError(f"Invalid {what}. Use " + ", ".join(f"'{a}'" for a in allowed) + ".")
    return normalized


def _set_mode(agent: Any, mode: str) -> str:
    agent.mode = normalize_mode(mode)
    return agent.mode


def _set_thinking_depth(agent: Any, level: int) -> int:
    """A rung of the thinking ladder, keeping the derived flags coherent."""
    agent.thinking_depth = clamp_thinking_depth(level)
    agent.thinking = agent.thinking_depth > 0
    agent.thinking_budget = THINKING_DEPTH_BUDGETS[agent.thinking_depth]
    return agent.thinking_depth


def _set_thinking(agent: Any, enabled: bool) -> bool:
    """The on/off switch: "on" means "auto" — the model calibrates its own depth."""
    _set_thinking_depth(agent, DEFAULT_THINKING_DEPTH if enabled else 0)
    return agent.thinking


def _set_thinking_budget(agent: Any, budget: int) -> int:
    """The token budget, without moving the rung (manual/advanced use)."""
    agent.thinking_budget = int(budget)
    return agent.thinking_budget


def _set_streaming(agent: Any, enabled: bool) -> bool:
    agent.streaming = bool(enabled)
    return agent.streaming


def _set_batch(agent: Any, enabled: bool) -> bool:
    agent.approvals.batch_mode = bool(enabled)
    return agent.approvals.batch_mode


def _set_context_mode(agent: Any, mode: str) -> str:
    agent.context_mode = _one_of(mode, ("compact", "full"), "context mode")
    return agent.context_mode


def _set_enforcement(agent: Any, level: str) -> str:
    agent.enforcement = _one_of(level, ("strict", "light", "off"), "enforcement level")
    return agent.enforcement


def _set_approval_mode(agent: Any, mode: str) -> str:
    """How much of the approval flow the user still answers by hand.

    ``manual`` (default) / ``auto`` (sensitive tools pass, leaving the workspace still
    asks) / ``auto_all`` (nothing asks). Session state — see
    ``ApprovalManager.approval_mode`` for what an auto mode does and does not lift.
    """
    agent.approvals.approval_mode = _one_of(mode, agent.approvals.APPROVAL_MODES,
                                            "approval mode")
    return agent.approvals.approval_mode


def _set_temperature(agent: Any, value: float | None) -> float | None:
    """This model's sampling temperature, persisted; None restores the model's own."""
    if value is not None:
        value = float(value)
        if not TEMPERATURE_MIN <= value <= TEMPERATURE_MAX:
            raise ValueError(
                f"Temperature must be between {TEMPERATURE_MIN:g} and {TEMPERATURE_MAX:g}.")
    agent.temperature = value
    save_temperature(agent.model, value)
    return value


def _set_subagent_level(agent: Any, level: str) -> str:
    """How far a sub-agent may go, persisted. An unknown rung lands on the weakest one
    rather than raising: "here is what is in force" is more use than an error."""
    agent.subagent_level = clamp_subagent_level(level)
    save_subagent_level(agent.subagent_level)
    return agent.subagent_level


def _set_backend(agent: Any, name: str) -> str:
    from ..query_engine.backends.factory import switch_backend
    switch_backend(name)
    agent.backend = name
    return name


def _set_model(agent: Any, model: str) -> str:
    """Switch the served model mid-session.

    The model is read live at every LLM call, so mutating it is enough for the next
    call to use it — no reconnect. ``enforcement`` and the temperature are re-derived
    from the new model, as they are resolved at construction.

    Nothing is written to ``os.environ``: a server's environment is a copy frozen at
    spawn, so a write there reaches no running server, and with concurrent sessions in
    one process it would change what *another* session's servers were told.
    """
    model = (model or "").strip()
    if not model:
        raise ValueError("Model name must not be empty.")
    agent.model = model
    agent.enforcement = enforcement_level(model)
    agent.temperature = load_temperature(model)
    return model


@dataclass(frozen=True)
class Setting:
    name: str
    apply: Callable[[Any, Any], Any]
    read: Callable[[Any], Any]
    # What is in force with no agent yet: the value the greeting reports.
    default: Callable[[], Any]


def _attr(name: str) -> Callable[[Any], Any]:
    return lambda agent: getattr(agent, name)


SETTINGS: dict[str, Setting] = {s.name: s for s in (
    Setting("mode", _set_mode, _attr("mode"), lambda: "agent"),
    Setting("model", _set_model, _attr("model"), lambda: ""),
    Setting("thinking", _set_thinking, _attr("thinking"), lambda: DEFAULT_THINKING_DEPTH > 0),
    Setting("thinking_depth", _set_thinking_depth, _attr("thinking_depth"),
            lambda: DEFAULT_THINKING_DEPTH),
    Setting("thinking_budget", _set_thinking_budget, _attr("thinking_budget"),
            lambda: THINKING_DEPTH_BUDGETS[DEFAULT_THINKING_DEPTH]),
    Setting("streaming", _set_streaming, _attr("streaming"), lambda: True),
    Setting("batch", _set_batch, lambda agent: agent.approvals.batch_mode, lambda: True),
    Setting("context_mode", _set_context_mode, _attr("context_mode"), lambda: "full"),
    Setting("enforcement", _set_enforcement, _attr("enforcement"), lambda: "light"),
    Setting("approval_mode", _set_approval_mode,
            lambda agent: agent.approvals.approval_mode, lambda: "manual"),
    Setting("temperature", _set_temperature, _attr("temperature"), lambda: None),
    Setting("subagent_level", _set_subagent_level, _attr("subagent_level"),
            lambda: DEFAULT_SUBAGENT_LEVEL),
    Setting("backend", _set_backend, _attr("backend"),
            lambda: os.environ.get("LLM_BACKEND", "vllm")),
)}


def apply_setting(agent: Any, name: str, value: Any) -> Any:
    """Apply *value* to *agent*'s setting *name*; returns the value now in force.

    Raises ``ValueError`` for a value the setting refuses, ``KeyError`` for an unknown
    setting.
    """
    return SETTINGS[name].apply(agent, value)


def read_setting(agent: Any, name: str) -> Any:
    """*agent*'s current value of *name* — or what is in force with no agent yet."""
    setting = SETTINGS[name]
    return setting.default() if agent is None else setting.read(agent)
