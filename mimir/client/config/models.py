from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_MODEL = os.environ.get("MIMIR_DEFAULT_MODEL", "")

VALID_MODES: tuple[str, ...] = ("agent", "plan", "ask")

# Modes whose tool surface is read-only: PLAN_BLOCKED tools are hidden from the
# model and the dual-use PLAN_READONLY exec tool is restricted, at call time, to
# read-only commands. "plan" produces a checklist for approval; "ask" just answers.
READONLY_MODES: frozenset[str] = frozenset({"plan", "ask"})

# LLM backend selection — LLM_BACKEND=vllm (local vLLM, default), anthropic/claude
# (hosted Claude API, for evaluation against local models), else Ollama.
LLM_BACKEND   = os.environ.get("LLM_BACKEND",   "vllm")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL",  "http://127.0.0.1:8000")
VLLM_API_KEY  = os.environ.get("VLLM_API_KEY",   "EMPTY")

def _load_vllm_model_profiles() -> dict[str, dict]:
    """Load shared vLLM model profiles from JSON.

    This file is consumed by both Python and the VS Code extension so new
    models are added in one place only.
    """
    profiles_path = Path(__file__).with_name("vllm_model_profiles.json")
    try:
        with profiles_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


# Shared per-model settings used by both Python request-time logic and
# extension-time vLLM launch command generation.
VLLM_MODEL_PROFILES: dict[str, dict] = _load_vllm_model_profiles()


def _normalize_model_key(name: str) -> str:
    """Fold a model name to the form profile keys are matched in.

    Version separators are not stable across publishers: the same family ships as
    ``Llama-3_1-Nemotron`` on one repo and ``Llama-3.1-Nemotron`` on another. Matching
    the raw spelling means a family key silently covers one and misses the other — and
    a missed profile costs that model its reasoning, which is precisely the failure the
    profile exists to prevent. So ``_`` and ``.`` are treated as the same separator.
    """
    return (name or "").lower().replace("\\", "/").replace("_", ".").strip()


def profile_for_model(model: str) -> dict:
    """Return the matching profile for *model* (longest case-insensitive prefix).

    Matched against the whole name, its last path segment, and each segment, so
    ``nvidia/Llama-3_1-Nemotron-Ultra-253B-v1`` reaches a ``llama-3.1-nemotron``
    family key. Returns ``{}`` on no match, which is a normal outcome: every default
    is chosen so that an unlisted model still behaves correctly.
    """
    model_lower = _normalize_model_key(model)
    parts = [p for p in model_lower.split("/") if p]
    candidates: list[str] = [model_lower]
    if parts:
        candidates.append(parts[-1])
        candidates.extend(parts)
    best_key = max(
        (
            k for k in VLLM_MODEL_PROFILES
            if any(c.startswith(_normalize_model_key(k)) for c in candidates)
        ),
        key=len,
        default=None,
    )
    return dict(VLLM_MODEL_PROFILES.get(best_key, {})) if best_key else {}


# How a model is told to reason. "kwarg" is the default for every model, listed or
# not: `chat_template_kwargs.enable_thinking` is what thinking-capable vLLM templates
# read, and a template that doesn't know the kwarg ignores it. Only a model steered
# some other way needs a profile entry.
THINKING_MECHANISMS: frozenset[str] = frozenset({"kwarg", "directive", "effort"})
DEFAULT_THINKING_MECHANISM = "kwarg"

# Effort rungs, weakest first, for the "effort" mechanism when a family does not name
# its own. OpenAI's scale is the default only because it is the most common one — it
# is NOT universal: DeepSeek-V4 and GLM take low/high/max, and sending them "medium"
# lands on the template's fallback instead of the rung the user picked. Any family
# whose ladder differs must declare `effort_levels`.
DEFAULT_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high")
DEFAULT_EFFORT_PARAM = "reasoning_effort"


def thinking_profile(model: str) -> dict:
    """Everything needed to switch *model*'s reasoning, resolved from its family.

    One descriptor feeds both sides — the request builder and the panel's depth
    control — so the rungs a user sees are the rungs actually sent. Keys:

    - ``mechanism``    — ``"kwarg"`` (send ``chat_template_kwargs.enable_thinking``,
      the default), ``"directive"`` (a trained-on line at the head of the system
      message) or ``"effort"`` (a request param naming a rung).
    - ``effort_param`` / ``levels`` — the param carrying the rung, and the family's
      own ladder weakest-first. Ladders differ between families, so they are data.
    - ``toggle_param`` / ``toggle_values`` — an explicit on/off param, where the
      family has one. Without it an "effort" family cannot stop reasoning, and the
      control drops its "off" rung rather than offering a switch that does nothing.
    - ``directive``    — the ``{on, off}`` strings for the directive mechanism.

    Defaulting to ``"kwarg"`` rather than to "no thinking" is deliberate: the served
    model name is whatever ``--served-model-name`` says, so a name that matches no
    family is the normal case, not an error, and it must not silently cost the user
    their reasoning.
    """
    profile = profile_for_model(model)
    declared = profile.get("thinking")
    mechanism = declared if declared in THINKING_MECHANISMS else DEFAULT_THINKING_MECHANISM

    levels = profile.get("effort_levels")
    if not (isinstance(levels, list) and all(isinstance(x, str) and x for x in levels)):
        levels = list(DEFAULT_EFFORT_LEVELS)

    toggle_values = profile.get("toggle_values")
    if not (isinstance(toggle_values, dict) and toggle_values.get("on") and toggle_values.get("off")):
        toggle_values = {}
    toggle_param = profile.get("toggle_param") or ""

    directive = profile.get("thinking_directive")
    if not (isinstance(directive, dict) and directive.get("on") and directive.get("off")):
        directive = {}

    return {
        "mechanism": mechanism,
        "effort_param": profile.get("effort_param") or DEFAULT_EFFORT_PARAM,
        "levels": list(levels),
        "toggle_param": toggle_param if toggle_values else "",
        "toggle_values": toggle_values if toggle_param else {},
        "directive": directive,
    }


def thinking_mechanism(model: str) -> str:
    """Shorthand for ``thinking_profile(model)["mechanism"]``."""
    return thinking_profile(model)["mechanism"]


def thinking_can_disable(model: str) -> bool:
    """Whether reasoning can actually be turned off for *model*.

    False only for an "effort" family with no on/off param: those always reason, so
    an "off" rung in the UI would be a control that does nothing.
    """
    profile = thinking_profile(model)
    return profile["mechanism"] != "effort" or bool(profile["toggle_param"])


def enforcement_level(model: str) -> str:
    """How much reasoning babysitting to apply, from the model's vLLM profile.

    Governs ONLY the **guidance** nudge layer (env resolution/cleanup, discovery, doc,
    state, blast-radius, creation, todo, validation) plus the plan-mode explore phase —
    never safety, approval, write-policy or verification guards, which run at every
    level. Which categories survive per level and mode is the single table
    ``_GUIDANCE_BY_LEVEL_MODE`` in ``guardrails/nudges/engine.py``:
      - "strict": every guidance category.
      - "light":  only the ones guarding a costly, hard-to-detect, non-self-correcting
                  mistake — blast-radius, env cleanup, validation.
      - "off":    no guidance at all.

    **Default "light"**, with "strict" available as an explicit opt-in. The `light` set
    is already defined by the right criterion — a mistake that is expensive, hard to
    notice and does not self-correct — and everything it drops is procedural
    hand-holding a capable model does unprompted, paid for in tokens and in
    interruptions to its own plan on every step. Having that as the opt-down rather
    than the default meant the burden of proof sat on the wrong side.

    Weak local models opt *up* by declaring ``"enforcement": "strict"`` in their
    profile. Unknown values fall back to "light".
    """
    level = profile_for_model(model).get("enforcement", "light")
    return level if level in ("strict", "light", "off") else "light"


def resolve_enforcement(agent: object) -> str:
    """Read an agent's resolved enforcement level.

    Prefers the cached ``agent.enforcement`` attribute — set once at ``MimirAgent``
    construction from the model profile (``enforcement_level``) and overridable at
    runtime via ``MimirAgent.set_enforcement`` / the ``/enforcement`` command. The
    model is immutable for an agent's lifetime, so there is nothing to re-resolve
    per turn. Falls back to resolving from ``agent.model`` for agent-like objects
    that predate the attribute (e.g. lightweight test fakes).
    """
    level = getattr(agent, "enforcement", None)
    if level in ("strict", "light", "off"):
        return level
    return enforcement_level(getattr(agent, "model", ""))
