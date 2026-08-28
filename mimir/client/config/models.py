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


def profile_for_model(model: str) -> dict:
    """Return the matching vLLM profile for *model* (longest case-insensitive prefix).

    Matches the resolution used for request ``extra_body`` (e.g. ``mistral-medium:128b``
    → ``mistral``), but returns the FULL profile (no key stripping) so client-only
    settings such as ``pin_role`` / ``max_tools`` can be read. Returns ``{}`` on no match.
    """
    model_lower = (model or "").lower().replace("\\", "/").strip()
    parts = [p for p in model_lower.split("/") if p]
    candidates: list[str] = [model_lower]
    if parts:
        candidates.append(parts[-1])
        candidates.extend(parts)
    best_key = max(
        (k for k in VLLM_MODEL_PROFILES if any(c.startswith(k.lower()) for c in candidates)),
        key=len,
        default=None,
    )
    return dict(VLLM_MODEL_PROFILES.get(best_key, {})) if best_key else {}


def resolve_pin_role(model: str, override: str = "") -> str:
    """Discovery-pin attachment role: explicit *override* wins, else the model's
    profile ``pin_role``, else ``"system"`` (see agent_loop._inject_pin)."""
    if override:
        return override
    role = profile_for_model(model).get("pin_role")
    return role if role in ("system", "user", "append_user") else "system"


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
