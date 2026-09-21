"""What each served model is worth, for picking a sub-agent's model.

The data lives in ``model_catalog.json`` (its ``_comment`` says what may go in it).
This module matches served ids to entries, estimates relative decode speed, and
renders the table the orchestrating model reads before it picks.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .models import longest_prefix_key

_CATALOG_PATH = Path(__file__).with_name("model_catalog.json")


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    """The whole JSON, ``{}`` when it is missing or unreadable."""
    try:
        with _CATALOG_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _entries() -> dict[str, dict]:
    return {k: v for k, v in load_catalog().items()
            if not k.startswith("_") and isinstance(v, dict)}


def catalog_entry(model: str, root: str = "") -> dict:
    """The entry for a served model, matched on its id and then on its ``root``.

    ``root`` is the checkpoint vLLM reports behind a ``--served-model-name``; it lets
    an arbitrary served name still find its family. ``{}`` when neither matches.
    """
    entries = _entries()
    for name in (model, root):
        key = longest_prefix_key(name, entries) if name else None
        if key:
            return dict(entries[key])
    return {}


def decode_cost(entry: dict) -> float | None:
    """Relative cost of decoding one token: active parameters × bytes per weight.

    Decoding is memory-bandwidth bound, so this orders models by speed on like
    hardware. ``None`` when the entry does not say how many parameters are active.
    """
    active = entry.get("active_params_b")
    if not isinstance(active, (int, float)) or active <= 0:
        return None
    weight_bytes = load_catalog().get("_weight_bytes") or {}
    return float(active) * float(weight_bytes.get(entry.get("weights"), 2))


def is_delegable(entry: dict) -> bool:
    return entry.get("delegable", True) is not False


def _fmt_params(b: float) -> str:
    return f"{b:g}B"


def _fmt_ctx(tokens: int) -> str:
    return f"{tokens // 1_000_000}M" if tokens >= 1_000_000 and tokens % 1_000_000 == 0 \
        else f"{round(tokens / 1000)}k"


def _fmt_scores(scores: dict, categories: list[str]) -> str:
    parts = []
    for cat in categories:
        values = scores.get(cat)
        if not isinstance(values, dict) or not values:
            parts.append(f"{cat} n/a")
            continue
        parts.append(f"{cat} " + "/".join(f"{v:g}" for v in values.values()))
    return ", ".join(parts)


def _legend() -> str:
    scale = load_catalog().get("_scale") or {}
    cats = scale.get("categories") or {}
    described = [f"{name} = {' / '.join(c.get('benchmarks') or [])} ({c.get('use', '')})"
                 for name, c in cats.items() if c.get("benchmarks")]
    unmeasured = [name for name, c in cats.items() if not c.get("benchmarks")]
    text = f"Scores from {scale.get('source', 'the catalog')}: " + "; ".join(described) + "."
    if unmeasured:
        text += f" Not measured comparably yet: {', '.join(unmeasured)}."
    return text


def describe_served_models(served: list[dict]) -> str:
    """The description of a sub-agent's ``model`` argument, for these served models.

    *served* is what the endpoint's /v1/models lists: dicts with ``id`` and, where the
    server reports them, ``root`` and ``max_model_len``. Rendered once per session, so
    it must not depend on anything that changes mid-session (the caller's own model
    included).
    """
    head = "Which model the sub-agent runs on. Leave it empty to use your own model."
    rows = []
    for m in served:
        mid = str(m.get("id") or "")
        if mid:
            rows.append((mid, m, catalog_entry(mid, str(m.get("root") or ""))))
    usable = [r for r in rows if is_delegable(r[2])]
    if len(usable) <= 1:
        return head + " No other model is available on this endpoint."

    categories = list(((load_catalog().get("_scale") or {}).get("categories") or {}).keys())
    costs = sorted({c for c in (decode_cost(e) for _, _, e in usable) if c is not None})
    lines = [
        head + " Pick another when the sub-task leans on a strength yours lacks, or when "
        "a faster model is enough: a broad, simple sweep → a fast model; hard reasoning, "
        "agentic work or delicate code → the strongest in that category.",
        "Speed rank 1 is the fastest, estimated from active parameters × weight size.",
    ]
    for mid, m, entry in usable:
        cost = decode_cost(entry)
        speed = f"speed {costs.index(cost) + 1}/{len(costs)}" if cost is not None else "speed ?"
        ctx = m.get("max_model_len") or entry.get("context_tokens")
        facts = []
        if entry.get("arch"):
            size = _fmt_params(entry["total_params_b"]) if entry.get("total_params_b") else ""
            if entry["arch"] == "moe" and entry.get("active_params_b"):
                size += f" ({_fmt_params(entry['active_params_b'])} active)"
            arch = {"moe": "MoE"}.get(entry["arch"], entry["arch"])
            facts.append(f"{arch} {size}".strip())
        if isinstance(ctx, int):
            facts.append(f"{_fmt_ctx(ctx)} ctx")
        facts.append(speed)
        scores = entry.get("scores")
        detail = _fmt_scores(scores, categories) if isinstance(scores, dict) else "no benchmark data"
        if entry.get("scored_variant"):
            detail += f" (scored {entry['scored_variant']}; a sub-agent runs at its lowest reasoning rung)"
        lines.append(f"- {mid}: {', '.join(facts)} | {detail}")
    refused = [(mid, e) for mid, _, e in rows if not is_delegable(e)]
    if refused:
        lines.append("Not for sub-agents: " + "; ".join(
            f"{mid} ({e.get('note') or 'marked not delegable'})" for mid, e in refused))
    lines.append(_legend())
    return "\n".join(lines)


def subagent_model_refusal(model: str, served: list[dict]) -> str | None:
    """Why *model* cannot run a sub-agent here, or None when it can."""
    ids = [str(m.get("id")) for m in served if m.get("id")]
    if model not in ids:
        usable = [i for i in ids if is_delegable(catalog_entry(i))]
        if not usable:
            return (f"unknown model {model!r}: this endpoint lists no other model; "
                    "leave `model` empty to use your own")
        return f"unknown model {model!r}: the served models are {', '.join(usable)}"
    root = next((str(m.get("root") or "") for m in served if m.get("id") == model), "")
    entry = catalog_entry(model, root)
    if not is_delegable(entry):
        return f"model {model!r} cannot run a sub-agent: {entry.get('note') or 'marked not delegable'}"
    return None
