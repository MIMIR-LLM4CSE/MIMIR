"""One-sentence session descriptions for the chat-history panel.

The panel used to show the first user query verbatim, which says what was
*asked* but not what the session actually did. This module asks the backend for
a short descriptive sentence built from the session transcript. When the model is
unreachable or unhelpful the first user query stands in, stored as *provisional*
(``PROVISIONAL_VERSION``) so the panel has something to show while the next turn
still retries the real generation — a failed call never freezes a non-description
into the session.

The call is blocking (backend ``chat``) — callers run it off the event loop.
"""

from __future__ import annotations

import logging

from ._ws_runtime import get_backend

logger = logging.getLogger(__name__)

# How much of each message to feed the summarizer, and how many messages.
_MSG_CHARS = 600
_MAX_MSGS = 12
# Bump when the prompt or the accepted output changes: stored summaries carrying
# an older version are regenerated on the next turn instead of sticking around.
SUMMARY_VERSION = 2
# Stamped on the query fallback (and matching the pre-summary default), so a
# provisional description is shown but always regenerated.
PROVISIONAL_VERSION = 0
# Generous answer budget: thinking-capable models (nemotron, qwen3, …) may still
# emit a reasoning block even with thinking disabled, and a tight budget lets it
# swallow the whole completion, leaving empty content and a fallback title.
_MAX_TOKENS = 400

_SYSTEM = (
    "You write one-sentence descriptions of coding-assistant sessions. "
    "Given a transcript, reply with a single sentence (max 15 words) describing "
    "what the session is about overall, naming the concrete files, features or "
    "problems involved. Say what was done, not just what was asked, and cover the "
    "whole transcript rather than only its last turn. No preamble, no quotes, "
    "no markdown, no trailing explanation. Write in the language of the transcript."
)


def _transcript(display_messages: list[dict]) -> str:
    """Flatten display messages into a compact user/assistant transcript."""
    lines: list[str] = []
    for msg in display_messages:
        if msg.get("kind") != "text":
            continue
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        role = msg.get("role")
        if role == "user":
            label = "User"
        elif role == "agent":
            label = "Assistant"
        else:
            continue
        lines.append(f"{label}: {text[:_MSG_CHARS]}")
    # Keep the first two turns (what was asked) and the tail (where it ended up).
    if len(lines) > _MAX_MSGS:
        lines = lines[:2] + lines[-(_MAX_MSGS - 2):]
    return "\n".join(lines)


# Lead-ins models like to prepend despite being told not to.
_PREFIXES = ("description:", "summary:", "one-sentence description:", "sentence:")


def _clean(text: str) -> str:
    """Strip quoting/markdown noise and keep a single sentence."""
    # Take the last non-empty line: a model that opens with "Sure, here it is:"
    # puts the actual sentence last, while a bare answer is unaffected.
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if not lines:
        return ""
    out = lines[-1]
    # Quotes and list/heading markers nest either way round ("- text." as well
    # as - "text."), so peel until nothing more comes off.
    for _ in range(4):
        peeled = out.strip('"').strip("'").lstrip("-*# ").strip()
        low = peeled.lower()
        for prefix in _PREFIXES:
            if low.startswith(prefix):
                peeled = peeled[len(prefix):].strip()
                break
        if peeled == out:
            break
        out = peeled
    if len(out) > 160:  # clip on a word boundary rather than mid-word
        out = out[:160].rsplit(" ", 1)[0] + "…"
    return out.strip()


def fallback_summary(display_messages: list[dict]) -> str:
    """The first user query, trimmed — the stand-in when generation fails."""
    for msg in display_messages:
        if msg.get("role") == "user":
            raw = (msg.get("text") or msg.get("content") or "").strip()
            if raw:
                return _clean(raw.replace("\n", " "))
    return ""


def generate_summary(model: str, display_messages: list[dict]) -> tuple[str, bool]:
    """Return ``(description, generated)`` for the session (blocking).

    ``generated`` is False when the model was unreachable or answered with
    nothing usable and the first user query stands in — the caller stores such a
    description as provisional so the next query tries the real thing again.
    """
    transcript = _transcript(display_messages)
    if not transcript:
        return fallback_summary(display_messages), False
    try:
        reply = get_backend().chat(
            model,
            [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": f"Transcript:\n{transcript}\n\nOne-sentence description:",
                },
            ],
            [],                 # no tools
            False,              # thinking off — this is a formatting task
            False,              # no streaming
            {"temperature": 0.2, "num_predict": _MAX_TOKENS, "max_tokens": _MAX_TOKENS},
        )
    except Exception as exc:
        logger.warning("Session summary generation failed: %s", exc)
        return fallback_summary(display_messages), False

    summary = _clean(str(reply.get("content") or ""))
    if not summary:
        # Thinking models sometimes answer entirely inside the reasoning block
        # (or run out of budget there). Salvage its last line before giving up.
        summary = _clean(str(reply.get("thinking") or ""))
    if not summary:
        logger.warning("Session summary generation returned no usable text")
        return fallback_summary(display_messages), False
    return summary, True
