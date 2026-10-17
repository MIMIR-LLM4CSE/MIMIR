"""A chat message as a provider receives it, and the invariants a history must keep
for every provider to accept it: each tool call answered under its id, user turns
alternating with the assistant's."""
from __future__ import annotations

import json

from .tool_result import json_error_payload


# The keys a message actually travels with. Serialising the dict wholesale would
# also count local bookkeeping a backend never sees, so the wire form is spelled
# out rather than inferred — an extra key added here later must be one the
# provider receives.
_WIRE_KEYS = ("role", "content", "name", "tool_call_id", "tool_calls", "thinking",
              "reasoning")


def message_wire_form(message: dict) -> str:
    """*message* as the provider is sent it: the JSON, envelope included.

    Content alone is not the message: in full-context mode history keeps the
    structured transcript, where an assistant turn carries its ``tool_calls`` and
    usually has empty content. Counting content only scored those turns at ~0 —
    including calls whose arguments hold a whole file — so the context bar
    under-reported and the pre-query trim under-trimmed.

    Serialising content and arguments as bare *text* closed that gap and left a
    second one of the same kind. What a window is measured against is
    ``json.dumps(messages)`` — see the guard in ``VLLMBackend.chat`` — so every
    ``\n`` that serialisation escapes, every quote it doubles and every ``role`` /
    ``tool_call_id`` key it adds is prompt that a text-shaped count never saw. On
    an ordinary tool-heavy history that envelope is ~18% of the total; on
    escape-dense content — HTML, LaTeX, JSON inside JSON, the shapes a web fetch
    returns — it reaches 30%. Measured in the wrong units, a budget cannot be
    conservative by accident: it is optimistic by construction, and it was an
    optimism of exactly this size that let a backstop announce a fit on a prompt
    the provider then rejected.

    Falls back to the joined-text approximation on anything unserialisable, which
    can only under-count.
    """
    # Keyed on presence, not on truth: a message that carries ``content: None``
    # is serialised with ``"content": null`` and paid for, so dropping the falsy
    # ones would reintroduce the same under-count one level down — ~16 characters
    # per assistant turn, and an assistant turn is every other message.
    payload = {k: message[k] for k in _WIRE_KEYS if k in message}
    if not payload:
        return ""
    try:
        return json.dumps(payload, default=str)
    except Exception:
        parts = [str(message.get("content") or "")]
        tool_calls = message.get("tool_calls")
        if tool_calls:
            parts.append(str(tool_calls))
        thinking = message.get("thinking")
        if thinking:
            parts.append(str(thinking))
        return "\n".join(p for p in parts if p)


# Placeholder result emitted for an assistant tool call whose real result is absent —
# keeps the assistant↔tool pairing valid and tells the model the output is gone.
EVICTED_TOOL_RESULT = json_error_payload(
    "tool result not in history (evicted to stay within the context budget)",
)


def _declared_call_ids(assistant: dict) -> list[str | None]:
    """Ids of an assistant turn's tool calls; ``None`` where the model emitted none.

    Ollama-style calls carry no id at all, so the caller must fall back to positional
    matching rather than assume a key that isn't there.
    """
    out: list[str | None] = []
    for tc in assistant.get("tool_calls") or []:
        cid = tc.get("id") if isinstance(tc, dict) else None
        out.append(cid if isinstance(cid, str) and cid else None)
    return out


def reconcile_tool_pairs(messages: list[dict]) -> list[dict]:
    """Repair assistant.tool_calls ↔ tool-message pairing.

    Two upstream mechanisms legitimately break the pairing: ``_trim_tool_history``
    evicts a tool result while preserving its assistant turn, and
    ``_maybe_compact_intra_query`` can slice between an assistant turn and its results.
    (The dispatcher answers every call it declines to run under that call's own id, so
    it leaves no gaps of its own.)

    The invariant belongs to the history, not to one provider: strict backends reject a
    dangling pair outright (the Claude API 400s on a ``tool_use`` with no ``tool_result``)
    and lenient ones render an incoherent prompt. For each assistant turn declaring
    calls, the following run of tool messages is matched by ``tool_call_id`` first, then
    positionally for any lacking a usable id; each declared call is emitted with its
    response or a stub. Tool messages belonging to no preceding call are dropped.
    """
    out: list[dict] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            declared = _declared_call_ids(m)

            # Contiguous run of tool messages answering this assistant turn.
            j = i + 1
            run: list[dict] = []
            while j < n and messages[j].get("role") == "tool":
                run.append(messages[j])
                j += 1

            responses: list[dict | None] = [None] * len(declared)
            leftovers: list[dict] = []
            for tmsg in run:
                tcid = tmsg.get("tool_call_id")
                slot = (
                    declared.index(tcid)
                    if isinstance(tcid, str) and tcid in declared
                    else -1
                )
                if slot >= 0 and responses[slot] is None:
                    responses[slot] = tmsg
                else:
                    leftovers.append(tmsg)
            for tmsg in leftovers:
                slot = next((k for k, r in enumerate(responses) if r is None), None)
                if slot is None:
                    break  # all calls answered — remaining tool messages are orphans
                fixed = dict(tmsg)
                if declared[slot] is not None:
                    fixed["tool_call_id"] = declared[slot]
                responses[slot] = fixed

            out.append(m)
            for cid, resp in zip(declared, responses):
                if resp is not None:
                    out.append(resp)
                    continue
                stub: dict = {"role": "tool", "content": EVICTED_TOOL_RESULT}
                if cid is not None:
                    stub["tool_call_id"] = cid
                out.append(stub)
            i = j
        elif m.get("role") == "tool":
            # Orphan tool message (no preceding assistant tool_calls run) — drop it.
            i += 1
        else:
            out.append(m)
            i += 1
    return out


def merge_consecutive_user_messages(prepared: list[dict]) -> list[dict]:
    """Collapse adjacent plain ``user`` messages into one.

    Lives here, beside :func:`reconcile_tool_pairs`, because every backend needs it
    (``LLMBackend.chat`` applies both): the Anthropic path requires strict alternation.

    A strict tokenizer (vLLM's ``--tokenizer-mode mistral`` is one) enforces
    strict role alternation and degenerates into token salad when it sees two
    consecutive ``user`` turns. Upstream can legitimately produce them (plan-mode
    nudges, a caller that already appended the current turn). Joining their text
    with a blank line preserves the content while keeping the sequence legal; only
    string-content user messages with no tool fields are merged, so tool pairing is
    untouched.
    """
    out: list[dict] = []
    for m in prepared:
        if (
            out
            and m.get("role") == "user"
            and out[-1].get("role") == "user"
            and isinstance(m.get("content"), str)
            and isinstance(out[-1].get("content"), str)
            and not m.get("tool_calls")
            and not out[-1].get("tool_calls")
        ):
            if m["content"] != out[-1]["content"]:
                out[-1] = {**out[-1], "content": out[-1]["content"] + "\n\n" + m["content"]}
            # identical duplicate → drop entirely
            continue
        out.append(m)
    return out
