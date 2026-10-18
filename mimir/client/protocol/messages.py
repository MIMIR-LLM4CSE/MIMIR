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
    """*message* as the provider receives it: its JSON, envelope included.

    A window is measured against ``json.dumps(messages)``; counting content alone misses
    the tool calls and the escaping, up to a third of a tool-heavy history.
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
    """Repair the pairing of assistant ``tool_calls`` and tool messages.

    Trimming and compaction can break it, and strict backends reject a dangling call.
    Results match by id, then by position; a call with none gets a stub; orphans are dropped.
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
                free = next((k for k, r in enumerate(responses) if r is None), None)
                if free is None:
                    break  # all calls answered — remaining tool messages are orphans
                fixed = dict(tmsg)
                if declared[free] is not None:
                    fixed["tool_call_id"] = declared[free]
                responses[free] = fixed

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
    """Join adjacent plain ``user`` messages with a blank line.

    Strict tokenizers and the Anthropic API require alternating roles; tool messages are
    left alone.
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
