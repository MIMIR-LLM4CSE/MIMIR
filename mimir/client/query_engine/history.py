"""Context-window budgeting: keep the message list inside the model window.

Trims oldest tool output, compacts the middle of the conversation, and — as a
deterministic backstop — force-fits to a token target. ``_enforce_context_budget``
orchestrates trim → compact → force-fit before each model call.
``served_compaction_instruction`` is the handoff note used when summarizing.
Extracted from ``agent_loop.py``.
"""
from __future__ import annotations

import json
from typing import Any

from ..event_sink import emit
from ..config.constants import (
    TOOL_HISTORY_CHAR_BUDGET as _TOOL_HISTORY_CHAR_BUDGET,
    TOOL_HISTORY_TOKEN_BUDGET as _TOOL_HISTORY_TOKEN_BUDGET,
    INTRA_QUERY_COMPACT_TOKENS as _INTRA_QUERY_COMPACT_TOKENS,
    INTRA_QUERY_COMPACT_CHARS as _INTRA_QUERY_COMPACT_CHARS,
    context_budget_for,
)


_TOOL_OUTPUT_MAX_CHARS = 4000
_TOOL_OUTPUT_MAX_LINES = 60

# Placeholder result emitted for an assistant tool call whose real result is absent —
# keeps the assistant↔tool pairing valid and tells the model the output is gone.
EVICTED_TOOL_RESULT = json.dumps({
    "status": "error",
    "error": "tool result not in history (evicted to stay within the context budget)",
})


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

    Three upstream mechanisms legitimately break the pairing: ``_trim_tool_history``
    evicts a tool result while preserving its assistant turn, ``_maybe_compact_intra_query``
    can slice between an assistant turn and its results, and the dispatcher drops
    duplicate/already-dispatched calls without emitting a tool message for them.

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


def served_compaction_instruction() -> str:
    """The handoff-note prompt used when summarizing older conversation turns.

    Shared by the loop's intra-query compaction (via ``MimirAgent.compact_messages``)
    so the summarization behaviour is defined in exactly one place.
    """
    return (
        "The above is a conversation history with a coding agent. "
        "Produce a concise HANDOFF NOTE that a new session of the same agent "
        "can read to continue work without re-discovering what was already done. "
        "Cover:\n"
        "1. Task(s) requested — one sentence each\n"
        "2. Repository structure discovered — directories, key files, and their purpose\n"
        "3. Files created or modified — path + one-sentence description\n"
        "4. Key decisions and their rationale\n"
        "5. What was validated and the result\n"
        "6. What is still pending or incomplete\n\n"
        "Rules:\n"
        "- Be specific: always use full file paths, class names, function names\n"
        "- Prefer structure over prose — use short lists\n"
        "- Keep it under 800 words\n"
        "- Do NOT repeat what can be inferred from file names alone"
    )


def _truncate_output(text: Any) -> str:
    """Clip a tool result to a UI-friendly preview (line- and char-bounded)."""
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    lines = s.splitlines()
    clipped = False
    if len(lines) > _TOOL_OUTPUT_MAX_LINES:
        lines = lines[:_TOOL_OUTPUT_MAX_LINES]
        clipped = True
    s = "\n".join(lines)
    if len(s) > _TOOL_OUTPUT_MAX_CHARS:
        s = s[:_TOOL_OUTPUT_MAX_CHARS]
        clipped = True
    if clipped:
        s = s.rstrip() + "\n… (truncated)"
    return s


def _trim_tool_history(
    messages: list[dict],
    char_budget: int = _TOOL_HISTORY_CHAR_BUDGET,
    execution_context: dict | None = None,
    token_counter: Any | None = None,
    token_budget: int = _TOOL_HISTORY_TOKEN_BUDGET,
) -> None:
    """Evict oldest tool-result messages when total tool content exceeds the budget.

    Only removes ``{"role": "tool"}`` entries; system, user, and assistant
    messages are always preserved so history coherence is maintained.

    Size is measured in tokens via *token_counter* (``text -> int``) against
    *token_budget* when a counter is supplied; otherwise it falls back to raw
    characters against *char_budget*. The eviction-selection logic (file
    protection, read invalidation) is identical in both modes — only the size
    metric changes.

    Tool messages whose content references a file currently being written
    (dirty_written_files or declared_edit_set) are protected from eviction.
    When a file has been read but then evicted from history, its path is also
    removed from execution_context["read_files"] so the policy cannot falsely
    allow an edit without a fresh re-read.
    """
    _size = token_counter if token_counter is not None else len
    budget = token_budget if token_counter is not None else char_budget
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    total_size = sum(_size(messages[i].get("content", "")) for i in tool_indices)
    if total_size <= budget:
        return

    # Build set of protected file paths — files being actively written/planned.
    protected_paths: set[str] = set()
    if execution_context:
        protected_paths |= execution_context.get("dirty_written_files", set())
        protected_paths |= execution_context.get("declared_edit_set", set())
        # Also protect reads for files being actively repaired: when dirty files
        # exist, the model may need earlier read_file results to produce correct
        # repair anchors — keep them from being evicted.
        if execution_context.get("dirty_written_files"):
            protected_paths |= execution_context.get("read_files", set())
    protected_paths = {p for p in protected_paths if p}  # drop empty strings

    # Authoritative per-message file association, recorded at dispatch time (see
    # _dispatch_tool_calls). Structural matching avoids the substring hazard: a grep
    # result that merely *mentions* a path neither protects it from eviction nor
    # invalidates that file's actual read.
    tool_msg_files: dict = (
        execution_context.get("tool_msg_files", {}) if execution_context else {}
    )

    def _structural_paths(m: dict) -> tuple[list[str], bool]:
        """Return (paths, is_structural) for a tool message.

        When the message's tool_call_id was recorded at dispatch we know exactly
        which files it concerns (possibly none) — authoritative. Messages with no
        record (e.g. history carried over from a prior query) return
        is_structural=False so the caller falls back to the legacy substring scan.
        """
        cid = m.get("tool_call_id")
        if cid is not None and cid in tool_msg_files:
            return tool_msg_files[cid], True
        return [], False

    to_remove: list[int] = []
    removed_size = 0
    for idx in tool_indices:
        if total_size - removed_size <= budget:
            break
        msg = messages[idx]
        content = msg.get("content", "")
        paths, structural = _structural_paths(msg)

        # Keep messages whose referenced file is actively being edited.
        if protected_paths:
            if structural:
                if any(p in protected_paths for p in paths):
                    continue
            elif any(p in content for p in protected_paths):  # legacy fallback
                continue

        to_remove.append(idx)
        removed_size += _size(content)
        # Keep execution_context in sync: evicting a file read invalidates its
        # read_files entry, so policy demands a fresh re-read before the next edit.
        # The substring fallback covers untracked (carried-over) messages.
        if execution_context is not None:
            read_files: set[str] = execution_context.get("read_files", set())
            if structural:
                for path in paths:
                    read_files.discard(path)
                tool_msg_files.pop(msg.get("tool_call_id"), None)
            else:
                for path in list(read_files):
                    if path and path in content:
                        read_files.discard(path)

    for i in sorted(to_remove, reverse=True):
        del messages[i]


def _maybe_compact_intra_query(
    messages: list[dict],
    system_content: str,
    execution_context: dict,
    compact_fn: Any | None,
    token_counter: Any | None = None,
    token_budget: int | None = None,
) -> None:
    """Compact the middle of the message list when total size exceeds the budget.

    Keeps: system message, first user message, last 4 messages (2 exchanges).
    Replaces everything in between with a single assistant summary produced by
    *compact_fn* (same signature as chat_session.compact_history).
    The discovery pin is a transient tail message (see _inject_pin), not part of
    messages[0], so compaction never disturbs it.

    Total size is measured in tokens via *token_counter* against the token
    budget when supplied, else in characters against the char budget.
    """
    if compact_fn is None:
        return
    _size = token_counter if token_counter is not None else len
    # token_budget=None means "use the module default" — resolved here (not as a
    # default arg) so tests patching _INTRA_QUERY_COMPACT_TOKENS still take effect.
    _token_budget = token_budget if token_budget is not None else _INTRA_QUERY_COMPACT_TOKENS
    budget = _token_budget if token_counter is not None else _INTRA_QUERY_COMPACT_CHARS
    total_size = sum(_size(m.get("content", "")) for m in messages)
    if total_size <= budget:
        return

    # Need at least: [system, user, ..., last4] — compact only if there's a
    # meaningful middle section (at least 3 messages between head and tail).
    if len(messages) < 8:
        return

    middle = messages[2:-4]  # messages[0]=system, messages[1]=first user msg
    if not middle:
        return

    emit({"type": "status", "text": "⚡ Intra-query compaction triggered — summarising intermediate steps..."})
    try:
        summary_messages = compact_fn(middle)
    except Exception:
        return  # compaction failed — silently continue without it

    messages[2:-4] = summary_messages
    # The discovery pin is a transient tail message (see _inject_pin), not part of
    # messages[0], so compaction of the middle never disturbs it — nothing to refresh.


def _truncate_text_to_tokens(text: str, max_tokens: int, token_counter: Any) -> str:
    """Shrink *text* to at most *max_tokens*, keeping a head and tail with a marker.

    Used as a last resort by _force_fit_to_window when whole-message eviction
    isn't enough (e.g. a single huge tool result or pasted blob). Keeps the start
    and end — usually the most informative parts — and drops the middle. The
    result is GUARANTEED to be <= max_tokens (verified, then shrunk if the
    marker/rounding overshot), so the caller's fitting loop always converges.
    """
    if max_tokens <= 0 or not text:
        return ""
    if token_counter(text) <= max_tokens:
        return text
    marker = "\n…[truncated]…\n"
    cpt = max(1, len(text) // max(1, token_counter(text)))
    budget_chars = max(1, max_tokens * cpt)
    # Shrink until the rendered result (head + marker + tail) is within budget.
    for _ in range(24):  # bounded; halving converges well before this
        if budget_chars >= len(text):
            return text
        head = (budget_chars * 2) // 3
        tail = budget_chars - head
        candidate = text[:head] + marker + (text[-tail:] if tail else "")
        if token_counter(candidate) <= max_tokens:
            return candidate
        budget_chars //= 2
    # Fallback: marker alone (or empty if even that is too big).
    return marker if token_counter(marker) <= max_tokens else ""


def _message_content_str(m: dict) -> str:
    """Return a message's ``content`` as a string (JSON-encode non-str payloads)."""
    c = m.get("content")
    if c is None:
        return ""
    return c if isinstance(c, str) else json.dumps(c)


def _force_fit_to_window(
    messages: list[dict],
    target_tokens: int,
    token_counter: Any,
) -> bool:
    """Guarantee the message list fits *target_tokens*, truncating as a last resort.

    The trim/compact helpers above only handle whole tool messages (and need a
    compaction callback). This is the deterministic backstop that runs on every
    call: it shrinks the largest reducible messages — oldest first among equals —
    until the total content fits. The system message (index 0) and the most recent
    user message (the active query) are never reduced, so the model always sees
    its instructions and the question. Returns True when the list fits afterwards,
    False when the irreducible core alone still exceeds the target (the caller /
    backend then surfaces a clear context-overflow error).
    """
    if target_tokens < 1:
        return False

    _content_str = _message_content_str

    def _mtok(m: dict) -> int:
        t = token_counter(_content_str(m))
        for tc in m.get("tool_calls") or []:
            args = (tc.get("function") or {}).get("arguments", "")
            if args:
                t += token_counter(args if isinstance(args, str) else json.dumps(args))
        return t

    cur = sum(_mtok(m) for m in messages)
    if cur <= target_tokens:
        return True

    last_user = max(
        (i for i, m in enumerate(messages) if m.get("role") == "user"),
        default=-1,
    )
    protected = {0, last_user}
    # Largest first; ties broken by oldest (lower index) so recent context survives.
    order = sorted(
        (i for i in range(len(messages)) if i not in protected),
        key=lambda i: (_mtok(messages[i]), -i),
        reverse=True,
    )
    for i in order:
        if cur <= target_tokens:
            break
        m = messages[i]
        content = _content_str(m)
        if not content:
            continue
        before = token_counter(content)
        need = cur - target_tokens
        keep = max(0, before - need)
        new = "…[truncated]…" if keep == 0 else _truncate_text_to_tokens(
            content, keep, token_counter
        )
        m["content"] = new
        cur -= before - token_counter(new)

    return cur <= target_tokens


def _enforce_context_budget(
    messages: list[dict],
    system_content: str,
    step_tools: list[dict] | None,
    execution_context: dict,
    model: str,
    context_mode: str,
    compact_fn: Any | None,
    token_counter: Any,
) -> None:
    """Trim/compact history so the *next* LLM prompt fits the model's window.

    Sizes the trim/compaction budgets to the model's real context window
    (vLLM ``max_model_len`` when available) and — crucially — subtracts the
    tools-schema token cost. The tools schema is sent on every call but is NOT
    part of ``messages``, so without this the message history alone can fill the
    window and the actual prompt (messages + tools) overflows, which vLLM rejects
    with the confusing ``max_tokens must be at least 1, got -N`` 400. On large
    windows the overhead is negligible; on a small window (e.g. a served 16K
    model) it is the difference between fitting and overflowing.

    Order: (1) evict oldest tool results, (2) compact the middle when a
    compaction callback is available, then (3) a deterministic hard-fit pass that
    truncates oversized content as a last resort. Step 3 is what actually
    *guarantees* the prompt fits regardless of message types or whether
    compaction is wired up. (4) repairs the assistant↔tool pairing that steps 1
    and 2 legitimately break, so every backend receives a coherent history.
    """
    total, reserved, trim_budget, compact_budget = context_budget_for(model, context_mode)
    overhead = token_counter(json.dumps(step_tools)) if step_tools else 0
    trim_budget = max(512, trim_budget - overhead)
    compact_budget = max(512, compact_budget - overhead)
    _trim_tool_history(messages, execution_context=execution_context,
                       token_counter=token_counter, token_budget=trim_budget)
    _maybe_compact_intra_query(messages, system_content, execution_context, compact_fn,
                               token_counter=token_counter, token_budget=compact_budget)
    # Hard backstop, reached only when eviction + summarization still didn't fit: keeps
    # a margin for chat-template scaffolding and a minimal answer allocation the
    # per-message estimate omits. Destructive, so the user is notified when it drops
    # content — that loss is otherwise silent.
    if total:
        usable = max(1, total - reserved - overhead)
        before = sum(token_counter(_message_content_str(m)) for m in messages)
        _force_fit_to_window(messages, usable, token_counter)
        after = sum(token_counter(_message_content_str(m)) for m in messages)
        if after < before:
            emit({"type": "status", "text": (
                f"  ⚠ Context backstop: truncated ~{before - after} tokens of older "
                f"content to fit the model's window."
            )})
    # Last: eviction and compaction above can strand an assistant tool call or a tool
    # result. Repair in place so the next model call is coherent whatever the backend.
    messages[:] = reconcile_tool_pairs(messages)
