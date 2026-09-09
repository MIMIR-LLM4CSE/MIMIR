"""Context-window budgeting: keep the message list inside the model window.

Trims oldest tool output, compacts the middle of the conversation, and — as a
deterministic backstop — force-fits to a token target. ``_enforce_context_budget``
orchestrates trim → compact → force-fit before each model call.
``served_compaction_instruction`` is the handoff note used when summarizing.
Extracted from ``agent_loop.py``.
"""
from __future__ import annotations

import hashlib
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


class ContextOverflowError(RuntimeError):
    """The prompt cannot be made to fit the model's window.

    Raised by :func:`_enforce_context_budget` when eviction, compaction and the
    force-fit backstop have all run and the irreducible core — the system message
    plus the current query, neither of which may be reduced — still exceeds the
    usable window. Without it the oversized prompt went to the backend anyway and
    came back as an opaque provider 400 (vLLM: ``max_tokens must be at least 1,
    got -N``), after a status line claiming the history had been trimmed to fit.
    """


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

    The tool results answering the most recent assistant tool-call turn are never
    evicted: they are the current turn's payload, and stubbing them out costs more
    than the space it frees. Only the force-fit backstop may shrink them.

    Tool messages whose content references a file currently being written
    (dirty_written_files or declared_edit_set) are protected from eviction.
    When a file has been read but then evicted from history, its path is also
    removed from execution_context["read_files"] so the policy cannot falsely
    allow an edit without a fresh re-read.
    """
    _size = token_counter if token_counter is not None else len
    budget = token_budget if token_counter is not None else char_budget
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    total_size = sum(_message_tokens(messages[i], _size) for i in tool_indices)
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

    # The newest results — those answering the last assistant turn that made calls —
    # are this turn's payload: a sub-agent's entire answer comes back as one of them.
    # Evicting oldest-first hit them anyway whenever they were the only tool messages
    # in the window, replacing the freshest evidence with EVICTED_TOOL_RESULT while
    # nothing older remained to drop. They are exempt here; the force-fit pass can
    # still shrink them, and its truncation keeps a head and a tail rather than
    # nothing at all.
    last_call_turn = _last_call_turn(messages)
    newest_results = {i for i in tool_indices if i > last_call_turn} if last_call_turn >= 0 else set()

    to_remove: list[int] = []
    removed_size = 0
    for idx in tool_indices:
        if idx in newest_results:
            continue
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
    The task checklist lives in messages[0], which is kept, so compaction never
    disturbs it.

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
    total_size = sum(_message_tokens(m, _size) for m in messages)
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
    # The task checklist lives in messages[0], which this slice does not touch, so
    # compaction of the middle never disturbs it — nothing to refresh.


def merge_consecutive_user_messages(prepared: list[dict]) -> list[dict]:
    """Collapse adjacent plain ``user`` messages into one.

    Lives here, beside :func:`reconcile_tool_pairs`, because every backend needs it
    and only one used to have it: it was defined inside the vLLM backend, so the same
    history produced a legal prompt there and two adjacent ``user`` entries on the
    Anthropic path — which requires strict alternation — and no normalization at all
    on Ollama.

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


def _message_tokens(m: dict, token_counter: Any) -> int:
    """Tokens for one message: its ``content`` PLUS its tool calls' arguments.

    The single size measure every budgeting pass uses. Counting only ``content``
    is what let a history whose weight lived in ``tool_calls[].function.arguments``
    — file bodies handed to write_file — read as barely half its real size: the
    trim and compaction triggers never fired, while the prompt the provider
    actually received (it serializes the whole message, arguments included) was
    nearly twice what they measured.

    Content and arguments are joined into ONE string and counted in a single call:
    ``count_text_tokens`` may make a blocking /tokenize round-trip (cached per
    model/length/hash), so one call per message beats one per argument.
    """
    parts = [_message_content_str(m)]
    for tc in m.get("tool_calls") or []:
        args = (tc.get("function") or {}).get("arguments", "")
        if args:
            parts.append(args if isinstance(args, str) else json.dumps(args))
    text = "\n".join(p for p in parts if p)
    return token_counter(text) if text else 0


def _last_call_turn(messages: list[dict]) -> int:
    """Index of the last assistant turn that made tool calls, or -1."""
    return max(
        (i for i, m in enumerate(messages)
         if m.get("role") == "assistant" and m.get("tool_calls")),
        default=-1,
    )


_ARG_DIGEST_MAX_CHARS = 400


def _elision_note(text: str) -> str:
    """A short, factual stand-in for an elided argument value."""
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]
    return (f"\u2026[elided: {len(text.splitlines())} lines, {len(text)} chars, "
            f"sha256:{digest}]")


def _digest_call_args(args: Any, max_chars: int = _ARG_DIGEST_MAX_CHARS) -> Any:
    """Replace a tool call's bulky string arguments with a short digest.

    Returns a NEW dict, never mutating *args*, so digesting cannot reach through
    into the untrimmed record that shares these message objects. Returns *args*
    itself — identity-comparable by the caller — when there was nothing to elide.

    The result stays a dict: ``anthropic_backend._prepare`` substitutes ``{}`` for
    arguments that are not one, and the vLLM path re-encodes dicts to a string on
    the way out, so a dict is the shape both expect internally. Small scalar
    arguments (path, overwrite, …) are kept verbatim so the call still reads as
    what it was; only long string values are replaced.
    """
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except (ValueError, TypeError):
            return {"_elided": _elision_note(args)} if len(args) > max_chars else args
        args = parsed
    if not isinstance(args, dict):
        return args
    out: dict = {}
    elided = False
    for k, v in args.items():
        if isinstance(v, str) and len(v) > max_chars:
            out[k] = _elision_note(v)
            elided = True
        else:
            out[k] = v
    return out if elided else args


def _digest_tool_call_args(m: dict) -> bool:
    """Digest every bulky argument of *m*'s tool calls. True when anything shrank.

    Rebuilds the ``tool_calls`` list out of new dicts rather than writing into the
    existing ones: the nested call/function dicts are shared with the untrimmed
    record, which a nested in-place write would silently rewrite too.
    """
    rebuilt: list = []
    elided = False
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        old = fn.get("arguments")
        new = _digest_call_args(old)
        if new is not old:
            rebuilt.append({**tc, "function": {**fn, "arguments": new}})
            elided = True
        else:
            rebuilt.append(tc)
    if elided:
        m["tool_calls"] = rebuilt
    return elided


def _tool_result_failed(m: dict) -> bool:
    """True when a tool message carries the dispatcher's ``status: error`` payload."""
    content = m.get("content")
    if not isinstance(content, str) or "error" not in content:
        return False
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "error"


def _digest_failed_call_args(messages: list[dict]) -> int:
    """Digest the arguments of tool calls whose result came back an error.

    Lossless by construction: the call did not take effect — the write was
    refused, the path already existed, the content did not parse — so its
    arguments describe something that never happened, while the error message
    saying why stays in history verbatim. One rejected ``write_file`` of a
    250-line file costs ~2.7k tokens, and a single session carried three of them
    at full weight from the moment they failed to the end of the run.

    The most recent tool-call turn is exempt: a model repairing the syntax error
    it was just told about needs to see what it wrote. Only older failures go.

    Returns the number of calls digested.
    """
    results = {m.get("tool_call_id"): m
               for m in messages if m.get("role") == "tool" and m.get("tool_call_id")}
    if not results:
        return 0
    newest = _last_call_turn(messages)
    digested = 0
    for i, m in enumerate(messages):
        if i >= newest or not m.get("tool_calls"):
            continue
        rebuilt: list = []
        elided = False
        for tc in m["tool_calls"]:
            fn = tc.get("function") or {}
            res = results.get(tc.get("id"))
            old = fn.get("arguments")
            new = _digest_call_args(old) if (res is not None and _tool_result_failed(res)) else old
            if new is not old:
                rebuilt.append({**tc, "function": {**fn, "arguments": new}})
                elided = True
                digested += 1
            else:
                rebuilt.append(tc)
        if elided:
            m["tool_calls"] = rebuilt
    return digested


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
        return _message_tokens(m, token_counter)

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
        before_msg = _mtok(m)
        before = token_counter(content)
        need = cur - target_tokens
        keep = max(0, before - need)
        new = "…[truncated]…" if keep == 0 else _truncate_text_to_tokens(
            content, keep, token_counter
        )
        m["content"] = new
        # Per-message deltas rather than content-only ones: `cur` is measured with
        # the unified metric, so subtracting a content-only difference would drift
        # from it and could report a fit as a failure.
        cur -= before_msg - _mtok(m)

    # Second pass. A message whose weight is entirely in its tool-call arguments
    # has no reducible content, so the loop above skipped it outright — the
    # backstop counted those bytes and had no grip on them, which is how a prompt
    # could stay stuck over the window with nothing left that this pass would
    # touch. Digesting the arguments is the grip. The newest tool-call turn is
    # exempt, as in _digest_failed_call_args: it is the turn being answered.
    if cur > target_tokens:
        newest = _last_call_turn(messages)
        for i in order:
            if cur <= target_tokens:
                break
            if i >= newest:
                continue
            m = messages[i]
            if not m.get("tool_calls"):
                continue
            before_msg = _mtok(m)
            if _digest_tool_call_args(m):
                cur -= before_msg - _mtok(m)

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

    Raises :class:`ContextOverflowError` when step 3 cannot make the prompt fit —
    the irreducible core (system message + current query) alone exceeds the usable
    window. The repair in step 4 still runs first, so the history left behind is
    coherent for whatever the caller does with it.
    """
    overflow: ContextOverflowError | None = None
    # First, and regardless of any budget: the arguments of calls that came back an
    # error describe work that never happened. Dropping them is lossless, so it is
    # not something to do only under pressure.
    digested = _digest_failed_call_args(messages)
    if digested:
        emit({"type": "status", "text": (
            f"  \u2702 Context: elided the arguments of {digested} failed tool "
            f"call{'s' if digested != 1 else ''} (the calls did not take effect)."
        )})
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
        before = sum(_message_tokens(m, token_counter) for m in messages)
        fitted = _force_fit_to_window(messages, usable, token_counter)
        after = sum(_message_tokens(m, token_counter) for m in messages)
        if after < before:
            emit({"type": "status", "text": (
                f"  ⚠ Context backstop: truncated ~{before - after} tokens of older "
                + ("content to fit the model's window."
                   if fitted else "content — the prompt is STILL over the window.")
            )})
        if not fitted:
            # Everything reducible has been reduced and the prompt still does not
            # fit. Sending it anyway is what produced the provider-level 400 the
            # user saw; failing here names the actual cause and the numbers behind
            # it. The turn ends either way — this only decides what it says.
            # Raised only after the repair below, so the history the session keeps
            # (and reloads on the next turn) is still a coherent one.
            overflow = ContextOverflowError(
                f"Context overflow: the prompt is ~{after + overhead:,} tokens but only "
                f"~{usable:,} fit in this model's {total:,}-token window "
                f"(answer reserve {reserved:,}, tools schema {overhead:,}). "
                "The system message and the current query cannot be reduced further — "
                "shorten the query, start a new session, or use a model with a larger window."
            )
    # Last: eviction and compaction above can strand an assistant tool call or a tool
    # result. Repair in place so the next model call is coherent whatever the backend.
    messages[:] = reconcile_tool_pairs(messages)
    if overflow is not None:
        raise overflow
