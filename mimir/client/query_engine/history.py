"""Context-window budgeting: keep the message list inside the model window.

Trims oldest tool output, compacts the middle of the conversation, and — as a
deterministic backstop — force-fits to a token target. ``enforce_context_budget``
orchestrates trim → compact → force-fit before each model call.
``served_compaction_instruction`` is the handoff note used when summarizing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from ..config.limits import (
    INTRA_QUERY_COMPACT_CHARS as _INTRA_QUERY_COMPACT_CHARS,
)
from ..config.limits import (
    INTRA_QUERY_COMPACT_TOKENS as _INTRA_QUERY_COMPACT_TOKENS,
)
from ..config.limits import (
    TOOL_HISTORY_CHAR_BUDGET as _TOOL_HISTORY_CHAR_BUDGET,
)
from ..config.limits import (
    TOOL_HISTORY_TOKEN_BUDGET as _TOOL_HISTORY_TOKEN_BUDGET,
)
from ..context.execution_context import ExecutionContext
from ..event_sink import emit
from ..protocol.messages import (
    message_wire_form,
    reconcile_tool_pairs,
)
from .context_budget import context_budget_for

logger = logging.getLogger(__name__)


class ContextOverflowError(RuntimeError):
    """The prompt cannot fit: the system message and the current query alone exceed the window."""


# Opening of the one message a compaction leaves behind. Both producers build it
# through ``compaction_summary_message`` below, so the marker exists once: a resume
# decides what to reload by looking for it, and a second spelling would silently read
# as "never compacted" and reload the whole untrimmed record instead.
COMPACTION_MARKER_PREFIX = "[Context summary"

# How far into a window the marker may sit and still be this window's own opening.
_COMPACTION_MARKER_SCAN = 3

# The count inside the marker. Read back so a second compaction can add what the
# summary it is about to swallow already stood for.
_MARKER_EXCHANGES = re.compile(r"(\d+)\s+prior\s+exchange", re.IGNORECASE)


def _marker_exchanges(m: dict) -> int | None:
    """Exchanges a summary *m* stands in for (0 if unreadable), or None if it is not one."""
    content = m.get("content")
    if not isinstance(content, str):
        return None
    if not content[:len(COMPACTION_MARKER_PREFIX)].lower() == COMPACTION_MARKER_PREFIX.lower():
        return None
    found = _MARKER_EXCHANGES.search(content[:200])
    return int(found.group(1)) if found else 0


def compacted_exchanges(messages: list[dict]) -> int:
    """How many exchanges a summary over *messages* stands in for, counting earlier summaries' share."""
    absorbed = raw = 0
    for m in messages:
        already = _marker_exchanges(m)
        if already is None:
            raw += 1
        else:
            absorbed += already
    return absorbed + raw // 2


def compaction_summary_message(n_exchanges: int, summary: str) -> dict:
    """The single assistant message that stands in for *n_exchanges* summarized turns.

    Assistant role, not user: the model reads the handoff note as its own prior memory
    rather than as something the user typed.
    """
    plural = "s" if n_exchanges != 1 else ""
    return {
        "role": "assistant",
        "content": (
            f"{COMPACTION_MARKER_PREFIX} \u2014 {n_exchanges} prior exchange{plural} "
            f"compacted]\n\n{summary}"
        ),
    }


def carries_compaction_summary(messages: list[dict]) -> bool:
    """True when *messages* opens on a compaction summary (among its first few messages).

    Only such a window is worth reloading in place of the untrimmed record; a front-trimmed
    tail summarizes nothing.
    """
    return any(m.get("role") == "assistant" and _marker_exchanges(m) is not None
               for m in messages[:_COMPACTION_MARKER_SCAN])


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


def _protected_paths(execution_context: ExecutionContext | None) -> set[str]:
    """Files being written or planned, plus every read while a repair is open (its
    anchors may need an earlier read)."""
    if execution_context is None:
        return set()
    paths = execution_context.dirty_written_files | execution_context.declared_edit_set
    if execution_context.dirty_written_files:
        paths = paths | execution_context.read_files
    return {p for p in paths if p}


def _recorded_paths(msg: dict, tool_msg_files: dict) -> list[str] | None:
    """The files recorded for this result at dispatch, or None when it was not tracked.

    Recorded paths avoid the substring hazard: a grep that merely mentions a path
    neither protects nor invalidates that file's read.
    """
    cid = msg.get("tool_call_id")
    return tool_msg_files[cid] if cid is not None and cid in tool_msg_files else None


def _exempt_results(messages: list[dict], tool_indices: list[int],
                    execution_context: ExecutionContext | None) -> set[int]:
    """The results of the last call turn (this turn's payload) and pinned results (a
    skill's instructions, kept in the tail rather than folded into messages[0])."""
    last_call_turn = _last_call_turn(messages)
    exempt = {i for i in tool_indices if i > last_call_turn} if last_call_turn >= 0 else set()
    pinned = execution_context.pinned_call_ids if execution_context else set()
    if pinned:
        exempt |= {i for i in tool_indices if messages[i].get("tool_call_id") in pinned}
    return exempt


def _forget_read(execution_context: ExecutionContext, msg: dict,
                 recorded: list[str] | None) -> None:
    """An evicted read leaves ``read_files``, so policy asks for a re-read before an edit."""
    read_files = execution_context.read_files
    if recorded is not None:
        read_files.difference_update(recorded)
        execution_context.tool_msg_files.pop(msg["tool_call_id"], None)
    else:
        content = msg.get("content", "")
        read_files.difference_update(p for p in list(read_files) if p and p in content)


def _trim_tool_history(
    messages: list[dict],
    char_budget: int = _TOOL_HISTORY_CHAR_BUDGET,
    execution_context: ExecutionContext | None = None,
    token_counter: Any | None = None,
    token_budget: int = _TOOL_HISTORY_TOKEN_BUDGET,
) -> None:
    """Evict the oldest tool results while their total exceeds the budget (tokens, else chars).

    Never evicted: the results of the last call turn, pinned results, results about files
    being written. An evicted read also leaves ``read_files``, so a re-read is required.
    """
    size = token_counter if token_counter is not None else len
    budget = token_budget if token_counter is not None else char_budget
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    excess = sum(_message_tokens(messages[i], size) for i in tool_indices) - budget
    if excess <= 0:
        return

    protected = _protected_paths(execution_context)
    tool_msg_files = execution_context.tool_msg_files if execution_context else {}
    exempt = _exempt_results(messages, tool_indices, execution_context)
    to_remove: list[int] = []
    for idx in tool_indices:
        if excess <= 0:
            break
        if idx in exempt:
            continue
        msg = messages[idx]
        recorded = _recorded_paths(msg, tool_msg_files)
        mentioned = recorded if recorded is not None else [
            p for p in protected if p in msg.get("content", "")]
        if protected and any(p in protected for p in mentioned):
            continue
        to_remove.append(idx)
        # Credited in the units the budget is spent in (the wire form).
        excess -= _message_tokens(msg, size)
        if execution_context is not None:
            _forget_read(execution_context, msg, recorded)

    for i in sorted(to_remove, reverse=True):
        del messages[i]


def _carve_out_pinned(
    middle: list[dict], pinned_call_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    """Split *middle* into ``(summarizable, kept)``, each pinned result kept with its assistant turn.

    Alone, ``reconcile_tool_pairs`` would drop it; the turn's unpinned results are stubbed.
    """
    kept_idx: set[int] = set()
    for i, m in enumerate(middle):
        if m.get("role") != "tool" or m.get("tool_call_id") not in pinned_call_ids:
            continue
        kept_idx.add(i)
        # Walk back over this turn's other results to the assistant turn that owns it.
        j = i - 1
        while j >= 0 and middle[j].get("role") == "tool":
            j -= 1
        if j >= 0 and middle[j].get("role") == "assistant" and middle[j].get("tool_calls"):
            kept_idx.add(j)
    if not kept_idx:
        return middle, []
    summarizable = [m for i, m in enumerate(middle) if i not in kept_idx]
    kept = [m for i, m in enumerate(middle) if i in kept_idx]
    return summarizable, kept


def _maybe_compact_intra_query(
    messages: list[dict],
    system_content: str,
    execution_context: ExecutionContext,
    compact_fn: Any | None,
    token_counter: Any | None = None,
    token_budget: int | None = None,
) -> None:
    """Replace the middle of the messages with a summary when they exceed the budget.

    Keeps the system message, the first user message and the last four; pinned results are
    carved out and put back after the summary.
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

    # A pinned result sits in this slice like any other and would be summarised away
    # with it — the one pass that could silently drop a skill's instructions without
    # the eviction path ever running. Carved out, with its assistant turn.
    pinned_call_ids: set[str] = execution_context.pinned_call_ids or set()
    middle, kept = (_carve_out_pinned(middle, pinned_call_ids)
                    if pinned_call_ids else (middle, []))
    if not middle:
        return  # nothing left to summarise once the pinned pairs are out

    emit({"type": "status", "text": "⚡ Intra-query compaction triggered — summarising intermediate steps..."})
    try:
        summary_messages = compact_fn(middle)
    except Exception:
        logger.debug("_maybe_compact_intra_query: a best-effort step failed and was skipped", exc_info=True)
        return  # compaction failed — silently continue without it

    # Kept pairs go after the summary: they are the methodology the following steps
    # act on, and the summary stands for the work that happened around them.
    messages[2:-4] = list(summary_messages) + kept
    # The task checklist lives in messages[0], which this slice does not touch, so
    # compaction of the middle never disturbs it — nothing to refresh.


def _truncate_text_to_tokens(text: str, max_tokens: int, token_counter: Any) -> str:
    """*text* cut to at most *max_tokens*, head and tail kept around a marker (guaranteed)."""
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


# What one tool result, and one step, may add: fractions of the usable window. A loose
# backstop behind the servers' own ceilings, set so a tool returning its documented
# maximum (``github_get_file``, 256 KB, ~65k tokens) is never cut on a 160k window.
_RESULT_WINDOW_SHARE = 2       # one result may take up to usable // 2
_STEP_NUM, _STEP_DEN = 3, 4    # one step's results, together, up to usable * 3/4

_TRUNCATION_NOTE = (
    "\n\nTOOL_RESULT_TRUNCATED: this result was {full} tokens and was cut to fit the "
    "context budget — the middle is missing. Ask for less (a narrower range, a more "
    "specific query) rather than repeating this call."
)


def _water_fill(sizes: list[int], budget: int) -> list[int]:
    """Per-item allowances summing to at most *budget*, smallest first (deterministic)."""
    allow = [0] * len(sizes)
    left = budget
    for rank, i in enumerate(sorted(range(len(sizes)), key=lambda j: sizes[j])):
        share = left // max(1, len(sizes) - rank)
        allow[i] = min(sizes[i], share)
        left -= allow[i]
    return allow


def bound_step_results(
    results: list[str],
    *,
    model: str,
    context_mode: str,
    token_counter: Any,
    context_ceiling: int | None = None,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Clamp one step's tool results before they enter the history; report each cut.

    Before the append, because afterwards the current step is exempt from eviction.
    """
    if not results:
        return results, []
    total, reserved, _trim, _compact = context_budget_for(
        model, context_mode, ceiling=context_ceiling)
    usable = max(1, total - reserved)
    per_result = max(512, usable // _RESULT_WINDOW_SHARE)
    per_step = max(1024, usable * _STEP_NUM // _STEP_DEN)

    texts = [r if isinstance(r, str) else str(r) for r in results]
    sizes = [token_counter(t) for t in texts]
    allow = [min(s, per_result) for s in sizes]
    if sum(allow) > per_step:
        allow = _water_fill(allow, per_step)

    out: list[str] = list(texts)
    cut: list[tuple[int, int]] = []
    for i, (text, size, cap) in enumerate(zip(texts, sizes, allow)):
        if size <= cap:
            continue
        note = _TRUNCATION_NOTE.format(full=size)
        # The note has to fit inside the allowance too, or bounding the result would
        # be the thing that broke the bound.
        body_cap = max(1, cap - token_counter(note))
        out[i] = _truncate_text_to_tokens(text, body_cap, token_counter) + note
        cut.append((size, token_counter(out[i])))
    return out, cut


def _message_content_str(m: dict) -> str:
    """Return a message's ``content`` as a string (JSON-encode non-str payloads)."""
    c = m.get("content")
    if c is None:
        return ""
    return c if isinstance(c, str) else json.dumps(c)


# What one message costs beyond its own wire form, once it sits in the list that is
# serialised and sent: the ", " that separates it from the next, its share of the
# enclosing brackets, and the sub-token remainder every per-message count floors.
# Measured at ~3 on a 241-message tool-heavy history; kept whole and slightly
# generous, because under-counting here is the failure this whole measure exists
# to prevent.
WIRE_LIST_TOKENS_PER_MESSAGE = 4


def _message_tokens(m: dict, token_counter: Any) -> int:
    """Tokens of one message in the form the provider is sent (``message_wire_form``).

    The budget must measure what the window is checked against; one count per message
    keeps the tokenize cache warm.
    """
    text = message_wire_form(m)
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
    """A new dict of *args* with long strings digested, or *args* itself when nothing changed."""
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
    """Digest the arguments of failed tool calls (lossless: they never took effect); return how many.

    The latest call turn is exempt: a model repairing it needs what it wrote.
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


def force_fit_to_window(
    messages: list[dict],
    target_tokens: int,
    token_counter: Any,
    pinned_call_ids: set[str] | None = None,
) -> bool:
    """Shrink the largest reducible messages until the list fits *target_tokens*.

    The system message and the current query are never reduced; *pinned_call_ids* go
    last. False when that core alone does not fit.
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
    pinned_ids = pinned_call_ids or set()
    pinned = (
        {i for i, m in enumerate(messages)
         if m.get("role") == "tool" and m.get("tool_call_id") in pinned_ids}
        if pinned_ids else set()
    )
    # Largest first; ties broken by oldest (lower index) so recent context survives.
    # A pinned message sorts behind every unpinned one whatever its size: reverse=True
    # puts the HIGHEST key first, so 0 for pinned and 1 for the rest sends the pinned
    # ones to the end of the order, where they are reached only after the rest has been
    # cut and usually not at all.
    order = sorted(
        (i for i in range(len(messages)) if i not in protected),
        key=lambda i: (0 if i in pinned else 1, _mtok(messages[i]), -i),
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


def enforce_context_budget(
    messages: list[dict],
    system_content: str,
    step_tools: list[dict] | None,
    execution_context: ExecutionContext,
    model: str,
    context_mode: str,
    compact_fn: Any | None,
    token_counter: Any,
    context_ceiling: int | None = None,
) -> None:
    """Make the next prompt fit the window, tools schema included.

    Evict old tool results, compact the middle, force-fit, then repair the pairing. Raises
    :class:`ContextOverflowError` (after the repair) when the core alone does not fit.
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
    total, reserved, trim_budget, compact_budget = context_budget_for(
        model, context_mode, ceiling=context_ceiling)
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
        # Per-message counts are the wire form of each message; the list that holds
        # them costs a little more still — the separators between the objects, the
        # enclosing brackets, and the rounding each per-message count floors away.
        # ~0.8% on a tool-heavy history, and the one part of the envelope a
        # per-message measure cannot see, so it is subtracted rather than hoped for:
        # the point of this budget is that it is never the optimistic one.
        usable = max(1, total - reserved - overhead
                     - len(messages) * WIRE_LIST_TOKENS_PER_MESSAGE)
        before = sum(_message_tokens(m, token_counter) for m in messages)
        fitted = force_fit_to_window(
            messages, usable, token_counter,
            pinned_call_ids=execution_context.pinned_call_ids or set(),
        )
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
