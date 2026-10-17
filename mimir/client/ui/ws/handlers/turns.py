"""A turn of the conversation on screen: the query, its budget, steering, the transcript.

A part of ``_Session`` (``ws_session``): a mixin over the connection's own state, so a
handler reads and writes the session it serves. ``MESSAGES`` names the inbound message
types it answers.
"""
from __future__ import annotations

import asyncio
import json
import logging

from ....protocol.messages import reconcile_tool_pairs
from ....session.conversation import trim_front
from ....tool_execution import run_channel
from .._ws_runtime import (
    context_budget_for,
    get_backend,
)
from ..ws_worker import _AgentWorker

logger = logging.getLogger(__name__)


class TurnHandlers:
    MESSAGES: dict[str, str] = {
        "query": "_handle_query",
        "transcript": "_handle_transcript",
        "steer": "_handle_steer",
        "divert_to_background": "_handle_divert_to_background",
    }

    def _ctx_budget(self) -> tuple:
        """Return (total_tokens, reserved_tokens) for the current context mode.

        For vLLM the window tracks the server's reported max_model_len (primed at
        startup so this runs against the cache, never blocking the event loop).
        """
        mode = self.worker.setting("context_mode", self.conv.record.context_mode)
        total, reserved, _, _ = context_budget_for(self.worker.model, mode)
        return total, reserved

    async def _emit_context_usage(self) -> None:
        """Push a context_usage event to the WS client (best-effort, never raises)."""
        try:
            total, reserved = self._ctx_budget()
            # While a query runs, the transcript that matters is the agent's in-flight
            # one: the conversation only gains the turn once the answer lands, so the bar
            # would sit frozen for the whole run.
            messages = self.worker.live_history()
            if messages is None:
                messages = self.conv.history
            # allow_network=False: on the WS event loop, which must not block on a
            # tokenize round-trip. Already-counted messages hit the shared cache for
            # exact numbers; the rest fall back to the heuristic.
            history_used = get_backend().count_messages_tokens(
                self.worker.model, messages, allow_network=False
            )
            # Include the fixed per-call overhead (system prompt + tools schema).
            # Without it the bar shows only the conversation and hides the tens of
            # thousands of tokens (~30k with every tool server on) that ride on each call.
            overhead = self.worker.context_overhead_tokens()
            used = history_used + overhead
            if used == self._last_context_usage:
                return  # nothing moved — don't spend a frame on an identical payload
            self._last_context_usage = used
            await self.ws.send(json.dumps({
                "type": "context_usage",
                "used_tokens": used,
                "total_tokens": total,
                "reserved_tokens": reserved,
                "overhead_tokens": overhead,
                "overhead_measured": self.worker.context_overhead_is_measured(),
                # No agent yet, so the fixed part — tens of thousands of tokens — is not
                # in `used_tokens` at all. Said out loud rather than left to be read off
                # a missing figure: this is the one moment the bar is too low by more
                # than a rounding error, and a bar that claims an overflow it cannot
                # have measured teaches the user to distrust the one that can.
                "provisional": overhead == 0,
                # What the model actually has this turn, against the untrimmed record
                # kept behind it — so a trimmed window is visible, not silent.
                "history_messages": len(self.conv.history),
                "history_messages_full": len(self.conv.history_full),
            }))
        except Exception:
            pass

    async def _announce_steer_turns(self, texts: list[str]) -> None:
        """Tell the client its unread steering is being answered by a turn of its own:
        ``steer_injected`` per message, the last one saying a turn started."""
        for i, one in enumerate(texts):
            try:
                await self.ws.send(json.dumps({
                    "type": "steer_injected", "text": one,
                    "starts_turn": i == len(texts) - 1,
                }))
            except Exception:
                return

    async def _announce_detached_answer(self, extras: dict) -> None:
        """A conversation not on screen finished a turn: say so, and that its row moved.

        The pool has already written it into its conversation, and carried on from it.
        """
        committed = extras.get("_committed") or {}
        if not committed:
            return
        title = committed.get("title") or ""
        await self._notify(f"“{title}” is waiting for your decision."
                           if committed.get("deferred")
                           else f"“{title}” finished its background turn.")
        await self._send_sessions_list()

    async def _compact_history(self) -> bool:
        """Summarize the middle of the session history. True when it actually shrank.

        Keeps the opening user message and the last two exchanges and replaces what
        lies between with a single summary — the same shape the agent loop's
        intra-query compaction uses. The summarization is an LLM call, so it happens
        on the worker thread (``_AgentWorker.compact_middle``); this coroutine only
        awaits its future. ``history_full`` is deliberately untouched: it stays the
        one complete account of the session, exactly as for a front-trim.

        Never raises — the caller falls back to front-trimming on a False.
        """
        history = self.conv.history
        middle = history[1:-4]
        if len(middle) < 3:
            return False
        await self.ws.send(json.dumps({"type": "output",
            "text": "  ⚡ Context budget reached — compacting older history…\n"}))
        try:
            summary = await asyncio.wrap_future(self.worker.compact_middle(list(middle)))
        except Exception:
            return False
        # compact_messages returns its input unchanged when summarization failed.
        if not summary or len(summary) >= len(middle):
            return False
        history[1:-4] = summary
        # Slicing across a turn boundary can strand an assistant tool call or its
        # result; strict backends reject that outright.
        history[:] = reconcile_tool_pairs(history)
        self.pool.bus.record_client_event(self._active_session_id, {
            "type": "context_compact",
            "dropped": len(middle),
            "kept": len(history),
            "archived": len(self.conv.history_full),
        })
        return True

    async def _ensure_worker(self) -> _AgentWorker | None:
        """This conversation's agent, built now if it has none and there is room.

        None means the pool is full of conversations that are working, waiting on the user,
        or watching a job — none of which may be evicted to make room. The caller queues
        instead, and the user is told, because a queue nobody can see reads as a hang.

        The build costs the LLM backend wait plus ~19 MCP server spawns, so it runs off the
        event loop — the other conversations keep streaming — and this one announces it.
        """
        session_id = self._active_session_id
        if not session_id:
            return None
        existing = self.pool.get(session_id)
        if existing is not None:
            return existing

        def _announce() -> None:
            self._schedule(self.ws.send(json.dumps({
                "type": "status",
                "session_id": session_id,
                "text": "  ⏳ Starting an agent for this conversation…",
            })))

        try:
            worker = await self.pool.worker_for(session_id, on_wait=_announce)
            if worker is not None:
                worker.session_title = self._session_title(session_id)
            return worker
        except Exception as exc:
            await self.ws.send(json.dumps({
                "type": "error",
                "session_id": session_id,
                "text": f"Could not start an agent for this conversation: {exc}",
            }))
            return None

    async def _fit_history_to_budget(self) -> None:
        """Make the window fit the budget before a new query is added to it.

        Compaction first, front-trim as the fallback, and both are real losses of
        context — so this must run against the budget the conversation actually has.
        Its caller resolves the agent first for that reason.
        """
        # Pre-query budget check: front-trim the oldest history so the new query fits.
        # Deliberately not compact_history — an LLM call from the event loop is unsafe
        # while the worker thread owns the agent.
        total, reserved = self._ctx_budget()
        # Subtract the per-call overhead (system prompt + tools schema) that rides on
        # every call: without it the bar reads past 100% while this check says it fits.
        usable_tokens = max(1, total - reserved - self.worker.context_overhead_tokens())
        # allow_network=False: never block the WS event loop on tokenize.
        backend = get_backend()
        history = self.conv.history
        counts = backend.message_token_counts(
            self.worker.model, history, allow_network=False
        )
        used_tokens = sum(counts)
        if used_tokens >= usable_tokens and len(history) > 1:
            # Compaction first: it carries forward what the older turns established,
            # where the front-trim below just forgets them. Only when it cannot run
            # (too short a middle) or the summary still doesn't fit do we drop turns.
            if await self._compact_history():
                counts = backend.message_token_counts(
                    self.worker.model, history, allow_network=False
                )
                used_tokens = sum(counts)
                await self._emit_context_usage()
        if used_tokens >= usable_tokens and len(history) > 1:
            await self.ws.send(json.dumps({"type": "output",
                "text": "  ⚡ Context budget reached — trimming oldest history…\n"}))
            dropped = trim_front(history, counts, usable_tokens)
            # `history_full` is deliberately untouched above — this records what the
            # window lost, so the log stays the one complete account of the session.
            self.pool.bus.record_client_event(self._active_session_id, {
                "type": "context_trim",
                "dropped": dropped,
                "kept": len(history),
                "archived": len(self.conv.history_full),
            })
            await self._emit_context_usage()

    async def _handle_query(self, msg: dict) -> None:
        text = (msg.get("text") or "").strip()
        if not text:
            return
        # Writing instead of answering moves past a deferred card: the call keeps its
        # "not run" result, and a late answer to the card goes nowhere.
        if self.conv.pending_interaction is not None:
            prompt_id = (self.conv.pending_interaction.get("prompt") or {}).get("id")
            if prompt_id:
                self._stale_prompt_ids.add(prompt_id)
            self.conv.pending_interaction = None

        # Resolve @<uri> mentions on the worker's loop (where the MCP sessions live),
        # then submit the augmented text. History and display keep the RAW `text`, so
        # the attachment is per-turn. Caveat: in full-context mode the augmented message
        # persists in the worker's `_last_full_messages`.
        effective_text = text
        # Built here, before anything is submitted: resolving @-mentions needs the
        # worker's own loop, where the MCP sessions live. None means the pool is full,
        # and the turn is queued below with the mentions left unresolved — the text is
        # still exactly what the user wrote.
        worker = await self._ensure_worker()
        # After the agent, never before it: the budget is sized from its context mode
        # and its prompt overhead, and neither can be read off a conversation that has
        # no agent yet. A resumed session asked too early was budgeted at compact's 32k,
        # which read as an overflow and compacted and trimmed a history that fit the
        # real window perfectly well.
        await self._fit_history_to_budget()
        attached_uris: list[str] = []
        if worker is not None:
            try:
                effective_text, attached_uris = await asyncio.wrap_future(
                    worker.resolve_resources(text)
                )
            except Exception:
                attached_uris = []
        if attached_uris:
            await self.ws.send(json.dumps({
                "type": "output",
                "text": "📎 Attached: " + ", ".join(attached_uris) + "\n",
            }))

        self.conv.add_user_message(text, show=True)
        self.pool.bus.record_client_event(
            self._active_session_id, {"type": "query", "text": text})
        # Save on arrival so a reconnect mid-turn (e.g. while waiting for an edit
        # approval) reloads from disk instead of minting a blank session ID, which
        # would wipe the chat on the frontend.
        self._save()
        messages = (list(self.conv.history[:-1])
                    + [{"role": "user", "content": effective_text}])
        session_id = self._active_session_id

        def _submit(worker: _AgentWorker) -> None:
            worker.submit_query(effective_text, messages, session_id=session_id)

        if worker is not None:
            _submit(worker)
            return
        # Every slot is held by a conversation that is working, waiting on the user, or
        # watching a job. Wait for one rather than evicting: taking a slot from a
        # conversation mid-task trades a visible wait for silently lost work.
        position = self.pool.enqueue(session_id, _submit)
        await self.ws.send(json.dumps({
            "type": "queued",
            "session_id": session_id,
            "position": position,
            "text": (f"  ⏸ Waiting for a free agent slot (#{position} in line). "
                     f"At most {self.pool.cap} conversations run at once."),
        }))

    @staticmethod
    def _text_count(messages: list) -> int:
        """Number of plain-text bubbles — the part of a transcript both sides share.

        A note this layer wrote itself (``role: "system"``) is not that part: the
        webview never received it, so counting it made every later transcript the
        client sent look short by one and the guard below refused them all. The stored
        chat then froze at the moment of the first notice and came back stripped of
        every tool row, diff card and reasoning panel after it.
        """
        return sum(1 for m in messages
                   if isinstance(m, dict) and m.get("kind", "text") == "text"
                   and m.get("role") != "system")

    async def _handle_transcript(self, msg: dict) -> None:
        """Store the client's rendered transcript as this session's display messages.

        The rich transcript — tool rows, reasoning panels, diff cards — is assembled by
        the webview's reducer and exists nowhere else; the server only ever appended the
        text bubbles, which is why a reload used to come back stripped to prose. Rather
        than rebuild that assembly here, we take the client's copy of it.

        Two things make that safe to trust: the transcript must name the session it
        belongs to (one arriving after a switch would otherwise overwrite the session the
        user just moved to), and it must not have fewer text bubbles than what we hold —
        a webview that just opened on an empty view must never blank a stored history.
        """
        if self._active_session_id is None:
            return
        if msg.get("session_id") != self._active_session_id:
            return
        sid = self._active_session_id
        # Per session: the client's copy as last applied, by id, and which revision of
        # which run of full-then-deltas it is. A delta is only ever applied to that.
        syncs = getattr(self, "_transcript_sync", None)
        if syncs is None:
            syncs = self._transcript_sync = {}
        if msg.get("delta"):
            held = syncs.get(sid)
            order = msg.get("order")
            changed = msg.get("changed")
            if (held is None or held["epoch"] != msg.get("epoch")
                    or held["rev"] != msg.get("base_rev")
                    or not isinstance(order, list) or not isinstance(changed, list)):
                await self._send_transcript_resync(sid)
                return
            by_id = dict(held["by_id"])
            for m in changed:
                if isinstance(m, dict) and m.get("id"):
                    by_id[m["id"]] = m
            if any(i not in by_id for i in order):
                await self._send_transcript_resync(sid)
                return
            messages = [by_id[i] for i in order]
        else:
            messages = msg.get("messages")
            if not isinstance(messages, list):
                return
        if self._text_count(messages) < self._text_count(self.conv.display):
            return
        syncs[sid] = {
            "epoch": msg.get("epoch"),
            "rev": msg.get("rev", 0),
            "by_id": {m["id"]: m for m in messages if isinstance(m, dict) and m.get("id")},
        }
        self.conv.display = messages
        # The client says how far the transcript it just sent reaches. Taken on trust
        # the same way the transcript itself is, and only ever forward: a webview that
        # reopened on less than we hold must not shorten the watermark either.
        through = msg.get("through_seq")
        if isinstance(through, int) and through > self._rendered_seq:
            self._rendered_seq = through
        self._save()

    async def _send_transcript_resync(self, session_id: str) -> None:
        """Ask for the transcript whole: a delta arrived on a revision not held here."""
        try:
            await self.ws.send(json.dumps(
                {"type": "transcript_resync", "session_id": session_id}))
        except Exception:
            pass

    async def _handle_steer(self, msg: dict) -> None:
        """A message typed while the agent is busy — inject it into the running run.

        The message is recorded in history/display (so it persists like a normal
        turn) and handed to the worker's steer queue, which the agent loop drains at
        its next step boundary. If no run is actually in flight (race: the agent just
        finished), fall back to the normal query path so the message isn't dropped.
        """
        text = (msg.get("text") or "").strip()
        if not text:
            return
        if not self._running_turn_is_ours():
            # Nothing of ours in flight — or a detached wake turn running in another
            # session, which this message must not be injected into. Either way the
            # normal query path is right: the serial query loop queues it.
            #
            # The client sent a steer because it still believes a run is going, and the
            # end of a run reaches the worker before its answer reaches the screen: a
            # message typed in that gap is answered as a turn of its own, so it is also
            # told so — unannounced, the bubble kept the "queued" tag of a run that was
            # already over.
            await self._announce_steer_turn(text)
            await self._handle_query(msg)
            return
        self.conv.add_steer(text)
        self.pool.bus.record_client_event(
            self._active_session_id, {"type": "steer", "text": text})
        self._save()
        self.worker.submit_steer(text)

    async def _announce_steer_turn(self, text: str) -> None:
        """Tell the client the steer it just sent became a turn of its own.

        The same ``steer_injected`` :meth:`_announce_steer_turns` sends, and for the
        same two reasons: the bubble drops the "queued" tag of a run that is over, and
        ``busy`` is set for a turn the user is waiting on but never pressed send for.

        It goes on the agent's output queue rather than to the socket, because what
        makes it read correctly is where it sits relative to the answer of the run the
        message was typed into. That answer is on that queue, ahead of this, and it is
        what clears ``busy``: sent straight to the socket this would arrive first and
        the turn would run with no stop button and no end the composer can see. With no
        agent there is no queue and no answer in flight either, so nothing to be ordered
        behind — the socket is then the only route.
        """
        ev = {"type": "steer_injected", "text": text, "starts_turn": True,
              # Unstamped it would pass the foreign-event filter and clear a tag in
              # whatever conversation happens to be on screen.
              "session_id": self._active_session_id}
        out_q = getattr(self.pool.get(self._active_session_id), "out_q", None)
        if out_q is not None:
            out_q.put(ev)
            return
        try:
            await self.ws.send(json.dumps(ev))
        except Exception:
            pass

    def _rows_for(self, session_id: str | None) -> dict[str, str]:
        """The divertible tool rows of one conversation's running turn."""
        return self._live_rows.setdefault(session_id or "", {})

    def _forget_row(self, call_id: str, session_id: str | None = None) -> None:
        """Drop a finished row from the divert/progress bookkeeping."""
        sid = session_id if session_id is not None else self._active_session_id
        self._rows_for(sid).pop(call_id, None)
        self._sent_progress.setdefault(sid or "", {}).pop(call_id, None)

    async def _handle_divert_to_background(self, msg: dict) -> None:
        """Move the run the user pointed at into the background.

        Served here, on the WS loop, and deliberately never through the model: the
        worker thread is parked awaiting the tool result for the whole call, and the
        steer queue is drained only at a step boundary, so an instruction routed that
        way could not arrive until after the run it was meant to divert had ended.
        Writing the request to the shared state dir is what reaches a server whose
        event loop that very call is holding.

        Which run is decided by the row the user clicked: the message carries the call
        id, and the row's tool name is the channel the owning server publishes under.
        Resolved here rather than shipped by the webview, which has no business
        knowing a tool name. Sending the request to the shell's channel whatever was
        clicked has a proxy run report "nothing to move" while a perfectly innocent shell
        command is the one that gets detached.

        Nothing worker- or agent-side is touched: the confirmation the user sees is
        the tool result that lands a moment later, carrying the work done so far and
        the job handle. Saying anything more here would be predicting it.
        """
        channel = self._rows_for(self._active_session_id).get(str(msg.get("id") or ""))
        if channel is None or run_channel.request_divert(
                channel, self._active_session_id) is None:
            await self.ws.send(json.dumps({
                "type": "status",
                "text": "  ⓘ Nothing to move — that run had already finished.",
            }))

    async def _admit_waiting(self) -> None:
        """Start the turn of a conversation that was waiting for an agent slot.

        Called the moment a turn ends, because that is when its conversation stops being
        busy and so becomes the one the pool may release to make room. Leaving it to the
        idle sweep makes a queued conversation wait out the sweep interval after the slot
        it needs has already freed — half a minute of nothing, having just been told it is
        next in line.
        """
        try:
            for session_id in await self.pool.pump():
                await self.ws.send(json.dumps({
                    "type": "status",
                    "session_id": session_id,
                    "text": "  ▶ An agent slot freed — this conversation is starting.",
                }))
        except Exception:
            logger.warning("could not admit a queued conversation", exc_info=True)

    def _running_turn_is_ours(self) -> bool:
        """True when the conversation on screen has a turn in flight.

        Asked of that conversation's own agent. Several can be working at once, so what
        matters for a message the user typed is whether *this* conversation is busy, in
        which case the message steers its turn instead of starting one.
        """
        return self.pool.is_busy(self._active_session_id)
