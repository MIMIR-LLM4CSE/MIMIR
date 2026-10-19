"""A turn of the conversation on screen: the query, its budget, steering, the transcript.

A handler group of ``Connection``, built with its view state, its socket, the pool
and the shared ``Screen``. ``MESSAGES`` names the inbound message types it answers.
"""
from __future__ import annotations

import asyncio
import logging

from .....protocol.messages import reconcile_tool_pairs
from .....session.conversation import trim_front
from .....config import paths
from ......shared import run_channel
from ...runtime import get_backend
from ...worker.worker import AgentWorker
from .base import Handlers

logger = logging.getLogger(__name__)


class TurnHandlers(Handlers):
    MESSAGES: dict[str, str] = {
        "query": "on_query",
        "transcript": "on_transcript",
        "steer": "on_steer",
        "divert_to_background": "on_divert_to_background",
    }

    async def announce_steer_turns(self, texts: list[str]) -> None:
        """Tell the client its unread steering is being answered by a turn of its own:
        ``steer_injected`` per message, the last one saying a turn started."""
        for i, one in enumerate(texts):
            try:
                await self.out.send({
                    "type": "steer_injected", "text": one,
                    "starts_turn": i == len(texts) - 1,
                })
            except Exception:
                logger.debug("_announce_steer_turns: the client went away mid-send", exc_info=True)
                return

    async def announce_detached_answer(self, extras: dict) -> None:
        """A conversation not on screen finished a turn: say so, and that its row moved.

        The pool has already written it into its conversation, and carried on from it.
        """
        committed = extras.get("_committed") or {}
        if not committed:
            return
        title = committed.get("title") or ""
        await self.out.notify(f"“{title}” is waiting for your decision."
                           if committed.get("deferred")
                           else f"“{title}” finished its background turn.")
        await self.screen.send_sessions_list()

    async def _compact_history(self) -> bool:
        """Summarize the middle of the window on the worker thread; True when it shrank. Never raises."""
        history = self.screen.conv.history
        middle = history[1:-4]
        if len(middle) < 3:
            return False
        await self.out.send({"type": "output",
            "text": "  ⚡ Context budget reached — compacting older history…\n"})
        try:
            summary = await asyncio.wrap_future(self.screen.worker.usage.compact_middle(list(middle)))
        except Exception:
            logger.debug("_compact_history: a best-effort step failed and was skipped", exc_info=True)
            return False
        # compact_messages returns its input unchanged when summarization failed.
        if not summary or len(summary) >= len(middle):
            return False
        history[1:-4] = summary
        # Slicing across a turn boundary can strand an assistant tool call or its
        # result; strict backends reject that outright.
        history[:] = reconcile_tool_pairs(history)
        self.pool.bus.record_client_event(self.screen.session_id, {
            "type": "context_compact",
            "dropped": len(middle),
            "kept": len(history),
            "archived": len(self.screen.conv.history_full),
        })
        return True

    async def _fit_history_to_budget(self) -> None:
        """Make the window fit the budget before a new query is added to it.

        Compaction first, front-trim as the fallback, and both are real losses of
        context — so this must run against the budget the conversation actually has.
        Its caller resolves the agent first for that reason.
        """
        # Pre-query budget check: front-trim the oldest history so the new query fits.
        # Deliberately not compact_history — an LLM call from the event loop is unsafe
        # while the worker thread owns the agent.
        total, reserved = self.screen.ctx_budget()
        # Subtract the per-call overhead (system prompt + tools schema) that rides on
        # every call: without it the bar reads past 100% while this check says it fits.
        usable_tokens = max(1, total - reserved - self.screen.worker.usage.overhead_tokens())
        # allow_network=False: never block the WS event loop on tokenize.
        backend = get_backend()
        history = self.screen.conv.history
        counts = backend.message_token_counts(
            self.screen.worker.model, history, allow_network=False
        )
        used_tokens = sum(counts)
        if used_tokens >= usable_tokens and len(history) > 1:
            # Compaction first: it carries forward what the older turns established,
            # where the front-trim below just forgets them. Only when it cannot run
            # (too short a middle) or the summary still doesn't fit do we drop turns.
            if await self._compact_history():
                counts = backend.message_token_counts(
                    self.screen.worker.model, history, allow_network=False
                )
                used_tokens = sum(counts)
                await self.screen.emit_context_usage()
        if used_tokens >= usable_tokens and len(history) > 1:
            await self.out.send({"type": "output",
                "text": "  ⚡ Context budget reached — trimming oldest history…\n"})
            dropped = trim_front(history, counts, usable_tokens)
            # `history_full` is deliberately untouched above — this records what the
            # window lost, so the log stays the one complete account of the session.
            self.pool.bus.record_client_event(self.screen.session_id, {
                "type": "context_trim",
                "dropped": dropped,
                "kept": len(history),
                "archived": len(self.screen.conv.history_full),
            })
            await self.screen.emit_context_usage()

    async def on_query(self, msg: dict) -> None:
        text = (msg.get("text") or "").strip()
        if not text:
            return
        # Writing instead of answering moves past a deferred card: the call keeps its
        # "not run" result, and a late answer to the card goes nowhere.
        if self.screen.conv.pending_interaction is not None:
            prompt_id = (self.screen.conv.pending_interaction.get("prompt") or {}).get("id")
            if prompt_id:
                self.view.stale_prompt_ids.add(prompt_id)
            self.screen.conv.pending_interaction = None

        # Resolve @<uri> mentions on the worker's loop (where the MCP sessions live),
        # then submit the augmented text. History and display keep the RAW `text`, so
        # the attachment is per-turn. Caveat: in full-context mode the augmented message
        # persists in the agent's `turn.last_full_messages`.
        effective_text = text
        # Built here, before anything is submitted: resolving @-mentions needs the
        # worker's own loop, where the MCP sessions live. None means the pool is full,
        # and the turn is queued below with the mentions left unresolved — the text is
        # still exactly what the user wrote.
        worker = await self.screen.ensure_worker()
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
                    worker.panel.resolve_resources(text)
                )
            except Exception:
                attached_uris = []
        if attached_uris:
            await self.out.send({
                "type": "output",
                "text": "📎 Attached: " + ", ".join(attached_uris) + "\n",
            })

        self.screen.conv.add_user_message(text, show=True)
        self.pool.bus.record_client_event(
            self.screen.session_id, {"type": "query", "text": text})
        # Save on arrival so a reconnect mid-turn (e.g. while waiting for an edit
        # approval) reloads from disk instead of minting a blank session ID, which
        # would wipe the chat on the frontend.
        self.screen.save()
        messages = (list(self.screen.conv.history[:-1])
                    + [{"role": "user", "content": effective_text}])
        session_id = self.view.active_session_id

        def _submit(worker: AgentWorker) -> None:
            worker.submit_query(effective_text, messages, session_id=session_id)

        if worker is not None:
            _submit(worker)
            return
        # Every slot is held by a conversation that is working, waiting on the user, or
        # watching a job. Wait for one rather than evicting: taking a slot from a
        # conversation mid-task trades a visible wait for silently lost work.
        position = self.pool.enqueue(self.screen.session_id, _submit)
        await self.out.send({
            "type": "queued",
            "session_id": session_id,
            "position": position,
            "text": (f"  ⏸ Waiting for a free agent slot (#{position} in line). "
                     f"At most {self.pool.cap} conversations run at once."),
        })

    async def on_transcript(self, msg: dict) -> None:
        """Store the client's rendered transcript (full or delta) as the session's display.

        Only for the session on screen, and never with fewer text bubbles than held.
        """
        if self.view.active_session_id is None:
            return
        if msg.get("session_id") != self.view.active_session_id:
            return
        sid = self.view.active_session_id
        # A delta is only ever applied to the copy it was computed against.
        syncs = self.view.transcript_sync
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
            sent = msg.get("messages")
            if not isinstance(sent, list):
                return
            messages = sent
        if self.screen.text_count(messages) < self.screen.text_count(self.screen.conv.display):
            return
        syncs[sid] = {
            "epoch": msg.get("epoch"),
            "rev": msg.get("rev", 0),
            "by_id": {m["id"]: m for m in messages if isinstance(m, dict) and m.get("id")},
        }
        self.screen.conv.display = messages
        # The client says how far the transcript it just sent reaches. Taken on trust
        # the same way the transcript itself is, and only ever forward: a webview that
        # reopened on less than we hold must not shorten the watermark either.
        through = msg.get("through_seq")
        if isinstance(through, int) and through > self.view.rendered_seq:
            self.view.rendered_seq = through
        self.screen.save()

    async def _send_transcript_resync(self, session_id: str) -> None:
        """Ask for the transcript whole: a delta arrived on a revision not held here."""
        try:
            await self.out.send({"type": "transcript_resync", "session_id": session_id})
        except Exception:
            logger.debug("_send_transcript_resync: the client went away mid-send", exc_info=True)

    async def on_steer(self, msg: dict) -> None:
        """Inject a message typed mid-run into the running turn, or start a turn if none is running."""
        text = (msg.get("text") or "").strip()
        if not text:
            return
        if not self.screen.running_turn_is_ours():
            # No turn of ours in flight (or a wake turn of another session): the query path queues
            # it, and the client is told the steer became a turn of its own.
            await self._announce_steer_turn(text)
            await self.on_query(msg)
            return
        self.screen.conv.add_steer(text)
        self.pool.bus.record_client_event(
            self.screen.session_id, {"type": "steer", "text": text})
        self.screen.save()
        self.screen.worker.submit_steer(text)

    async def _announce_steer_turn(self, text: str) -> None:
        """Tell the client its steer became a turn of its own, on the agent's queue behind the answer."""
        ev = {"type": "steer_injected", "text": text, "starts_turn": True,
              # Unstamped it would pass the foreign-event filter and clear a tag in
              # whatever conversation happens to be on screen.
              "session_id": self.view.active_session_id}
        out_q = getattr(self.pool.get(self.view.active_session_id), "out_q", None)
        if out_q is not None:
            out_q.put(ev)
            return
        try:
            await self.out.send(ev)
        except Exception:
            logger.debug("_announce_steer_turn: the client went away mid-send", exc_info=True)

    async def on_divert_to_background(self, msg: dict) -> None:
        """Move the run of the clicked row to the background, through its run channel.

        Never through the model, which is parked on that very call; the row's tool name is the
        channel. The tool result that follows is the confirmation.
        """
        channel = self.view.rows_for(self.view.active_session_id).get(str(msg.get("id") or ""))
        if channel is None or run_channel.request_divert(
                channel, paths.STATE_DIR, self.view.active_session_id) is None:
            await self.out.send({
                "type": "status",
                "text": "  ⓘ Nothing to move — that run had already finished.",
            })
