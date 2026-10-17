"""The person behind a WebSocket worker: cards raised, answers awaited, turns set aside.

The ``HumanIO`` (``core.ports``) the worker puts on its agent. An approval, an
out-of-workspace path or a question becomes a card sent through the worker's outbox,
and the thread that asked blocks on that card's own answer slot until the client
answers it, Stop cancels it, or — nobody being attached past the grace — the turn is
set aside (``query_engine.deferral``) to be resumed when somebody answers.

Split out of the worker, which used to monkeypatch three shims onto the agent and keep
all of this state among its own.
"""
from __future__ import annotations

import os
import queue as _queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ... import human_pause
from ...context.capabilities import (
    IRREVERSIBLE,
    kind_for,
    label_for,
    preview_spec,
    reversibility_of,
)
from ...query_engine.deferral import CURRENT_CALL_ID
from ...context.tool_display import shorten_display_args
from .file_preview import build_preview_diffs

if TYPE_CHECKING:
    from .ws_worker import _AgentWorker

# What a bounded prompt wait returns when its wall passes with nothing answered. A
# unique instance compared with ``is``, never read as a response: ``None`` already
# means cancelled, and an expired card has to be taken back down rather than treated
# as a user who said no.
TIMED_OUT: dict = {"timed_out": True}


@dataclass
class _Prompt:
    """One card put to the user, and the slot the thread that raised it waits on.

    Each card carries its own answer slot rather than sharing a queue per kind: with
    two cards in flight, a shared queue hands an answer to whichever thread happens to
    be listening, which is not necessarily the one whose card the user was reading.
    """

    payload: dict
    questions: list | None
    answer: "_queue.Queue[dict]"
    shown: bool = False


def _detach_grace() -> float:
    """How long a socket may be gone before this process treats it as gone.

    Not zero, and this is the whole reason the delay exists: reloading a VS Code window
    closes and reopens the socket, and acting on that blink would be a regression — the
    user is right there, and would find their question put away.

    One value, two readers: an unanswered card is set aside past it, and a server no
    conversation has asked to keep stops past it. Both are answering the same question
    — has the window actually gone — so both must answer it the same way.
    """
    try:
        return max(0.0, float(os.environ.get("MIMIR_DETACH_GRACE", "") or 30.0))
    except ValueError:
        return 30.0



class HumanBridge:
    """Every card one worker's agent raises, and the answers that settle them."""

    def __init__(self, worker: "_AgentWorker") -> None:
        self.worker = worker
        # Cards waiting on the user, oldest first, each holding the slot the thread
        # that raised it blocks on. One is shown at a time and each is answered by its
        # own id — see _emit_prompt for why that is not the single slot it used to be.
        self._prompts: "OrderedDict[str, _Prompt]" = OrderedDict()
        self._prompts_lock = threading.Lock()
        # The card currently on screen, set for exactly as long as it is up. Read by a
        # connection that arrives while the wait is on (see _emit_prompt).
        self._pending_prompt: dict | None = None
        # The raw questions of the pending question card — how a deferred question is
        # matched to the call that asked it.
        self._pending_questions: list | None = None
        # When this process last had a client listening, or None while one does. Written
        # by the pump on the event loop and read by the thread that waits, which is why
        # it is a plain float and not an asyncio primitive. See _unattended_past_grace.
        self.unattended_since: float | None = None
        # True while a turn of this conversation is set aside awaiting an answer. Read
        # by the pool, which must not reap the agent out from under it.
        self.has_deferral: bool = False
        # Set when the user leaves a conversation whose turn is parked on them: every
        # wait of that turn returns at once instead (query_engine.deferral).
        self._defer = threading.Event()
        # The answer a resume turn carries, handed to the first prompt of its kind
        # instead of putting the card up again: ``{"type", "response"}``.
        self._preanswer: dict | None = None

    def awaiting_user(self) -> bool:
        """Whether a card is up, waiting on the person."""
        return self._pending_prompt is not None

    def begin_turn(self, resume: dict | None, answer: dict | None) -> None:
        """A turn starts: a resume turn carries the answer its card would have asked for."""
        if resume is not None:
            self._preanswer = {"type": resume.get("prompt", {}).get("type"),
                               "response": answer or {}}

    def end_turn(self) -> None:
        """Deferral is scoped to the turn it was asked for, like the cancel flag."""
        self._defer.clear()
        self._preanswer = None

    def flush(self) -> None:
        """Drop every pending card: the turn they belong to was abandoned."""
        with self._prompts_lock:
            self._prompts.clear()
        self._pending_prompt = None
        self._pending_questions = None

    def resolve_approval(self, choice: str, approved_files: list | None = None,
                         req_id: str = "") -> None:
        """Answer one approval card. *req_id* is the id it was sent with."""
        self._deliver(req_id, {"choice": choice, "approved_files": approved_files},
                      "approval")

    def resolve_question(self, answers: list | None, req_id: str = "") -> None:
        """Answer one question card. *req_id* is the id it was sent with."""
        self._deliver(req_id, {"answers": answers or []}, "user_question")


    def _deferred_record(self) -> dict | None:
        """What the turn that just ended was set aside on, or None.

        The loop's own record (which calls, which mode) plus the card to put back:
        the first prompt deferred, whose id the answer will come back with.
        """
        agent = self.worker._agent
        turn = agent.turn.deferred_turn if agent is not None else None
        prompts = agent.turn.deferred_prompts if agent is not None else None
        if not turn or not prompts:
            return None
        return {**turn, "prompt": prompts[0]["prompt"]}

    def defer_turn(self) -> bool:
        """Set the parked turn aside instead of cancelling it (see deferral).

        True when a turn was parked on a prompt and is now unwinding; False when there
        was nothing to set aside, and the caller should fall back to cancelling.
        """
        if self._pending_prompt is None or not self.worker.is_busy():
            return False
        self._defer.set()
        return True

    def _emit_prompt(self, payload: dict, questions: list | None = None) -> dict:
        """Register a card the caller is about to park on, and show it when its turn comes.

        An agent outlives the connections that read it: a socket that drops while it is
        parked leaves the card on a client nobody is holding, and the turn waiting on an
        answer nobody can give. This agent's query loop is serial, so every later query of
        *this conversation* queues behind that wait and it reads as hung, with nothing on
        screen to explain it. Kept here, the card goes back in front of whoever reconnects
        (``_Session._resend_parked_prompt``).

        Cards queue rather than overwrite one another: one card on screen at a time, in
        the order they were raised, each answered by its own id (:meth:`_deliver`). A
        sub-agent still running after its delegating call has returned raises its card
        from a thread of its own, so two can be in flight at once; a single slot would
        mean the second erased the first while an answer went to whichever waiter
        happened to be listening.

        Returns the attributed payload, so a caller that has to take the card back down
        (an expired question) addresses the same conversation the card named.
        """
        # Which conversation is asking. Structured, never spelled into the label or the
        # question text: that text is also what the model sees, so a marker written there
        # lands in the conversation's own history. The client renders the attribution from
        # these and — the part that is not cosmetic — sends them back on the answer, which
        # is how the answer reaches the agent that asked rather than whichever one is on
        # screen.
        payload = {
            **payload,
            "session_id": self.worker._query_session_id or self.worker.session_id,
            "session_title": self.worker.session_title or "",
        }
        with self._prompts_lock:
            self._prompts[str(payload.get("id") or "")] = _Prompt(
                payload=dict(payload), questions=questions,
                answer=_queue.Queue(maxsize=1))
        self._show_next()
        return payload

    def _show_next(self) -> None:
        """Put the oldest card that has not been shown on screen, if none is up.

        Called whenever the set changes — a card raised, a card answered — so the queue
        advances on its own. Nothing is shown while the turn is deferring (nobody is
        there to read it) or while the head card is pre-answered (the wait settles it
        without a card): both are dropped a moment later, and this runs again for the
        one behind them.
        """
        with self._prompts_lock:
            if any(p.shown for p in self._prompts.values()):
                return
            nxt = next((p for p in self._prompts.values() if not p.shown), None)
            if nxt is None:
                self._pending_prompt = None
                self._pending_questions = None
                return
            if self._deferring() or self._preanswer_for(nxt.payload) is not None:
                return
            nxt.shown = True
            self._pending_prompt = dict(nxt.payload)
            self._pending_questions = nxt.questions
            payload = nxt.payload
        self.worker.out_q.put(payload)

    def _drop_prompt(self, req_id: str) -> None:
        """Forget a card that is answered, cancelled or set aside; show the next."""
        with self._prompts_lock:
            self._prompts.pop(req_id, None)
        self._show_next()

    def _deliver(self, req_id: str, response: dict, kind: str) -> None:
        """Hand *response* to the card that asked for it, or drop it.

        Matched by id — which the front-end has always sent back and this had always
        thrown away. An answer put on a queue shared by kind settles whatever prompt is
        listening, which with two cards in flight is not necessarily the one the user
        was reading. ``_Session._answer_deferred`` already refuses a stale answer on
        exactly this reasoning; this is the same rule for a live one.

        An answer carrying no id at all goes to the card on screen of that kind — the
        only one it could have been meant for.
        """
        with self._prompts_lock:
            prompt = self._prompts.get(req_id) if req_id else None
            if prompt is None and not req_id:
                prompt = next((p for p in self._prompts.values()
                               if p.shown and p.payload.get("type") == kind), None)
            if prompt is None:
                return      # a card that is gone: answering it would settle another
            try:
                prompt.answer.put_nowait(response)
            except _queue.Full:
                pass        # answered twice (a copy of the card on screen): the first wins
    def _unattended_past_grace(self) -> bool:
        """Whether this process has had no client for longer than the grace period.

        The criterion is *attached*, not *elapsed*: a card with somebody there waits
        for ever, which is deliberate and what ``test_approval_wait`` pins. What
        changes when nobody is there is not how long the wait may be but whether there
        is anything to wait for.
        """
        # Absent until the pump's first tick reaches this worker, which is a real
        # state and not only a test's: the safe reading of no information is that
        # somebody is there, because the cost of being wrong the other way is a card
        # put away under the user's nose.
        since = getattr(self, "unattended_since", None)
        if since is None:
            return False
        return (time.monotonic() - since) >= _detach_grace()

    def _deferring(self) -> bool:
        defer = getattr(self, "_defer", None)
        return defer is not None and defer.is_set()

    def _preanswer_for(self, payload: dict | None) -> dict | None:
        pre = getattr(self, "_preanswer", None)
        if pre is None or payload is None or pre.get("type") != payload.get("type"):
            return None
        return pre

    def pending_prompt(self) -> dict | None:
        """The card the parked turn is waiting on, once it is nobody's to deliver.

        None while the card is still queued: the drain loop will hand it to whoever is
        connected, and returning it here as well would put the same card on screen
        twice. That is not merely cosmetic — an approval renders as one card carrying
        both ids and is then answered twice, leaving a spare answer on the queue for
        the *next* prompt to consume, which is the exact failure ``flush_prompts``
        exists to prevent.
        """
        prompt = self._pending_prompt
        if prompt is None:
            return None
        with self.worker.out_q.mutex:
            queued = [ev.get("id") for ev in self.worker.out_q.queue if isinstance(ev, dict)]
        return None if prompt.get("id") in queued else prompt

    def _await_response(self, req_id: str,
                        timeout_secs: float | None = None) -> dict | None:
        """Block until the answer to card *req_id* lands. ``None`` when it does not.

        An unanswered approval must keep the agent *parked*: it must never silently
        proceed just because the user was slow to respond, so it waits with no
        wall-clock limit at all. To stay responsive to the Stop button, we poll in
        short slices and bail the moment the agent's cancel flag is set (from the WS
        thread), returning ``None`` for cancelled.

        *timeout_secs* gives up waiting after that long and returns :data:`TIMED_OUT`
        — told apart from the cancelled ``None`` because the caller has a card on
        screen to take back down, and the two endings read differently to the model.
        Only a clarification question passes one; whoever raised the card decides,
        because the same seam carries approvals (a deadline would authorise a command
        nobody saw) and plan decisions (one would execute a plan nobody approved).

        This is the single seam every WS prompt (approval, out-of-workspace path,
        question) blocks on, so it is where the wait is marked as *human*
        time — excluded from the tool-call timeout budget it sits inside.

        Waits on this card's own slot, so a thread only ever wakes on the answer to the
        card it raised.
        """
        with self._prompts_lock:
            prompt = self._prompts.get(req_id)
        if prompt is None:
            return None     # already flushed: the turn it belonged to was abandoned
        pre = self._preanswer_for(prompt.payload)
        if pre is not None:
            # One answer, one prompt: a second card in the resumed step is asked live.
            self._preanswer = None
            self._drop_prompt(req_id)
            return pre.get("response") or {}
        deadline = None if timeout_secs is None else time.monotonic() + timeout_secs
        try:
            with human_pause.human_pause():
                while True:
                    agent = self.worker._agent
                    if agent is not None and agent.turn.cancel_flag.is_set():
                        return None
                    if self._deferring():
                        self._record_deferral(prompt)
                        return None
                    if deadline is not None and time.monotonic() >= deadline:
                        return TIMED_OUT
                    try:
                        return prompt.answer.get(timeout=0.25)
                    except _queue.Empty:
                        pass
                    # Only once the queue has had its turn and come back empty, and
                    # this order is the point: an answer already in hand must win over
                    # setting the card aside, or a reply that crossed with the grace
                    # elapsing is thrown away and the user answered into a void.
                    #
                    # Nobody to answer, for long enough that it is not a window
                    # reload. Set aside rather than waited out: the wait has no
                    # timeout by design, so without this a detached run meets its
                    # first sensitive tool and holds this thread for ever — and the
                    # pool then reaps the agent out from under it.
                    #
                    # Only the indefinite waits. A question carries its own five
                    # minutes and a documented meaning for running out of them
                    # ("go ahead with what you recommend"); overriding that here
                    # would change an answer the model has already been promised.
                    if deadline is None and self._unattended_past_grace():
                        self._defer.set()
        finally:
            # Answered, cancelled, expired or raised through: this card is not waiting
            # on anyone any more, and one resent past this point would be unheld.
            self._drop_prompt(req_id)

    def _record_deferral(self, prompt: "_Prompt") -> None:
        """Note the prompt being set aside, and which call it holds up.

        Takes the card rather than reading the one on screen: they are the same for a
        turn's own prompt, and for a card still queued behind another they are not.
        """

        agent = self.worker._agent
        if agent is None:
            return
        agent.turn.deferred_prompts = [*(agent.turn.deferred_prompts or []), {
            "call_id": CURRENT_CALL_ID.get(),
            "prompt": dict(prompt.payload),
            "questions": list(prompt.questions or []),
        }]
        # The pool reads this: a deferred turn has cleared `_pending_prompt`, so
        # `is_parked()` is False and nothing else would stop the agent being released
        # while the user still owes it an answer.
        self.has_deferral = True

    def approve_tool(
        self, agent: Any, tool_name: str, arguments: dict, max_attempts: int = 3
    ) -> tuple[bool, str]:
        """Sync approval — blocks background thread until WS client responds.

        The WS event loop (main thread) is unaffected; it forwards the approval
        prompt to the client and answers this card's own slot (see _deliver).
        """

        server_name = agent.tool_owner.get(tool_name, "unknown")
        risk = agent.approvals.describe_risk(tool_name)
        scope_label = agent.approvals._approval_scope_label(tool_name, server_name, arguments)
        # Canonical human label ("Proxy exec: run") so the card header matches the
        # tool-activity row instead of showing a bare op name; None when the tool
        # declares no template, and the client falls back to server·tool.
        label = label_for(
            tool_name,
            shorten_display_args(tool_name, arguments or {}, agent.tool_caps),
            agent.tool_caps,
        )

        # A declared diff-preview spec marks a previewable file mutation: auto-approve
        # (batch review + /undo cover it) and send a live card with the reconstructed
        # diff. No spec — including foreign-server writers — means the normal prompt.
        if preview_spec(tool_name, agent.tool_caps) is not None:
            diffs = build_preview_diffs(tool_name, arguments, agent.tool_caps)
            if diffs:
                self.worker.out_q.put({"type": "file_progress", "diffs": diffs})
            return True, ""

        req_id = str(uuid.uuid4())
        payload: dict = {
            "type": "approval",
            "id": req_id,
            "tool": tool_name,
            "server": server_name,
            "args": arguments,
            "risk": risk,
            # The declared undo level, so the card can show severity from a value
            # instead of keyword-sniffing the risk sentence for "destructive".
            "reversibility": reversibility_of(tool_name, agent.tool_caps),
            "scope": scope_label,
            "label": label,
            # The tool's work family, so the card is drawn with the same icon as the
            # activity row for the same call. Read off the registry like everything else
            # here — the front-end must not learn which tool happens to be a shell.
            "kind": kind_for(tool_name, agent.tool_caps),
        }
        self._emit_prompt(payload)

        # Block background thread (not WS event loop) until the client responds.
        # No timeout: an unanswered prompt keeps the agent parked (Stop cancels).
        response = self._await_response(req_id)
        if response is None:
            return False, "cancelled"

        choice = response.get("choice", "n")

        if choice == "a":
            scope = agent.approvals._approval_scope(tool_name, server_name, arguments)
            agent.approvals.approved_scopes.add(scope)
            return True, f"approved always ({scope_label})"
        if choice == "y":
            return True, "approved once"
        return False, "denied"

    def approve_paths(
        self, agent: Any, paths: list[str], tool_name: str, arguments: dict | None = None
    ) -> tuple[bool, bool]:
        """Out-of-workspace access prompt — reuses the approval UI (y/n/a).

        Carries the same descriptive payload as a normal approval (canonical label,
        the call's real arguments, the owning server) so the card reads like any
        other tool card; ``oow_paths`` is what makes it an out-of-workspace prompt and
        names the offending paths. Returns (approved, always). Blocks the worker thread
        on its card like ``approve_tool``; the engine's out-of-workspace
        gate records the grants.

        **One card for the whole call.** Every outside path the call names travels in a
        single payload: the user judges the command, not each of its operands, and the
        per-path loop this replaced parked the agent on one queue behind several
        identical questions.
        """

        arguments = arguments or {}
        scope = (f"this path ({os.path.basename(paths[0])})" if len(paths) == 1
                 else f"these {len(paths)} paths")
        req_id = str(uuid.uuid4())
        self._emit_prompt({
            "type": "approval",
            "id": req_id,
            "tool": tool_name,
            "server": agent.tool_owner.get(tool_name, "filesystem"),
            "args": arguments or {"path": paths[0]},
            "risk": agent.approvals.describe_risk(tool_name),
            # Reaching outside the workspace is irreversible by situation, not by tool:
            # whatever the tool's own level, nothing here can undo a write landing
            # outside the sandbox, so the card must not soften it to the tool's rating.
            "reversibility": IRREVERSIBLE,
            "scope": scope,
            # Short label in the header; the card renders `oow_paths` beneath it as an
            # explicit "outside workspace" line. The absolute paths are the decision
            # being made, so they are never the thing that gets shortened.
            "label": label_for(
                tool_name,
                shorten_display_args(tool_name, arguments or {}, agent.tool_caps),
                agent.tool_caps,
            ),
            # Same family (and so the same icon) as the row for this call.
            "kind": kind_for(tool_name, agent.tool_caps),
            "oow_paths": list(paths),
            # Kept alongside the list so a client built against the single-path payload
            # still renders a path rather than nothing.
            "oow_path": paths[0],
        })
        # No timeout: keep the agent parked until answered (Stop cancels).
        response = self._await_response(req_id)
        if response is None:
            return (False, False)
        choice = response.get("choice", "n")
        if choice == "a":
            return (True, True)
        if choice == "y":
            return (True, False)
        return (False, False)

    def ask(self, agent: Any, questions: list, origin: dict | None = None,
            timeout_secs: float | None = None) -> dict:
        """Sync clarification questions — blocks the calling thread until answered.

        Mirrors ``approve_tool``: emits a ``user_question`` card carrying the whole
        batch of questions to the client and blocks the thread that raised it (not the
        WS event loop) until a ``user_question_response`` arrives. The frontend shows
        the questions one at a time and returns all ``answers`` together. A cancel
        returns no answers so the agent proceeds with its best judgment.

        *origin* names the asker when it is not the turn on screen — a sub-agent
        working alongside it, which since it can be one of several is not something
        the user can infer. It travels as data so the card can render it as a badge,
        and it is what tells the front end this card does not belong to the main turn.

        *timeout_secs* is how long the asker is willing to wait (``ask_user_question``
        passes one; plan approval does not). When it runs out, a ``prompt_expired``
        event takes the card off the client's screen — it is answering a question
        nobody is waiting on any more — and the answer comes back empty and flagged
        ``timed_out``, which the asking server reads as "decide for yourself". That is
        against an empty answer with no flag, which is a refusal and means "do not
        choose for them"; confusing the two is what left a long run stopped on a card
        nobody would read.
        """
        req_id = str(uuid.uuid4())
        emitted = self._emit_prompt({
            "type": "user_question",
            "id": req_id,
            "questions": list(questions),
            # So the card can show what is left of the wait. Closing it is this side's
            # call (``prompt_expired``); the client only displays the countdown, which
            # is what keeps the card disappearing from reading as a glitch.
            "timeout_secs": int(timeout_secs) if timeout_secs else None,
            **({"origin": origin} if origin else {}),
        }, questions=list(questions))
        response = self._await_response(req_id, timeout_secs)
        if response is TIMED_OUT:
            self.worker.out_q.put({
                "type": "prompt_expired",
                "id": req_id,
                "kind": "user_question",
                "session_id": emitted.get("session_id"),
                "session_title": emitted.get("session_title") or "",
                "timeout_secs": int(timeout_secs or 0),
            })
            return {"answers": [], "timed_out": True, "timeout_secs": timeout_secs}
        if response is None:
            # Stop or a session switch: the person is no longer there at all, which is
            # not the same as the wall above and must not read as "carry on without me".
            return {"answers": []}
        answers: list[dict] = []
        for a in response.get("answers") or []:
            a = a or {}
            answers.append({
                "selected": [str(s) for s in (a.get("selected") or [])],
                "other_text": a.get("otherText") or a.get("other_text") or None,
            })
        return {"answers": answers}

