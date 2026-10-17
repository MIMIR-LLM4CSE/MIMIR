"""Putting a reminder into the conversation, and taking the transient ones out again."""
from __future__ import annotations

from __future__ import annotations
from typing import Any
from ...event_sink import emit
from ...context.execution_context import ExecutionContext


# Prefix marking every workflow nudge as machine-generated rather than user speech.
# Nudges must be injected with role "user" (the Claude backend rejects mid-array
# "system" messages), so without this tag the model cannot tell an automated reminder
# from a real instruction and would let it override the system prompt's latitude clause.
_NUDGE_TAG = "[automated workflow reminder — not from the user; advisory, apply judgment]\n\n"


def inject_reminder(
    messages: list[dict[str, Any]],
    content: str,
    *,
    category: str,
    tagged: bool = True,
    execution_context: ExecutionContext | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    """Append a machine-generated reminder as a user turn and announce the injection.

    EVERY injection point must go through here, not just the nudge table. The event is
    not decoration: the webview holds the turn in flight *outside* the transcript and
    commits it only once the loop accepts it, so a reminder injected silently leaves
    the rejected prose on screen looking like the answer, to be replaced without
    explanation when the real one lands. The loop-control and plan-mode reminders were
    injected that way, which is why the "answer appears then changes" symptom survived
    the draft fix.

    *tagged* is False for the reminders that are protocol, not advice (the plan-mode
    control flow): prefixing those with an "advisory, apply judgment" banner invites
    the model to skip a step the loop actually requires.

    A reminder is TRANSIENT when *execution_context* is given: it is recorded there and
    :func:`drop_transient_reminders` takes it back out after the call it was meant to
    influence, so it never reaches persisted history. Without that the same reminder
    was appended again on every occasion and stayed forever — one session carried 21
    copies of a single sentence and 17 duplicate workflow reminders, ~10% of its prompt.
    The emitted event is what a reader diagnoses from, and it is unaffected: it records
    the category and step whether or not the message itself is kept.

    Returns the appended message dict.
    """
    event: dict[str, Any] = {"type": "nudge_injected", "category": category, "text": content}
    if step is not None:
        event["step"] = step
    emit(event)
    msg: dict[str, Any] = {
        "role": "user",
        "content": (_NUDGE_TAG + content) if tagged else content,
    }
    messages.append(msg)
    if execution_context is not None:
        execution_context.transient_reminders.append((category, msg))
    return msg


def drop_transient_reminders(
    messages: list[dict[str, Any]],
    execution_context: ExecutionContext | None,
) -> int:
    """Take back the reminders injected for the call that has just been made.

    Removal is by IDENTITY, not by position: budget enforcement may have trimmed,
    compacted or reordered the list between the injection and the call, so a tail-only
    pop would silently do nothing whenever anything else was appended in the meantime.
    A reminder that compaction already summarised away is simply not found, which is
    the correct outcome.

    The categories removed are left in ``_last_call_reminders`` so a caller can still
    say what the prompt carried after the fact — the empty-turn diagnostic reads it,
    and it necessarily runs after this.

    Returns how many were removed.
    """
    if execution_context is None:
        return 0
    pending = execution_context.transient_reminders
    if not pending:
        execution_context.last_call_reminders = []
        return 0
    doomed = {id(m) for _cat, m in pending}
    before = len(messages)
    messages[:] = [m for m in messages if id(m) not in doomed]
    execution_context.last_call_reminders = [cat for cat, _m in pending]
    execution_context.transient_reminders = []
    return before - len(messages)
