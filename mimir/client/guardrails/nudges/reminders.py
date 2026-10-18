"""Putting a reminder into the conversation, and taking the transient ones out again."""
from __future__ import annotations

from typing import Any
from ...protocol.event_sink import emit
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
    """Append a reminder as a user turn and announce it (``nudge_injected``). Returns it.

    Every injection goes through here, so the webview can hold the turn it rejects. *tagged*
    False for protocol steps. With *execution_context* it is transient: taken back after
    the call by :func:`drop_transient_reminders`.
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
        execution_context.nudges.transient_reminders.append((category, msg))
    return msg


def drop_transient_reminders(
    messages: list[dict[str, Any]],
    execution_context: ExecutionContext | None,
) -> int:
    """Remove the reminders injected for the call just made, by identity; return how many.

    Their categories stay in ``_last_call_reminders`` for the empty-turn diagnostic.
    """
    if execution_context is None:
        return 0
    pending = execution_context.nudges.transient_reminders
    if not pending:
        execution_context.nudges.last_call_reminders = []
        return 0
    doomed = {id(m) for _cat, m in pending}
    before = len(messages)
    messages[:] = [m for m in messages if id(m) not in doomed]
    execution_context.nudges.last_call_reminders = [cat for cat, _m in pending]
    execution_context.nudges.transient_reminders = []
    return before - len(messages)
