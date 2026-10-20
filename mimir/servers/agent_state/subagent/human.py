"""Reaching the user from inside a sub-agent: its questions and approvals relayed to the
session that delegated it, and the time the user took to answer kept off the budget."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import threading
import time
from . import limits

# ── Reaching the user from inside a sub-agent ─────────────────────────────────
# Through the MCP session (one per connection, so a detached child can use it too). One
# card at a time across children (this lock), and each carries its ``origin``.
ASK_LOCK = asyncio.Lock()


def child_origin(agent, task: str) -> dict:
    """Which child this is (sub-agent and task), as ``origin`` data for its cards and runs."""
    label = task.strip().splitlines()[0][:60] if task.strip() else "a sub-agent"
    return {"kind": "subagent", "label": label,
            "session": getattr(agent, "session_id", "") or ""}


APPROVAL_OPTIONS = [
    {"label": "Allow once", "description": "Run it this time."},
    {"label": "Allow for the session",
     "description": "Stop asking for this kind of call until the session ends."},
    {"label": "Refuse", "description": "Do not run it; the sub-agent carries on without."},
]


async def ask_the_user(session, header: str, question: str, options: list[dict],
                        origin: dict | None = None) -> str:
    """Put one card to the user, after any other child's; return the chosen label, or "".

    *session* is the MCP ``ServerSession``, so it works after the call returned; *origin*
    travels as data for a badge.
    """
    spec_questions = [{
        "question": question,
        "header": header[:24],
        "options": options,
        "multi_select": False,
    }]
    x_mimir: dict = {"kind": "user_question", "questions": spec_questions}
    if origin:
        x_mimir["origin"] = origin
    schema = {
        "type": "object",
        "title": header[:24] or "Sub-agent",
        "x_mimir": x_mimir,
        "properties": {"answers": {"type": "array", "items": {"type": "object"}}},
        "required": [],
    }
    async with ASK_LOCK:
        try:
            result = await session.elicit_form(message=question, requestedSchema=schema)
        except Exception as exc:
            print(f"spawn_agent: could not reach the user: {exc}", file=sys.stderr)
            return ""
    if getattr(result, "action", "") != "accept":
        return ""
    content = getattr(result, "content", None) or {}
    try:
        answers = json.loads(content.get("answers") or "[]")
        selected = (answers[0] or {}).get("selected") or []
        return str(selected[0]) if selected else str((answers[0] or {}).get("other_text") or "")
    except Exception:
        return ""


PAUSED_LOCK = threading.Lock()


# Seconds a child has spent in front of the user, by session. A card can sit for as
# long as the person takes, and that wait happens outside every clock that would
# normally exempt it: `human_pause` is thread-local and the wait lands on whichever
# thread the client's elicitation callback runs on. Uncounted, a child parked on a
# question ages out of its own budget and is reported lost while its card is still on
# screen. This is where the cards are raised, so this is where the clock is kept.
PAUSED: dict[str, float] = {}


def record_pause(session: str, seconds: float) -> None:
    if not session or seconds <= 0:
        return
    with PAUSED_LOCK:
        PAUSED[session] = PAUSED.get(session, 0.0) + seconds


def paused_secs(session: str) -> float:
    with PAUSED_LOCK:
        return PAUSED.get(session, 0.0)


class RelayedHuman:
    """The user of the session that delegated this child, reached through the caller
    (``core.ports.HumanIO``). *ask* puts one question to them and returns the option
    they chose, or "" when nobody answered."""

    def __init__(self, ask) -> None:
        self._ask = ask

    def approve_tool(self, agent, tool_name: str, arguments: dict,
                     max_attempts: int = 3) -> tuple[bool, str]:
        detail = json.dumps(arguments, ensure_ascii=False, default=str)[:400]
        # The name of the child is in the origin, not repeated here: the card shows it
        # as a badge, and saying it twice on one card reads as two different facts.
        answer = self._ask("Sub-agent", f"Run {tool_name}?\n{detail}", APPROVAL_OPTIONS)
        if answer.startswith("Allow for the session"):
            # Recorded under the same key a front-end "always" uses, so a grant given
            # here narrows the way theirs does — a command family rather than the whole
            # tool. A call whose scope cannot be derived falls back to trusting the tool
            # itself, which is what the coarse path has always meant.
            scope = agent.approval_scope(tool_name, arguments)
            if scope:
                agent.approvals.approved_scopes.add(scope)
            else:
                agent.approvals.trust_tool(tool_name, agent.tool_owner.get(tool_name))
            return True, "approved for this session"
        if answer.startswith("Allow"):
            return True, "approved once"
        return False, "the user refused"

    def approve_paths(self, agent, paths: list[str], tool_name: str,
                      arguments: dict | None = None) -> tuple[bool, bool]:
        shown = ", ".join(paths[:5]) + ("…" if len(paths) > 5 else "")
        answer = self._ask("Outside", f"Reach outside the workspace: {shown}",
                           APPROVAL_OPTIONS)
        if answer.startswith("Allow for the session"):
            return True, True
        return (answer.startswith("Allow"), False)

    def ask(self, agent, questions: list, origin: dict | None = None,
            timeout_secs: float | None = None) -> dict:
        answers = []
        for q in questions or []:
            chosen = self._ask(str(q.get("header") or "Sub-agent"),
                               str(q.get("question") or ""),
                               list(q.get("options") or []))
            answers.append({"header": q.get("header", ""),
                            "selected": [chosen] if chosen else [],
                            "other_text": None})
        return {"answers": answers}


def install_user_channel(agent, session, loop, task: str) -> None:
    """Route this child's approvals and questions to the user through the caller's session.

    Each relay hops onto the server's loop and blocks the child's thread until answered.
    """
    origin = child_origin(agent, task)

    def _ask(header: str, question: str, options: list[dict]) -> str:
        future = asyncio.run_coroutine_threadsafe(
            ask_the_user(session, header, question, options, origin), loop)
        started = time.monotonic()
        try:
            return future.result(timeout=limits.SUBAGENT_BLOCKING_WAIT_SECS)
        except Exception:
            return ""
        finally:
            record_pause(origin["session"], time.monotonic() - started)

    agent.attach(dataclasses.replace(agent.host, human=RelayedHuman(_ask)))
    # It is no longer unattended: there is a person at the end of the channel, and the
    # mode gate must ask them rather than refuse on their behalf.
    agent.approvals.unattended = False
