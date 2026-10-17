"""Reaching the user from inside a sub-agent: its questions and approvals relayed to the
session that delegated it, and the time the user took to answer kept off the budget."""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from . import spawn_limits

# ── Reaching the user from inside a sub-agent ─────────────────────────────────
# A child that needs an approval, or has a question, must be able to reach the person
# — otherwise the work stops on a step nobody was asked about. The route is the MCP
# session, which is one per stdio connection and not one per request: it is there for
# a detached child exactly as much as for a blocking one, and the client renders the
# card it already renders for any other server.
#
# Two rules the relay owes the user:
#   * ONE AT A TIME. Several children can be working at once; several cards at once is
#     a pile nobody can answer in order. They queue on this lock, so a card is shown,
#     answered, and only then is the next one raised. The client enforces the same rule
#     across every source (the WebSocket worker's _show_next) — this lock keeps a fan-out from
#     filling that queue in the first place.
#   * WHO IS ASKING. Every card carries its sub-agent and its task as `origin`,
#     because "allow this command?" from an unnamed process is a question the user
#     cannot weigh.
ASK_LOCK = asyncio.Lock()


def child_origin(agent, task: str) -> dict:
    """Which child this is, as data — for a card it raises and a run it leaves behind.

    Every approval card carries it, because "allow this command?" from an unnamed
    process is a question the user cannot weigh, and because several children can be
    working at once. A badge on a run carries the same dict for the same reason: a job
    on a node is worth nothing to read if it does not say whose.
    """
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
    """Put one card to the user and return the label they chose, or "" .

    Queued behind every other sub-agent's card, so the answers cannot be attributed to
    the wrong question. A channel that fails is silence, not an answer: the caller
    reads "" as "they did not say", which every caller here treats as a refusal.

    *session* is the MCP ``ServerSession`` — one per stdio connection, not one per
    request — which is why this still reaches the user after the delegating call has
    returned. *origin* says who is asking, and travels as data rather than as a prefix
    glued to the question text: the front end shows it as a badge, and a badge is
    something a person can read at a glance and the interface can style.
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
    """Route this child's approvals and questions to the user, through the caller.

    Installed for every child, detached or not. *session* is the MCP ``ServerSession``
    — one per stdio connection, not one per request — so it still reaches the user
    after the delegating call has returned; the client registers its elicitation
    callback on connect, not per call, and routes whatever arrives to the front end.

    The child runs in its own thread with its own event loop, so each relay hops back
    onto the server's loop and blocks its own thread until the user answers. Blocking
    is correct here: the child has nothing to do until it has its answer.
    """
    origin = child_origin(agent, task)

    def _ask(header: str, question: str, options: list[dict]) -> str:
        future = asyncio.run_coroutine_threadsafe(
            ask_the_user(session, header, question, options, origin), loop)
        started = time.monotonic()
        try:
            return future.result(timeout=spawn_limits.SUBAGENT_HARD_CAP_SECS)
        except Exception:
            return ""
        finally:
            record_pause(origin["session"], time.monotonic() - started)

    agent.human = RelayedHuman(_ask)
    # It is no longer unattended: there is a person at the end of the channel, and the
    # mode gate must ask them rather than refuse on their behalf.
    agent.approvals.unattended = False
