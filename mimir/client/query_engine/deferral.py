"""A turn parked on the user, set aside when the user leaves its conversation.

A turn parks on a person: a tool approval, a clarification, a plan awaiting a
decision. One worker serves every conversation and runs one turn at a time, so a turn
left parked would hold every other conversation behind it — and cancelling it, as a
session switch used to, threw the card away with the work that led to it.

Deferral is the third way out. The front-end asks the worker to set the wait aside;
each parked prompt then returns at once and is recorded here, the turn finishes the
step it was on and ends. What it was waiting on is stored with the conversation, and
the card comes back when the user does. Their answer starts a resume turn that runs
the deferred calls with that answer and carries on exactly where the turn stopped —
no model call in between, so nothing about the resumption is left to chance.

The record is plain data: ``kind`` is ``"calls"`` (tool calls, identified by their
``tool_call_id``) or ``"plan_decision"`` (the plan loop's approval question, which is
asked between model calls rather than inside one).
"""
from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any

# The tool call whose execution is running in this context. Set around each call's
# task, so a prompt raised from inside the call can say which call it holds up.
CURRENT_CALL_ID: ContextVar[str | None] = ContextVar("mimir_current_call_id", default=None)

KIND_CALLS = "calls"
KIND_PLAN_DECISION = "plan_decision"

# What a deferred call's result reads as while the decision is pending. Only ever seen
# by the model if the user moves on without deciding.
DEFERRED_RESULT = json.dumps({
    "status": "not_run",
    "note": (
        "This call did not run: it was waiting for the user's decision when they left "
        "the conversation, and they have not decided since. Do not assume either answer."
    ),
})


def deferred_prompts(agent: Any) -> list[dict]:
    """The prompts set aside during this turn, as ``{"call_id", "prompt", "questions"}``."""
    return list(getattr(agent, "_deferred_prompts", None) or [])


def park_results(messages: list[dict], call_ids: list[str]) -> None:
    """Give each deferred call the placeholder result, whatever the call returned."""
    wanted = set(call_ids)
    for i, m in enumerate(messages):
        if m.get("role") == "tool" and m.get("tool_call_id") in wanted:
            messages[i] = {**m, "content": DEFERRED_RESULT}


def mark_deferred(agent: Any, *, kind: str, query: str, mode: str,
                  call_ids: list[str] | None = None) -> None:
    """Record that this turn ended deferred, for the front-end to store."""
    agent._deferred_turn = {
        "kind": kind,
        "query": query,
        "mode": mode,
        "call_ids": list(call_ids or []),
    }


def _call_asking(tool_calls: list, questions: list) -> str | None:
    """The call of this step whose arguments carry exactly these questions.

    A question reaches the client through MCP elicitation, handled on the session's
    reader task rather than the call's, so it cannot say which call raised it. The
    asking tool passes its questions through untouched, so the text identifies it.
    """
    texts = [str(q.get("question")) for q in questions
             if isinstance(q, dict) and q.get("question")]
    if not texts:
        return None
    for tc in tool_calls or []:
        fn = tc.get("function") if isinstance(tc, dict) else None
        args = (fn or {}).get("arguments")
        blob = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        if all(t in blob or json.dumps(t, ensure_ascii=False)[1:-1] in blob for t in texts):
            return tc.get("id")
    return None


def end_deferred_calls(agent: Any, messages: list[dict], tool_calls: list, *,
                       query: str, mode: str) -> bool:
    """After a dispatch: if a call was deferred, park its result and mark the turn.

    Returns True when the turn must end here. Every prompt raised during the dispatch
    belongs to one of its calls; the plan decision, asked between calls, is the plan
    loop's to handle.
    """
    call_ids = []
    for d in deferred_prompts(agent):
        cid = d.get("call_id") or _call_asking(tool_calls, d.get("questions") or [])
        if cid and cid not in call_ids:
            call_ids.append(cid)
    if not call_ids:
        return False
    park_results(messages, call_ids)
    mark_deferred(agent, kind=KIND_CALLS, query=query, mode=mode, call_ids=call_ids)
    return True


def end_deferred_turn(agent: Any, execution_context: dict) -> str:
    """End a turn whose calls were set aside. The answer is empty: nothing concluded.

    What the turn learned is kept for the one that resumes it; its transcript is
    recorded on the way out of the loop, like any other ending.
    """
    update = getattr(agent, "_update_carry_context", None)
    if callable(update):
        update(execution_context)
    return ""


def take_deferred_calls(messages: list[dict], call_ids: list[str]) -> list[dict]:
    """Remove the placeholders and return the calls to run, as the model issued them.

    The calls are read from the latest assistant message declaring them; a call whose
    declaration is gone (compacted away) cannot be run again and is left out.
    """
    wanted = set(call_ids)
    messages[:] = [m for m in messages
                   if not (m.get("role") == "tool" and m.get("tool_call_id") in wanted)]
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        calls = [tc for tc in (m.get("tool_calls") or [])
                 if isinstance(tc, dict) and tc.get("id") in wanted]
        if calls:
            return calls
    return []
