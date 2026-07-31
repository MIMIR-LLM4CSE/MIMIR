"""Interaction MCP server — lets the agent ask the user structured questions.

Exposes a single tool, ``ask_user_question``, which pauses the run and surfaces one
or more clickable/selectable prompts to the human via MCP **elicitation**
(``elicitation/create``). The client supplies an ``elicitation_callback`` (see
``mimir/client/integration/server_manager.py``) that renders the questions in
whatever frontend is active — the terminal CLI or the VSCode webview — and returns
the user's selections.

Multiple questions are asked **sequentially** in a single elicitation: the frontend
shows one question at a time and, when the user submits an answer, advances to the
next (Claude-style), returning all answers in one shot — so the agent spends a
single turn regardless of how many questions it bundles.

Because this rides the standard elicitation channel, no bespoke protocol is needed
and any future server can elicit the same way. The rich answer shape (per-question
header, per-option descriptions, single/multi-select, free-text "Other") is carried
in an ``x_mimir`` extension on the requested JSON schema; the user's reply comes
back as ``{"answers": [{"selected": [...], "other_text": ...}, ...]}``.
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import Context, FastMCP
from responses import err, ok
from capabilities import tool_caps

mcp = FastMCP(
    "InteractionServer",
    debug=False,
    log_level="ERROR",
)


# Every frontend already renders an always-present free-text "Other" field, so an
# explicit escape-hatch option ("Other", "Request changes", "Add precisions", "None of
# these", …) is dead weight: it duplicates the field and costs the user a click to get
# to the same box. Models add them anyway, so they are filtered out server-side.
_REDUNDANT_OPTION_LABELS = frozenset({
    "other", "others", "other option", "other answer", "another option",
    "something else", "something different", "anything else",
    "none", "none of these", "none of the above", "neither", "no preference",
    "custom", "custom answer", "custom option", "free text", "free form", "open answer",
    "specify", "let me specify", "i will specify", "i'll specify",
    "type my own", "type my own answer", "write my own", "write my own answer",
    "describe it myself", "i will describe", "i'll describe",
    "request changes", "request a change", "request modifications",
    "suggest changes", "propose changes", "describe changes", "describe the changes",
    "add precision", "add precisions", "add details", "add more details",
    "more details", "provide details", "add context", "add more context",
    "add a clarification", "add a comment", "add a note",
})
# Bare verbs like "edit"/"revise"/"clarify" are deliberately *not* all listed: they can
# be genuine choices in their own right ("Edit / Overwrite / Skip"). Only phrasings whose
# whole point is "let me type something instead" are filtered.


def _is_redundant_option(label: str) -> bool:
    """True when a label merely restates the always-available free-text entry.

    Compared on the whole (normalized) label only — a parenthetical aside is dropped
    ("Other (type your own)") and trailing punctuation/ellipsis ignored — so genuine
    choices that happen to contain one of these words ("Revise the migration script")
    are never filtered.
    """
    text = label.split("(")[0]
    text = text.strip().strip(".…!?:;-–— ").lower()
    return text in _REDUNDANT_OPTION_LABELS


def _normalize_options(options: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Coerce caller options into a clean ``[{label, description}]`` list.

    Escape-hatch options that duplicate the frontend's free-text "Other" field are
    dropped here — see :func:`_is_redundant_option`.
    """
    cleaned: list[dict[str, str]] = []
    for opt in options:
        if isinstance(opt, str):
            opt = {"label": opt}
        if not isinstance(opt, dict):
            continue
        label = str(opt.get("label", "")).strip()
        if not label or _is_redundant_option(label):
            continue
        desc = str(opt.get("description", "")).strip()
        cleaned.append({"label": label, "description": desc})
    return cleaned


def _normalize_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce caller questions into a clean per-question spec list.

    Each item becomes ``{header, question, multiSelect, options}`` with options run
    through :func:`_normalize_options`. Items without a question text or without any
    valid option are dropped.
    """
    cleaned: list[dict[str, Any]] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        question = str(q.get("question", "")).strip()
        options = _normalize_options(q.get("options") or [])
        if not question or not options:
            continue
        cleaned.append({
            "header": str(q.get("header", "")).strip(),
            "question": question,
            "multiSelect": bool(q.get("multi_select") or q.get("multiSelect")),
            "options": options,
        })
    return cleaned


def _parse_answer(raw_answer: Any) -> dict[str, Any]:
    """Normalize one raw ``{selected, other_text}`` reply into a clean answer."""
    if not isinstance(raw_answer, dict):
        raw_answer = {}

    raw = raw_answer.get("selected")
    if isinstance(raw, list):
        selected = [str(s) for s in raw if str(s).strip()]
    elif raw:
        selected = [str(raw)]
    else:
        selected = []

    other = raw_answer.get("other_text") or raw_answer.get("otherText")
    other_text = str(other).strip() if other else None
    if other_text and other_text not in selected:
        selected.append(other_text)

    return {"selected": selected, "other_text": other_text}


@mcp.tool(**tool_caps(read_only=True, label="Asking the user"))
async def ask_user_question(
    questions: list[dict[str, Any]],
    *,
    ctx: Context,
) -> dict:
    """Ask the user one or more clarifying questions and wait for their answers.

    Use this when you genuinely cannot proceed well without the user's input —
    ambiguous requirements, an architectural fork (e.g. "Postgres or SQLite?"), or
    an irreversible/destructive choice. Do NOT use it for things you can infer from
    the code, the conversation, or sensible defaults; prefer acting over asking.

    Bundle several related questions in one call: the frontend shows them one at a
    time and advances to the next when the user submits an answer, then returns all
    answers together (a single agent turn). Set ``multi_select`` per question.

    Args:
        questions: The questions to ask, in order. Each item is
            ``{"question": "...", "header": "...",
              "options": [{"label": "...", "description": "..."}],
              "multi_select": false}``. ``header`` is a short label (a few words,
            e.g. "Database"); ``description`` per option is optional.
            ``multi_select`` (default false) lets the user pick several options for
            that question. List only *substantive* choices: a free-text "Other" entry
            is always shown by the frontend, so escape-hatch options ("Other",
            "Something else", "Request changes", "Add precisions", "None of these")
            are redundant and are stripped from your list.

    Returns:
        ``{"answers": [{"header": ..., "selected": [<labels>], "other_text":
        <str|None>}, ...]}`` — one entry per question, in order — on an answer, or
        ``{"answers": [], "note": "user did not answer; proceed with best
        judgment"}`` if the user declines/dismisses or no interactive frontend is
        connected — in which case continue with your best judgment.
    """
    clean_questions = _normalize_questions(questions)
    if not clean_questions:
        return err(
            "ask_user_question requires at least one question with a substantive option.",
            hint=(
                'Pass questions like [{"question": "Which DB?", "header": "Database", '
                '"options": [{"label": "Postgres"}, {"label": "SQLite"}]}]. Escape-hatch '
                'options ("Other", "Something else", "Request changes") are dropped — the '
                "frontend always shows a free-text field — so list real choices only, or "
                "just ask the question in your answer instead."
            ),
        )

    requested_schema: dict[str, Any] = {
        "type": "object",
        "title": clean_questions[0]["header"] or "Questions",
        # Authoritative rich spec read by the client's elicitation callback.
        "x_mimir": {
            "kind": "user_question",
            "questions": clean_questions,
        },
        "properties": {
            "answers": {
                "type": "array",
                "title": "Answers",
                "items": {
                    "type": "object",
                    "properties": {
                        "selected": {
                            "type": "array",
                            "items": {"type": "string"},
                            "title": "Selection",
                        },
                        "other_text": {"type": "string", "title": "Other"},
                    },
                },
            },
        },
        "required": [],
    }

    # A single elicit message summarizing the batch (per-question text rides x_mimir).
    message = (
        clean_questions[0]["question"]
        if len(clean_questions) == 1
        else f"{len(clean_questions)} questions"
    )

    try:
        result = await ctx.session.elicit_form(
            message=message,
            requestedSchema=requested_schema,
        )
    except Exception as exc:  # client declined elicitation / not supported
        return ok({
            "answers": [],
            "note": f"could not ask the user ({exc}); proceed with best judgment",
        })

    if result.action != "accept" or not result.content:
        return ok({
            "answers": [],
            "note": "user did not answer; proceed with best judgment",
        })

    raw_answers = result.content.get("answers")
    if not isinstance(raw_answers, list):
        raw_answers = []

    answers: list[dict[str, Any]] = []
    for i, q in enumerate(clean_questions):
        raw_answer = raw_answers[i] if i < len(raw_answers) else {}
        parsed = _parse_answer(raw_answer)
        answers.append({
            "header": q["header"],
            "selected": parsed["selected"],
            "other_text": parsed["other_text"],
        })

    return ok({"answers": answers})


if __name__ == "__main__":
    mcp.run()
