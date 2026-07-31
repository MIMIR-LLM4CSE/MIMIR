"""Unit coverage for the ``interaction`` MCP server (``ask_user_question``).

Loads the server module directly (like ``test_server_contracts``) and drives the
tool with a fake elicitation session, so we exercise the multi-question batch
contract — ``x_mimir.questions`` construction, per-question answer parsing (single /
multi / free-text "Other"), and the decline / empty-input fallbacks — without a live
MCP transport or any frontend.
"""

from __future__ import annotations

import asyncio
import importlib.util
import types
import unittest
from pathlib import Path

SERVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "servers" / "interaction" / "server_interaction.py"
)


def _load_server():
    spec = importlib.util.spec_from_file_location("server_interaction", SERVER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = _load_server()


class _FakeSession:
    """Records the elicit call and returns a canned result."""

    def __init__(self, result):
        self._result = result
        self.calls: list[tuple] = []

    async def elicit_form(self, message, requestedSchema):
        self.calls.append((message, requestedSchema))
        return self._result


def _ctx(result):
    session = _FakeSession(result)
    return types.SimpleNamespace(session=session), session


def _result(action, content):
    return types.SimpleNamespace(action=action, content=content)


def _run(questions, result):
    ctx, session = _ctx(result)
    payload = asyncio.run(server.ask_user_question(questions, ctx=ctx))
    return payload, session


_Q_DB = {
    "question": "Which database?",
    "header": "Database",
    "options": [{"label": "Postgres"}, {"label": "SQLite"}],
}
_Q_FEATURES = {
    "question": "Which features?",
    "header": "Features",
    "multi_select": True,
    "options": [{"label": "Auth"}, {"label": "Cache"}, {"label": "Search"}],
}


class AskUserQuestionTests(unittest.TestCase):
    def test_builds_x_mimir_questions_list(self) -> None:
        accept = _result("accept", {"answers": [{"selected": ["Postgres"]},
                                                 {"selected": ["Auth", "Cache"]}]})
        _, session = _run([_Q_DB, _Q_FEATURES], accept)

        self.assertEqual(len(session.calls), 1)
        _msg, schema = session.calls[0]
        spec = schema["x_mimir"]
        self.assertEqual(spec["kind"], "user_question")
        self.assertEqual(len(spec["questions"]), 2)
        self.assertEqual(spec["questions"][0]["header"], "Database")
        self.assertFalse(spec["questions"][0]["multiSelect"])
        self.assertTrue(spec["questions"][1]["multiSelect"])
        # Answers schema is a list, matching the batch contract.
        self.assertEqual(schema["properties"]["answers"]["type"], "array")

    def test_parses_answers_in_order(self) -> None:
        accept = _result("accept", {"answers": [{"selected": ["Postgres"]},
                                                 {"selected": ["Auth", "Cache"]}]})
        payload, _ = _run([_Q_DB, _Q_FEATURES], accept)

        self.assertEqual(payload["status"], "ok")
        answers = payload["answers"]
        self.assertEqual(len(answers), 2)
        self.assertEqual(answers[0], {"header": "Database", "selected": ["Postgres"],
                                      "other_text": None})
        self.assertEqual(answers[1]["header"], "Features")
        self.assertEqual(answers[1]["selected"], ["Auth", "Cache"])

    def test_single_string_selection_is_coerced_to_list(self) -> None:
        accept = _result("accept", {"answers": [{"selected": "SQLite"}]})
        payload, _ = _run([_Q_DB], accept)
        self.assertEqual(payload["answers"][0]["selected"], ["SQLite"])

    def test_other_text_merged_into_selection(self) -> None:
        accept = _result("accept", {"answers": [{"selected": [], "other_text": "MySQL"}]})
        payload, _ = _run([_Q_DB], accept)
        answer = payload["answers"][0]
        self.assertEqual(answer["selected"], ["MySQL"])
        self.assertEqual(answer["other_text"], "MySQL")

    def test_missing_answer_yields_empty_selection(self) -> None:
        # Fewer answers than questions → the tail question comes back empty.
        accept = _result("accept", {"answers": [{"selected": ["Postgres"]}]})
        payload, _ = _run([_Q_DB, _Q_FEATURES], accept)
        self.assertEqual(payload["answers"][1]["selected"], [])

    def test_decline_returns_note_and_no_answers(self) -> None:
        declined = _result("decline", None)
        payload, _ = _run([_Q_DB], declined)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["answers"], [])
        self.assertIn("note", payload)

    def test_empty_questions_is_structured_error(self) -> None:
        payload, session = _run([], _result("accept", {"answers": []}))
        self.assertEqual(payload["status"], "error")
        # Never even reaches the elicitation channel.
        self.assertEqual(session.calls, [])

    def test_escape_hatch_options_are_stripped(self) -> None:
        # The frontend always shows a free-text "Other" field, so options that only
        # restate it are dropped before the question reaches the user.
        question = {
            "question": "Which database?",
            "header": "Database",
            "options": [
                {"label": "Postgres"},
                {"label": "Other (type your own answer)"},
                {"label": "Request changes"},
                {"label": "Add precisions…"},
                {"label": "None of these"},
                {"label": "SQLite"},
            ],
        }
        _, session = _run([question], _result("accept", {"answers": []}))
        sent = session.calls[0][1]["x_mimir"]["questions"][0]["options"]
        self.assertEqual([o["label"] for o in sent], ["Postgres", "SQLite"])

    def test_substantive_options_containing_those_words_are_kept(self) -> None:
        # Only whole-label matches are filtered — real choices survive.
        question = {
            "question": "What next?",
            "header": "Next",
            "options": [
                {"label": "Request changes from the reviewer"},
                {"label": "Add details to the changelog"},
                {"label": "Custom kernel"},
            ],
        }
        _, session = _run([question], _result("accept", {"answers": []}))
        sent = session.calls[0][1]["x_mimir"]["questions"][0]["options"]
        self.assertEqual(len(sent), 3)

    def test_question_with_only_escape_hatch_options_is_dropped(self) -> None:
        payload, session = _run(
            [{"question": "Anything to add?", "header": "Input",
              "options": [{"label": "Other"}, {"label": "Something else"}]}],
            _result("accept", {"answers": []}),
        )
        self.assertEqual(payload["status"], "error")
        self.assertEqual(session.calls, [])

    def test_question_without_options_is_dropped(self) -> None:
        payload, _ = _run(
            [{"question": "No options here", "header": "Bad"}],
            _result("accept", {"answers": []}),
        )
        self.assertEqual(payload["status"], "error")


if __name__ == "__main__":
    unittest.main()
