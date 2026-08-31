"""What survives a reload — the rendered chat, and the context behind it.

Two losses used to happen on every reconnect. The session file only ever held the text
bubbles, so tool rows, reasoning panels and diff cards came back stripped to prose; and
`llm_history` is the *window*, mutated in place by the budget trim, so the turns it drops
left no record anywhere and a resumed session started amnesiac. These tests pin the two
halves of the fix: a client transcript is stored verbatim (with the guards that stop it
overwriting the wrong session or a longer history), and the untrimmed record is kept
beside the window and is what a load restores.
"""
import os
import tempfile
import unittest
from unittest import mock

from mimir.client.ui.ws.session_store import FullSession, SessionStore
from mimir.client.ui.ws.ws_session import _Session


RICH_MESSAGES = [
    {"id": "m1", "role": "user", "kind": "text", "text": "profile the solver"},
    {"id": "m2", "role": "agent", "kind": "thinking", "thinking": "where does it spend time",
     "thinkingDurationMs": 4200, "thinkingTokens": 310},
    {"id": "m3", "role": "agent", "kind": "tools", "tools": [
        {"id": "c1", "name": "bash", "icon": "💻", "label": "Running: pytest",
         "detail": "-q", "status": "ok", "summary": "12 passed", "durationMs": 8100,
         "startedAt": 1_700_000_000_000,
         "exec": {"command": "pytest -q", "stdout": "12 passed", "stderr": "",
                  "returncode": 0}},
        {"id": "c1:1", "name": "grep", "icon": "🔍", "label": "Searching", "detail": "solve",
         "status": "ok", "startedAt": 1_700_000_000_001,
         "parentId": "c1", "origin": "explore #1"},
    ]},
    {"id": "m4", "role": "agent", "kind": "editing",
     "diffs": [{"file": "solver.py", "patch": "@@ -1 +1 @@", "is_new": False}]},
    {"id": "m5", "role": "agent", "kind": "text", "text": "The hot loop is in solve()."},
]


class _FakeStore:
    """In-memory stand-in with the three methods _autosave_session calls."""

    def __init__(self):
        self.saved: dict[str, FullSession] = {}

    def session_exists(self, sid):
        return sid in self.saved

    def load_session(self, sid):
        return self.saved[sid]

    def save_session(self, session):
        self.saved[session.id] = session


class _FakeWorker:
    def __init__(self):
        self.active_session_id = None

    def export_agent_state(self):
        return {"carry_context": {}}

    def _load_todos(self):
        return []

    def load_agent_state(self, state):
        pass

    def reset_session_guards(self):
        pass


def _session(active="s1"):
    sess = object.__new__(_Session)
    sess.worker = _FakeWorker()
    sess.store = _FakeStore()
    sess._active_session_id = active
    sess._unsaved_session_meta = None
    sess._display_messages = []
    sess.history = []
    sess.history_full = []
    sess.transcript = mock.Mock()
    return sess


class RichTranscriptRoundTripTests(unittest.TestCase):
    """A stored transcript must come back with every block it went in with."""

    def test_tool_thinking_and_diff_blocks_survive_the_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("mimir.client.ui.ws.session_store._sessions_dir",
                            return_value=tmp):
                store = SessionStore()
                session = store.new_session()
                session.display_messages = RICH_MESSAGES
                store.save_session(session)
                back = store.load_session(session.id)
        self.assertEqual(back.display_messages, RICH_MESSAGES)
        tools = back.display_messages[2]["tools"]
        self.assertEqual(tools[0]["exec"]["returncode"], 0)
        self.assertEqual(tools[1]["parentId"], "c1")  # sub-agent nesting kept

    def test_a_session_written_before_the_full_history_existed_still_loads(self):
        old = FullSession.from_dict({
            "id": "s1", "title": "t", "created_at": "x", "updated_at": "y",
            "llm_history": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(old.llm_history_full, [])


class TranscriptHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_client_transcript_replaces_the_text_only_messages(self):
        sess = _session()
        sess._display_messages = [{"role": "user", "kind": "text", "text": "profile the solver"}]
        await sess._handle_transcript({"session_id": "s1", "messages": RICH_MESSAGES})
        self.assertEqual(sess._display_messages, RICH_MESSAGES)
        self.assertEqual(sess.store.saved["s1"].display_messages, RICH_MESSAGES)

    async def test_a_transcript_for_another_session_is_ignored(self):
        """It would land on the session the user just switched to."""
        sess = _session(active="s1")
        await sess._handle_transcript({"session_id": "s2", "messages": RICH_MESSAGES})
        self.assertEqual(sess._display_messages, [])

    async def test_a_shorter_transcript_never_blanks_a_stored_history(self):
        """A webview that just opened on an empty view must not erase the session."""
        sess = _session()
        sess._display_messages = list(RICH_MESSAGES)
        await sess._handle_transcript({"session_id": "s1", "messages": []})
        self.assertEqual(sess._display_messages, RICH_MESSAGES)

    async def test_only_prose_counts_when_comparing_lengths(self):
        """Tool rows are re-derived per turn; they must not gate the comparison."""
        sess = _session()
        sess._display_messages = [
            {"role": "user", "kind": "text", "text": "a"},
            {"role": "agent", "kind": "tools", "tools": []},
            {"role": "agent", "kind": "tools", "tools": []},
        ]
        incoming = [
            {"role": "user", "kind": "text", "text": "a"},
            {"role": "agent", "kind": "text", "text": "done"},
        ]
        await sess._handle_transcript({"session_id": "s1", "messages": incoming})
        self.assertEqual(sess._display_messages, incoming)


class UntrimmedHistoryTests(unittest.TestCase):
    def test_the_record_keeps_what_the_window_drops(self):
        sess = _session()
        for i in range(4):
            msg = {"role": "user", "content": f"turn {i}"}
            sess.history.append(msg)
            sess.history_full.append(dict(msg))
        # What the pre-query front-trim does to the window, and only to it.
        sess.history.pop(0)
        sess.history.pop(0)
        sess._autosave_session(list(sess._display_messages))
        saved = sess.store.saved["s1"]
        self.assertEqual(len(saved.llm_history), 2)
        self.assertEqual(len(saved.llm_history_full), 4)
        self.assertEqual(saved.llm_history_full[0]["content"], "turn 0")


class AnswerDeltaTests(unittest.TestCase):
    """What the record takes from a finished turn.

    The agent hands back its whole transcript, but the loop is free to trim or compact
    the prefix it inherited — and that prefix is exactly what the record exists to
    preserve, so only the turn's own messages may be taken from it.
    """

    @staticmethod
    def _added(submitted_len: int, full: list[dict]) -> list[dict]:
        return full[submitted_len:] if len(full) > submitted_len else full[-1:]

    def test_only_the_turn_takes_its_place_in_the_record(self):
        full = [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]
        added = self._added(2, full)
        self.assertEqual(len(added), 3)
        self.assertEqual(added[0]["tool_calls"][0]["id"], "c1")

    def test_a_compacted_turn_still_contributes_its_answer(self):
        """The loop replaced the middle with a handoff note — the answer is what is left."""
        full = [{"role": "assistant", "content": "done"}]
        self.assertEqual(self._added(4, full), [{"role": "assistant", "content": "done"}])


class ResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_loading_restores_the_untrimmed_record_not_the_window(self):
        sess = _session()
        stored = FullSession(
            id="s1", title="t", created_at="x", updated_at="y",
            llm_history=[{"role": "user", "content": "turn 3"}],
            llm_history_full=[{"role": "user", "content": f"turn {i}"} for i in range(4)],
        )
        sess.store.saved["s1"] = stored
        sess.ws = mock.AsyncMock()
        sess._emit_context_usage = mock.AsyncMock()
        with mock.patch("mimir.client.ui.ws.ws_session._write_active_session"):
            await sess._load_session("s1")
        self.assertEqual(len(sess.history), 4)
        self.assertEqual(len(sess.history_full), 4)

    async def test_an_older_session_falls_back_to_the_window_it_saved(self):
        sess = _session()
        sess.store.saved["s1"] = FullSession(
            id="s1", title="t", created_at="x", updated_at="y",
            llm_history=[{"role": "user", "content": "only this"}],
        )
        sess.ws = mock.AsyncMock()
        sess._emit_context_usage = mock.AsyncMock()
        with mock.patch("mimir.client.ui.ws.ws_session._write_active_session"):
            await sess._load_session("s1")
        self.assertEqual(sess.history_full, [{"role": "user", "content": "only this"}])


if __name__ == "__main__":
    unittest.main()
