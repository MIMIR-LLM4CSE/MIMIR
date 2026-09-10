"""What survives a reload — the rendered chat, and the context behind it.

Two losses used to happen on every reconnect. The session file only ever held the text
bubbles, so tool rows, reasoning panels and diff cards came back stripped to prose; and
`llm_history` is the *window*, mutated in place by the budget trim, so the turns it drops
left no record anywhere and a resumed session started amnesiac. These tests pin the two
halves of the fix: a client transcript is stored verbatim (with the guards that stop it
overwriting the wrong session or a longer history), and the untrimmed record is kept
beside the window.

What a load hands the model is the third thing pinned here. A front-trimmed tail resumes
from the record, because its prefix was summarized nowhere; a window that already
carries a compaction summary resumes from itself, because reloading the record over it
throws that summary away and starts the session back at its pre-compaction size.
"""
import concurrent.futures
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

from mimir.client.query_engine.history import (
    carries_compaction_summary, compacted_exchanges, compaction_summary_message,
)
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
    def __init__(self, turn_start=None):
        self.active_session_id = None
        self._turn_start = turn_start

    def last_turn_start(self):
        return self._turn_start

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


class _FakeWS:
    def __init__(self):
        self.sent: list[str] = []

    async def send(self, payload):
        self.sent.append(payload)


class PreQueryCompactionTests(unittest.IsolatedAsyncioTestCase):
    """The window is summarized before it is amputated.

    The CLI compacts before each query; the WS path only ever front-trimmed, because
    summarizing means an LLM call and that call must not run on the event loop. It runs
    on the worker thread now, and dropping turns is the fallback.
    """

    @staticmethod
    def _sess_with_history(n_middle):
        sess = _session()
        sess.ws = _FakeWS()
        sess.transcript = []
        sess.history = [{"role": "user", "content": "opening question"}]
        sess.history += [{"role": "assistant", "content": f"step {i}"} for i in range(n_middle)]
        sess.history += [{"role": "user", "content": "last"},
                         {"role": "assistant", "content": "answer"},
                         {"role": "user", "content": "follow-up"},
                         {"role": "assistant", "content": "ok"}]
        return sess

    async def test_the_middle_is_summarized_and_the_head_and_tail_survive(self):
        sess = self._sess_with_history(5)
        summary = [{"role": "assistant", "content": "[Context summary] what happened"}]
        sess.worker.compact_middle = lambda middle: _done_future(summary)

        self.assertTrue(await sess._compact_history())
        self.assertEqual(sess.history[0]["content"], "opening question")
        self.assertEqual(sess.history[1], summary[0])
        self.assertEqual([m["content"] for m in sess.history[-4:]],
                         ["last", "answer", "follow-up", "ok"])
        self.assertEqual(sess.transcript[-1]["type"], "context_compact")

    async def test_a_failed_summarization_leaves_the_history_untouched(self):
        # compact_messages returns its input unchanged when the backend call fails —
        # the caller must read that as "no compaction" and fall back to trimming.
        sess = self._sess_with_history(5)
        before = list(sess.history)
        sess.worker.compact_middle = lambda middle: _done_future(middle)

        self.assertFalse(await sess._compact_history())
        self.assertEqual(sess.history, before)

    async def test_too_short_a_middle_is_not_worth_an_llm_call(self):
        sess = self._sess_with_history(1)
        called = []
        sess.worker.compact_middle = lambda middle: called.append(middle) or _done_future([])

        self.assertFalse(await sess._compact_history())
        self.assertEqual(called, [])
        self.assertEqual(sess.ws.sent, [])  # not even an announcement


def _done_future(value):
    fut = concurrent.futures.Future()
    fut.set_result(value)
    return fut


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

    The agent hands back its whole transcript, but the loop is free to rewrite the
    prefix it inherited while the turn runs — and that prefix is exactly what the
    record exists to preserve, so only the turn's own messages may be taken from it.

    These drive `_Session._turn_messages` itself. They used to carry a copy of its
    arithmetic instead, which is why they kept passing while the real path was cutting
    an archive to ribbons: a test that restates the code cannot disagree with it.
    """

    TURN = [
        {"role": "user", "content": "old"},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]

    @staticmethod
    def _sess(turn_start, submitted_len):
        sess = _session()
        sess.worker = _FakeWorker(turn_start=turn_start)
        sess._submitted_len = submitted_len
        return sess

    def test_only_the_turn_takes_its_place_in_the_record(self):
        added = self._sess(2, 2)._turn_messages(self.TURN)
        self.assertEqual(len(added), 3)
        self.assertEqual(added[0]["tool_calls"][0]["id"], "c1")

    def test_the_boundary_the_loop_gives_beats_the_length_we_submitted(self):
        """The bug, in one assertion.

        We submitted 2 messages; the in-turn budget pass rewrote the prefix and the
        turn now starts at 1. Trusting our own count re-archives message 1 — already
        in the record — and, in the session this came from, cut into an assistant↔tool
        pair and left the tool result orphaned.
        """
        added = self._sess(1, 2)._turn_messages(self.TURN)
        self.assertEqual([m["content"] for m in added][:1], ["new"])
        self.assertEqual(len(added), 4)

    def test_a_stale_boundary_is_not_used_when_the_loop_supplies_one(self):
        """The mirror case: our count is too large, and would drop the turn's own work."""
        added = self._sess(2, 4)._turn_messages(self.TURN)
        self.assertEqual(len(added), 3)

    def test_without_a_boundary_the_submitted_length_still_serves(self):
        added = self._sess(None, 2)._turn_messages(self.TURN)
        self.assertEqual(len(added), 3)

    def test_a_compacted_turn_still_contributes_its_answer(self):
        """The loop replaced the middle with a handoff note — the answer is what is left."""
        full = [{"role": "assistant", "content": "done"}]
        self.assertEqual(self._sess(None, 4)._turn_messages(full),
                         [{"role": "assistant", "content": "done"}])

    def test_a_turn_that_added_nothing_is_recorded_by_its_answer(self):
        """The boundary points past the end — an empty slice must not erase the turn."""
        self.assertEqual(self._sess(5, 5)._turn_messages(self.TURN),
                         [{"role": "assistant", "content": "done"}])


class TodoRestoreTests(unittest.IsolatedAsyncioTestCase):
    """A reconnect must not overwrite a checklist newer than the snapshot it holds.

    `session.todos` is captured at autosave time, so a query cut short by a dropped
    connection leaves the store holding the list as it stood at the last save while the
    file on disk holds the one the model has written since. Restoring unconditionally
    destroys the live list and hands the model a finished one. Observed in session
    ba8eee87: a disconnect during an optimisation task restored the *previous* task's
    five completed steps, so the next query ran with a plan of record reading
    "0 pending" and the completion gate had nothing left to block on.
    """

    def _sess_with_todo_file(self, body: str | None, mtime_offset: float = 0.0):
        """A session whose todo file holds *body* (None = no file), with a set mtime."""
        d = tempfile.mkdtemp()
        todo_file = os.path.join(d, "todo_list.md")
        if body is not None:
            with open(todo_file, "w", encoding="utf-8") as fh:
                fh.write(body)
            base = datetime.now(timezone.utc).timestamp() + mtime_offset
            os.utime(todo_file, (base, base))
        sess = _session()
        sess.ws = _FakeWS()
        sess._emit_context_usage = mock.AsyncMock()
        return sess, todo_file

    async def _load(self, sess, todo_file, todos, updated_at):
        sess.store.saved["s1"] = FullSession(
            id="s1", title="t", created_at="x", updated_at=updated_at, todos=todos,
        )
        with mock.patch("mimir.client.ui.ws.ws_session._write_active_session"), \
             mock.patch("mimir.client.ui.ws.ws_session._todo_file_for_session",
                        lambda _sid: todo_file):
            await sess._load_session("s1")

    SNAPSHOT = [{"text": "old step", "done": True}]
    LIVE = "- [ ] the step the model is actually working on\n"

    async def test_a_newer_file_on_disk_is_kept(self) -> None:
        # The regression: the file is 60s newer than the snapshot — it is the live list.
        now = datetime.now(timezone.utc)
        sess, todo_file = self._sess_with_todo_file(self.LIVE, mtime_offset=+60)
        await self._load(sess, todo_file, self.SNAPSHOT, now.isoformat())
        with open(todo_file, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), self.LIVE)  # untouched

    async def test_keeping_it_is_announced(self) -> None:
        # A silently skipped restore is as hard to diagnose as the overwrite it avoids.
        now = datetime.now(timezone.utc)
        sess, todo_file = self._sess_with_todo_file(self.LIVE, mtime_offset=+60)
        await self._load(sess, todo_file, self.SNAPSHOT, now.isoformat())
        self.assertTrue(
            any("newer" in payload for payload in sess.ws.sent),
            f"no status about keeping the newer checklist: {sess.ws.sent}",
        )

    async def test_an_older_file_is_still_restored(self) -> None:
        now = datetime.now(timezone.utc)
        sess, todo_file = self._sess_with_todo_file(self.LIVE, mtime_offset=-60)
        await self._load(sess, todo_file, self.SNAPSHOT, now.isoformat())
        with open(todo_file, encoding="utf-8") as fh:
            self.assertIn("old step", fh.read())

    async def test_a_missing_file_is_restored(self) -> None:
        now = datetime.now(timezone.utc)
        sess, todo_file = self._sess_with_todo_file(None)
        await self._load(sess, todo_file, self.SNAPSHOT, now.isoformat())
        with open(todo_file, encoding="utf-8") as fh:
            self.assertIn("old step", fh.read())

    async def test_an_unreadable_timestamp_restores(self) -> None:
        # The guard stops one specific loss; it must not become a second way to fail.
        sess, todo_file = self._sess_with_todo_file(self.LIVE, mtime_offset=+60)
        await self._load(sess, todo_file, self.SNAPSHOT, "not-a-timestamp")
        with open(todo_file, encoding="utf-8") as fh:
            self.assertIn("old step", fh.read())

    async def test_the_deps_sidecar_follows_the_list(self) -> None:
        # Restoring deps over a list that was kept yields dependencies pointing at
        # steps that are not there.
        now = datetime.now(timezone.utc)
        sess, todo_file = self._sess_with_todo_file(self.LIVE, mtime_offset=+60)
        deps_file = os.path.join(os.path.dirname(todo_file), "todo_deps.json")
        sess.store.saved["s1"] = FullSession(
            id="s1", title="t", created_at="x", updated_at=now.isoformat(),
            todos=self.SNAPSHOT, todo_deps=[[], [0]],
        )
        with mock.patch("mimir.client.ui.ws.ws_session._write_active_session"), \
             mock.patch("mimir.client.ui.ws.ws_session._todo_file_for_session",
                        lambda _sid: todo_file):
            await sess._load_session("s1")
        self.assertFalse(os.path.exists(deps_file))


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

    async def test_the_window_does_not_share_message_objects_with_the_archive(self):
        """The budgeting pass rewrites `content` and `tool_calls` in place.

        Sharing the dicts let those writes reach straight into the untrimmed
        record, which is the one thing a resume is supposed to be able to trust.
        """
        sess = _session()
        sess.store.saved["s1"] = FullSession(
            id="s1", title="t", created_at="x", updated_at="y",
            llm_history_full=[{"role": "user", "content": f"turn {i}"} for i in range(4)],
        )
        sess.ws = mock.AsyncMock()
        sess._emit_context_usage = mock.AsyncMock()
        with mock.patch("mimir.client.ui.ws.ws_session._write_active_session"):
            await sess._load_session("s1")
        sess.history[0]["content"] = "…[truncated]…"
        self.assertEqual(sess.history_full[0]["content"], "turn 0")

    async def test_a_compacted_window_is_what_a_resume_reloads(self):
        """A summary already stands in for what was cut — reloading the record undoes it.

        The bug this pins: a long session compacted its 1346-message history down to a
        162-message window, and the reload handed the model the 1346 again. The context
        bar sat full before the user had typed, and the first turn spent an LLM call
        re-compacting what was already compacted.
        """
        sess = _session()
        window = [
            {"role": "user", "content": "port the build to the new target"},
            compaction_summary_message(257, "# Handoff Note\n\nthe arch is set in cmake/"),
            {"role": "assistant", "content": "picking it back up"},
        ]
        sess.store.saved["s1"] = FullSession(
            id="s1", title="t", created_at="x", updated_at="y",
            llm_history=window,
            llm_history_full=[{"role": "user", "content": f"turn {i}"} for i in range(600)],
        )
        sess.ws = mock.AsyncMock()
        sess._emit_context_usage = mock.AsyncMock()
        with mock.patch("mimir.client.ui.ws.ws_session._write_active_session"):
            await sess._load_session("s1")
        self.assertEqual(len(sess.history), 3)
        self.assertIn("Handoff Note", sess.history[1]["content"])
        # The record is still kept whole beside it: nothing is lost, it is just not
        # what the model is handed.
        self.assertEqual(len(sess.history_full), 600)
        self.assertEqual(sess._submitted_len, 3)

    async def test_a_compacted_window_is_not_shared_with_the_archive_either(self):
        sess = _session()
        sess.store.saved["s1"] = FullSession(
            id="s1", title="t", created_at="x", updated_at="y",
            llm_history=[compaction_summary_message(9, "note")],
            llm_history_full=[compaction_summary_message(9, "note"),
                              {"role": "user", "content": "after"}],
        )
        sess.ws = mock.AsyncMock()
        sess._emit_context_usage = mock.AsyncMock()
        with mock.patch("mimir.client.ui.ws.ws_session._write_active_session"):
            await sess._load_session("s1")
        sess.history[0]["content"] = "…[truncated]…"
        self.assertIn("note", sess.history_full[0]["content"])

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


class CompactionMarkerTests(unittest.TestCase):
    """The marker is how a resume tells a compacted window from a front-trimmed tail.

    It is matched by prefix, so the builder and the predicate have to agree exactly.
    They lived apart once and disagreed on a capital letter — which would have read as
    "never compacted" and quietly reloaded the whole record.
    """

    def test_what_the_builder_writes_is_what_the_predicate_reads(self):
        self.assertTrue(carries_compaction_summary(
            [compaction_summary_message(12, "note")]))

    def test_the_summary_is_found_behind_the_task_that_is_kept_ahead_of_it(self):
        self.assertTrue(carries_compaction_summary([
            {"role": "user", "content": "the task"},
            compaction_summary_message(12, "note"),
        ]))

    def test_a_front_trimmed_tail_carries_no_summary(self):
        self.assertFalse(carries_compaction_summary([
            {"role": "user", "content": "turn 8"},
            {"role": "assistant", "content": "turn 9"},
        ]))

    def test_an_empty_window_carries_no_summary(self):
        self.assertFalse(carries_compaction_summary([]))

    def test_a_summary_buried_mid_history_is_not_this_window_opening(self):
        """An older compaction a later front-trim reduced to a tail again.

        The window no longer opens on the note, so what precedes the note inside it
        was never summarized — the record is still the faithful resume.
        """
        self.assertFalse(carries_compaction_summary([
            {"role": "assistant", "content": f"turn {i}"} for i in range(6)
        ] + [compaction_summary_message(3, "note")]))

    def test_the_old_capital_s_spelling_is_still_recognised(self):
        """Windows the CLI compacted before the two producers merged are still on disk."""
        self.assertTrue(carries_compaction_summary([
            {"role": "assistant",
             "content": "[Context Summary — 12 prior exchanges compacted]\n\nnote"},
        ]))

    def test_a_user_message_quoting_the_marker_is_not_a_summary(self):
        self.assertFalse(carries_compaction_summary([
            {"role": "user", "content": "[Context summary — why did it say that?"},
        ]))


class CompactedExchangeCountTests(unittest.TestCase):
    """What the marker claims the summary stands for.

    The model reads this number to judge how much of its own past it can no longer
    see. Counting the slice's messages made every pass after the first announce less
    than the one before it, because a summary absorbing hundreds of exchanges counts
    as a single message.
    """

    @staticmethod
    def _raw(n):
        return [{"role": "assistant", "content": f"m{i}"} for i in range(n)]

    def test_a_first_pass_counts_exchanges(self):
        self.assertEqual(compacted_exchanges(self._raw(16)), 8)

    def test_a_second_pass_adds_what_the_first_already_absorbed(self):
        middle = [compaction_summary_message(8, "note")] + self._raw(8)
        self.assertEqual(compacted_exchanges(middle), 12)   # 8 absorbed + 4 fresh

    def test_the_count_never_shrinks_across_passes(self):
        """The regression, stated directly: pass N+1 cannot claim less than pass N."""
        counts = []
        middle = self._raw(12)
        for _ in range(4):
            n = compacted_exchanges(middle)
            counts.append(n)
            middle = [compaction_summary_message(n, "note")] + self._raw(12)
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts, [6, 12, 18, 24])

    def test_an_older_capital_s_marker_still_contributes_its_count(self):
        middle = [{"role": "assistant",
                   "content": "[Context Summary — 30 prior exchanges compacted]\n\nx"}]
        self.assertEqual(compacted_exchanges(middle), 30)

    def test_a_summary_with_no_readable_count_is_not_read_as_conversation(self):
        """It is still a summary: worth 0 exchanges, never one raw message."""
        self.assertEqual(compacted_exchanges(
            [{"role": "assistant", "content": "[Context summary] lost its count"}]), 0)

    def test_the_singular_marker_reads_back(self):
        self.assertEqual(compacted_exchanges([compaction_summary_message(1, "n")]), 1)


if __name__ == "__main__":
    unittest.main()
