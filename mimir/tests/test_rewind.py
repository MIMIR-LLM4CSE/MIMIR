"""Editing a sent message: the checkpoints a rewind reads, and the conversation it cuts.

Pinned here: the checkpoint store (``context.checkpoints``) keeps the content a file had
before the message that led to its first write, across every later turn; it leaves a
file someone else changed since; the conversation goes back to its length before the
message; and a shell write's original is recovered from git or from the scan.
"""
import asyncio
import os
import subprocess
import tempfile
import threading
import unittest

from mimir.client.context.checkpoints import ABSENT, Checkpoints
from mimir.client.core.turn_state import TurnState
from mimir.client.ui.ws.worker.worker import AgentWorker
from mimir.client.session.conversation import Conversations
from mimir.client.session.store import SessionStore
from mimir.client.tool_execution import bash_effect
from mimir.tests.support.state import pin_state_dir


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state = os.path.join(tmp.name, "state")
        self.work = os.path.join(tmp.name, "work")
        os.makedirs(self.work)
        pin_state_dir(self, self.state)
        self.cps = Checkpoints.of("s1")

    def _open(self, message_id: str, **lengths) -> None:
        self.cps.open(message_id, full_len=lengths.get("full_len", 0),
                      history_len=lengths.get("history_len", 0), window_epoch=0)

    def _agent_writes(self, path: str, text: str) -> None:
        self.cps.note_before(path)
        _write(path, text)
        self.cps.note_written(path)


class CheckpointTests(_Case):
    def test_the_first_content_before_the_message_is_what_comes_back(self):
        a = os.path.join(self.work, "a.txt")
        _write(a, "v0")
        self._open("m1")
        self._agent_writes(a, "v1")
        self._open("m2")
        self._agent_writes(a, "v2")
        plan = self.cps.plan("m1")
        self.assertEqual(len(plan.restore), 1)
        self.assertTrue(self.cps.restore(*plan.restore[0]))
        self.assertEqual(_read(a), "v0")
        # From the second message, the first turn's write stays.
        _write(a, "v2")
        path, digest = self.cps.plan("m2").restore[0]
        self.cps.restore(path, digest)
        self.assertEqual(_read(a), "v1")

    def test_a_created_file_is_removed(self):
        b = os.path.join(self.work, "b.txt")
        self._open("m1")
        self._agent_writes(b, "new")
        plan = self.cps.plan("m1")
        self.assertEqual(plan.restore, [(b, ABSENT)])
        self.assertEqual(plan.preview()["files"], [{"path": b, "action": "delete"}])
        self.cps.restore(*plan.restore[0])
        self.assertFalse(os.path.exists(b))

    def test_a_file_changed_by_someone_else_since_is_left(self):
        a = os.path.join(self.work, "a.txt")
        _write(a, "v0")
        self._open("m1")
        self._agent_writes(a, "v1")
        _write(a, "the user's edit")
        plan = self.cps.plan("m1")
        self.assertEqual(plan.restore, [])
        self.assertEqual(plan.conflicts, [a])

    def test_an_unknown_original_is_named(self):
        self._open("m1")
        self.cps.note_unknown(os.path.join(self.work, "big.bin"))
        self.assertEqual(self.cps.plan("m1").unrestorable,
                         [os.path.join(self.work, "big.bin")])

    def test_runs_since_the_message_are_in_the_plan(self):
        self._open("m1")
        self.cps.note_run({"job_key": "j1"})
        self._open("m2")
        self.cps.note_run({"job_key": "j2"})
        self.assertEqual([r["job_key"] for r in self.cps.plan("m1").runs], ["j1", "j2"])
        self.assertEqual([r["job_key"] for r in self.cps.plan("m2").runs], ["j2"])

    def test_dropping_forgets_later_checkpoints_and_survives_a_reload(self):
        self._open("m1")
        self._open("m2")
        self._open("m3")
        self.cps.drop_from("m2")
        self.assertEqual(self.cps.ids(), ["m1"])
        Checkpoints._open.clear()
        self.assertEqual(Checkpoints.of("s1").ids(), ["m1"])

    def test_nothing_is_kept_without_a_checkpoint(self):
        a = os.path.join(self.work, "a.txt")
        self.cps.note_before(a)
        self.assertIsNone(self.cps.plan("m1"))


class ConversationRewindTests(_Case):
    def setUp(self) -> None:
        super().setUp()
        self.conversations = Conversations(SessionStore())
        self.conv = self.conversations.new("s1")

    def _turn(self, message_id: str, text: str, answer: str) -> None:
        self.conv.open_checkpoint(message_id, {"k": message_id})
        self.conv.add_user_message(text, show=True, message_id=message_id)
        self.conv.commit_answer({"text": answer}, {})

    def test_the_conversation_goes_back_to_before_the_message(self):
        self._turn("m1", "first", "a1")
        self._turn("m2", "second", "a2")
        self.conv.rewind(self.cps.plan("m2").checkpoint)
        self.assertEqual([m["content"] for m in self.conv.history], ["first", "a1"])
        self.assertEqual([m["content"] for m in self.conv.history_full], ["first", "a1"])
        self.assertEqual([m["text"] for m in self.conv.display], ["first", "a1"])
        self.assertEqual(self.conv.record.carry_context, {"k": "m2"})

    def test_a_window_rewritten_since_is_rebuilt_from_the_record(self):
        self._turn("m1", "first", "a1")
        self._turn("m2", "second", "a2")
        del self.conv.history[0]          # trimmed from the front
        self.conv.window_rewritten()
        self.conv.rewind(self.cps.plan("m2").checkpoint)
        self.assertEqual([m["content"] for m in self.conv.history], ["first", "a1"])

    def test_the_bubble_carries_its_id(self):
        self._turn("m1", "first", "a1")
        self.assertEqual(self.conv.display[0]["id"], "m1")
        self.assertTrue(self.conv.display[0]["opensTurn"])


class _Agent:
    def __init__(self, root: str) -> None:
        self.workspace_root = root
        self.tool_caps: dict = {}

    def normalize_workspace_path(self, path):
        return path


def _git(root: str, *args: str) -> None:
    subprocess.run(("git", *args), cwd=root, check=True, capture_output=True)


class ShellWriteTests(_Case):
    def _probe(self, command: str):
        from unittest import mock
        with mock.patch.object(bash_effect, "shell_command_of", return_value=command):
            return bash_effect.capture(_Agent(self.work), "bash", {})

    def test_in_a_repository_clean_dirty_and_created_files_come_back(self):
        _git(self.work, "init", "-q")
        _write(os.path.join(self.work, "clean.txt"), "c0\n")
        _write(os.path.join(self.work, "dirty.txt"), "d0\n")
        _git(self.work, "add", ".")
        _git(self.work, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
        _write(os.path.join(self.work, "dirty.txt"), "d1\n")
        self._open("m1")
        probe = self._probe("sed -i s/0/9/ clean.txt dirty.txt && touch new.txt")
        _write(os.path.join(self.work, "clean.txt"), "c9\n")
        _write(os.path.join(self.work, "dirty.txt"), "d9\n")
        _write(os.path.join(self.work, "new.txt"), "")
        bash_effect.keep_originals(probe, self.cps)
        plan = self.cps.plan("m1")
        for path, digest in plan.restore:
            self.cps.restore(path, digest)
        self.assertEqual(_read(os.path.join(self.work, "clean.txt")), "c0\n")
        self.assertEqual(_read(os.path.join(self.work, "dirty.txt")), "d1\n")
        self.assertFalse(os.path.exists(os.path.join(self.work, "new.txt")))

    def test_outside_a_repository_a_scanned_file_comes_back(self):
        src = os.path.join(self.work, "solver.py")
        _write(src, "x = 1\n")
        self._open("m1")
        probe = self._probe(f"sed -i s/1/2/ {src}")
        _write(src, "x = 2\n")
        bash_effect.keep_originals(probe, self.cps)
        path, digest = self.cps.plan("m1").restore[0]
        self.cps.restore(path, digest)
        self.assertEqual(_read(src), "x = 1\n")


class _StuckAgent:
    """A turn that works until it is cancelled."""

    def __init__(self) -> None:
        self.turn = TurnState()
        self.turn.cancel_flag = threading.Event()
        self.turn.deferred_prompts = []
        self.turn.deferred_turn = None
        self.turn.last_full_messages = None
        self.turn.last_turn_start = None
        self.context_mode = "flat"
        self.streaming = False
        self.thinking = False

    async def run(self, **_kw) -> str:
        await asyncio.Event().wait()
        return "never"


def _stuck_worker() -> AgentWorker:
    w = AgentWorker("test-model", start=False)
    w.session_id = "s1"
    w._agent = _StuckAgent()
    w.human._defer = set()
    w.review.todos = lambda: []
    w.usage.count_tokens = lambda _text: 0
    return w


class StoppingTheTurnTests(unittest.TestCase):
    """What a rewind waits on: the cancelled turn's answer written, the loop alive."""

    async def _until(self, predicate) -> None:
        for _ in range(500):
            if predicate():
                return
            await asyncio.sleep(0.01)
        self.fail("never happened")

    def test_the_turn_is_open_until_its_answer_is_on_the_outbox(self):
        async def scenario():
            w = _stuck_worker()
            w._loop = asyncio.get_running_loop()
            loop_task = asyncio.create_task(w._query_loop())
            await asyncio.sleep(0)
            w._query_q.put({"text": "go", "history": [], "session_id": "s1"})
            w._query_wake.set()
            await self._until(lambda: w._current_task not in (None, w._loop_task))
            self.assertTrue(w.turn_open())
            w.cancel()
            await self._until(lambda: not w.turn_open())
            answers = [ev for ev in list(w.out_q.queue) if ev.get("type") == "answer"]
            self.assertEqual(len(answers), 1)
            self.assertTrue(answers[0]["cancelled"])
            self.assertFalse(loop_task.done())
            w._query_q.put(None)
            w._query_wake.set()
            await loop_task
        asyncio.run(scenario())

    def test_a_cancel_while_a_turn_is_set_up_leaves_the_loop_running(self):
        async def scenario():
            w = _stuck_worker()
            w._loop = asyncio.get_running_loop()
            loop_task = asyncio.create_task(w._query_loop())
            await asyncio.sleep(0)
            w._current_task = w._loop_task      # the moment before the agent's task exists
            self.assertFalse(w.cancel())
            await asyncio.sleep(0.01)
            self.assertFalse(loop_task.done())
            w._current_task = None
            w._query_q.put(None)
            w._query_wake.set()
            await loop_task
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
