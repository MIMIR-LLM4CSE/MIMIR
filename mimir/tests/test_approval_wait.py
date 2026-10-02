"""What each kind of unanswered prompt does to a parked agent.

An approval waits indefinitely: it must never resume on its own and answer for the
user, so the only ways out are an answer and the cancel flag (the Stop button). A
clarification question carries a wall instead — nothing bad happens at the end of it,
the agent goes on with the option it recommended — and the wait reports that ending
distinctly so the card can be taken off the screen.
"""

import queue as _queue
import threading
import time
import unittest

from mimir.client.ui.ws.ws_worker import TIMED_OUT, _AgentWorker


class _StubAgent:
    def __init__(self) -> None:
        self._cancel_flag = threading.Event()


def _make_worker() -> _AgentWorker:
    w = _AgentWorker.__new__(_AgentWorker)  # bypass __init__ (spawns a thread)
    w._agent = _StubAgent()
    return w


class AwaitResponseTests(unittest.TestCase):
    def test_returns_response_when_answered(self) -> None:
        w = _make_worker()
        q: _queue.Queue = _queue.Queue()
        q.put({"choice": "y"})
        self.assertEqual(w._await_response(q), {"choice": "y"})

    def test_blocks_until_answered_does_not_time_out(self) -> None:
        """No response, no cancel → the call stays blocked (never returns on its own)."""
        w = _make_worker()
        q: _queue.Queue = _queue.Queue()
        result: list = []

        def _run() -> None:
            result.append(w._await_response(q))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        # Give it well past a poll slice; it must still be waiting (not proceeded).
        t.join(timeout=1.0)
        self.assertTrue(t.is_alive(), "shim should stay parked while unanswered")
        self.assertEqual(result, [])
        # A late answer unblocks it.
        q.put({"choice": "n"})
        t.join(timeout=2.0)
        self.assertEqual(result, [{"choice": "n"}])

    def test_cancel_flag_interrupts_wait(self) -> None:
        """Pressing Stop (cancel flag) unblocks the wait with None (cancelled)."""
        w = _make_worker()
        q: _queue.Queue = _queue.Queue()
        result: list = []

        def _run() -> None:
            result.append(w._await_response(q))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.05)
        w._agent._cancel_flag.set()
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(result, [None])


def _make_question_worker() -> _AgentWorker:
    w = _AgentWorker.__new__(_AgentWorker)
    w._agent = _StubAgent()
    w.out_q = _queue.Queue()
    w._question_q = _queue.Queue()
    w._pending_prompt = None
    w._pending_questions = None
    w._expired_prompt_ids = set()
    w.session_id = "s1"
    w.session_title = "Picking a database"
    w._query_session_id = "s1"
    return w


def _events(w: _AgentWorker) -> list:
    out = []
    while not w.out_q.empty():
        out.append(w.out_q.get_nowait())
    return out


class AnswerDeadlineTests(unittest.TestCase):
    """A question's wall: it ends the wait, says so, and closes the card."""

    def test_timeout_is_told_apart_from_cancelled(self) -> None:
        w = _make_worker()
        q: _queue.Queue = _queue.Queue()
        self.assertIs(w._await_response(q, timeout=0.05), TIMED_OUT)

    def test_an_answer_within_the_wall_wins(self) -> None:
        w = _make_worker()
        q: _queue.Queue = _queue.Queue()
        q.put({"answers": [{"selected": ["Postgres"]}]})
        self.assertEqual(
            w._await_response(q, timeout=5),
            {"answers": [{"selected": ["Postgres"]}]},
        )

    def test_expired_question_closes_its_card_and_reports_it(self) -> None:
        w = _make_question_worker()
        result = w._question_shim([{"question": "Which DB?", "header": "Database"}],
                                  timeout_secs=0.05)

        self.assertEqual(result["answers"], [])
        self.assertTrue(result["timed_out"])

        card, expiry = _events(w)
        self.assertEqual(card["type"], "user_question")
        self.assertEqual(expiry["type"], "prompt_expired")
        # Addressed to the conversation that asked: the card may be showing in the
        # foreign-prompt strip of whatever the user is reading instead.
        self.assertEqual(expiry["id"], card["id"])
        self.assertEqual(expiry["session_id"], "s1")

    def test_a_late_answer_cannot_settle_the_next_question(self) -> None:
        w = _make_question_worker()
        w._question_shim([{"question": "Which DB?", "header": "Database"}],
                         timeout_secs=0.05)
        expired_id = _events(w)[0]["id"]

        w.resolve_question([{"selected": ["Postgres"]}], expired_id)

        self.assertTrue(w._question_q.empty())

    def test_a_question_with_no_wall_stays_parked(self) -> None:
        """Plan approval passes no timeout: reading a plan takes as long as it takes."""
        w = _make_question_worker()
        done: list = []

        t = threading.Thread(
            target=lambda: done.append(w._question_shim([{"header": "Plan approval"}])),
            daemon=True,
        )
        t.start()
        t.join(timeout=1.0)
        self.assertTrue(t.is_alive())
        self.assertEqual(done, [])
        w.resolve_question([{"selected": ["Accept"]}])
        t.join(timeout=2.0)
        self.assertEqual(done, [{"answers": [{"selected": ["Accept"],
                                              "other_text": None}]}])


if __name__ == "__main__":
    unittest.main()
