"""What each kind of unanswered prompt does to a parked agent.

An approval waits indefinitely: it must never resume on its own and answer for the
user, so the only ways out are an answer and the cancel flag (the Stop button). A
clarification question carries a wall instead — nothing bad happens at the end of it,
the agent goes on with the option it recommended — and the wait reports that ending
distinctly so the card can be taken off the screen.

Each card has an answer slot of its own and is matched to its answer by id. A
sub-agent still running after its delegating call has returned raises its card from
another thread, so two can be in flight at once; the cards queue — one on screen at a
time — rather than share one slot.
"""

import queue as _queue
import threading
import time
import unittest
from collections import OrderedDict

from mimir.client.ui.ws.ws_worker import TIMED_OUT, _AgentWorker


class _StubAgent:
    def __init__(self) -> None:
        self._cancel_flag = threading.Event()


def _make_worker() -> _AgentWorker:
    w = _AgentWorker.__new__(_AgentWorker)  # bypass __init__ (spawns a thread)
    w._agent = _StubAgent()
    w.out_q = _queue.Queue()
    w._prompts = OrderedDict()
    w._prompts_lock = threading.Lock()
    w._pending_prompt = None
    w._pending_questions = None
    w._defer = threading.Event()
    w._preanswer = None
    w.active_session_id = None
    w._query_session_id = None
    w.session_id = None
    w.session_title = ""
    return w


def _card(req_id: str, kind: str = "approval") -> dict:
    return {"type": kind, "id": req_id, "tool": "bash_run"}


class AwaitResponseTests(unittest.TestCase):
    def test_returns_response_when_answered(self) -> None:
        w = _make_worker()
        w._emit_prompt(_card("c1"))
        w.resolve_approval("y", None, "c1")
        self.assertEqual(w._await_response("c1"), {"choice": "y", "approved_files": None})

    def test_blocks_until_answered_does_not_time_out(self) -> None:
        """No response, no cancel → the call stays blocked (never returns on its own)."""
        w = _make_worker()
        w._emit_prompt(_card("c1"))
        result: list = []

        def _run() -> None:
            result.append(w._await_response("c1"))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        # Give it well past a poll slice; it must still be waiting (not proceeded).
        t.join(timeout=1.0)
        self.assertTrue(t.is_alive(), "shim should stay parked while unanswered")
        self.assertEqual(result, [])
        # A late answer unblocks it.
        w.resolve_approval("n", None, "c1")
        t.join(timeout=2.0)
        self.assertEqual(result, [{"choice": "n", "approved_files": None}])

    def test_cancel_flag_interrupts_wait(self) -> None:
        """Pressing Stop (cancel flag) unblocks the wait with None (cancelled)."""
        w = _make_worker()
        w._emit_prompt(_card("c1"))
        result: list = []

        def _run() -> None:
            result.append(w._await_response("c1"))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.05)
        w._agent._cancel_flag.set()
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(result, [None])


class TwoCardsAtOnceTests(unittest.TestCase):
    """What a detached sub-agent introduces: two cards, raised from two threads.

    Every prompt of a turn comes from the worker thread, so on their own they cannot
    coexist. A sub-agent whose delegating call has already returned raises its own
    from wherever the elicitation lands, and a slot shared between them would let the
    second card erase the first while its answer went to whichever thread happened to
    be listening.
    """

    def _park(self, w, req_id, results):
        def _run():
            results[req_id] = w._await_response(req_id)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def test_an_answer_goes_to_the_card_it_names(self) -> None:
        w = _make_worker()
        results: dict = {}
        w._emit_prompt(_card("c1"))
        w._emit_prompt(_card("c2"))
        first = self._park(w, "c1", results)
        second = self._park(w, "c2", results)
        time.sleep(0.05)

        w.resolve_approval("y", None, "c2")
        second.join(timeout=2.0)
        self.assertEqual(results.get("c2"), {"choice": "y", "approved_files": None})
        # The other card is untouched: its user has not answered yet.
        self.assertTrue(first.is_alive(), "the wrong card was settled")
        self.assertNotIn("c1", results)

        w.resolve_approval("n", None, "c1")
        first.join(timeout=2.0)
        self.assertEqual(results.get("c1"), {"choice": "n", "approved_files": None})

    def test_only_one_card_is_on_screen_at_a_time(self) -> None:
        """Serialised, so the user answers one question at a time and the answers
        cannot be attributed to the wrong one."""
        w = _make_worker()
        w._emit_prompt(_card("c1"))
        w._emit_prompt(_card("c2"))
        self.assertEqual([ev["id"] for ev in list(w.out_q.queue)], ["c1"])

        results: dict = {}
        self._park(w, "c1", results)
        time.sleep(0.05)
        w.resolve_approval("y", None, "c1")
        time.sleep(0.1)
        # The first is answered and gone; the next one takes its place.
        self.assertEqual([ev["id"] for ev in list(w.out_q.queue)], ["c1", "c2"])
        self.assertEqual((w._pending_prompt or {}).get("id"), "c2")

    def test_an_answer_to_a_card_that_is_gone_settles_nothing(self) -> None:
        """The hazard _Session._answer_deferred already guards for a stale card: an
        answer nobody is waiting on would otherwise settle the next prompt."""
        w = _make_worker()
        results: dict = {}
        w._emit_prompt(_card("c1"))
        t = self._park(w, "c1", results)
        time.sleep(0.05)
        w.resolve_approval("y", None, "card-that-was-dismissed")
        t.join(timeout=0.5)
        self.assertTrue(t.is_alive(), "a stale answer settled a live card")
        w.resolve_approval("n", None, "c1")
        t.join(timeout=2.0)
        self.assertEqual(results.get("c1"), {"choice": "n", "approved_files": None})

    def test_an_answer_with_no_id_goes_to_the_card_on_screen(self) -> None:
        """A client that sends none can only have meant the one it was shown."""
        w = _make_worker()
        results: dict = {}
        w._emit_prompt(_card("c1"))
        w._emit_prompt(_card("c2"))
        t = self._park(w, "c1", results)
        time.sleep(0.05)
        w.resolve_approval("y", None, "")
        t.join(timeout=2.0)
        self.assertEqual(results.get("c1"), {"choice": "y", "approved_files": None})

    def test_a_question_and_an_approval_do_not_answer_each_other(self) -> None:
        w = _make_worker()
        results: dict = {}
        w._emit_prompt(_card("c1", kind="user_question"), questions=[{"question": "q?"}])
        t = self._park(w, "c1", results)
        time.sleep(0.05)
        w.resolve_approval("y", None, "")     # no id, and the wrong kind
        t.join(timeout=0.5)
        self.assertTrue(t.is_alive(), "an approval answered a question card")
        w.resolve_question([{"selected": ["a"]}], "c1")
        t.join(timeout=2.0)
        self.assertEqual(results.get("c1"), {"answers": [{"selected": ["a"]}]})


class DeadlineTests(unittest.TestCase):
    """A clarification gives up on the user; an approval never does.

    The wall exists so a long run is not stopped for the night by a question nobody
    will read. Applying it to an approval would mean a coffee break authorises a
    command the user never saw, so the deadline is passed by whoever raises the card
    and no card gets one by default.
    """

    def test_a_card_with_a_deadline_gives_up_on_its_own(self) -> None:
        """And says so with TIMED_OUT, which is not the cancelled None: the caller has
        a card on screen to take back down, and the two endings read differently to
        the model."""
        w = _make_worker()
        w._emit_prompt(_card("c1", kind="user_question"))
        started = time.monotonic()
        self.assertIs(w._await_response("c1", timeout_secs=0.3), TIMED_OUT)
        self.assertGreaterEqual(time.monotonic() - started, 0.3)

    def test_a_card_without_one_keeps_waiting(self) -> None:
        """The default, and what every approval uses."""
        w = _make_worker()
        w._emit_prompt(_card("c1"))
        result: list = []
        t = threading.Thread(target=lambda: result.append(w._await_response("c1")),
                             daemon=True)
        t.start()
        t.join(timeout=1.0)
        self.assertTrue(t.is_alive(), "an approval gave up on the user")
        w.resolve_approval("n", None, "c1")
        t.join(timeout=2.0)
        self.assertEqual(result, [{"choice": "n", "approved_files": None}])

    def test_an_answer_that_arrives_before_the_deadline_still_wins(self) -> None:
        w = _make_worker()
        w._emit_prompt(_card("c1", kind="user_question"))
        w.resolve_question([{"selected": ["a"]}], "c1")
        self.assertEqual(w._await_response("c1", timeout_secs=5),
                         {"answers": [{"selected": ["a"]}]})

    def test_the_expired_card_is_dropped_so_the_next_one_shows(self) -> None:
        """Otherwise a card nobody is waiting on holds the queue behind it."""
        w = _make_worker()
        w._emit_prompt(_card("c1", kind="user_question"))
        w._emit_prompt(_card("c2"))
        w._await_response("c1", timeout_secs=0.2)
        self.assertEqual((w._pending_prompt or {}).get("id"), "c2")


class ExpiryIsNotRefusalTests(unittest.TestCase):
    """What the shim reports back, which is what the two cases turn on."""

    def test_running_out_of_time_is_flagged_as_such(self) -> None:
        w = _make_worker()
        out = w._question_shim([{"question": "which?"}], None, 0.2)
        # The wall it ran out of travels with the verdict: the card says so on screen.
        self.assertEqual(out, {"answers": [], "timed_out": True, "timeout_secs": 0.2})

    def test_being_cancelled_is_not(self) -> None:
        """Stop means the person is gone, not that they delegated the choice."""
        w = _make_worker()
        w._agent._cancel_flag.set()
        out = w._question_shim([{"question": "which?"}], None, 30)
        self.assertEqual(out, {"answers": []})

    def test_a_shim_with_no_deadline_never_reports_a_timeout(self) -> None:
        w = _make_worker()
        w._agent._cancel_flag.set()
        self.assertEqual(w._question_shim([{"question": "which?"}]), {"answers": []})


class ExpiryClosesTheCardTests(unittest.TestCase):
    """The wall's other half: the card goes away, addressed to whoever asked."""

    def _events(self, w) -> list:
        out = []
        while not w.out_q.empty():
            out.append(w.out_q.get_nowait())
        return out

    def test_an_expired_question_closes_its_card_and_reports_it(self) -> None:
        w = _make_worker()
        w.session_id = "s1"
        w.session_title = "Picking a database"
        result = w._question_shim([{"question": "Which DB?", "header": "Database"}],
                                  timeout_secs=0.05)

        self.assertEqual(result["answers"], [])
        self.assertTrue(result["timed_out"])

        card, expiry = self._events(w)
        self.assertEqual(card["type"], "user_question")
        self.assertEqual(expiry["type"], "prompt_expired")
        # Addressed to the conversation that asked: the card may be showing in the
        # foreign-prompt strip of whatever the user is reading instead.
        self.assertEqual(expiry["id"], card["id"])
        self.assertEqual(expiry["session_id"], "s1")

    def test_a_late_answer_cannot_settle_the_next_question(self) -> None:
        """The card was dropped on the way out, so its id names nothing to settle."""
        w = _make_worker()
        w._question_shim([{"question": "Which DB?", "header": "Database"}],
                         timeout_secs=0.05)
        expired_id = self._events(w)[0]["id"]

        results: dict = {}
        w._emit_prompt(_card("next", kind="user_question"))
        t = threading.Thread(
            target=lambda: results.update(next=w._await_response("next")), daemon=True)
        t.start()
        time.sleep(0.05)
        w.resolve_question([{"selected": ["Postgres"]}], expired_id)
        t.join(timeout=0.5)
        self.assertTrue(t.is_alive(), "a late answer settled the card behind it")
        w.resolve_question([{"selected": ["SQLite"]}], "next")
        t.join(timeout=2.0)
        self.assertEqual(results["next"], {"answers": [{"selected": ["SQLite"]}]})

    def test_a_question_with_no_wall_stays_parked(self) -> None:
        """Plan approval passes no timeout: reading a plan takes as long as it takes."""
        w = _make_worker()
        done: list = []

        t = threading.Thread(
            target=lambda: done.append(w._question_shim([{"header": "Plan approval"}])),
            daemon=True,
        )
        t.start()
        t.join(timeout=1.0)
        self.assertTrue(t.is_alive())
        self.assertEqual(done, [])
        card_id = self._events(w)[0]["id"]
        w.resolve_question([{"selected": ["Accept"]}], card_id)
        t.join(timeout=2.0)
        self.assertEqual(done, [{"answers": [{"selected": ["Accept"],
                                              "other_text": None}]}])


if __name__ == "__main__":
    unittest.main()
