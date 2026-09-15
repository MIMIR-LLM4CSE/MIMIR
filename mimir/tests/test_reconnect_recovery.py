"""A connection that drops under a running turn must not cost the turn.

One ``_AgentWorker`` serves the whole server and outlives every connection, but a
``_Session`` is per-socket. A drop mid-run therefore left a turn alive with nobody
reading it, and the next connection did three things to it: it emptied the queue the
turn was writing into, it never put back the card the turn was parked on, and it told
the client nothing was running. The turn then waited on an answer that could not
arrive — the query loop is serial, so every later query queued behind that wait — and
the session read as hung with nothing on screen to say why.
"""
import json
import queue as _queue
import unittest

from mimir.client.ui.ws.ws_worker import _AgentWorker
from mimir.tests.test_session_isolation import _FakeWS, _bare_worker, SessionFencingTests


def _parked_worker(prompt: dict | None = None) -> _AgentWorker:
    w = _bare_worker()
    w._pending_prompt = prompt
    return w


class PendingPromptTests(unittest.TestCase):
    def test_emitting_a_card_marks_the_turn_parked_on_it(self):
        w = _parked_worker()
        card = {"type": "user_question", "id": "q1", "questions": []}
        w._emit_prompt(card)
        self.assertEqual(w.out_q.get_nowait(), card)
        self.assertEqual(w.pending_prompt(), card)

    def test_the_card_is_a_copy_of_what_was_sent(self):
        """The queued dict is drained and stamped downstream; the record must not follow."""
        w = _parked_worker()
        card = {"type": "approval", "id": "a1"}
        w._emit_prompt(card)
        w.out_q.get_nowait()["session_id"] = "s1"
        self.assertEqual(w.pending_prompt(), {"type": "approval", "id": "a1"})

    def test_an_answer_ends_the_park(self):
        w = _parked_worker({"type": "approval", "id": "a1"})
        w._agent = None
        w._approval_q.put({"choice": "y"})
        self.assertEqual(w._await_response(w._approval_q), {"choice": "y"})
        self.assertIsNone(w.pending_prompt())

    def test_a_cancelled_wait_ends_the_park(self):
        class _Cancelled:
            class _Flag:
                @staticmethod
                def is_set():
                    return True
            _cancel_flag = _Flag()

        w = _parked_worker({"type": "approval", "id": "a1"})
        w._agent = _Cancelled()
        self.assertIsNone(w._await_response(w._approval_q))
        self.assertIsNone(w.pending_prompt())

    def test_a_card_still_on_its_way_out_is_not_handed_out_twice(self):
        """The drain loop will deliver it; resending it too is a card answered twice."""
        w = _parked_worker()
        w._emit_prompt({"type": "approval", "id": "a1"})
        self.assertIsNone(w.pending_prompt())
        w.out_q.get_nowait()                      # delivered to whoever was connected
        self.assertEqual(w.pending_prompt(), {"type": "approval", "id": "a1"})

    def test_abandoning_the_turn_forgets_the_card(self):
        w = _parked_worker({"type": "continue_prompt", "id": "c1", "summary": "3 left"})
        w.flush_prompts()
        self.assertIsNone(w.pending_prompt())


class ReconnectTests(unittest.IsolatedAsyncioTestCase):
    def _session(self, worker):
        return SessionFencingTests._session(self, worker)

    async def test_the_card_a_turn_is_parked_on_comes_back(self):
        card = {"type": "user_question", "id": "q1", "questions": [{"header": "Plan approval"}]}
        sess = self._session(_parked_worker(card))
        await sess._resend_parked_prompt()
        self.assertEqual([json.loads(p) for p in sess.ws.sent], [card])

    async def test_nothing_is_sent_when_no_turn_is_parked(self):
        sess = self._session(_parked_worker(None))
        await sess._resend_parked_prompt()
        self.assertEqual(sess.ws.sent, [])

    async def test_a_socket_that_fails_on_the_resend_does_not_break_the_connection(self):
        sess = self._session(_parked_worker({"type": "approval", "id": "a1"}))

        async def _boom(_payload):
            raise ConnectionResetError("gone")

        sess.ws.send = _boom
        await sess._resend_parked_prompt()  # must not raise

    def test_a_busy_worker_keeps_the_events_of_its_running_turn(self):
        w = _parked_worker()
        w.is_busy = lambda: True
        w.out_q.put({"type": "tool_call", "id": "t1"})
        w.out_q.put({"type": "answer", "text": "what I did"})
        self._session(w)._drop_stale_events()
        self.assertEqual(w.out_q.qsize(), 2)

    def test_an_idle_worker_leaves_only_debris_behind(self):
        w = _parked_worker()
        w.is_busy = lambda: False
        w.out_q.put({"type": "status", "text": "left over"})
        self._session(w)._drop_stale_events()
        self.assertTrue(w.out_q.empty())


if __name__ == "__main__":
    unittest.main()
