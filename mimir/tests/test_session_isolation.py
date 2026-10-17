"""An agent per conversation — events and prompts must not cross over.

A turn parked on a plan-approval prompt survives a session switch, so its card, its
answer and its `open_editor` must all reach the conversation it started in and not the one
now on screen. These tests pin the three seams that fence a turn to its own session.
"""
import json
import unittest

from mimir.tests.support.pools import FakePool
from mimir.tests.support.connections import bare_session

from mimir.client.ui.ws.worker.worker import AgentWorker


def _bare_worker(session_id: str | None = None) -> AgentWorker:
    """A worker with only the queue/state fields the tests touch (no agent, no loop)."""
    w = AgentWorker("test-model", start=False)
    w.session_id = session_id
    return w


class WorkerStampTests(unittest.TestCase):
    def test_drain_stamps_events_with_the_running_query_session(self):
        w = _bare_worker()
        w._query_session_id = "s1"
        w.out_q.put({"type": "open_editor", "path": "/plans/a.md"})
        self.assertEqual(w.drain()[0]["session_id"], "s1")

    def test_drain_leaves_an_explicit_stamp_alone(self):
        w = _bare_worker()
        w._query_session_id = "s1"
        w.out_q.put({"type": "output", "text": "x", "session_id": "s2"})
        self.assertEqual(w.drain()[0]["session_id"], "s2")

    def test_events_outside_a_query_carry_the_workers_own_session(self):
        """Setup output belongs to its conversation as much as a turn's does.

        A worker built lazily emits its ``ready``, and any setup failure, while the user
        may well be reading a different conversation — so the stamp has to name the worker's
        own session rather than mark the event unattributable.
        """
        w = _bare_worker(session_id="s1")
        w.out_q.put({"type": "output", "text": "x"})
        self.assertEqual(w.drain()[0]["session_id"], "s1")

    def test_a_worker_with_no_session_of_its_own_still_stamps_nothing(self):
        """A bare worker (tests, standalone) has no conversation to name."""
        w = _bare_worker()
        w.out_q.put({"type": "output", "text": "x"})
        self.assertIsNone(w.drain()[0]["session_id"])

    def test_a_running_turn_outranks_the_workers_own_session(self):
        """A wake turn names the session it resumes, whoever hosts it."""
        w = _bare_worker(session_id="s1")
        w._query_session_id = "s2"
        w.out_q.put({"type": "output", "text": "x"})
        self.assertEqual(w.drain()[0]["session_id"], "s2")

    def test_flush_prompts_drops_every_pending_card(self):
        """A card left behind would be answered by the next turn's response.

        Each card's answer sits in its own slot, so dropping the cards is what drops
        the answers with them.
        """
        w = _bare_worker()
        w.human._emit_prompt({"type": "approval", "id": "a1"})
        w.human._emit_prompt({"type": "user_question", "id": "q1"}, questions=[])
        w._steer_q.put("hurry up")
        w.flush_prompts()
        self.assertEqual(dict(w.human._prompts), {})
        self.assertIsNone(w.human.pending_prompt())
        self.assertTrue(w._steer_q.empty())


class _FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)


class SessionFencingTests(unittest.IsolatedAsyncioTestCase):
    def _session(self, worker, active="s1"):
        pool = FakePool(worker, active=active)
        for sid in ("s1", "s2"):
            pool.conversations.get(sid) or pool.conversations.new(sid)
        return bare_session(pool, active=active, ws=_FakeWS())

    def test_foreign_events_are_dropped_and_own_events_kept(self):
        sess = self._session(_bare_worker())
        self.assertTrue(sess._is_foreign_event({"type": "open_editor", "session_id": "s2"}))
        self.assertFalse(sess._is_foreign_event({"type": "open_editor", "session_id": "s1"}))
        self.assertFalse(sess._is_foreign_event({"type": "output"}))  # unstamped

    async def test_leaving_a_conversation_leaves_its_turn_running(self):
        """The property the whole feature is for.

With an agent per conversation the turn has somewhere to go: it streams into
        its own transcript, and a card it is parked on carries the conversation that raised
        it, so neither needs the user to be looking at it.
        """
        w = _bare_worker(session_id="s1")
        w._query_session_id = "s1"
        w.is_busy = lambda: True
        w.cancel = lambda: self.fail("the turn of the conversation being left was cancelled")
        w.defer = lambda: self.fail("the turn of the conversation being left was deferred")
        w.flush_prompts = lambda: self.fail("its pending card was thrown away")
        sess = self._session(w)
        sess._save = lambda: None
        sess._send_sessions_list = _noop_async
        sess._load_session = _noop_async
        await sess._handle_switch_session({"session_id": "s2"})

    async def test_the_answer_of_a_conversation_left_still_lands_in_it(self):
        """Its answer arrives after the user has moved on, and goes to the conversation
        it belongs to — the pool writes it, whichever one is on screen."""
        w = _bare_worker(session_id="s1")
        w._query_session_id = "s1"
        w.is_busy = lambda: True
        sess = self._session(w)
        sess.pool.workers = {"s1": w}
        sess._send_sessions_list = _noop_async
        sess._load_session = _noop_async
        await sess._handle_switch_session({"session_id": "s2"})
        w.out_q.put({"type": "answer", "text": "done", "session_id": "s1"})
        sess.pool.bus.pump_once()
        self.assertEqual(sess.pool.conversations.get("s1").display[-1]["text"], "done")
        self.assertEqual(sess.pool.conversations.get("s2").display, [])

    async def test_set_model_reports_the_new_model_and_derived_settings(self):
        w = _bare_worker()
        w.model = "old-model"
        w.apply_setting = lambda name, v: (setattr(w, name, v), "")[1]
        w.get_thinking_profile = lambda: {"mechanism": "kwarg"}
        w.setting = lambda name, default=None: {"enforcement": "light"}.get(name, default)
        w.get_temperature_state = lambda: {"supported": True, "value": 0.6}
        sess = self._session(w)
        await sess._handle_set_model({"model": "qwen3:30b"})
        self.assertEqual(len(sess.ws.sent), 1)
        payload = json.loads(sess.ws.sent[0])
        self.assertEqual(payload["type"], "model_changed")
        self.assertEqual(payload["model"], "qwen3:30b")
        self.assertEqual(payload["thinking"], {"mechanism": "kwarg"})
        self.assertEqual(payload["enforcement"], "light")
        self.assertEqual(payload["temperature"], {"supported": True, "value": 0.6})

    async def test_served_models_reach_the_client_so_a_choice_can_be_offered(self):
        # The panel's model picker exists only when it knows of more than one model.
        # Learning that from the extension host's own probe of the address made the
        # picker hostage to a corporate proxy with no route to the cluster; the agent
        # server is connected to the endpoint either way.
        w = _bare_worker()
        w.served_models = lambda: ["a", "b"]
        sess = self._session(w)
        await sess._send_served_models()
        payload = json.loads(sess.ws.sent[0])
        self.assertEqual(payload["type"], "served_models")
        self.assertEqual(payload["models"], ["a", "b"])

    async def test_an_endpoint_that_names_nothing_sends_nothing(self):
        # An empty list is not a choice, and would replace a list the extension
        # host's probe may have got through with on its own.
        w = _bare_worker()
        w.served_models = lambda: []
        sess = self._session(w)
        await sess._send_served_models()
        self.assertEqual(sess.ws.sent, [])

    async def test_a_failing_lookup_does_not_break_the_connection(self):
        # This runs inside the greeting sequence; raising here would abort a session
        # that is otherwise perfectly usable at its default model.
        w = _bare_worker()
        def _boom():
            raise RuntimeError("endpoint down")
        w.served_models = _boom
        sess = self._session(w)
        await sess._send_served_models()
        self.assertEqual(sess.ws.sent, [])

    async def test_set_model_with_no_name_sends_an_error_not_a_change(self):
        sess = self._session(_bare_worker())
        await sess._handle_set_model({"model": "  "})
        self.assertEqual(len(sess.ws.sent), 1)
        payload = json.loads(sess.ws.sent[0])
        self.assertEqual(payload["type"], "error")
        self.assertNotEqual(payload["type"], "model_changed")

    async def test_set_model_failure_is_surfaced_as_an_error(self):
        w = _bare_worker()
        w.model = "old-model"
        w.apply_setting = lambda name, v: "Agent is not ready yet."
        sess = self._session(w)
        await sess._handle_set_model({"model": "qwen3:30b"})
        self.assertEqual(len(sess.ws.sent), 1)
        payload = json.loads(sess.ws.sent[0])
        self.assertEqual(payload["type"], "error")
        self.assertIn("not ready", payload["text"])


async def _noop_async(*args, **kwargs):
    return None


if __name__ == "__main__":
    unittest.main()
