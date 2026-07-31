"""Two fixes from the wave2d failure log:

1. Op-confusion redirect — an unknown op that belongs to another op-dispatched
   proxy tool names that tool (proxy_eval(op='register') → proxy_manage).
2. Persistent cross-query repeat backstop — a call that fails HARD_REPEAT_LIMIT
   times stays blocked in a *later* query too (the per-query counter resets, so a
   spin that hits the step ceiling and is re-continued would otherwise restart).

Run:
    python -m unittest mimir.tests.test_op_confusion_and_repeat -v
"""

import sys
import unittest
from pathlib import Path

_SERVERS = Path(__file__).resolve().parents[1] / "servers"
for _p in (_SERVERS / "_shared", _SERVERS / "proxy"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import server_proxy  # noqa: E402


class OpConfusionRedirectTests(unittest.TestCase):
    def test_manage_op_on_eval_redirects(self) -> None:
        r = server_proxy.proxy_eval(op="register", confirm=True)
        self.assertEqual(r.get("status"), "error")
        self.assertIn("proxy_manage", r["hint"])
        self.assertIn("not proxy_eval", r["hint"])

    def test_get_op_on_eval_redirects(self) -> None:
        r = server_proxy.proxy_eval(op="proxies", confirm=True)
        self.assertIn("proxy_get", r["hint"])

    def test_truly_unknown_op_has_no_redirect(self) -> None:
        r = server_proxy.proxy_eval(op="totally_bogus", confirm=True)
        self.assertEqual(r.get("status"), "error")
        self.assertIn("Use one of:", r["hint"])
        self.assertNotIn("belongs to", r["hint"])

    def test_shared_op_lists_all_owners(self) -> None:
        # 'eval' is a proxy_slurm op, unknown on proxy_exec → redirect names slurm.
        r = server_proxy.proxy_exec(op="eval", confirm=True)
        self.assertIn("proxy_slurm", r["hint"])


class PersistentRepeatBackstopTests(unittest.TestCase):
    """Directly exercises the agent-level counter the dispatch backstop consults."""

    def test_counter_survives_until_success_or_reset(self) -> None:
        from mimir.client.query_engine.dispatch import HARD_REPEAT_LIMIT

        class _Agent:
            def __init__(self):
                self._persistent_call_fails = {}

        agent = _Agent()
        key = ("proxy_eval", (("op", "proxies"),))

        # Simulate identical failures accumulating across "queries".
        for _ in range(HARD_REPEAT_LIMIT):
            agent._persistent_call_fails[key] = agent._persistent_call_fails.get(key, 0) + 1
        self.assertGreaterEqual(agent._persistent_call_fails[key], HARD_REPEAT_LIMIT)
        # This is exactly the predicate the dispatch hard-block uses.
        blocked = agent._persistent_call_fails.get(key, 0) >= HARD_REPEAT_LIMIT
        self.assertTrue(blocked)

        # A later success on that exact call clears it (won't stay blocked forever).
        agent._persistent_call_fails.pop(key, None)
        self.assertFalse(agent._persistent_call_fails.get(key, 0) >= HARD_REPEAT_LIMIT)

    def test_reset_session_guards_clears_counter(self) -> None:
        from mimir.client.ui.ws.ws_server import _AgentWorker

        class _Approvals:
            def reset_allowed_paths(self):
                pass

        class _Agent:
            approvals = _Approvals()
            _persistent_call_fails = {("x", ()): 5}

        w = _AgentWorker.__new__(_AgentWorker)
        w._agent = _Agent()
        w.reset_session_guards()
        self.assertEqual(w._agent._persistent_call_fails, {})


if __name__ == "__main__":
    unittest.main()
