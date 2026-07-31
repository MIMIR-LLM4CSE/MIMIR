"""Background-jobs feature: server descriptor, client detection, worker watcher.

The feature lets the agent launch a long detached run (``background=True``), end
its turn, and be auto-resumed when a watcher detects completion. These tests
cover the three seams in isolation (no live WebSocket / no real subprocess where
avoidable).

Run:
    python -m unittest mimir.tests.test_background_jobs -v
"""

import asyncio
import json
import os
import unittest

from mimir.client.context.capabilities import BACKGROUNDABLE, ToolCaps
from mimir.client.query_engine.background import _maybe_register_background_job
from mimir.tests.test_proxy_ops import _TmpStorageTest, server_proxy


# ── 1. Server: the launch op emits a background_job descriptor on demand ────────

class BackgroundDescriptorTests(_TmpStorageTest):
    def _init_session(self) -> str:
        import sys
        exe = os.path.join(self.root, "fast.py")
        with open(exe, "w") as fh:
            fh.write(
                "import sys\n"
                "import numpy as np\n"
                "out = sys.argv[1]\n"
                "if not out.endswith('.npz'):\n"
                "    out += '.npz'\n"
                "np.savez(out, field=np.ones(4))\n"
                "print('PROXY_METRICS_BEGIN')\nprint('time_s=0.01')\n"
                "print(f'output_file={out}')\nprint('PROXY_METRICS_END')\n"
            )
        server_proxy.proxy_manage(
            op="register", name="fast", executable_path=exe,
            run_cmd_template=f"{sys.executable} {{executable}} {{output_file}}",
            output_format="npz", confirm=True,
        )
        server_proxy.proxy_manage(
            op="suite_define", name="fastb",
            cases=[{"case_id": "a", "proxy_name": "fast"}], confirm=True,
        )
        server_proxy.proxy_eval(
            op="init", proxy_name="fast", benchmark_name="fastb",
            proxy_source_path=exe, primary_metric="time_s",
            requirements=[{"metric": "time_s", "operator": "lt", "threshold": 100.0}],
            confirm=True,
        )
        return exe

    def test_background_true_attaches_descriptor(self) -> None:
        self._init_session()
        res = server_proxy.proxy_eval(op="run", proxy_name="fast",
                                      background=True, confirm=True)
        self.assertEqual(res.get("status"), "ok")
        job = res.get("background_job")
        self.assertIsInstance(job, dict)
        self.assertEqual(job["server"], "proxy")
        self.assertEqual(job["job_key"], "fast")
        self.assertEqual(job["status_op"]["tool"], "proxy_eval_status")
        self.assertEqual(job["summary_op"]["args"]["op"], "results")
        # Stop the detached run so it doesn't linger.
        server_proxy.proxy_eval(op="stop", proxy_name="fast", confirm=True)

    def test_background_false_has_no_descriptor(self) -> None:
        self._init_session()
        res = server_proxy.proxy_eval(op="run", proxy_name="fast", confirm=True)
        self.assertEqual(res.get("status"), "ok")
        self.assertNotIn("background_job", res)
        server_proxy.proxy_eval(op="stop", proxy_name="fast", confirm=True)


# ── 2. Client: dispatch detects the descriptor and registers a watcher ─────────

class _FakeAgent:
    def __init__(self, cap: bool, with_hook: bool) -> None:
        caps = frozenset({BACKGROUNDABLE}) if cap else frozenset()
        self.tool_caps = {"proxy_eval": ToolCaps(name="proxy_eval", capabilities=caps)}
        self.registered: list[dict] = []
        if with_hook:
            self._register_background_job = self._hook

    def _hook(self, descriptor: dict) -> bool:
        self.registered.append(descriptor)
        return True


class RegisterBackgroundJobTests(unittest.TestCase):
    _RESULT = json.dumps({
        "status": "ok", "run_dir": "/x", "proxy_name": "fast",
        "background_job": {"server": "proxy", "job_key": "fast",
                           "status_op": {"tool": "proxy_eval_status", "args": {}}},
    })

    def test_registers_and_augments_when_hook_present(self) -> None:
        agent = _FakeAgent(cap=True, with_hook=True)
        out = _maybe_register_background_job("proxy_eval", self._RESULT, agent)
        self.assertEqual(len(agent.registered), 1)
        self.assertIn("background job", out)
        self.assertIn("end your turn", out)

    def test_no_hook_leaves_result_unchanged(self) -> None:
        # CLI path: no _register_background_job hook → normal poll contract kept.
        agent = _FakeAgent(cap=True, with_hook=False)
        out = _maybe_register_background_job("proxy_eval", self._RESULT, agent)
        self.assertEqual(out, self._RESULT)

    def test_without_capability_is_noop(self) -> None:
        agent = _FakeAgent(cap=False, with_hook=True)
        out = _maybe_register_background_job("proxy_eval", self._RESULT, agent)
        self.assertEqual(out, self._RESULT)
        self.assertEqual(agent.registered, [])

    def test_no_descriptor_is_noop(self) -> None:
        agent = _FakeAgent(cap=True, with_hook=True)
        plain = json.dumps({"status": "ok", "note": "no job here"})
        out = _maybe_register_background_job("proxy_eval", plain, agent)
        self.assertEqual(out, plain)
        self.assertEqual(agent.registered, [])

    def test_malformed_result_is_noop(self) -> None:
        agent = _FakeAgent(cap=True, with_hook=True)
        out = _maybe_register_background_job("proxy_eval", "not json", agent)
        self.assertEqual(out, "not json")


# ── 3. Worker: _watch_job polls to completion and emits job_complete ───────────

class _ScriptedAgent:
    """Returns scripted status states, then a summary, via _run_tool."""

    def __init__(self, states: list[str], summary: dict) -> None:
        self._states = list(states)
        self._summary = summary
        self.status_calls = 0

    async def _run_tool(self, tool: str, args: dict, execution_context=None) -> str:
        if args.get("op") == "results":
            return json.dumps(self._summary)
        self.status_calls += 1
        state = self._states.pop(0) if self._states else "done"
        return json.dumps({"state": state})


class WatchJobTests(unittest.TestCase):
    def _make_worker(self, agent) -> object:
        import queue as _queue
        from mimir.client.ui.ws.ws_server import _AgentWorker
        w = _AgentWorker.__new__(_AgentWorker)   # bypass __init__ (spawns a thread)
        w._agent = agent
        w.out_q = _queue.Queue()
        w._bg_jobs = {}
        return w

    def _descriptor(self) -> dict:
        return {
            "server": "proxy", "job_key": "fast", "kind": "proxy-optimization",
            "status_op":  {"tool": "proxy_eval_status", "args": {"proxy_name": "fast"}},
            "summary_op": {"tool": "proxy_eval_status",
                           "args": {"op": "results", "proxy_name": "fast"}},
        }

    def _run(self, worker, descriptor):
        async def _drive():
            # Zero the poll interval so the test doesn't wait 5s per tick.
            # _watch_job lives in ws_worker, so patch asyncio.sleep there.
            import mimir.client.ui.ws.ws_worker as ws
            orig_sleep = asyncio.sleep

            async def _fast_sleep(_):
                await orig_sleep(0)
            ws.asyncio.sleep = _fast_sleep
            try:
                await worker._watch_job("fast", descriptor)
            finally:
                ws.asyncio.sleep = orig_sleep
        asyncio.run(_drive())

    def test_emits_job_complete_after_done(self) -> None:
        agent = _ScriptedAgent(states=["running", "running", "done"],
                               summary={"verdict": "accept",
                                        "best": {"primary_value": 0.007},
                                        "primary_metric": "time_s"})
        worker = self._make_worker(agent)
        self._run(worker, self._descriptor())
        ev = worker.out_q.get_nowait()
        self.assertEqual(ev["type"], "job_complete")
        self.assertEqual(ev["state"], "done")
        self.assertEqual(ev["summary"]["verdict"], "accept")
        self.assertNotIn("fast", worker._bg_jobs)  # cleaned up
        self.assertGreaterEqual(agent.status_calls, 3)

    def test_crashed_still_emits(self) -> None:
        agent = _ScriptedAgent(states=["running", "crashed"], summary={})
        worker = self._make_worker(agent)
        self._run(worker, self._descriptor())
        ev = worker.out_q.get_nowait()
        self.assertEqual(ev["state"], "crashed")


# ── 3b. proxy_slurm(op='eval', background=True) attaches the descriptor ─────────

class ProxySlurmBackgroundTests(_TmpStorageTest):
    def _init_session(self) -> None:
        import sys
        exe = os.path.join(self.root, "fast.py")
        with open(exe, "w") as fh:
            fh.write("print('PROXY_METRICS_BEGIN')\nprint('time_s=0.01')\n"
                     "print('PROXY_METRICS_END')\n")
        server_proxy.proxy_manage(
            op="register", name="fast", executable_path=exe,
            run_cmd_template=f"{sys.executable} {{executable}} {{output_file}}",
            output_format="npz", confirm=True,
        )
        server_proxy.proxy_manage(
            op="suite_define", name="fastb",
            cases=[{"case_id": "a", "proxy_name": "fast"}], confirm=True,
        )
        server_proxy.proxy_eval(
            op="init", proxy_name="fast", benchmark_name="fastb",
            proxy_source_path=exe, primary_metric="time_s",
            requirements=[{"metric": "time_s", "operator": "lt", "threshold": 100.0}],
            confirm=True,
        )

    def test_eval_background_attaches_descriptor(self) -> None:
        from unittest import mock
        from mimir.tests.test_proxy_ops import slurm as slurm_ops
        self._init_session()
        with mock.patch.object(slurm_ops, "_submit_sbatch", return_value=(4242, None)):
            res = server_proxy.proxy_slurm(op="eval", partition="cpu",
                                           proxy_name="fast", background=True,
                                           confirm=True)
        self.assertEqual(res.get("status"), "ok")
        job = res.get("background_job")
        self.assertIsInstance(job, dict)
        self.assertEqual(job["server"], "proxy")
        self.assertEqual(job["job_key"], "fast")
        self.assertEqual(job["status_op"]["tool"], "proxy_eval_status")

    def test_eval_without_background_has_no_descriptor(self) -> None:
        from unittest import mock
        from mimir.tests.test_proxy_ops import slurm as slurm_ops
        self._init_session()
        with mock.patch.object(slurm_ops, "_submit_sbatch", return_value=(4243, None)):
            res = server_proxy.proxy_slurm(op="eval", partition="cpu",
                                           proxy_name="fast", confirm=True)
        self.assertEqual(res.get("status"), "ok")
        self.assertNotIn("background_job", res)


# ── 3a. CLI efficient-await: poll in-turn (no model calls), append the summary ──

class AwaitBackgroundJobTests(unittest.TestCase):
    def _run(self, agent, descriptor, result):
        from mimir.client.query_engine import background as al

        async def _drive():
            orig = al.asyncio.sleep

            async def _fast(_):
                await orig(0)
            al.asyncio.sleep = _fast
            try:
                return await al._await_background_job(descriptor, agent, result)
            finally:
                al.asyncio.sleep = orig
        return asyncio.run(_drive())

    def _descriptor(self) -> dict:
        return {"status_op":  {"tool": "proxy_eval_status", "args": {}},
                "summary_op": {"tool": "proxy_eval_status", "args": {"op": "results"}}}

    def test_awaits_then_folds_in_summary(self) -> None:
        agent = _ScriptedAgent(states=["running", "running", "done"],
                               summary={"verdict": "accept"})
        out = self._run(agent, self._descriptor(), '{"status":"ok"}')
        self.assertIn("[background:awaited]", out)
        self.assertIn("done", out)
        self.assertIn("accept", out)
        self.assertGreaterEqual(agent.status_calls, 3)


# ── 4. Auto-resume: the wake text the session synthesizes from job_complete ────

class WakeTextTests(unittest.TestCase):
    def _wake(self, ev: dict) -> str:
        from mimir.client.ui.ws.ws_server import _Session
        return _Session._wake_text(ev)

    def test_accept_summary_is_enriched(self) -> None:
        ev = {"job_key": "wave1d", "state": "done",
              "summary": {"verdict": "accept", "primary_metric": "time_s",
                          "best": {"primary_value": 0.0073},
                          "next_step": "continue the loop"}}
        out = self._wake(ev)
        self.assertIn("wave1d", out)
        self.assertIn("accept", out)
        self.assertIn("0.0073", out)
        self.assertIn("continue the loop", out)

    def test_crashed_proxy_points_to_proxy_log(self) -> None:
        out = self._wake({"job_key": "wave1d", "state": "crashed",
                          "server": "proxy", "summary": {}})
        self.assertIn("crashed", out)
        self.assertIn("op='log'", out)

    def test_crashed_non_proxy_points_to_slurm_logs(self) -> None:
        out = self._wake({"job_key": "9001", "state": "crashed",
                          "server": "hpc", "summary": {}})
        self.assertIn("crashed", out)
        self.assertIn("sacct", out)

    def test_missing_summary_is_generic(self) -> None:
        out = self._wake({"job_key": "j", "state": "done"})
        self.assertIn("finished", out)
        self.assertIn("results", out)


if __name__ == "__main__":
    unittest.main()
