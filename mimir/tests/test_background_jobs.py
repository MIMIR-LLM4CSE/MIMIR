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
import types
import unittest

from mimir.client.context.capabilities import BACKGROUNDABLE, ToolCaps
from mimir.client.query_engine.background import _maybe_register_background_job
from mimir.tests.test_proxy_ops import _TmpStorageTest, server_proxy


def _eval(*args, **kwargs):
    """Call the now-async ``proxy_eval`` from a synchronous test.

    ``op='run'`` awaits the run it launches, so the tool is a coroutine function.
    """
    return asyncio.run(server_proxy.proxy_eval(*args, **kwargs))


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
        _eval(
            op="init", proxy_name="fast", benchmark_name="fastb",
            proxy_source_path=exe, optimize_paths=[self._tracked()], primary_metric="time_s",
            requirements=[{"metric": "time_s", "operator": "lt", "threshold": 100.0}],
            confirm=True,
        )
        return exe

    def test_background_true_attaches_descriptor(self) -> None:
        self._init_session()
        res = _eval(op="run", proxy_name="fast",
                                      background=True, confirm=True)
        self.assertEqual(res.get("status"), "ok")
        job = res.get("background_job")
        self.assertIsInstance(job, dict)
        self.assertEqual(job["server"], "proxy")
        self.assertEqual(job["job_key"], "fast")
        self.assertEqual(job["status_op"]["tool"], "proxy_eval_status")
        self.assertEqual(job["summary_op"]["args"]["op"], "results")
        # Stop the detached run so it doesn't linger.
        _eval(op="stop", proxy_name="fast", confirm=True)

    def test_background_false_has_no_descriptor(self) -> None:
        self._init_session()
        res = _eval(op="run", proxy_name="fast", confirm=True)
        self.assertEqual(res.get("status"), "ok")
        self.assertNotIn("background_job", res)
        _eval(op="stop", proxy_name="fast", confirm=True)


# ── 2. Client: dispatch detects the descriptor and registers a watcher ─────────

class _FakeAgent:
    def __init__(self, cap: bool, with_hook: bool, outcome: object = True) -> None:
        caps = frozenset({BACKGROUNDABLE}) if cap else frozenset()
        self.tool_caps = {"proxy_eval": ToolCaps(name="proxy_eval", capabilities=caps)}
        self.registered: list[dict] = []
        # What the front-end hook does: True registers, False declines, an Exception
        # instance is raised. The last two are the paths the field failure took.
        self._outcome = outcome
        if with_hook:
            self._register_background_job = self._hook

    def _hook(self, descriptor: dict) -> bool:
        self.registered.append(descriptor)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return bool(self._outcome)


class RegisterBackgroundJobTests(unittest.TestCase):
    _RESULT = json.dumps({
        "status": "ok", "run_dir": "/x", "proxy_name": "fast",
        "background_job": {"server": "proxy", "job_key": "fast",
                           "status_op": {"tool": "proxy_eval_status", "args": {}}},
    })

    def test_registers_and_augments_when_hook_present(self) -> None:
        agent = _FakeAgent(cap=True, with_hook=True)
        out, registered = _maybe_register_background_job("proxy_eval", self._RESULT, agent)
        self.assertEqual(len(agent.registered), 1)
        self.assertTrue(registered)
        self.assertIn("background job", out)
        self.assertIn("end your turn", out)

    def test_no_hook_leaves_result_unchanged(self) -> None:
        # CLI path: no _register_background_job hook → normal poll contract kept.
        agent = _FakeAgent(cap=True, with_hook=False)
        out, registered = _maybe_register_background_job("proxy_eval", self._RESULT, agent)
        self.assertEqual(out, self._RESULT)
        self.assertFalse(registered)

    def test_without_capability_is_noop(self) -> None:
        agent = _FakeAgent(cap=False, with_hook=True)
        out, registered = _maybe_register_background_job("proxy_eval", self._RESULT, agent)
        self.assertEqual(out, self._RESULT)
        self.assertFalse(registered)
        self.assertEqual(agent.registered, [])

    def test_no_descriptor_is_noop(self) -> None:
        agent = _FakeAgent(cap=True, with_hook=True)
        plain = json.dumps({"status": "ok", "note": "no job here"})
        out, registered = _maybe_register_background_job("proxy_eval", plain, agent)
        self.assertEqual(out, plain)
        self.assertFalse(registered)
        self.assertEqual(agent.registered, [])

    def test_malformed_result_is_noop(self) -> None:
        agent = _FakeAgent(cap=True, with_hook=True)
        out, registered = _maybe_register_background_job("proxy_eval", "not json", agent)
        self.assertEqual(out, "not json")
        self.assertFalse(registered)

    # ── The three ways registration fails, which used to be indistinguishable ──
    # from success. Each one promised the model a resume nobody was holding.

    def test_declined_registration_promises_nothing(self) -> None:
        agent = _FakeAgent(cap=True, with_hook=True, outcome=False)
        with self.assertLogs("mimir.client.query_engine.background", "WARNING") as log:
            out, registered = _maybe_register_background_job(
                "proxy_eval", self._RESULT, agent)
        self.assertFalse(registered)
        # The note is the promise of a resume. A declined registration must not carry
        # it: that is exactly what let the model end its turn on a wake that never came.
        self.assertEqual(out, self._RESULT)
        self.assertIn("declined", "\n".join(log.output))

    def test_raising_hook_promises_nothing(self) -> None:
        agent = _FakeAgent(cap=True, with_hook=True, outcome=RuntimeError("no loop"))
        with self.assertLogs("mimir.client.query_engine.background", "WARNING") as log:
            out, registered = _maybe_register_background_job(
                "proxy_eval", self._RESULT, agent)
        self.assertFalse(registered)
        self.assertEqual(out, self._RESULT)
        self.assertIn("raised", "\n".join(log.output))

    def test_annotated_result_still_finds_its_descriptor(self) -> None:
        """A post-tool annotation must not be able to hide a launched run.

        The dispatcher appends advisory prose to the same string the descriptor lives
        in, and a result is an envelope plus its text blocks besides. Read with a bare
        ``json.loads`` that is one broken document and the run goes unwatched — which
        is why detection goes through ``parse_tool_payload``, like every other
        consumer of a tool payload.
        """
        from mimir.client.query_engine.background import _detect_background_job
        agent = _FakeAgent(cap=True, with_hook=True)
        annotated = self._RESULT + "\n\nAUTO_VALIDATION\nlooks fine"
        self.assertIsNotNone(_detect_background_job("proxy_eval", annotated, agent))
        out, registered = _maybe_register_background_job("proxy_eval", annotated, agent)
        self.assertTrue(registered)
        self.assertIn("[background]", out)

    def test_annotated_result_still_opens_the_editor(self) -> None:
        """Same defect, same fix, on the sibling that opens a written plan."""
        from mimir.client.query_engine import background as bg
        payload = json.dumps({"status": "ok", "name": "p",
                              "path": "/tmp/plan.md", "open_in_editor": True})
        seen: list[dict] = []
        orig, bg.emit = bg.emit, seen.append
        try:
            bg._maybe_emit_open_editor(payload + "\n\nAUTO_VALIDATION\nlooks fine")
        finally:
            bg.emit = orig
        self.assertEqual(seen, [{"type": "open_editor", "path": "/tmp/plan.md"}])


# ── 2b. The joint: a real launch result, read through the real live registry ────

class LiveRegistrySeamTests(unittest.TestCase):
    """The seam that actually failed in the field, and that nothing covered.

    Section 2 builds the capability registry by hand, so its tests pass whether or not
    a server really declares ``backgroundable``. In September 2026 ``bash_run`` did not
    yet, on the client's side of the wire: every launch came back with the server's own
    "you will be resumed" note, no watcher was ever registered, and five two-hour builds
    finished into silence while the model kept promising a resume nobody was holding.
    Both halves were tested. The joint between them was not.

    So this one goes end to end on the client's side: the ``_meta`` the server really
    advertises, decoded by the real ``infer_tool_caps``, against a real launch result.
    """

    def setUp(self) -> None:
        import shutil
        import tempfile
        from mimir.tests.test_bash_background import server_bash, _bash_jobs
        self.server_bash, self._bash_jobs = server_bash, _bash_jobs
        self._tmp = tempfile.mkdtemp(prefix="mimir-seam-jobs-")
        self._orig_root = _bash_jobs.JOBS_ROOT
        _bash_jobs.JOBS_ROOT = self._tmp
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self.addCleanup(setattr, _bash_jobs, "JOBS_ROOT", self._orig_root)

    def _live_registry(self, tool_name: str) -> dict:
        """The capabilities the client would resolve for *tool_name*, over the wire."""
        from mimir.client.context.capabilities import infer_tool_caps
        tools = asyncio.run(self.server_bash.mcp.list_tools())
        tool = (tools[tool_name] if isinstance(tools, dict)
                else next(t for t in tools if t.name == tool_name))
        return {tool_name: infer_tool_caps(tool)}

    def test_a_backgroundable_launch_is_detected_and_the_watcher_promised(self) -> None:
        import types
        registry = self._live_registry("bash_run")
        self.assertIn(BACKGROUNDABLE, registry["bash_run"].capabilities,
                      "bash_run must advertise 'backgroundable' in the _meta it sends: "
                      "without it the client never looks for the descriptor")

        result = self.server_bash.bash_run("sleep 5", background=True)
        job_key = result["background_job"]["job_key"]
        self.addCleanup(self._bash_jobs.stop, job_key)

        registered: list[dict] = []
        agent = types.SimpleNamespace(
            tool_caps=registry,
            _register_background_job=lambda d: bool(registered.append(d) or True),
        )
        out, was_registered = _maybe_register_background_job(
            "bash_run", json.dumps(result), agent)

        self.assertEqual(len(registered), 1)
        self.assertEqual(registered[0]["job_key"], job_key)
        self.assertTrue(was_registered)
        # The note is the only promise of a resume the model should ever see, and the
        # only proof a watcher exists. Its absence is what the field failure looked like.
        self.assertIn("[background]", out)
        self.assertIn(job_key, out)

    def test_the_server_does_not_promise_a_resume_on_its_own(self) -> None:
        # The launch note describes what was started. Whether anything is watching is
        # the client's to say — a server that guarantees it produces a model that ends
        # its turn on a promise nobody is holding.
        result = self.server_bash.bash_run("sleep 5", background=True)
        self.addCleanup(self._bash_jobs.stop, result["background_job"]["job_key"])
        note = result.get("note", "")
        self.assertNotIn("resumed", note)
        self.assertIn("background", note)


# ── 3. Worker: _watch_job polls to completion and emits job_complete ───────────

class _ScriptedAgent:
    """Returns scripted status states, then a summary, via _run_tool.

    Records the ``record_observations`` each probe was made with: a watcher tick is
    the client asking a question on its own account, and crediting it to the agent
    would put a step it never took into what the guardrails have seen.
    """

    def __init__(self, states: list[str], summary: dict) -> None:
        self._states = list(states)
        self._summary = summary
        self.status_calls = 0
        self.observed: list[bool] = []

    async def _run_tool(self, tool: str, args: dict, execution_context=None,
                        record_observations: bool = True) -> str:
        self.observed.append(record_observations)
        if args.get("op") == "results":
            return json.dumps(self._summary)
        self.status_calls += 1
        state = self._states.pop(0) if self._states else "done"
        return json.dumps({"state": state})


class _MuteAgent:
    """An agent whose status op answers, but never with a state anyone can act on.

    The shape of a probe that stopped working: a policy violation, a dead server, a
    renamed op. Indistinguishable from a running job if a watcher only looks for a
    terminal state — which is how a wake that can never come stays invisible.
    """

    def __init__(self, payload: dict | None = None, raises: bool = False) -> None:
        self._payload = payload if payload is not None else {"status": "error",
                                                             "error": "policy: refused"}
        self._raises = raises
        self.status_calls = 0

    async def _run_tool(self, tool: str, args: dict, execution_context=None,
                        record_observations: bool = True) -> str:
        self.status_calls += 1
        if self._raises:
            raise RuntimeError("server is gone")
        return json.dumps(self._payload)


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

    def _run(self, worker, descriptor, session_id=None):
        async def _drive():
            # Zero the poll interval so the test doesn't wait 5s per tick.
            # _watch_job lives in ws_worker, so patch asyncio.sleep there.
            import mimir.client.ui.ws.ws_worker as ws
            orig_sleep = asyncio.sleep

            async def _fast_sleep(_):
                await orig_sleep(0)
            ws.asyncio.sleep = _fast_sleep
            try:
                await worker._watch_job("fast", descriptor, session_id)
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

    def test_the_event_carries_the_launching_session_and_the_descriptor_ops(self) -> None:
        # Both are what the session layer needs and cannot reconstruct: which
        # conversation to wake, and — for a job with no summary — the only op it may
        # name, because it came from the descriptor rather than from the client.
        agent = _ScriptedAgent(states=["done"], summary={})
        worker = self._make_worker(agent)
        self._run(worker, self._descriptor(), session_id="sess-A")
        ev = worker.out_q.get_nowait()
        self.assertEqual(ev["session_id"], "sess-A")
        self.assertEqual(ev["status_op"]["tool"], "proxy_eval_status")
        self.assertEqual(ev["kind"], "proxy-optimization")

    def test_probes_are_not_recorded_as_steps_the_model_took(self) -> None:
        agent = _ScriptedAgent(states=["running", "done"], summary={"verdict": "accept"})
        worker = self._make_worker(agent)
        self._run(worker, self._descriptor())
        self.assertTrue(agent.observed)                    # it was asked
        self.assertNotIn(True, agent.observed)             # never on the agent's record

    def test_an_unreadable_probe_ends_as_unknown_rather_than_polling_forever(self) -> None:
        # The failure this guard exists for: a status op that answers but never with a
        # state, so no terminal state is ever reached, the watcher spins, and the wake
        # it promised never arrives — silently, which is the worst part.
        agent = _MuteAgent()
        worker = self._make_worker(agent)
        self._run(worker, self._descriptor())
        ev = worker.out_q.get_nowait()
        self.assertEqual(ev["state"], "unknown")
        self.assertIn("refused", ev["reason"])
        self.assertLessEqual(agent.status_calls, 6)        # bounded, not forever
        self.assertNotIn("fast", worker._bg_jobs)

    def test_a_probe_that_raises_also_ends_as_unknown(self) -> None:
        agent = _MuteAgent(raises=True)
        worker = self._make_worker(agent)
        self._run(worker, self._descriptor())
        ev = worker.out_q.get_nowait()
        self.assertEqual(ev["state"], "unknown")
        self.assertIn("RuntimeError", ev["reason"])

    def test_a_transient_failure_does_not_trip_the_guard(self) -> None:
        # One bad tick in the middle of a healthy run must not end the watch: the
        # counter resets on any state the descriptor's own op owns.
        class _Flaky(_ScriptedAgent):
            async def _run_tool(self, tool, args, execution_context=None,
                                record_observations: bool = True):
                self.status_calls += 1
                if self.status_calls in (2, 5):
                    return json.dumps({"status": "error", "error": "blip"})
                if self.status_calls < 8:
                    return json.dumps({"state": "running"})
                return json.dumps({"state": "done"})

        agent = _Flaky(states=[], summary={})
        worker = self._make_worker(agent)
        self._run(worker, self._descriptor())
        ev = worker.out_q.get_nowait()
        self.assertEqual(ev["state"], "done")

    def test_crashed_still_emits(self) -> None:
        agent = _ScriptedAgent(states=["running", "crashed"], summary={})
        worker = self._make_worker(agent)
        self._run(worker, self._descriptor())
        ev = worker.out_q.get_nowait()
        self.assertEqual(ev["state"], "crashed")


class RegisterBgJobTests(unittest.TestCase):
    """``_register_bg_job``'s refusals: each one must say so.

    Its return value is the whole contract — False means no watcher is holding the
    run, and the caller must await it in-turn instead. Refusing silently is what made
    a run finish into nothing with no line anywhere to explain it.
    """

    def _worker(self) -> object:
        from mimir.client.ui.ws.ws_server import _AgentWorker
        w = _AgentWorker.__new__(_AgentWorker)   # bypass __init__ (spawns a thread)
        w._bg_jobs = {}
        w._query_session_id = None
        w.active_session_id = None
        w._loop = None
        return w

    def test_descriptor_without_a_key_is_refused_out_loud(self) -> None:
        w = self._worker()
        with self.assertLogs("mimir.client.ui.ws.ws_worker", "WARNING") as log:
            self.assertFalse(w._register_bg_job({"server": "bash", "status_op": {}}))
        self.assertIn("job_key", "\n".join(log.output))
        self.assertEqual(w._bg_jobs, {})

    def test_non_dict_descriptor_is_refused_out_loud(self) -> None:
        w = self._worker()
        with self.assertLogs("mimir.client.ui.ws.ws_worker", "WARNING") as log:
            self.assertFalse(w._register_bg_job("not a descriptor"))
        self.assertIn("not a dict", "\n".join(log.output))

    def test_no_running_loop_is_refused_out_loud(self) -> None:
        """Called off the worker loop there is nothing to host the watcher."""
        w = self._worker()
        with self.assertLogs("mimir.client.ui.ws.ws_worker", "WARNING") as log:
            self.assertFalse(w._register_bg_job({"job_key": "fast", "status_op": {}}))
        self.assertIn("event loop", "\n".join(log.output))
        self.assertEqual(w._bg_jobs, {})


# ── 3c. The dispatch guards: a watched run is neither polled nor waited on ─────

class _GuardAgent:
    """Just enough agent for the two dispatch guards."""

    def __init__(self, watched: list[dict] | None = None, *, with_hook: bool = True,
                 shell_tool: str = "bash_run") -> None:
        from mimir.client.context.capabilities import ToolCaps
        self.tool_caps = {
            shell_tool: ToolCaps(name=shell_tool, capabilities=frozenset(),
                                 scope={"args": ["command"], "kind": "command_prefix"}),
            "bash_job": ToolCaps(name="bash_job", capabilities=frozenset()),
        }
        self._watched = watched or []
        if with_hook:
            self._watched_background_jobs = lambda: list(self._watched)


def _descriptor(job_key: str = "j1") -> dict:
    return {
        "server": "bash", "kind": "shell-command", "job_key": job_key,
        "status_op":  {"tool": "bash_job", "args": {"job_key": job_key}},
        "summary_op": {"tool": "bash_job",
                       "args": {"op": "output", "job_key": job_key}},
    }


class WatchedRunPollGuardTests(unittest.TestCase):
    def _asks(self, agent, name, args):
        from mimir.client.query_engine.dispatch import (
            _asks_whether_a_watched_run_is_done)
        return _asks_whether_a_watched_run_is_done(agent, name, args)

    def test_state_of_a_watched_run_is_refused(self) -> None:
        agent = _GuardAgent([_descriptor()])
        self.assertEqual(self._asks(agent, "bash_job", {"job_key": "j1"}), "j1")

    def test_an_explicit_default_is_the_same_question(self) -> None:
        """Containment, not equality: spelling out the implicit op must not slip past."""
        agent = _GuardAgent([_descriptor()])
        self.assertEqual(
            self._asks(agent, "bash_job", {"op": "status", "job_key": "j1"}), "j1")

    def test_reading_the_output_stays_allowed(self) -> None:
        """Progress mid-run is information the watcher does not report until the end."""
        agent = _GuardAgent([_descriptor()])
        self.assertIsNone(
            self._asks(agent, "bash_job", {"op": "output", "job_key": "j1"}))

    def test_an_unwatched_run_is_not_guarded(self) -> None:
        agent = _GuardAgent([_descriptor("j1")])
        self.assertIsNone(self._asks(agent, "bash_job", {"job_key": "other"}))

    def test_listing_every_job_is_not_polling_one(self) -> None:
        agent = _GuardAgent([_descriptor()])
        self.assertIsNone(self._asks(agent, "bash_job", {"op": "list"}))

    def test_without_the_hook_the_guard_abstains(self) -> None:
        """The CLI has no watcher, so nothing may be refused on its behalf."""
        agent = _GuardAgent([_descriptor()], with_hook=False)
        self.assertIsNone(self._asks(agent, "bash_job", {"job_key": "j1"}))

    def test_a_raising_hook_abstains(self) -> None:
        agent = _GuardAgent()
        def _boom():
            raise RuntimeError("worker gone")
        agent._watched_background_jobs = _boom
        self.assertIsNone(self._asks(agent, "bash_job", {"job_key": "j1"}))


class BlockingWaitGuardTests(unittest.TestCase):
    def _waits(self, agent, command, name="bash_run"):
        from mimir.client.query_engine.dispatch import _waits_by_blocking_the_turn
        return _waits_by_blocking_the_turn(agent, name, {"command": command})

    def test_leading_sleep_is_refused_while_a_run_is_watched(self) -> None:
        agent = _GuardAgent([_descriptor()])
        self.assertTrue(self._waits(agent, "sleep 60; tail -2 build.log"))

    def test_wrappers_and_env_assignments_are_seen_through(self) -> None:
        agent = _GuardAgent([_descriptor()])
        self.assertTrue(self._waits(agent, "FOO=1 sleep 60"))
        self.assertTrue(self._waits(agent, "env sleep 90"))

    def test_sleep_that_is_not_the_leading_program_is_allowed(self) -> None:
        """A flag named --sleep is not a wait, and work already done is not one either."""
        agent = _GuardAgent([_descriptor()])
        self.assertFalse(self._waits(agent, "./run.sh --sleep 60"))
        self.assertFalse(self._waits(agent, "make -j8; sleep 1"))

    def test_a_tool_that_takes_no_command_line_abstains(self) -> None:
        agent = _GuardAgent([_descriptor()])
        self.assertFalse(self._waits(agent, "sleep 60", name="bash_job"))


class WatcherProbeIsNeverGuardedTests(unittest.TestCase):
    """The failure mode this guard could introduce, pinned down.

    The watcher polls the very status op the guard refuses. It reaches the tool through
    ``agent._run_tool``, never through ``_dispatch_tool_calls``, so the separation is
    structural — but nothing said so out loud until this test, and getting it wrong
    would silently kill every wake.
    """

    def test_dispatch_is_the_only_caller_that_consults_the_guards(self) -> None:
        import inspect
        from mimir.client.query_engine import dispatch, background
        from mimir.client.ui.ws import ws_worker

        guard_names = ("_asks_whether_a_watched_run_is_done",
                       "_waits_by_blocking_the_turn")
        # The watcher (WS) and the in-turn await (CLI) must not reference either guard.
        for module in (background, ws_worker):
            src = inspect.getsource(module)
            for guard in guard_names:
                self.assertNotIn(guard, src, f"{module.__name__} must not gate its own probes")
        # And both probe paths go through _run_tool, not the dispatcher.
        for probe in (background._await_background_job, ws_worker._AgentWorker._watch_job):
            src = inspect.getsource(probe)
            self.assertIn("_run_tool", src)
            self.assertNotIn("_dispatch_tool_calls", src)


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
        _eval(
            op="init", proxy_name="fast", benchmark_name="fastb",
            proxy_source_path=exe, optimize_paths=[self._tracked()], primary_metric="time_s",
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

    def test_a_shell_job_is_not_told_to_review_proxy_results(self) -> None:
        # The bug this whole family exists to prevent: a two-hour compile finishing and
        # the wake telling the model to read proxy results and continue an optimization
        # loop that does not exist. The client does not know what kind of job ran, so
        # it may not name an op of its own — it passes the job's own record through.
        out = self._wake({
            "job_key": "20260909T170922Z-d123", "kind": "shell-command", "state": "done",
            "server": "bash",
            "status_op": {"tool": "bash_job", "args": {}},
            "summary": {"state": "done", "exit_code": 0,
                        "log_tail": "[100%] Built target sem-solver"},
        })
        self.assertNotIn("proxy_eval_status", out)
        self.assertNotIn("optimization loop", out)
        self.assertNotIn("sacct", out)
        self.assertIn("finished", out)
        self.assertIn("Built target sem-solver", out)     # its own result, passed through
        self.assertIn("exit_code", out)

    def test_a_crashed_shell_job_carries_its_log_not_slurm_advice(self) -> None:
        out = self._wake({
            "job_key": "j", "kind": "shell-command", "state": "crashed", "server": "bash",
            "summary": {"state": "crashed", "exit_code": 1,
                        "log_tail": "CMake Error: no CUDA toolset found"},
        })
        self.assertIn("crashed", out)
        self.assertIn("CMake Error", out)
        self.assertNotIn("sacct", out)
        self.assertNotIn("proxy_eval_status", out)

    def test_a_server_authored_next_step_is_relayed_verbatim(self) -> None:
        # An instruction that came from something which knows the job is the one thing
        # worth repeating — the proxy writes one, so a proxy run still reads as before.
        out = self._wake({"job_key": "wave1d", "kind": "proxy-optimization",
                          "state": "done",
                          "summary": {"verdict": "accept",
                                      "next_step": "proxy_eval_status(op='results')"}})
        self.assertIn("verdict=accept", out)
        self.assertIn("proxy_eval_status(op='results')", out)

    def test_a_job_with_no_summary_points_at_its_own_status_op(self) -> None:
        # sbatch_submit declares no summary_op. The descriptor's own status tool is
        # registry data travelling on the event, so naming it is not the client
        # choosing a tool — and it is the only thing left to point at.
        out = self._wake({"job_key": "9001", "kind": "slurm-batch", "state": "crashed",
                          "status_op": {"tool": "slurm_job_status", "args": {}},
                          "summary": {}})
        self.assertIn("crashed", out)
        self.assertIn("slurm_job_status", out)

    def test_a_job_that_cannot_be_tracked_says_so(self) -> None:
        out = self._wake({"job_key": "j", "kind": "shell-command", "state": "unknown",
                          "reason": "'bash_job' returned no state", "summary": {}})
        self.assertIn("no longer be tracked", out)
        self.assertIn("returned no state", out)

    def test_an_oversized_summary_is_cut_and_says_so(self) -> None:
        out = self._wake({"job_key": "j", "state": "done",
                          "summary": {"log_tail": "x" * 50_000}})
        self.assertLess(len(out), 4_000)
        self.assertIn("cut:", out)


# ── 5. The wake belongs to the session that launched the job ───────────────────

class _FakeWS:
    """Collects the frames a session would have sent to its client."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


class _FakeWorker:
    """Records submitted turns, the way _AgentWorker's query queue would."""

    def __init__(self) -> None:
        self.submitted: list[tuple] = []
        self.steered: list[str] = []
        self._query_session_id = None
        self._agent = types.SimpleNamespace(context_mode="flat")
        self.model = "test-model"

    def submit_query(self, text, history, session_id=None) -> None:
        self.submitted.append((text, list(history), session_id))
        self._query_session_id = session_id

    def submit_steer(self, text) -> None:
        self.steered.append(text)

    def full_history(self):
        return None

    def is_busy(self) -> bool:
        return self._query_session_id is not None


class DetachedSessionResumeTests(unittest.TestCase):
    """A two-hour build outlives the conversation on screen.

    Its result belongs to the session that asked for it. Dropped into whichever one the
    user happens to be reading, it is an answer to a question that conversation never
    put — and the conversation that did put it never hears back.
    """

    def setUp(self) -> None:
        import shutil
        import tempfile
        from unittest import mock
        from mimir.client.ui.ws import session_store, transcript_log
        from mimir.client.ui.ws.ws_session import _Session

        self._tmp = tempfile.mkdtemp(prefix="mimir-detached-")
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        for target, attr in ((session_store, "STATE_DIR"), (transcript_log, "_MIMIR_DIR_WS")):
            patcher = mock.patch.object(target, attr, self._tmp)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.ws, self.worker = _FakeWS(), _FakeWorker()
        self.session = _Session(self.ws, self.worker)

        # A: the conversation that launched the job. B: the one being read now.
        self.a = self.session.store.new_session()
        self.a.title = "Install the toolchain"
        self.a.llm_history = [{"role": "user", "content": "build it"}]
        self.a.llm_history_full = list(self.a.llm_history)
        self.session.store.save_session(self.a)

        self.b = self.session.store.new_session()
        self.session.store.save_session(self.b)
        self.session._active_session_id = self.b.id
        self.session.history = [{"role": "user", "content": "something else"}]
        self.session.history_full = list(self.session.history)
        self.session._display_messages = []

    def _event(self, **over) -> dict:
        ev = {"type": "job_complete", "job_key": "20260909T170922Z-d123",
              "kind": "shell-command", "state": "done", "session_id": self.a.id,
              "status_op": {"tool": "bash_job", "args": {}},
              "summary": {"state": "done", "exit_code": 0, "log_tail": "Built target"}}
        ev.update(over)
        return ev

    def test_the_wake_runs_against_the_launching_session_not_the_visible_one(self) -> None:
        asyncio.run(self.session._handle_job_complete(self._event()))

        self.assertEqual(len(self.worker.submitted), 1)
        _text, history, session_id = self.worker.submitted[0]
        self.assertEqual(session_id, self.a.id)
        self.assertEqual(history[0]["content"], "build it")   # A's own history
        self.assertIn("Built target", history[-1]["content"])

        stored_a = self.session.store.load_session(self.a.id)
        self.assertEqual(len(stored_a.llm_history), 2)
        self.assertTrue(any("🔔" in m.get("text", "")
                            for m in stored_a.display_messages))

        # B — the conversation on screen — is untouched.
        self.assertEqual(self.session.history, [{"role": "user", "content": "something else"}])
        self.assertEqual(self.session._display_messages, [])

    def test_a_detached_answer_is_written_to_its_own_session(self) -> None:
        asyncio.run(self.session._handle_job_complete(self._event()))
        answer = {"type": "answer", "text": "Build finished, 0 errors.",
                  "session_id": self.a.id}

        # The drain loop would filter it out of this socket's stream ...
        self.assertTrue(self.session._is_foreign_event(answer))
        # ... which is exactly why it has to be persisted on its own path.
        asyncio.run(self.session._persist_detached_answer(answer))

        stored_a = self.session.store.load_session(self.a.id)
        self.assertEqual(stored_a.llm_history[-1],
                         {"role": "assistant", "content": "Build finished, 0 errors."})
        self.assertTrue(any(m.get("text") == "Build finished, 0 errors."
                            for m in stored_a.display_messages))
        self.assertEqual(self.session.history,
                         [{"role": "user", "content": "something else"}])

    def test_two_detached_turns_are_tracked_apart(self) -> None:
        # Two builds can finish into two different conversations before either
        # answers; the second wake must not evict the first's bookkeeping.
        c = self.session.store.new_session()
        c.llm_history = [{"role": "user", "content": "other work"}]
        c.llm_history_full = list(c.llm_history)
        self.session.store.save_session(c)

        asyncio.run(self.session._handle_job_complete(self._event()))
        asyncio.run(self.session._handle_job_complete(
            self._event(session_id=c.id, job_key="other-job")))

        asyncio.run(self.session._persist_detached_answer(
            {"type": "answer", "text": "A done", "session_id": self.a.id}))
        asyncio.run(self.session._persist_detached_answer(
            {"type": "answer", "text": "C done", "session_id": c.id}))

        self.assertEqual(self.session.store.load_session(self.a.id).llm_history[-1],
                         {"role": "assistant", "content": "A done"})
        self.assertEqual(self.session.store.load_session(c.id).llm_history[-1],
                         {"role": "assistant", "content": "C done"})

    def test_an_answer_for_a_turn_we_did_not_detach_is_ignored(self) -> None:
        # No detached turn in flight: nothing to write, and nothing to invent.
        asyncio.run(self.session._persist_detached_answer(
            {"type": "answer", "text": "stray", "session_id": self.a.id}))
        self.assertEqual(self.session.store.load_session(self.a.id).llm_history,
                         [{"role": "user", "content": "build it"}])

    def test_a_wake_for_the_visible_session_lands_in_it(self) -> None:
        asyncio.run(self.session._handle_job_complete(self._event(session_id=self.b.id)))
        self.assertEqual(self.worker.submitted[0][2], self.b.id)
        self.assertIn("Built target", self.session.history[-1]["content"])
        self.assertTrue(any("🔔" in m.get("text", "")
                            for m in self.session._display_messages))

    def test_a_prompt_from_a_detached_turn_still_reaches_the_user(self) -> None:
        # An approval or a question parks the turn until it is answered. Filtered as
        # foreign — which it is — it would park that turn forever.
        for kind in ("approval", "continue_prompt", "user_question"):
            with self.subTest(kind=kind):
                self.assertFalse(
                    self.session._is_foreign_event({"type": kind, "session_id": self.a.id}))
        self.assertTrue(
            self.session._is_foreign_event({"type": "token", "session_id": self.a.id}))

    def test_a_message_typed_here_is_not_steered_into_a_detached_turn(self) -> None:
        self.worker._query_session_id = self.a.id      # a detached turn is running
        self.assertFalse(self.session._running_turn_is_ours())
        self.worker._query_session_id = self.b.id      # our own turn is running
        self.assertTrue(self.session._running_turn_is_ours())


if __name__ == "__main__":
    unittest.main()


# ── 6. A burst of finished jobs is one turn, not one turn apiece ───────────────

class WakeCoalescingTests(unittest.TestCase):
    """Three jobs finishing together used to mean three turns and three answers.

    Measured in the field: three near-identical "the work is done" summaries inside
    3.5 minutes, one of them saying two jobs had finished and the next, seven seconds
    later, saying three — because each turn had seen a different subset. The wake
    mechanism working is what created this; nothing about it was wrong before it fired.
    """

    def setUp(self) -> None:
        import shutil
        import tempfile
        from unittest import mock
        from mimir.client.ui.ws import session_store, transcript_log
        from mimir.client.ui.ws.ws_session import _Session

        self._tmp = tempfile.mkdtemp(prefix="mimir-coalesce-")
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        for target, attr in ((session_store, "STATE_DIR"), (transcript_log, "_MIMIR_DIR_WS")):
            patcher = mock.patch.object(target, attr, self._tmp)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.ws, self.worker = _FakeWS(), _FakeWorker()
        self.session = _Session(self.ws, self.worker)
        self.here = self.session.store.new_session()
        self.session.store.save_session(self.here)
        self.session._active_session_id = self.here.id
        self.session.history = [{"role": "user", "content": "go"}]
        self.session.history_full = list(self.session.history)
        self.session._display_messages = []

    def _event(self, job_key: str, session_id: str | None = None) -> dict:
        return {"type": "job_complete", "job_key": job_key, "kind": "shell-command",
                "state": "done", "session_id": session_id or self.here.id,
                "status_op": {"tool": "bash_job", "args": {}},
                "summary": {"state": "done", "exit_code": 0}}

    def _running(self, session_id: str | None) -> None:
        """Pretend a turn of *session_id* is in flight."""
        self.worker._query_session_id = session_id

    def _wake(self, ev: dict) -> None:
        asyncio.run(self.session._handle_job_complete(ev))

    def _answer(self) -> None:
        self._running(None)
        asyncio.run(self.session._flush_pending_wakes(self.here.id))

    # ── idle: unchanged behaviour ────────────────────────────────────────────

    def test_a_job_finishing_while_idle_starts_its_turn_at_once(self) -> None:
        self._wake(self._event("j1"))
        self.assertEqual(len(self.worker.submitted), 1)
        self.assertIn("j1", self.worker.submitted[0][0])
        self.assertEqual(self.worker.steered, [])

    # ── busy: steered into the turn that is already running ──────────────────

    def test_a_job_finishing_mid_turn_is_handed_to_that_turn(self) -> None:
        self._running(self.here.id)
        self._wake(self._event("j1"))
        # No second turn: the running one is told, and carries on.
        self.assertEqual(self.worker.submitted, [])
        self.assertEqual(len(self.worker.steered), 1)
        self.assertIn("j1", self.worker.steered[0])

    def test_a_burst_becomes_one_catch_up_turn(self) -> None:
        self._running(self.here.id)
        for key in ("j1", "j2", "j3"):
            self._wake(self._event(key))
        self.assertEqual(self.worker.submitted, [])
        self._answer()
        # One turn, naming all three — not three turns naming one each.
        self.assertEqual(len(self.worker.submitted), 1)
        text = self.worker.submitted[0][0]
        for key in ("j1", "j2", "j3"):
            self.assertIn(key, text)

    def test_an_injected_wake_needs_no_catch_up(self) -> None:
        self._running(self.here.id)
        self._wake(self._event("j1"))
        self.session._drop_injected_wakes(self.worker.steered[0])
        self._answer()
        self.assertEqual(self.worker.submitted, [], "the running turn already had it")

    def test_only_the_unconfirmed_half_is_carried(self) -> None:
        self._running(self.here.id)
        self._wake(self._event("j1"))
        self._wake(self._event("j2"))
        self.session._drop_injected_wakes(self.worker.steered[0])   # j1 only
        self._answer()
        self.assertEqual(len(self.worker.submitted), 1)
        text = self.worker.submitted[0][0]
        self.assertIn("j2", text)
        self.assertNotIn("j1", text)

    # ── the invariant: nothing is ever swallowed ─────────────────────────────

    def test_every_job_is_reported_exactly_once(self) -> None:
        """A steer the loop never took in must still reach a turn.

        This is the failure the opportunistic injection could introduce, and it would
        be invisible in a session: the job finished, the history carries the message,
        and no turn ever reads it.
        """
        self._running(self.here.id)
        for key in ("j1", "j2", "j3"):
            self._wake(self._event(key))
        self.session._drop_injected_wakes(self.worker.steered[1])   # j2 taken in
        self._answer()

        carried = "".join(t for t, _h, _s in self.worker.submitted)
        absorbed = self.worker.steered[1]        # the one the loop confirmed taking in
        for key in ("j1", "j2", "j3"):
            routes = [key in absorbed, key in carried]
            self.assertEqual(routes.count(True), 1,
                             f"{key} reached {routes.count(True)} turns, not exactly one")
        self.assertIn("j2", absorbed)
        self.assertNotIn("j2", carried)

    # ── the record, whichever route the wake took ────────────────────────────

    def test_a_steered_wake_is_still_written_to_the_record(self) -> None:
        """The log is how the mechanism is checked; the quiet route must appear in it.

        The history is the one thing it must NOT write: the running turn receives the
        wake in its own messages, and adding it here as well put the same text in twice.
        """
        before = list(self.session.history)
        self._running(self.here.id)
        self._wake(self._event("j1"))
        self.assertEqual(self.worker.submitted, [])
        self.assertTrue(any("🔔" in m.get("text", "")
                            for m in self.session._display_messages))
        self.assertEqual(self.session.history, before)

    def test_a_steered_wake_is_recorded_once_even_if_a_later_flush_retells_it(self) -> None:
        """The re-tell is for the model, not for the record.

        An unconfirmed steer must still reach a turn, so the catch-up message repeats
        it — but the user was already shown it and the log already has it. Counting it
        twice was the first thing this coalescing got wrong.
        """
        self._running(self.here.id)
        self._wake(self._event("jA"))            # steered
        self._running(None)
        self._wake(self._event("jB"))            # idle → flushes jA and jB together

        notes = [m for m in self.session._display_messages if "🔔" in m.get("text", "")]
        self.assertEqual(len(notes), 2, "one notice per job, not one per telling")
        # The model is told about both, because jA's steer was never confirmed read.
        self.assertEqual(len(self.worker.submitted), 1)
        text = self.worker.submitted[0][0]
        self.assertIn("jA", text)
        self.assertIn("jB", text)
        # ...but the history carries jA once, not once per route.
        occurrences = sum(str(m.get("content", "")).count("jA") for m in self.session.history)
        self.assertEqual(occurrences, 1)

    # ── a wake for another conversation is never steered into this turn ──────

    def test_a_detached_wake_is_not_injected_into_the_visible_turn(self) -> None:
        other = self.session.store.new_session()
        self.session.store.save_session(other)
        self._running(self.here.id)          # the turn in flight is THIS session's
        self._wake(self._event("j9", session_id=other.id))
        self.assertEqual(self.worker.steered, [], "that turn is not its conversation")
        self.assertEqual(len(self.worker.submitted), 1)
        self.assertEqual(self.worker.submitted[0][2], other.id)
