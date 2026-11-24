"""Other agents on the same tasks: a stand-in agent run end to end through the meter, and
what each real agent is given (its config, its command, its environment)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from mimir.evals.baselines import Endpoint, ExternalAgent, metered, run_external
from mimir.evals.baselines.agents import Copilot, MiniSweAgent, OpenCode
from mimir.evals.meter import Meter, make_server
from mimir.runner import BenchTask
from mimir.tests.test_meter import _Upstream

# A stand-in agent: one model call through the endpoint it is given, then a file written
# in its working directory, then its answer on stdout.
_AGENT = """
import json, os, sys, time, httpx
base, key, mode = sys.argv[1], sys.argv[2], sys.argv[3]
httpx.post(base + "/chat/completions", headers={"Authorization": "Bearer " + key},
           json={"model": "glm", "messages": [{"role": "user", "content": "q"}]}, timeout=10)
if mode == "slow":
    time.sleep(30)
open("fixed.txt", "w").write("done")
print("all fixed, home " + os.environ["XDG_CONFIG_HOME"])
"""


class _StandIn(ExternalAgent):
    name = "stand-in"

    def __init__(self, script: str, mode: str = "ok") -> None:
        self.script, self.mode = script, mode
        self.executable = sys.executable

    def command(self, task, workspace, home, endpoint):
        return [sys.executable, self.script, endpoint.base_url, endpoint.run_tag, self.mode]


class _Files:
    """Passes when the agent left ``fixed.txt`` in the workspace."""

    name = "files"

    def __init__(self, timeout: float | None = None) -> None:
        self.timeout = timeout
        self.answers: list[str] = []

    def load_tasks(self, limit=None):
        return [BenchTask(id="fix", query="fix it", servers=(), timeout_secs=self.timeout)]

    async def score(self, task, answer, ctx):
        self.answers.append(answer)
        return os.path.exists(os.path.join(ctx.workspace, "fixed.txt")), "checked"


def _serve(server) -> str:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return "http://%s:%d" % server.server_address[:2]


class RunExternalTests(unittest.TestCase):
    def setUp(self) -> None:
        _Upstream.seen, _Upstream.paths = [], []
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
        self.addCleanup(upstream.shutdown)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.log = os.path.join(self.dir, "meter.jsonl")
        meter = make_server(Meter(_serve(upstream), {}, self.log))
        self.addCleanup(meter.shutdown)
        self.meter = _serve(meter)
        self.script = os.path.join(self.dir, "agent.py")
        Path(self.script).write_text(_AGENT)

    def _run(self, agent, adapter, tag="run-1"):
        task = adapter.load_tasks()[0]
        return asyncio.run(run_external(task, adapter, agent,
                                        Endpoint(self.meter + "/v1", "glm", tag), self.log))

    def test_the_agent_s_work_is_graded_and_its_calls_counted_from_the_meter(self) -> None:
        adapter = _Files()
        result = self._run(_StandIn(self.script), adapter)
        self.assertTrue(result.passed, result.error)
        self.assertEqual((result.model_calls, result.prompt_tokens, result.cached_tokens),
                         (1, 120, 100))
        self.assertEqual((result.variant, result.run_tag), ("stand-in", "run-1"))
        self.assertTrue(adapter.answers[0].startswith("all fixed"))

    def test_the_agent_runs_in_a_home_of_its_own_that_goes_after(self) -> None:
        adapter = _Files()
        self._run(_StandIn(self.script), adapter)
        config_home = adapter.answers[0].split("home ", 1)[1]
        self.assertNotEqual(config_home, os.path.expanduser("~/.config"))
        self.assertFalse(os.path.exists(config_home))

    def test_past_its_timeout_the_agent_is_stopped_and_graded_as_it_stands(self) -> None:
        result = self._run(_StandIn(self.script, mode="slow"), _Files(timeout=3))
        self.assertTrue(result.timed_out)
        self.assertFalse(result.passed)
        self.assertEqual(result.model_calls, 1)

    def test_an_agent_not_installed_is_skipped(self) -> None:
        agent = _StandIn(self.script)
        agent.executable = "no-such-agent-mimir-test"
        result = self._run(agent, _Files())
        self.assertTrue(result.skipped)
        self.assertIn("no-such-agent-mimir-test", result.detail)

    def test_no_record_means_nothing_reported(self) -> None:
        self.assertEqual(metered([])["prompt_tokens"], None)


class ConditionTests(unittest.TestCase):
    """The agents run in the evaluation loop, MIMIR named ``mimir`` beside them."""

    def test_each_task_runs_under_mimir_then_each_agent_in_one_report(self) -> None:
        from mimir.evals.__main__ import _run
        from mimir.evals.report import load_runs
        from mimir.tests.support.backend import ScriptedBackend

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = os.path.join(tmp.name, "report.json")
        log = os.path.join(tmp.name, "meter.jsonl")
        script = os.path.join(tmp.name, "agent.py")
        Path(script).write_text(_AGENT)
        args = argparse.Namespace(adapter="mimir.tests.test_baselines:_Files", adapter_arg=[],
                                  variant=[], repeat=1, limit=None, model="glm",
                                  backend="vllm", enforcement=None, out=out, resume=False,
                                  keep_workspaces=False, agent=["stand-in"],
                                  meter="http://127.0.0.1:9", meter_log=log)

        async def fake_external(task, adapter, agent, endpoint, meter_log, keep_workspace):
            from mimir.runner import RunResult
            return RunResult(task.id, True, "stand-in ran", variant=agent.name,
                             run_tag=endpoint.run_tag)

        with patch("mimir.evals.__main__.known_agents", lambda: {"stand-in": _StandIn(script)}), \
                patch("mimir.evals.__main__.run_external", fake_external):
            asyncio.run(_run(args, get_backend_override=lambda: ScriptedBackend([])))
        runs = load_runs([out])
        self.assertEqual([r["variant"] for r in runs], ["mimir", "stand-in"])
        self.assertTrue(runs[0]["run_tag"].startswith("mimir-fix-r0-"))
        self.assertTrue(runs[1]["run_tag"].startswith("stand-in-fix-r0-"))

    def test_other_agents_need_the_meter(self) -> None:
        from mimir.evals.__main__ import _conditions

        args = argparse.Namespace(agent=["opencode"], meter=None, meter_log=None)
        with self.assertRaises(SystemExit):
            _conditions(args, [None], None)


class AgentConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.endpoint = Endpoint("http://meter:8099/v1", "glm-5.3", "run-7")
        self.task = BenchTask(id="t", query="fix the bug", max_steps=60)

    def test_mini_swe_agent_calls_the_meter_in_the_workspace_within_the_steps(self) -> None:
        agent = MiniSweAgent()
        agent.prepare(self.task, "/ws", self.home, self.endpoint)
        config = json.loads((self.home / "mimir-run.yaml").read_text())
        self.assertEqual(config["model"]["model_kwargs"],
                         {"api_base": "http://meter:8099/v1", "api_key": "run-7"})
        self.assertEqual(config["environment"]["cwd"], "/ws")
        self.assertEqual(config["agent"]["step_limit"], 60)
        command = agent.command(self.task, "/ws", self.home, self.endpoint)
        self.assertEqual(command[:3], ["mini", "-t", "fix the bug"])
        self.assertIn("--exit-immediately", command)

    def test_mini_swe_agent_answers_with_its_submission(self) -> None:
        (self.home / "trajectory.json").write_text(json.dumps({"info": {"submission": "patched"}}))
        self.assertEqual(MiniSweAgent().answer("noise", self.home), "patched")
        (self.home / "trajectory.json").unlink()
        self.assertEqual(MiniSweAgent().answer(" printed ", self.home), "printed")

    def test_opencode_s_provider_is_the_meter_and_its_tools_are_allowed(self) -> None:
        agent = OpenCode(compaction=False)
        agent.prepare(self.task, "/ws", self.home, self.endpoint)
        config = json.loads((self.home / ".config/opencode/opencode.json").read_text())
        self.assertEqual(config["model"], "meter/glm-5.3")
        self.assertEqual(config["providers"]["meter"]["settings"]["baseURL"],
                         "http://meter:8099/v1")
        self.assertEqual(config["compaction"], {"auto": False})
        self.assertEqual(agent.environment(self.home, self.endpoint),
                         {"MIMIR_METER_KEY": "run-7"})

    def test_copilot_runs_offline_on_our_model(self) -> None:
        env = Copilot().environment(self.home, self.endpoint)
        self.assertEqual((env["COPILOT_PROVIDER_BASE_URL"], env["COPILOT_PROVIDER_API_KEY"],
                          env["COPILOT_MODEL"], env["COPILOT_OFFLINE"]),
                         ("http://meter:8099/v1", "run-7", "glm-5.3", "true"))
        self.assertIn("--allow-all-tools",
                      Copilot().command(self.task, "/ws", self.home, self.endpoint))


if __name__ == "__main__":
    unittest.main()
