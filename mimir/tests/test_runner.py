"""The headless run engine, model-free with a ``ScriptedBackend`` and no servers.

The report shape, the auto-approve hook, a fresh agent and workspace per task, and the
skip path.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import unittest

from mimir.runner import BenchTask, CheckContext, run_benchmark
from mimir.tests.support.backend import ScriptedBackend


def _scripted_override():
    backend = ScriptedBackend([])  # empty → every chat() ends the turn

    def _get_backend():
        return backend

    return _get_backend


class _RecordingAdapter:
    """Minimal adapter: two server-less tasks; records what each scorer saw."""

    name = "test_recording"

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def load_tasks(self, limit=None):
        def _seed_a(ws: str) -> None:
            with open(os.path.join(ws, "from_task_a.txt"), "w") as fh:
                fh.write("a")

        tasks = [
            BenchTask(id="task_a", query="q", servers=(), max_steps=2, setup=_seed_a),
            BenchTask(id="task_b", query="q", servers=(), max_steps=2),
        ]
        return tasks[:limit] if limit is not None else tasks

    async def score(self, task, answer, ctx: CheckContext):
        self.seen.append({
            "task_id": task.id,
            "workspace": ctx.workspace,
            # The agent itself, not id(): an address freed between tasks is handed
            # straight back to the next allocation, so two distinct agents compared
            # by id() can and do collide. Holding the reference makes reuse
            # impossible, and `is` asks the question the test actually means.
            "agent": ctx.agent,
            # task_b must NOT see task_a's seeded file (workspace isolation):
            "leaked_a_file": os.path.isfile(os.path.join(ctx.workspace, "from_task_a.txt")),
        })
        return task.id == "task_a", f"scored {task.id}"


class RunBenchmarkReportTests(unittest.TestCase):
    def test_report_shape_and_per_task_results(self) -> None:
        adapter = _RecordingAdapter()
        summary = asyncio.run(run_benchmark(
            adapter, model="dummy", get_backend_override=_scripted_override(),
        ))
        self.assertEqual(summary["benchmark"], "test_recording")
        self.assertEqual(summary["backend"], "scripted")
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["passed"], 1)        # only task_a passes
        self.assertEqual(summary["pass_rate"], 0.5)
        ids = [r["task_id"] for r in summary["results"]]
        self.assertEqual(ids, ["task_a", "task_b"])
        self.assertEqual([r["error"] for r in summary["results"]], ["", ""])

    def test_limit_caps_task_count(self) -> None:
        adapter = _RecordingAdapter()
        summary = asyncio.run(run_benchmark(
            adapter, model="dummy", limit=1, get_backend_override=_scripted_override(),
        ))
        self.assertEqual(summary["total"], 1)
        self.assertEqual([r["task_id"] for r in summary["results"]], ["task_a"])


class IsolationTests(unittest.TestCase):
    def test_fresh_agent_and_workspace_per_task(self) -> None:
        adapter = _RecordingAdapter()
        asyncio.run(run_benchmark(
            adapter, model="dummy", get_backend_override=_scripted_override(),
        ))
        self.assertEqual(len(adapter.seen), 2)
        a, b = adapter.seen
        # Distinct workspaces and distinct agent instances → no cross-task leakage.
        self.assertNotEqual(a["workspace"], b["workspace"])
        self.assertIsNot(a["agent"], b["agent"])
        # task_b's workspace never received task_a's seed file.
        self.assertFalse(b["leaked_a_file"])

    def test_cwd_restored_after_run(self) -> None:
        before = os.getcwd()
        asyncio.run(run_benchmark(
            _RecordingAdapter(), model="dummy", get_backend_override=_scripted_override(),
        ))
        self.assertEqual(os.getcwd(), before)


class _ApprovalAdapter:
    """One task whose scorer asks the live agent for an approval nobody is there to give."""

    name = "test_approval"

    def __init__(self) -> None:
        self.verdict: tuple[bool, str] | None = None

    def load_tasks(self, limit=None):
        return [BenchTask(id="task", query="q", servers=(), max_steps=1)]

    async def score(self, task, answer, ctx: CheckContext):
        self.verdict = ctx.agent.request_tool_approval("bash_run", {"command": "ls"})
        return True, "asked"


class AutoApproveTests(unittest.TestCase):
    def test_an_unattended_run_approves_without_asking_anyone(self) -> None:
        """The agent asks whoever its host lends; a benchmark lends nobody at a terminal."""
        adapter = _ApprovalAdapter()
        asyncio.run(run_benchmark(adapter, model="dummy",
                                  get_backend_override=_scripted_override()))
        approved, reason = adapter.verdict
        self.assertTrue(approved, reason)
        self.assertIn("auto-approve", reason)


class SkipTaskAdapter:
    """One task that requires a non-existent executable → must be skipped."""

    name = "test_skip"

    def load_tasks(self, limit=None):
        return [BenchTask(
            id="needs_bogus", query="q", servers=(), max_steps=2,
            requires=("definitely-not-a-real-binary-xyz",),
        )]

    async def score(self, task, answer, ctx):  # must never be called for a skipped task
        raise AssertionError("score() must not run for a skipped task")


class SkipSupportTests(unittest.TestCase):
    def test_missing_requirement_is_skipped_not_run(self) -> None:
        summary = asyncio.run(run_benchmark(
            SkipTaskAdapter(), model="dummy", get_backend_override=_scripted_override(),
        ))
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["passed"], 0)
        self.assertEqual(summary["pass_rate"], 0.0)  # scored=0 → 0.0, no ZeroDivision
        self.assertTrue(summary["results"][0]["skipped"])



class _ModuleAdapter:
    name = "test_module"

    def load_tasks(self, limit=None):
        return [BenchTask(id="needs_a_module", query="q", servers=(),
                          modules=("no_such_module_mimir_test",))]

    async def score(self, task, answer, ctx):
        raise AssertionError("a task missing its module is never run")


class MissingModuleTests(unittest.TestCase):
    def test_a_task_missing_a_python_module_is_skipped_not_failed(self) -> None:
        summary = asyncio.run(run_benchmark(_ModuleAdapter(), model="dummy",
                                            get_backend_override=_scripted_override()))
        result = summary["results"][0]
        self.assertTrue(result["skipped"])
        self.assertIn("no_such_module_mimir_test", result["detail"])



class MeasuresTests(unittest.TestCase):
    """What a run reports beyond pass or fail: its cost, what it was told, under which prose."""

    def _run(self, responses: list[dict], variant=None) -> dict:
        from mimir.client.prompt.variants import use_variant
        backend = ScriptedBackend(responses)
        with use_variant(variant):
            summary = asyncio.run(run_benchmark(_RecordingAdapter(), model="dummy", limit=1,
                                                get_backend_override=lambda: backend))
        return summary["results"][0]

    def test_model_calls_and_reported_prompt_tokens_are_counted(self) -> None:
        result = self._run([{"role": "assistant", "content": "done", "prompt_tokens": 1200}])
        self.assertGreaterEqual(result["model_calls"], 1)
        self.assertEqual(result["prompt_tokens"], 1200)
        self.assertGreaterEqual(result["elapsed_secs"], 0.0)

    def test_a_backend_that_reports_no_size_says_so(self) -> None:
        self.assertIsNone(self._run([{"role": "assistant", "content": "done"}])["prompt_tokens"])

    def test_the_variant_in_force_is_named(self) -> None:
        self.assertEqual(self._run([], {"name": "terse"})["variant"], "terse")
        self.assertEqual(self._run([])["variant"], "")

    def test_reminders_are_counted_by_category(self) -> None:
        from mimir.runner.engine import _CountingBackend, _measures
        events = [{"type": "nudge_injected", "category": "todo"},
                  {"type": "nudge_injected", "category": "todo"},
                  {"type": "nudge_injected", "category": "validation"},
                  {"type": "tool_call"}]
        measures = _measures(events, _CountingBackend(lambda: None), 0.0)
        self.assertEqual(measures["nudges"], {"todo": 2, "validation": 1})
        self.assertEqual(measures["tool_calls"], 1)


class WorkspaceLifetimeTests(unittest.TestCase):
    def test_a_workspace_goes_once_graded_unless_kept(self) -> None:
        for keep in (False, True):
            adapter = _RecordingAdapter()
            asyncio.run(run_benchmark(adapter, model="dummy", limit=1, keep_workspaces=keep,
                                      get_backend_override=_scripted_override()))
            workspace = adapter.seen[0]["workspace"]
            self.assertEqual(os.path.isdir(workspace), keep)
            if keep:
                shutil.rmtree(workspace)


class _SlowBackend(ScriptedBackend):
    """A model call that lasts until the turn is stopped, or ten seconds."""

    def chat(self, *args, cancel_flag=None, **kwargs):
        if cancel_flag is not None:
            cancel_flag.wait(10)
        return super().chat(*args, cancel_flag=cancel_flag, **kwargs)


class _TimedAdapter:
    name = "test_timed"

    def __init__(self) -> None:
        self.answers: list[str] = []

    def load_tasks(self, limit=None):
        return [BenchTask(id="slow", query="q", servers=(), max_steps=2, timeout_secs=0.2)]

    async def score(self, task, answer, ctx):
        self.answers.append(answer)
        return False, "graded as it stood"


class TimeoutTests(unittest.TestCase):
    def test_a_run_past_its_timeout_is_stopped_and_graded_as_it_stands(self) -> None:
        adapter, backend = _TimedAdapter(), _SlowBackend([])
        summary = asyncio.run(run_benchmark(adapter, model="dummy",
                                            get_backend_override=lambda: backend))
        result = summary["results"][0]
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["detail"], "graded as it stood")
        self.assertEqual(adapter.answers, [""])
        self.assertLess(result["elapsed_secs"], 5)


class _BatchAdapter:
    """Two tasks collected from their workspace, graded together; *verdicts* is what grade says."""

    name = "test_batch"

    def __init__(self, verdicts=None, crash: bool = False) -> None:
        self.verdicts = verdicts
        self.crash = crash
        self.batches: list[tuple[str, list[dict]]] = []

    def load_tasks(self, limit=None):
        def _seed(ws: str) -> None:
            with open(os.path.join(ws, "work.txt"), "w") as fh:
                fh.write("seeded")

        return [BenchTask(id=i, query="q", servers=(), max_steps=1, setup=_seed)
                for i in ("one", "two")]

    async def collect(self, task, answer, ctx):
        with open(os.path.join(ctx.workspace, "work.txt")) as fh:
            return {"content": fh.read(), "task": task.id}

    async def grade(self, runs, label):
        self.batches.append((label, [r.artifact for r in runs]))
        if self.crash:
            raise RuntimeError("harness down")
        return self.verdicts


class BatchGradeTests(unittest.TestCase):
    def test_what_each_workspace_held_is_graded_in_one_batch(self) -> None:
        adapter = _BatchAdapter({"one": (True, "resolved"), "two": (False, "unresolved")})
        summary = asyncio.run(run_benchmark(adapter, model="dummy",
                                            get_backend_override=_scripted_override()))
        self.assertEqual(len(adapter.batches), 1)
        label, artifacts = adapter.batches[0]
        self.assertTrue(label.startswith("test_batch-"))
        self.assertEqual(artifacts, [{"content": "seeded", "task": "one"},
                                     {"content": "seeded", "task": "two"}])
        self.assertEqual([(r["passed"], r["detail"], r["pending"]) for r in summary["results"]],
                         [(True, "resolved", False), (False, "unresolved", False)])
        self.assertEqual(summary["pass_rate"], 0.5)

    def test_a_grader_may_skip_a_run(self) -> None:
        adapter = _BatchAdapter({"one": (False, "no docker", True), "two": (False, "no docker", True)})
        summary = asyncio.run(run_benchmark(adapter, model="dummy",
                                            get_backend_override=_scripted_override()))
        self.assertEqual(summary["skipped"], 2)

    def test_a_run_without_a_verdict_fails_and_says_so(self) -> None:
        adapter = _BatchAdapter({"one": (True, "resolved")})
        summary = asyncio.run(run_benchmark(adapter, model="dummy",
                                            get_backend_override=_scripted_override()))
        self.assertEqual(summary["results"][1]["detail"], "the grader gave no verdict")
        self.assertFalse(summary["results"][1]["passed"])

    def test_a_crashed_grade_fails_every_run_with_its_trace(self) -> None:
        summary = asyncio.run(run_benchmark(_BatchAdapter(crash=True), model="dummy",
                                            get_backend_override=_scripted_override()))
        for result in summary["results"]:
            self.assertEqual(result["detail"], "the grade crashed")
            self.assertIn("harness down", result["error"])


class _ProbeAdapter:
    """One task; its scorer runs *probe(task, ctx)* and keeps what it returns."""

    name = "test_probe"

    def __init__(self, probe, **task_fields) -> None:
        self.probe = probe
        self.task_fields = task_fields
        self.seen = None

    def load_tasks(self, limit=None):
        return [BenchTask(id="probe", query="q", servers=(), max_steps=2, **self.task_fields)]

    async def score(self, task, answer, ctx):
        self.seen = self.probe(task, ctx)
        return True, "probed"


class UsageTests(unittest.TestCase):
    def _run(self, adapter, responses) -> dict:
        backend = ScriptedBackend(responses)
        summary = asyncio.run(run_benchmark(adapter, model="dummy",
                                            get_backend_override=lambda: backend))
        return summary["results"][0]

    def test_every_call_s_usage_is_summed_and_the_loop_s_prompt_sizes_kept(self) -> None:
        result = self._run(_ProbeAdapter(lambda t, c: None), [
            {"role": "assistant", "content": "done", "prompt_tokens": 900,
             "completion_tokens": 40, "cached_tokens": 600}])
        self.assertEqual((result["prompt_tokens"], result["completion_tokens"],
                          result["cached_tokens"]), (900, 40, 600))
        self.assertEqual(result["prompt_curve"], [900])
        self.assertEqual(result["peak_prompt_tokens"], 900)
        self.assertGreaterEqual(result["answer_secs"], 0.0)
        self.assertLessEqual(result["answer_secs"], result["elapsed_secs"])

    def test_a_compaction_goes_through_the_run_s_backend_as_a_side_call(self) -> None:
        def compact(task, ctx):
            return ctx.agent.compact_messages([{"role": "user", "content": "long"},
                                               {"role": "assistant", "content": "work"}])

        adapter = _ProbeAdapter(compact)
        result = self._run(adapter, [
            {"role": "assistant", "content": "done", "prompt_tokens": 500},
            {"role": "assistant", "content": "the summary", "prompt_tokens": 300}])
        self.assertIn("the summary", adapter.seen[0]["content"])
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["side_calls"], {"compaction": 1})
        self.assertEqual(result["prompt_tokens"], 800)
        self.assertEqual(result["prompt_curve"], [500])

    def test_the_context_a_task_asks_for_is_the_agent_s(self) -> None:
        adapter = _ProbeAdapter(
            lambda t, c: (c.agent.context_mode, c.agent.max_context_tokens,
                          c.agent.intra_query_compaction),
            context_mode="compact", max_context_tokens=16_000, compaction=False)
        self._run(adapter, [])
        self.assertEqual(adapter.seen, ("compact", 16_000, False))

    def test_a_refusing_run_denies_and_records_each_request(self) -> None:
        adapter = _ProbeAdapter(
            lambda t, c: c.agent.request_tool_approval("bash_run", {"command": "curl x"}),
            approval="refuse")
        result = self._run(adapter, [])
        approved, _ = adapter.seen
        self.assertFalse(approved)
        self.assertEqual(result["refused"],
                         [{"tool": "bash_run", "arguments": {"command": "curl x"}}])

    def test_an_unknown_approval_mode_fails_the_task(self) -> None:
        result = self._run(_ProbeAdapter(lambda t, c: None, approval="maybe"), [])
        self.assertEqual(result["detail"], "run crashed")
        self.assertIn("unknown approval", result["error"])

    def test_context_passes_are_counted_by_kind(self) -> None:
        from mimir.runner.engine import _CountingBackend, _measures
        events = [{"type": "status", "text": "x", "context": {"kind": "compact"}},
                  {"type": "status", "text": "y", "context": {"kind": "trim"}},
                  {"type": "status", "text": "z", "context": {"kind": "trim"}},
                  {"type": "status", "text": "plain"}]
        measures = _measures(events, _CountingBackend(lambda: None), 0.0)
        self.assertEqual(measures["context_events"], {"compact": 1, "trim": 2})


if __name__ == "__main__":
    unittest.main()
