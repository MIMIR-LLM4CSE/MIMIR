"""Integration tests for the proxy optimization ratchet (accept/reject/converge).

Drives ``eval_session.results`` over fabricated completed runs — the same
"never actually launch the runner" approach as test_proxy_ops.py — and asserts
the ratchet verdict, best-so-far tracking, ledger monotonicity, and the
reset_to_best rollback op.

Run:
    python -m unittest mimir.tests.test_proxy_opt_loop -v
"""

import json
import os

from mimir.tests.test_proxy_ops import _TmpStorageTest, procs, ratchet, server_proxy, store


class RatchetLoopTests(_TmpStorageTest):
    def _init(self, max_stall: int = 2) -> None:
        self._register()
        server_proxy.proxy_manage(
            op="suite_define", name="bench",
            cases=[{"case_id": "a", "proxy_name": "tiny"}], confirm=True,
        )
        self.source = os.path.join(self.root, "source.py")
        self._write_source("v0")
        res = server_proxy.proxy_eval(
            op="init", proxy_name="tiny", benchmark_name="bench",
            requirements=[{"metric": "time_s", "operator": "lt", "threshold": 100.0}],
            proxy_source_path=self.source,
            primary_metric="time_s", primary_goal="min", max_stall=max_stall,
            confirm=True,
        )
        self.assertEqual(res.get("status"), "ok")
        self.assertIn("minimize time_s", res["objective"])

    def _write_source(self, tag: str) -> None:
        with open(self.source, "w") as fh:
            fh.write(f"VERSION = '{tag}'\n")

    def _complete_run(self, run_id: str, all_passed: bool, time_s: float,
                      wall_time_s: float | None = None) -> str:
        """Fabricate a finished run dir and point the active symlink at it."""
        session_dir = store._opt_session_runs_dir("tiny")
        run_dir = os.path.join(session_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        case_metrics = {"time_s": time_s}
        if wall_time_s is not None:
            case_metrics["wall_time_s"] = wall_time_s
        with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
            json.dump({
                "all_passed": all_passed,
                "best_time_s": time_s if all_passed else None,
                "results": [{"metrics": case_metrics}],
            }, fh)
        # A summary line so the log-parsing path is exercised too.
        with open(procs._log_path(run_dir), "w") as fh:
            fh.write(f"[proxy_runner] summary cases_passed={1 if all_passed else 0} "
                     f"cases_total=1 all_passed={all_passed} best_case=a "
                     f"best_time_s={time_s}\n")
        procs._update_opt_active_link("tiny", run_dir)
        return run_dir

    def _results(self) -> dict:
        return server_proxy.proxy_eval_status(op="results")

    def test_full_ratchet_lifecycle(self) -> None:
        self._init(max_stall=2)

        # Run A: first feasible run -> accepted, becomes best (source snapshot 'A').
        self._write_source("A")
        self._complete_run("20240101T000000Z", all_passed=True, time_s=5.0)
        res = self._results()
        self.assertEqual(res["verdict"], "accept")
        self.assertEqual(res["best"]["run_id"], "20240101T000000Z")
        self.assertEqual(res["best"]["primary_value"], 5.0)

        # Run B: regression (breaks the constraint) -> rejected, best unchanged,
        # next_step steers to reset_to_best.
        self._write_source("B-broken")
        self._complete_run("20240101T000100Z", all_passed=False, time_s=0.1)
        res = self._results()
        self.assertEqual(res["verdict"], "reject")
        self.assertIn("reset_to_best", res["next_step"])
        self.assertEqual(res["best"]["run_id"], "20240101T000000Z")

        # reset_to_best restores the best-so-far source (version 'A'), not canonical.
        rb = server_proxy.proxy_eval(op="reset_to_best", confirm=True)
        self.assertEqual(rb.get("status"), "ok")
        with open(self.source) as fh:
            self.assertEqual(fh.read(), "VERSION = 'A'\n")

        # Run C: feasible improvement -> accepted, new best (3.0).
        self._write_source("C")
        self._complete_run("20240101T000200Z", all_passed=True, time_s=3.0)
        res = self._results()
        self.assertEqual(res["verdict"], "accept")
        self.assertEqual(res["best"]["primary_value"], 3.0)

        # Runs D, E: feasible but no improvement -> stall climbs; at max_stall=2
        # the session reports converged.
        self._write_source("D")
        self._complete_run("20240101T000300Z", all_passed=True, time_s=3.0)
        res_d = self._results()
        self.assertEqual(res_d["verdict"], "reject")
        self.assertEqual(res_d["stall"], 1)

        self._write_source("E")
        self._complete_run("20240101T000400Z", all_passed=True, time_s=3.0)
        res_e = self._results()
        self.assertEqual(res_e["verdict"], "converged")
        self.assertIn("reset_to_best", res_e["next_step"])

        # Ledger: best primary_value is monotone non-increasing (min goal).
        with open(store._opt_ledger_file("tiny")) as fh:
            entries = [json.loads(x) for x in fh if x.strip()]
        self.assertEqual(len(entries), 5)
        # best-so-far as recorded per entry never regresses
        seen_best = [e["best_run_id"] for e in entries]
        self.assertEqual(seen_best[0], "20240101T000000Z")   # A
        self.assertEqual(seen_best[-1], "20240101T000200Z")  # C stays best through the end

    def test_results_are_idempotent_across_polls(self) -> None:
        """Polling results() repeatedly must not move best/stall twice."""
        self._init(max_stall=5)
        self._write_source("A")
        self._complete_run("20240101T000000Z", all_passed=True, time_s=5.0)
        first = self._results()
        second = self._results()
        self.assertEqual(first["verdict"], second["verdict"])
        self.assertEqual(first["stall"], second["stall"])
        # Exactly one ledger entry despite two polls.
        with open(store._opt_ledger_file("tiny")) as fh:
            self.assertEqual(sum(1 for x in fh if x.strip()), 1)

    def test_reset_to_best_errors_without_a_best(self) -> None:
        self._init()
        res = server_proxy.proxy_eval(op="reset_to_best", confirm=True)
        self.assertEqual(res.get("status"), "error")
        self.assertIn("No best-so-far", res["error"])

    def test_infeasible_without_incumbent_keeps_editing(self) -> None:
        self._init()
        self._write_source("A")
        self._complete_run("20240101T000000Z", all_passed=False, time_s=0.1)
        res = self._results()
        self.assertIsNone(res["verdict"])
        self.assertIn("edit", res["next_step"])

    def test_timing_warning_when_wall_time_regresses(self) -> None:
        """time_s is self-reported: an accepted 'improvement' whose server-
        measured wall time regressed vs the incumbent must be flagged."""
        self._init(max_stall=5)
        self._write_source("A")
        self._complete_run("20240101T000000Z", all_passed=True, time_s=5.0,
                           wall_time_s=5.0)
        res = self._results()
        self.assertEqual(res["verdict"], "accept")
        self.assertNotIn("timing_warning", res)
        self.assertEqual(ratchet._load_best("tiny")["wall_value"], 5.0)

        # Self-reported time_s "improves" while the measured wall time blows
        # up: still accepted (the objective is time_s) but flagged for audit.
        self._write_source("B")
        self._complete_run("20240101T000100Z", all_passed=True, time_s=3.0,
                           wall_time_s=9.0)
        res = self._results()
        self.assertEqual(res["verdict"], "accept")
        self.assertIn("wall time", res["timing_warning"])
        self.assertIn("WARNING", res["recommendation"])
        self.assertEqual(res["wall_value"], 9.0)
        # The audit values persist in the ledger and the best pointer.
        self.assertEqual(ratchet._load_best("tiny")["wall_value"], 9.0)
        with open(store._opt_ledger_file("tiny")) as fh:
            entries = [json.loads(x) for x in fh if x.strip()]
        self.assertIn("timing_warning", entries[-1])
        self.assertEqual(entries[-1]["wall_value"], 9.0)


class RunnerConvergenceRequirementTests(_TmpStorageTest):
    """convergence_order is run-level: requiring it must not fail every case.

    Regression: the runner used to feed the full requirements list to the
    per-case evaluation, where convergence_order can never exist (it is
    fitted across the sweep), so any session requiring it had 0/N cases
    passing regardless of the actual metrics.
    """

    def _write_sweep_proxy(self) -> str:
        """A proxy whose error scales as h^2 across the {n} sweep."""
        path = os.path.join(self.root, "sweep.py")
        with open(path, "w") as fh:
            fh.write(
                "import sys\n"
                "import numpy as np\n"
                "out, n = sys.argv[1], int(sys.argv[2])\n"
                "if not out.endswith('.npz'):\n"
                "    out += '.npz'\n"
                "np.savez(out, field=np.ones(4))\n"
                "print('PROXY_METRICS_BEGIN')\n"
                "print('time_s=0.01')\n"
                "print(f'h={1.0 / n}')\n"
                "print(f'err={(1.0 / n) ** 2}')\n"
                "print(f'output_file={out}')\n"
                "print('PROXY_METRICS_END')\n"
            )
        return path

    def test_convergence_requirement_evaluated_at_run_level_only(self) -> None:
        import sys
        from unittest import mock

        exe = self._write_sweep_proxy()
        res = server_proxy.proxy_manage(
            op="register", name="conv", executable_path=exe,
            run_cmd_template=f"{sys.executable} {{executable}} {{output_file}} {{n}}",
            output_format="npz", confirm=True,
        )
        self.assertEqual(res.get("status"), "ok")
        res = server_proxy.proxy_exec(
            op="benchmark_create", proxy_name="conv", benchmark_name="convb",
            param_sweeps=[{"n": 10}, {"n": 20}, {"n": 40}],
            reference_params={"n": 40}, confirm=True,
        )
        self.assertEqual(res.get("status"), "ok")
        res = server_proxy.proxy_eval(
            op="init", proxy_name="conv", benchmark_name="convb",
            proxy_source_path=exe, primary_metric="time_s",
            convergence={"h_param": "h", "error_metric": "err"},
            requirements=[
                {"metric": "convergence_order", "operator": "gte", "threshold": 1.7},
                {"metric": "time_s", "operator": "lt", "threshold": 100.0},
            ],
            confirm=True,
        )
        self.assertEqual(res.get("status"), "ok")

        from mimir.tests.test_proxy_ops import eval_session
        cfg, err_response, run_dir = eval_session._prepare_run("conv")
        self.assertIsNone(err_response)

        import _proxy_runner
        with mock.patch.object(sys, "argv", ["_proxy_runner", "--run-dir", run_dir]):
            _proxy_runner.main()

        with open(os.path.join(run_dir, "metrics.json")) as fh:
            metrics = json.load(fh)
        self.assertEqual(metrics["cases_passed"], 3)
        self.assertTrue(metrics["all_passed"])
        self.assertAlmostEqual(metrics["convergence_order"], 2.0, delta=0.05)


class RunnerSettlesRatchetTests(_TmpStorageTest):
    """The runner settles ledger/best at completion — bookkeeping must not
    depend on the agent ever calling results().

    Regression: a winning run completed after the agent's tool call timed
    out; the agent only polled status, so the run was absent from the ledger
    and never became best until someone manually called results.
    """

    def _setup_session(self) -> str:
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
                "print('PROXY_METRICS_BEGIN')\n"
                "print('time_s=0.01')\n"
                "print(f'output_file={out}')\n"
                "print('PROXY_METRICS_END')\n"
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
        res = server_proxy.proxy_eval(
            op="init", proxy_name="fast", benchmark_name="fastb",
            proxy_source_path=exe, primary_metric="time_s",
            requirements=[{"metric": "time_s", "operator": "lt",
                           "threshold": 100.0}],
            confirm=True,
        )
        self.assertEqual(res.get("status"), "ok")
        return exe

    def _run_runner(self) -> str:
        import sys
        from unittest import mock
        from mimir.tests.test_proxy_ops import eval_session
        cfg, err_response, run_dir = eval_session._prepare_run("fast")
        self.assertIsNone(err_response)
        procs._update_opt_active_link("fast", run_dir)
        import _proxy_runner
        with mock.patch.object(sys, "argv",
                               ["_proxy_runner", "--run-dir", run_dir]):
            _proxy_runner.main()
        return run_dir

    def _ledger_lines(self) -> list[dict]:
        path = os.path.join(store._opt_session_runs_dir("fast"), "ledger.jsonl")
        if not os.path.isfile(path):
            return []
        with open(path) as fh:
            return [json.loads(x) for x in fh if x.strip()]

    def test_ledger_and_best_settle_without_results_call(self) -> None:
        self._setup_session()
        run_dir = self._run_runner()

        # No results() call — the runner itself must have settled everything.
        lines = self._ledger_lines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["verdict"], "accept")
        best = ratchet._load_best("fast")
        self.assertIsNotNone(best)
        self.assertEqual(best["run_id"], os.path.basename(run_dir))
        self.assertTrue(os.path.isfile(os.path.join(run_dir, "ratchet.json")))
        # Best snapshot comes from the launch-time copy of the source.
        self.assertTrue(os.path.isfile(
            os.path.join(run_dir, "source_at_launch.py")))

    def test_results_after_runner_settle_is_idempotent(self) -> None:
        self._setup_session()
        self._run_runner()
        res = server_proxy.proxy_eval_status(op="results", proxy_name="fast")
        self.assertEqual(res.get("verdict"), "accept")
        # Replay, not a second settle: still exactly one ledger entry.
        self.assertEqual(len(self._ledger_lines()), 1)


class ReferenceRequirementGuardTests(_TmpStorageTest):
    """init/configure must reject reference-dependent requirements that the
    benchmark setup can never satisfy (the dead-end an agent once escaped by
    forging the metric from inside the proxy)."""

    def _source(self) -> str:
        path = os.path.join(self.root, "source.py")
        with open(path, "w") as fh:
            fh.write("VERSION = 'v0'\n")
        return path

    def _init(self, requirements: list[dict], benchmark: str = "bench") -> dict:
        return server_proxy.proxy_eval(
            op="init", proxy_name="tiny", benchmark_name=benchmark,
            requirements=requirements, proxy_source_path=self._source(),
            confirm=True,
        )

    def test_init_rejects_reference_metric_without_reference(self) -> None:
        self._register()
        server_proxy.proxy_manage(
            op="suite_define", name="bench",
            cases=[{"case_id": "a", "proxy_name": "tiny"}], confirm=True,
        )
        res = self._init([{"metric": "conservation_residual",
                           "operator": "lt", "threshold": 1e-3}])
        self.assertEqual(res.get("status"), "error")
        self.assertIn("sealed reference", res["error"])
        self.assertIn("benchmark_create", res["error"])

    def test_init_rejects_conservation_without_conserved_metric(self) -> None:
        self._register()  # no conserved_metric in metadata
        server_proxy.proxy_manage(
            op="suite_define", name="bench",
            cases=[{"case_id": "a", "proxy_name": "tiny",
                    "reference_name": "some_ref"}], confirm=True,
        )
        res = self._init([{"metric": "conservation_residual",
                           "operator": "lt", "threshold": 1e-3}])
        self.assertEqual(res.get("status"), "error")
        self.assertIn("conserved_metric", res["error"])

    def test_init_accepts_when_setup_is_complete(self) -> None:
        self._register(metadata={"conserved_metric": "time_s"})
        server_proxy.proxy_manage(
            op="suite_define", name="bench",
            cases=[{"case_id": "a", "proxy_name": "tiny",
                    "reference_name": "some_ref"}], confirm=True,
        )
        res = self._init([
            {"metric": "conservation_residual", "operator": "lt",
             "threshold": 1e-3},
            {"metric": "time_s", "operator": "lt", "threshold": 100.0},
        ])
        self.assertEqual(res.get("status"), "ok")

    def test_configure_recheck_blocks_bad_requirements(self) -> None:
        self._register()
        server_proxy.proxy_manage(
            op="suite_define", name="bench",
            cases=[{"case_id": "a", "proxy_name": "tiny"}], confirm=True,
        )
        res = self._init([{"metric": "time_s", "operator": "lt",
                           "threshold": 100.0}])
        self.assertEqual(res.get("status"), "ok")
        res = server_proxy.proxy_eval(
            op="configure", proxy_name="tiny",
            requirements=[{"metric": "l2_rel", "operator": "lt",
                           "threshold": 1e-3}],
            confirm=True,
        )
        self.assertEqual(res.get("status"), "error")
        self.assertIn("sealed reference", res["error"])


if __name__ == "__main__":
    import unittest
    unittest.main()
