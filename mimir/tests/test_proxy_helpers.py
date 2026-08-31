"""Unit tests for the proxy helper library in servers/proxy/_lib/.

Locks in the behavior of the consolidated helpers (_seal_reference,
_diff_run_dirs), the correctness fixes (_select_best_case, the _fallback_scan
leak, per-case timeout plumbing), the integrity guards (reserved metrics,
time_s plausibility, returncode gate), and the frozen
``MIMIR_PROXY_BENCH_DIR`` root override.

Run:
    python -m unittest tests.test_proxy_helpers -v
"""

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in (SERVERS_DIR / "_shared", SERVERS_DIR / "proxy"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from _lib import execute, metrics, procs, ratchet, report, store  # noqa: E402

try:
    import numpy as _np_top
    _HAVE_NUMPY_TOP = True
except ImportError:
    _HAVE_NUMPY_TOP = False


class CoerceTests(unittest.TestCase):
    def test_bool_int_float_str(self) -> None:
        self.assertIs(metrics._coerce("true"), True)
        self.assertIs(metrics._coerce("FALSE"), False)
        self.assertEqual(metrics._coerce("42"), 42)
        self.assertEqual(metrics._coerce("3.5"), 3.5)
        self.assertEqual(metrics._coerce("/abs/path"), "/abs/path")


class ParseMetricsBlockTests(unittest.TestCase):
    def test_block_is_parsed(self) -> None:
        stdout = textwrap.dedent("""\
            noise line
            PROXY_METRICS_BEGIN
            misfit=1.23e-3
            time_s=4.72
            converged=true
            output_file=/abs/out.npz
            PROXY_METRICS_END
            trailing noise
        """)
        m = metrics._parse_metrics_block(stdout)
        self.assertEqual(m["misfit"], 1.23e-3)
        self.assertEqual(m["time_s"], 4.72)
        self.assertIs(m["converged"], True)
        self.assertEqual(m["output_file"], "/abs/out.npz")
        self.assertNotIn("_fallback_scan", m)

    def test_fallback_scan_does_not_leak_marker(self) -> None:
        # No block markers -> fallback scan path. Must NOT inject _fallback_scan.
        stdout = "time_s=2.0\ndof=100\n"
        m = metrics._parse_metrics_block(stdout)
        self.assertEqual(m.get("time_s"), 2.0)
        self.assertEqual(m.get("dof"), 100)
        self.assertNotIn("_fallback_scan", m)


class EvaluateRequirementsTests(unittest.TestCase):
    def test_operators(self) -> None:
        mets = {"time_s": 5.0, "misfit": 0.001}
        reqs = [
            {"metric": "time_s", "operator": "lt", "threshold": 10},
            {"metric": "misfit", "operator": "lte", "threshold": 0.001},
        ]
        res = metrics._evaluate_requirements(mets, reqs)
        self.assertTrue(res["passed"])

    def test_missing_metric_fails(self) -> None:
        res = metrics._evaluate_requirements({}, [{"metric": "x", "operator": "lt", "threshold": 1}])
        self.assertFalse(res["passed"])
        self.assertEqual(res["results"][0]["note"], "metric not found in results")

    def test_non_numeric_fails(self) -> None:
        res = metrics._evaluate_requirements(
            {"x": "abc"}, [{"metric": "x", "operator": "gt", "threshold": 1}]
        )
        self.assertFalse(res["passed"])

    def test_invalid_operator_fails(self) -> None:
        res = metrics._evaluate_requirements(
            {"x": 1}, [{"metric": "x", "operator": "approx", "threshold": 1}]
        )
        self.assertFalse(res["passed"])


class SelectBestCaseTests(unittest.TestCase):
    def test_passing_case_with_lowest_time_wins(self) -> None:
        best, t = ratchet._select_best_case([("a", True, 5.0), ("b", True, 3.0), ("c", True, 8.0)])
        self.assertEqual(best, "b")
        self.assertEqual(t, 3.0)

    def test_failing_case_never_selected(self) -> None:
        best, t = ratchet._select_best_case([("a", False, 1.0), ("b", False, 2.0)])
        self.assertIsNone(best)
        self.assertIsNone(t)

    def test_none_time_sorts_last(self) -> None:
        # A passing case with a real time must beat a passing case whose time is None,
        # regardless of order (this was the pre-fix bug).
        best, t = ratchet._select_best_case([("a", True, None), ("b", True, 3.0)])
        self.assertEqual(best, "b")
        self.assertEqual(t, 3.0)

    def test_all_none_keeps_first_passing(self) -> None:
        best, t = ratchet._select_best_case([("a", True, None), ("b", True, None)])
        self.assertEqual(best, "a")
        self.assertIsNone(t)


class DiffRunDirsTests(unittest.TestCase):
    def _make_run(self, root: str, name: str, config: dict, metrics: dict) -> str:
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as fh:
            json.dump(config, fh)
        with open(os.path.join(d, "metrics.json"), "w") as fh:
            json.dump(metrics, fh)
        return d

    def test_numeric_and_nonnumeric_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = self._make_run(tmp, "a", {"n": 64, "mode": "cpu"},
                               {"time_s": 10.0, "label": "old"})
            b = self._make_run(tmp, "b", {"n": 128, "mode": "cpu"},
                               {"time_s": 8.0, "label": "new"})
            diff = report._diff_run_dirs(a, b)
            self.assertEqual(diff["config_diff"], {"n": [64, 128]})
            self.assertNotIn("mode", diff["config_diff"])
            md = diff["metrics_diff"]
            self.assertEqual(md["time_s"]["a"], 10.0)
            self.assertEqual(md["time_s"]["b"], 8.0)
            self.assertEqual(md["time_s"]["delta"], -2.0)
            self.assertAlmostEqual(md["time_s"]["pct_change"], -20.0)
            # non-numeric metric change is reported as {a, b}
            self.assertEqual(md["label"], {"a": "old", "b": "new"})


class SealReferenceTests(unittest.TestCase):
    """Integration test for _seal_reference using a tiny real proxy subprocess."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = self._tmp.name
        # Redirect all cache I/O into the temp dir (every path derives from the root).
        self._saved_root = store._CACHE_DIR
        store._CACHE_DIR = root

        # A minimal proxy: emits a metrics block + writes an .npz field.
        self._proxy = os.path.join(root, "tiny_proxy.py")
        with open(self._proxy, "w") as fh:
            fh.write(textwrap.dedent("""\
                import argparse, numpy as np
                p = argparse.ArgumentParser()
                p.add_argument("--output", required=True)
                a = p.parse_args()
                np.savez(a.output, field=np.arange(16, dtype=np.float64))
                print("PROXY_METRICS_BEGIN")
                print("time_s=0.01")
                print("misfit=0.5")
                print(f"output_file={a.output}")
                print("PROXY_METRICS_END")
            """))

    def tearDown(self) -> None:
        store._CACHE_DIR = self._saved_root
        self._tmp.cleanup()

    def test_seal_reference_stores_metrics_and_output(self) -> None:
        entry = {
            "name": "tiny",
            "executable_path": self._proxy,
            "run_cmd_template": f"{sys.executable} {{executable}} --output {{output_file}}",
            "output_format": "npz",
        }
        result, error = execute._seal_reference(entry, "tiny_ref")
        self.assertIsNone(error, msg=str(error))
        self.assertEqual(result["reference_name"], "tiny_ref")
        self.assertEqual(result["metrics"]["misfit"], 0.5)
        # output_file is consumed (stored), not echoed back as a metric.
        self.assertNotIn("output_file", result["metrics"])
        self.assertTrue(os.path.isfile(os.path.join(result["ref_dir"], "metrics.json")))
        self.assertTrue(result["output_stored"] and os.path.isfile(result["output_stored"]))

    def test_seal_reference_reports_launch_failure(self) -> None:
        entry = {
            "name": "broken",
            "executable_path": "/nonexistent/exe",
            "run_cmd_template": "/nonexistent/exe --output {output_file}",
            "output_format": "npz",
        }
        result, error = execute._seal_reference(entry, "broken_ref")
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertEqual(error.get("status"), "error")
        self.assertIn("error", error)


@unittest.skipUnless(_HAVE_NUMPY_TOP, "numpy required")
class PostRunFinalizeTests(unittest.TestCase):
    """A detached run must settle with the same metrics a synchronous one gets.

    The invariants (`finite`, lifted error norms, `conservation_residual`) are
    computed server-side precisely so the proxy cannot forge them — a detached
    path that skips them hands the ratchet an unguarded metrics.json.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self._saved_root = store._CACHE_DIR
        store._CACHE_DIR = self.root

        # Sealed reference: a field plus a conserved quantity to compare against.
        import numpy as np
        rd = store._ref_dir("ref1")
        os.makedirs(rd, exist_ok=True)
        np.savez(os.path.join(rd, "output.npz"), field=np.arange(16, dtype=np.float64))
        store._write_json_atomic(os.path.join(rd, "metrics.json"),
                                 {"mass": 120.0, "time_s": 1.0})
        store._write_json_atomic(store.registry_path(), {"tiny": {
            "name": "tiny", "executable_path": "/bin/true",
            "run_cmd_template": "true", "output_format": "npz",
            "conserved_metric": "mass",
        }})

    def tearDown(self) -> None:
        store._CACHE_DIR = self._saved_root
        self._tmp.cleanup()

    def _run_dir(self, mass: float = 120.0) -> str:
        import numpy as np
        d = procs._new_run_dir(store._proxy_runs_dir("tiny"))
        np.savez(os.path.join(d, "output.npz"), field=np.arange(16, dtype=np.float64))
        with open(procs._log_path(d), "w") as fh:
            fh.write("PROXY_METRICS_BEGIN\n"
                     "time_s=0.5\n"
                     f"mass={mass}\n"
                     "PROXY_METRICS_END\n")
        return d

    def test_detached_run_carries_the_invariants(self) -> None:
        d = self._run_dir()
        execute._post_run_finalize(d, "tiny", "ref1", "npz", wall_s=0.6, returncode=0)
        m = store._read_json(os.path.join(d, "metrics.json"))

        self.assertEqual(m["comparison_to_reference"]["reference"], "ref1")
        # Error norms lifted to top level so requirements can gate on them.
        self.assertIn("l2_rel", m)
        self.assertEqual(m["l2_rel"], m["comparison_to_reference"]["l2_rel"])
        self.assertEqual(m["finite"], 1)
        self.assertEqual(m["conservation_residual"], 0.0)
        # And the shared settling still applies.
        self.assertEqual(m["wall_time_s"], 0.6)
        self.assertEqual(m["returncode"], 0)

    def test_conservation_drift_surfaces_as_a_metric(self) -> None:
        d = self._run_dir(mass=126.0)
        execute._post_run_finalize(d, "tiny", "ref1", "npz", wall_s=0.6, returncode=0)
        m = store._read_json(os.path.join(d, "metrics.json"))
        self.assertAlmostEqual(m["conservation_residual"], 0.05, places=6)

    def test_unknown_proxy_still_settles(self) -> None:
        d = self._run_dir()
        execute._post_run_finalize(d, "ghost", "ref1", "npz", wall_s=0.6, returncode=0)
        m = store._read_json(os.path.join(d, "metrics.json"))
        self.assertEqual(m["finite"], 1)          # entry-independent invariant
        self.assertNotIn("conservation_residual", m)   # needs conserved_metric


class ValidateSlurmArgsTests(unittest.TestCase):
    def test_valid_args_pass(self) -> None:
        self.assertIsNone(procs._validate_slurm_args("gpu", 1, 8, "32G", "04:00:00"))
        self.assertIsNone(procs._validate_slurm_args("cpu", 0, 1, "64000M", "1-12:00:00"))

    def test_invalid_args_rejected(self) -> None:
        self.assertIn("partition", procs._validate_slurm_args("", 0, 8, "32G", "04:00:00"))
        self.assertIn("gpus", procs._validate_slurm_args("p", -1, 8, "32G", "04:00:00"))
        self.assertIn("mem", procs._validate_slurm_args("p", 0, 8, "lots", "04:00:00"))
        self.assertIn("wall_time", procs._validate_slurm_args("p", 0, 8, "32G", "4 hours"))


class RunLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_new_run_dir_creates_start_time_and_suffix(self) -> None:
        base = os.path.join(self.root, "runs", "p")
        d = procs._new_run_dir(base, tag_suffix="scase_0")
        self.assertTrue(os.path.isdir(d))
        self.assertTrue(os.path.basename(d).endswith("_scase_0"))
        self.assertTrue(os.path.isfile(os.path.join(d, "start_time")))

    def test_write_run_config(self) -> None:
        d = procs._new_run_dir(self.root)
        procs._write_run_config(d, {"proxy_name": "p", "n": 3})
        with open(os.path.join(d, "config.json")) as fh:
            self.assertEqual(json.load(fh), {"proxy_name": "p", "n": 3})

    def test_launch_detached_records_pid_and_log(self) -> None:
        d = procs._new_run_dir(self.root)
        log = procs._log_path(d)
        pid = procs._launch_detached(
            [sys.executable, "-c", "print('hello from child')"], d, log_file=log,
        )
        self.assertEqual(procs._read_pid(d), pid)
        for _ in range(50):  # wait for the detached child to finish
            if not procs._is_running(pid):
                break
            import time as _t
            _t.sleep(0.1)
        with open(log) as fh:
            self.assertIn("hello from child", fh.read())

    def test_cancel_run_terminates_local_process(self) -> None:
        d = procs._new_run_dir(self.root)
        procs._launch_detached([sys.executable, "-c", "import time; time.sleep(60)"], d)
        result = procs._cancel_run(d)
        self.assertIn(result.get("cancelled"), ("sigterm", "sigkill"))

    def test_cancel_run_inactive_is_noop(self) -> None:
        d = procs._new_run_dir(self.root)  # no pid, no metrics -> state 'crashed'
        result = procs._cancel_run(d)
        self.assertEqual(result.get("note"), "Run is not active.")
        self.assertNotIn("cancelled", result)


class SubmitSbatchTests(unittest.TestCase):
    """Uses a fake `sbatch` on PATH so no real scheduler is ever touched."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.bin_dir = os.path.join(self.root, "bin")
        os.makedirs(self.bin_dir)
        self._saved_path = os.environ["PATH"]

    def tearDown(self) -> None:
        os.environ["PATH"] = self._saved_path
        self._tmp.cleanup()

    def _fake_sbatch(self, script: str) -> None:
        p = os.path.join(self.bin_dir, "sbatch")
        with open(p, "w") as fh:
            fh.write(script)
        os.chmod(p, 0o755)

    def test_success_parses_and_records_job_id(self) -> None:
        self._fake_sbatch("#!/bin/sh\necho 'Submitted batch job 4242'\n")
        os.environ["PATH"] = self.bin_dir
        d = procs._new_run_dir(self.root)
        job_id, error = procs._submit_sbatch(d, "#!/bin/bash\ntrue\n")
        self.assertIsNone(error)
        self.assertEqual(job_id, 4242)
        self.assertEqual(procs._read_slurm_id(d), 4242)
        self.assertTrue(os.path.isfile(os.path.join(d, "batch_script.sh")))

    def test_sbatch_failure_returns_err(self) -> None:
        self._fake_sbatch("#!/bin/sh\necho 'bad partition' >&2\nexit 1\n")
        os.environ["PATH"] = self.bin_dir
        d = procs._new_run_dir(self.root)
        job_id, error = procs._submit_sbatch(d, "#!/bin/bash\ntrue\n")
        self.assertIsNone(job_id)
        self.assertIn("sbatch failed", error["error"])

    def test_sbatch_missing_suggests_local_alternative(self) -> None:
        os.environ["PATH"] = self.bin_dir  # empty dir: no sbatch
        d = procs._new_run_dir(self.root)
        job_id, error = procs._submit_sbatch(
            d, "#!/bin/bash\ntrue\n",
            local_alternative="proxy_exec(op='run', confirm=True)",
        )
        self.assertIsNone(job_id)
        self.assertIn("sbatch not found", error["error"])
        self.assertIn("proxy_exec(op='run'", error["hint"])


try:
    import numpy as _np
    _HAVE_NUMPY = True
except ImportError:
    _HAVE_NUMPY = False


@unittest.skipUnless(_HAVE_NUMPY, "numpy required")
class FieldNormsTests(unittest.TestCase):
    def test_identical_fields_have_zero_error(self) -> None:
        a = _np.array([1.0, 2.0, 3.0])
        r = metrics._field_norms(a, a.copy())
        self.assertEqual(r["l2_abs"], 0.0)
        self.assertEqual(r["l2_rel"], 0.0)
        self.assertEqual(r["linf_rel"], 0.0)

    def test_known_relative_error(self) -> None:
        a = _np.array([0.0, 3.0, 4.0])   # ||a|| = 5
        b = _np.array([0.0, 3.0, 0.0])   # diff = [0,0,4], ||diff|| = 4
        r = metrics._field_norms(a, b)
        self.assertAlmostEqual(r["l2_abs"], 4.0)
        self.assertAlmostEqual(r["l2_rel"], 0.8)

    def test_shape_mismatch_flattens_and_flags(self) -> None:
        r = metrics._field_norms(_np.zeros((2, 2)), _np.zeros(3))
        self.assertIn("shape_error", r)
        self.assertEqual(r["l2_abs"], 0.0)

    def test_zero_norm_reference_yields_none_rel(self) -> None:
        r = metrics._field_norms(_np.zeros(3), _np.zeros(3))
        self.assertIsNone(r["l2_rel"])
        self.assertIsNone(r["linf_rel"])


@unittest.skipUnless(_HAVE_NUMPY, "numpy required")
class FiniteCheckTests(unittest.TestCase):
    def test_finite_field(self) -> None:
        self.assertTrue(metrics._finite_check(_np.array([1.0, 2.0, -3.5])))

    def test_nan_and_inf_are_not_finite(self) -> None:
        self.assertFalse(metrics._finite_check(_np.array([1.0, _np.nan])))
        self.assertFalse(metrics._finite_check(_np.array([1.0, _np.inf])))

    def test_empty_or_none_is_not_finite(self) -> None:
        self.assertFalse(metrics._finite_check(_np.array([])))
        self.assertFalse(metrics._finite_check(None))


class ConservationResidualTests(unittest.TestCase):
    def test_relative_residual(self) -> None:
        r = metrics._conservation_residual({"mass": 10.5}, {"mass": 10.0}, "mass")
        self.assertAlmostEqual(r, 0.05)

    def test_zero_reference_uses_absolute(self) -> None:
        r = metrics._conservation_residual({"mass": 0.2}, {"mass": 0.0}, "mass")
        self.assertAlmostEqual(r, 0.2)

    def test_missing_or_nonnumeric_returns_none(self) -> None:
        self.assertIsNone(metrics._conservation_residual({}, {"mass": 1.0}, "mass"))
        self.assertIsNone(metrics._conservation_residual({"mass": "x"}, {"mass": 1.0}, "mass"))


@unittest.skipUnless(_HAVE_NUMPY, "numpy required")
class ConvergenceOrderTests(unittest.TestCase):
    def test_second_order_recovered(self) -> None:
        # error = C * h^2  ->  fitted slope ~ 2.0
        pairs = [(h, 0.5 * h ** 2) for h in (1.0, 0.5, 0.25, 0.125)]
        self.assertAlmostEqual(metrics._convergence_order(pairs), 2.0, places=6)

    def test_needs_two_points(self) -> None:
        self.assertIsNone(metrics._convergence_order([(1.0, 0.1)]))

    def test_nonpositive_values_are_dropped(self) -> None:
        # only one usable pair remains -> None
        self.assertIsNone(metrics._convergence_order([(1.0, 0.0), (0.5, 0.25)]))


class IsImprovementTests(unittest.TestCase):
    def test_min_goal(self) -> None:
        self.assertTrue(ratchet._is_improvement(3.0, 5.0, "min"))
        self.assertFalse(ratchet._is_improvement(5.0, 5.0, "min"))
        self.assertFalse(ratchet._is_improvement(6.0, 5.0, "min"))

    def test_max_goal(self) -> None:
        self.assertTrue(ratchet._is_improvement(6.0, 5.0, "max"))
        self.assertFalse(ratchet._is_improvement(5.0, 5.0, "max"))

    def test_min_improvement_margin(self) -> None:
        # needs to beat 10 by >10% -> < 9.0
        self.assertFalse(ratchet._is_improvement(9.5, 10.0, "min", 0.1))
        self.assertTrue(ratchet._is_improvement(8.5, 10.0, "min", 0.1))

    def test_missing_incumbent_is_improvement(self) -> None:
        self.assertTrue(ratchet._is_improvement(1.0, None, "min"))
        self.assertFalse(ratchet._is_improvement(None, 1.0, "min"))


class RatchetVerdictTests(unittest.TestCase):
    def test_first_feasible_accepts(self) -> None:
        self.assertEqual(ratchet._ratchet_verdict(True, 5.0, None, "min"), "accept")

    def test_feasible_improvement_accepts(self) -> None:
        best = {"primary_value": 5.0}
        self.assertEqual(ratchet._ratchet_verdict(True, 4.0, best, "min"), "accept")

    def test_feasible_no_improvement_rejects(self) -> None:
        best = {"primary_value": 5.0}
        self.assertEqual(ratchet._ratchet_verdict(True, 5.0, best, "min"), "reject")

    def test_infeasible_with_incumbent_rejects(self) -> None:
        best = {"primary_value": 5.0}
        self.assertEqual(ratchet._ratchet_verdict(False, 4.0, best, "min"), "reject")

    def test_infeasible_without_incumbent_is_none(self) -> None:
        self.assertIsNone(ratchet._ratchet_verdict(False, 4.0, None, "min"))


class RatchetStoreTests(unittest.TestCase):
    """best.json / ledger.jsonl persistence."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._saved_root = store._CACHE_DIR
        store._CACHE_DIR = self._tmp.name

    def tearDown(self) -> None:
        store._CACHE_DIR = self._saved_root
        self._tmp.cleanup()

    def test_save_and_load_best_with_snapshot(self) -> None:
        src = os.path.join(self._tmp.name, "proxy.py")
        with open(src, "w") as fh:
            fh.write("# v1\n")
        snap = ratchet._save_best("p", "run1", 3.0, src)
        self.assertTrue(os.path.isfile(snap))
        best = ratchet._load_best("p")
        self.assertEqual(best["run_id"], "run1")
        self.assertEqual(best["primary_value"], 3.0)
        self.assertEqual(best["source_snapshot"], snap)

    def test_ledger_appends_lines(self) -> None:
        ratchet._append_ledger("p", {"run_id": "r1", "verdict": "accept"})
        ratchet._append_ledger("p", {"run_id": "r2", "verdict": "reject"})
        with open(store._opt_ledger_file("p")) as fh:
            lines = [json.loads(x) for x in fh if x.strip()]
        self.assertEqual([e["verdict"] for e in lines], ["accept", "reject"])


class ReservedMetricsTests(unittest.TestCase):
    """The proxy under optimization cannot forge server-side invariants.

    Regression: an agent-edited proxy printed conservation_residual in its
    metrics block to satisfy a requirement whose sealed reference was missing,
    and the forged value flowed straight into requirement evaluation.
    """

    def test_strip_reserved_metrics(self) -> None:
        mets = {"time_s": 0.1, "finite": 1, "conservation_residual": 0.0,
                "l2_rel": 0.0, "convergence_order": 2.0, "custom": 7}
        dropped = metrics._strip_reserved_metrics(mets)
        self.assertEqual(dropped, ["conservation_residual", "convergence_order",
                                   "finite", "l2_rel"])
        self.assertEqual(mets, {"time_s": 0.1, "custom": 7})

    def test_run_benchmark_case_ignores_forged_invariants(self) -> None:
        import time as _time
        with tempfile.TemporaryDirectory() as tmp:
            old_root, store._CACHE_DIR = store._CACHE_DIR, tmp
            try:
                exe = os.path.join(tmp, "forger.py")
                with open(exe, "w") as fh:
                    fh.write(textwrap.dedent(f"""\
                        import numpy as np
                        out = r"{tmp}/out.npz"
                        np.savez(out, field=np.ones(4))
                        print("PROXY_METRICS_BEGIN")
                        print("time_s=0.01")
                        print("conservation_residual=0.0")
                        print("l2_rel=0.0")
                        print(f"output_file={{out}}")
                        print("PROXY_METRICS_END")
                    """))
                entry = {"name": "forger", "executable_path": exe,
                         "run_cmd_template": f"{sys.executable} {{executable}}",
                         "output_format": "npz"}
                run_dir, row = execute._run_benchmark_case(
                    entry=entry, proxy_name="forger", reference_name="",
                    extra_params="", param_overrides=None, extra_metrics=[],
                    deadline=_time.monotonic() + 60,
                )
                self.assertNotIn("error", row)
                with open(os.path.join(run_dir, "metrics.json")) as fh:
                    metrics = json.load(fh)
                # Forged values are gone; the real server-side finite remains.
                self.assertNotIn("conservation_residual", metrics)
                self.assertNotIn("l2_rel", metrics)
                self.assertEqual(metrics["reserved_metrics_ignored"],
                                 ["conservation_residual", "l2_rel"])
                self.assertEqual(metrics["finite"], 1)
            finally:
                store._CACHE_DIR = old_root


class NormalizeTimeMetricsTests(unittest.TestCase):
    """time_s is self-reported: impossible claims are discarded, wall_time_s is
    always the server's own measurement (the ratchet's default objective must
    not be forgeable from inside the proxy)."""

    def test_missing_time_s_falls_back_to_wall(self) -> None:
        m: dict = {}
        metrics._normalize_time_metrics(m, 4.0)
        self.assertEqual(m["wall_time_s"], 4.0)
        self.assertEqual(m["time_s"], 4.0)
        self.assertNotIn("time_s_ignored", m)

    def test_honest_kernel_time_is_kept(self) -> None:
        m = {"time_s": 3.2}
        metrics._normalize_time_metrics(m, 4.0)
        self.assertEqual(m["time_s"], 3.2)
        self.assertEqual(m["wall_time_s"], 4.0)
        self.assertNotIn("time_s_ignored", m)

    def test_time_exceeding_wall_is_discarded(self) -> None:
        m = {"time_s": 9.9}
        metrics._normalize_time_metrics(m, 4.0)
        self.assertEqual(m["time_s"], 4.0)
        self.assertEqual(m["time_s_ignored"], 9.9)

    def test_non_positive_time_is_discarded(self) -> None:
        m = {"time_s": 0.0}
        metrics._normalize_time_metrics(m, 4.0)
        self.assertEqual(m["time_s"], 4.0)
        self.assertEqual(m["time_s_ignored"], 0.0)

    def test_non_numeric_time_replaced_without_audit_key(self) -> None:
        m = {"time_s": "fast"}
        metrics._normalize_time_metrics(m, 4.0)
        self.assertEqual(m["time_s"], 4.0)
        self.assertNotIn("time_s_ignored", m)


class RunBenchmarkCaseIntegrityTests(unittest.TestCase):
    """returncode gate + time_s plausibility guard on the synchronous case runner."""

    def _run_case(self, tmp: str, body: str) -> tuple[str, dict]:
        import time as _time
        exe = os.path.join(tmp, "p.py")
        with open(exe, "w") as fh:
            fh.write(body)
        entry = {"name": "p", "executable_path": exe,
                 "run_cmd_template": f"{sys.executable} {{executable}}",
                 "output_format": "npz"}
        return execute._run_benchmark_case(
            entry=entry, proxy_name="p", reference_name="",
            extra_params="", param_overrides=None, extra_metrics=[],
            deadline=_time.monotonic() + 60,
        )

    def test_forged_time_s_is_replaced_by_wall_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_root, store._CACHE_DIR = store._CACHE_DIR, tmp
            try:
                run_dir, row = self._run_case(tmp, textwrap.dedent("""\
                    print("PROXY_METRICS_BEGIN")
                    print("time_s=99999")
                    print("wall_time_s=0.001")
                    print("PROXY_METRICS_END")
                """))
                self.assertNotIn("error", row)
                with open(os.path.join(run_dir, "metrics.json")) as fh:
                    metrics = json.load(fh)
                # The printed wall_time_s is reserved -> stripped; the stored
                # one is the server's own measurement.
                self.assertIn("wall_time_s", metrics["reserved_metrics_ignored"])
                self.assertLess(metrics["wall_time_s"], 60)
                # The impossible time_s claim is discarded but kept for audit.
                self.assertEqual(metrics["time_s"], metrics["wall_time_s"])
                self.assertEqual(metrics["time_s_ignored"], 99999)
                self.assertEqual(metrics["returncode"], 0)
            finally:
                store._CACHE_DIR = old_root

    def test_nonzero_exit_fails_case_even_with_metrics_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_root, store._CACHE_DIR = store._CACHE_DIR, tmp
            try:
                run_dir, row = self._run_case(tmp, textwrap.dedent("""\
                    import sys
                    print("PROXY_METRICS_BEGIN")
                    print("time_s=0.01")
                    print("PROXY_METRICS_END")
                    sys.exit(3)
                """))
                self.assertEqual(row["error"], "solver exited with code 3")
                self.assertEqual(row["returncode"], 3)
                # No metrics.json -> the run dir reads as 'crashed'.
                self.assertFalse(os.path.isfile(os.path.join(run_dir, "metrics.json")))
                self.assertEqual(procs._run_state(run_dir)["state"], "crashed")
            finally:
                store._CACHE_DIR = old_root


class RegistryLockThreadingTests(unittest.TestCase):
    """The re-entrancy depth is thread-local: same-thread nesting must not
    deadlock, while two threads must still exclude each other via flock
    (a shared depth counter would let the second thread skip the lock)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._saved_root = store._CACHE_DIR
        store._CACHE_DIR = self._tmp.name

    def tearDown(self) -> None:
        store._CACHE_DIR = self._saved_root
        self._tmp.cleanup()

    def test_reentrant_in_one_thread(self) -> None:
        with store._registry_lock():
            with store._registry_lock():
                pass  # nesting must not deadlock

    def test_threads_exclude_each_other(self) -> None:
        import threading
        import time as _time

        held = threading.Event()
        release = threading.Event()
        acquired: list[float] = []

        def holder() -> None:
            with store._registry_lock():
                held.set()
                release.wait(5)

        def contender() -> None:
            with store._registry_lock():
                acquired.append(_time.monotonic())

        t1 = threading.Thread(target=holder)
        t1.start()
        self.assertTrue(held.wait(5))
        t2 = threading.Thread(target=contender)
        t2.start()
        _time.sleep(0.3)
        self.assertEqual(acquired, [], "contender acquired the lock while held")
        release.set()
        t2.join(5)
        t1.join(5)
        self.assertEqual(len(acquired), 1)


class StorageContractTests(unittest.TestCase):
    """Frozen external contract, checked in a fresh interpreter: the storage
    root honors ``MIMIR_PROXY_BENCH_DIR``, which is read at import time.
    """

    _PATH_SETUP = (f"import sys; sys.path[:0] = ["
                   f"{str(SERVERS_DIR / 'proxy')!r}, {str(SERVERS_DIR / '_shared')!r}]; ")

    def _run(self, code: str, env_extra: dict | None = None) -> str:
        import subprocess
        env = dict(os.environ, **(env_extra or {}))
        res = subprocess.run([sys.executable, "-c", self._PATH_SETUP + code],
                             capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        return res.stdout.strip()

    def test_cache_root_env_override(self) -> None:
        out = self._run("from _lib import store; print(store.cache_dir())",
                        env_extra={"MIMIR_PROXY_BENCH_DIR": "/tmp/proxy_bench_env_test"})
        self.assertEqual(out, "/tmp/proxy_bench_env_test")


if __name__ == "__main__":
    unittest.main()
