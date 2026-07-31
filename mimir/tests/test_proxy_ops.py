"""Unit tests for the op-dispatched proxy server (servers/proxy/server_proxy.py).

Covers the dispatch layer (unknown ops, per-op required args, confirm gating),
the `metadata` dict merge in register/update, the `next_step` contract on the
eval loop, and the Slurm arg validation.  All storage is redirected into a
temp dir; no scheduler or real proxy is ever touched.

Run:
    python -m unittest mimir.tests.test_proxy_ops -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in (SERVERS_DIR / "_shared", SERVERS_DIR / "proxy"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import server_proxy  # noqa: E402
from _lib import procs, ratchet, store  # noqa: E402  (procs/ratchet re-exported for test_proxy_opt_loop)
from _ops import eval_session, references, registry, runs, scaffold, slurm, suites  # noqa: E402,F401  (re-exported for sibling test modules)


class _TmpStorageTest(unittest.TestCase):
    """Base: redirects the proxy storage root into a per-test temp dir.

    Every path is derived from ``store._CACHE_DIR`` at call time, so one
    repoint makes the entire store hermetic.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._saved_root = store._CACHE_DIR
        store._CACHE_DIR = self._tmp.name
        self.root = self._tmp.name

    def tearDown(self) -> None:
        store._CACHE_DIR = self._saved_root
        self._tmp.cleanup()

    # -- fixtures ------------------------------------------------------------

    def _make_exe(self, name: str = "tiny.py") -> str:
        path = os.path.join(self.root, name)
        with open(path, "w") as fh:
            fh.write("print('PROXY_METRICS_BEGIN')\nprint('time_s=0.1')\nprint('PROXY_METRICS_END')\n")
        return path

    def _register(self, name: str = "tiny", metadata: dict | None = None) -> dict:
        return server_proxy.proxy_manage(
            op="register", name=name, executable_path=self._make_exe(),
            run_cmd_template="python3 {executable}", metadata=metadata, confirm=True,
        )


class DispatchTests(_TmpStorageTest):
    def test_unknown_op_lists_valid_ops(self) -> None:
        for tool, kwargs in (
            (server_proxy.proxy_get, {}),
            (server_proxy.proxy_runs, {}),
            (server_proxy.proxy_eval_status, {}),
            (server_proxy.proxy_manage, {"confirm": True}),
            (server_proxy.proxy_exec, {"confirm": True}),
            (server_proxy.proxy_eval, {"confirm": True}),
            (server_proxy.proxy_slurm, {"partition": "p", "confirm": True}),
        ):
            with self.subTest(tool=tool.__name__):
                res = tool(op="bogus", **kwargs)
                self.assertEqual(res.get("status"), "error")
                self.assertIn("Unknown op 'bogus'", res["error"])
                self.assertIn("Use one of:", res["hint"])

    def test_missing_required_args_named(self) -> None:
        res = server_proxy.proxy_get(op="proxy")
        self.assertEqual(res.get("status"), "error")
        self.assertIn("op 'proxy' requires: name", res["error"])

        res = server_proxy.proxy_runs(op="logs")
        self.assertIn("op 'logs' requires: run_id", res["error"])

        res = server_proxy.proxy_exec(op="run", confirm=True)
        self.assertIn("op 'run' requires: proxy_name", res["error"])

        res = server_proxy.proxy_exec(op="reference", proxy_name="p", confirm=True)
        self.assertIn("op 'reference' requires: reference_name", res["error"])

    def test_confirm_gate_on_every_sensitive_tool(self) -> None:
        for tool, kwargs in (
            (server_proxy.proxy_manage, {"op": "unregister", "name": "x"}),
            (server_proxy.proxy_exec, {"op": "run", "proxy_name": "x"}),
            (server_proxy.proxy_eval, {"op": "run"}),
            (server_proxy.proxy_slurm, {"op": "run", "partition": "p", "proxy_name": "x"}),
        ):
            with self.subTest(tool=tool.__name__):
                res = tool(**kwargs)
                self.assertEqual(res.get("status"), "error")
                self.assertIn("Not confirmed", res["error"])
                self.assertIn("confirm=True", res["hint"])

    def test_slurm_resource_validation_before_submission(self) -> None:
        res = server_proxy.proxy_slurm(op="run", partition="p", proxy_name="x",
                                       mem="lots", confirm=True)
        self.assertEqual(res.get("status"), "error")
        self.assertIn("Invalid mem format", res["error"])

        res = server_proxy.proxy_slurm(op="run", partition="", proxy_name="x", confirm=True)
        self.assertIn("partition is required", res["error"])


class RegisterMetadataTests(_TmpStorageTest):
    def test_register_merges_metadata_flat(self) -> None:
        res = self._register(metadata={"arch": "x86", "tags": ["cfd"], "notes": "n1"})
        self.assertEqual(res.get("status"), "ok")
        entry = res["registered"]
        self.assertEqual(entry["arch"], "x86")
        self.assertEqual(entry["tags"], ["cfd"])
        self.assertEqual(entry["notes"], "n1")
        # Unspecified metadata keys keep their flat on-disk defaults.
        self.assertEqual(entry["backend"], "")
        self.assertEqual(entry["usage_examples"], [])

    def test_register_rejects_unknown_metadata_key(self) -> None:
        res = self._register(metadata={"archh": "typo"})
        self.assertEqual(res.get("status"), "error")
        self.assertIn("Unknown metadata key", res["error"])
        self.assertIn("arch", res["error"])  # valid keys listed

    def test_update_patches_only_provided_metadata(self) -> None:
        self._register(metadata={"arch": "x86", "notes": "n1"})
        res = server_proxy.proxy_manage(op="update", name="tiny",
                                        metadata={"notes": "n2"}, confirm=True)
        self.assertEqual(res.get("status"), "ok")
        self.assertEqual(res["updated"]["notes"], "n2")
        self.assertEqual(res["updated"]["arch"], "x86")

    def test_inspect_returns_readme_and_recent_runs(self) -> None:
        self._register()
        res = server_proxy.proxy_get(op="proxy", name="tiny")
        self.assertEqual(res.get("status"), "ok")
        self.assertIn("# Proxy: tiny", res["readme"])
        self.assertEqual(res["recent_runs"], [])
        self.assertEqual(server_proxy.proxy_get(op="proxies")["count"], 1)

    def test_unregister_keeps_history_note(self) -> None:
        self._register()
        res = server_proxy.proxy_manage(op="unregister", name="tiny", confirm=True)
        self.assertEqual(res.get("status"), "ok")
        self.assertEqual(server_proxy.proxy_get(op="proxies")["count"], 0)


class SuiteTests(_TmpStorageTest):
    def test_suite_define_validates_cases(self) -> None:
        self._register()
        res = server_proxy.proxy_manage(
            op="suite_define", name="s1",
            cases=[{"case_id": "a", "proxy_name": "tiny"}], confirm=True,
        )
        self.assertEqual(res.get("status"), "ok")
        self.assertIn("proxy_exec(op='suite'", res["next_step"])

        res = server_proxy.proxy_manage(
            op="suite_define", name="s2",
            cases=[{"case_id": "a", "proxy_name": "ghost"}], confirm=True,
        )
        self.assertEqual(res.get("status"), "error")
        self.assertIn("ghost", res["error"])

    def test_suite_inspect_flags_missing_reference(self) -> None:
        self._register()
        server_proxy.proxy_manage(
            op="suite_define", name="s1",
            cases=[{"case_id": "a", "proxy_name": "tiny", "reference_name": "missing"}],
            confirm=True,
        )
        res = server_proxy.proxy_get(op="suite", name="s1")
        self.assertEqual(res.get("status"), "ok")
        self.assertFalse(res["validation"][0]["ok"])


class EvalLoopTests(_TmpStorageTest):
    def _init_session(self) -> dict:
        self._register()
        server_proxy.proxy_manage(
            op="suite_define", name="bench",
            cases=[{"case_id": "a", "proxy_name": "tiny"}], confirm=True,
        )
        self.source = os.path.join(self.root, "source.py")
        with open(self.source, "w") as fh:
            fh.write("VERSION = 1\n")
        return server_proxy.proxy_eval(
            op="init", proxy_name="tiny", benchmark_name="bench",
            requirements=[{"metric": "time_s", "operator": "lt", "threshold": 2.0}],
            proxy_source_path=self.source, confirm=True,
        )

    def test_init_snapshots_canonical_and_sets_next_step(self) -> None:
        res = self._init_session()
        self.assertEqual(res.get("status"), "ok")
        self.assertFalse(res["canonical_existed"])
        self.assertIn("proxy_eval(op='run'", res["next_step"])

        # Second init never overwrites the canonical.
        res2 = server_proxy.proxy_eval(
            op="init", proxy_name="tiny", benchmark_name="bench",
            requirements=[{"metric": "time_s", "operator": "lt", "threshold": 2.0}],
            proxy_source_path=self.source, confirm=True,
        )
        self.assertTrue(res2["canonical_existed"])

    def test_init_requires_existing_suite(self) -> None:
        self._register()
        src = self._make_exe("src.py")
        res = server_proxy.proxy_eval(
            op="init", proxy_name="tiny", benchmark_name="nope",
            requirements=[{"metric": "t", "operator": "lt", "threshold": 1}],
            proxy_source_path=src, confirm=True,
        )
        self.assertEqual(res.get("status"), "error")
        self.assertIn("not found", res["error"])

    def test_init_rejects_bad_requirements(self) -> None:
        self._register()
        server_proxy.proxy_manage(
            op="suite_define", name="bench",
            cases=[{"case_id": "a", "proxy_name": "tiny"}], confirm=True,
        )
        res = server_proxy.proxy_eval(
            op="init", proxy_name="tiny", benchmark_name="bench",
            requirements=[{"metric": "t", "operator": "approx", "threshold": 1}],
            proxy_source_path=self._make_exe("s.py"), confirm=True,
        )
        self.assertEqual(res.get("status"), "error")
        self.assertIn("invalid operator", res["error"])

    def test_init_rejects_negative_min_improvement(self) -> None:
        self._register()
        server_proxy.proxy_manage(
            op="suite_define", name="bench",
            cases=[{"case_id": "a", "proxy_name": "tiny"}], confirm=True,
        )
        res = server_proxy.proxy_eval(
            op="init", proxy_name="tiny", benchmark_name="bench",
            requirements=[{"metric": "time_s", "operator": "lt", "threshold": 2.0}],
            proxy_source_path=self._make_exe("s.py"), min_improvement=-0.1, confirm=True,
        )
        self.assertEqual(res.get("status"), "error")
        self.assertIn("min_improvement", res["error"])

    def test_status_next_step_guides_the_loop(self) -> None:
        # No session at all -> init.
        res = server_proxy.proxy_eval_status()
        self.assertEqual(res["state"], "no_session")
        self.assertIn("init", res["next_step"])

        self._init_session()
        # Session but no runs -> run.
        res = server_proxy.proxy_eval_status()
        self.assertEqual(res["state"], "no_runs")
        self.assertIn("proxy_eval(op='run'", res["next_step"])

        res = server_proxy.proxy_eval_status(op="config")
        self.assertEqual(res["config"]["benchmark_name"], "bench")

    def test_configure_patches_without_resnapshot(self) -> None:
        self._init_session()
        res = server_proxy.proxy_eval(
            op="configure",
            requirements=[{"metric": "misfit", "operator": "lt", "threshold": 0.1}],
            confirm=True,
        )
        self.assertEqual(res.get("status"), "ok")
        self.assertEqual(res["config"]["requirements"][0]["metric"], "misfit")
        self.assertIn("next_step", res)

    def test_reset_restores_canonical_source(self) -> None:
        self._init_session()
        with open(self.source, "w") as fh:
            fh.write("VERSION = 2  # broken attempt\n")
        res = server_proxy.proxy_eval(op="reset", confirm=True)
        self.assertEqual(res.get("status"), "ok")
        with open(self.source) as fh:
            self.assertEqual(fh.read(), "VERSION = 1\n")


class ReadOnlyOpsTests(_TmpStorageTest):
    def test_empty_listings(self) -> None:
        self.assertEqual(server_proxy.proxy_get(op="references")["count"], 0)
        self.assertEqual(server_proxy.proxy_get(op="suites")["count"], 0)
        self.assertEqual(server_proxy.proxy_runs()["count"], 0)

    def test_aggregate_is_readonly_no_confirm_needed(self) -> None:
        res = server_proxy.proxy_runs(op="aggregate", run_ids=["ghost/run"])
        self.assertEqual(res.get("status"), "ok")
        self.assertEqual(res["rows"], [])
        self.assertEqual(res["errors"][0]["error"], "directory not found")


class ScaffoldTests(_TmpStorageTest):
    def test_scaffold_generates_harnesses_with_op_syntax(self) -> None:
        src = os.path.join(self.root, "proxy_src.py")
        with open(src, "w") as fh:
            fh.write("def heat_kernel(n):\n    return n\n")
        res = server_proxy.proxy_manage(
            op="scaffold", proxy_path=src, component_hint="heat_kernel", confirm=True,
        )
        self.assertEqual(res.get("status"), "ok")
        self.assertEqual(res["component"], "heat_kernel")
        self.assertTrue(os.path.isfile(res["ref_harness_path"]))
        self.assertTrue(os.path.isfile(res["test_harness_path"]))
        # Suggested follow-ups use the live op syntax, not dead tool names.
        self.assertIn("proxy_manage(", res["suggested_ref_register"])
        self.assertIn("op='register'", res["suggested_ref_register"])


if __name__ == "__main__":
    unittest.main()
