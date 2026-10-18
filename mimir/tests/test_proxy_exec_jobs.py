"""proxy_exec's long ops (reference/suite/benchmark_create) run as detached jobs.

They used to run inside the tool call, so a baseline longer than the client's tool wall
could not be created at all. Now the call waits for the job, and detaches it to the
background watcher on request or when the wait budget runs out.

Run:
    python -m unittest mimir.tests.test_proxy_exec_jobs -v
"""

import asyncio
import os
import time

from mimir.tests.support.proxy import _TmpStorageTest, _call, server_proxy, store
from mimir.servers.proxy._ops import exec_jobs, references, runs  # noqa: E402


class ExecJobTests(_TmpStorageTest):
    def test_reference_waits_and_answers_inline(self) -> None:
        self._register()
        res = _call(server_proxy.proxy_exec, op="reference", proxy_name="tiny",
                    reference_name="r1", confirm=True)
        self.assertEqual(res.get("status"), "ok", res)
        self.assertEqual(res["reference_name"], "r1")
        self.assertTrue(exec_jobs.is_job_dir(res["job_dir"]))
        self.assertTrue(os.path.isdir(store.reference_dir("r1")))

    def test_background_detaches_with_a_watchable_handle(self) -> None:
        self._register()
        res = _call(server_proxy.proxy_exec, op="benchmark_create", proxy_name="tiny",
                    benchmark_name="b1", background=True, confirm=True)
        self.assertEqual(res.get("status"), "ok", res)
        job = res["background_job"]
        self.assertEqual(job["status_op"]["tool"], "proxy_runs")
        run_id = job["status_op"]["args"]["run_id"]

        deadline = time.monotonic() + 60
        while True:
            st = runs.run_status(run_id)
            if st["state"] not in ("running", "pending") or time.monotonic() > deadline:
                break
            time.sleep(0.2)
        self.assertEqual(st["state"], "done", st)
        self.assertEqual(st["result"]["benchmark_name"], "b1")

    def test_budget_exhausted_detaches_instead_of_failing(self) -> None:
        self._register()
        res = asyncio.run(exec_jobs.run_awaited(
            "reference", {"proxy_name": "tiny", "reference_name": "r2"}, wait_s=0))
        self.assertEqual(res.get("status"), "ok", res)
        self.assertIn("background_job", res)
        self.assertIn("Still running", res["note"])
        deadline = time.monotonic() + 60
        while (exec_jobs.job_state(res["job_dir"])["state"] == "running"
               and time.monotonic() < deadline):
            time.sleep(0.2)

    def test_failed_op_reads_as_crashed(self) -> None:
        res = _call(server_proxy.proxy_exec, op="reference", proxy_name="missing",
                    reference_name="r3", confirm=True)
        self.assertEqual(res.get("status"), "error", res)
        self.assertEqual(exec_jobs.job_state(res["job_dir"])["state"], "crashed")


class SealFieldTests(_TmpStorageTest):
    """The field is sealed from the {output_file} path, echoed back or not."""

    def _register_writer(self, echo: bool) -> None:
        exe = os.path.join(self.root, "writer.py")
        with open(exe, "w") as fh:
            fh.write("import sys\n"
                     "open(sys.argv[1], 'wb').write(b'\\0' * 64)\n"
                     "print('PROXY_METRICS_BEGIN')\n"
                     "print('time_s=0.01')\n"
                     + ("print(f'output_file={sys.argv[1]}')\n" if echo else "")
                     + "print('PROXY_METRICS_END')\n")
        res = server_proxy.proxy_manage(
            op="register", name="w", executable_path=exe,
            run_cmd_template="python3 {executable} {output_file}",
            output_format="raw_float64", confirm=True)
        self.assertEqual(res.get("status"), "ok", res)

    def test_field_sealed_without_echo(self) -> None:
        self._register_writer(echo=False)
        res = references.create("w", "ref_noecho")
        self.assertEqual(res.get("status"), "ok", res)
        self.assertTrue(res["output_stored"])
        self.assertTrue(os.path.isfile(res["output_stored"]))

    def test_field_sealed_with_echo(self) -> None:
        self._register_writer(echo=True)
        res = references.create("w", "ref_echo")
        self.assertTrue(res["output_stored"])

    def test_missing_field_is_reported_not_silent(self) -> None:
        self._register()
        res = references.create("tiny", "ref_none")
        self.assertIsNone(res["output_stored"])
        self.assertIn("No field output found", res["metrics"]["output_file_warning"])


if __name__ == "__main__":
    import unittest
    unittest.main()
