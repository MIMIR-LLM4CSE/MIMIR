"""The proxy store is created by registering a proxy, never by reading.

``proxy_get`` is a read tool and stays visible in plan mode (the mutating proxy
tools are PLAN_BLOCKED), so it must leave nothing behind on a project that has never
registered anything. Taking the registry flock on every read has the flock create its own
directory, leaving ``<workspace>/proxy_bench/`` and a ``registry.json.lock`` in the user's
tree — a read tool writing, during a phase that must not write at all. Reads skip the lock
when there is no store to race over, and ``proxy_manage(op='register')`` is the one op that
brings the store into existence.

Run:
    python -m unittest mimir.tests.test_proxy_store_creation -v
"""

import os
import shutil
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
from _lib import store  # noqa: E402


class StoreCreationTests(unittest.TestCase):
    """Storage root points at a path that does NOT exist yet — the real first-run shape.

    ``_TmpStorageTest`` deliberately points the store at an existing temp dir, so it
    cannot see this; here the workspace exists and ``proxy_bench/`` under it does not.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = self._tmp.name
        self.cache = os.path.join(self.workspace, "proxy_bench")
        self._saved_root = store._CACHE_DIR
        store._CACHE_DIR = self.cache
        self._saved_ws = os.environ.get("MCP_FILES_ROOT")
        os.environ["MCP_FILES_ROOT"] = self.workspace

    def tearDown(self) -> None:
        store._CACHE_DIR = self._saved_root
        if self._saved_ws is None:
            os.environ.pop("MCP_FILES_ROOT", None)
        else:
            os.environ["MCP_FILES_ROOT"] = self._saved_ws
        self._tmp.cleanup()

    def _assert_untouched(self) -> None:
        self.assertFalse(
            os.path.exists(self.cache),
            f"a read created {self.cache}: {os.listdir(self.workspace)}")

    # -- reads create nothing ------------------------------------------------

    def test_reads_create_nothing(self) -> None:
        for label, call, expect in (
            ("get/proxies",    lambda: server_proxy.proxy_get(op="proxies"),    "ok"),
            ("get/suites",     lambda: server_proxy.proxy_get(op="suites"),     "ok"),
            ("get/references", lambda: server_proxy.proxy_get(op="references"), "ok"),
            ("get/proxy",      lambda: server_proxy.proxy_get(op="proxy", name="absent"),
             "error"),
            ("runs/list",      lambda: server_proxy.proxy_runs(op="list", proxy_name="absent"),
             None),
        ):
            with self.subTest(op=label):
                res = call()
                if expect:
                    self.assertEqual(res.get("status"), expect)
                self._assert_untouched()

    def test_failed_mutations_create_nothing(self) -> None:
        """update/unregister on an absent proxy: an error, and still no store."""
        for res in (
            server_proxy.proxy_manage(op="unregister", name="absent", confirm=True),
            server_proxy.proxy_manage(op="update", name="absent", description="x", confirm=True),
        ):
            self.assertEqual(res.get("status"), "error")
        self._assert_untouched()

    # -- registering is what creates it --------------------------------------

    def test_register_creates_the_store(self) -> None:
        exe = os.path.join(self.workspace, "tiny.py")
        with open(exe, "w") as fh:
            fh.write("print('hi')\n")
        res = server_proxy.proxy_manage(
            op="register", name="tiny", executable_path=exe,
            run_cmd_template="python3 {executable}", confirm=True)
        self.assertEqual(res.get("status"), "ok")
        self.assertTrue(os.path.isfile(store.registry_path()))
        # And the read that used to create it now finds the real entry.
        listed = server_proxy.proxy_get(op="proxies")
        self.assertEqual(listed.get("status"), "ok")
        self.assertIn("tiny", str(listed))



class StoreLocationTests(unittest.TestCase):
    """Where the root lands. Read at import, so each case reloads the module.

    A store outside the workspace outlives the project: deleting a project left the
    registry and an "in progress" optimisation behind, and a fresh start silently
    resumed it.
    """

    def test_the_default_is_under_the_workspace(self) -> None:
        import importlib
        wt = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, wt, True)
        os.environ["MCP_FILES_ROOT"] = wt
        os.environ.pop("MIMIR_PROXY_BENCH_DIR", None)
        self.addCleanup(os.environ.pop, "MCP_FILES_ROOT", None)
        from _lib import store
        importlib.reload(store)
        self.assertEqual(store.cache_dir(), os.path.join(wt, "proxy_bench"))

    def test_the_env_override_still_wins(self) -> None:
        import importlib
        os.environ["MIMIR_PROXY_BENCH_DIR"] = "/tmp/explicit_store"
        self.addCleanup(os.environ.pop, "MIMIR_PROXY_BENCH_DIR", None)
        from _lib import store
        importlib.reload(store)
        self.assertEqual(store.cache_dir(), "/tmp/explicit_store")


if __name__ == "__main__":
    unittest.main()
