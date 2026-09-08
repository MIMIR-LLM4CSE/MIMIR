"""The proxy store is created by registering a proxy, never by reading.

``proxy_get`` is a read tool and stays visible in plan mode (the mutating proxy
tools are PLAN_BLOCKED). It used to leave ``<workspace>/proxy_bench/`` and a
``registry.json.lock`` behind on a project that had never registered anything,
because every registry read took the registry flock and the flock created its own
directory — a read tool writing into the user's tree, during a phase that must not
write at all. Reads now skip the lock when there is no store to race over, and
``proxy_manage(op='register')`` is the one op that brings the store into existence.

Run:
    python -m unittest mimir.tests.test_proxy_store_creation -v
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

    def test_get_proxies_creates_nothing(self) -> None:
        res = server_proxy.proxy_get(op="proxies")
        self.assertEqual(res.get("status"), "ok")
        self._assert_untouched()

    def test_get_suites_creates_nothing(self) -> None:
        self.assertEqual(server_proxy.proxy_get(op="suites").get("status"), "ok")
        self._assert_untouched()

    def test_get_references_creates_nothing(self) -> None:
        self.assertEqual(server_proxy.proxy_get(op="references").get("status"), "ok")
        self._assert_untouched()

    def test_inspect_missing_proxy_creates_nothing(self) -> None:
        self.assertEqual(server_proxy.proxy_get(op="proxy", name="absent").get("status"), "error")
        self._assert_untouched()

    def test_runs_list_creates_nothing(self) -> None:
        server_proxy.proxy_runs(op="list", proxy_name="absent")
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


if __name__ == "__main__":
    unittest.main()
