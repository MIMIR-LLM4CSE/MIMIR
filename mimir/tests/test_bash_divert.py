"""The two ends of the divert channel agree on one pair of files.

The client writes the request and a bash server in another process consumes it, so
nothing typechecks the format between them: only a test that runs both halves against
one state dir can. The properties that matter are that a request reaches its own run
and nothing else, and that every failure mode is silent — a missing, stale or corrupt
sidecar must leave a blocking run behaving exactly as it did before this existed.

See servers/workspace/_bash_divert.py and client/tool_execution/bash_divert.py.
"""
import importlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in [SERVERS_DIR / "_shared", SERVERS_DIR / "workspace"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _bash_divert as server_side  # noqa: E402

_KEY = "20260101T120000Z-ab12"


class DivertSeamTests(unittest.TestCase):
    def setUp(self) -> None:
        self._state = tempfile.mkdtemp(prefix="mimir-divert-")
        self._orig = os.environ.get("MIMIR_STATE_DIR")
        os.environ["MIMIR_STATE_DIR"] = self._state
        # The client half reads STATE_DIR at import time, so it is reloaded under the
        # temp dir rather than monkeypatched — the same value the server resolves.
        from mimir.client.config import constants
        importlib.reload(constants)
        from mimir.client.tool_execution import bash_divert
        self.client_side = importlib.reload(bash_divert)

    def tearDown(self) -> None:
        if self._orig is None:
            os.environ.pop("MIMIR_STATE_DIR", None)
        else:
            os.environ["MIMIR_STATE_DIR"] = self._orig
        from mimir.client.config import constants
        importlib.reload(constants)
        from mimir.client.tool_execution import bash_divert
        importlib.reload(bash_divert)
        shutil.rmtree(self._state, ignore_errors=True)

    def test_both_ends_resolve_the_same_directory(self) -> None:
        self.assertEqual(self.client_side._dir(), server_side._dir())

    def test_a_request_reaches_the_run_the_server_announced(self) -> None:
        server_side.publish(_KEY, 4242, "make -j8", "/w", 60.0)
        run = self.client_side.request_divert()
        self.assertIsNotNone(run)
        self.assertEqual(run["job_key"], _KEY)
        self.assertEqual(run["pid"], 4242)
        self.assertTrue(server_side.requested(_KEY))

    def test_a_request_is_consumed_once(self) -> None:
        # A second read must not detach whatever runs next.
        server_side.publish(_KEY, 1, "x", "/w", 60.0)
        self.client_side.request_divert()
        self.assertTrue(server_side.requested(_KEY))
        self.assertFalse(server_side.requested(_KEY))

    def test_a_request_naming_another_run_is_left_alone(self) -> None:
        server_side.publish(_KEY, 1, "x", "/w", 60.0)
        self.client_side.request_divert()
        self.assertFalse(server_side.requested("20260101T120000Z-ffff"))

    def test_nothing_to_divert_says_so_rather_than_pretending(self) -> None:
        self.assertIsNone(self.client_side.current_run())
        self.assertIsNone(self.client_side.request_divert())
        self.assertFalse(server_side.requested(_KEY))

    def test_publishing_clears_a_request_left_by_an_earlier_run(self) -> None:
        # The race this exists for: the click lands just after its target finished.
        # Left in place, the request would detach the next command the agent ran.
        server_side.publish(_KEY, 1, "x", "/w", 60.0)
        self.client_side.request_divert()
        server_side.clear(_KEY)
        other = "20260101T120001Z-cd34"
        server_side.publish(other, 2, "y", "/w", 60.0)
        self.assertFalse(server_side.requested(other))

    def test_a_stale_request_is_not_honoured(self) -> None:
        server_side.publish(_KEY, 1, "x", "/w", 60.0)
        self.client_side.request_divert()
        path = os.path.join(server_side._dir(), "divert")
        old = time.time() - server_side._STALE_S - 30
        os.utime(path, (old, old))
        self.assertFalse(server_side.requested(_KEY))
        self.assertFalse(os.path.exists(path))

    def test_clear_leaves_another_run_s_announcement_standing(self) -> None:
        server_side.publish(_KEY, 1, "x", "/w", 60.0)
        server_side.clear("20260101T120000Z-ffff")
        self.assertIsNotNone(self.client_side.current_run())

    def test_every_read_is_fail_open_on_a_corrupt_sidecar(self) -> None:
        os.makedirs(server_side._dir(), exist_ok=True)
        with open(os.path.join(server_side._dir(), "current.json"), "w") as fh:
            fh.write("{not json")
        self.assertIsNone(self.client_side.current_run())
        self.assertIsNone(self.client_side.request_divert())
        self.assertFalse(server_side.requested(_KEY))
        server_side.clear(_KEY)  # must not raise


if __name__ == "__main__":
    unittest.main()
