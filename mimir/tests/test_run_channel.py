"""The two ends of the divert channel agree on one pair of files.

The client writes the request and a server in another process consumes it, so
nothing typechecks the format between them: only a test that runs both halves against
one state dir can. The properties that matter are that a request reaches its own run
and nothing else, that one tool's channel is invisible to another's, and that every
failure mode is silent — a missing, stale or corrupt sidecar must leave a blocking run
behaving exactly as it did before this existed.

See servers/_shared/run_channel.py and client/tool_execution/run_channel.py.
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
for _p in [SERVERS_DIR / "_shared"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_channel as server_side  # noqa: E402

_KEY = "20260101T120000Z-ab12"
# Every case runs against both real channels: the properties below are about the
# mechanism, and a channel that only ever held one tool would not prove it.
_CHANNELS = ("bash_run", "proxy_eval")


class _ChannelFixture(unittest.TestCase):
    """One temp state dir, with both halves resolving to it."""

    def setUp(self) -> None:
        self._state = tempfile.mkdtemp(prefix="mimir-divert-")
        self._orig = os.environ.get("MIMIR_STATE_DIR")
        os.environ["MIMIR_STATE_DIR"] = self._state
        # The client half reads STATE_DIR at import time, so it is reloaded under the
        # temp dir rather than monkeypatched — the same value the server resolves.
        from mimir.client.config import constants
        importlib.reload(constants)
        from mimir.client.tool_execution import run_channel
        self.client_side = importlib.reload(run_channel)

    def tearDown(self) -> None:
        if self._orig is None:
            os.environ.pop("MIMIR_STATE_DIR", None)
        else:
            os.environ["MIMIR_STATE_DIR"] = self._orig
        from mimir.client.config import constants
        importlib.reload(constants)
        from mimir.client.tool_execution import run_channel
        importlib.reload(run_channel)
        shutil.rmtree(self._state, ignore_errors=True)

class DivertSeamTests(_ChannelFixture):
    """A request reaches its own run, and every failure mode is silent."""

    def test_both_ends_resolve_the_same_directory(self) -> None:
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                self.assertEqual(self.client_side._dir(ch), server_side._dir(ch))

    def test_a_request_reaches_the_run_the_server_announced(self) -> None:
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                server_side.publish(ch, _KEY, 4242, "make -j8", "/w", 60.0)
                run = self.client_side.request_divert(ch)
                self.assertIsNotNone(run)
                self.assertEqual(run["job_key"], _KEY)
                self.assertEqual(run["pid"], 4242)
                self.assertTrue(server_side.requested(ch, _KEY))

    def test_a_request_is_consumed_once(self) -> None:
        # A second read must not detach whatever runs next.
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                server_side.publish(ch, _KEY, 1, "x", "/w", 60.0)
                self.client_side.request_divert(ch)
                self.assertTrue(server_side.requested(ch, _KEY))
                self.assertFalse(server_side.requested(ch, _KEY))

    def test_a_request_naming_another_run_is_left_alone(self) -> None:
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                server_side.publish(ch, _KEY, 1, "x", "/w", 60.0)
                self.client_side.request_divert(ch)
                self.assertFalse(server_side.requested(ch, "20260101T120000Z-ffff"))

    def test_nothing_to_divert_says_so_rather_than_pretending(self) -> None:
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                self.assertIsNone(self.client_side.current_run(ch))
                self.assertIsNone(self.client_side.request_divert(ch))
                self.assertFalse(server_side.requested(ch, _KEY))

    def test_publishing_clears_a_request_left_by_an_earlier_run(self) -> None:
        # The race this exists for: the click lands just after its target finished.
        # Left in place, the request would detach the next command the agent ran.
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                server_side.publish(ch, _KEY, 1, "x", "/w", 60.0)
                self.client_side.request_divert(ch)
                server_side.clear(ch, _KEY)
                other = "20260101T120001Z-cd34"
                server_side.publish(ch, other, 2, "y", "/w", 60.0)
                self.assertFalse(server_side.requested(ch, other))

    def test_a_stale_request_is_not_honoured(self) -> None:
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                server_side.publish(ch, _KEY, 1, "x", "/w", 60.0)
                self.client_side.request_divert(ch)
                path = os.path.join(server_side._dir(ch), "divert")
                old = time.time() - server_side._STALE_S - 30
                os.utime(path, (old, old))
                self.assertFalse(server_side.requested(ch, _KEY))
                self.assertFalse(os.path.exists(path))

    def test_clear_leaves_another_run_s_announcement_standing(self) -> None:
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                server_side.publish(ch, _KEY, 1, "x", "/w", 60.0)
                server_side.clear(ch, "20260101T120000Z-ffff")
                self.assertIsNotNone(self.client_side.current_run(ch))

    def test_every_read_is_fail_open_on_a_corrupt_sidecar(self) -> None:
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                os.makedirs(server_side._dir(ch), exist_ok=True)
                with open(os.path.join(server_side._dir(ch), "current.json"), "w") as fh:
                    fh.write("{not json")
                self.assertIsNone(self.client_side.current_run(ch))
                self.assertIsNone(self.client_side.request_divert(ch))
                self.assertFalse(server_side.requested(ch, _KEY))
                server_side.clear(ch, _KEY)  # must not raise


class ChannelIsolationTests(_ChannelFixture):
    """One tool's channel must be invisible to another's.

    The whole point of keying by tool: before this, a click on a proxy row wrote into
    the shell's channel and detached whatever the shell happened to be running.
    """

    def test_a_request_on_one_channel_does_not_reach_another(self) -> None:
        a, b = _CHANNELS
        server_side.publish(a, _KEY, 1, "x", "/w", 60.0)
        server_side.publish(b, _KEY, 2, "y", "/w", 60.0)
        self.client_side.request_divert(a)
        self.assertFalse(server_side.requested(b, _KEY))
        self.assertTrue(server_side.requested(a, _KEY))

    def test_publishing_on_one_channel_does_not_clear_another_s_request(self) -> None:
        a, b = _CHANNELS
        server_side.publish(a, _KEY, 1, "x", "/w", 60.0)
        self.client_side.request_divert(a)
        server_side.publish(b, "other", 2, "y", "/w", 60.0)
        self.assertTrue(server_side.requested(a, _KEY))

    def test_one_channel_s_announcement_is_not_the_other_s(self) -> None:
        a, b = _CHANNELS
        server_side.publish(a, _KEY, 1, "x", "/w", 60.0)
        self.assertIsNotNone(self.client_side.current_run(a))
        self.assertIsNone(self.client_side.current_run(b))


class PhaseUpdateTests(_ChannelFixture):
    """``update`` refreshes what the run is doing, and nothing else."""

    def test_a_phase_reaches_the_client(self) -> None:
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                server_side.publish(ch, _KEY, 1, "x", "/w", 60.0)
                server_side.update(ch, _KEY, phase="building (1/2)", percent=34.0)
                run = self.client_side.current_run(ch)
                self.assertEqual(run["phase"], "building (1/2)")
                self.assertEqual(run["percent"], 34.0)

    def test_a_phase_for_another_run_is_ignored(self) -> None:
        # A late tick from a wait loop whose run already ended must not relabel the
        # run that took its place.
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                server_side.publish(ch, _KEY, 1, "x", "/w", 60.0)
                server_side.update(ch, "someone-else", phase="wrong")
                self.assertEqual(self.client_side.current_run(ch)["phase"], "")

    def test_updating_does_not_consume_a_pending_divert(self) -> None:
        # The click lands between two ticks of the wait loop. If `update` cleared the
        # request the way `publish` does, the divert would be silently dropped.
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                server_side.publish(ch, _KEY, 1, "x", "/w", 60.0)
                self.client_side.request_divert(ch)
                server_side.update(ch, _KEY, phase="still going", percent=50.0)
                self.assertTrue(server_side.requested(ch, _KEY))

    def test_updating_an_absent_channel_is_silent(self) -> None:
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                server_side.update(ch, _KEY, phase="nobody is listening")
                self.assertIsNone(self.client_side.current_run(ch))

    def test_percent_is_dropped_when_a_tick_no_longer_has_one(self) -> None:
        # Build ends, measurement begins: the bar must go away rather than freeze at
        # whatever the compiler last said.
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                server_side.publish(ch, _KEY, 1, "x", "/w", 60.0)
                server_side.update(ch, _KEY, phase="building", percent=80.0)
                server_side.update(ch, _KEY, phase="case a (1/2)")
                run = self.client_side.current_run(ch)
                self.assertEqual(run["phase"], "case a (1/2)")
                self.assertIsNone(run["percent"])


class DeadServerTests(_ChannelFixture):
    """An announcement outlives the server that made it; the client must notice."""

    def test_an_announcement_from_a_dead_server_reads_as_absent(self) -> None:
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                server_side.publish(ch, _KEY, 1, "x", "/w", 60.0)
                path = os.path.join(server_side._dir(ch), "current.json")
                import json
                run = json.load(open(path))
                # A pid that cannot be running: above the kernel's own ceiling.
                run["server_pid"] = 4_294_967_294
                with open(path, "w") as fh:
                    json.dump(run, fh)
                self.assertIsNone(self.client_side.current_run(ch))
                self.assertIsNone(self.client_side.request_divert(ch))

    def test_an_announcement_with_no_server_pid_is_accepted(self) -> None:
        # Fail-open: the field is the improvement, not the precondition.
        for ch in _CHANNELS:
            with self.subTest(channel=ch):
                os.makedirs(server_side._dir(ch), exist_ok=True)
                import json
                with open(os.path.join(server_side._dir(ch), "current.json"), "w") as fh:
                    json.dump({"job_key": _KEY, "pid": 1}, fh)
                self.assertIsNotNone(self.client_side.current_run(ch))


if __name__ == "__main__":
    unittest.main()
