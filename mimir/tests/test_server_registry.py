"""Finding a server that outlived the window which started it.

The port used to be learned by regexing the child's stdout, which works exactly as long
as the extension host is the parent holding that pipe. A server meant to survive that
window has to leave its address on disk — and a reader has to be able to tell a live
entry from the leftovers of one that crashed.

Liveness here is three questions, and the tests are mostly about the third: a pid can be
alive with its listener already gone.
"""
import json
import os
import socket
import tempfile
import unittest
from unittest import mock

from mimir.client.ui.ws import job_scan, server_registry


class _RegistryCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        for target in (server_registry, job_scan):
            patcher = mock.patch.object(target, "_MIMIR_DIR_WS", self._tmp.name)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _listening(self) -> socket.socket:
        """A real listening socket, so the probe has something true to find."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        self.addCleanup(sock.close)
        return sock

    def _free_port(self) -> int:
        """A port nothing is listening on."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    @staticmethod
    def _dead_pid() -> int:
        for pid in range(4194300, 4194200, -1):
            try:
                os.kill(pid, 0)
            except OSError:
                return pid
        raise unittest.SkipTest("no free pid to use as a dead one")


class PublishAndReadTests(_RegistryCase):
    def test_a_published_entry_reads_back(self):
        entry = server_registry.publish(url="ws://127.0.0.1:1234",
                                        host="127.0.0.1", port=1234, model="m")
        self.assertIsNotNone(entry)
        back = server_registry.read()
        self.assertEqual(back["url"], "ws://127.0.0.1:1234")
        self.assertEqual(back["port"], 1234)
        self.assertEqual(back["pid"], os.getpid())
        self.assertEqual(back["model"], "m")
        self.assertEqual(back["protocol"], server_registry.PROTOCOL)

    def test_the_entry_records_this_process_start_time_too(self):
        # Without it a recycled pid reads as the same server.
        server_registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1)
        self.assertEqual(server_registry.read()["pid_starttime"],
                         job_scan._proc_starttime(os.getpid()))

    def test_nothing_published_reads_as_nothing(self):
        self.assertIsNone(server_registry.read())

    def test_an_unreadable_file_reads_as_nothing_rather_than_raising(self):
        with open(server_registry.registry_path(), "w") as fh:
            fh.write("{not json")
        self.assertIsNone(server_registry.read())

    def test_an_entry_from_another_protocol_is_refused(self):
        # Connecting to a server whose contract has changed is worse than deciding
        # there is none.
        with open(server_registry.registry_path(), "w") as fh:
            json.dump({"protocol": server_registry.PROTOCOL + 1,
                       "pid": os.getpid(), "url": "ws://x", "port": 1}, fh)
        self.assertIsNone(server_registry.read())

    def test_publishing_twice_replaces_the_entry(self):
        server_registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1)
        server_registry.publish(url="ws://127.0.0.1:2", host="127.0.0.1", port=2)
        self.assertEqual(server_registry.read()["port"], 2)

    def test_a_write_that_fails_leaves_no_half_file(self):
        with mock.patch("json.dump", side_effect=OSError("disk full")):
            self.assertIsNone(
                server_registry.publish(url="ws://x", host="h", port=1))
        self.assertFalse(os.path.exists(server_registry.registry_path() + ".tmp"))
        self.assertIsNone(server_registry.read())


class LivenessTests(_RegistryCase):
    def test_a_live_pid_with_a_listening_port_is_alive(self):
        sock = self._listening()
        host, port = sock.getsockname()
        server_registry.publish(url=f"ws://{host}:{port}", host=host, port=port)
        self.assertTrue(server_registry.alive(server_registry.read()))

    def test_a_live_pid_whose_listener_is_gone_is_not_alive(self):
        # The question the process table cannot answer: this very process is running,
        # and nothing is accepting on that port.
        port = self._free_port()
        server_registry.publish(url=f"ws://127.0.0.1:{port}",
                                host="127.0.0.1", port=port)
        entry = server_registry.read()
        self.assertFalse(server_registry.alive(entry))
        # …and the weaker, process-only answer still says yes, which is why the probe
        # is not optional in the ordinary path.
        self.assertTrue(server_registry.alive(entry, probe=False))

    def test_a_dead_pid_is_not_alive(self):
        server_registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1)
        entry = server_registry.read()
        entry["pid"] = self._dead_pid()
        self.assertFalse(server_registry.alive(entry))

    def test_a_recycled_pid_is_not_mistaken_for_the_server(self):
        sock = self._listening()
        host, port = sock.getsockname()
        server_registry.publish(url=f"ws://{host}:{port}", host=host, port=port)
        entry = server_registry.read()
        entry["pid_starttime"] = 1          # a different process wears that number now
        self.assertFalse(server_registry.alive(entry))

    def test_no_entry_is_not_alive(self):
        self.assertFalse(server_registry.alive(None))
        self.assertFalse(server_registry.alive({}))

    def test_a_malformed_pid_is_not_alive(self):
        self.assertFalse(server_registry.alive({"pid": "not a number", "port": 1}))

    def test_current_hands_back_only_a_live_entry(self):
        sock = self._listening()
        host, port = sock.getsockname()
        server_registry.publish(url=f"ws://{host}:{port}", host=host, port=port)
        self.assertIsNotNone(server_registry.current())
        sock.close()
        self.assertIsNone(server_registry.current())

    def test_probing_port_zero_is_never_alive(self):
        self.assertFalse(server_registry.port_answers("127.0.0.1", 0))

    def test_an_unresolvable_host_is_not_alive(self):
        self.assertFalse(
            server_registry.port_answers("no-such-host.invalid", 1234))


class ClearTests(_RegistryCase):
    def test_our_own_entry_is_removed(self):
        server_registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1)
        server_registry.clear()
        self.assertIsNone(server_registry.read())
        self.assertFalse(os.path.exists(server_registry.registry_path()))

    def test_a_strangers_entry_is_left_alone(self):
        # Deleting another server's address makes a perfectly good server
        # undiscoverable.
        server_registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1)
        entry = server_registry.read()
        entry["pid"] = self._dead_pid()
        with open(server_registry.registry_path(), "w") as fh:
            json.dump(entry, fh)
        server_registry.clear()
        self.assertIsNotNone(server_registry.read())

    def test_a_strangers_entry_can_be_removed_deliberately(self):
        server_registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1)
        entry = server_registry.read()
        entry["pid"] = self._dead_pid()
        with open(server_registry.registry_path(), "w") as fh:
            json.dump(entry, fh)
        server_registry.clear(only_if_ours=False)
        self.assertIsNone(server_registry.read())

    def test_clearing_nothing_is_not_an_error(self):
        server_registry.clear()


class WorkspaceScopeTests(unittest.TestCase):
    """One file per workspace is where "one server per workspace" comes from."""

    def test_the_path_sits_in_the_per_workspace_state_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(server_registry, "_MIMIR_DIR_WS", tmp):
                self.assertEqual(server_registry.registry_path(),
                                 os.path.join(tmp, "server.json"))

    def test_two_workspaces_do_not_share_an_entry(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            with mock.patch.object(server_registry, "_MIMIR_DIR_WS", a):
                server_registry.publish(url="ws://127.0.0.1:1",
                                        host="127.0.0.1", port=1)
            with mock.patch.object(server_registry, "_MIMIR_DIR_WS", b):
                self.assertIsNone(server_registry.read())


if __name__ == "__main__":
    unittest.main()
