"""Finding a server that outlived the window which started it, through its registry entry.

An entry is live only if its pid is alive and the server at its port names that pid.
"""
import http.server
import json
import os
import socket
import tempfile
import threading
import unittest
from unittest import mock

from mimir.shared.processes import proc_starttime
from mimir.client.ui.ws.process import registry as server_registry
from mimir.tests.support.state import pin_state_dir


class _RegistryCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        pin_state_dir(self, self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _answering(self, body: str | None = None, status: int = 200):
        """A real server answering the probe with *body* (this process's pid by default).

        Returns its address and a stop that frees the port.
        """
        answer = (str(os.getpid()) if body is None else body).encode()

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                found = self.path == server_registry.ALIVE_PATH
                self.send_response(status if found else 404)
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

            def log_message(self, *_args) -> None:
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def stop() -> None:
            server.shutdown()
            server.server_close()

        self.addCleanup(stop)
        return server.server_address[0], server.server_address[1], stop

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

    def test_only_this_user_can_read_the_token(self):
        server_registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1, token="t")
        self.assertEqual(os.stat(server_registry.registry_path()).st_mode & 0o777, 0o600)

    def test_an_update_keeps_the_token(self):
        server_registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1, token="t")
        server_registry.claim(["s"], "auto")
        self.assertEqual(server_registry.read()["token"], "t")

    def test_the_entry_records_this_process_start_time_too(self):
        # Without it a recycled pid reads as the same server.
        server_registry.publish(url="ws://127.0.0.1:1", host="127.0.0.1", port=1)
        self.assertEqual(server_registry.read()["pid_starttime"],
                         proc_starttime(os.getpid()))

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
        with mock.patch("os.replace", side_effect=OSError("disk full")):
            self.assertIsNone(
                server_registry.publish(url="ws://x", host="h", port=1))
        self.assertFalse(os.path.exists(server_registry.registry_path() + ".tmp"))
        self.assertIsNone(server_registry.read())


class LivenessTests(_RegistryCase):
    def test_a_live_pid_whose_server_names_it_is_alive(self):
        host, port, _stop = self._answering()
        server_registry.publish(url=f"ws://{host}:{port}", host=host, port=port)
        self.assertTrue(server_registry.alive(server_registry.read()))

    def test_a_stranger_on_the_port_is_not_the_server(self):
        # The pid is alive, and the port is held by another process that answers.
        host, port, _stop = self._answering(body="1")
        server_registry.publish(url=f"ws://{host}:{port}", host=host, port=port)
        self.assertFalse(server_registry.alive(server_registry.read()))

    def test_a_listener_that_does_not_answer_the_probe_is_not_the_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        self.addCleanup(sock.close)
        host, port = sock.getsockname()
        self.assertFalse(server_registry.server_answers(host, port, os.getpid(),
                                                        timeout=0.3))

    def test_a_server_from_before_the_probe_is_alive(self):
        # It refuses a plain request with 426, as every WebSocket server does.
        host, port, _stop = self._answering(body="", status=426)
        self.assertTrue(server_registry.server_answers(host, port, os.getpid()))

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
        host, port, _stop = self._answering()
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
        host, port, stop = self._answering()
        server_registry.publish(url=f"ws://{host}:{port}", host=host, port=port)
        self.assertIsNotNone(server_registry.current())
        stop()
        self.assertIsNone(server_registry.current())

    def test_probing_port_zero_is_never_alive(self):
        self.assertFalse(server_registry.server_answers("127.0.0.1", 0, os.getpid()))

    def test_an_unresolvable_host_is_not_alive(self):
        self.assertFalse(
            server_registry.server_answers("no-such-host.invalid", 1234, os.getpid()))


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


class OneServerPerWorkspaceTests(_RegistryCase):
    """One server per workspace, claimed with ``flock``: two would interleave one transcript
    and its ``seq``, and the kernel releases the lock however the process dies.
    """

    def tearDown(self) -> None:
        server_registry.release()

    def test_the_first_claim_succeeds(self):
        self.assertTrue(server_registry.acquire())

    def test_claiming_twice_in_one_process_is_idempotent(self):
        self.assertTrue(server_registry.acquire())
        self.assertTrue(server_registry.acquire())

    def test_a_state_dir_it_cannot_write_raises_rather_than_reads_as_held(self):
        # A refusal means another server holds the lock; a directory that cannot be made
        # is a different fault, and the server says so instead of waiting on nobody.
        blocker = os.path.join(self._tmp.name, "not-a-dir")
        open(blocker, "w").close()
        with mock.patch("mimir.client.config.paths.STATE_DIR", os.path.join(blocker, "state")):
            with self.assertRaises(OSError):
                server_registry.acquire()

    def test_a_second_process_is_refused_while_the_first_holds_it(self):
        # Real subprocesses: flock is a kernel object, and a mock of it would prove
        # nothing about the thing that has to hold.
        import subprocess
        import sys
        import textwrap

        holder = textwrap.dedent(f"""
            import os, sys, time
            sys.path.insert(0, {json.dumps(os.getcwd())})
            os.environ["MIMIR_STATE_DIR"] = {json.dumps(self._tmp.name)}
            from unittest import mock
            from mimir.client.ui.ws.process import registry as r
            mock.patch("mimir.client.config.paths.STATE_DIR", {json.dumps(self._tmp.name)}).start()
            print("HELD" if r.acquire() else "REFUSED", flush=True)
            time.sleep(10)
        """)
        contender = textwrap.dedent(f"""
            import os, sys
            sys.path.insert(0, {json.dumps(os.getcwd())})
            os.environ["MIMIR_STATE_DIR"] = {json.dumps(self._tmp.name)}
            from unittest import mock
            from mimir.client.ui.ws.process import registry as r
            mock.patch("mimir.client.config.paths.STATE_DIR", {json.dumps(self._tmp.name)}).start()
            print("HELD" if r.acquire() else "REFUSED", flush=True)
        """)

        first = subprocess.Popen([sys.executable, "-c", holder],
                                 stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(first.stdout.readline().strip(), "HELD")
            second = subprocess.run([sys.executable, "-c", contender],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(second.stdout.strip(), "REFUSED",
                             "a second server claimed the same workspace")
        finally:
            first.kill()
            first.wait(timeout=10)

    def test_the_lock_is_free_once_the_holder_dies(self):
        # However it dies: a crashed server must not keep a workspace locked, which is
        # the whole reason this is the kernel's lock and not a file we write.
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(f"""
            import os, sys, time
            sys.path.insert(0, {json.dumps(os.getcwd())})
            from unittest import mock
            from mimir.client.ui.ws.process import registry as r
            mock.patch("mimir.client.config.paths.STATE_DIR", {json.dumps(self._tmp.name)}).start()
            print("HELD" if r.acquire() else "REFUSED", flush=True)
            time.sleep(30)
        """)
        holder = subprocess.Popen([sys.executable, "-c", script],
                                  stdout=subprocess.PIPE, text=True)
        self.assertEqual(holder.stdout.readline().strip(), "HELD")
        holder.kill()                     # not a graceful stop
        holder.wait(timeout=10)
        self.assertTrue(server_registry.acquire(),
                        "the lock outlived the process that held it")

    def test_releasing_lets_the_next_one_in(self):
        self.assertTrue(server_registry.acquire())
        server_registry.release()
        self.assertTrue(server_registry.acquire())

    def test_releasing_nothing_is_not_an_error(self):
        server_registry.release()
        server_registry.release()

    def test_two_workspaces_do_not_contend(self):
        import tempfile as _tf
        self.assertTrue(server_registry.acquire())
        with _tf.TemporaryDirectory() as other:
            with mock.patch("mimir.client.config.paths.STATE_DIR", other):
                # A different workspace is a different lock file; the global handle is
                # this process's own, so it is released first to make the test honest.
                server_registry.release()
                self.assertTrue(server_registry.acquire())
                server_registry.release()


class WorkspaceScopeTests(unittest.TestCase):
    """One file per workspace is where "one server per workspace" comes from."""

    def test_the_path_sits_in_the_per_workspace_state_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("mimir.client.config.paths.STATE_DIR", tmp):
                self.assertEqual(server_registry.registry_path(),
                                 os.path.join(tmp, "server.json"))

    def test_two_workspaces_do_not_share_an_entry(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            with mock.patch("mimir.client.config.paths.STATE_DIR", a):
                server_registry.publish(url="ws://127.0.0.1:1",
                                        host="127.0.0.1", port=1)
            with mock.patch("mimir.client.config.paths.STATE_DIR", b):
                self.assertIsNone(server_registry.read())


if __name__ == "__main__":
    unittest.main()
