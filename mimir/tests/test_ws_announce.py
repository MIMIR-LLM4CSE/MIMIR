"""The "Listening on" line must name an address that answers, and the probe an answer.

With ``--port 0`` on ``localhost`` the server binds 127.0.0.1 and ::1 on two
different ports, in an order that changes between launches. The line used to say
``localhost`` with the first socket's port; when that socket was the IPv6 one, the
extension dialled 127.0.0.1 on the IPv6 port and was refused on every retry.
"""
import asyncio
import os
import unittest
from unittest import mock

import websockets

from mimir.client.ui.ws.process import registry as server_registry
from mimir.client.ui.ws.server import serving
from mimir.client.ui.ws.server.serving import _announced_url, _gate, _is_loopback, _server_token


class _Sock:
    def __init__(self, name):
        self._name = name

    def getsockname(self):
        return self._name


class AnnouncedUrlTests(unittest.TestCase):
    def test_ipv4_socket_is_named_by_its_address(self):
        self.assertEqual(_announced_url([_Sock(("127.0.0.1", 4242))]), "ws://127.0.0.1:4242")

    def test_ipv6_socket_is_bracketed(self):
        self.assertEqual(_announced_url([_Sock(("::1", 4242, 0, 0))]), "ws://[::1]:4242")

    def test_ipv4_socket_is_named_whatever_the_order(self):
        # A proxy rarely exempts ::1, so the IPv6 address is only a fallback.
        sockets = [_Sock(("::1", 4243, 0, 0)), _Sock(("127.0.0.1", 4242))]
        self.assertEqual(_announced_url(sockets), "ws://127.0.0.1:4242")


class AnnouncedUrlAnswersTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_announced_address_accepts_a_connection(self):
        async def _hello(ws):
            await ws.send("hi")

        # Several launches: the socket order is what varies.
        for _ in range(6):
            async with websockets.serve(_hello, "localhost", 0) as server:
                # proxy=None: the proxy variables of the machine running the tests
                # must not decide the outcome. The IPv4 preference is tested above.
                url = _announced_url(server.sockets)
                async with websockets.connect(url, proxy=None) as ws:
                    self.assertEqual(await asyncio.wait_for(ws.recv(), 5), "hi")


class LivenessProbeTests(unittest.IsolatedAsyncioTestCase):
    """The server answers the registry's probe with its pid, and logs nothing for it."""

    async def _probe(self, **serve_kwargs) -> tuple[bool, list[str]]:
        async def _hello(ws):
            await ws.send("hi")

        with self.assertLogs("websockets.server", level="INFO") as logs:
            async with websockets.serve(_hello, "127.0.0.1", 0, **serve_kwargs) as server:
                port = server.sockets[0].getsockname()[1]
                answered = await asyncio.to_thread(
                    server_registry.server_answers, "127.0.0.1", port, os.getpid())
        return answered, [r.getMessage() for r in logs.records if r.levelname == "ERROR"]

    async def test_the_probe_is_answered_without_an_error(self):
        self.assertEqual(await self._probe(process_request=_gate("t")), (True, []))

    async def test_a_server_without_the_answer_still_passes(self):
        # A server started before the probe existed refuses the plain request (426).
        answered, _errors = await self._probe()
        self.assertTrue(answered)



class TokenGateTests(unittest.IsolatedAsyncioTestCase):
    """A handshake is taken on the server's token, and on nothing else."""

    @staticmethod
    async def _status(**connect_kwargs) -> int:
        """The handshake's status: 101 when it is taken, else what refused it."""
        async def _hello(ws):
            await ws.send("hi")

        async with websockets.serve(_hello, "127.0.0.1", 0,
                                    process_request=_gate("right")) as server:
            try:
                async with websockets.connect(_announced_url(server.sockets), proxy=None,
                                              **connect_kwargs) as ws:
                    assert await asyncio.wait_for(ws.recv(), 5) == "hi"
                    return 101
            except websockets.InvalidStatus as refused:
                return refused.response.status_code

    async def test_the_token_opens_the_door(self):
        self.assertEqual(
            await self._status(additional_headers={"Authorization": "Bearer right"}), 101)

    async def test_no_token_is_refused(self):
        self.assertEqual(await self._status(), 401)

    async def test_a_wrong_token_is_refused(self):
        self.assertEqual(
            await self._status(additional_headers={"Authorization": "Bearer wrong"}), 401)

    async def test_a_page_of_any_origin_is_refused(self):
        # A browser sends an Origin and cannot add an Authorization header.
        self.assertEqual(await self._status(origin="https://evil.example"), 401)

    async def test_a_second_authorization_header_is_refused(self):
        headers = [("Authorization", "Bearer right"), ("Authorization", "Bearer wrong")]
        self.assertEqual(await self._status(additional_headers=headers), 401)


class ServerTokenTests(unittest.TestCase):
    def test_each_server_generates_its_own(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(serving.TOKEN_ENV, None)
            first, second = _server_token(), _server_token()
        self.assertGreaterEqual(len(first), 32)
        self.assertNotEqual(first, second)

    def test_the_variable_sets_it_and_nothing_started_later_inherits_it(self):
        with mock.patch.dict(os.environ, {serving.TOKEN_ENV: " chosen "}):
            self.assertEqual(_server_token(), "chosen")
            self.assertNotIn(serving.TOKEN_ENV, os.environ)

    def test_only_a_loopback_address_stays_on_the_machine(self):
        self.assertTrue(_is_loopback("127.0.0.1"))
        self.assertTrue(_is_loopback("::1"))
        self.assertFalse(_is_loopback("0.0.0.0"))
        self.assertFalse(_is_loopback("10.1.2.3"))


class EntryPointTests(unittest.TestCase):
    def test_the_entry_point_resolves_no_workspace_before_main(self) -> None:
        """``--cwd`` is applied in ``main``; anything that resolved the workspace root
        while the entry module was imported would have resolved the launch directory.

        It did: the package's own ``__init__`` imported the agent, so ``--cwd`` changed
        the directory after ``config.paths`` had already fixed the workspace root.
        """
        import subprocess
        import sys
        code = ("import sys; import mimir.client.ui.ws.server.main; "
                "print(any(m.startswith('mimir.client.config') for m in sys.modules))")
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        self.assertEqual(out, "False")

if __name__ == "__main__":
    unittest.main()
