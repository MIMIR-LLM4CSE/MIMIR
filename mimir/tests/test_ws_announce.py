"""The "Listening on" line must name an address that answers.

With ``--port 0`` on ``localhost`` the server binds 127.0.0.1 and ::1 on two
different ports, in an order that changes between launches. The line used to say
``localhost`` with the first socket's port; when that socket was the IPv6 one, the
extension dialled 127.0.0.1 on the IPv6 port and was refused on every retry.
"""
import asyncio
import socket
import unittest

import websockets

from mimir.client.ui.ws.ws_server import _announced_url


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


class AnnouncedUrlAnswersTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_announced_address_accepts_a_connection(self):
        async def _hello(ws):
            await ws.send("hi")

        # Several launches: the socket order is what varies.
        for _ in range(6):
            async with websockets.serve(_hello, "localhost", 0) as server:
                async with websockets.connect(_announced_url(server.sockets)) as ws:
                    self.assertEqual(await asyncio.wait_for(ws.recv(), 5), "hi")


if __name__ == "__main__":
    unittest.main()
