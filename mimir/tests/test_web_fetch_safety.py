"""What ``http_get`` may reach: the address it connects to is the one it checked.

``_safe_url`` resolves a name to refuse internal targets; a connection that resolved it
again could be answered differently (DNS rebinding). No test here touches the network:
resolution is stubbed, and every refusal happens before a connect.
"""
from __future__ import annotations

import ipaddress
import os
import socket
import unittest
import urllib.request
from unittest import mock

from mimir.servers.external import server_web as web


def _answers(*addresses: str):
    """A getaddrinfo answering each call with the next address in turn."""
    queue = list(addresses)

    def getaddrinfo(host, port, *args, **kwargs):
        ip = queue.pop(0) if len(queue) > 1 else queue[0]
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port or 0))]
    return getaddrinfo


class WebFetchSafetyTests(unittest.TestCase):
    def test_carrier_grade_nat_and_benchmark_ranges_are_blocked(self) -> None:
        for ip in ("100.100.100.200", "198.18.0.1", "169.254.169.254", "10.0.0.1"):
            self.assertTrue(web._is_blocked_ip(ipaddress.ip_address(ip)), ip)
        self.assertFalse(web._is_blocked_ip(ipaddress.ip_address("93.184.216.34")))

    def test_a_name_that_rebinds_to_loopback_is_refused_at_connect(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(web.socket, "getaddrinfo",
                                  _answers("93.184.216.34", "127.0.0.1")), \
                mock.patch.object(web.socket.socket, "connect",
                                  side_effect=AssertionError("connected")):
            for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                os.environ.pop(var, None)
            with self.assertRaises(Exception) as caught:
                web._http_request("GET", "http://rebind.example/")
            self.assertIn("blocked", str(caught.exception))

    def test_a_proxied_request_keeps_the_name_check_only(self) -> None:
        with mock.patch.dict(os.environ, {"https_proxy": "http://proxy.site:3128",
                                          "no_proxy": ""}):
            handlers = [type(h) for h in web._opener("https://example.com/").handlers]
        self.assertNotIn(web._PinnedHTTPSHandler, handlers)
        with mock.patch.dict(os.environ, {}, clear=True):
            handlers = [type(h) for h in web._opener("https://example.com/").handlers]
        self.assertIn(web._PinnedHTTPSHandler, handlers)

    def test_headers_do_not_follow_a_redirect_to_another_host(self) -> None:
        handler = web._SafeRedirectHandler()
        req = urllib.request.Request("https://a.example/x", headers={"Authorization": "t"})
        with mock.patch.object(web, "_safe_url", lambda url: url):
            other = handler.redirect_request(req, None, 302, "Found", {},
                                             "https://b.example/y")
            same = handler.redirect_request(req, None, 302, "Found", {},
                                            "https://a.example/z")
        self.assertNotIn("Authorization", other.headers)
        self.assertIn("Authorization", same.headers)


class GithubTokenScopeTests(unittest.TestCase):
    """The GitHub token goes to GitHub: a raw URL from an answer may point anywhere."""

    def test_the_token_is_sent_to_github_hosts_only(self) -> None:
        from mimir.servers.external import server_github as gh
        with mock.patch.object(gh, "_TOKEN", "t"):
            self.assertIn("Authorization", gh._headers("https://api.github.com/repos/a/b"))
            self.assertIn("Authorization",
                          gh._headers("https://raw.githubusercontent.com/a/b/main/f"))
            self.assertNotIn("Authorization", gh._headers("https://evil.example/f"))
            self.assertNotIn("Authorization", gh._headers("http://raw.githubusercontent.com/f"))

    def test_owner_and_repo_stay_one_segment_each(self) -> None:
        from mimir.servers.external import server_github as gh
        self.assertEqual(gh._repo_path("a/../../x", "b?q"), "/repos/a%2F..%2F..%2Fx/b%3Fq")


if __name__ == "__main__":
    unittest.main()
