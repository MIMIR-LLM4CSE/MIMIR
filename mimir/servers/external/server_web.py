"""
MCP Web Server
==============
Provides safe HTTP fetch and JSON utilities.
Only http:// and https:// schemes are accepted.
Requests to loopback / link-local / private RFC-1918 addresses are blocked.
"""

import ipaddress
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps, PLAN_BLOCKED, IRREVERSIBLE
from responses import err, ok

mcp = FastMCP(
    "WebServer",
    debug=False,
    log_level="ERROR",
)

# ── security helpers ──────────────────────────────────────────────────────────

_BLOCKED_NETS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local
    ipaddress.ip_network("fc00::/7"),          # ULA IPv6
]

_TIMEOUT = 10   # seconds
_MAX_BYTES = 512 * 1024  # 512 KB


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    # Block IP classes commonly used for SSRF pivoting and non-routable targets.
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or any(ip in net for net in _BLOCKED_NETS)
    )


def _resolve_all_ips(host: str) -> list[ipaddress._BaseAddress]:
    # Resolve all A/AAAA records to avoid TOCTOU on single-record checks.
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    ips: dict[str, ipaddress._BaseAddress] = {}
    for info in infos:
        sockaddr = info[4]
        ip_text = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_text)
            ips[str(ip_obj)] = ip_obj
        except ValueError:
            continue
    return list(ips.values())


def _safe_url(url: str) -> str:
    """Raise ValueError for non-http(s), unresolved, or internal/private targets."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Scheme '{parsed.scheme}' is not allowed. Use http or https.")
    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a hostname.")

    try:
        ips = _resolve_all_ips(host)
    except socket.gaierror as exc:
        raise ValueError(f"DNS resolution failed for host '{host}': {exc}")

    if not ips:
        raise ValueError(f"Could not resolve any IP address for host '{host}'.")

    blocked = [str(ip) for ip in ips if _is_blocked_ip(ip)]
    if blocked:
        raise ValueError(f"Requests to internal/non-routable addresses are blocked: {blocked}")

    return url


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate each redirect target before allowing urllib to follow it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        resolved = urllib.parse.urljoin(req.full_url, newurl)
        _safe_url(resolved)
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def _http_request(method: str, url: str, headers: dict = None, data: bytes = None) -> dict:
    safe_url = _safe_url(url)
    req = urllib.request.Request(safe_url, headers=headers or {}, data=data, method=method)
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    with opener.open(req, timeout=_TIMEOUT) as resp:
        body = resp.read(_MAX_BYTES)
        charset = resp.headers.get_content_charset("utf-8")
        return ok({
            "method": method,
            "url": url,
            "final_url": resp.geturl(),
            "http_status": getattr(resp, "status", None),
            "content_type": resp.headers.get("Content-Type", ""),
            "body": body.decode(charset, errors="replace"),
        })


# ── tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(**tool_caps(
    # Not unconditionally sensitive: a GET is read-only, but the client treats it as
    # sensitive when the URL targets an authenticated/mutating endpoint. The `host`
    # scope (which arg carries the URL) is what drives that conditional gate and
    # narrows any "always" approval to one destination host.
    scope={"args": ["url"], "kind": "host"},
    risk_note="fetches from an authenticated or otherwise sensitive endpoint",
    label="Fetching {url}",
))
def http_get(url: str, headers: dict = None) -> dict:
    """Perform an HTTP GET request and return the response body (text).

    Args:
        url:     The target URL (http or https only).
        headers: Optional dict of extra request headers.
    """
    try:
        return _http_request("GET", url, headers=headers)
    except ValueError as e:
        return err(str(e), hint="Use a public http(s) URL that does not resolve to internal addresses.")
    except urllib.error.HTTPError as e:
        body = e.read(_MAX_BYTES).decode("utf-8", errors="replace")
        return err(
            f"HTTP {e.code}: {e.reason}",
            http_status=e.code,
            body=body,
            url=url,
        )
    except Exception as e:
        return err(str(e), url=url)


@mcp.tool(**tool_caps(
    caps=[PLAN_BLOCKED], reversibility=IRREVERSIBLE, non_batch=True,
    scope={"args": ["url"], "kind": "host"},
    risk_note="sends data to an external service",
    label="Posting to {url}",
))
def http_post(url: str, payload: dict, headers: dict = None) -> dict:
    """Perform an HTTP POST request with a JSON payload and return the response body.

    Args:
        url:     The target URL (http or https only).
        payload: Dict to send as JSON body.
        headers: Optional dict of extra request headers.
    """
    try:
        data = json.dumps(payload).encode("utf-8")
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        return _http_request("POST", url, headers=h, data=data)
    except ValueError as e:
        return err(str(e), hint="Use a public http(s) URL that does not resolve to internal addresses.")
    except urllib.error.HTTPError as e:
        body = e.read(_MAX_BYTES).decode("utf-8", errors="replace")
        return err(
            f"HTTP {e.code}: {e.reason}",
            http_status=e.code,
            body=body,
            url=url,
        )
    except Exception as e:
        return err(str(e), url=url)


@mcp.tool()
def parse_json(text: str) -> dict:
    """Parse a JSON string and return a pretty-printed version.

    Args:
        text: Raw JSON string.
    """
    try:
        data = json.loads(text)
        return ok({"data": data})
    except Exception as e:
        return err(str(e), hint="Ensure the input text is valid JSON.")


@mcp.tool()
def json_extract(text: str, key_path: str) -> dict:
    """Extract a value from a JSON string by dotted key path.

    Args:
        text:      Raw JSON string.
        key_path:  Dot-separated path, e.g. 'results.0.title'.
    """
    try:
        obj = json.loads(text)
        for key in key_path.split("."):
            if isinstance(obj, list):
                obj = obj[int(key)]
            else:
                obj = obj[key]
        return ok({"key_path": key_path, "value": obj})
    except (KeyError, IndexError, TypeError):
        return err(f"Key path '{key_path}' not found.")
    except Exception as e:
        return err(str(e), hint="Ensure the input text is valid JSON and the key_path is correct.")


if __name__ == "__main__":
    mcp.run()
