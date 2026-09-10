"""
MCP Web Server
==============
Provides safe HTTP fetch and JSON utilities.
Only http:// and https:// schemes are accepted.
Requests to loopback / link-local / private RFC-1918 addresses are blocked.
"""

import ipaddress
from html.parser import HTMLParser
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps, EXTERNAL_FETCH, PLAN_BLOCKED, IRREVERSIBLE
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

# Two different ceilings, because they defend against two different things.
#
# _MAX_BYTES bounds the socket read: it stops a hostile or runaway endpoint from
# streaming forever, and 512 KB has always been the right order for that.
#
# _MAX_TEXT_CHARS bounds what comes back to the *model*, which is a separate
# question nobody was asking. A 512 KB body is ~170k tokens on escape-dense
# markup — around 80% of a 256k window, so two fetches could and did put a
# session over it with no single call doing anything unusual. A ceiling
# expressed only in socket bytes cannot see that; this one is sized against the
# window it has to share.
_MAX_BYTES = 512 * 1024        # 512 KB — what we will read off the wire
_MAX_TEXT_CHARS = 128 * 1024   # 128 KB — what we will hand back to the model


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


class _TextExtractor(HTMLParser):
    """Visible text from an HTML document, script and style discarded.

    Stdlib only, and deliberately crude: the aim is to stop shipping markup to a
    model that wanted prose, not to render the page. Markup is most of a modern
    page's bytes — the tags, the class attributes, the inline JSON — and every one
    of those bytes is escaped again when the message is serialised, so it is paid
    for twice before anyone reads it.
    """

    _SKIP = {"script", "style", "noscript", "svg", "head"}
    _BREAK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BREAK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        joined = "".join(self._parts)
        lines = [ln.strip() for ln in joined.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def _looks_like_html(content_type: str, body: str) -> bool:
    if "html" in content_type.lower():
        return True
    head = body[:2048].lstrip().lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


def _readable_body(body: str, content_type: str, raw: bool) -> tuple[str, dict]:
    """The body as the model should receive it, plus what was done to it.

    Extraction happens *before* the size ceiling, not after: markup is what the
    ceiling would otherwise spend itself on, and a page whose prose fits easily
    should not be cut just because its tags did not.
    """
    note: dict = {}
    if not raw and _looks_like_html(content_type, body):
        parser = _TextExtractor()
        try:
            parser.feed(body)
            parser.close()
            extracted = parser.text()
        except Exception:
            extracted = ""
        # Empty is the only reliable sign this parser did not understand the page
        # — a script-shell app, or markup it choked on — and there returning the
        # markup beats returning nothing. Short-but-present text is not a failure:
        # a page really can be three sentences, and `extracted_text` in the reply
        # says what happened, so a caller who wanted the markup can ask for it.
        if extracted.strip():
            note["extracted_text"] = True
            note["html_chars"] = len(body)
            body = extracted
    if len(body) > _MAX_TEXT_CHARS:
        note["truncated"] = True
        note["full_chars"] = len(body)
        note["hint"] = (
            "Only the first part of this document was returned. Fetch a more "
            "specific URL, or ask for the section you need by name."
        )
        body = body[:_MAX_TEXT_CHARS]
    return body, note


def _http_request(method: str, url: str, headers: dict = None, data: bytes = None,
                  raw: bool = False) -> dict:
    safe_url = _safe_url(url)
    req = urllib.request.Request(safe_url, headers=headers or {}, data=data, method=method)
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    with opener.open(req, timeout=_TIMEOUT) as resp:
        # One byte past the cap, so a body that exactly fills it can be told from
        # one that was cut off.
        raw_bytes = resp.read(_MAX_BYTES + 1)
        wire_truncated = len(raw_bytes) > _MAX_BYTES
        raw_bytes = raw_bytes[:_MAX_BYTES]
        charset = resp.headers.get_content_charset("utf-8")
        content_type = resp.headers.get("Content-Type", "")
        body, note = _readable_body(
            raw_bytes.decode(charset, errors="replace"), content_type, raw)
        if wire_truncated:
            note["truncated"] = True
            note.setdefault("hint", (
                f"The response exceeded the {_MAX_BYTES // 1024} KB read limit and "
                f"was cut off. Fetch a more specific URL."
            ))
        return ok({
            "method": method,
            "url": url,
            "final_url": resp.geturl(),
            "http_status": getattr(resp, "status", None),
            "content_type": content_type,
            "body": body,
            **note,
        })


# ── tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(**tool_caps(
    # Not unconditionally sensitive: a GET is read-only, but the client treats it as
    # sensitive when the URL targets an authenticated/mutating endpoint. The `host`
    # scope (which arg carries the URL) is what drives that conditional gate and
    # narrows any "always" approval to one destination host.
    caps=[EXTERNAL_FETCH],
    scope={"args": ["url"], "kind": "host"},
    risk_note="fetches from an authenticated or otherwise sensitive endpoint",
    label="Fetching {url}",
))
def http_get(url: str, headers: dict = None, raw: bool = False) -> dict:
    """Perform an HTTP GET request and return the response body as readable text.

    An HTML page comes back as its visible text, with script, style and markup
    dropped; anything else comes back as-is. A long document is cut and the reply
    says so — ``truncated`` with a ``full_chars`` count — so what you get back is
    never silently a fragment.

    Args:
        url:     The target URL (http or https only).
        headers: Optional dict of extra request headers.
        raw:     Return the HTML untouched instead of its text. For reading the
                 markup itself — structure, attributes, embedded data.
    """
    try:
        return _http_request("GET", url, headers=headers, raw=raw)
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
    caps=[EXTERNAL_FETCH, PLAN_BLOCKED], reversibility=IRREVERSIBLE, non_batch=True,
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
