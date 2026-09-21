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
# Raising this used to be unaffordable: what we read and what we handed the model were
# one and the same, so a bigger read meant a bigger prompt. The two ceilings below
# decoupled them — prose, no-prose and error bodies are each bounded on their own — so
# the read now costs time and memory, not context. It matters because metadata lives in
# <head> and a page can put a great deal in front of it: on the thesis record page that
# prompted all this, <meta name="description"> sits at byte 700 336, behind ~700 KB of
# inline CSS. At 512 KB the abstract was not slow to reach, it was unreachable.
_MAX_BYTES = 2 * 1024 * 1024   # 2 MB — what we will read off the wire
_MAX_TEXT_CHARS = 128 * 1024   # 128 KB — what we will hand back to the model

# What a document the parser found NO prose in may spend. It used to share the
# ceiling above, on the reasoning that markup beats nothing; measured, that is
# backwards. A thesis record page: 754 KB, 92% of it one inline <style>. The socket
# read stops at 512 KB, entirely inside that block, so the body never arrives, the
# parser correctly reports no text — and the fallback then spent 131 072 chars,
# ~34k tokens, on CSS. Empty extraction is the strongest evidence there is that the
# markup holds no prose; the only thing it is still good for is showing the caller
# what kind of page this is, which costs a sample and not a window.
_MARKUP_FALLBACK_CHARS = 8 * 1024

# What a FAILED request may spend. The error branches used to hand back up to
# _MAX_BYTES of the error body untouched — no extraction, no text ceiling, nothing.
# Measured on a paper host answering 403: the block page came back as 131 164 tokens,
# and four fetches issued in one step put a 200k window at 215k before anything could
# be trimmed. An error body is a diagnosis, never content: what is wanted from it is
# "Access Denied", "rate limited", "captcha", and that fits in a couple of KB.
_ERROR_BODY_READ = 64 * 1024   # what we read off a failed response
_ERROR_BODY_CHARS = 2 * 1024   # what we hand back of it

# Targeted reading. A document that does not fit is not thereby unusable: what the
# caller wanted is nearly always one region of it, and `truncated` alone left them
# with the first 128 KB and no way to ask for the rest.
_MATCH_WINDOW_CHARS = 600      # text kept around an occurrence
_MAX_MATCHES = 10              # occurrences reported for one query


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

    Headings survive as headings. `h1`…`h6` used to sit in `_BREAK` and produce a
    newline like any `<div>`, which left a long document as one undifferentiated wall
    of prose: nothing said where the abstract ended and the bibliography began. They
    are marked instead, in the one notation a model reads without being told —
    Markdown's — so a document comes back as named regions, and so a reader can ask
    for one of them by name.
    """

    _SKIP = {"script", "style", "noscript", "svg", "head"}
    _BREAK = {"p", "br", "div", "li", "tr"}
    _HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._HEADINGS:
            self._parts.append("\n" + "#" * self._HEADINGS[tag] + " ")
        elif tag in self._BREAK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._HEADINGS:
            # Closed explicitly: the next run of text is body, not more heading.
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        joined = "".join(self._parts)
        lines = [ln.strip() for ln in joined.splitlines()]
        # A heading whose element held no text would leave its bare marker behind.
        return "\n".join(ln for ln in lines if ln and ln.strip("#").strip())


class _EmbeddedData(HTMLParser):
    """The JSON a script-shell page carries its own content in.

    The pages prose extraction fails on are largely the ones that ship their record
    as data rather than as text: `application/ld+json` for a citation or a product,
    `__NEXT_DATA__` and its equivalents for a whole rendered view. Dropping those and
    keeping only `<head>` metadata would answer a 34k-token page by throwing away the
    one part of it that held the answer — so they are harvested before the markup
    sample, under the same budget.
    """

    _TYPES = ("application/json", "application/ld+json")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._blocks: list[str] = []
        self._keep: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "script":
            return
        a = {k.lower(): (v or "") for k, v in attrs}
        ctype = a.get("type", "").split(";")[0].strip().lower()
        if ctype in self._TYPES:
            self._keep = a.get("id") or ctype

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._keep = None

    def handle_data(self, data: str) -> None:
        if self._keep and data.strip():
            self._blocks.append(f"<{self._keep}>\n{data.strip()}")

    def blocks(self) -> list[str]:
        return self._blocks


def _embedded_data(body: str, budget: int) -> str:
    """Embedded JSON blocks, largest first, up to *budget* chars.

    Largest first because on a script shell the big blob is the content and the small
    ones are analytics config. Each block is whole or absent: half a JSON document
    costs as much to read and cannot be parsed.
    """
    parser = _EmbeddedData()
    try:
        parser.feed(body)
        parser.close()
    except Exception:
        return ""
    kept: list[str] = []
    left = budget
    for block in sorted(parser.blocks(), key=len, reverse=True):
        if len(block) + 1 <= left:
            kept.append(block)
            left -= len(block) + 1
    return "\n".join(kept)


class _HeadSummary(HTMLParser):
    """The page's own summary of itself: its title and its descriptive metadata.

    Worth having because the pages prose extraction fails on are largely the ones
    that describe themselves best. A record page built as a script shell, or one
    whose body never arrived, still carries `og:title`, `description` and the
    `citation_*` set in its `<head>` — which is the first thing off the wire, so it
    survives a read that was cut long before the body.
    """

    _WANTED = ("description", "og:title", "og:description", "citation_title",
               "citation_author", "citation_publication_date", "citation_doi")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._meta: dict[str, str] = {}
        self._title: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._in_title = True
            return
        if tag != "meta":
            return
        a = {k.lower(): (v or "") for k, v in attrs}
        key = (a.get("name") or a.get("property") or "").lower()
        content = a.get("content", "").strip()
        # First wins: a page repeating og:description sets it once meaningfully and
        # again in a share widget, and the head's own copy is the one it means.
        if key in self._WANTED and content and key not in self._meta:
            self._meta[key] = content

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)

    def text(self) -> str:
        lines = []
        title = "".join(self._title).strip()
        if title:
            lines.append(title)
        for key in self._WANTED:
            if key in self._meta and self._meta[key] != title:
                lines.append(f"{key}: {self._meta[key]}")
        return "\n".join(lines)


def _head_summary(body: str) -> str:
    """``_HeadSummary`` over a document, or "" when it yields nothing."""
    parser = _HeadSummary()
    try:
        parser.feed(body)
        parser.close()
    except Exception:
        return ""
    return parser.text().strip()


def _looks_like_html(content_type: str, body: str) -> bool:
    if "html" in content_type.lower():
        return True
    head = body[:2048].lstrip().lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


# application/* subtypes that are text despite the family. Anything outside this
# set (pdf, zip, octet-stream, …) is treated as bytes, as is every non-text family.
_TEXTUAL_APPLICATION_SUBTYPES = frozenset({
    "json", "xml", "javascript", "ecmascript", "x-javascript",
    "yaml", "x-yaml", "toml", "x-toml", "csv", "x-www-form-urlencoded",
    "sql", "graphql", "x-ndjson", "ld+json", "rss+xml", "atom+xml",
})


def _looks_binary(body: str, content_type: str) -> bool:
    """True when *body* is bytes that were never text, whatever the ceiling allows.

    A PDF, an image or an archive decoded with ``errors="replace"`` is not shorter
    than a page — it is the same size in mojibake, and it JSON-escapes to more than
    twice that on the way into the model's context (measured 2026-09-20: a 131072-char
    PDF prefix went in as 502529 characters, ~125K tokens of catalog objects and
    stream bytes, in a window budgeted for 160K). None of it can be read, so the
    ceiling is the wrong instrument: the answer is not a smaller slice of binary.

    Judged on the decoded text rather than on the declared type alone, since a server
    that mislabels a PDF as octet-stream — or as nothing at all — must not get a pass.
    """
    main = content_type.split(";")[0].strip().lower()
    if main and "/" in main:
        family, _, sub = main.partition("/")
        # text/* is text by declaration; the handful of application/* subtypes below
        # are the textual ones a fetch legitimately lands on. Everything else — pdf,
        # zip, octet-stream, image, audio, video, font — is bytes.
        textual = family == "text" or (
            family == "application"
            and (sub in _TEXTUAL_APPLICATION_SUBTYPES
                 or sub.endswith(("+json", "+xml")))
        )
        if not textual:
            return True
    head = body[:4096]
    if not head:
        return False
    if "\x00" in head:
        return True
    if head.lstrip().startswith("%PDF-"):
        return True
    # U+FFFD is what a byte that is not this charset decodes to. Prose does not
    # produce them; a compressed stream produces little else.
    return head.count("\ufffd") > len(head) // 20


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
        # Empty is the only reliable sign this parser did not understand the page —
        # a script-shell app, markup it choked on, or a body that never arrived.
        # Short-but-present text is not a failure: a page really can be three
        # sentences, and `extracted_text` in the reply says what happened, so a
        # caller who wanted the markup can ask for it with raw=True.
        if extracted.strip():
            note["extracted_text"] = True
            note["html_chars"] = len(body)
            body = extracted
        else:
            note["extracted_text"] = False
            note["html_chars"] = len(body)
            summary = _head_summary(body)
            # The page's own metadata first — it is small and it is always about the
            # page — then whatever structured data it carries, in what is left of the
            # budget. On a script shell that data IS the content.
            data = _embedded_data(
                body, max(0, _MARKUP_FALLBACK_CHARS - len(summary) - 2))
            if summary or data:
                note["from_metadata"] = bool(summary)
                note["embedded_data"] = bool(data)
                # body_note, not hint: `hint` is a reserved protocol key that
                # responses.ok() strips from success payloads, so every one of these
                # lines had been written for a model that never received it.
                note["body_note"] = (
                    "No readable text could be extracted from this page — it is a "
                    "script shell, or its body did not arrive. What is below is the "
                    "page's <head> metadata and any structured data it embeds, not "
                    "its rendered text. Fetch the site's API or a direct document "
                    "URL for the rest, or pass raw=True to get the markup."
                )
                body = "\n\n".join(p for p in (summary, data) if p)
            else:
                note["markup_only"] = True
                note["body_note"] = (
                    f"No readable text could be extracted from this page and it "
                    f"carries no descriptive metadata, so only the first "
                    f"{_MARKUP_FALLBACK_CHARS // 1024} KB of markup is returned, to "
                    f"show what kind of page it is. Fetch the site's API or a direct "
                    f"document URL, or pass raw=True for the whole body."
                )
                body = body[:_MARKUP_FALLBACK_CHARS]
    return body, note


def _fit_body(body: str, note: dict, offset: int = 0) -> str:
    """Apply the text ceiling, and say where to resume.

    Deliberately the LAST step, after any targeting. It used to live at the end of
    `_readable_body`, which meant the document was cut to 128 KB before `offset` was
    applied to it — so page two of a long read came back empty, and the resume this
    ceiling exists to enable could never actually be taken. `next_offset` is absolute
    for the same reason: it is handed straight back as the next call's `offset`.
    """
    if len(body) <= _MAX_TEXT_CHARS:
        return body
    note["truncated"] = True
    note["full_chars"] = offset + len(body)
    note["next_offset"] = offset + _MAX_TEXT_CHARS
    note["body_note"] = (
        "Only part of this document was returned. Pass `offset=<next_offset>` to "
        "continue from where this stops, or `contains=<what you are after>` to get "
        "just that part of it."
    )
    return body[:_MAX_TEXT_CHARS]


def _sections(text: str) -> list[tuple[str, int, int]]:
    """The document's regions as ``(heading, start, end)`` offsets into *text*.

    Built on the Markdown headings ``_TextExtractor`` now emits. Text before the
    first heading is its own region with an empty heading, so an offset always falls
    inside exactly one of them.
    """
    out: list[tuple[str, int, int]] = []
    heading = ""
    start = 0
    pos = 0
    for line in text.splitlines(keepends=True):
        if line.startswith("#") and line.strip("#").strip():
            if pos > start or heading:
                out.append((heading, start, pos))
            heading = line.strip().lstrip("#").strip()
            start = pos
        pos += len(line)
    out.append((heading, start, len(text)))
    return [r for r in out if r[2] > r[1]]


def _find_matches(text: str, query: str) -> list[dict]:
    """Regions and windows of *text* that answer *query*, best first.

    Two passes, because a document names its own parts. A query matching a heading
    wants that whole region — asking a thesis page for "résumé" should return the
    abstract, not six sentences containing the word. Only when no heading matches do
    we fall back to windows around the occurrences themselves.

    Ranking reuses ``lexical_rank`` (servers/_shared/embed.py), the deterministic
    token-overlap scorer already shared by the memory and platform searches.
    """
    if not query.strip():
        return []
    from embed import lexical_rank, relevance_tokens

    qtok = relevance_tokens(query)
    regions = _sections(text)
    named = [r for r in regions if r[0] and (qtok & relevance_tokens(r[0]))]
    if named:
        return [{"at": start, "heading": head, "text": text[start:end]}
                for head, start, end in named[:_MAX_MATCHES]]

    lowered = text.lower()
    needle = query.strip().lower()
    # (window start, window end, position of the first hit inside it). The hit is
    # tracked apart from the window because the window opens half its width EARLIER,
    # which on a short document lands in the previous region — naming that region
    # would tell the caller the excerpt is somewhere it is not.
    spans: list[tuple[int, int, int]] = []
    at = lowered.find(needle)
    while at != -1 and len(spans) < _MAX_MATCHES * 4:
        lo = max(0, at - _MATCH_WINDOW_CHARS // 2)
        hi = min(len(text), at + len(needle) + _MATCH_WINDOW_CHARS // 2)
        # Overlapping windows are one window: two hits a line apart must not be
        # reported as two nearly identical excerpts.
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], hi, spans[-1][2])
        else:
            spans.append((lo, hi, at))
        at = lowered.find(needle, at + len(needle))
    if not spans:
        return []

    def _heading_at(pos: int) -> str:
        for head, start, end in regions:
            if start <= pos < end:
                return head
        return ""

    windows = [text[lo:hi] for lo, hi, _hit in spans]
    ranked = lexical_rank(query, windows)
    return [{"at": spans[i][0], "heading": _heading_at(spans[i][2]),
             "text": windows[i]}
            for i, _score in ranked[:_MAX_MATCHES]]


def _targeted(body: str, contains: str, offset: int) -> tuple[str, dict]:
    """Apply ``contains`` / ``offset`` to a body that is already readable text.

    ``contains`` wins when both are given: naming what you want is more specific than
    naming where it starts. Resume keys follow ``read_file_lines`` — ``truncated``,
    ``total_chars``, ``next_offset`` — because the client already reads that contract
    (tool_execution/executor.py `_build_continuation_hint`) and turns it into a
    MORE_CONTENT hint on its own.
    """
    note: dict = {}
    if contains.strip():
        matches = _find_matches(body, contains)
        note["query"] = contains
        note["match_count"] = len(matches)
        note["total_chars"] = len(body)
        if not matches:
            note["body_note"] = (
                f"Nothing in this document matches {contains!r}. It has "
                f"{len(body)} characters; drop `contains` to read it from the top, "
                f"or try the wording the page itself would use."
            )
            return "", note
        kept: list[str] = []
        used = 0
        for m in matches:
            block = (f"[at {m['at']}"
                     + (f" · {m['heading']}" if m["heading"] else "") + "]\n"
                     + m["text"])
            if used + len(block) > _MAX_TEXT_CHARS:
                note["truncated"] = True
                break
            kept.append(block)
            used += len(block)
        note["matches_returned"] = len(kept)
        return "\n\n".join(kept), note

    if offset:
        offset = max(0, min(int(offset), len(body)))
        note["offset"] = offset
        note["total_chars"] = len(body)
        body = body[offset:]
    return body, note


def _read_body(decoded: str, content_type: str, raw: bool,
               contains: str, offset: int) -> tuple[str, dict]:
    """Extract, then target, then fit — in that order, which is the whole point.

    Any other order loses: fitting before targeting cuts away the very part the
    caller asked for, and targeting before extracting searches markup instead of
    prose.
    """
    if _looks_binary(decoded, content_type):
        # Handing the caller the bytes would spend the window on something no reader
        # can use. Saying what it is leaves the next move available — a text mirror,
        # an extraction service, a download — which is what the model did on its own
        # once the PDF's bytes had already cost it the context.
        return "", {
            "binary": True,
            "content_type": content_type,
            "body_omitted": True,
            "hint": (
                "The response is binary, not text, so its bytes are not returned. "
                "Fetch a text rendering of this resource instead."
            ),
        }
    body, note = _readable_body(decoded, content_type, raw)
    if contains or offset:
        body, targeted = _targeted(body, contains, offset)
        note.update(targeted)
    return _fit_body(body, note, offset=note.get("offset", 0)), note


def _error_body(exc: urllib.error.HTTPError) -> tuple[str, dict]:
    """The readable part of a failed response, and what was done to it.

    Same treatment as a successful body — markup becomes text, because a block page
    is mostly markup and its one useful sentence is buried in it — then a far tighter
    ceiling, because nothing downstream is going to *use* this text. It only has to
    say why the request failed.
    """
    try:
        body = exc.read(_ERROR_BODY_READ).decode("utf-8", errors="replace")
    except Exception:
        return "", {}
    ctype = ""
    try:
        ctype = exc.headers.get("Content-Type", "") or ""
    except Exception:
        pass
    if _looks_like_html(ctype, body):
        parser = _TextExtractor()
        try:
            parser.feed(body)
            parser.close()
            extracted = parser.text().strip()
        except Exception:
            extracted = ""
        if extracted:
            body = extracted
    note: dict = {}
    if len(body) > _ERROR_BODY_CHARS:
        note["body_truncated"] = True
        body = body[:_ERROR_BODY_CHARS]
    return body, note


def _http_request(method: str, url: str, headers: dict = None, data: bytes = None,
                  raw: bool = False, contains: str = "", offset: int = 0) -> dict:
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
        # Targeting applies to whatever body would have been returned, extracted or
        # raw: the caller asked for part of what they would otherwise have received.
        body, note = _read_body(
            raw_bytes.decode(charset, errors="replace"), content_type, raw,
            contains, offset)
        if wire_truncated:
            note["truncated"] = True
            # setdefault: a ceiling further down has already said something more
            # specific about what came back, and that is the better advice.
            note.setdefault("body_note", (
                f"The document is larger than the {_MAX_BYTES // (1024 * 1024)} MB "
                f"read limit and was cut off at the source, so its end is missing "
                f"here whatever offset you pass. Fetch a more specific URL, or the "
                f"site's API."
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
def http_get(url: str, headers: dict = None, raw: bool = False,
             contains: str = "", offset: int = 0) -> dict:
    """Perform an HTTP GET request and return the response body as readable text.

    An HTML page comes back as its visible text, with script, style and markup
    dropped, and its headings kept as Markdown (``## Résumé``) so the document reads
    as named regions. Anything else comes back as-is. A long document is cut and the
    reply says so — ``truncated``, ``full_chars``, ``next_offset`` — so what you get
    back is never silently a fragment.

    Prefer ``contains`` over reading a long page whole: it is one request either way,
    and the reply is the part you asked about instead of the first 128 KB.

    Args:
        url:      The target URL (http or https only).
        headers:  Optional dict of extra request headers.
        raw:      Return the HTML untouched instead of its text. For reading the
                  markup itself — structure, attributes, embedded data.
        contains: Return only the parts of the document that answer this, best first,
                  each with its offset and the heading it sits under. A word matching
                  a heading returns that whole region; otherwise you get windows
                  around the occurrences. ``match_count`` says how many were found.
        offset:   Start reading at this character offset instead of the beginning —
                  pass the ``next_offset`` of a truncated reply to continue. Ignored
                  when ``contains`` is given.
    """
    try:
        return _http_request("GET", url, headers=headers, raw=raw,
                             contains=contains, offset=offset)
    except ValueError as e:
        return err(str(e), hint="Use a public http(s) URL that does not resolve to internal addresses.")
    except urllib.error.HTTPError as e:
        body, note = _error_body(e)
        return err(
            f"HTTP {e.code}: {e.reason}",
            http_status=e.code,
            body=body,
            url=url,
            **note,
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
        body, note = _error_body(e)
        return err(
            f"HTTP {e.code}: {e.reason}",
            http_status=e.code,
            body=body,
            url=url,
            **note,
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
