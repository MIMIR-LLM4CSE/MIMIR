"""User-attached context via ``@``-mention (Claude/Copilot-style).

Two kinds of things can be attached to a single turn, both by the *user* (never the
model — these are context, not model-invokable tools):

1. **MCP resources** — read-only, URI-addressed data a server exposes alongside its
   tools. Referenced as ``@<uri>`` (e.g. ``@memory://all``) or ``@<name>`` shorthand.
   The registry lives on the agent (``agent.resources``, populated at connect time by
   ``integration/server_manager``); reads dispatch via ``agent.read_resource``.
2. **Workspace files** — a path in the workspace, optionally with a line range:
   ``@src/foo.py`` (whole file) or ``@src/foo.py:10-20`` / ``@src/foo.py:10`` (a slice).
   Read locally against the workspace root (cwd), mirroring Copilot's ``#file`` / Claude's
   file attach. This needs no server and no registry.

This module is frontend-agnostic. Both the CLI and the WebSocket server call
:func:`augment_query_with_resources` to turn a raw user message into an effective query
with the referenced content prepended. Unknown ``@x`` tokens (not a known resource, not an
existing file) are left untouched so ordinary uses of ``@`` in prose (emails, handles) are
never swallowed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from ..tool_execution.normalizer import normalize_workspace_path
from ..tool_execution.validation import absolute_workspace_path

# A mention is ``@`` followed by a run of non-whitespace. We resolve the captured
# token against the registry / filesystem rather than constraining the pattern to a URI
# shape, so full URIs (``memory://all``), name shorthands (``memory``) and file paths
# (``src/foo.py:10-20``) are all supported.
_MENTION_RE = re.compile(r"(?<!\S)@(\S+)")

# ``path:start`` or ``path:start-end`` — a trailing line-range suffix on a file mention.
_RANGE_RE = re.compile(r"^(?P<path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$")

# Soft cap on a *whole-file* attach so a huge file can't blow the context window; a
# line range (the user asking for specific lines) is never capped.
_WHOLE_FILE_CHAR_CAP = 20_000


def _resolve_token(token: str, registry: dict[str, dict]) -> str | None:
    """Resolve a mention token to a registered resource URI, or None.

    Matches an exact URI first, then falls back to a case-insensitive resource-name
    match. Trailing punctuation (``.``, ``,`` …) commonly follows a mention in prose,
    so it is stripped progressively before giving up.
    """
    candidate = token
    while candidate:
        if candidate in registry:
            return candidate
        lowered = candidate.lower()
        for uri, info in registry.items():
            if str(info.get("name", "")).lower() == lowered:
                return uri
        if candidate[-1] in ".,;:!?)]}\"'":
            candidate = candidate[:-1]
            continue
        break
    return None


def parse_resource_mentions(
    text: str, registry: dict[str, dict]
) -> tuple[str, list[str]]:
    """Extract ``@<uri|name>`` mentions that resolve against ``registry``.

    Returns ``(cleaned_text, uris)`` where ``uris`` is the ordered, de-duplicated list
    of resolved resource URIs and ``cleaned_text`` has the matched mention tokens
    removed (whitespace collapsed). Unresolved ``@x`` tokens are preserved verbatim.
    """
    uris: list[str] = []
    seen: set[str] = set()

    def _sub(match: re.Match[str]) -> str:
        token = match.group(1)
        uri = _resolve_token(token, registry)
        if uri is None:
            return match.group(0)  # not a known resource — leave the text alone
        if uri not in seen:
            seen.add(uri)
            uris.append(uri)
        # Preserve any trailing punctuation the token-resolver peeled off.
        trailing = token[len(uri):] if token.startswith(uri) else ""
        matched_name = registry[uri].get("name", "")
        if not trailing and not token.startswith(uri):
            # name-shorthand match: recover trailing punctuation after the name
            trailing = token[len(str(matched_name)):]
        return trailing

    cleaned = _MENTION_RE.sub(_sub, text)
    # Collapse the double spaces / dangling spaces left by removed tokens.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, uris


def _render_blocks(blocks: list[tuple[str, str]]) -> str:
    """Join ``(header, content)`` pairs into one delimited attachment preamble.

    Empty-content entries are still shown (labelled ``(empty)``) so an intentional attach
    of currently-empty content is visible to the model rather than silently dropped.
    """
    rendered: list[str] = []
    for header, content in blocks:
        body = content if content.strip() else "(empty)"
        rendered.append(f"{header}\n{body}")
    return "\n---\n".join(rendered)


def build_attachment_block(items: list[tuple[str, str, str]]) -> str:
    """Format read resources into a single labelled context preamble.

    ``items`` is a list of ``(uri, name, content)``.
    """
    blocks = [
        (
            f"[Attached resource: {uri}"
            + (f" ({name})" if name and name != uri else "")
            + "]",
            content,
        )
        for uri, name, content in items
    ]
    return _render_blocks(blocks)


@dataclass
class _FileSpec:
    """A resolved workspace-file mention: display label, absolute path, optional range."""

    display: str          # workspace-relative label, e.g. ``src/foo.py`` or ``src/foo.py:10-20``
    abspath: str          # absolute path used for the actual read
    start: int | None     # 1-indexed first line, or None for the whole file
    end: int | None       # 1-indexed last line (inclusive), or None
    trailing: str = ""    # prose punctuation peeled off the token, re-inserted into the text


def _resolve_file_token(token: str, cwd: str | None) -> _FileSpec | None:
    """Resolve a mention token to an existing workspace file (with optional range).

    Returns a :class:`_FileSpec` if the token names a real file, else ``None`` (so the
    caller leaves the ``@token`` text untouched). A ``:start[-end]`` suffix is treated as
    a line range; otherwise the whole file is attached. Trailing prose punctuation on a
    bare path is peeled progressively (the full path — extension included — is tried first).
    """
    range_match = _RANGE_RE.match(token)
    if range_match:
        path = range_match.group("path")
        start = int(range_match.group("start"))
        end = int(range_match.group("end")) if range_match.group("end") else start
        abspath = absolute_workspace_path(path, cwd=cwd)
        if os.path.isfile(abspath):
            disp = normalize_workspace_path(path, cwd=cwd) or path
            label = f"{disp}:{start}" + (f"-{end}" if end != start else "")
            return _FileSpec(display=label, abspath=abspath, start=start, end=end)
        return None

    candidate = token
    trailing = ""
    while candidate:
        abspath = absolute_workspace_path(candidate, cwd=cwd)
        if os.path.isfile(abspath):
            disp = normalize_workspace_path(candidate, cwd=cwd) or candidate
            return _FileSpec(display=disp, abspath=abspath, start=None, end=None, trailing=trailing)
        if candidate[-1] in ".,;:!?)]}\"'":
            trailing = candidate[-1] + trailing
            candidate = candidate[:-1]
            continue
        break
    return None


def parse_file_mentions(
    text: str, cwd: str | None = None
) -> tuple[str, list[_FileSpec]]:
    """Extract ``@<path>[:start[-end]]`` mentions that name existing workspace files.

    Returns ``(cleaned_text, specs)`` with matched tokens removed (their trailing prose
    punctuation preserved) and de-duplicated file specs in first-seen order. Tokens that
    do not resolve to a real file are left verbatim.
    """
    specs: list[_FileSpec] = []
    seen: set[tuple[str, int | None, int | None]] = set()

    def _sub(match: re.Match[str]) -> str:
        spec = _resolve_file_token(match.group(1), cwd)
        if spec is None:
            return match.group(0)
        key = (spec.abspath, spec.start, spec.end)
        if key not in seen:
            seen.add(key)
            specs.append(spec)
        return spec.trailing

    cleaned = _MENTION_RE.sub(_sub, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, specs


def read_file_slice(spec: _FileSpec) -> str:
    """Read a file (or a 1-indexed inclusive line slice) for attachment; best-effort.

    Whole-file reads are soft-capped at ``_WHOLE_FILE_CHAR_CAP`` with a truncation note;
    an explicit line range is never capped. Returns a short error note (never raises) when
    the file cannot be read.
    """
    try:
        with open(spec.abspath, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        return f"(could not read file: {exc})"

    if spec.start is None:
        content = "".join(lines)
        if len(content) > _WHOLE_FILE_CHAR_CAP:
            content = (
                content[:_WHOLE_FILE_CHAR_CAP]
                + f"\n… (truncated at {_WHOLE_FILE_CHAR_CAP} chars — attach a line range like "
                f"@{spec.display}:1-50 for a specific section)"
            )
        return content

    start = max(spec.start, 1)
    end = spec.end if spec.end is not None else start
    return "".join(lines[start - 1:end])


async def augment_query_with_resources(
    agent: Any, text: str, *, cwd: str | None = None
) -> tuple[str, list[str]]:
    """Resolve ``@``-mentions in ``text`` and prepend the referenced content.

    Handles both MCP resources (``@<uri|name>``) and workspace files
    (``@<path>[:start[-end]]``). Returns ``(effective_query, attached_labels)``; when
    nothing is referenced, returns ``text`` unchanged and an empty list — so callers can
    keep passing the raw text with zero behavioural change. Frontends use the returned
    labels to print an attach confirmation while storing the *raw* text in history.

    ``cwd`` defaults to the process working directory (the workspace root), which is where
    both the CLI and the WS worker run — so callers need not pass it.
    """
    if "@" not in text:
        return text, []

    registry = getattr(agent, "resources", None) or {}
    # Resource mentions first (they win over a like-named file); file mentions then scan
    # whatever survived — unresolved resource tokens are left verbatim for the file pass.
    cleaned, uris = parse_resource_mentions(text, registry) if registry else (text, [])
    cleaned, file_specs = parse_file_mentions(cleaned, cwd=cwd)
    if not uris and not file_specs:
        return text, []

    blocks: list[tuple[str, str]] = []
    labels: list[str] = []

    for uri in uris:
        content = await agent.read_resource(uri)
        name = registry.get(uri, {}).get("name", uri)
        header = (
            f"[Attached resource: {uri}"
            + (f" ({name})" if name and name != uri else "")
            + "]"
        )
        blocks.append((header, content))
        labels.append(uri)

    for spec in file_specs:
        blocks.append((f"[Attached file: {spec.display}]", read_file_slice(spec)))
        labels.append(spec.display)

    block = _render_blocks(blocks)
    effective = f"{block}\n---\n\n{cleaned}" if cleaned else block
    return effective, labels


__all__ = [
    "parse_resource_mentions",
    "parse_file_mentions",
    "build_attachment_block",
    "read_file_slice",
    "augment_query_with_resources",
]
