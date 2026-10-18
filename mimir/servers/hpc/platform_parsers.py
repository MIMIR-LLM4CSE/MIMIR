"""Reading what the module system prints: ``module avail`` (terse), ``module whatis``
and Lmod's spider JSON, into the entries of the module catalogue."""
from __future__ import annotations

import os
import re
import json

# Lmod decorates a name with (D)efault, (L)oaded, and friends.
_AVAIL_MARKER_RE = re.compile(r"\(([A-Za-z],?)+\)$")


# Chatter Lmod prints among the names when stdout and stderr are merged.
_AVAIL_NOISE = ("Where:", "Use \"module", "Use 'module", "If the avail list",
                "To learn more", "The following")


def split_name(load: str) -> tuple[str, str]:
    """'cuda/12.2' -> ('cuda', '12.2'); a bare 'cmake' -> ('cmake', '')."""
    if "/" in load:
        head, _, tail = load.rpartition("/")
        return head, tail
    return load, ""


def parse_avail_terse(stdout: str) -> list[dict]:
    """Parse `module -t avail` with stderr merged in.

    The merge is deliberate: Lmod writes the MODULEPATH section headers to stderr and
    the names to stdout, so merging is the only way to attribute a name to its tree.
    Headers are therefore *consumed* (they set the current root), never just dropped.
    """
    entries: list[dict] = []
    seen: set[str] = set()
    root = ""
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("-"):
            continue
        if any(line.startswith(prefix) for prefix in _AVAIL_NOISE):
            continue
        if line.endswith(":"):
            candidate = line[:-1].strip()
            if candidate.startswith("/") or os.path.isdir(candidate):
                root = candidate
            continue
        default = False
        marker = _AVAIL_MARKER_RE.search(line)
        if marker:
            flags = marker.group(0).strip("()").split(",")
            default = any(f.strip().upper() in ("D", "DEFAULT") for f in flags)
            line = line[:marker.start()].strip()
        if not line or " " in line:
            continue
        if line in seen:
            if default:
                for e in entries:
                    if e["load"] == line:
                        e["default"] = True
            continue
        seen.add(line)
        name, version = split_name(line)
        entries.append({
            "kind": "module",
            "key": line,
            "load": line,
            "name": name,
            "version": version,
            "default": default,
            "description": "",
            "category": "",
            "keywords": [],
            "source": os.path.join(root, line) if root else "",
            "tier": "name",
        })
    return entries


def _whatis_description(lines: list[str], load: str) -> str:
    """Squeeze a one-line description out of whatis output, whatever shape it took.

    Lmod and Tcl Modules disagree on the layout, and sites add their own; rather than
    match a format, take the text after the first colon on each line, drop the parts
    that only repeat the module's own name, and join what is left.
    """
    pieces = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if ":" in text:
            head, _, tail = text.partition(":")
            if head.strip() in (load, load.split("/")[0], "Description", "description"):
                text = tail.strip()
            elif head.strip().startswith(load.split("/")[0]):
                text = tail.strip()
        if text.lower().startswith("description:"):
            text = text.split(":", 1)[1].strip()
        if not text or text == load:
            continue
        pieces.append(text)
    return " ".join(pieces)[:400]


def parse_whatis_bulk(stdout: str, names: list[str]) -> dict:
    """Parse the '@@<name>'-delimited whatis loop into {load: description}."""
    out: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for raw in stdout.splitlines():
        if raw.startswith("@@"):
            if current is not None:
                out[current] = _whatis_description(buffer, current)
            current = raw[2:].strip()
            buffer = []
            continue
        buffer.append(raw)
    if current is not None:
        out[current] = _whatis_description(buffer, current)
    return {k: v for k, v in out.items() if v}


def parse_whatis_flat(stdout: str, known: set[str]) -> dict:
    """Parse Tcl Modules' argument-less `module whatis`, one line per module."""
    out: dict[str, str] = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        head, _, tail = line.partition(":")
        load = head.strip()
        if load in known and tail.strip():
            out[load] = tail.strip()[:400]
    return out


def _spider_versions(package: dict) -> list[dict]:
    versions = package.get("versions")
    return versions if isinstance(versions, list) else []


def parse_spider_json(stdout: str) -> dict:
    """Either Lmod spider JSON shape as ``{load: {description, category, ...}}``; {} if unrecognized."""
    try:
        data = json.loads(stdout)
    except (TypeError, ValueError):
        return {}

    out: dict[str, dict] = {}

    def _record(load, description, path, category, keywords, default):
        if not load:
            return
        out[load] = {
            "description": (description or "").strip()[:400],
            "source": path or "",
            "category": category or "",
            "keywords": keywords or [],
            "default": bool(default),
        }

    if isinstance(data, list):
        for package in data:
            if not isinstance(package, dict):
                continue
            base_desc = package.get("description") or ""
            categories = package.get("categories") or package.get("category") or ""
            if isinstance(categories, list):
                categories = ", ".join(str(c) for c in categories)
            keywords = package.get("keywords") or []
            if not isinstance(keywords, list):
                keywords = []
            for version in _spider_versions(package):
                if not isinstance(version, dict):
                    continue
                _record(
                    version.get("full") or version.get("fullName"),
                    version.get("description") or base_desc,
                    version.get("path") or version.get("pathToDB"),
                    categories,
                    [str(k) for k in keywords],
                    version.get("markedDefault") or version.get("default"),
                )
        return out

    if isinstance(data, dict):
        for _name, by_path in data.items():
            if not isinstance(by_path, dict):
                continue
            for path, record in by_path.items():
                if not isinstance(record, dict):
                    continue
                keywords = record.get("keywords") or record.get("propT") or []
                if not isinstance(keywords, list):
                    keywords = []
                _record(
                    record.get("fullName") or record.get("full"),
                    record.get("Description") or record.get("description"),
                    record.get("path") or path,
                    record.get("category") or "",
                    [str(k) for k in keywords],
                    record.get("markedDefault") or record.get("default"),
                )
        return out

    return {}
