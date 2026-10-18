"""How a tool call reads on screen: its status line, its shortened arguments, its result."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from urllib.parse import urlparse

from .capabilities import CONVENTIONAL_PATH_ARGS, READ, SEARCH_WITH_PATH, ToolCaps, arg_role, has_cap

logger = logging.getLogger(__name__)

def _relpath(path: str) -> str:
    """The file name alone, for an activity row.

    Never for an approval prompt, where the full location is the decision.
    """
    if not path:
        return path
    return os.path.basename(path.rstrip("/")) or path


def shorten_display_args(name: str, args: dict[str, Any], tool_caps: dict[str, ToolCaps] | None = None) -> dict[str, Any]:
    """*args* with absolute path arguments reduced to their file name, for display.

    Paths are the ``path`` arg-role, or the conventional argument names for tools that do
    not declare it. Only absolute values are shortened: a relative one is already the
    path the row means.
    """
    if not isinstance(args, dict):
        return args
    declared = arg_role(name, "path", tool_caps) or ()
    keys = declared or CONVENTIONAL_PATH_ARGS
    shortened = dict(args)
    # A url's ``user:password@`` never belongs on screen: removed from the display copy,
    # it leaves the row, the approval card and the stored transcript. The call itself still
    # carries it.
    url = shortened.get("url")
    if isinstance(url, str) and "@" in url:
        shortened["url"] = _without_userinfo(url)
    for key in keys:
        val = shortened.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        val = val.strip()
        if os.path.isabs(val):
            shortened[key] = _relpath(val)
    return shortened


# ---------------------------------------------------------------------------
# Status labels from the tool's name alone: the verb becomes a gerund ("Reading…"), the
# rest the object. A ``tool_caps(label=…)`` template takes precedence.
# ---------------------------------------------------------------------------

# A vocabulary of verbs, not a registry of tools: a new tool needs no entry here, and
# an unknown verb still humanises sensibly (see ``_humanize_tool_name``).
_VERBS: frozenset[str] = frozenset({
    "read", "write", "append", "delete", "remove", "list", "find", "search",
    "grep", "replace", "apply", "get", "set", "run", "compile", "execute",
    "check", "format", "add", "update", "register", "unregister", "submit",
    "inspect", "compare", "cancel", "stop", "diff", "aggregate", "scaffold",
    "init", "initialize", "promote", "parse", "reset", "install", "show",
    "import", "export", "query", "build", "probe", "configure", "evaluate",
    "summarize", "clear", "create", "store", "fetch", "analyze", "retrieve",
})

# Irregular / spelling-sensitive gerunds that the generic rules below would get
# wrong. Still verb-keyed, not tool-keyed.
_GERUND_OVERRIDES: dict[str, str] = {
    "set": "Setting",
    "get": "Getting",
    "run": "Running",
    "submit": "Submitting",
    "stop": "Stopping",
}

_VOWELS = frozenset("aeiou")


def _gerund(verb: str) -> str:
    """Return the capitalised ``-ing`` form of an English verb."""
    override = _GERUND_OVERRIDES.get(verb)
    if override:
        return override
    v = verb.lower()
    if v.endswith("ie"):
        base = v[:-2] + "ying"
    elif v.endswith("e") and not v.endswith("ee"):
        base = v[:-1] + "ing"
    elif (
        len(v) >= 3
        and v[-1] not in _VOWELS
        and v[-1] not in "wxy"
        and v[-2] in _VOWELS
        and v[-3] not in _VOWELS
    ):
        # short consonant-vowel-consonant → double the final consonant
        base = v + v[-1] + "ing"
    else:
        base = v + "ing"
    return base[:1].upper() + base[1:]


def _humanize_tool_name(name: str) -> str:
    """Turn a snake_case tool name into a readable gerund status phrase.

    ``list_directory`` → "Listing directory"; ``salloc_submit`` →
    "Submitting salloc"; ``slurm_partitions`` (no verb) → "Slurm
    partitions". Purely name-derived — no per-tool table.
    """
    if not name:
        return "Performing tool.."
    tokens = [t for t in name.split("_") if t]
    if not tokens:
        return "Performing tool.."
    verb_idx = next((i for i, t in enumerate(tokens) if t in _VERBS), None)
    if verb_idx is not None:
        rest = tokens[:verb_idx] + tokens[verb_idx + 1:]
        phrase = _gerund(tokens[verb_idx])
        if rest:
            phrase += " " + " ".join(rest)
        return phrase
    # No recognised verb: plain humanisation (capitalise the first token).
    return " ".join(tokens)[:1].upper() + " ".join(tokens)[1:]


def tool_status_message(name: str, args: dict[str, Any]) -> str:
    return _humanize_tool_name(name)


# Argument keys that carry a runnable command / code body, in priority order.
COMMAND_KEYS = ("command", "cmd", "script", "code")
# The *object* of a call, in priority order: what it acts on or asks about. The model's
# description says what the call is for; this says what it is for *on* (the verdict, the
# url, the job id). Argument names, never tool identities, so a new tool needs no entry.
#
# `op` comes last: it selects an action, which the description already carries, so a
# row falls back to it only when the call names no object (`system`, `date_op`).
_OBJECT_KEYS = (
    "verdict", "query", "expression", "equation",
    "symbol", "name", "job_id", "job_key", "target", "packages",
    "key_path", "partition", "title", "steps", "role", "scope", "op",
)
# A pair of arguments that names one thing between them. Checked before the single keys,
# since either half alone is the wrong answer: `repo` without its owner does not say
# which repository, and that is the whole of what a GitHub row is for.
_OBJECT_PAIRS = (("owner", "repo"),)
# Upper bound on a preview: the row's elastic slot is the description, and this one sits
# beside it. Long enough for a url with a path or a short expression, short enough that
# it cannot become the row.
_PREVIEW_LIMIT = 48
# Result-list key → its singular, for the row count of a search that reports hits.
_SEARCH_RESULT_KEYS = {
    "matches": "match",
    "references": "reference",
    "definitions": "definition",
}


def _text(args: dict[str, Any], key: str) -> str:
    """``args[key]`` stripped when it is a non-blank string, else ""."""
    val = args.get(key)
    return val.strip() if isinstance(val, str) else ""


def _first_path(args: dict[str, Any]) -> str:
    return next((v for v in (_text(args, k) for k in CONVENTIONAL_PATH_ARGS) if v), "")


def _preview_command(args: dict[str, Any]) -> str:
    """A runnable command or code body: its first non-empty line, up to 80 characters."""
    for key in COMMAND_KEYS:
        val = _text(args, key)
        if val:
            return next(ln for ln in val.splitlines() if ln.strip()).strip()[:80]
    return ""


def _preview_url(args: dict[str, Any]) -> str:
    """Host and path without the scheme: the host alone does not say what for."""
    url = _text(args, "url")
    return _url_preview(url) if url else ""


def _preview_pair(args: dict[str, Any]) -> str:
    """Two arguments naming one thing (owner/repo), with the path inside it, clipped
    from the left: the end names the thing."""
    for left, right in _OBJECT_PAIRS:
        a, b = _text(args, left), _text(args, right)
        if a and b:
            whole = f"{a}/{b}"
            path = _first_path(args)
            if path:
                whole += "/" + path.lstrip("/")
            return whole if len(whole) <= _PREVIEW_LIMIT else "…" + whole[-(_PREVIEW_LIMIT - 1):]
    return ""


def _preview_path(args: dict[str, Any]) -> str:
    """An absolute path as its file name (as in ``shorten_display_args``); a relative one,
    already short on a remote-fetch tool, as written."""
    path = _first_path(args)
    if not path:
        return ""
    return _relpath(path) if os.path.isabs(path) else _clip(path, _PREVIEW_LIMIT)


def _preview_value(key: str, val: Any) -> str:
    """One object argument. A list is joined while it fits and counted once it does not,
    the count named after the argument."""
    if isinstance(val, str) and val.strip():
        return _clip(" ".join(val.split()), _PREVIEW_LIMIT)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return str(val)
    if isinstance(val, (list, tuple)):
        parts = [" ".join(str(v).split()) for v in val if str(v).strip()]
        if not parts:
            return ""
        joined = " · ".join(parts)
        if len(joined) <= _PREVIEW_LIMIT:
            return joined
        return f"{len(parts)} {key}" if len(parts) > 1 else _clip(parts[0], _PREVIEW_LIMIT)
    return ""


def _preview_object(args: dict[str, Any]) -> str:
    """What the call acts on, else the action it selects (see _OBJECT_KEYS)."""
    return next((v for v in (_preview_value(k, args.get(k)) for k in _OBJECT_KEYS) if v), "")


def tool_arg_preview(name: str, args: dict[str, Any]) -> str:
    """The salient argument of a call, shown beside its description, or "".

    Tried in order: command, search pattern, url, owner/repo, file, object keys, ``op``.
    """
    if not isinstance(args, dict):
        return ""
    for preview in (_preview_command, lambda a: _text(a, "pattern")[:80], _preview_url,
                    _preview_pair, _preview_path, _preview_object):
        found = preview(args)
        if found:
            return found
    return ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _without_userinfo(url: str) -> str:
    """*url* with any ``user:password@`` removed, scheme and query intact.

    For the label and the approval card, which keep the whole url otherwise: a consent
    prompt asks about a precise call, and only the credential is never part of it.
    """
    head, sep, rest = url.partition("://")
    authority = (rest if sep else head).split("/", 1)[0]
    if "@" not in authority:
        return url
    cleaned = authority[authority.rindex("@") + 1:]
    return url.replace(authority, cleaned, 1)


def _bare_authority(url: str) -> str:
    """*url* without ``user:password@`` or ``?query``, for the fallback that echoes it.

    An unparseable url is where a credential would survive onto the row.
    """
    text = url.split("?", 1)[0].split("#", 1)[0]
    head = text.split("/", 1)[0]
    if "@" in head:
        text = text[text.index("@") + 1:]
    return text


def _url_preview(url: str) -> str:
    """``host/path`` of *url*, clipped — the scheme and the query are not the point.

    A path is kept because it is what distinguishes one call from the next, and clipped
    from the *left* of its tail rather than the right when it is long: the end of a path
    names the thing, the middle is navigation.
    """
    try:
        parsed = urlparse(url)
        # `hostname`, never `netloc`: a url may carry `user:password@`, and a row is
        # read over a shoulder and saved into a transcript. The port is added back
        # because it is what tells two local services apart.
        host, path = parsed.hostname or "", parsed.path or ""
        if host and parsed.port:
            host = f"{host}:{parsed.port}"
    except Exception:
        logger.debug("_url_preview: a best-effort step failed and was skipped", exc_info=True)
        return _clip(_bare_authority(url), _PREVIEW_LIMIT)
    if not host:
        return _clip(_bare_authority(url), _PREVIEW_LIMIT)
    whole = host + path.rstrip("/")
    if len(whole) <= _PREVIEW_LIMIT:
        return whole
    room = _PREVIEW_LIMIT - len(host) - 2
    tail = path.rstrip("/")
    return host + "/…" + tail[-room:] if room > 4 else _clip(host, _PREVIEW_LIMIT)


# Upper bound on the model's per-call description. The 15-word limit is asked of the
# model, not enforced here: a sentence cut mid-word reads worse than a long one, and
# this exists only so a model that ignores the limit entirely cannot push the rest of
# the row off screen (the stylesheet ellipsises what is still too long for the window).
_DOING_LIMIT = 120


def clip_doing(doing: Any) -> str:
    """The model's description of a call, as one line fit for a row.

    First line only: the row is a single line, and a description that arrives with a
    newline in it would otherwise take the layout with it.
    """
    if not isinstance(doing, str) or not doing.strip():
        return ""
    first = next((ln for ln in doing.splitlines() if ln.strip()), "").strip()
    return first if len(first) <= _DOING_LIMIT else first[:_DOING_LIMIT - 1] + "…"


def dedup_row_detail(shown: str, detail: str) -> str:
    """*detail*, unless the row's own text *shown* already says it (on word boundaries).

    Kept when *shown* is empty or the match is uncertain.
    """
    if not detail or not shown:
        return detail
    if re.search(r"\b" + re.escape(detail) + r"\b", shown, re.IGNORECASE):
        return ""
    return detail


# Short, human labels for a blocked-by-policy row (keyed on the violation's
# ``policy_stage``). Keeps the row readable instead of a cropped JSON error.
_POLICY_STAGE_LABELS = {
    "approval":       "needs approval",
    "write_policy":   "read the file first",
    "state_guard":    "validate current edits first",
    "external_fetch": "gather local context first",
    "cluster_submit": "validate locally first",
}


# Upper bound on the error body shipped to the UI. The row summary is a clipped
# one-liner; this is the full text the user expands to read, so it must be
# generous — but still bounded, since a failing tool can return a huge payload.
_ERROR_DETAIL_LIMIT = 4000


def error_detail(result: str) -> str:
    """The full error text of a failed result (with any policy reason), or "".

    For the expandable panel under the row; the ``hint`` is meant for the model and is
    left out.
    """
    if not isinstance(result, str) or not result.strip():
        return ""

    text = result.strip()
    payload = None
    if text.startswith("{"):
        try:
            payload, _ = json.JSONDecoder().raw_decode(text)
        except (ValueError, TypeError):
            payload = None
        if not isinstance(payload, dict):
            payload = None

    if isinstance(payload, dict):
        parts = []
        err = payload.get("error") or payload.get("message") or payload.get("reason")
        if err:
            parts.append(str(err).strip())
        stage = payload.get("policy_stage")
        if stage and not err:
            parts.append(f"blocked by policy ({stage})")
        detail = "\n\n".join(p for p in parts if p)
    else:
        detail = text

    detail = detail.strip()
    if len(detail) > _ERROR_DETAIL_LIMIT:
        detail = detail[:_ERROR_DETAIL_LIMIT] + "\n… (truncated)"
    return detail


def summarize_tool_result(name: str, result: str, tool_caps: dict[str, ToolCaps] | None = None) -> tuple[bool, str]:
    """Return ``(ok, summary)`` for a finished tool call.

    Tolerant of non-JSON results (returns ``(True, "")``). A policy block renders a
    short "⛔ blocked · <reason>"; other errors carry the first line of the error
    message; search/read tools carry a count.
    """
    if not isinstance(result, str) or not result.strip():
        return True, ""

    payload = None
    text = result.strip()
    if text.startswith("{"):
        # Parse only the LEADING JSON object: the client appends advisory text
        # (AUTO_VALIDATION, MORE_CONTENT, OUTLINE) after the payload, and a full
        # json.loads on the combined string fails into the plain-text heuristic below —
        # where a validator's embedded "status": "error" flips a successful edit to a
        # failed row. raw_decode ignores the trailing text.
        try:
            payload, _ = json.JSONDecoder().raw_decode(text)
        except (ValueError, TypeError):
            payload = None
        if not isinstance(payload, dict):
            payload = None

    if isinstance(payload, dict):
        # A policy precondition blocked the call — the tool never ran. Render a short
        # explicit reason, not the raw JSON cropped mid-sentence, so the row reads as
        # "blocked by policy" rather than as a genuine failure. Covers status="error"
        # (approval/write_policy/…) and status="blocked" (state_guard).
        stage = payload.get("policy_stage")
        if stage:
            return False, f"⛔ blocked · {_POLICY_STAGE_LABELS.get(stage, stage)}"
        if payload.get("status") == "error":
            err = str(payload.get("error") or "failed")
            return False, err.splitlines()[0].strip()[:100]
        if has_cap(name, SEARCH_WITH_PATH, tool_caps):
            # Each search names its own result list; counting only "matches" left the
            # row of every tool that carries this capability blank.
            for plural, singular in _SEARCH_RESULT_KEYS.items():
                hits = payload.get(plural)
                if isinstance(hits, list):
                    n = len(hits)
                    return True, f"{n} {singular if n == 1 else plural}"
        if has_cap(name, READ, tool_caps):
            # Prefer the exact line range read (read_file_lines returns the actual
            # start/end) so the activity row says "lines 207-211" instead of a bare
            # "2 lines" — informative and de-duplicates otherwise-identical rows.
            s, e = payload.get("start_line"), payload.get("end_line")
            if isinstance(s, int) and isinstance(e, int) and e >= s:
                return True, f"line {s}" if s == e else f"lines {s}-{e}"
            content = payload.get("content")
            if isinstance(content, str):
                n = content.count("\n") + 1 if content else 0
                return True, f"{n} line{'s' if n != 1 else ''}"

        # Structured payload parsed, status not error/block: the tool succeeded. Return
        # here so appended advisory text never reaches the plain-text heuristic below,
        # whose nested "status": "error" would flip this row to failed.
        return True, ""

    # Plain-text error heuristic (only for results that are not a JSON payload).
    low = text.lower()
    if low.startswith("error") or '"status": "error"' in low:
        return False, text.splitlines()[0].strip()[:100]
    return True, ""
