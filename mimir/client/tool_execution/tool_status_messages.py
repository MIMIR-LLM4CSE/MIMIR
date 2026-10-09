"""
Status message formatting for MCP agent tool calls.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from ..context.capabilities import READ, SEARCH_WITH_PATH, has_cap

def _relpath(path: str) -> str:
    """The file name alone — what a tool-activity row should show.

    Tools now take and report absolute paths (see ``server_files._require_abs``),
    which is right for the model and unreadable for a person: a row reading
    ``Reading file: /long/absolute/path/.../mimir/client/foo.py`` buries the one
    token that matters. The row answers "what is it touching right now", and the
    file name answers that.

    Deliberately NOT used for approval prompts. When the user is being asked to
    authorise access *outside* the workspace, the exact location is the decision —
    those carry the absolute path (``oow_path`` on the card, an explicit line in
    the CLI prompt). Readability wins in the activity log; precision wins in a
    consent prompt.
    """
    if not path:
        return path
    return os.path.basename(path.rstrip("/")) or path


def shorten_display_args(name: str, args: dict, tool_caps=None) -> dict:
    """*args* with path arguments reduced to their file name, for display.

    Capability-driven first: the ``path`` arg-role names which arguments are paths,
    so this needs no tool-name list and covers new tools automatically.

    File tools do not all declare that role, though — the read tool does and every
    write, edit and delete tool does not — so a declared role alone left exactly the
    rows the user reads most showing an absolute path: "Editing file:" carried the
    whole thing while "Reading file:" carried the name. The generic argument-name
    fallback is the same convention ``file_preview.build_preview_diffs`` and
    ``guardrails.observations`` already apply for the same reason, and these are
    argument names, never tool identities.

    Only values that are absolute are shortened, since that is the whole of the problem:
    a tool that names a workspace file is given an absolute path (``_require_abs``), so
    every path this exists for is one. An argument named ``path`` that holds a path
    relative to somewhere else — a repository path on a remote-fetch tool — is already
    short, and is the one the row means: reduced to its basename, the GitHub row said
    ``ci.yml`` where the call was for ``.github/workflows/ci.yml``. That holds whether or
    not the tool declared the role, which is why the rule is one rule.
    """
    if not isinstance(args, dict):
        return args
    from ..context.capabilities import arg_role
    declared = arg_role(name, "path", tool_caps) or ()
    keys = declared or _PATH_KEYS
    shortened = dict(args)
    # A url's `user:password@` never belongs on screen. It is not part of any decision
    # — a consent prompt asks about the host and the path — and the label it lands in
    # is the row's tooltip, the approval card's header and a line of the stored
    # transcript. This removes it from all four at once, which is the whole reason the
    # display copy of the arguments exists.
    #
    # It does NOT make the credential confidential: the model wrote it into the call,
    # and the call is in the conversation history either way. What it stops is the
    # incidental copy — on screen, and in the transcript the user shares.
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
# Generic name → status label, with NO hardcoded tool-name lists. A label is derived
# from the tool's *name* alone: an action verb becomes its gerund ("Reading…") and the
# remaining tokens become the object. The salient argument is surfaced separately by
# ``tool_arg_preview``. Servers wanting exact wording declare a ``tool_caps(label=…)``
# template, which ``label_for`` renders ahead of this fallback.
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


def tool_status_message(name: str, args: dict) -> str:
    return _humanize_tool_name(name)


# Argument keys that carry a runnable command / code body, in priority order.
_COMMAND_KEYS = ("command", "cmd", "script", "code")
_PATH_KEYS = ("path", "filepath", "file")
# The *object* of a call, in priority order: the thing it is acting on or asking about.
# Argument names, never tool identities — the same convention `shorten_display_args`
# and `file_preview` already follow, and the reason a new tool needs no entry anywhere.
#
# These used to reach the row through each server's label template ("Slurm cancel
# {job_id}", "Searching modules: {query}", "Verdict: {verdict}"), which the row no
# longer shows. The description says what the call is for; this says what it is for
# *on*, and losing it took the verdict, the url, the job id and the queried module name
# off the screen with it.
#
# Identifiers come before `op`, which is last on purpose: an op selects an action, which
# is the half the model's own description already carries, so it is what a row falls back
# to when the call names no object at all (`system`, `date_op`).
_OBJECT_KEYS = (
    "verdict", "query", "expression", "equation",
    "symbol", "name", "job_id", "job_key", "target", "packages",
    "key_path", "partition", "title", "role", "scope", "op",
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


def tool_arg_preview(name: str, args: dict) -> str:
    """The salient argument of a call, as the row shows it beside the description.

    What the call is *on*, where the description says what it is *for*. Keyed on
    argument names in priority order — a command or code body, a search pattern, a url,
    a file, then the object keys and finally ``op`` (see :data:`_OBJECT_KEYS`) — so any
    tool is covered and a new one needs no entry. Returns "" when the call names nothing
    worth showing (``{"max_depth": 2, "use_cache": True}``).

    A command keeps its first line up to 80 characters, since on a collapsed failed row
    it is the only trace of what ran; everything else is held to
    :data:`_PREVIEW_LIMIT`, which is what fits beside a description without competing
    with it. The stylesheet truncates whatever is still too wide for the pane.
    """
    if not isinstance(args, dict):
        return ""

    # Runnable command / code body → first non-empty line, clipped.
    for key in _COMMAND_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            first = next((ln for ln in val.splitlines() if ln.strip()), "").strip()
            return first[:80]

    # Search pattern.
    pattern = args.get("pattern")
    if isinstance(pattern, str) and pattern.strip():
        return pattern.strip()[:80]

    # Web URL → host and path, without the scheme. The host alone answered "is it
    # reaching the network" but not "for what", and a row whose whole point is the
    # outbound call said `api.github.com` for every one of them.
    url = args.get("url")
    if isinstance(url, str) and url.strip():
        return _url_preview(url.strip())

    # Two arguments that name one thing between them — an owner and a repository (see
    # _OBJECT_PAIRS). Before the path, and it takes the path with it: a remote file is
    # identified by the repository it is in, and `ci.yml` alone does not say which one.
    # That whole string is what a GitHub row is for.
    for left, right in _OBJECT_PAIRS:
        a, b = args.get(left), args.get(right)
        if isinstance(a, str) and a.strip() and isinstance(b, str) and b.strip():
            whole = f"{a.strip()}/{b.strip()}"
            for key in _PATH_KEYS:
                val = args.get(key)
                if isinstance(val, str) and val.strip():
                    whole += "/" + val.strip().lstrip("/")
                    break
            # Clipped from the left of the tail, like a url: the end names the thing.
            return whole if len(whole) <= _PREVIEW_LIMIT else "…" + whole[-(_PREVIEW_LIMIT - 1):]

    # A file this call names. Absolute → its file name, the same rule and the same
    # reason as `shorten_display_args`; relative → as written, because a repository
    # path on a remote-fetch tool is already short and its leading segments are what
    # identify it. Applied here and not only there because `policy.gates` previews the
    # raw arguments.
    for key in _PATH_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            val = val.strip()
            return _relpath(val) if os.path.isabs(val) else _clip(val, _PREVIEW_LIMIT)

    # What the call is acting on, else the action it selects (see _OBJECT_KEYS).
    for key in _OBJECT_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return _clip(" ".join(val.split()), _PREVIEW_LIMIT)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return str(val)
        # A list names several things at once — the packages of an install, the states
        # of a query. Joined rather than counted: "numpy scipy" is the row, "2 items"
        # is a row that has to be expanded to say anything.
        if isinstance(val, (list, tuple)) and val:
            joined = " ".join(str(v).strip() for v in val if str(v).strip())
            if joined:
                return _clip(joined, _PREVIEW_LIMIT)

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
    """*url* with any ``user:password@`` and any ``?query`` removed, for the fallback.

    The parsed path above never carries either — ``hostname`` drops the userinfo and
    ``path`` stops before the query. The fallback echoes the value as written, though,
    and a url the parser cannot read is exactly where a credential survives: a
    schemeless ``user:secret@host/x`` parses to no host at all, and the secret went
    onto the row and into the stored transcript with it. A query is dropped on the
    same grounds — an api key is usually in one — and it is not what identifies the
    call either.
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
        from urllib.parse import urlparse
        parsed = urlparse(url)
        # `hostname`, never `netloc`: a url may carry `user:password@`, and a row is
        # read over a shoulder and saved into a transcript. The port is added back
        # because it is what tells two local services apart.
        host, path = parsed.hostname or "", parsed.path or ""
        if host and parsed.port:
            host = f"{host}:{parsed.port}"
    except Exception:
        return _clip(_bare_authority(url), _PREVIEW_LIMIT)
    if not host:
        return _clip(_bare_authority(url), _PREVIEW_LIMIT)
    whole = host + path.rstrip("/")
    if len(whole) <= _PREVIEW_LIMIT:
        return whole
    room = _PREVIEW_LIMIT - len(host) - 2
    tail = path.rstrip("/")
    return host + "/…" + tail[-room:] if room > 4 else _clip(host, _PREVIEW_LIMIT)


# The argument the model's per-call description arrives in. Named here, with the helper
# that renders it, so the schema builder that adds the parameter and the dispatcher that
# strips it cannot disagree about its spelling.
DOING_ARG = "doing"

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
    """Drop a row *detail* that just repeats what the row already says.

    *shown* is the text the row puts beside it — the model's own description of the
    call. A description that already names the file or the url makes the preview pure
    duplication; it stays whenever it adds something, which is most of the time, since
    a description says what the call is for and this says what it is for *on*.

    Empty *shown* (no description was written) keeps the detail unconditionally: it is
    then the only thing on the row besides its family.

    Matched on word boundaries, not as a substring. The previews are now short words —
    a verdict is "pass", an op is "now" or "info" — and a substring test loses every one
    of them to an ordinary sentence: "pass" inside "recording the passing run", "now"
    inside "knowing the time", "info" inside "informing the user". The verdict going
    missing was the whole reason these previews came back to the row.

    Fails towards keeping: a detail whose edges are not word characters (a trailing
    slash, a closing bracket) simply does not match, and a repeated preview reads better
    than a lost one.
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
    """Return the FULL error text of a failed tool result, for the expandable panel.

    ``summarize_tool_result`` deliberately clips its summary to a single 100-char
    line so the activity row stays compact; that clipped line is useless on its own
    for diagnosing a failure. This returns the untruncated message (plus any policy
    reason, when the payload carries one) so the UI can show it in a full-width
    panel under the row. The payload's ``hint`` is guidance aimed at the model, not
    at the user, and is deliberately left out.

    Returns "" when no error text can be extracted (the caller then falls back to
    the summary).
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


def summarize_tool_result(name: str, result: str, tool_caps=None) -> tuple[bool, str]:
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
