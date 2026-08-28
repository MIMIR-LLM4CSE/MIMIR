"""
Status message formatting for MCP agent tool calls.
"""

from __future__ import annotations

import json
import os

from ..context.capabilities import READ, SEARCH_WITH_PATH, has_cap

def _relpath(path: str) -> str:
    """The file name alone — what a tool-activity row should show.

    Tools now take and report absolute paths (see ``server_files._require_abs``),
    which is right for the model and unreadable for a person: a row reading
    ``Reading file: /shared/data1/Projects/.../mimir/client/foo.py`` buries the one
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
    """*args* with declared path arguments reduced to their file name, for display.

    Capability-driven: the ``path`` arg-role names which arguments are paths, so
    this needs no tool-name list and covers new tools automatically. Returns the
    original dict untouched when the tool declares no path role.
    """
    if not isinstance(args, dict):
        return args
    from ..context.capabilities import arg_role
    keys = arg_role(name, "path", tool_caps) or ()
    if not keys:
        return args
    shortened = dict(args)
    for key in keys:
        val = shortened.get(key)
        if isinstance(val, str) and val.strip():
            shortened[key] = _relpath(val.strip())
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
# Result-list key → its singular, for the row count of a search that reports hits.
_SEARCH_RESULT_KEYS = {
    "matches": "match",
    "references": "reference",
    "definitions": "definition",
}


def tool_arg_preview(name: str, args: dict) -> str:
    """Return a short, human-meaningful preview of a tool's key argument.

    Generic and key-based so it works for any tool (including ones not in the
    hardcoded status map): commands/code show their first line, search tools
    show the pattern, web fetches show the host, file tools show the basename.
    Returns "" when there is nothing useful to show.
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

    # Web URL → hostname only.
    url = args.get("url")
    if isinstance(url, str) and url.strip():
        try:
            from urllib.parse import urlparse
            host = urlparse(url).hostname
            if host:
                return host
        except Exception:
            pass
        return url.strip()[:80]

    # Fall back to a file basename.
    for key in _PATH_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return _relpath(val.strip())

    return ""


def dedup_row_detail(label: str, detail: str) -> str:
    """Drop a row *detail* that just repeats what the *label* already shows.

    A server label template like "Reading file: {path}" already names the target,
    so the basename detail is pure duplication. The detail stays when it adds
    something new (e.g. the command line under a generic "Running shell command").
    """
    if detail and label and detail.lower() in label.lower():
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
