from __future__ import annotations

import json
import os
import re
import shlex
from typing import Callable
from urllib.parse import urlparse

from ... import human_pause
from ...config.settings import APPROVAL_MODES
from ...tools.capabilities import (
    IRREVERSIBLE,
    RECOVERABLE,
    ToolCaps,
    arg_role,
    reversibility_of,
    risk_note_of,
    scope_spec,
)
from .batch_review import BatchReview
from .path_grants import PathGrants


def _packages_scope_token(packages) -> str:
    """Stable scope token for a packages arg (str or list): sorted, space-joined."""
    if isinstance(packages, str):
        items = packages.split()
    elif isinstance(packages, (list, tuple)):
        items = [str(p).strip() for p in packages]
    else:
        items = []
    return " ".join(sorted(p for p in items if p))


def _first(arguments: dict, names: tuple[str, ...] | list[str]) -> str:
    """First non-empty value among the named args (stringified, stripped)."""
    for n in names:
        val = arguments.get(n)
        if val not in (None, ""):
            return str(val).strip()
    return ""


# --- Generic scope-token builders, keyed by the declared scope *kind* ----------
# Each builder maps (arguments, scope_arg_names) -> a stable token narrowing an
# "always" approval. They are keyed by a generic `kind` string a tool declares
# (the scope `kind`), never by tool identity — a new tool reuses a kind.

def _scope_command_prefix(arguments: dict, args: tuple[str, ...]) -> str:
    """First two argv tokens of a shell command, skipping leading `cd`/connectors.

    So `cd /work && python3 -c ...` and `python3 -c ...` share one scope.
    """
    command = _first(arguments, args or ("command",))
    first_line = command.splitlines()[0] if command else ""
    first_segment = first_line.split("|", 1)[0].strip()
    try:
        argv = shlex.split(first_segment)
    except ValueError:
        # Unclosed quote (e.g. a multiline `-c "..."`): strip from the first quote
        # onward so `python3 -c "..."` maps to the same scope regardless of body.
        pre_quote = re.split(r'["\']', first_segment)[0].strip()
        try:
            argv = shlex.split(pre_quote) if pre_quote else []
        except ValueError:
            return ""
    if not argv:
        return ""
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "cd":
            i += 2  # skip cd + its argument
        elif tok in ("&&", ";", "&"):
            i += 1
        else:
            break
    effective = argv[i:] if i < len(argv) else argv
    return " ".join(effective[:2])


def _scope_host(arguments: dict, args: tuple[str, ...]) -> str:
    """Hostname (netloc) of a URL argument."""
    url = _first(arguments, args or ("url",))
    return urlparse(url).netloc.lower()


def _scope_basename(arguments: dict, args: tuple[str, ...]) -> str:
    """Basename of the first present path-like argument."""
    target = _first(arguments, args)
    return os.path.basename(target) if target else ""


def _scope_packages(arguments: dict, args: tuple[str, ...]) -> str:
    """Sorted, space-joined package set."""
    key = (args or ("packages",))[0]
    return _packages_scope_token(arguments.get(key))


def _scope_lang_target(arguments: dict, args: tuple[str, ...]) -> str:
    """`language:target` for a code job (inline filename or existing filepath)."""
    names = list(args) if args else ["language", "filepath", "filename"]
    language = str(arguments.get(names[0], "")).strip().lower() if names else ""
    filepath = str(arguments.get("filepath", "")).strip()
    target = os.path.basename(filepath) if filepath else str(arguments.get("filename", "")).strip()
    if not (language or target):
        return ""
    return f"{language}:{target}"


# kind -> (builder, default label phrase). The phrase is the full leading noun shown
# in the "always" prompt (it carries its own determiner, since "these packages" and
# "this command family" don't share one). Generic strategies, reused across tools.
_SCOPE_KINDS: dict[str, tuple[Callable[[dict, tuple[str, ...]], str], str]] = {
    "command_prefix": (_scope_command_prefix, "this command family"),
    "host": (_scope_host, "this destination host"),
    "basename": (_scope_basename, "this target file"),
    "packages": (_scope_packages, "these packages"),
    "lang_target": (_scope_lang_target, "this code job"),
}


# --- Denial kinds -------------------------------------------------------------
# Every approval front-end already returns a human-readable note beside its verdict
# ("denied by user", "cancelled", …). Those notes are what the *user* would read, so
# they stay as they are; these tokens are what the *policy layer* branches on, so no
# caller ever has to match prose. A note we do not recognise degrades to a plain
# refusal rather than raising — an unknown note is still a refusal.

DENIAL_DENIED: str = "denied"                    # the user answered "no"
DENIAL_CANCELLED: str = "cancelled"              # the run was stopped while the card was open
DENIAL_BATCH_REJECT: str = "batch_reject"        # the pending batch review was rejected
DENIAL_INVALID_RESPONSE: str = "invalid_response"  # the prompt's retry budget ran out
DENIAL_PATH: str = "path_denied"                 # out-of-workspace access refused
DENIAL_AUTO_REPEAT: str = "auto_denied_repeat"   # refused so often we stopped asking
DENIAL_MODE: str = "mode_denied"                 # nobody to ask, and the mode does not allow it

# The request-``_meta`` key under which the client sends its approval mode with every
# tool call; the spawn server reads it to run a sub-agent in the same mode.
APPROVAL_MODE_META: str = "mimir/approval_mode"

# The note an unattended agent (a sub-agent) gives when its approval mode does not
# cover an action. Nobody refused it: there was nobody to ask.
MODE_DENIED_NOTE: str = "not allowed by the approval mode"

_DENIAL_KINDS: dict[str, str] = {
    "denied by user": DENIAL_DENIED,
    "denied": DENIAL_DENIED,
    "cancelled": DENIAL_CANCELLED,
    "denied because pending batch review was rejected": DENIAL_BATCH_REJECT,
    "denied after invalid approval responses": DENIAL_INVALID_RESPONSE,
    "path outside the workspace was not approved": DENIAL_PATH,
    "already refused; not asked again": DENIAL_AUTO_REPEAT,
    MODE_DENIED_NOTE: DENIAL_MODE,
}


def denial_kind(note: str | None) -> str:
    """Map an approval front-end's refusal note to a stable kind token."""
    return _DENIAL_KINDS.get((note or "").strip().lower(), DENIAL_DENIED)


class TrustedToolsView:
    """
    Lightweight dynamic view over trusted tool names.

    Not a mutable set: it reflects the ApprovalManager's state, so status and help
    code can read it without holding a copy.
    """

    def __init__(self, manager: "ApprovalManager"):
        self._manager = manager

    def __iter__(self):
        return iter(sorted(self._manager.trusted_tool_names()))

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str):
            return False
        return item in self._manager.trusted_tool_names()

    def __len__(self) -> int:
        return len(self._manager.trusted_tool_names())

    def __repr__(self) -> str:
        return repr(sorted(self._manager.trusted_tool_names()))


class ApprovalManager:
    def __init__(
        self,
        sensitive_tools: set[str] | None = None,
        fallback_tools: dict[str, tuple[str, ...]] | None = None,
        non_batch_tools: set[str] | None = None,
        tool_caps: dict[str, ToolCaps] | None = None,
        session_id: str | None = None,
    ):
        # Out-of-workspace paths the user let this conversation reach (see PathGrants:
        # per session, mirrored to the sidecar the servers read).
        self.paths = PathGrants(session_id)
        # Writes held for one review at the end of the turn (see BatchReview).
        self.batch = BatchReview()
        # Classification (sensitive / non-batch / fallback) plus scope-narrowing and
        # risk notes are seeded from the per-agent live registry after servers
        # connect, via MimirAgent.seed_classification_from_caps(). Constructed empty
        # by default because no classification happens before connect.
        self.sensitive_tools = set(sensitive_tools or ())
        self.fallback_tools = dict(fallback_tools or {})
        self.non_batch_tools = set(non_batch_tools or ())
        # Live tool-capability registry (agent.tool_caps). Drives scope narrowing,
        # the confirm/URL sensitivity gates, and risk notes — read so approval logic
        # never re-lists literal tool names.
        self.tool_caps: dict[str, ToolCaps] = dict(tool_caps or {})

        # Single source of truth for session approvals.
        # Examples:
        #   "*:bash_run"                  -> trust bash_run globally for session
        #   "bash:bash_run"              -> trust bash_run from server 'bash'
        #   "bash:bash_run:git status"   -> trust only this narrower scope
        self.approved_scopes: set[str] = set()

        # How much the user still answers by hand, for this session only (never persisted):
        #   manual   — ask for sensitive tools and out-of-workspace paths
        #   auto     — sensitive tools pass; leaving the workspace still asks
        #   auto_all — nothing asks
        # It removes the question, never the policy: denylists and the verification gates still refuse.
        self.approval_mode: str = "manual"

        # True when no one can answer a card (a sub-agent: its stdin is the MCP pipe).
        # What the mode does not cover is then refused without asking, and recorded in
        # ``mode_blocked`` as ``{"action", "needs"}`` so the caller can report it.
        self.unattended: bool = False
        self.mode_blocked: list[dict] = []

        # Whether writes are held for one review at the end of the turn.
        self.batch_mode: bool = True

    @property
    def session_id(self) -> str | None:
        """Which conversation's grants this manager holds."""
        return self.paths.session_id

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self.paths.session_id = value

    # ------------------------------------------------------------------
    # Basic metadata helpers
    # ------------------------------------------------------------------

    APPROVAL_MODES: tuple[str, ...] = APPROVAL_MODES

    def auto_tools(self) -> bool:
        """True when a sensitive tool no longer raises a card."""
        return self.approval_mode in ("auto", "auto_all")

    def auto_paths(self) -> bool:
        """True when reaching outside the workspace no longer raises a card."""
        return self.approval_mode == "auto_all"

    def block_for_mode(self, action: str, needs: str) -> None:
        """Record an action an unattended agent skipped because its mode is too low."""
        entry = {"action": action, "needs": needs}
        if entry not in self.mode_blocked:
            self.mode_blocked.append(entry)

    def is_sensitive(self, tool_name: str, arguments: dict) -> bool:
        # Confirm-gated tools (a `confirm_gate` arg-role) are dry-run/preview when the
        # gate arg is falsy and a real mutation when truthy — sensitivity tracks the
        # gate. The `not in sensitive_tools` guard means the gate can only *add*
        # sensitivity to an otherwise-unclassified tool, never downgrade a tool that
        # is independently declared SENSITIVE.
        if tool_name not in self.sensitive_tools:
            gate = arg_role(tool_name, "confirm_gate", self.tool_caps)
            if gate:
                return bool(arguments.get(gate[0], False))

        if tool_name in self.sensitive_tools:
            return True

        # A tool that narrows by URL host (e.g. an HTTP GET) is sensitive only when the
        # URL targets an authenticated/mutating endpoint. Keyed on the declared `host`
        # scope kind, not the tool name.
        spec = scope_spec(tool_name, self.tool_caps)
        if spec and spec.get("kind") == "host":
            url = _first(arguments, tuple(spec.get("args") or ("url",))).lower()
            parsed = urlparse(url)
            path_parts = set(parsed.path.strip("/").split("/"))
            # Specific patterns that unambiguously indicate authenticated/mutating endpoints.
            if any(t in url for t in ("/api/", "oauth", "bearer", "/authorize", "apikey")):
                return True
            # Segment-level check for words that are sensitive only as path components,
            # not as substrings of hostnames or parameter names.
            if path_parts & {"token", "auth", "login", "secret", "credential", "password"}:
                return True
            return False

        return False

    def describe_risk(self, tool_name: str) -> str:
        return risk_note_of(tool_name, self.tool_caps) or "performs a sensitive action"

    def describe_reversibility(self, tool_name: str) -> str:
        """One line on how far this action can be taken back, for the prompt."""
        level = reversibility_of(tool_name, self.tool_caps)
        if level == IRREVERSIBLE:
            return "IRREVERSIBLE — leaves this machine or spends real resources; nothing here can undo it"
        if level == RECOVERABLE:
            return "recoverable — undoable, but by hand: MIMIR keeps no snapshot of it"
        return "reversible — MIMIR snapshots it and can restore it this session"

    def fallback_suggestions(self, tool_name: str) -> tuple[str, ...]:
        return self.fallback_tools.get(tool_name, tuple())

    # ------------------------------------------------------------------
    # Public approval API
    # ------------------------------------------------------------------

    @staticmethod
    def _wildcard_tool_scope(tool_name: str) -> str:
        return f"*:{tool_name}"

    @staticmethod
    def _base_scope(tool_name: str, server_name: str) -> str:
        return f"{server_name}:{tool_name}"

    def trust_tool(self, tool_name: str, server_name: str | None = None) -> None:
        """
        Trust a tool for the current session.

        - server_name=None  -> trust the tool globally across servers:  *:tool
        - server_name=...   -> trust base scope for that server:         server:tool
        """
        scope = (
            self._wildcard_tool_scope(tool_name)
            if server_name is None
            else self._base_scope(tool_name, server_name)
        )
        self.approved_scopes.add(scope)

    def untrust_tool(self, tool_name: str, server_name: str | None = None) -> None:
        """Remove trust for *tool_name*, on *server_name* or (None) on every server, narrower scopes included."""
        if server_name is None:
            wildcard = self._wildcard_tool_scope(tool_name)
            self.approved_scopes.discard(wildcard)
            to_remove = []
            for scope in self.approved_scopes:
                parts = scope.split(":", 2)
                if scope.startswith("*:"):
                    current_tool = scope.split(":", 1)[1]
                elif len(parts) >= 2:
                    current_tool = parts[1]
                else:
                    continue
                if current_tool == tool_name:
                    to_remove.append(scope)
            for scope in to_remove:
                self.approved_scopes.discard(scope)
            return

        base = self._base_scope(tool_name, server_name)
        to_remove = [scope for scope in self.approved_scopes if scope == base or scope.startswith(base + ":")]
        for scope in to_remove:
            self.approved_scopes.discard(scope)

    def trusted_tool_names(self) -> set[str]:
        """
        Return the set of tool names that currently have any trust scope
        (global, base, or narrower).
        """
        names: set[str] = set()
        for scope in self.approved_scopes:
            if scope.startswith("*:"):
                names.add(scope.split(":", 1)[1])
                continue
            parts = scope.split(":", 2)
            if len(parts) >= 2:
                names.add(parts[1])
        return names

    def trusted_tools_view(self) -> TrustedToolsView:
        return TrustedToolsView(self)

    # ------------------------------------------------------------------
    # Approval scope helpers
    # ------------------------------------------------------------------

    def _scope_token(self, tool_name: str, arguments: dict) -> tuple[str, str]:
        """``(token, noun)`` from the tool's declared scope spec, or ``("", "")`` without one."""
        spec = scope_spec(tool_name, self.tool_caps)
        if not spec:
            return "", ""
        builder_noun = _SCOPE_KINDS.get(str(spec.get("kind") or ""))
        if not builder_noun:
            return "", ""
        builder, default_noun = builder_noun
        token = builder(arguments, tuple(spec.get("args") or ()))
        noun = str(spec.get("noun") or default_noun)
        return token, noun

    def _approval_scope(self, tool_name: str, server_name: str, arguments: dict) -> str:
        """The "always" scope of this call: server:tool, narrowed by a declared ``scope`` spec."""
        base = self._base_scope(tool_name, server_name)
        token, _ = self._scope_token(tool_name, arguments)
        return f"{base}:{token}" if token else base

    def _is_scope_approved(self, tool_name: str, server_name: str, scope: str) -> bool:
        """
        Approval resolution order:
        1. exact scope
        2. server:tool base scope
        3. global tool wildcard scope (*:tool)
        """
        if scope in self.approved_scopes:
            return True

        base = self._base_scope(tool_name, server_name)
        if base in self.approved_scopes:
            return True

        wildcard = self._wildcard_tool_scope(tool_name)
        if wildcard in self.approved_scopes:
            return True

        return False

    def _approval_scope_label(self, tool_name: str, server_name: str, arguments: dict) -> str:
        """What "always" will approve, in words: ``<phrase> (<token>)``, ``<phrase>``, or "this tool"."""
        token, phrase = self._scope_token(tool_name, arguments)
        if not phrase:
            return "this tool"
        return f"{phrase} ({token})" if token else phrase

    def render_prompt(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict,
    ) -> str:
        args_json = json.dumps(arguments, indent=2, ensure_ascii=False, default=str)
        always_scope = self._approval_scope_label(tool_name, server_name, arguments)
        return (
            f"\n🔑 Approval required\n"
            f"Server : {server_name}\n"
            f"Reason : {self.describe_risk(tool_name)}\n"
            f"Undo   : {self.describe_reversibility(tool_name)}\n"
            f"Scope  : [a]lways will approve {always_scope} for this session\n"
            f"Args   :\n{args_json}\n\n"
            f"Allow? [y]es / [n]o / [a]lways: "
        )

    def request(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict,
        max_attempts: int = 3,
        input_func: Callable[[str], str] | None = None,
        output_func: Callable[[str], None] | None = None,
    ) -> tuple[bool, str]:
        _ask = input if input_func is None else input_func
        output_func = print if output_func is None else output_func

        def ask(prompt: str) -> str:
            # Blocking on the user, not on the tool: excluded from the tool-call
            # timeout budget this approval sits inside (see human_pause).
            with human_pause.human_pause():
                return _ask(prompt)

        # Critical safety rule:
        # if there are pending batched file edits and we are about to run a
        # non-batch tool (execution, network, DB write, etc.), force review now.
        if tool_name in self.non_batch_tools and self.batch.has_pending:
            approved_batch = self.batch.flush(
                output_fn=output_func,
                input_fn=ask,
            )
            if not approved_batch:
                return False, "denied because pending batch review was rejected"

        scope = self._approval_scope(tool_name, server_name, arguments)
        if self._is_scope_approved(tool_name, server_name, scope):
            return True, "approved for this session"

        # In batch mode: auto-approve and queue for end-of-turn review,
        # EXCEPT for tools in non_batch_tools (e.g. code execution).
        if self.batch_mode and tool_name not in self.non_batch_tools:
            self.batch.queue(tool_name, server_name, arguments)
            return True, "queued for batch review"

        for attempt in range(max_attempts):
            choice = ask(self.render_prompt(tool_name, server_name, arguments)).strip().lower()
            if choice in {"y", "yes"}:
                return True, "approved once"
            if choice in {"a", "always"}:
                self.approved_scopes.add(scope)
                return True, "approved for this session"
            if choice in {"n", "no", ""}:
                return False, "denied by user"
            remaining = max_attempts - attempt - 1
            output_func(f"Please answer with y, n, or a. ({remaining} attempt(s) remaining)")

        return False, "denied after invalid approval responses"