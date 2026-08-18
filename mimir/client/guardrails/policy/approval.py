from __future__ import annotations

import difflib
import json
import os
import re
import shlex
from typing import Callable
from urllib.parse import urlparse

from ... import human_pause
from ...context.capabilities import (
    IRREVERSIBLE,
    RECOVERABLE,
    ToolCaps,
    arg_role,
    reversibility_of,
    risk_note_of,
    scope_spec,
)


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

_DENIAL_KINDS: dict[str, str] = {
    "denied by user": DENIAL_DENIED,
    "denied": DENIAL_DENIED,
    "cancelled": DENIAL_CANCELLED,
    "denied because pending batch review was rejected": DENIAL_BATCH_REJECT,
    "denied after invalid approval responses": DENIAL_INVALID_RESPONSE,
    "path outside the workspace was not approved": DENIAL_PATH,
    "already refused; not asked again": DENIAL_AUTO_REPEAT,
}


def denial_kind(note: str | None) -> str:
    """Map an approval front-end's refusal note to a stable kind token."""
    return _DENIAL_KINDS.get((note or "").strip().lower(), DENIAL_DENIED)


class TrustedToolsView:
    """
    Lightweight dynamic view over trusted tool names.

    This is intentionally *not* a mutable set. It reflects the ApprovalManager
    state dynamically so callers can pass it to status/help code without
    reintroducing legacy `approved_tools`.
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
    ):
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

        # Out-of-workspace access approved this session. ``_allowed_paths`` is the
        # full set (allow-once + always) mirrored to the sidecar file the servers
        # read per call; the ``path:<abspath>`` tokens in ``approved_scopes`` mark
        # the "always" grants that skip re-prompting. Reset on session change.
        self._allowed_paths: set[str] = set()

        # Batch-approval state
        self.batch_mode: bool = True
        self._pending_review: list[dict] = []
        # path → original content (None means file did not exist before the batch)
        self._file_snapshots: dict[str, str | None] = {}

    # ------------------------------------------------------------------
    # Basic metadata helpers
    # ------------------------------------------------------------------

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
        """One line on how far this action can be taken back, for the prompt.

        A binary "approval required" spent the same friction on a file the client can
        restore and on a Slurm submission burning allocation hours. Naming the level
        is what lets the user read the difference before answering — the prompt is the
        only place the distinction reaches them.
        """
        level = reversibility_of(tool_name, self.tool_caps)
        if level == IRREVERSIBLE:
            return "IRREVERSIBLE — leaves this machine or spends real resources; nothing here can undo it"
        if level == RECOVERABLE:
            return "recoverable — undoable, but by hand: MIMIR keeps no snapshot of it"
        return "reversible — MIMIR snapshots it and can restore it this session"

    def fallback_suggestions(self, tool_name: str) -> tuple[str, ...]:
        return self.fallback_tools.get(tool_name, tuple())

    # ------------------------------------------------------------------
    # Public approval API (new, explicit, non-legacy)
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
        """
        Remove trust for a tool.

        - server_name=None  -> remove wildcard trust and any narrower scopes
                               for that tool across all servers
        - server_name=...   -> remove base scope and any narrower scopes
                               for that server/tool
        """
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
        """Resolve (token, label-noun) from the tool's declared scope spec.

        The spec names the scope arg(s) and a generic derivation ``kind``; the kind
        selects a builder in ``_SCOPE_KINDS``. Returns ``("", "")`` when the tool
        declares no scope spec (or an unknown kind) — callers fall back to the coarse
        server:tool scope. No tool names appear here; everything is registry-driven.
        """
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
        """
        Return the session-approval scope for this tool call.

        Default behavior is coarse-grained (server + tool); tools that declare a
        ``scope`` spec narrow the "always" scope (e.g. to one command family / host /
        package set) so a single approval doesn't blanket every future call.
        """
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

    # ------------------------------------------------------------------
    # Out-of-workspace path approvals (session-scoped)
    # ------------------------------------------------------------------

    @staticmethod
    def _path_scope_token(abspath: str) -> str:
        """Scope token pinning an 'always' grant to one exact absolute path."""
        return f"path:{os.path.realpath(os.path.abspath(abspath))}"

    def is_path_approved(self, abspath: str) -> bool:
        """True if an 'always' grant this session covers *abspath*.

        Containment, not equality: the sidecar the servers read admits a granted
        path as a *root* (``resolve_path_in_root``'s ``extra_roots``), so once a
        directory is approved the server already allows everything under it and a
        prompt for a child could no longer deny anything.
        """
        real = os.path.realpath(os.path.abspath(abspath))
        for scope in self.approved_scopes:
            if not scope.startswith("path:"):
                continue
            granted = scope[len("path:"):]
            if real == granted or real.startswith(granted + os.sep):
                return True
        return False

    def grant_path(self, abspath: str, *, always: bool) -> None:
        """Record an approved out-of-workspace path and mirror it to the sidecar.

        Both allow-once and always add the path to the sidecar so the server honors
        this access; only ``always`` also records the scope token that skips future
        prompts for the same path this session.
        """
        real = os.path.realpath(os.path.abspath(abspath))
        self._allowed_paths.add(real)
        if always:
            self.approved_scopes.add(self._path_scope_token(real))
        self._flush_allowed_paths()

    def reset_allowed_paths(self) -> None:
        """Clear session path grants (on session change) and truncate the sidecar."""
        self._allowed_paths.clear()
        self.approved_scopes = {s for s in self.approved_scopes if not s.startswith("path:")}
        self._flush_allowed_paths()

    def _flush_allowed_paths(self) -> None:
        """Write the current allowed-paths set to the sidecar the servers read."""
        try:
            from ...config.constants import STATE_DIR
            os.makedirs(STATE_DIR, exist_ok=True)
            path = os.path.join(STATE_DIR, "approved_paths.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(sorted(self._allowed_paths), fh)
            os.replace(tmp, path)
        except OSError:
            pass  # best-effort; a failed write simply keeps the server strict

    def _approval_scope_label(self, tool_name: str, server_name: str, arguments: dict) -> str:
        """Human-readable description of what 'always' will approve.

        Driven by the same scope spec as ``_approval_scope``: ``<phrase> (<token>)``
        when the tool narrows its scope, falling back to the bare ``<phrase>`` (no
        usable token) or ``this tool`` (no scope spec). The phrase carries its own
        determiner (kind default or spec ``noun``). No tool names appear here.
        """
        token, phrase = self._scope_token(tool_name, arguments)
        if not phrase:
            return "this tool"
        return f"{phrase} ({token})" if token else phrase

    # ------------------------------------------------------------------
    # Batch-approval helpers
    # ------------------------------------------------------------------

    @property
    def has_pending_review(self) -> bool:
        return bool(self._pending_review)

    def record_snapshot(self, path: str) -> None:
        """Capture the current content of *path* before a write modifies it.

        Only the first snapshot per path is kept so subsequent edits within
        the same batch don't overwrite the original baseline.
        """
        if path in self._file_snapshots:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                self._file_snapshots[path] = fh.read()
        except FileNotFoundError:
            self._file_snapshots[path] = None  # file will be created by this batch
        except OSError:
            pass  # can't snapshot; skip gracefully

    def flush_pending_review(
        self,
        output_fn: Callable[[str], None] | None = None,
        input_fn: Callable[[str], str] | None = None,
    ) -> bool:
        """Show a combined diff of all batch edits and ask for a single approval.

        Returns True if the user accepted, False if they rejected (in which case
        all snapshotted files are restored to their pre-batch state).
        """
        output_fn = output_fn or print
        input_fn = input_fn or input

        if not self._pending_review:
            return True

        output_fn(
            f"\n📝 Batch review — {len(self._pending_review)} write operation(s) performed:"
        )

        # Per-operation summary
        for i, edit in enumerate(self._pending_review, 1):
            args_display = {
                k: v
                for k, v in edit["arguments"].items()
                if k not in ("new_text", "content")
            }
            new_val = edit["arguments"].get("new_text") or edit["arguments"].get("content", "")
            if new_val:
                preview = new_val[:120].replace("\n", "↵")
                args_display["new_text"] = preview + ("…" if len(new_val) > 120 else "")
            output_fn(
                f"\n  [{i}] {edit['tool_name']}  (server: {edit['server_name']})\n"
                + json.dumps(args_display, indent=4, ensure_ascii=False, default=str)
            )

        # Unified diffs for every snapshotted/modified file
        for path, original in self._file_snapshots.items():
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    current = fh.read()
            except OSError:
                current = ""

            before_lines = original.splitlines(keepends=True) if original is not None else []
            after_lines = current.splitlines(keepends=True)

            if before_lines == after_lines:
                continue

            rel = os.path.relpath(path)
            diff = "".join(
                difflib.unified_diff(
                    before_lines,
                    after_lines,
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                    n=3,
                )
            )
            output_fn(f"\n{diff}" if diff else f"\n(binary change: {rel})")

        # Single approval prompt
        choice = input_fn(
            "\nApply all changes? [y]es / [n]o (undo all): "
        ).strip().lower()
        approved = choice in {"y", "yes"}

        if not approved:
            for path, original in self._file_snapshots.items():
                try:
                    if original is None:
                        os.remove(path)
                    else:
                        with open(path, "w", encoding="utf-8") as fh:
                            fh.write(original)
                except OSError as exc:
                    output_fn(f"  ⚠️  Could not restore {path}: {exc}")
            output_fn("  ↩  Changes reverted.")
        else:
            output_fn("  ✔  Changes applied.")

        self._pending_review.clear()
        self._file_snapshots.clear()
        return approved

    def revert_last(self, output_fn: Callable[[str], None] | None = None) -> list[str]:
        """Restore all files that were snapshotted in the current batch.

        Returns a list of paths that were reverted. If a snapshot has
        ``None`` as the original (the file was newly created by the agent),
        the file is deleted.

        This is called by the ``/undo`` slash command.
        """
        output_fn = output_fn or print

        if not self._file_snapshots:
            output_fn("Nothing to undo — no file snapshots are available for this session.")
            return []

        reverted: list[str] = []
        for path, original in self._file_snapshots.items():
            try:
                if original is None:
                    if os.path.exists(path):
                        os.remove(path)
                        output_fn(f"  ↩  Deleted (newly created): {path}")
                    reverted.append(path)
                else:
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(original)
                    output_fn(f"  ↩  Restored: {path}")
                    reverted.append(path)
            except OSError as exc:
                output_fn(f"  ⚠️  Could not restore {path}: {exc}")

        self._pending_review.clear()
        self._file_snapshots.clear()
        return reverted

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

        def input_func(prompt: str) -> str:
            # Blocking on the user, not on the tool: excluded from the tool-call
            # timeout budget this approval sits inside (see human_pause).
            with human_pause.human_pause():
                return _ask(prompt)

        # Critical safety rule:
        # if there are pending batched file edits and we are about to run a
        # non-batch tool (execution, network, DB write, etc.), force review now.
        if tool_name in self.non_batch_tools and self.has_pending_review:
            approved_batch = self.flush_pending_review(
                output_fn=output_func,
                input_fn=input_func,
            )
            if not approved_batch:
                return False, "denied because pending batch review was rejected"

        scope = self._approval_scope(tool_name, server_name, arguments)
        if self._is_scope_approved(tool_name, server_name, scope):
            return True, "approved for this session"

        # In batch mode: auto-approve and queue for end-of-turn review,
        # EXCEPT for tools in non_batch_tools (e.g. code execution).
        if self.batch_mode and tool_name not in self.non_batch_tools:
            self._pending_review.append(
                {"tool_name": tool_name, "server_name": server_name, "arguments": arguments}
            )
            return True, "queued for batch review"

        for attempt in range(max_attempts):
            choice = input_func(self.render_prompt(tool_name, server_name, arguments)).strip().lower()
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