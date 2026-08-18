"""Server-side capability declaration helper.

A first-party MCP server uses :func:`tool_caps` to declare what each of its tools
*means* so the client classifies it correctly with zero client-side edits::

    from _shared.capabilities import tool_caps, READ, SEARCH

    @mcp.tool(**tool_caps(caps=[READ], path_args=["path"], label="Reading file: {path}"))
    def read_file_lines(path: str, start_line: int = 1, end_line: int = 200) -> str:
        ...

The helper emits the kwargs FastMCP's ``@mcp.tool`` accepts:

* ``meta={"mimir": {...}}`` — our authoritative descriptor, read by the client's
  :func:`mimir.client.context.capabilities.infer_tool_caps` (Layer 1).
* ``annotations=ToolAnnotations(...)`` — standard ``readOnlyHint``/``destructiveHint``
  so foreign/older clients still get a coarse classification (Layer 2).

The descriptor schema mirrors the client contract exactly:

    {
      "capabilities": [...],            # semantic flags (vocab below)
      "arg_roles": {"path": [...], ...},
      "approval": {"sensitive": bool, "non_batch": bool, "fallbacks": [...]},
      "label": "Reading file: {path}",  # optional status template
      "preview": {"kind": "...", "args": [...]},  # optional pre-write diff shape
    }

The capability vocabulary is duplicated here as plain strings (rather than importing
the client) to keep the server layer free of client dependencies; the two lists must
stay in sync — the client's parity test guards classification.
"""

from __future__ import annotations

from typing import Any, Iterable

# --- capability vocabulary (mirror of context/capabilities.py) -------------
# Grouped by behavioral family; values and membership must stay identical to the
# client module (guarded by test_capabilities.test_vocab_in_sync).

# Discovery: reads, searches, inspections the policy tracks as evidence
READ = "read"
CACHEABLE = "cacheable"
INSPECT_DIR = "inspect_dir"
CHECK_EXISTENCE = "check_existence"
CODE_NAV = "code_nav"                     # code navigation: symbol → definition, or file → symbol outline
ENV_DISCOVERY = "env_discovery"           # enumerates available runtime environments (conda/venv interpreters)
SEARCH = "search"
SEARCH_WITH_PATH = "search_with_path"
CANDIDATE_SEARCH = "candidate_search"

# Mutation (delete and whole-file overwrite are distinct policy categories;
# finer details are arg-roles, not flags). is_write / clears_edit_loop are
# *derived* client-side, not declared — see context/capabilities.py.
EDIT = "edit"
CONTENT_WRITE = "content_write"
OVERWRITE = "overwrite"                  # replaces an existing file's entire content (subset of content_write)
REMOVE = "remove"                        # deletes a file
REPLACEMENT_TRACK = "replacement_track"

# Validates code (syntax/lint/typecheck/test). The first-party stack validates through
# the bash server, but the flag stays declarable so an extension-pack server can ship
# its own validator (a success marks the target file validated + clears the edit streak).
VALIDATE = "validate"

# Task planning (the ordered-steps checklist is identified by the `plan_steps` arg-role)
TASK_PLANNING = "task_planning"          # records a task plan (checklist and/or prose rationale)

# Approval & mode policy
SENSITIVE = "sensitive"
NON_BATCH = "non_batch"
# Both names predate ask mode; they gate every read-only mode (plan and ask), not plan alone.
PLAN_BLOCKED = "plan_blocked"
PLAN_READONLY = "plan_readonly"          # dual-use exec tool kept available in a read-only mode, but only
                                         # for read-only/discovery invocations (client gates its exec use)
CODE_EXEC = "code_exec"                  # runs a program / shell command / code payload. Marks the tool
                                         # for guards that must inspect *what* is executed (e.g. the proxy
                                         # exec guard); it does NOT by itself imply a block.

# Reach & cost
EXTERNAL_FETCH = "external_fetch"         # reaches outside the workspace (e.g. GitHub API)
CLUSTER_SUBMIT = "cluster_submit"        # expensive cluster launch (Slurm submit / batch run)
ENV_MUTATE = "env_mutate"                # installs a package / creates an env (records a cleanup obligation)
BACKGROUNDABLE = "backgroundable"        # launches a long detached run; result may carry a
                                         # background_job descriptor a client watcher polls to completion


# ── Reversibility: how much of an action can be taken back ────────────────────
#
# One ordered dimension instead of a `sensitive` boolean, which put `mkdir` and a Slurm
# submission behind the same door and left no way to spend friction where it matters.
# Sensitivity is *derived* from this (client `_caps_from_meta`), the way plan-blocking
# is derived from the write caps — one concept declared, not two kept in sync.
#
#   reversible   the client itself can undo it — an in-workspace file mutation, which
#                the approval manager snapshots and `revert_last` restores. NOT
#                approval-gated.
#   recoverable  undoable, but by hand and off the client's own record: a delete, an
#                environment mutation, a memory wipe.
#   irreversible it leaves the machine or spends something real: a cluster submission
#                burning allocation hours, an outbound POST, a write outside the
#                workspace. No enforcement level may ever soften these.
REVERSIBLE = "reversible"
RECOVERABLE = "recoverable"
IRREVERSIBLE = "irreversible"
REVERSIBILITY_LEVELS = (REVERSIBLE, RECOVERABLE, IRREVERSIBLE)


def build_descriptor(
    *,
    caps: Iterable[str] | None = None,
    path_args: Iterable[str] | None = None,
    arg_roles: dict[str, Iterable[str]] | None = None,
    reversibility: str | None = None,
    non_batch: bool = False,
    fallbacks: Iterable[str] | None = None,
    label: str | None = None,
    scope: dict[str, Any] | None = None,
    risk_note: str | None = None,
    preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``meta["mimir"]`` descriptor dict (pure / stdlib-only).

    Kept separate from :func:`tool_caps` so it is unit-testable without importing
    the ``mcp`` package; round-trip it through the client's ``infer_tool_caps``.

    *reversibility* is the declared level (see :data:`REVERSIBILITY_LEVELS`); omit it
    and the client derives one from the capabilities. An unrecognised value is dropped
    rather than trusted, so the client's conservative derivation applies.
    """
    descriptor: dict[str, Any] = {}

    cap_list = list(caps or [])
    if cap_list:
        descriptor["capabilities"] = cap_list

    roles: dict[str, list[str]] = {role: list(names) for role, names in (arg_roles or {}).items()}
    if path_args:
        roles["path"] = list(path_args)
    if roles:
        descriptor["arg_roles"] = roles

    approval: dict[str, Any] = {}
    if reversibility in REVERSIBILITY_LEVELS:
        approval["reversibility"] = reversibility
    if non_batch:
        approval["non_batch"] = True
    if fallbacks:
        approval["fallbacks"] = list(fallbacks)
    if approval:
        descriptor["approval"] = approval

    if label:
        descriptor["label"] = label

    # Session-approval scope narrowing: which arg(s) carry the scope, which generic
    # client-side derivation `kind` to apply, and the human label noun. Normalised so
    # `args` is always a list; an empty/argless spec is dropped.
    if scope:
        norm: dict[str, Any] = {}
        args = scope.get("args")
        if isinstance(args, str):
            args = [args]
        if args:
            norm["args"] = list(args)
        if scope.get("kind"):
            norm["kind"] = scope["kind"]
        if scope.get("noun"):
            norm["noun"] = scope["noun"]
        # Require both a derivation kind and the arg(s) it reads; an argless or
        # kindless spec has nothing to narrow on and is dropped.
        if norm.get("kind") and norm.get("args"):
            descriptor["scope"] = norm

    if risk_note:
        descriptor["risk_note"] = risk_note

    # Pre-write diff preview: the generic edit shape (`kind`) plus the arg(s) the
    # client builder reads to reconstruct the post-write content. Unlike `scope`, an
    # argless spec is valid (a deletion has a shape but no content args); a kindless
    # one has nothing to dispatch on and is dropped.
    if preview:
        norm_prev: dict[str, Any] = {}
        prev_args = preview.get("args")
        if isinstance(prev_args, str):
            prev_args = [prev_args]
        if prev_args:
            norm_prev["args"] = list(prev_args)
        if preview.get("kind"):
            norm_prev["kind"] = preview["kind"]
        if norm_prev.get("kind"):
            descriptor["preview"] = norm_prev

    return descriptor


def tool_caps(
    *,
    caps: Iterable[str] | None = None,
    path_args: Iterable[str] | None = None,
    arg_roles: dict[str, Iterable[str]] | None = None,
    reversibility: str | None = None,
    non_batch: bool = False,
    fallbacks: Iterable[str] | None = None,
    label: str | None = None,
    scope: dict[str, Any] | None = None,
    risk_note: str | None = None,
    preview: dict[str, Any] | None = None,
    read_only: bool | None = None,
    destructive: bool | None = None,
) -> dict[str, Any]:
    """Return ``@mcp.tool(**...)`` kwargs declaring this tool's capabilities.

    Declare *reversibility* (see :data:`REVERSIBILITY_LEVELS`) rather than a
    ``sensitive`` flag: approval-gating is derived from it client-side, so a tool
    states one fact about its effect instead of two that can drift apart.

    ``read_only`` / ``destructive`` populate the standard ``ToolAnnotations`` hints
    for cross-client compatibility; when omitted, sensible defaults are derived from
    the declared capabilities (a write/edit/plan-blocked/executes tool is not
    read-only; anything beyond ``reversible`` is destructive).
    """
    descriptor = build_descriptor(
        caps=caps,
        path_args=path_args,
        arg_roles=arg_roles,
        reversibility=reversibility,
        non_batch=non_batch,
        fallbacks=fallbacks,
        label=label,
        scope=scope,
        risk_note=risk_note,
        preview=preview,
    )
    kwargs: dict[str, Any] = {"meta": {"mimir": descriptor}}

    # A tool needing approval is one whose effect the client cannot itself undo.
    gated = reversibility in (RECOVERABLE, IRREVERSIBLE)
    cap_set = set(caps or [])
    mutating = bool(cap_set & {CONTENT_WRITE, EDIT, REMOVE, PLAN_BLOCKED})
    if read_only is None and (cap_set or gated):
        read_only = not (mutating or gated)
    if destructive is None and gated:
        destructive = True

    if read_only is not None or destructive is not None:
        from mcp.types import ToolAnnotations

        kwargs["annotations"] = ToolAnnotations(
            readOnlyHint=read_only,
            destructiveHint=destructive,
        )
    return kwargs


__all__ = [
    "tool_caps",
    "build_descriptor",
    "READ", "CACHEABLE", "SEARCH", "SEARCH_WITH_PATH", "CANDIDATE_SEARCH",
    "INSPECT_DIR", "CHECK_EXISTENCE", "CONTENT_WRITE", "EDIT",
    "REPLACEMENT_TRACK", "VALIDATE",
    "PLAN_BLOCKED", "PLAN_READONLY", "SENSITIVE", "NON_BATCH",
    # Reversibility is the dimension a server *declares*; SENSITIVE is derived from it.
    "REVERSIBLE", "RECOVERABLE", "IRREVERSIBLE", "REVERSIBILITY_LEVELS",
    "CODE_NAV", "ENV_DISCOVERY", "EXTERNAL_FETCH", "CLUSTER_SUBMIT", "ENV_MUTATE",
    "BACKGROUNDABLE", "REMOVE", "OVERWRITE", "TASK_PLANNING",
]
