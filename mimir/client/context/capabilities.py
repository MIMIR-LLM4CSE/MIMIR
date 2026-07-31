"""Tool capability registry — the single source of truth for tool *semantics*.

Historically the client hardcoded the meaning of ~85 specific server tool names
across ~20 sets scattered through ``config``, ``context``, ``policy``,
``tool_execution``, ``query_engine``, ``integration`` and ``ui``: which tools are
sensitive, plan-blocked, writes, edits, searches, validations, reads,
dir-inspections, cacheable, which args carry paths, what fallbacks they have, etc.
Adding or renaming a server tool silently broke classification with *no error*.

This module centralises that knowledge.  A tool is described by a :class:`ToolCaps`
descriptor — an orthogonal set of capability flags plus structured metadata
(argument roles, approval fallbacks, validation ordering, status label).  Every
client consumer derives its behaviour from the registry instead of re-listing tool
names, so a new MCP server "just works".

There are **no hardcoded classification lists** here: each server declares its
tools' capabilities via ``@mcp.tool(**tool_caps(...))`` and the client builds a
**per-agent** live registry (``agent.tool_caps``) in ``connect_server`` from
:func:`infer_tool_caps`.  Consumers read that registry (passed explicitly); with no
registry the helpers resolve to an *empty* one — there is no static fallback table.
``infer_tool_caps`` uses 3-layer precedence: ``tool.meta["mimir"]`` (our
descriptor) › standard ``tool.annotations`` (foreign servers) › conservative default.

The module is a pure-stdlib leaf (no client imports) so every consumer can import it
without circular-import risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Capability vocabulary — orthogonal semantic flags a server declares per tool.
# Mirrored on the server side in servers/_shared/capabilities.py (kept in sync by
# test_capabilities.test_vocab_in_sync). Consumers query these via has_cap /
# names_with_cap. A flag is a reusable behavioral contract the client reacts to,
# never a tool's identity (see module docstring).
#
# The vocabulary is organised by the *policy or nudge that consumes each flag* —
# that is what justifies a flag's existence. There is exactly one declared flag
# per real policy distinction; anything derivable from the others is a helper
# (see ``is_write`` / ``clears_edit_loop`` below), never a separately-declared
# flag. A single tool typically carries flags from several groups (e.g.
# write_file is CONTENT_WRITE + OVERWRITE + EDIT).
# ---------------------------------------------------------------------------

# --- Discovery: reads, searches, inspections the policy tracks as evidence -
# The write policy reads these to build edit/delete preconditions, and the
# discovery/blast-radius nudges read them as "did exploration happen".
READ = "read"                          # _observe_read -> read_files (edit/overwrite precondition; evidence)
SEARCH = "search"                      # _observe_search_flags (evidence; blast-radius)
SEARCH_WITH_PATH = "search_with_path"  # results embed file paths the executor auto-follows
CANDIDATE_SEARCH = "candidate_search"  # _observe_candidates -> path-clarification
INSPECT_DIR = "inspect_dir"            # _observe_dir_inspect -> delete context
CHECK_EXISTENCE = "check_existence"    # _observe_existence_check -> weaker delete context
CODE_NAV = "code_nav"                  # symbol nav; relevance-cap "core keep" set
ENV_DISCOVERY = "env_discovery"        # _observe_env_probe -> env-resolution nudge (enumerates conda/venv interpreters)
CACHEABLE = "cacheable"                # executor result cache + invalidation

# --- Mutation: kinds the write policy treats distinctly --------------------
# delete needs existence+dir context; whole-file overwrite needs a prior read.
# Finer details (where a batch tool's edits live, etc.) are arg-roles, not flags.
EDIT = "edit"                          # any file writer/editor (state machine, dirty tracking, validation entry)
CONTENT_WRITE = "content_write"        # create/append whole files (query-intent create-vs-edit filter)
OVERWRITE = "overwrite"                # replaces an existing file's entire content (subset of content_write)
REMOVE = "remove"                      # deletes a file (needs context; _observe_delete)
REPLACEMENT_TRACK = "replacement_track"  # _observe_replacement_tracking -> cross-file completeness

# --- Validation ------------------------------------------------------------
# A tool that validates code (syntax/lint/typecheck/test). The first-party stack
# validates through the bash server (no dedicated tool carries this), but the flag
# stays in the vocabulary so an extension-pack server can ship its own validator: a
# successful VALIDATE call marks its target file validated (`_observe_validation_tool`)
# and clears the edit-failure streak (`clears_edit_loop`), the same conclude-gate
# contribution a bash validation command makes.
VALIDATE = "validate"

# --- Task planning ---------------------------------------------------------
# One flag for "records a task plan"; the ordered-steps checklist is identified by
# the `plan_steps` arg-role (which carries the steps), not a separate capability.
TASK_PLANNING = "task_planning"        # records a task plan (checklist and/or prose rationale)

# --- Approval & mode policy ------------------------------------------------
# How much of a tool's effect can be taken back. Mirrors the server-side vocabulary in
# `servers/_shared/capabilities.py`. This is the *declared* dimension; SENSITIVE below
# is derived from it (see `_caps_from_meta`), so a tool states one fact about its
# effect instead of two that can drift apart — the same treatment PLAN_BLOCKED already
# gets from the write caps.
REVERSIBLE = "reversible"              # the client itself can undo it (approval snapshots + revert_last)
RECOVERABLE = "recoverable"            # undoable by hand, off the client's record (delete, env mutation)
IRREVERSIBLE = "irreversible"          # leaves the machine or spends something real (cluster hours, outbound POST)
REVERSIBILITY_LEVELS = (REVERSIBLE, RECOVERABLE, IRREVERSIBLE)

SENSITIVE = "sensitive"                # requires approval — DERIVED: reversibility != REVERSIBLE
NON_BATCH = "non_batch"                # always prompt immediately; never batch
# Both names predate ask mode and are kept for compatibility: they gate every mode in
# config.models.READONLY_MODES (plan and ask), not plan alone.
PLAN_BLOCKED = "plan_blocked"          # never called in a read-only mode (writes/exec/mutations)
PLAN_READONLY = "plan_readonly"        # dual-use exec tool allowed in a read-only mode ONLY for read-only/
                                       # discovery invocations (e.g. bash_run for `rg`/`ls`/`cat`); a
                                       # call-time gate rejects its build/exec invocations there.
CODE_EXEC = "code_exec"                # runs a program/shell command/code payload. Scoping signal for guards
                                       # that must inspect *what* is executed (proxy exec guard); not a block.

# --- Reach & cost ----------------------------------------------------------
EXTERNAL_FETCH = "external_fetch"      # reaches outside the workspace (e.g. GitHub API)
CLUSTER_SUBMIT = "cluster_submit"      # expensive cluster launch (Slurm submit / batch run)
ENV_MUTATE = "env_mutate"              # _observe_env_mutation -> conclude-phase cleanup-offer nudge
BACKGROUNDABLE = "backgroundable"      # launches a long detached run; result may carry a
                                       # background_job descriptor a watcher polls to completion



# --- servers whose tools are all read-only (spawn_agent readonly subset) ---
# A server-name allowlist (not tool classification): the subset a read-only
# sub-agent connects. Consumed by servers/agent_state/server_spawn_agent.py.
_READONLY_SERVERS: frozenset[str] = frozenset({
    "files", "search", "code", "math", "strings",
    "datetime", "memory", "system", "platform",
})


# ===========================================================================
# Capability descriptor + registry
# ===========================================================================

@dataclass(frozen=True)
class ToolCaps:
    """Resolved semantics for a single tool name."""

    name: str
    capabilities: frozenset[str] = frozenset()
    # role -> tuple of argument names that carry a value of that role
    arg_roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fallbacks: tuple[str, ...] = ()
    label: str | None = None
    # Session-approval scope narrowing: {"args": [...], "kind": "<kind>", "noun": ...}.
    # The approval manager derives a per-call scope token from `args` using the generic
    # `kind` strategy, so "always" narrows to e.g. one command family / host / package
    # set instead of the whole tool. None -> coarse server:tool scope.
    scope: dict[str, Any] | None = None
    # One-line risk sentence shown in the approval prompt (registry-driven describe_risk).
    risk_note: str | None = None
    # Pre-write diff preview: {"kind": "<edit shape>", "args": [...]}. A file-mutating
    # tool declares the generic shape of its edit (full content / append / replace /
    # line splice / delete) and which args carry it, so a UI can reconstruct the
    # proposed post-write content and render a diff before execution. None -> the
    # tool's mutations can't be previewed (and get the normal approval flow).
    preview: dict[str, Any] | None = None
    # How much of this tool's effect can be taken back: reversible | recoverable |
    # irreversible (see REVERSIBILITY_LEVELS). SENSITIVE is derived from it, so this is
    # the one declared fact rather than two that can drift. Always populated —
    # `_derive_reversibility` supplies a conservative value when nothing is declared.
    reversibility: str = RECOVERABLE

    def has(self, cap: str) -> bool:
        return cap in self.capabilities




# ===========================================================================
# Inference — resolve a live MCP Tool to a ToolCaps with 3-layer precedence
# ===========================================================================

def _meta_descriptor(tool: Any) -> dict | None:
    """Return ``tool.meta["mimir"]`` if present (our authoritative descriptor)."""
    meta = getattr(tool, "meta", None)
    if isinstance(meta, dict):
        desc = meta.get("mimir")
        if isinstance(desc, dict):
            return desc
    return None


def _derive_reversibility(caps: set[str]) -> str:
    """How much of this tool's effect can be taken back, from its capabilities.

    The fallback for a tool that declares no ``reversibility`` — a third-party server,
    or one written before the field existed. Since SENSITIVE is derived from the
    result, this doubles as the approval-gating default, so it has to distinguish
    *unclassified* from *without effect*: defaulting everything to ``recoverable``
    would gate every read tool in the catalog behind an approval prompt.

    ``reversible`` is therefore the base case, matching the pre-existing rule that a
    tool is gated only when it says so. An in-workspace file mutation lands there too,
    because the client *itself* holds the undo: the approval manager snapshots the file
    before the write and ``revert_last`` restores it. That is a fact about MIMIR rather
    than about file writes in general, which is why it is derived here and assumed
    nowhere else.

    ``EXTERNAL_FETCH`` is deliberately absent: it means "reaches outside the
    workspace", which a read-only GitHub query does as much as a POST. What makes an
    outbound call irreversible is that it *sends*, and no capability expresses that —
    such tools declare ``IRREVERSIBLE`` themselves.
    """
    if caps & {CLUSTER_SUBMIT}:
        return IRREVERSIBLE
    if caps & {REMOVE, ENV_MUTATE, CODE_EXEC}:
        return RECOVERABLE
    return REVERSIBLE


def _caps_from_meta(name: str, desc: dict) -> ToolCaps:
    caps = set(desc.get("capabilities", []) or [])
    # Plan-blocking is *derived* for file mutations: any file writer/editor is
    # inherently unsafe in plan mode, so a server needn't declare PLAN_BLOCKED
    # alongside its write caps (and can't forget to). Non-file side effects
    # (exec / db / web / cluster) still declare PLAN_BLOCKED explicitly.
    if caps & {CONTENT_WRITE, EDIT, REMOVE}:
        caps.add(PLAN_BLOCKED)
    approval = desc.get("approval", {}) or {}
    # Sensitivity is derived, not declared — the same way PLAN_BLOCKED is above. A tool
    # states one fact about its effect (how far it can be taken back) and the approval
    # requirement follows: anything the client cannot itself undo is gated. Two
    # independently declared fields would eventually disagree, and the failure mode of
    # that disagreement is a tool running unasked.
    reversibility = approval.get("reversibility")
    if reversibility not in REVERSIBILITY_LEVELS:
        # `sensitive: true` is the pre-reversibility spelling. Honour it as
        # "recoverable" (gated, but not treated as irreversible) so a third-party
        # server written against the old descriptor keeps its prompt.
        reversibility = RECOVERABLE if approval.get("sensitive") else _derive_reversibility(caps)
    if reversibility != REVERSIBLE:
        caps.add(SENSITIVE)
    if approval.get("non_batch"):
        caps.add(NON_BATCH)
    arg_roles = {
        role: tuple(args) for role, args in (desc.get("arg_roles", {}) or {}).items()
    }
    scope = desc.get("scope")
    scope = dict(scope) if isinstance(scope, dict) else None
    preview = desc.get("preview")
    preview = dict(preview) if isinstance(preview, dict) else None
    return ToolCaps(
        name=name,
        capabilities=frozenset(caps),
        arg_roles=arg_roles,
        fallbacks=tuple(approval.get("fallbacks", []) or ()),
        label=desc.get("label"),
        scope=scope,
        risk_note=desc.get("risk_note"),
        preview=preview,
        reversibility=reversibility,
    )


def _caps_from_annotations(name: str, tool: Any) -> ToolCaps | None:
    """Coarse classification from standard MCP ``annotations`` (foreign servers)."""
    ann = getattr(tool, "annotations", None)
    if ann is None:
        return None

    def _hint(key: str) -> Any:
        if isinstance(ann, dict):
            return ann.get(key)
        return getattr(ann, key, None)

    read_only = _hint("readOnlyHint")
    destructive = _hint("destructiveHint")
    if read_only is None and destructive is None:
        return None

    caps: set[str] = set()
    if read_only is True:
        caps.add(CACHEABLE)
    elif read_only is False:
        caps.add(PLAN_BLOCKED)  # mutating -> block in plan mode
    if destructive is True:
        caps.update({SENSITIVE, NON_BATCH})
    return ToolCaps(name=name, capabilities=frozenset(caps))


def _arg_roles_from_schema(tool: Any) -> dict[str, tuple[str, ...]]:
    """Last-resort path-arg inference from the input schema property names."""
    schema = getattr(tool, "inputSchema", None)
    if not isinstance(schema, dict):
        return {}
    props = schema.get("properties")
    if not isinstance(props, dict):
        return {}
    path_like = tuple(k for k in props if k in ("path", "filepath", "file_path", "directory", "subdir"))
    return {"path": path_like} if path_like else {}


def infer_tool_caps(tool: Any) -> ToolCaps:
    """Resolve ``ToolCaps`` for a live MCP ``Tool`` with 3-layer precedence.

    1. ``tool.meta["mimir"]``     — our authoritative descriptor (first-party
       servers declare every classified tool this way).
    2. ``tool.annotations``          — standard readOnly/destructive hints (the
       coarse path for foreign servers that don't carry our descriptor).
    3. conservative default         — empty caps + schema-inferred path args.
    """
    name = getattr(tool, "name", None) or ""

    desc = _meta_descriptor(tool)
    if desc is not None:
        return _caps_from_meta(name, desc)

    ann_caps = _caps_from_annotations(name, tool)
    if ann_caps is not None:
        return ann_caps

    return ToolCaps(name=name, capabilities=frozenset(), arg_roles=_arg_roles_from_schema(tool))


def build_tool_caps(tools: Iterable[Any]) -> dict[str, ToolCaps]:
    """Build ``{name: ToolCaps}`` for a server's advertised tools."""
    return {getattr(t, "name", "") or "": infer_tool_caps(t) for t in tools if getattr(t, "name", None)}


# ===========================================================================
# Query helpers — consumers use these instead of re-listing tool names.
#
# Each requires the per-agent ``registry`` (``agent.tool_caps``). When omitted it
# resolves to an empty registry (everything unclassified) — there is no static
# fallback table; classification is owned by the connected servers.
# ===========================================================================

def _registry(registry: dict[str, ToolCaps] | None) -> dict[str, ToolCaps]:
    return registry if registry is not None else {}


def names_with_cap(cap: str, registry: dict[str, ToolCaps] | None = None) -> set[str]:
    return {name for name, c in _registry(registry).items() if cap in c.capabilities}


def has_cap(name: str, cap: str, registry: dict[str, ToolCaps] | None = None) -> bool:
    c = _registry(registry).get(name)
    return bool(c and cap in c.capabilities)


def is_write(name: str, registry: dict[str, ToolCaps] | None = None) -> bool:
    """Whether ``name`` mutates a file (write/edit/create or delete).

    Derived rather than declared: a tool is a write iff it edits/creates content
    or removes a file. Replaces the former ``IS_WRITE`` flag so a server can't
    declare its write caps yet forget the umbrella. Used for snapshot/targeting
    (agent_core).
    """
    c = _registry(registry).get(name)
    if not c:
        return False
    return bool(c.capabilities & {EDIT, CONTENT_WRITE, REMOVE})


def clears_edit_loop(name: str, registry: dict[str, ToolCaps] | None = None) -> bool:
    """Whether a successful call to ``name`` resets a file's edit-failure tracking.

    Derived rather than declared: re-reading a file or re-running a validator on it
    gives the model fresh ground truth, so the repeated-failed-edit counter clears.
    Replaces the former ``EDIT_LOOP_CLEAR`` flag (== READ or VALIDATE).
    """
    c = _registry(registry).get(name)
    if not c:
        return False
    return bool(c.capabilities & {READ, VALIDATE})


def name_for_cap(cap: str, registry: dict[str, ToolCaps] | None = None) -> str | None:
    """The single connected tool name carrying ``cap``, or ``None`` if absent.

    For the rare site that must *invoke* a capability's tool (e.g. resetting the task
    checklist at session start) without hardcoding its name: the concrete name is
    resolved from the live registry, so a renamed or replaced server tool keeps working.
    Used only for invocation — classification/control-flow uses :func:`has_cap` /
    :func:`names_with_cap`, and prompts describe the capability generically. When several
    tools match, the lexicographically first is returned for stability.
    """
    names = names_with_cap(cap, registry)
    return min(names) if names else None


def caps_for(name: str, registry: dict[str, ToolCaps] | None = None) -> ToolCaps:
    return _registry(registry).get(name) or ToolCaps(name=name)


def path_args(name: str, registry: dict[str, ToolCaps] | None = None) -> tuple[str, ...]:
    """Path-carrying argument names declared for ``name`` (the ``path`` arg role)."""
    c = _registry(registry).get(name)
    return c.arg_roles.get("path", ()) if c else ()


def arg_role(name: str, role: str, registry: dict[str, ToolCaps] | None = None) -> tuple[str, ...]:
    c = _registry(registry).get(name)
    return c.arg_roles.get(role, ()) if c else ()


def names_with_arg_role(role: str, registry: dict[str, ToolCaps] | None = None) -> set[str]:
    """Tool names that declare ``role`` in their arg-roles.

    Lets a consumer select a tool by a structural contract (e.g. ``plan_steps`` —
    the planning tool that carries an ordered step list) rather than by name, the
    arg-role analogue of :func:`names_with_cap`.
    """
    return {name for name, c in _registry(registry).items() if c.arg_roles.get(role)}


def name_with_arg_role(role: str, registry: dict[str, ToolCaps] | None = None) -> str | None:
    """The single tool declaring ``role`` (lexicographically first), for invocation."""
    names = names_with_arg_role(role, registry)
    return min(names) if names else None


def fallbacks(name: str, registry: dict[str, ToolCaps] | None = None) -> tuple[str, ...]:
    c = _registry(registry).get(name)
    return c.fallbacks if c else ()


def label_for(name: str, args: dict | None = None, registry: dict[str, ToolCaps] | None = None) -> str | None:
    """Render a tool's status label template, or ``None`` to fall back to the table.

    Returns ``None`` (rather than the raw template) when a referenced argument is
    absent, so a template that interpolates an *optional* arg — e.g.
    ``"Reading lines {start_line}-{end_line}: {path}"`` — degrades gracefully to
    the generic name-derived label instead of leaking ``{braces}`` to the UI.
    """
    c = _registry(registry).get(name)
    if not c or not c.label:
        return None
    try:
        return c.label.format(**(args or {}))
    except (KeyError, IndexError):
        return None


def scope_spec(name: str, registry: dict[str, ToolCaps] | None = None) -> dict[str, Any] | None:
    """The session-approval scope-narrowing spec declared for ``name``, or ``None``.

    Lets the approval manager narrow an "always" approval by a tool-declared key
    (command family, host, package set, …) without re-listing tool names: the spec
    names the scope arg(s) and the generic derivation ``kind``.
    """
    c = _registry(registry).get(name)
    return c.scope if c else None


def risk_note_of(name: str, registry: dict[str, ToolCaps] | None = None) -> str | None:
    """The per-tool risk sentence declared for ``name`` (approval prompt), or ``None``."""
    c = _registry(registry).get(name)
    return c.risk_note if c else None


def reversibility_of(name: str, registry: dict[str, ToolCaps] | None = None) -> str:
    """How much of ``name``'s effect can be taken back (see REVERSIBILITY_LEVELS).

    Falls back to ``RECOVERABLE`` for an unknown tool: the conservative end, since the
    caller uses this to decide how much friction an action deserves.
    """
    c = _registry(registry).get(name)
    return c.reversibility if c else RECOVERABLE


def preview_spec(name: str, registry: dict[str, ToolCaps] | None = None) -> dict[str, Any] | None:
    """The pre-write diff-preview spec declared for ``name``, or ``None``.

    A non-None spec is also the signal that the tool is a previewable file
    mutation: the WS UI auto-approves such calls (they are batch-reviewed and
    undoable) and renders the reconstructed diff instead of prompting.
    """
    c = _registry(registry).get(name)
    return c.preview if c else None


def readonly_servers() -> frozenset[str]:
    return _READONLY_SERVERS


def unannotated_live_tools(registry: dict[str, ToolCaps]) -> list[str]:
    """Connected tools whose resolved descriptor carries no capabilities.

    With classification owned by the servers, a connected tool that resolves to
    an empty ``ToolCaps`` is either genuinely pure (math, string ops, read-only
    queries) or a first-party tool that *forgot to declare* its caps via
    ``@mcp.tool(**tool_caps(...))``.  Surfacing it is the silent-drift signal the
    registry exists to catch.  Returned sorted for stable reporting.
    """
    return sorted(
        name for name, caps in registry.items()
        if not caps.capabilities and not caps.arg_roles
    )


__all__ = [
    "ToolCaps",
    "infer_tool_caps",
    "build_tool_caps",
    "names_with_cap",
    "has_cap",
    "is_write",
    "clears_edit_loop",
    "name_for_cap",
    "caps_for",
    "path_args",
    "arg_role",
    "names_with_arg_role",
    "name_with_arg_role",
    "scope_spec",
    "risk_note_of",
    "fallbacks",
    "label_for",
    "readonly_servers",
    "unannotated_live_tools",
    # capability constants
    "READ", "CACHEABLE", "SEARCH", "SEARCH_WITH_PATH", "CANDIDATE_SEARCH",
    "INSPECT_DIR", "CHECK_EXISTENCE", "CONTENT_WRITE", "EDIT",
    "REPLACEMENT_TRACK", "VALIDATE",
    "PLAN_BLOCKED", "PLAN_READONLY", "SENSITIVE", "NON_BATCH",
    "CODE_NAV", "ENV_DISCOVERY", "EXTERNAL_FETCH", "CLUSTER_SUBMIT", "ENV_MUTATE",
    "BACKGROUNDABLE", "REMOVE", "OVERWRITE", "TASK_PLANNING",
]
