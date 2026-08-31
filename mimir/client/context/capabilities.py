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

from ..config.constants import TOOL_CALL_TIMEOUT_MAX_SECS, TOOL_CALL_TIMEOUT_SECS

# ---------------------------------------------------------------------------
# Capability vocabulary — orthogonal semantic flags a server declares per tool,
# mirrored server-side in servers/_shared/capabilities.py (test_capabilities
# .test_vocab_in_sync keeps them level). A flag is a reusable behavioral contract,
# never a tool's identity. Organised by the *policy or nudge that consumes it* —
# that is what justifies a flag existing. Exactly one declared flag per real policy
# distinction; anything derivable is a helper (``is_write``, ``clears_edit_loop``).
# ---------------------------------------------------------------------------

# --- Discovery: reads, searches, inspections the policy tracks as evidence -
# Feeds edit/delete preconditions and the "did exploration happen" nudges.
READ = "read"                          # _observe_read -> read_files (edit/overwrite precondition; evidence)
SEARCH = "search"                      # _observe_search_flags (evidence; blast-radius)
SEARCH_WITH_PATH = "search_with_path"  # results locate hits by path and line
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
# Validates code (syntax/lint/typecheck/test). No first-party tool carries this — the
# stack validates through the bash server — but the flag stays so an extension-pack
# server can ship its own validator, with the same conclude-gate contribution a bash
# validation makes (`_observe_validation_tool` + `clears_edit_loop`).
VALIDATE = "validate"

# --- Task planning ---------------------------------------------------------
# One flag for "records a task plan"; the ordered-steps checklist is identified by
# the `plan_steps` arg-role (which carries the steps), not a separate capability.
TASK_PLANNING = "task_planning"        # records a task plan (checklist and/or prose rationale)

# --- Verdict ---------------------------------------------------------------
# The channel through which the model states what a run's output showed
# (_observe_verdict_tool -> guardrails/verdict.apply_verdict). Every consumer that has
# to name the tool to the model resolves it from here, so the name lives in one place
# — the server's declaration — and never in prompt or nudge copy.
JUDGE = "judge"                        # records the model's verdict on a run's output
DELEGATE = "delegate"                  # hands a self-contained sub-task to a fresh child agent that
                                       # runs to completion and returns its answer

# --- Approval & mode policy ------------------------------------------------
# How much of a tool's effect can be taken back — the *declared* dimension, from which
# SENSITIVE below is derived (see `_caps_from_meta`), so a tool states one fact about
# its effect rather than two that can drift.
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



# The subset an exploring sub-agent connects — a *connection-cost* filter, not the
# read-only guarantee: `files` carries the write tools too. What makes the child
# read-only is the read-only mode it runs in, where the capability-driven tool filter
# and the dual-use call gate both apply. Consumed by server_spawn_agent.py.
_EXPLORER_SERVERS: frozenset[str] = frozenset({
    "files", "search", "code_intel", "bash", "web", "math", "strings",
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
    # Lets "always" narrow to one command family / host / package set instead of the
    # whole tool. None -> coarse server:tool scope.
    scope: dict[str, Any] | None = None
    # One-line risk sentence shown in the approval prompt (registry-driven describe_risk).
    risk_note: str | None = None
    # Pre-write diff preview: {"kind": "<edit shape>", "args": [...]}. Declares the
    # generic shape of the edit (full content / append / replace / splice / delete) and
    # which args carry it, so a UI can reconstruct the post-write content and diff it.
    # None -> not previewable (normal approval flow).
    preview: dict[str, Any] | None = None
    # reversible | recoverable | irreversible (see REVERSIBILITY_LEVELS). Always
    # populated — `_derive_reversibility` supplies a conservative value when undeclared.
    reversibility: str = RECOVERABLE
    # The tool's own per-call wall, when the global default cannot bound its work.
    # None -> the dispatcher's default applies.
    timeout_secs: int | None = None
    # For a PLAN_READONLY (dual-use) tool: {"arg": name, "values": [...]} naming the
    # invocations that are the read-only ones. None -> the guard falls back to
    # classifying the shell command it carries.
    readonly_when: dict[str, Any] | None = None
    # What the server itself saw of a run it performed: {"id": field, "crashed_when":
    # {...}, "failed_when": {...}, "rows": {...}}. The floor under the model's stated
    # verdict for a non-shell execution tool; withholds credit only, never grants it.
    # None -> the client has no machine reading of this tool's runs.
    run_outcome: dict[str, Any] | None = None

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


def _declared_timeout(value: Any) -> int | None:
    """A declared per-call wall, clamped. A third-party server does not get to hang the loop."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    return min(value, TOOL_CALL_TIMEOUT_MAX_SECS)


def _readonly_when(value: Any) -> dict[str, Any] | None:
    """Normalise a dual-use tool's read-only-invocation spec; drop an incomplete one."""
    if not isinstance(value, dict):
        return None
    arg, values = value.get("arg"), value.get("values")
    if not isinstance(arg, str) or not isinstance(values, (list, tuple)) or not values:
        return None
    return {"arg": arg, "values": [str(v) for v in values]}


def _caps_from_meta(name: str, desc: dict) -> ToolCaps:
    caps = set(desc.get("capabilities", []) or [])
    # Derived for file mutations so a server can't forget to declare it alongside its
    # write caps. Non-file side effects (exec/db/web/cluster) still declare it.
    if caps & {CONTENT_WRITE, EDIT, REMOVE}:
        caps.add(PLAN_BLOCKED)
    approval = desc.get("approval", {}) or {}
    # Sensitivity is derived, like PLAN_BLOCKED above: anything the client cannot itself
    # undo is gated. Two independently declared fields would eventually disagree, and
    # the failure mode of that disagreement is a tool running unasked.
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
    outcome = desc.get("run_outcome")
    outcome = dict(outcome) if isinstance(outcome, dict) else None
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
        timeout_secs=_declared_timeout(desc.get("timeout_secs")),
        readonly_when=_readonly_when(desc.get("readonly_when")),
        run_outcome=outcome,
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
# Query helpers — consumers use these instead of re-listing tool names. Each takes
# the per-agent ``registry`` (``agent.tool_caps``); omitting it resolves to an empty
# one (everything unclassified). There is no static fallback table.
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


def readonly_invocation_spec(name: str, registry: dict[str, ToolCaps] | None = None) -> dict[str, Any] | None:
    """A dual-use tool's declared read-only-invocation spec, if it has one."""
    c = _registry(registry).get(name)
    return c.readonly_when if c else None


def timeout_for(name: str, registry: dict[str, ToolCaps] | None = None) -> int:
    """This tool's per-call wall: what it declares, else the global default.

    The default is calibrated on a search or a build step; a tool whose work is an
    agent run of its own declares its own, and the dispatcher asks here instead of
    holding a name-keyed table.
    """
    c = _registry(registry).get(name)
    return c.timeout_secs if c and c.timeout_secs else TOOL_CALL_TIMEOUT_SECS


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


def run_outcome_spec(
    name: str, registry: dict[str, ToolCaps] | None = None,
) -> dict[str, Any] | None:
    """The machine run-outcome spec declared for ``name``, or ``None``.

    Lets a server that performs runs itself tell the client what it saw of them,
    so the run ledger has a floor under the model's stated verdict.
    """
    c = _registry(registry).get(name)
    return c.run_outcome if c else None


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


def explorer_servers() -> frozenset[str]:
    return _EXPLORER_SERVERS


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
    "explorer_servers",
    "timeout_for",
    "readonly_invocation_spec",
    "unannotated_live_tools",
    # capability constants
    "READ", "CACHEABLE", "SEARCH", "SEARCH_WITH_PATH", "CANDIDATE_SEARCH",
    "INSPECT_DIR", "CHECK_EXISTENCE", "CONTENT_WRITE", "EDIT",
    "REPLACEMENT_TRACK", "VALIDATE",
    "PLAN_BLOCKED", "PLAN_READONLY", "SENSITIVE", "NON_BATCH",
    "CODE_NAV", "ENV_DISCOVERY", "EXTERNAL_FETCH", "CLUSTER_SUBMIT", "ENV_MUTATE",
    "BACKGROUNDABLE", "REMOVE", "OVERWRITE", "TASK_PLANNING", "JUDGE", "DELEGATE",
]
