"""The tool capability registry: what each tool is, as its server declares it.

A :class:`ToolCaps` per tool, read by every consumer instead of a list of tool names.
``infer_tool_caps`` resolves it from ``tool.meta["mimir"]``, else the MCP annotations,
else a conservative default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..config.limits import TOOL_CALL_TIMEOUT_MAX_SECS, TOOL_CALL_TIMEOUT_SECS

# The capability vocabulary is the servers' declaration language, defined once in
# shared/capabilities.py and re-exported here for the client's readers and plugin packs.
from ...shared.capabilities import (  # noqa: F401  (re-exported)
    BACKGROUNDABLE,
    CACHEABLE,
    CANDIDATE_SEARCH,
    CHECK_EXISTENCE,
    CLUSTER_SUBMIT,
    CODE_EXEC,
    CODE_MAP,
    CODE_NAV,
    CONTENT_WRITE,
    DELEGATE,
    DIVERTIBLE,
    EDIT,
    ENV_DISCOVERY,
    ENV_MUTATE,
    EXTERNAL_FETCH,
    INSPECT_DIR,
    IRREVERSIBLE,
    JUDGE,
    MAIN_ONLY,
    NON_BATCH,
    OVERWRITE,
    PANEL_REPORT,
    PLAN_BLOCKED,
    PLAN_READONLY,
    READ,
    RECOVERABLE,
    REMOVE,
    REPLACEMENT_TRACK,
    REVERSIBILITY_LEVELS,
    REVERSIBLE,
    SEARCH,
    SEARCH_WITH_PATH,
    SENSITIVE,
    TASK_PLANNING,
    TOOL_KINDS,
    VALIDATE,
)

# The family of a tool that declares none and whose capabilities derive none.
DEFAULT_TOOL_KIND = TOOL_KINDS[-1]


# The subset an exploring sub-agent connects — a *connection-cost* filter, not the
# read-only guarantee: `files` carries the write tools too. What makes the child
# read-only is the read-only mode it runs in, where the capability-driven tool filter
# and the dual-use call gate both apply. Consumed by server_spawn_agent.py.
# Kept to what reading a repository actually needs: every entry costs the child a
# subprocess to start before its first step, and a fan-out pays that per child.
_EXPLORER_SERVERS: frozenset[str] = frozenset({
    "files", "search", "code_intel", "bash",
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
    # The work family this tool belongs to (see TOOL_KINDS) — the word its activity row
    # shows and the icon drawn beside it. None -> `kind_for` derives one.
    kind: str | None = None
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
    # Which section of the scientific-computing panel this tool fills, and how to call
    # it: {"section", "order", "args", "refresh_arg"?}. Read by ``panel_sections``.
    # ``refresh_arg`` names the parameter that bypasses the tool's own cache, set only
    # when the reader asked for a re-read rather than when a timer did.
    panel: dict[str, Any] | None = None

    def has(self, cap: str) -> bool:
        return cap in self.capabilities




# ===========================================================================
# Inference — resolve a live MCP Tool to a ToolCaps with 3-layer precedence
# ===========================================================================

def _meta_descriptor(tool: Any) -> dict[str, Any] | None:
    """Return ``tool.meta["mimir"]`` if present (our authoritative descriptor)."""
    meta = getattr(tool, "meta", None)
    if isinstance(meta, dict):
        desc = meta.get("mimir")
        if isinstance(desc, dict):
            return desc
    return None


def _derive_reversibility(caps: set[str]) -> str:
    """How much of a tool's effect can be taken back, for one that declares none.

    ``reversible`` is the base case, so read tools are not gated; an in-workspace write is
    too, since the approval manager snapshots the file before it. A tool that sends declares
    ``IRREVERSIBLE`` itself.
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


def _kind(value: Any) -> str | None:
    """Normalise a declared work family, keeping any usable word (not only ``TOOL_KINDS``)."""
    if not isinstance(value, str) or not value.strip():
        return None
    return " ".join(value.lower().split())[:16]


def _caps_from_meta(name: str, desc: dict[str, Any]) -> ToolCaps:
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
        # A third-party server may still declare `sensitive: true`: read it as
        # "recoverable" (gated, not irreversible) so its tool keeps its prompt.
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
    panel = desc.get("panel")
    panel = dict(panel) if isinstance(panel, dict) else None
    return ToolCaps(
        name=name,
        capabilities=frozenset(caps),
        arg_roles=arg_roles,
        fallbacks=tuple(approval.get("fallbacks", []) or ()),
        label=desc.get("label"),
        kind=_kind(desc.get("kind")),
        scope=scope,
        risk_note=desc.get("risk_note"),
        preview=preview,
        reversibility=reversibility,
        timeout_secs=_declared_timeout(desc.get("timeout_secs")),
        readonly_when=_readonly_when(desc.get("readonly_when")),
        run_outcome=outcome,
        panel=panel,
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
    """Resolve ``ToolCaps`` for a live MCP ``Tool``.

    Our ``tool.meta["mimir"]`` descriptor first, then the standard annotations, then empty
    caps with path arguments inferred from the schema.
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


def panel_sections(registry: dict[str, ToolCaps] | None = None) -> list[tuple[str, dict[str, Any]]]:
    """The tools that fill a panel section, in the order they declared, as (name, spec).

    By capability, so the panel asks whoever can answer — a plugin server adds its own
    section by declaring ``PANEL_REPORT`` and a ``panel`` spec, with no client edit.
    """
    reg = _registry(registry)
    found = [
        (name, dict(c.panel or {}))
        for name, c in reg.items()
        if PANEL_REPORT in c.capabilities and c.panel and c.panel.get("section")
    ]
    return sorted(found, key=lambda item: (item[1].get("order", 100), item[0]))


def code_map_tools(registry: dict[str, ToolCaps] | None = None) -> list[str]:
    """The tools that answer the code map, by capability, in name order."""
    return sorted(name for name, c in _registry(registry).items() if CODE_MAP in c.capabilities)


def reserved_for_main(registry: dict[str, ToolCaps] | None = None) -> set[str]:
    """Tools a sub-agent is never granted: ``MAIN_ONLY`` and ``DELEGATE``.

    The user's plan, questions and memory; a delegation tree nobody budgeted.
    ``TASK_PLANNING`` is not here: a sub-agent keeps its own checklist. ``CLUSTER_SUBMIT``
    is not either: it is granted at the "build" rung (``toollist.grantable_tools``).
    """
    reg = _registry(registry)
    return {
        name for name, c in reg.items()
        if c.capabilities & {MAIN_ONLY, DELEGATE}
    }


def is_write(name: str, registry: dict[str, ToolCaps] | None = None) -> bool:
    """Whether ``name`` mutates a file (write/edit/create or delete).

    Derived rather than declared: a tool is a write iff it edits/creates content
    or removes a file, so a server cannot declare its write caps and forget an
    umbrella flag. Used for snapshot/targeting (agent_core).
    """
    c = _registry(registry).get(name)
    if not c:
        return False
    return bool(c.capabilities & {EDIT, CONTENT_WRITE, REMOVE})


def clears_edit_loop(name: str, registry: dict[str, ToolCaps] | None = None) -> bool:
    """Whether a successful call to ``name`` resets a file's edit-failure tracking.

    Derived rather than declared: re-reading a file or re-running a validator on it
    gives the model fresh ground truth, so the repeated-failed-edit counter clears
    (READ or VALIDATE).
    """
    c = _registry(registry).get(name)
    if not c:
        return False
    return bool(c.capabilities & {READ, VALIDATE})


def name_for_cap(cap: str, registry: dict[str, ToolCaps] | None = None) -> str | None:
    """The connected tool carrying *cap* (the first by name), or ``None``.

    Only for invoking a capability's tool; classification goes through :func:`has_cap`.
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


# Where a tool that declares no ``path`` arg-role conventionally carries the file it
# targets, and where a listing tool carries the directory. The one list every consumer
# reads — the snapshot taken before a write, the observers, the guards, the status row
# — so a tool is never one file's target to one of them and nobody's to another.
CONVENTIONAL_PATH_ARGS: tuple[str, ...] = ("path", "filepath", "filePath", "file_path", "file")
DIRECTORY_ARGS: tuple[str, ...] = ("subdir", "directory")


def target_path_arg(name: str, arguments: dict[str, Any],
                    registry: dict[str, ToolCaps] | None = None, *,
                    directories: bool = False) -> str:
    """The raw path argument this call targets, or "".

    The tool's declared ``path`` arg-role first, else the conventional names; with
    *directories*, a listing tool's directory argument counts too. Unnormalised: the
    caller resolves it against the workspace it works in.
    """
    keys = arg_role(name, "path", registry) or CONVENTIONAL_PATH_ARGS
    if directories:
        keys = (*keys, *DIRECTORY_ARGS)
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


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


def label_for(name: str, args: dict[str, Any] | None = None, registry: dict[str, ToolCaps] | None = None) -> str | None:
    """Render a tool's status label template, or ``None`` when an argument it names is absent."""
    c = _registry(registry).get(name)
    if not c or not c.label:
        return None
    try:
        return c.label.format(**(args or {}))
    except (KeyError, IndexError):
        return None


def kind_for(name: str, registry: dict[str, ToolCaps] | None = None) -> str:
    """The work family of *name*: declared, else derived from its capabilities, else default.

    Derived by effect: a tool that reads and rewrites is an ``edit``; reading is checked
    last because almost everything reads.
    """
    c = _registry(registry).get(name)
    if c and c.kind:
        return c.kind
    caps = c.capabilities if c else frozenset()
    for cap, kind in _DERIVED_KINDS:
        if cap in caps:
            return kind
    return DEFAULT_TOOL_KIND


# Capability -> work family, in order of precedence. Consulted only for a tool that
# declares no kind of its own. Ordered by effect: what a call *does* outranks what it
# merely also reads, so an editor is an "edit" and not a "read".
_DERIVED_KINDS: tuple[tuple[str, str], ...] = (
    (REMOVE, "delete"),
    (EDIT, "edit"),
    (CONTENT_WRITE, "write"),
    (CLUSTER_SUBMIT, "slurm"),
    (ENV_MUTATE, "env"),
    (CODE_EXEC, "shell"),
    (DELEGATE, "agent"),
    (JUDGE, "verdict"),
    (TASK_PLANNING, "todo"),
    (EXTERNAL_FETCH, "web"),
    (SEARCH_WITH_PATH, "search"),
    (SEARCH, "search"),
    (CANDIDATE_SEARCH, "search"),
    (CODE_NAV, "search"),
    (INSPECT_DIR, "list"),
    (ENV_DISCOVERY, "env"),
    (READ, "read"),
)


def run_outcome_spec(
    name: str, registry: dict[str, ToolCaps] | None = None,
) -> dict[str, Any] | None:
    """The machine run-outcome spec declared for ``name``, or ``None``.

    Lets a server that performs runs itself tell the client what it saw of them,
    so the run ledger has a floor under the model's stated verdict.
    """
    c = _registry(registry).get(name)
    return c.run_outcome if c else None


# The scope kind a tool taking a raw shell command line declares. What identifies a
# shell tool everywhere — never its name: the guards and observers that read arguments
# *as shell* key off this declaration about the tool's interface.
SHELL_SCOPE_KIND = "command_prefix"


def shell_command_args(name: str, registry: dict[str, ToolCaps] | None = None
                       ) -> tuple[str, ...] | None:
    """The argument names that carry a raw shell command, if ``name`` takes one, else None."""
    spec = scope_spec(name, registry)
    if not spec or spec.get("kind") != SHELL_SCOPE_KIND:
        return None
    return tuple(spec.get("args") or ("command",))


def shell_command_of(name: str, arguments: dict[str, Any],
                     registry: dict[str, ToolCaps] | None = None) -> str:
    """The shell command this call carries, or "" when it carries none."""
    for arg in shell_command_args(name, registry) or ():
        value = arguments.get(arg)
        if isinstance(value, str) and value.strip():
            return value
    return ""


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
    """Connected tools whose descriptor carries no capabilities, sorted.

    Either genuinely pure, or a first-party tool that forgot its ``tool_caps``.
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
    "BACKGROUNDABLE", "DIVERTIBLE", "REMOVE", "OVERWRITE", "TASK_PLANNING", "JUDGE",
    "DELEGATE", "MAIN_ONLY", "PANEL_REPORT", "CODE_MAP", "reserved_for_main", "panel_sections",
    "code_map_tools",
]
