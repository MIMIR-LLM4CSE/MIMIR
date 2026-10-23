"""The server side of tool capabilities: :func:`tool_caps` declares what a tool means.

It emits ``meta={"mimir": descriptor}`` (read by the client's ``infer_tool_caps``) and
standard ``ToolAnnotations``. The vocabulary below is the one definition of the flags;
the client imports it.
"""

from __future__ import annotations

from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Capability vocabulary: flags a server declares per tool, each a behavioural contract
# consumed by a client policy or nudge, never a tool's identity. Defined here once; the
# client re-exports it, and `mimir_api(topic="capabilities")` lists it from this source,
# so every flag carries a note saying what it drives.
# ---------------------------------------------------------------------------

# --- Discovery: reads, searches, inspections the policy tracks as evidence -
# Feeds edit/delete preconditions and the "did exploration happen" nudges.
READ = "read"                          # files.read -> read_files (edit/overwrite precondition; evidence)
SEARCH = "search"                      # discovery.search_flags (evidence; blast-radius)
SEARCH_WITH_PATH = "search_with_path"  # results locate hits by path and line
CANDIDATE_SEARCH = "candidate_search"  # discovery.candidates -> path-clarification
INSPECT_DIR = "inspect_dir"            # discovery.dir_inspect -> delete context
CHECK_EXISTENCE = "check_existence"    # discovery.existence_check -> weaker delete context
CODE_NAV = "code_nav"                  # symbol nav; relevance-cap "core keep" set
ENV_DISCOVERY = "env_discovery"        # env.env_probe -> env-resolution nudge (enumerates conda/venv interpreters)
CACHEABLE = "cacheable"                # executor result cache + invalidation

# --- Mutation: kinds the write policy treats distinctly --------------------
# delete needs existence+dir context; whole-file overwrite needs a prior read.
# Finer details (where a batch tool's edits live, etc.) are arg-roles, not flags.
# is_write / clears_edit_loop are derived client-side, not declared.
EDIT = "edit"                          # any file writer/editor (state machine, dirty tracking, validation entry)
CONTENT_WRITE = "content_write"        # create/append whole files (query-intent create-vs-edit filter)
OVERWRITE = "overwrite"                # replaces an existing file's entire content (subset of content_write)
REMOVE = "remove"                      # deletes a file (needs context; files.delete)
REPLACEMENT_TRACK = "replacement_track"  # files.replacement_tracking -> cross-file completeness

# --- Validation ------------------------------------------------------------
# Validates code (syntax/lint/typecheck/test). No first-party tool carries this — the
# stack validates through the bash server — but the flag stays so an extension-pack
# server can ship its own validator, with the same conclude-gate contribution a bash
# validation makes (`runs.validation_tool` + `clears_edit_loop`).
VALIDATE = "validate"

# --- Task planning ---------------------------------------------------------
# One flag for "records a task plan"; the ordered-steps checklist is identified by
# the `plan_steps` arg-role (which carries the steps), not a separate capability.
TASK_PLANNING = "task_planning"        # records a task plan (checklist and/or prose rationale)

# --- Verdict and delegation ------------------------------------------------
# Declared, not derived, so the client finds each channel by capability instead of by
# name: the name lives in one place, the server's declaration, and never in prompt or
# nudge copy.
JUDGE = "judge"                        # records the model's verdict on a run's output (evidence.verdict_tool)
DELEGATE = "delegate"                  # hands a self-contained sub-task to a fresh child agent that
                                       # runs to completion and returns its answer
MAIN_ONLY = "main_only"                # reserved to the orchestrating agent; never granted to a child
                                       # (the approved plan, the user's questions, the shared memory)
PANEL_REPORT = "panel_report"          # fills one section of the scientific-computing panel, declared
                                       # with a `panel` spec {"section", "order", "args", "refresh_arg"?}
CODE_MAP = "code_map"                  # answers the workspace's code map: shared.code_map's dict
                                       # when called with shared.code_map.FULL_ARGS, for the map tab
CODE_MAP_STEPS = "code_map_steps"      # answers the optimisation session's steps for the map tab:
                                       # each run's verdict, metric and the files and edges it changed

# --- Reversibility: how much of an action can be taken back ----------------
# One ordered dimension, the *declared* one, from which SENSITIVE below is derived
# client-side, so a tool states one fact about its effect rather than two that can drift.
REVERSIBLE = "reversible"              # the client itself can undo it (approval snapshots + revert_last)
RECOVERABLE = "recoverable"            # undoable by hand, off the client's record (delete, env mutation)
IRREVERSIBLE = "irreversible"          # leaves the machine or spends something real (cluster hours, outbound POST)
REVERSIBILITY_LEVELS = (REVERSIBLE, RECOVERABLE, IRREVERSIBLE)

# --- Approval & mode policy ------------------------------------------------
SENSITIVE = "sensitive"                # requires approval — DERIVED: reversibility != REVERSIBLE
NON_BATCH = "non_batch"                # always prompt immediately; never batch
# Both names gate every mode in config.models.READONLY_MODES (plan and ask), not plan alone.
PLAN_BLOCKED = "plan_blocked"          # never called in a read-only mode (writes/exec/mutations)
PLAN_READONLY = "plan_readonly"        # dual-use exec tool allowed in a read-only mode ONLY for read-only/
                                       # discovery invocations (e.g. bash_run for `rg`/`ls`/`cat`); a
                                       # call-time gate rejects its build/exec invocations there.
CODE_EXEC = "code_exec"                # runs a program/shell command/code payload. Scoping signal for guards
                                       # that must inspect *what* is executed (proxy exec guard); not a block.

# --- Reach & cost ----------------------------------------------------------
EXTERNAL_FETCH = "external_fetch"      # reaches outside the workspace (e.g. GitHub API)
CLUSTER_SUBMIT = "cluster_submit"      # expensive cluster launch (Slurm submit / batch run)
ENV_MUTATE = "env_mutate"              # installs a package / creates an env (env.env_mutation ->
                                       # conclude-phase cleanup-offer nudge)
BACKGROUNDABLE = "backgroundable"      # launches a long detached run; result may carry a
                                       # background_job descriptor a watcher polls to completion
DIVERTIBLE = "divertible"              # publishes a run channel under its own name while it
                                       # blocks, so the user can move the run it is waiting on
                                       # to the background mid-flight (see shared/run_channel.py)


# ── Work families: the word a tool-activity row shows, and its icon ───────────
# Declared because it cannot always be derived (the client's ``kind_for`` derives one
# for a tool that declares none); the row's sentence comes from the model's ``doing``
# argument.
TOOL_KINDS = (
    # discovery
    "read", "search", "list", "outline",
    # file mutation
    "write", "edit", "delete",
    # local execution. `bash` is what the shell tool actually runs (and `/bin/sh` is
    # refused outright, so it is never vaguer than that); `shell` is the generic family a
    # third-party execution tool derives, since nothing says it runs bash.
    "bash", "shell", "job", "verdict",
    # calculation
    "eval", "symbolic", "string", "date",
    # cluster & environment
    "slurm", "env", "modules",
    # the proxy harness
    "proxy", "proxy eval",
    # agent state. `plan` is a saved, titled plan in prose; `todo` is the ordered
    # checklist of steps. One server holds both, and they are not the same object:
    # writing the checklist and recording a plan are different acts on different things.
    "memory", "plan", "todo", "skill", "agent", "ask",
    # outside the workspace
    "web", "github", "system",
    # the unknown default (the client supplies it; never declared)
    "tool",
)


# Condition kinds a run-outcome spec may carry. The `_present` forms fire on a field
# merely being there and truthy — a per-case error string has no enumerable value set.
# `measured_when` is the one positive form, and it credits *evidence level* only —
# that the file was exercised and measured, never that the measurement was good. That
# claim is a verdict on the run, and no server may grant one for itself.
_OUTCOME_CONDITIONS = frozenset({
    "crashed_when", "failed_when", "measured_when",
    "crashed_when_present", "failed_when_present", "measured_when_present",
})


def _norm_conditions(src: dict[str, Any], dest: dict[str, Any]) -> None:
    """Copy the normalised outcome conditions of *src* into *dest*."""
    for key in ("crashed_when", "failed_when", "measured_when"):
        conds = src.get(key)
        if isinstance(conds, dict) and conds:
            dest[key] = {
                str(field): list(values) if isinstance(values, (list, tuple)) else [values]
                for field, values in conds.items()
            }
    for key in ("crashed_when_present", "failed_when_present", "measured_when_present"):
        fields = src.get(key)
        if isinstance(fields, (list, tuple)) and fields:
            dest[key] = [str(f) for f in fields]


def _as_list(value: Any) -> list:
    """A string or a sequence as a list; anything falsy as an empty list."""
    if isinstance(value, str):
        return [value]
    return list(value or [])


def _norm_arg_roles(arg_roles: dict[str, Iterable[str]] | None,
                    path_args: Iterable[str] | None) -> dict[str, list[str]] | None:
    roles = {role: list(names) for role, names in (arg_roles or {}).items()}
    if path_args:
        roles["path"] = list(path_args)
    return roles or None


def _norm_approval(reversibility: str | None, non_batch: bool,
                   fallbacks: Iterable[str] | None) -> dict[str, Any] | None:
    # An unrecognised reversibility is dropped, so the client derives one.
    approval: dict[str, Any] = {}
    if reversibility in REVERSIBILITY_LEVELS:
        approval["reversibility"] = reversibility
    if non_batch:
        approval["non_batch"] = True
    if fallbacks:
        approval["fallbacks"] = list(fallbacks)
    return approval or None


def _norm_kind(kind: str | None) -> str | None:
    """The work family (see TOOL_KINDS), normalised but not restricted: a pack's own
    family gets its word on the row with the default icon."""
    if isinstance(kind, str) and kind.strip():
        return " ".join(kind.lower().split())[:16]
    return None


def _norm_scope(scope: dict[str, Any] | None) -> dict[str, Any] | None:
    """Session-approval narrowing: the args carrying the scope, the client derivation
    ``kind`` and the label ``noun``. Without both args and kind there is nothing to
    narrow on."""
    if not scope:
        return None
    norm: dict[str, Any] = {}
    args = _as_list(scope.get("args"))
    if args:
        norm["args"] = args
    for key in ("kind", "noun"):
        if scope.get(key):
            norm[key] = scope[key]
    return norm if norm.get("kind") and norm.get("args") else None


def _norm_panel(panel: dict[str, Any] | None) -> dict[str, Any] | None:
    """Which section of the client's panel this tool fills, its order and call args."""
    if not panel or not panel.get("section"):
        return None
    norm = {"section": str(panel["section"]), "order": int(panel.get("order") or 100),
            "args": dict(panel.get("args") or {})}
    if panel.get("refresh_arg"):
        norm["refresh_arg"] = str(panel["refresh_arg"])
    return norm


def _norm_readonly_when(readonly_when: dict[str, Any] | None) -> dict[str, Any] | None:
    """``{"arg", "values"}``: the argument values that make a dual-use tool read-only."""
    if readonly_when and readonly_when.get("arg") and readonly_when.get("values"):
        return {"arg": str(readonly_when["arg"]),
                "values": [str(v) for v in readonly_when["values"]]}
    return None


def _norm_preview(preview: dict[str, Any] | None) -> dict[str, Any] | None:
    """The pre-write diff shape (``kind``) and the args it reads. Unlike a scope, an
    argless preview is valid (a deletion); a kindless one is dropped."""
    if not preview or not preview.get("kind"):
        return None
    norm: dict[str, Any] = {}
    args = _as_list(preview.get("args"))
    if args:
        norm["args"] = args
    norm["kind"] = preview["kind"]
    return norm


def _norm_run_outcome(run_outcome: dict[str, Any] | None) -> dict[str, Any] | None:
    """What the server saw of a run it performed: ``id`` names the payload field,
    ``*_when`` map fields to red values, ``rows`` the same nested in a list field.
    An id with nothing to judge on is dropped."""
    if not run_outcome:
        return None
    norm: dict[str, Any] = {}
    if run_outcome.get("id"):
        norm["id"] = str(run_outcome["id"])
    _norm_conditions(run_outcome, norm)
    rows = run_outcome.get("rows")
    if isinstance(rows, dict) and rows.get("field"):
        norm_rows: dict[str, Any] = {"field": str(rows["field"])}
        if rows.get("id"):
            norm_rows["id"] = str(rows["id"])
        _norm_conditions(rows, norm_rows)
        norm["rows"] = norm_rows
    if norm.get("id") and (norm.keys() & _OUTCOME_CONDITIONS or norm.get("rows")):
        return norm
    return None


def build_descriptor(
    *,
    caps: Iterable[str] | None = None,
    path_args: Iterable[str] | None = None,
    arg_roles: dict[str, Iterable[str]] | None = None,
    reversibility: str | None = None,
    non_batch: bool = False,
    fallbacks: Iterable[str] | None = None,
    label: str | None = None,
    kind: str | None = None,
    scope: dict[str, Any] | None = None,
    risk_note: str | None = None,
    preview: dict[str, Any] | None = None,
    timeout_secs: int | None = None,
    readonly_when: dict[str, Any] | None = None,
    run_outcome: dict[str, Any] | None = None,
    panel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``meta["mimir"]`` descriptor dict (stdlib only, testable without ``mcp``).

    Each section is normalised on its own; an empty one is left out.
    """
    sections: dict[str, Any] = {
        "capabilities": list(caps or []) or None,
        "arg_roles": _norm_arg_roles(arg_roles, path_args),
        "approval": _norm_approval(reversibility, non_batch, fallbacks),
        "label": label or None,
        "kind": _norm_kind(kind),
        "scope": _norm_scope(scope),
        "risk_note": risk_note or None,
        "panel": _norm_panel(panel),
        # A tool's own wall, which the client clamps.
        "timeout_secs": timeout_secs if isinstance(timeout_secs, int) and timeout_secs > 0
        else None,
        "readonly_when": _norm_readonly_when(readonly_when),
        "preview": _norm_preview(preview),
        "run_outcome": _norm_run_outcome(run_outcome),
    }
    return {key: value for key, value in sections.items() if value is not None}


def tool_caps(
    *,
    caps: Iterable[str] | None = None,
    path_args: Iterable[str] | None = None,
    arg_roles: dict[str, Iterable[str]] | None = None,
    reversibility: str | None = None,
    non_batch: bool = False,
    fallbacks: Iterable[str] | None = None,
    label: str | None = None,
    kind: str | None = None,
    scope: dict[str, Any] | None = None,
    risk_note: str | None = None,
    preview: dict[str, Any] | None = None,
    timeout_secs: int | None = None,
    readonly_when: dict[str, Any] | None = None,
    run_outcome: dict[str, Any] | None = None,
    panel: dict[str, Any] | None = None,
    read_only: bool | None = None,
    destructive: bool | None = None,
) -> dict[str, Any]:
    """``@mcp.tool(**...)`` kwargs declaring this tool's capabilities.

    Declare *reversibility*, not "sensitive": gating is derived from it. ``read_only`` and
    ``destructive`` default from the capabilities.
    """
    descriptor = build_descriptor(
        caps=caps,
        path_args=path_args,
        arg_roles=arg_roles,
        reversibility=reversibility,
        non_batch=non_batch,
        fallbacks=fallbacks,
        label=label,
        kind=kind,
        scope=scope,
        risk_note=risk_note,
        preview=preview,
        timeout_secs=timeout_secs,
        readonly_when=readonly_when,
        run_outcome=run_outcome,
        panel=panel,
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
    "BACKGROUNDABLE", "DIVERTIBLE", "REMOVE", "OVERWRITE", "TASK_PLANNING", "JUDGE",
    "DELEGATE", "MAIN_ONLY", "PANEL_REPORT", "CODE_MAP", "CODE_MAP_STEPS",
]
