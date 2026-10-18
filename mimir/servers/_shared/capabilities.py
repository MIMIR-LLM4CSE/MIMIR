"""The server side of tool capabilities: :func:`tool_caps` declares what a tool means.

It emits ``meta={"mimir": descriptor}`` (read by the client's ``infer_tool_caps``) and
standard ``ToolAnnotations``. The vocabulary is mirrored from the client as plain
strings; a parity test keeps them level.
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

# Records the model's reading of what a run's output showed. Declared, not derived, so
# the client can find the channel by capability instead of by name.
JUDGE = "judge"

# Hands a self-contained sub-task to a fresh child agent that runs to completion and
# returns its answer. Declared so prompt, guidance and observation can all speak of the
# delegation channel without naming the tool.
DELEGATE = "delegate"

# Reserved to the orchestrating agent: never granted to a sub-agent. For what only the
# run the user is watching may do — the plan it agreed to, the questions it asks, the
# memory shared by every session. Declared by the tool, so the client filters by
# capability and no list of names lives client-side.
MAIN_ONLY = "main_only"

# Fills one section of the client's panel, declared with a ``panel`` spec
# (``{"section", "order", "args", "refresh_arg"?}``) and answering
# ``{"title", "lines": [{"label", "value", "state"?}], "detail"?}``. ``refresh_arg`` names
# the argument that bypasses the server's cache, set only for the user's ⟳.
PANEL_REPORT = "panel_report"

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
DIVERTIBLE = "divertible"                # publishes a run channel under its own name while it
                                         # blocks, so the user can move the run it is waiting on
                                         # to the background mid-flight (see _shared/run_channel.py)


# ── Reversibility: how much of an action can be taken back ────────────────────
# One ordered dimension; approval-gating is derived from it client-side.
#   reversible   the client can undo it (an in-workspace write it snapshotted); not gated.
#   recoverable  undoable by hand: a delete, an environment mutation, a memory wipe.
#   irreversible it leaves the machine or spends something: a submission, a POST.
REVERSIBLE = "reversible"
RECOVERABLE = "recoverable"
IRREVERSIBLE = "irreversible"
REVERSIBILITY_LEVELS = (REVERSIBLE, RECOVERABLE, IRREVERSIBLE)


# ── Work families: the word a tool-activity row shows, and its icon ───────────
# Declared because it cannot always be derived; the row's sentence comes from the model's
# ``doing`` argument.
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
    "DELEGATE", "MAIN_ONLY", "PANEL_REPORT",
]
