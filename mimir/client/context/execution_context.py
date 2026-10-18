from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields
from collections.abc import Iterable, Set as AbstractSet
from typing import Any, Callable

from ..config.limits import (
    DISCOVERY_EVIDENCE_MIN_DISTINCT_PLAN,
    PLAN_EVIDENCE_MIN_FILES_READ,
)
from .bash_classify import Kind, classify_bash_command
from ..config import paths


# Per-query loop-control bookkeeping used only by the tool-dispatch dedup and the
# repeated-failing-call guard: private dedup implementation details, reset fresh each
# query, not semantic discovery/edit/validation state — so kept in one object of its
# own (``ExecutionContext.loop_control``) rather than spread over the context.
@dataclass
class LoopControlState:
    write_calls: set[tuple[Any, ...]] = field(default_factory=set)          # (tool, args) writes already dispatched this query
    call_fails: dict[tuple[Any, ...], int] = field(default_factory=dict)         # (tool, args) -> identical-FAILED dispatch count
    repeat_warned: set[Any] = field(default_factory=set)        # keys the soft repeated-call corrective fired for
    call_results: dict[tuple[Any, ...], tuple[str, int]] = field(default_factory=dict)       # (tool, args) -> (result digest, identical-SUCCESS count)
    repeat_noted: set[Any] = field(default_factory=set)         # keys the identical-success annotation fired for


# ── Recency-preserving sets ────────────────────────────────────────────────────
# A path field shown as a bounded slice should show the paths just touched. A
# ``RecencySet`` is a ``set`` that also keeps insertion order; operators that build a new
# set return a plain one, which ``recent_first`` sorts.
class RecencySet(set[Any]):
    """A ``set`` that also remembers the order elements were first added."""

    __slots__ = ("_order",)

    def __init__(self, iterable: Iterable[Any] = ()) -> None:
        super().__init__()
        self._order: list[Any] = []
        self.update(iterable)

    def add(self, item: Any) -> None:
        if item not in self:
            self._order.append(item)
        super().add(item)

    def update(self, *iterables: Iterable[Any]) -> None:
        for it in iterables:
            for item in it:
                self.add(item)

    def discard(self, item: Any) -> None:
        if item in self:
            try:
                self._order.remove(item)
            except ValueError:
                pass
        super().discard(item)

    def remove(self, item: Any) -> None:
        self.discard(item)

    def clear(self) -> None:
        self._order.clear()
        super().clear()

    def difference_update(self, *iterables: Iterable[Any]) -> None:
        for it in iterables:
            for item in list(it):
                self.discard(item)

    def __isub__(self, other: AbstractSet[object]) -> RecencySet:
        self.difference_update(other)
        return self

    def insertion_order(self) -> list[Any]:
        """Elements oldest-first. Filtered against the set so it cannot drift."""
        return [x for x in self._order if x in self]


def recent_first(value: Any) -> list[Any]:
    """Elements of *value* most-recent-first for a :class:`RecencySet`, else sorted."""
    if isinstance(value, RecencySet):
        return list(reversed(value.insertion_order()))
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return sorted(value or [])


# ── Field traits ───────────────────────────────────────────────────────────────
# What a field is, declared beside it; every derived list below is computed from them.
CARRY = "carry"              # merged into the next query's context (session memory)
FILE_PATH = "file_path"      # holds workspace FILE paths -> a deleted file must be purged
KNOWN_FILE = "known_file"    # holds a file path the model has encountered this session
DISCOVERY = "discovery"      # counts as the model's own discovery evidence

FIELD_TRAITS: frozenset[str] = frozenset({CARRY, FILE_PATH, KNOWN_FILE, DISCOVERY})

# ``inspected_dirs`` is deliberately CARRY + DISCOVERY but NOT FILE_PATH: it holds
# directories, and deleting a file does not un-inspect the directory that contained it.

def _spec(factory: Callable[[], Any], types: tuple[type, ...], *traits: str) -> Any:
    """A context field: its default, the types it may hold, and what it *is* (traits)."""
    return field(default_factory=factory,
                 metadata={"types": types, "traits": frozenset(traits)})


_NoneType = type(None)


@dataclass(eq=False)
class ExecutionContext:
    """Everything one query has established, read by every guard, nudge and report.

    Each field declares its default, types and traits; the derived lists (carried fields,
    path fields, discovery signals) are computed from those declarations.
    """

    # ── Discovery: what the agent has searched, inspected, and read ────────────
    # True once any SEARCH-capability tool has run (discovery-evidence signal)
    searched: bool = _spec(lambda: False, (bool,), DISCOVERY)
    # directories listed/inspected (delete context, discovery-evidence signal)
    inspected_dirs: set[str] = _spec(set, (set,), CARRY, DISCOVERY)
    # paths whose existence was probed via an existence check
    checked_paths: set[str] = _spec(set, (set,), CARRY, FILE_PATH, KNOWN_FILE, DISCOVERY)
    # RecencySet: shown as a bounded slice, where recency is what makes the slice
    # useful (see recent_first).
    # files whose content was read (edit/overwrite precondition; dropped to force a re-read after repeated edit failures)
    read_files: set[str] = _spec(RecencySet, (set,), CARRY, FILE_PATH, KNOWN_FILE, DISCOVERY)
    # Not DISCOVERY: a path's existence can be established as a side effect of the
    # loop's own bookkeeping rather than by the model's own exploration, which is
    # exactly what the discovery gates must not accept as evidence.
    # paths proven to exist on disk (reads, existence hits, candidate matches, written files + their parent dirs)
    existing_paths: set[str] = _spec(RecencySet, (set,), CARRY, FILE_PATH, KNOWN_FILE)
    # Read by a sub-agent the model sent, and reported back with the finding: the
    # model's own discovery, done at arm's length. Kept apart from `read_files`
    # because that field also gates editing, and the model has not read these.
    # files a delegated sub-agent read and reported back; discovery evidence for the caller, never a substitute for read_files
    delegated_read_files: set[str] = _spec(RecencySet, (set,), CARRY, FILE_PATH, KNOWN_FILE, DISCOVERY)
    # per-directory near-name candidates from candidate searches (path clarification)
    similar_candidates_by_dir: dict[str, set[str]] = _spec(dict, (dict,))
    # count of successful search calls (blast-radius gate)
    search_tool_calls: int = _spec(lambda: 0, (int,))
    # Count of successful substantive actions (writes/exec/mutations — any PLAN_BLOCKED
    # tool). Read by the todo nudge so a many-operation task is recognised as
    # multi-step even when it touches only one (or zero) files.
    # count of successful substantive actions (PLAN_BLOCKED: writes/exec/mutations); drives the todo op-count trigger
    action_op_count: int = _spec(lambda: 0, (int,))
    # Empty means "the workspace the client was started in" — the module constant.
    # Set per run, because an agent's root is a property of the agent and a sub-agent
    # may work in a copy of the repository.
    # the tree THIS agent works in; a sub-agent given a copy of the repository declares its paths under that copy
    workspace_root: str = _spec(lambda: "", (str,))
    # ── Edit: planned and in-flight mutations ──────────────────────────────────
    # files the model intends to / has begun editing (union'd with dirty for todo + blast gating)
    planned_edit_targets: set[str] = _spec(RecencySet, (set,))
    # source files named in a task-checklist declaration (multi-file plan completeness)
    declared_edit_set: set[str] = _spec(set, (set,))
    # code files successfully written this query and not yet re-validated
    dirty_written_files: set[str] = _spec(RecencySet, (set,), FILE_PATH)
    # old_text of a confirmed workspace-wide replace, seeding the cross-file completeness check
    cross_file_grep_old_text: str | None = _spec(lambda: None, (str, _NoneType,))
    # file where that workspace-wide replacement originated
    cross_file_grep_source: str | None = _spec(lambda: None, (str, _NoneType,))
    # old_text of the most recent tracked replacement (completeness checking)
    last_replace_old_text: str | None = _spec(lambda: None, (str, _NoneType,))
    # new_text of that most recent tracked replacement
    last_replace_new_text: str = _spec(lambda: "", (str,))
    # file the most recent tracked replacement targeted
    last_replace_file: str = _spec(lambda: "", (str,))
    # path -> (last_signature: str|None, identical consecutive failures: int); guards the identical-patch hard block
    edit_loop_state: dict[str, tuple[Any, ...]] = _spec(dict, (dict,))
    # path -> consecutive edit failures regardless of patch (force-re-read + error_recovery)
    edit_fail_streak_by_file: dict[str, int] = _spec(dict, (dict,))
    # agent-loop steps since the last successful edit (idle gating)
    steps_since_last_edit: int = _spec(lambda: 0, (int,))
    # path of last successful code edit (cleared after injection)
    last_edit_success_path: str = _spec(lambda: "", (str,))
    # ── Validation ─────────────────────────────────────────────────────────────
    # dirty files that have since passed a checker (syntax/imports/lint) — never an execution
    validated_files: set[str] = _spec(set, (set,), FILE_PATH)
    # per-file strength of that validation: structural|syntax|static|compiled (completion ledger)
    validation_tier_by_file: dict[str, str] = _spec(dict, (dict,))
    # per-file count of failed checks (retry-budget exhaustion)
    validation_fail_count_by_file: dict[str, int] = _spec(dict, (dict,))
    # dirty files the built-in floor cannot read at all (binary, undecodable): reported, never demanded
    unverifiable_files: set[str] = _spec(set, (set,), FILE_PATH)
    # path -> (mtime_ns, size) at its last built-in check: what makes the conclusion sweep check each revision once
    builtin_check_stamp: dict[str, tuple[int, int]] = _spec(dict, (dict,))
    # path -> the built-in floor's diagnostic for a file it rejected (drives the validation nudge)
    builtin_check_findings: dict[str, str] = _spec(dict, (dict,))
    # command -> {call_id, completed: bool, effect: run|build, verdict: ''|pass|fail|unknown, reason, failures: int, attempts: [str]}: everything about an execution, which is judged on its own and credits no file
    runs: dict[str, dict[str, Any]] = _spec(dict, (dict,))
    # True once the model answered the run/verdict advice; the shared exercise budget never re-arms after that
    exercise_advice_closed: bool = _spec(lambda: False, (bool,))
    # why running the change was out of reach (no exec tool, unresolved imports, needs a build): reported in the ledger instead of silently dropping the advice
    exercise_blocked_reason: str = _spec(lambda: "", (str,))
    # True once the model was handed the completion ledger to rewrite its summary (once per query)
    evidence_handback_used: bool = _spec(lambda: False, (bool,))
    # ── Workflow + policy ──────────────────────────────────────────────────────
    # coarse phase of the loop: "discover" | "edit" | "validate" | "conclude"
    workflow_state: str = _spec(lambda: "discover", (str,))
    # True once the first code file was successfully edited this query
    code_mutation_started: bool = _spec(lambda: False, (bool,))
    # per-category nudge fire counts (frequency caps). keys: discovery, validation, denial, state, creation, blast_radius, doc, todo, error_recovery, regression
    nudge_counts: dict[str, int] = _spec(dict, (dict,))
    # tool calls blocked by policy/approval (denial nudge + finalization)
    denied_tool_calls: list[dict[str, Any]] = _spec(list, (list,))
    # append-only refusal log keyed by approval scope; drives the escalation ladder (never cleared, unlike denied_tool_calls)
    denial_history: list[dict[str, Any]] = _spec(list, (list,))
    # files passed to a tests-kind validator this query (regression guard)
    tests_run: set[str] = _spec(set, (set,))
    # test file -> distinct failing-node-id sets seen this query (oracle nudge)
    test_failure_signatures: dict[str, list[str]] = _spec(dict, (dict,))
    # ── Environment ────────────────────────────────────────────────────────────
    # imports a run failed to resolve (env-resolution nudge)
    unresolved_modules: set[str] = _spec(set, (set,))
    # ids a scheduler listed that no descriptor covers — nothing watches them, no wake is owed (unwatched-run nudge)
    unwatched_runs: set[str] = _spec(set, (set,))
    # True once the model enumerated the available interpreters/envs
    env_probed: bool = _spec(lambda: False, (bool,))
    # installs / env creations recorded this query (conclude-phase cleanup nudge)
    env_mutations: list[dict[str, Any]] = _spec(list, (list,))
    # ── Todo / planning ────────────────────────────────────────────────────────
    # True once an ordered checklist (plan_steps) was recorded
    todo_written: bool = _spec(lambda: False, (bool,))
    # True once a prose plan/rationale was recorded
    plan_written: bool = _spec(lambda: False, (bool,))
    # absolute path to the active todo_list.md (set at query start)
    todo_file_path: str = _spec(lambda: "", (str,))
    # ── Cross-query and loop bookkeeping ───────────────────────────────────────
    # These were live without being declared: written by the loop, the dispatcher or
    # the carry merge, and read by the history budget — but invisible to the template,
    # to validate_execution_context and to the trait derivations.
    # tool_call_id -> files that tool message concerns; lets history trimming match messages to files structurally
    tool_msg_files: dict[str, list[str]] = _spec(dict, (dict,))
    # consecutive bare final-answer turns with no tool call (nudge cutoff)
    consecutive_noop_turns: int = _spec(lambda: 0, (int,))
    # the context backstop dropped older content this query; a cached read may no longer be "above in your context"
    history_truncated: bool = _spec(lambda: False, (bool,))
    # ── Skills loaded on demand ────────────────────────────────────────────────
    # Insertion order is the load order; the instructions live in the messages and in
    # agent.skills. Mirrored into carry_context by hand, like ``searched``.
    # skill name -> tool_call_id carrying its instructions ("" = folded into messages[0] by /<name>, "pending" = pull in flight)
    skills_loaded: dict[str, str] = _spec(dict, (dict,))
    # tool_call_ids whose results the history budget may not evict or summarise away
    # while anything else is reducible. The one pin mechanism; see history.py. CARRY
    # because the pinned message is archived and replayed into the next query, so its
    # protection has to outlive the query too.
    # tool_call_ids the history budget may not evict or summarise while anything else is reducible
    pinned_call_ids: set[str] = _spec(set, (set,), CARRY)

    # ── Loop bookkeeping: state the loop and the correctives keep between two steps ──
    # The per-query dedup and spin guards (see LoopControlState).
    loop_control: LoopControlState = field(default_factory=LoopControlState)
    # The agent's mode as last seen, so only a *change* reads as the user switching
    # (loop_base._live_mode); None until the query starts.
    observed_agent_mode: str | None = None
    # The /<skill> block folded into the system message, put back on every rebuild.
    skill_suffix: str = ""
    # mtime of the checklist file the system message currently renders.
    checklist_stamp: int | None = None
    # One-shot correctives staged during a step and consumed by the next injection.
    repeat_alert: tuple[Any, ...] | None = None          # (tool, failures) of a repeated failing call
    moving_test_alert: str | None = None        # a test file whose failing set keeps moving
    moving_test_told: list[Any] = field(default_factory=list)   # test files already told about
    handback_told: bool = False                 # the end of the denial ladder was said
    last_todo_tick_path: str = ""               # the file the last "tick the step" was about
    # Reminders injected for the next call only, and the categories of the last call's.
    transient_reminders: list[Any] = field(default_factory=list)
    last_call_reminders: list[Any] = field(default_factory=list)

    def take(self, name: str) -> Any:
        """Read a one-shot field and reset it to its default: the staged correctives
        are consumed by whoever delivers them."""
        value = getattr(self, name)
        spec = self.__dataclass_fields__[name]
        setattr(self, name, spec.default_factory() if spec.default_factory is not MISSING
                else spec.default)
        return value



# ── The carry-context schema ───────────────────────────────────────────────────
# Keys that exist only in the carry context (``context.carry``), declared so the JSON
# conversion and the deleted-path purge are derived.
#   (name, default_factory, json_kind, holds_paths)
_CarryExtraSpec = tuple[str, Callable[[], Any], str, bool]

_CARRY_EXTRA_SPECS: tuple[_CarryExtraSpec, ...] = (
    ("searched", lambda: False, "scalar", False),
    # Reported by a sub-agent's hand-back (see server_spawn_agent).
    ("last_query_written_files", set, "set", True),
    # path -> mtime at the time of the read; used to evict reads a later edit staled.
    ("_read_mtimes", dict, "dict", False),
)


def carry_set_fields() -> tuple[str, ...]:
    """Carry keys holding a ``set`` — the ones needing list⇄set conversion for JSON."""
    return fields_with(CARRY) + tuple(
        name for name, _f, kind, _p in _CARRY_EXTRA_SPECS if kind == "set"
    )


def carry_path_fields() -> tuple[str, ...]:
    """Carry keys holding workspace file paths — purged when a file is deleted."""
    return fields_with(CARRY, FILE_PATH) + tuple(
        name for name, _f, _k, holds_paths in _CARRY_EXTRA_SPECS if holds_paths
    )


def carry_context_to_json(carry: dict[str, Any]) -> dict[str, Any]:
    """Return *carry* with every set rendered as a sorted list, ready for ``json.dump``.

    Schema-driven for the declared keys, then a defensive sweep for anything else: the
    session store serialises with ``default=str``, which turns an unconverted set into
    a string instead of failing, so a missed key corrupts silently rather than loudly.
    """
    out = dict(carry)
    for key in carry_set_fields():
        if isinstance(out.get(key), set):
            out[key] = sorted(out[key])
    for key, value in out.items():  # backstop for any undeclared key
        if isinstance(value, (set, frozenset)):
            out[key] = sorted(value)
    return out


def carry_context_from_json(carry: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`carry_context_to_json`: restore declared set fields."""
    out = dict(carry)
    for key in carry_set_fields():
        if key in out and not isinstance(out[key], set):
            value = out[key]
            out[key] = set(value) if isinstance(value, (list, tuple, set)) else set()
    return out


def fields_with(*traits: str) -> tuple[str, ...]:
    """Field names carrying **every** trait in *traits*, in declaration order.

    The derivation that replaces the hand-maintained name lists. Declaration order is
    stable and meaningful (fields are grouped by concern), so callers that render or
    iterate get a deterministic result without sorting.
    """
    wanted = frozenset(traits)
    return tuple(f.name for f in fields(ExecutionContext)
                 if wanted <= f.metadata.get("traits", frozenset()))


def was_read(execution_context: ExecutionContext, path: str) -> bool:
    """Whether *path* was read through a direct read tool this query.

    Says nothing about *how much* of it was read, and deliberately so: reads are
    targeted, a capped window is the normal case, and a gate that asked for the whole
    file would be asking for the one thing the read policy tells the model not to do.
    """
    return path in (execution_context.read_files or set())


def is_known_to_exist(execution_context: ExecutionContext, path: str) -> bool:
    """Whether *path* is proven to exist — a successful read or an existence hit.

    Stronger than :func:`was_checked_for`, which only records that a look happened.
    """
    return (
        path in (execution_context.existing_paths or set())
        or was_read(execution_context, path)
    )


def was_checked_for(execution_context: ExecutionContext, path: str) -> bool:
    """Whether a pre-check was attempted for *path* — not proof that it exists."""
    return path in (execution_context.checked_paths or set())


def known_existing_files(execution_context: ExecutionContext) -> set[str]:
    """Every file path the model has encountered this session (proof they exist).

    One implementation for a question that had two: the policy engine's path-clarification
    lookup and the nudge engine's ``_known_existing_files`` walked the same four fields
    with the same intent, in two modules, with nothing keeping them equal.
    """
    known: set[str] = set()
    for name in fields_with(KNOWN_FILE):
        value = getattr(execution_context, name)
        if isinstance(value, set):
            known.update(str(p) for p in value if p)
    return known


# ── State producer → consumer map ──────────────────────────────────────────────
# Each field group has one writer: see CLIENT_DETAILED.md, "Who writes what". Never
# mutate the context outside its declared writer.


# ── Discovery evidence: what counts as the model's own exploration this query ──
# Every consumer reads these helpers. Only the model's own tool calls set the signals,
# so presence is the whole test.

# Derived from the ``DISCOVERY`` trait rather than hand-listed, so a new discovery
# field cannot be added to the schema and forgotten here. Membership:
# searched, read_files, checked_paths, inspected_dirs, delegated_read_files.
DISCOVERY_EVIDENCE_SIGNALS: tuple[str, ...] = fields_with(DISCOVERY)


def discovery_signal_count(execution_context: ExecutionContext) -> int:
    """Number of distinct model-initiated discovery signals present in *execution_context*."""
    return sum(1 for s in DISCOVERY_EVIDENCE_SIGNALS
               if getattr(execution_context, s))


def nudge_count(execution_context: ExecutionContext, category: str) -> int:
    """How many times the *category* nudge has fired this query (0 if never).

    Named accessor for the ``execution_context.nudge_counts.get(category, 0)``
    pattern that the per-category frequency caps repeat across every nudge branch.
    """
    return execution_context.nudge_counts.get(category, 0)


def idle_steps(execution_context: ExecutionContext) -> int:
    """Steps since the last successful edit (large sentinel when none yet).

    Named accessor for ``int(execution_context.steps_since_last_edit)``;
    the 99 sentinel means "the model has not edited anything, so it is maximally
    idle" — used by the validation/state idle gates.
    """
    return int(execution_context.steps_since_last_edit)


def has_discovery_evidence(execution_context: ExecutionContext, *, min_distinct: int = 1) -> bool:
    """True once the model has done at least *min_distinct* distinct discovery actions."""
    return discovery_signal_count(execution_context) >= min_distinct


def plan_evidence_ready(
    execution_context: ExecutionContext,
    *,
    min_files: int = PLAN_EVIDENCE_MIN_FILES_READ,
    min_distinct: int = DISCOVERY_EVIDENCE_MIN_DISTINCT_PLAN,
) -> bool:
    """True once the model has read code, not merely located it: the bar for the plan tool."""
    read = set(execution_context.read_files or ())
    # A file a sub-agent read and reported on counts here: the plan's "WHERE" rests on
    # code that was read and whose finding is in the model's context. It does NOT count
    # toward the pre-edit gate, which asks whether *this* agent holds the lines.
    read |= set(execution_context.delegated_read_files or ())
    return (
        len(read) >= min_files
        and has_discovery_evidence(execution_context, min_distinct=min_distinct)
    )


def _declared_target_written(declared: str, written: set[str], root: str) -> bool:
    """True when the declared edit target *declared* is one of the *written* paths.

    Both sides are resolved against the workspace root; a declared entry with no
    directory matches on basename.
    """
    if declared in written:
        return True
    base = os.path.basename(declared)
    same_name = [w for w in written if os.path.basename(w) == base]
    if not same_name:
        return False
    if not os.path.dirname(declared):
        return True
    target = os.path.abspath(os.path.join(root, declared))
    return any(os.path.abspath(os.path.join(root, w)) == target for w in same_name)


def unwritten_declared_files(execution_context: ExecutionContext) -> list[str]:
    """Declared edit targets no write of this query honoured, sorted.

    Guidance mid-run only: the set is scraped from checklist prose.
    """
    declared = execution_context.declared_edit_set or set()
    if not declared:
        return []
    dirty = set(execution_context.dirty_written_files or set())
    # Off the context, not off the import: a sub-agent in a copy of the repository
    # declares paths under its own root, and this function never sees the agent.
    root = os.path.abspath(execution_context.workspace_root or paths.WORKSPACE_ROOT)
    return sorted(d for d in declared if not _declared_target_written(d, dirty, root))


def declared_edit_set_complete(execution_context: ExecutionContext) -> bool:
    """True when every file the model declared it would edit has been written (vacuous if none)."""
    return not unwritten_declared_files(execution_context)


# ── Validation strength: what a green check proved ─────────────────────────────
# The checker ladder, weakest to strongest, for a file (a run is judged as a run, see
# ``runs``). Report-only: the completion ledger reads it. ``structural`` is the built-in
# floor every checked file starts at; ``measured`` only comes from a server that ran the
# file and knows which. See CLIENT_DETAILED.md, "Two axes, never mixed".
VALIDATION_TIERS: tuple[str, ...] = ("structural", "syntax", "static", "compiled", "measured")
_TIER_RANK = {name: i for i, name in enumerate(VALIDATION_TIERS)}


def validation_tier(execution_context: ExecutionContext, path: str) -> str | None:
    """The strength of *path*'s validation, or None if it was never validated."""
    return execution_context.validation_tier_by_file.get(path)


def raise_validation_tier(execution_context: ExecutionContext, path: str, tier: str) -> None:
    """Record *tier* for *path*, keeping the strongest evidence seen.

    Cleared, never lowered, wherever ``validated_files`` drops the path.
    """
    if tier not in _TIER_RANK:
        return
    tiers = execution_context.validation_tier_by_file
    current = tiers.get(path)
    if current is None or _TIER_RANK[tier] > _TIER_RANK[current]:
        tiers[path] = tier


def weakest_validation_tier(
    execution_context: ExecutionContext, paths: Iterable[str],
) -> str | None:
    """The lowest tier across *paths*, or None if none of them was validated.

    The honest summary of a multi-file change: a set is only as well established
    as its least-checked member.
    """
    ranks = [
        _TIER_RANK[t]
        for t in (validation_tier(execution_context, p) for p in paths)
        if t is not None
    ]
    return VALIDATION_TIERS[min(ranks)] if ranks else None


def files_below_tier(execution_context: ExecutionContext, tier: str) -> list[str]:
    """Validated files whose evidence is weaker than *tier* (sorted).

    Unvalidated files are excluded — they are already reported as pending, and
    listing them here would double-count them in the ledger.
    """
    threshold = _TIER_RANK.get(tier)
    if threshold is None:
        return []
    tiers = execution_context.validation_tier_by_file
    return sorted(p for p, t in tiers.items() if _TIER_RANK.get(t, 0) < threshold)


# ── Output verdicts: what the model said a run's output showed ─────────────────
# A claim about a run, kept apart from the file evidence above. ``blocked`` re-imputes a
# failure to the environment; ``rejected`` is a run that measured correctly without
# improving, which must not charge the repair budget the way ``fail`` does.
VERDICTS: tuple[str, ...] = ("pass", "fail", "unknown", "blocked", "rejected")


def run_ledger_key(command: str) -> str:
    """The identity a run is remembered by: the executing head plus the files it named.

    Flags and everything after a pipe are dropped, so retries of one failure share a key
    and the repair budget accumulates. A segment that named no file keeps its argv; an
    unreadable command keys on its text. Idempotent.
    """
    try:
        segments = classify_bash_command(command, allow_expansion=True) or []
    except Exception:  # pragma: no cover - the classifier owns its own refusals
        segments = []
    parts: list[str] = []
    for seg in segments:
        if seg.kind not in (Kind.EXEC, Kind.UNKNOWN):
            continue
        head = seg.head or (seg.argv[0] if seg.argv else "")
        if not head:
            continue
        if seg.operands:
            parts.append(" ".join([head, *sorted(seg.operands)]).strip())
        else:
            parts.append(" ".join(seg.argv).strip() or head)
    return " && ".join(parts) if parts else command


def record_run(
    execution_context: ExecutionContext, command: str, *, completed: bool, call_id: str = "",
    effect: str = "run",
) -> dict[str, Any]:
    """Register an execution, or re-register one run again, last and with its failure history.

    Keyed by :func:`run_ledger_key`; ``command`` keeps the text a report shows. ``effect``
    tells a build from a run.
    """
    runs = execution_context.runs
    key = run_ledger_key(command)
    previous = runs.pop(key, None) or {}
    run = {
        "call_id": call_id,
        "command": command,
        "completed": completed,
        "effect": effect,
        "verdict": "",
        "reason": "",
        "failures": int(previous.get("failures", 0)),
        "attempts": list(previous.get("attempts") or ()),
        # Not carried over from `previous`, unlike the two above: a re-run is a fresh
        # attempt at the wall, so re-running retracts the blocked standing by itself.
        "blocked": "",
    }
    runs[key] = run
    return run


def unsettled_runs(execution_context: ExecutionContext) -> dict[str, dict[str, Any]]:
    """Runs that reached their end and still owe a reading of what they showed.

    A run that did not complete owes nothing: its non-zero exit *is* the finding, and
    asking the model to judge output that never came would be asking for a guess. A run
    judged ``unknown`` stays here — "I cannot tell" is not a settlement.
    """
    return {
        command: run for command, run in (execution_context.runs or {}).items()
        if run.get("completed") and run.get("verdict") in ("", "unknown")
    }


#: What a run's record says about it, most actionable first.
RUN_CLASSES: tuple[str, ...] = ("blocked", "broken", "unknown", "pass", "unjudged")


def run_class(run: dict[str, Any]) -> str:
    """Which of :data:`RUN_CLASSES` *run* falls in.

    Blocked (a wall of this environment) comes first; broken is a run that did not
    reach its end or was judged wrong; any other verdict reads as pass.
    """
    if run.get("blocked"):
        return "blocked"
    if not run.get("completed") or run.get("verdict") == "fail":
        return "broken"
    verdict = run.get("verdict")
    if verdict == "unknown":
        return "unknown"
    return "pass" if verdict else "unjudged"


def failed_runs(execution_context: ExecutionContext) -> dict[str, dict[str, Any]]:
    """Runs that did not reach their end, or whose output the model judged wrong.

    A blocked run is excluded: it charges no repair budget, so leaving it here would hold
    the "every failure is spent" release below its threshold for the rest of the query.
    """
    return {
        command: run for command, run in (execution_context.runs or {}).items()
        if run_class(run) == "broken"
    }


def build_execution_context(**values: Any) -> ExecutionContext:
    """A fresh context, with *values* set over the defaults."""
    return ExecutionContext(**values)


def validate_execution_context(execution_context: ExecutionContext) -> None:
    """Fail fast if the runtime policy contract is broken."""
    for f in fields(execution_context):
        types = f.metadata.get("types")
        value = getattr(execution_context, f.name)
        if types and not isinstance(value, types):
            type_names = ", ".join(tp.__name__ for tp in types)
            raise TypeError(
                f"Invalid execution_context.{f.name} type: {type(value).__name__}; "
                f"expected one of: {type_names}"
            )

    for parent, candidates in execution_context.similar_candidates_by_dir.items():
        if not isinstance(parent, str):
            raise TypeError("execution_context.similar_candidates_by_dir keys must be str")
        if not isinstance(candidates, set):
            raise TypeError("execution_context.similar_candidates_by_dir values must be set[str]")

    for target in execution_context.planned_edit_targets:
        if not isinstance(target, str):
            raise TypeError("execution_context.planned_edit_targets values must be str")

    for key, count in execution_context.validation_fail_count_by_file.items():
        if not isinstance(key, str):
            raise TypeError("execution_context.validation_fail_count_by_file keys must be str")
        if not isinstance(count, int):
            raise TypeError("execution_context.validation_fail_count_by_file values must be int")

    for key, tier in execution_context.validation_tier_by_file.items():
        if not isinstance(key, str):
            raise TypeError("execution_context.validation_tier_by_file keys must be str")
        if tier not in VALIDATION_TIERS:
            raise ValueError(
                f"execution_context.validation_tier_by_file[{key!r}] must be one of "
                f"{VALIDATION_TIERS}; got {tier!r}"
            )

    for key, run in execution_context.runs.items():
        if not isinstance(key, str):
            raise TypeError("execution_context.runs keys must be str")
        if not isinstance(run, dict) or not isinstance(run.get("attempts"), list):
            raise TypeError("execution_context.runs values must be dict with an 'attempts' list")
        if run.get("verdict") not in ("",) + VERDICTS:
            raise ValueError(
                f"execution_context.runs[{key!r}]['verdict'] must be '' or one of "
                f"{VERDICTS}; got {run.get('verdict')!r}"
            )


def ensure_execution_context(execution_context: ExecutionContext | None) -> ExecutionContext | None:
    """*execution_context*, checked to be one. A tool run outside a query has none."""
    if execution_context is not None and not isinstance(execution_context, ExecutionContext):
        raise TypeError(f"expected an ExecutionContext, got {type(execution_context).__name__}")
    return execution_context
