from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields
from collections.abc import Iterable
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
    write_calls: set = field(default_factory=set)          # (tool, args) writes already dispatched this query
    call_fails: dict = field(default_factory=dict)         # (tool, args) -> identical-FAILED dispatch count
    repeat_warned: set = field(default_factory=set)        # keys the soft repeated-call corrective fired for
    call_results: dict = field(default_factory=dict)       # (tool, args) -> (result digest, identical-SUCCESS count)
    repeat_noted: set = field(default_factory=set)         # keys the identical-success annotation fired for


# ── Recency-preserving sets ────────────────────────────────────────────────────
#
# Several path fields can only ever be shown as a bounded slice — a summary, a nudge,
# a hand-back. A plain ``set`` carries no order, so the only deterministic slice is an
# alphabetical one, and the alphabetically-last ten paths are not the ten that matter.
# ``RecencySet`` keeps insertion order alongside the set so a slice can be the paths
# the model just touched.
#
# It IS a ``set``: every existing membership test, ``|``, ``-`` and ``isinstance``
# check keeps working unchanged. Operators that build a new set return a plain one and
# simply lose the order, which ``recent_first`` degrades to sorted order for — the
# previous behaviour, never a crash.
class RecencySet(set):
    """A ``set`` that also remembers the order elements were first added."""

    __slots__ = ("_order",)

    def __init__(self, iterable=()):
        super().__init__()
        self._order: list = []
        self.update(iterable)

    def add(self, item) -> None:
        if item not in self:
            self._order.append(item)
        super().add(item)

    def update(self, *iterables) -> None:  # type: ignore[override]
        for it in iterables:
            for item in it:
                self.add(item)

    def discard(self, item) -> None:
        if item in self:
            try:
                self._order.remove(item)
            except ValueError:
                pass
        super().discard(item)

    def remove(self, item) -> None:
        self.discard(item)

    def clear(self) -> None:
        self._order.clear()
        super().clear()

    def difference_update(self, *iterables) -> None:  # type: ignore[override]
        for it in iterables:
            for item in list(it):
                self.discard(item)

    def __isub__(self, other):  # type: ignore[override]
        self.difference_update(other)
        return self

    def insertion_order(self) -> list:
        """Elements oldest-first. Filtered against the set so it cannot drift."""
        return [x for x in self._order if x in self]


def recent_first(value) -> list:
    """Elements of *value* most-recent-first, falling back to sorted order.

    The pin shows a bounded slice, so which elements it picks is a real decision. A
    :class:`RecencySet` answers it with recency; anything else (a plain set rebuilt by
    ``|``, a list restored from JSON) has no such answer and gets deterministic
    alphabetical order rather than an arbitrary one.
    """
    if isinstance(value, RecencySet):
        return list(reversed(value.insertion_order()))
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return sorted(value or [])


# ── Field traits ───────────────────────────────────────────────────────────────
#
# What a field *is*, declared next to the field itself, so the question is asked once,
# where the field is defined. Every derived list below is computed from this table —
# hand-maintained name lists in agent_core, observations, the policy engine and the nudge
# engine have nothing keeping them correct when a field is added, and go stale silently or
# compute the same thing twice.
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

    One object per query, built by the agent (``new_execution_context``) and handed to
    every layer. Each field declares its default, the types it may hold and its traits
    — what the field *is* — and every derived list (the carried fields, the path fields
    a deletion purges, the discovery signals) is computed from those declarations, so a
    new field is declared once, here.
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
    edit_loop_state: dict[str, tuple] = _spec(dict, (dict,))
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
    # These three were live for a long time without being declared here at all: seeded
    # by a bootstrap helper, written by the observers, read by the nudges — and so
    # invisible to the template, to validate_execution_context, and to the TypedDict
    # parity test. Declaring them is what lets a backfill be derived from this table
    # instead of hand-listed per module.
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
    # skill name -> the tool_call_id of the result carrying its instructions; "" for
    # one the user's /<name> folded into messages[0], "pending" while a pull is in
    # flight (the gate reserves the slot so two parallel pulls cannot both pass the
    # cap). Insertion order is the load order. No body is copied in: it lives in the
    # message list and in agent.skills.
    # Mirrored into carry_context by hand, like ``searched``: the trait-driven mirror
    # only handles sets, and the instructions outlive the query that pulled them.
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
    repeat_alert: tuple | None = None          # (tool, failures) of a repeated failing call
    moving_test_alert: str | None = None        # a test file whose failing set keeps moving
    moving_test_told: list = field(default_factory=list)   # test files already told about
    handback_told: bool = False                 # the end of the denial ladder was said
    last_todo_tick_path: str = ""               # the file the last "tick the step" was about
    # Reminders injected for the next call only, and the categories of the last call's.
    transient_reminders: list = field(default_factory=list)
    last_call_reminders: list = field(default_factory=list)

    def take(self, name: str) -> Any:
        """Read a one-shot field and reset it to its default: the staged correctives
        are consumed by whoever delivers them."""
        value = getattr(self, name)
        spec = self.__dataclass_fields__[name]
        setattr(self, name, spec.default_factory() if spec.default_factory is not MISSING
                else spec.default)
        return value



# ── The carry-context schema ───────────────────────────────────────────────────
#
# ``carry_context`` is the session-level dict that survives between queries. Most of it
# mirrors the CARRY-trait fields above, but three keys exist only there — and because
# they were never declared anywhere, ``export_state`` did not know to normalise them
# for JSON. ``last_query_written_files`` is a ``set``: the session store serialises with
# ``default=str``, so it round-tripped as the *string* "{'a.py', 'b.py'}", and the next
# ``apply_carry_context`` did ``set(...)`` on that string, exploding it into single
# characters that the pin then rendered as file paths.
#
# Declaring the extras here is what lets the JSON conversion and the deleted-path purge
# be derived instead of hand-listed, so the next key added cannot repeat the bug.
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
    """Whether a pre-check was *attempted* for *path*.

    Explicitly **not** proof that it exists: a check that came back negative lands here
    too. Kept separate from :func:`is_known_to_exist` because the destructive guards
    need the stronger fact, and a name is the only thing that keeps the two apart at
    the call site.
    """
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
# ExecutionContext is a large, mutable, shared blackboard. The single biggest source
# of cognitive load in the policy layer is that reasoning about one nudge or guard
# requires knowing *who writes* the fields it reads. This map is the authoritative
# answer. It is grouped by writer; every reader is a consumer of the corresponding
# field group. Keep it in sync when a field's writer/reader set changes.
#
#   WRITER                                   FIELDS WRITTEN                              READ BY
#   guardrails.observations.record_tool_observation   searched, read_files, inspected_dirs,      engine (tools_for_context,
#     (the ordered _observe_* handlers,        checked_paths, existing_paths,             _missing_evidence, guards),
#      run after every tool call)              search_tool_calls,                          runtime (check_write_policy),
#                                              similar_candidates_by_dir,                 state_machine (guards),
#                                              dirty_written_files, declared_edit_set,    nudge_logic (all nudges),
#                                              edit_loop_state, edit_fail_streak_by_file,  agent_loop (correctives)
#                                              last_replace_*,
#                                              code_mutation_started, validated_files,
#                                              validation_tier_by_file,
#                                              validation_fail_count_by_file, tests_run,
#                                              runs,
#                                              action_op_count,
#                                              todo_written, plan_written
#   guardrails.verdict.apply_verdict           runs[cmd].verdict/reason/failures,        verification (ledger rows),
#     (runs on the judging tool's call,        attempts                                   workflow (completion report)
#      via _observe_verdict_tool)
#   context/workflow.set_workflow_state       workflow_state                             every layer's state gates
#   query_engine/nudge_logic._fire_nudge      nudge_counts[<category>]                    nudge_logic (per-category caps)
#   query_engine/agent_loop                   steps_since_last_edit (increment),          nudge_logic (idle gates),
#     (loop body + loop-control correctives)    LoopControlState (the private             agent_loop (dedup +
#                                              _loop_control object: write_calls,          repeated-failure guard)
#                                              call_fails, repeat_warned),
#                                              evidence_handback_used                      agent_loop (once per query)
#   policy/engine + agent_core (approval)  denied_tool_calls, denial_history          state_machine (blocking denials),
#                                                                                         nudge_logic (denial nudge),
#                                                                                         workflow (escalation ladder)
#
# Rule of thumb: fields are WRITTEN in exactly one place (the observation pass or the
# loop) and READ in many. Never mutate context outside its declared writer above: a second
# writer is how the layers come to disagree about what the context says.


# ── Discovery-evidence semantics: the single definition of "what counts as the
# model's own exploration this query." Every consumer (the agent-mode discovery
# nudge, the plan-mode explore phase, engine._missing_evidence) reads these helpers
# instead of hand-picking its own subset of fields — that duplicated subset-picking
# was the source of cross-layer coherence bugs.
#
# Every signal below is only ever set by the model's own tool calls this query (see
# guardrails.observations._observe_*), so presence is the whole test. Nothing seeds
# these fields any more: a structural snapshot used to pre-fill ``inspected_dirs``,
# which meant a gate could be satisfied before the model had done anything, and the
# discount that existed to undo it was itself a field only some consumers applied.

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
    """True once the model has actually read code, not merely located it.

    The bar plan mode holds the plan-document tool behind. Distinct signal *kinds*
    alone do not express it: listing a directory and running a find scores two while
    grounding nothing, which is how a plan could be written over file names. Reading
    file contents is what a plan's "WHERE" rests on, so the read count carries the
    gate and the kind count stays as the breadth term beside it.
    """
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
    """True when one declared edit target was honoured by one of the *written* paths.

    The two sides are spelled differently by construction. A write records the path
    the file tools were given — absolute outside the workspace, root-relative inside
    — while the declared set is scraped from the checklist's own prose, where the
    model names the file however it reads best: bare (``write wave_solver_2d.py in
    ../other/``, the directory living in the surrounding words) or relative to the
    workspace. So the comparison resolves both sides against the workspace root, and
    a declared entry carrying no directory at all matches on basename — a bare
    mention never carried a location to contradict.
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
    """Declared edit targets that no write this query honoured, sorted.

    Read while the work is still going — the edit→validate transition, the empty-turn
    corrective, the missing-evidence text — and no longer at completion time. The end-of
    -run readers (completion issues, residual risk, the ledger) were dropped: the
    declared set is scraped from checklist prose and replaced only when a new checklist
    is written, so what it reports is "named earlier, not written yet", which is useful
    guidance mid-run and an accusation at the end, where a revised plan is
    indistinguishable from a skipped step.

    Matching is by resolved path, not by string (see :func:`_declared_target_written`) —
    a raw set difference reported a file as never written whenever the plan spelled it
    differently from the write.
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
    """True when every file the model declared it would edit has been dirtied.

    The single definition of "the planned multi-file refactor is finished writing"
    — read by the edit→validate transition (runtime), the validation state guard
    (state_machine), and the validation/state nudges (nudge_logic). An empty
    declared set (no explicit plan) is vacuously complete.
    """
    return not unwritten_declared_files(execution_context)


# ── Validation strength: how much a green check actually proved ───────────────
#
# ``validated_files`` answers "was this file checked?"; this ladder answers "with
# what?". Both are about **checkers** only — a parse, an import resolution, a lint,
# a compile: tools whose output *is* a list of problems, so an empty one is the
# finding and nobody has to read anything. Running the code is not on this ladder at
# all; an execution is judged as a run (see ``runs`` and ``guardrails/verdict.py``),
# because "it reached the end" and "it gave the right answer" are different claims
# and only the second one is what the user asked for.
#
# Ordered weakest→strongest. Report-only: nothing gates, blocks, or nudges on the
# tier — it is read by the completion ledger so the *answer* can say what was
# actually established.
#
# ``structural`` sits at the bottom: the built-in floor established that the file parses
# (or, for a language with no stdlib parser, that it is not truncated) without any tool
# being installed. It is the tier every checked file starts at, and any external checker
# the model happens to run raises it from there.
#
# ``compiled`` needs a toolchain the environment may not have, so a build is
# recommended when it is available, never required. It still credits the file, a
# fortiori — a translation unit that linked also parsed.
#
# ``measured`` sits at the top and is reached by exactly one route: a server that ran
# the file itself and knows which file it ran. Attribution is what normally keeps a run
# off this axis — ``python main.py`` exercises ``mesh.py`` without naming it — and an
# optimisation session is the one place where it is not a guess, because the session
# config names a single source. It is demanded nowhere either: a cluster, a queue or a
# dataset may be missing. It says the file was exercised and measured, never that the
# measurement was good; that claim is a verdict on the *run*.
VALIDATION_TIERS: tuple[str, ...] = ("structural", "syntax", "static", "compiled", "measured")
_TIER_RANK = {name: i for i, name in enumerate(VALIDATION_TIERS)}


def validation_tier(execution_context: ExecutionContext, path: str) -> str | None:
    """The strength of *path*'s validation, or None if it was never validated."""
    return execution_context.validation_tier_by_file.get(path)


def raise_validation_tier(execution_context: ExecutionContext, path: str, tier: str) -> None:
    """Record *tier* for *path*, keeping the strongest evidence seen.

    Monotone by design: a file validated by a reference comparison and then
    re-checked with a bare syntax pass has still been compared. The tier is
    cleared (not lowered) when the file is re-edited or fails validation —
    wherever ``validated_files`` is discarded, this is popped alongside it.
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


# ── Output verdicts: what the MODEL said a run's output showed ────────────────
#
# The tier ladder above grades machine-observed evidence about a *file*; this grades
# nothing — it records a claim about a *run*. Exit 0 only ever means the program
# reached its end, and no parser generalises across plots, tables, logs and physical
# units, so the only thing that can read arbitrary output is the model. Mimir
# therefore never parses the *program's* output for a pass/fail; it parses the model's
# own statement about it, which is a format the model controls. Kept on the run rather
# than on a file, and rendered under its own heading in the ledger, so a claim is never
# mistaken for an observation and "the file lints" is never read as "the answer is right".
# ``blocked`` grades nothing either, and settles nothing: it re-imputes a red exit from
# the change to the environment, leaving the run as red as the machine saw it.
#
# ``rejected`` is the fifth because four were not enough to describe an experiment. In a
# search loop most runs lose, and losing is not failing: the program ran, the number was
# read, the candidate was simply not better. With no word for that the model reached for
# ``fail`` — recorded doing exactly this: `fail`, "OMP_NUM_THREADS=64 rejected:
# time_s=0.064247 > best 0.055586" — and `fail` charges the repair budget and addresses
# every outstanding run at once, so one honest sentence about a thread count marked
# thirteen unrelated runs as defects and asked whose fault a healthy run was. The
# machine side already knew better: `observations._settle_from_machine` declines to
# charge a run that measured correctly without improving, "the ordinary outcome of an
# experiment, and charging it would punish most of them". This is the same judgement,
# made available to the only party that can actually recognise it.
VERDICTS: tuple[str, ...] = ("pass", "fail", "unknown", "blocked", "rejected")


def run_ledger_key(command: str) -> str:
    """The identity a run is remembered by: WHAT it ran, not how it was typed.

    The repair budget counts attempts at the same command, so what counts as "the same"
    decides whether it ever accumulates. Keyed on the raw string it never did: a model
    narrowing down a failure rewrites the tail of its command every time, and

        pytest tests/test_x.py -q 2>&1 | tail -15
        pytest tests/test_x.py -q 2>&1 | tail -8
        pytest tests/test_x.py -v  2>&1 | tail -12

    are three ledger entries of one failure apiece — never the three strikes that arm
    the ladder. (Observed: eleven pytest invocations in one session, no two keys equal,
    the budget never reached.)

    The key is the executing segment's head plus the files it named, taken from the same
    classifier the policy layer parses with, so ``timeout 280 python3 -m pytest a.py``
    and ``pytest a.py`` agree. Flags are dropped — they change how a run reports, not
    what it exercises — and so is everything downstream of a pipe.

    Parsed with ``allow_expansion=True``, unlike every other caller. A bare ``$VAR``
    makes a command opaque to *classification* — its kind depends on what the value
    turns out to be — but this function wants an identity, not a verdict, and it gates
    nothing: the policy layer calls the classifier itself, with the strict default.
    Without it, ``pytest a.py; echo "exit=$?"`` fell back to its raw text while the same
    command without the echo keyed on ``pytest a.py``, so a model adding a status line
    to a command it was already retrying opened a fresh ledger entry every time and the
    repair budget never accumulated. Command *substitution* stays opaque either way: it
    runs code.

    A command the classifier still cannot read keys on its raw text: nothing about it is
    known well enough to call two of them the same.

    A segment that named no file keeps its whole argv instead of collapsing to its head.
    Otherwise every ``python3 -c "…"`` probe in a session — the idiom the base prompt
    actively steers one-off checks towards — would share one ledger entry, and three
    unrelated one-liners would read as three attempts at a single failing thing.

    Idempotent, so a caller holding a key may pass it back in place of a command.
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
    """Register an execution, or re-register one being run again.

    Re-registered last, not in place: an unscoped ``pass`` credits the most recent run,
    and a re-run is newer than the record it replaces. Its failure history is carried
    over — the repair budget counts attempts at the same command, and a re-run is the
    next attempt, not a fresh start.

    Identity comes from :func:`run_ledger_key`, not from the command text; the text is
    kept on the record as ``command`` because that is what a report must show. Readers
    render ``run["command"]``, never the dict key.

    ``effect`` distinguishes a build from a run of the project's code, so the completion
    report can name what actually hit a wall.
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


def failed_runs(execution_context: ExecutionContext) -> dict[str, dict[str, Any]]:
    """Runs that did not reach their end, or whose output the model judged wrong.

    A blocked run is excluded: it charges no repair budget, so leaving it here would hold
    the "every failure is spent" release below its threshold for the rest of the query.
    """
    return {
        command: run for command, run in (execution_context.runs or {}).items()
        if (not run.get("completed") or run.get("verdict") == "fail")
        and not run.get("blocked")
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
