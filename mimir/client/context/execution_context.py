from __future__ import annotations

import os
from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any, Callable, TypedDict, cast


class ExecutionContext(TypedDict):
    # ── Discovery: what the agent has searched, inspected, and read ────────────
    searched: bool                                  # True once any SEARCH-capability tool has run (discovery-evidence signal)
    planned_edit_targets: set[str]                  # files the model intends to / has begun editing (union'd with dirty for todo + blast gating)
    inspected_dirs: set[str]                        # directories listed/inspected (delete context, discovery); counts as evidence beyond BASELINE_SEEDED_DIRS
    checked_paths: set[str]                         # paths whose existence was probed via an existence check
    read_files: set[str]                            # files whose content was read (edit/overwrite precondition; dropped to force a re-read after repeated edit failures)
    existing_paths: set[str]                        # paths proven to exist on disk (reads, existence hits, candidate matches, written files + their parent dirs)
    similar_candidates_by_dir: dict[str, set[str]]  # per-directory near-name candidates from candidate searches (path clarification)
    search_tool_calls: int                          # count of successful search calls (blast-radius gate)
    action_op_count: int                            # count of successful substantive actions (PLAN_BLOCKED: writes/exec/mutations); drives the todo op-count trigger
    # ── Edit: planned and in-flight mutations ──────────────────────────────────
    dirty_written_files: set[str]                   # code files successfully written this query and not yet re-validated
    validated_files: set[str]                       # dirty files that have since passed syntax validation
    validation_tier_by_file: dict[str, str]         # per-file strength of that validation: syntax|static|executed|oracle (completion ledger)
    validation_fail_count_by_file: dict[str, int]   # per-file count of failed syntax validations (retry-budget exhaustion)
    executed_failures: set[str]                     # files an `executed`-tier check was observed FAILING on, ever, this query — the red half of red→green (see VALIDATION_TIERS)
    denied_tool_calls: list[dict[str, Any]]         # tool calls blocked by policy/approval (denial nudge + finalization)
    denial_history: list[dict[str, Any]]            # append-only refusal log keyed by approval scope; drives the escalation ladder (never cleared, unlike denied_tool_calls)
    workflow_state: str                             # coarse phase of the loop: "discover" | "edit" | "validate" | "conclude"
    code_mutation_started: bool                     # True once the first code file was successfully edited this query
    cluster_submit_warned: bool                     # True once the pre-submission HPC guard has fired once this query
    nudge_counts: dict[str, int]                    # per-category nudge fire counts (frequency caps). keys: discovery, validation, denial, state, creation, blast_radius, doc, todo, error_recovery, regression
    search_queries_used: set[str]                   # normalized query/pattern strings already searched (redundant-search detection)
    read_file_line_counts: dict[str, int]           # per-file cumulative lines read (10_000 sentinel == whole file)
    cross_file_grep_old_text: str | None            # old_text of a confirmed workspace-wide replace, seeding the cross-file completeness check
    cross_file_grep_source: str | None              # file where that workspace-wide replacement originated
    last_replace_old_text: str | None               # old_text of the most recent tracked replacement (completeness checking)
    last_replace_new_text: str                      # new_text of that most recent tracked replacement
    last_replace_file: str                          # file the most recent tracked replacement targeted
    edit_loop_state: dict[str, tuple]               # path -> (last_signature: str|None, identical consecutive failures: int); guards the identical-patch hard block
    edit_fail_streak_by_file: dict[str, int]        # path -> consecutive edit failures regardless of patch (force-re-read + error_recovery)
    steps_since_last_edit: int                      # agent-loop steps since the last successful edit (idle gating)
    declared_edit_set: set[str]                     # source files named in a task-checklist declaration (multi-file plan completeness)
    snippet_read_files: set                         # files whose content was only partially/snippet-read
    tests_run: set[str]                             # files passed to a tests-kind validator this query (regression guard)
    # ── Environment: what the run needs and what it changed ────────────────────
    unresolved_modules: set[str]                     # imports a run failed to resolve (env-resolution nudge)
    env_probed: bool                                 # True once the model enumerated the available interpreters/envs
    env_mutations: list[dict[str, Any]]              # installs / env creations recorded this query (conclude-phase cleanup nudge)
    # ── Todo / planning ────────────────────────────────────────────────────────
    rearmed_domains: set[str]                       # domain-group keys unlocked mid-query after the work revealed a need the query never signaled (bounded by DOMAIN_REARM_MAX_PER_QUERY)
    todo_written: bool                              # True once an ordered checklist (plan_steps) was recorded
    plan_written: bool                              # True once a prose plan/rationale was recorded
    todo_file_path: str                             # absolute path to the active todo_list.md (set at query start)
    last_edit_success_path: str                     # path of last successful code edit (cleared after injection)
    # ── Cross-query and loop bookkeeping ───────────────────────────────────────
    prev_query_written_files: set[str]               # files written by the PREVIOUS query, forwarded from carry_context so the pin can warn to re-read them
    tool_msg_files: dict[str, list[str]]             # tool_call_id -> files that tool message concerns; lets history trimming match messages to files structurally
    consecutive_noop_turns: int                      # consecutive bare final-answer turns with no tool call (nudge cutoff)


# Per-query loop-control bookkeeping used only by agent_loop's tool-dispatch spin
# guards (repeated-failure + redundant-success). Deliberately kept OUT of the
# ExecutionContext contract above: these are private dedup implementation details,
# reset fresh each query, not semantic discovery/edit/validation state. They live in
# one object stored under the private ``_loop_control`` context key (via
# ``loop_control()``) so the public schema stays about meaning, not plumbing.
@dataclass
class LoopControlState:
    write_calls: set = field(default_factory=set)          # (tool, args) writes already dispatched this query
    call_fails: dict = field(default_factory=dict)         # (tool, args) -> identical-FAILED dispatch count
    repeat_warned: set = field(default_factory=set)        # keys the soft repeated-call corrective fired for
    call_results: dict = field(default_factory=dict)       # (tool, args) -> (result_hash, redundant_repeat_count)
    redundant_warned: set = field(default_factory=set)     # keys the soft redundant-read corrective fired for
    redundant_call_ids: dict = field(default_factory=dict)  # (tool, args) -> ordered identical-content call_ids


_LOOP_CONTROL_KEY = "_loop_control"


def loop_control(execution_context: dict[str, Any]) -> LoopControlState:
    """Return the per-query loop-control state, creating it on first use.

    Not part of the validated ExecutionContext schema — the state is created lazily
    the first time agent_loop needs it and discarded with the per-query context.
    """
    lc = execution_context.get(_LOOP_CONTROL_KEY)
    if not isinstance(lc, LoopControlState):
        lc = LoopControlState()
        execution_context[_LOOP_CONTROL_KEY] = lc
    return lc


# ── Recency-preserving sets ────────────────────────────────────────────────────
#
# Several fields are rendered into the discovery pin, which can only show a bounded
# slice of them. A plain ``set`` carries no order, so the only deterministic slice is
# an alphabetical one — and the alphabetically-last ten paths are not the ten that
# matter. ``RecencySet`` keeps insertion order alongside the set so the pin can show
# what the model just touched.
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
# What a field *is*, declared next to the field itself. Before this, the answers lived
# in eight hand-maintained name lists scattered across agent_core, observations, the
# policy engine and the nudge engine — two of which computed the same thing twice, two
# of which had gone stale (an entry for a field that is never in carry_context at all).
# Nothing kept them correct when a field was added; each derived list below is now
# computed from this table, so the question is asked once, where the field is defined.
CARRY = "carry"              # merged into the next query's context (session memory)
FILE_PATH = "file_path"      # holds workspace FILE paths -> a deleted file must be purged
KNOWN_FILE = "known_file"    # holds a file path the model has encountered this session
DISCOVERY = "discovery"      # counts as the model's own discovery evidence

FIELD_TRAITS: frozenset[str] = frozenset({CARRY, FILE_PATH, KNOWN_FILE, DISCOVERY})

# ``inspected_dirs`` is deliberately CARRY + DISCOVERY but NOT FILE_PATH: it holds
# directories, and deleting a file does not un-inspect the directory that contained it.

# Single source of truth for the execution-context schema: ``(name, default_factory,
# accepted_types, traits)``.  ``execution_context_template``, ``validate_execution_context``
# and every derived field list are computed from this table, so a new field only needs to
# be added once here (plus the ``ExecutionContext`` TypedDict above, kept in sync by the
# contract test asserting the two key sets match).  Grouped by concern for readability.
_FieldSpec = tuple[str, Callable[[], Any], tuple[type, ...], frozenset[str]]
_NoneType = type(None)
_NO_TRAITS: frozenset[str] = frozenset()

_FIELD_SPECS: tuple[_FieldSpec, ...] = (
    # ── Discovery: what the agent has searched, inspected, and read ────────────
    ("searched", lambda: False, (bool,), frozenset({DISCOVERY})),
    ("inspected_dirs", set, (set,), frozenset({CARRY, DISCOVERY})),
    ("checked_paths", set, (set,), frozenset({CARRY, FILE_PATH, KNOWN_FILE, DISCOVERY})),
    # RecencySet: rendered into the pin, which shows a bounded slice — recency is what
    # makes that slice useful (see recent_first).
    ("read_files", RecencySet, (set,), frozenset({CARRY, FILE_PATH, KNOWN_FILE, DISCOVERY})),
    # Not DISCOVERY: existence can be established by the repo baseline rather than by
    # the model's own exploration, which is exactly what the discovery gates must not
    # accept as evidence.
    ("existing_paths", RecencySet, (set,), frozenset({CARRY, FILE_PATH, KNOWN_FILE})),
    ("similar_candidates_by_dir", dict, (dict,), _NO_TRAITS),
    ("search_tool_calls", lambda: 0, (int,), _NO_TRAITS),
    # Count of successful substantive actions (writes/exec/mutations — any PLAN_BLOCKED
    # tool). Read by the todo nudge so a many-operation task is recognised as
    # multi-step even when it touches only one (or zero) files.
    ("action_op_count", lambda: 0, (int,), _NO_TRAITS),
    # Carried, but NOT FILE_PATH: these are search patterns, not paths.
    ("search_queries_used", RecencySet, (set,), frozenset({CARRY})),
    ("read_file_line_counts", dict, (dict,), _NO_TRAITS),
    # Partial context only — never CARRY, and never proof of a full read (see
    # ``was_fully_read``). It is still KNOWN_FILE: seeing a snippet proves the file exists.
    ("snippet_read_files", set, (set,), frozenset({FILE_PATH, KNOWN_FILE, DISCOVERY})),
    # ── Edit: planned and in-flight mutations ──────────────────────────────────
    ("planned_edit_targets", RecencySet, (set,), _NO_TRAITS),
    ("declared_edit_set", set, (set,), _NO_TRAITS),
    ("dirty_written_files", RecencySet, (set,), frozenset({FILE_PATH})),
    ("cross_file_grep_old_text", lambda: None, (str, _NoneType), _NO_TRAITS),
    ("cross_file_grep_source", lambda: None, (str, _NoneType), _NO_TRAITS),
    ("last_replace_old_text", lambda: None, (str, _NoneType), _NO_TRAITS),
    ("last_replace_new_text", lambda: "", (str,), _NO_TRAITS),
    ("last_replace_file", lambda: "", (str,), _NO_TRAITS),
    ("edit_loop_state", dict, (dict,), _NO_TRAITS),
    ("edit_fail_streak_by_file", dict, (dict,), _NO_TRAITS),
    ("steps_since_last_edit", lambda: 0, (int,), _NO_TRAITS),
    ("last_edit_success_path", lambda: "", (str,), _NO_TRAITS),
    # ── Validation ─────────────────────────────────────────────────────────────
    ("validated_files", set, (set,), frozenset({FILE_PATH})),
    ("validation_tier_by_file", dict, (dict,), _NO_TRAITS),
    ("validation_fail_count_by_file", dict, (dict,), _NO_TRAITS),
    ("executed_failures", set, (set,), _NO_TRAITS),
    # ── Workflow + policy ──────────────────────────────────────────────────────
    ("workflow_state", lambda: "discover", (str,), _NO_TRAITS),
    ("code_mutation_started", lambda: False, (bool,), _NO_TRAITS),
    ("cluster_submit_warned", lambda: False, (bool,), _NO_TRAITS),
    ("nudge_counts", dict, (dict,), _NO_TRAITS),
    ("denied_tool_calls", list, (list,), _NO_TRAITS),
    ("denial_history", list, (list,), _NO_TRAITS),
    ("tests_run", set, (set,), _NO_TRAITS),
    # ── Environment ────────────────────────────────────────────────────────────
    # These three were live for a long time without being declared here at all: seeded
    # by a bootstrap helper, written by the observers, read by the nudges — and so
    # invisible to the template, to validate_execution_context, and to the TypedDict
    # parity test. Declaring them is what lets a backfill be derived from this table
    # instead of hand-listed per module.
    ("unresolved_modules", set, (set,), _NO_TRAITS),
    ("env_probed", lambda: False, (bool,), _NO_TRAITS),
    ("env_mutations", list, (list,), _NO_TRAITS),
    # ── Todo / planning ────────────────────────────────────────────────────────
    ("todo_written", lambda: False, (bool,), _NO_TRAITS),
    ("plan_written", lambda: False, (bool,), _NO_TRAITS),
    ("todo_file_path", lambda: "", (str,), _NO_TRAITS),
    # ── Tool visibility ────────────────────────────────────────────────────────
    ("rearmed_domains", set, (set,), _NO_TRAITS),
    # ── Cross-query and loop bookkeeping ───────────────────────────────────────
    # These three were live without being declared: written by the loop, the dispatcher
    # or the carry merge, and read by the pin and the history budget — but invisible to
    # the template, to validate_execution_context and to the trait derivations. That is
    # why a deleted file kept appearing in the pin's "written last query" list: the
    # delete observer purges fields_with(FILE_PATH), and an undeclared field has no
    # traits to be found by.
    ("prev_query_written_files", set, (set,), frozenset({FILE_PATH})),
    ("tool_msg_files", dict, (dict,), _NO_TRAITS),
    ("consecutive_noop_turns", lambda: 0, (int,), _NO_TRAITS),
)


# ── The carry-context schema ───────────────────────────────────────────────────
#
# ``carry_context`` is the session-level dict that survives between queries. Most of it
# mirrors the CARRY-trait fields above, but three keys exist only there — and because
# they were never declared anywhere, ``export_state`` did not know to normalise them
# for JSON. ``last_query_written_files`` is a ``set``: the session store serialises with
# ``default=str``, so it round-tripped as the *string* "{'a.py', 'b.py'}", and the next
# ``_apply_carry_context`` did ``set(...)`` on that string, exploding it into single
# characters that the pin then rendered as file paths.
#
# Declaring the extras here is what lets the JSON conversion and the deleted-path purge
# be derived instead of hand-listed, so the next key added cannot repeat the bug.
#   (name, default_factory, json_kind, holds_paths)
_CarryExtraSpec = tuple[str, Callable[[], Any], str, bool]

_CARRY_EXTRA_SPECS: tuple[_CarryExtraSpec, ...] = (
    ("searched", lambda: False, "scalar", False),
    # Forwarded into the next query's execution_context as prev_query_written_files.
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
    return tuple(name for name, _f, _t, field_traits in _FIELD_SPECS if wanted <= field_traits)


def was_fully_read(execution_context: dict[str, Any], path: str) -> bool:
    """Whether *path* was read in full through a direct read tool.

    **A snippet does not count.** ``snippet_read_files`` records partial context — a
    grep hit, a neighbour excerpt — which is enough to plan against but not to rewrite
    a file from. POLICY.md has always said so; until this predicate existed the rule
    lived only in prose, while the call site was a bare ``path in ec["read_files"]``
    sitting next to an equally plausible ``ec["snippet_read_files"]``.
    """
    return path in (execution_context.get("read_files") or set())


def is_known_to_exist(execution_context: dict[str, Any], path: str) -> bool:
    """Whether *path* is proven to exist — a successful read or an existence hit.

    Stronger than :func:`was_checked_for`, which only records that a look happened.
    """
    return (
        path in (execution_context.get("existing_paths") or set())
        or was_fully_read(execution_context, path)
    )


def was_checked_for(execution_context: dict[str, Any], path: str) -> bool:
    """Whether a pre-check was *attempted* for *path*.

    Explicitly **not** proof that it exists: a check that came back negative lands here
    too. Kept separate from :func:`is_known_to_exist` because the destructive guards
    need the stronger fact, and a name is the only thing that keeps the two apart at
    the call site.
    """
    return path in (execution_context.get("checked_paths") or set())


def known_existing_files(execution_context: dict[str, Any]) -> set[str]:
    """Every file path the model has encountered this session (proof they exist).

    One implementation for a question that had two: the policy engine's path-clarification
    lookup and the nudge engine's ``_known_existing_files`` walked the same four fields
    with the same intent, in two modules, with nothing keeping them equal.
    """
    known: set[str] = set()
    for name in fields_with(KNOWN_FILE):
        value = execution_context.get(name)
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
#      run after every tool call)              search_tool_calls, snippet_read_files,     runtime (check_write_policy),
#                                              similar_candidates_by_dir,                 state_machine (guards),
#                                              dirty_written_files, declared_edit_set,    nudge_logic (all nudges),
#                                              edit_loop_state, edit_fail_streak_by_file,  agent_loop (correctives)
#                                              last_replace_*,
#                                              code_mutation_started, validated_files,
#                                              validation_tier_by_file,
#                                              validation_fail_count_by_file, tests_run,
#                                              action_op_count,
#                                              todo_written, plan_written
#   context/workflow.set_workflow_state       workflow_state                             every layer's state gates
#   query_engine/nudge_logic._fire_nudge      nudge_counts[<category>]                    nudge_logic (per-category caps)
#   query_engine/agent_loop                   steps_since_last_edit (increment),          nudge_logic (idle gates),
#     (loop body + loop-control correctives)    LoopControlState (the private             agent_loop (repeat/redundant
#                                              _loop_control object: write_calls,          guards)
#                                              call_fails, call_results, …)
#   policy/engine + agent_core (approval)  denied_tool_calls, denial_history          state_machine (blocking denials),
#                                                                                         nudge_logic (denial nudge),
#                                                                                         workflow (escalation ladder)
#
# Rule of thumb: fields are WRITTEN in exactly one place (the observation pass or the
# loop) and READ in many. Never mutate context outside its declared writer above —
# that is what previously caused cross-layer incoherence.


# ── Discovery-evidence semantics: the single definition of "what counts as the
# model's own exploration this query." Every consumer (the agent-mode discovery
# nudge, the plan-mode evidence gate, engine._missing_evidence) reads these helpers
# instead of hand-picking its own subset of fields — that duplicated subset-picking
# was the source of cross-layer coherence bugs.
#
# ``inspected_dirs`` counts, but only beyond what the repo baseline seeded: a
# snapshot must never pre-satisfy a gate on the model's behalf, while a directory
# the model actually inspected is real exploration. Excluding the field outright
# (as it was) made structural discovery — listing/summarising a subtree — worth
# nothing, so a gate could only ever be cleared by a grep or a file read, which is
# the wrong bar for a task that is about layout rather than about one symbol.
# Every other signal below is only ever set by the model's own tool calls this
# query (see guardrails.observations._observe_*).
BASELINE_SEEDED_DIRS: frozenset[str] = frozenset({"."})

# Derived from the ``DISCOVERY`` trait rather than hand-listed, so a new discovery
# field cannot be added to the schema and forgotten here. The membership is unchanged:
# searched, read_files, snippet_read_files, checked_paths, inspected_dirs.
DISCOVERY_EVIDENCE_SIGNALS: tuple[str, ...] = fields_with(DISCOVERY)


def _signal_present(execution_context: dict[str, Any], signal: str) -> bool:
    """Whether *signal* carries the model's own evidence.

    ``inspected_dirs`` is the one field with a pre-seeded component, so the seeded
    entries are discounted before the truth test; every other signal is taken as-is.
    """
    value = execution_context.get(signal)
    if signal == "inspected_dirs":
        return bool(set(value or ()) - BASELINE_SEEDED_DIRS)
    return bool(value)


def discovery_signal_count(execution_context: dict[str, Any]) -> int:
    """Number of distinct model-initiated discovery signals present in *execution_context*."""
    return sum(1 for s in DISCOVERY_EVIDENCE_SIGNALS if _signal_present(execution_context, s))


def nudge_count(execution_context: dict[str, Any], category: str) -> int:
    """How many times the *category* nudge has fired this query (0 if never).

    Named accessor for the ``execution_context["nudge_counts"].get(category, 0)``
    pattern that the per-category frequency caps repeat across every nudge branch.
    """
    return execution_context.get("nudge_counts", {}).get(category, 0)


def idle_steps(execution_context: dict[str, Any]) -> int:
    """Steps since the last successful edit (large sentinel when none yet).

    Named accessor for ``int(execution_context.get("steps_since_last_edit", 99))``;
    the 99 sentinel means "the model has not edited anything, so it is maximally
    idle" — used by the validation/state idle gates.
    """
    return int(execution_context.get("steps_since_last_edit", 99))


def has_discovery_evidence(execution_context: dict[str, Any], *, min_distinct: int = 1) -> bool:
    """True once the model has done at least *min_distinct* distinct discovery actions."""
    return discovery_signal_count(execution_context) >= min_distinct


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


def unwritten_declared_files(execution_context: dict[str, Any]) -> list[str]:
    """Declared edit targets that no write this query honoured, sorted.

    The single definition of "promised and then skipped", read by the completion
    issues, the residual-risk level, the verification ledger and the validation
    nudge. Matching is by resolved path, not by string (see
    :func:`_declared_target_written`) — a raw set difference reported a file as
    never written whenever the plan spelled it differently from the write.
    """
    declared = execution_context.get("declared_edit_set", set()) or set()
    if not declared:
        return []
    dirty = set(execution_context.get("dirty_written_files", set()) or set())
    from ..config.constants import WORKSPACE_ROOT
    root = os.path.abspath(WORKSPACE_ROOT)
    return sorted(d for d in declared if not _declared_target_written(d, dirty, root))


def declared_edit_set_complete(execution_context: dict[str, Any]) -> bool:
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
# what?". The distinction exists because every rung below "oracle" tests only that
# the code *runs* — a test asserting no-NaN and a loose magnitude bound exits 0
# exactly like one that compares against an analytic solution, and for a long time
# the two were indistinguishable everywhere in the client.
#
# Ordered weakest→strongest. Report-only: nothing gates, blocks, or nudges on the
# tier — it is read by the completion ledger so the *answer* can say what was
# actually established. That is deliberate; a wrong "your evidence is weak" verdict
# would loop the model, whereas a pessimistic ledger line costs nothing.
VALIDATION_TIERS: tuple[str, ...] = ("syntax", "static", "executed", "oracle")
_TIER_RANK = {name: i for i, name in enumerate(VALIDATION_TIERS)}


def validation_tier(execution_context: dict[str, Any], path: str) -> str | None:
    """The strength of *path*'s validation, or None if it was never validated."""
    return execution_context.get("validation_tier_by_file", {}).get(path)


def raise_validation_tier(execution_context: dict[str, Any], path: str, tier: str) -> None:
    """Record *tier* for *path*, keeping the strongest evidence seen.

    Monotone by design: a file validated by a reference comparison and then
    re-checked with a bare syntax pass has still been compared. The tier is
    cleared (not lowered) when the file is re-edited or fails validation —
    wherever ``validated_files`` is discarded, this is popped alongside it.
    """
    if tier not in _TIER_RANK:
        return
    tiers = execution_context.setdefault("validation_tier_by_file", {})
    current = tiers.get(path)
    if current is None or _TIER_RANK[tier] > _TIER_RANK[current]:
        tiers[path] = tier


def weakest_validation_tier(
    execution_context: dict[str, Any], paths: Iterable[str],
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


def files_below_tier(execution_context: dict[str, Any], tier: str) -> list[str]:
    """Validated files whose evidence is weaker than *tier* (sorted).

    Unvalidated files are excluded — they are already reported as pending, and
    listing them here would double-count them in the ledger.
    """
    threshold = _TIER_RANK.get(tier)
    if threshold is None:
        return []
    tiers = execution_context.get("validation_tier_by_file", {})
    return sorted(p for p, t in tiers.items() if _TIER_RANK.get(t, 0) < threshold)


def execution_context_template() -> ExecutionContext:
    template = {name: factory() for name, factory, _types, _traits in _FIELD_SPECS}
    return cast(ExecutionContext, template)


def build_execution_context() -> ExecutionContext:
    return execution_context_template()


def validate_execution_context(execution_context: dict[str, Any]) -> None:
    """Fail fast if the runtime policy contract is broken."""
    for key, _factory, types, _traits in _FIELD_SPECS:
        if key not in execution_context:
            raise KeyError(f"Missing execution_context key: {key}")
        if not isinstance(execution_context[key], types):
            type_names = ", ".join(tp.__name__ for tp in types)
            raise TypeError(
                f"Invalid execution_context[{key!r}] type: "
                f"{type(execution_context[key]).__name__}; expected one of: {type_names}"
            )

    for parent, candidates in execution_context["similar_candidates_by_dir"].items():
        if not isinstance(parent, str):
            raise TypeError("execution_context['similar_candidates_by_dir'] keys must be str")
        if not isinstance(candidates, set):
            raise TypeError("execution_context['similar_candidates_by_dir'] values must be set[str]")

    for target in execution_context["planned_edit_targets"]:
        if not isinstance(target, str):
            raise TypeError("execution_context['planned_edit_targets'] values must be str")

    for key, count in execution_context["validation_fail_count_by_file"].items():
        if not isinstance(key, str):
            raise TypeError("execution_context['validation_fail_count_by_file'] keys must be str")
        if not isinstance(count, int):
            raise TypeError("execution_context['validation_fail_count_by_file'] values must be int")

    for key, tier in execution_context["validation_tier_by_file"].items():
        if not isinstance(key, str):
            raise TypeError("execution_context['validation_tier_by_file'] keys must be str")
        if tier not in VALIDATION_TIERS:
            raise ValueError(
                f"execution_context['validation_tier_by_file'][{key!r}] must be one of "
                f"{VALIDATION_TIERS}; got {tier!r}"
            )

    for query in execution_context["search_queries_used"]:
        if not isinstance(query, str):
            raise TypeError("execution_context['search_queries_used'] values must be str")


def ensure_execution_context(execution_context: dict[str, Any] | None) -> ExecutionContext | None:
    if execution_context is None:
        return None

    # Defensive backfill so policy checks do not crash when keys are missing.
    defaults = execution_context_template()
    for key, value in defaults.items():
        if key not in execution_context:
            execution_context[key] = value

    validate_execution_context(execution_context)
    return cast(ExecutionContext, execution_context)


def backfill_execution_context(execution_context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Give *execution_context* every declared field, without validating it.

    The one seeding helper. There were four — ``bootstrap_engine_context``,
    ``bootstrap_runtime_context``, ``bootstrap_state_context`` and the nudge engine's
    own ``_bootstrap_nudge_context`` — each ``setdefault``-ing the subset of fields its
    own module happened to read. Four hand-picked subsets of one schema is four chances
    to miss a field, and they had already diverged: the nudge one defaulted
    ``steps_since_last_edit`` to **99** where the others (and the schema) use 0, so the
    same absent field meant "maximally idle" to the idle gates and "just edited" to the
    message builder.

    Deriving from :data:`_FIELD_SPECS` removes the choice: a field that exists is
    seeded, wherever the context came from. Callers can then read ``ec["field"]``
    instead of defending with ``ec.get("field", set())`` at every site.

    Distinct from :func:`ensure_execution_context` only by what it *skips*: no
    validation pass, so it stays cheap enough for a per-tool-call path. The saving was
    never about seeding fewer fields.
    """
    if execution_context is None:
        return None
    for name, factory, _types, _traits in _FIELD_SPECS:
        if name not in execution_context:
            execution_context[name] = factory()
    return execution_context


# Names the call sites still use. Kept as aliases so this change stays a pure
# consolidation; they carry no distinct behaviour any more.
bootstrap_engine_context = backfill_execution_context
bootstrap_runtime_context = backfill_execution_context
bootstrap_state_context = backfill_execution_context
