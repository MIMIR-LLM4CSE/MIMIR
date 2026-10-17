"""What a shell command did: the checks it ran, the files it touched or read."""
from __future__ import annotations

from __future__ import annotations
import os
from typing import Any
from ...context import (
    VALIDATION_TIERS,
)
from ..runner_output import (
    TEST_RUNNER_FAILURE,
    exit_status_settles,
    judge_run,
    test_failure_signature,
)
from ..ladder import (
    mark_file_validated,
    record_run_outcome,
    register_validation_failure,
)
from ..workflow import (
    pending_validation_paths,
    set_workflow_state,
)
from ...context.capabilities import (
    shell_command_args,
)
from ...context.paths import is_python_test_filepath
from ....servers._shared.shell_paths import (
    EFFECT_BUILD,
    EFFECT_RUN,
    EFFECT_VALIDATE,
    any_command_on_path,
    is_path_like_command,
)
from ...context.bash_classify import (
    Kind,
    classify_bash_command,
    opaque_command_executes,
    FILE_LIKE_RE,
)
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort
from .call import Call
from .files import enter_post_edit_state, record_code_edit, register_known_path


# Checkers that cover a whole tree in one call. A *successful* run of one naming no
# specific source file (`ruff check .`, `mypy src/`) clears every pending file at once;
# per-file crediting is already language-agnostic, this set only adds the shortcut.
#
# INVARIANT (test-enforced): every entry is in `shell_paths.EXEC_EFFECTS` with effect
# `validate`, and in `VALIDATOR_TIER` below. That is also why `pytest` and `ctest` are
# not here: they are declared `run`, and running a suite is judged on its own output —
# it credits no file however green it exits.
PROJECT_VALIDATORS = frozenset({
    "ruff", "mypy", "pyflakes", "black", "py_compile",
})


# The checkers, keyed by command head, and what kind of check each one is. Reached only
# for a segment `shell_paths.EXEC_EFFECTS` already declared a *validator*: a head absent
# from this table credits nothing, it does not thereby become an execution.
VALIDATOR_TIER = {
    "py_compile": "syntax",
    "ruff": "static", "mypy": "static", "pyflakes": "static", "black": "static",
    # A compiler is a checker, not a run: it emits diagnostics and exit 0 means there
    # were none. Like the linters, its output *is* the verdict — nothing was executed,
    # so there is no result for anyone to judge. It sits on its own tier because it is
    # the one check that needs a toolchain to exist: demanded of nobody, credited when
    # it happens.
    "gcc": "compiled", "g++": "compiled", "gfortran": "compiled", "nvcc": "compiled",
    "javac": "compiled",
}


# Reformatting is not checking: `ruff format` and a bare `black` rewrite the file and
# exit 0 whether or not the code is correct, so they credit nothing. `--check`/`--diff`
# report instead of writing, which is what a checker does.
_REPORT_ONLY_FLAGS = ("--check", "--diff")


def _rewrites_instead_of_checking(head: str, argv: tuple[str, ...]) -> bool:
    """True when this validator invocation formats the file rather than judging it."""
    if head not in ("ruff", "black"):
        return False
    if any(flag in argv for flag in _REPORT_ONLY_FLAGS):
        return False
    if head == "black":
        return True
    # ruff formats only under its `format` sub-command; bare `ruff` and `ruff check`
    # are checks. The head is not argv[0] for `python -m ruff`, so read the sub-command
    # after the head rather than at a fixed index.
    rest = argv[argv.index(head) + 1:] if head in argv else argv
    return next((a for a in rest if not a.startswith("-")), "") == "format"


# What turns a compiler invocation back into the cheap, always-affordable check: it
# parses and type-checks the translation unit without emitting an object, so it is the
# mandatory floor for a compiled language the way ``py_compile`` is for Python.
_SYNTAX_ONLY_FLAGS = ("-fsyntax-only", "--fsyntax-only")


def _call_was_refused(status: Any, payload: dict[str, Any]) -> bool:
    """Whether the server rejected this call before any shell opened.

    A shell tool answers ``status: "error"`` for two different things, and only one of
    them is evidence. A command that ran and exited non-zero — or timed out — is a
    finding about the change; a call the server *refused* (a heredoc, a substitution, a
    denied command, a path outside the workspace) never reached a shell, so it says
    nothing about anything and must charge nothing: not a file's retry budget, not a
    run's.

    The server says which it is, rather than this inferring it from the payload's shape:
    a rejection is the one answer it produces before knowing anything about a run, and
    a flag it sets deliberately cannot drift the way "no ``returncode`` field" can.
    """
    return status != "ok" and bool(payload.get("refused"))


def bash_validation_scan(
    agent: AgentPort, command: str,
) -> tuple[list[str], bool, str, str, str, bool]:
    """Scan a bash command, telling checks from builds from executions.

    Returns ``(explicit_targets, whole_project, tier, execution, missing_head,
    exit_is_the_run_s)``:
    - ``explicit_targets`` — normalized workspace paths of the files named by EXEC
      segments, with a leading ``cd`` rebasing later relative operands (``cd sub &&
      ruff check t.py`` → ``sub/t.py``).
    - ``whole_project`` — True when a recognised project **checker** ran over no specific
      file (``ruff check .``, ``mypy src/``): a green run then validates every pending
      file, not just a named one.
    - ``tier`` — the strongest check any segment performed, or ``""`` when none did.
    - ``execution`` — ``EFFECT_RUN`` when a segment ran the project's code, ``EFFECT_BUILD``
      when the strongest thing it did was drive a build, ``""`` when nothing was executed.
      A run outranks a build in the same chain (``make && ./solver``): the build's exit
      code settles it, the run's output still owes a reading. Which of the two a segment
      is comes from :data:`shell_paths.EXEC_EFFECTS`, declared beside the command
      groups — never from a head's absence from the tier table below, which is how
      ``source env.sh`` came to be recorded as a run owing a verdict.
    - ``missing_head`` — the first command in the chain that is not installed here, or
      ``""``. The one imputation the machine can settle alone: a command the shell could
      not find said nothing about the change, so its red exit is a wall, not a defect.
      Three conditions, each guarding a real trap. It reads ``argv[0]``, never ``head``,
      which is the *module* for ``python -m X`` and so never on PATH (``py_compile`` is a
      checker with no command of its own). It skips a path-like argv0, because a missing
      ``./solver`` is a build that did not happen, which is attributable. And it only looks
      at segments that ran, built or checked something — ``source`` is a shell builtin and
      would otherwise fail this test on every chain that sets up an environment.
    For a command the classifier cannot read at all, the first three are empty — nothing
    is attributable — but ``execution`` still comes from its command-position heads
    (:func:`opaque_command_executes`). Opacity is a reason not to credit; it is not
    a reason to believe nothing ran, and inline payloads (which the base prompt asks for)
    are opaque by construction.
    """
    segments = classify_bash_command(command)
    if not segments:
        # Opaque: something may have run, but which segment produced the status is
        # exactly what could not be parsed, so the status is not attributable.
        return [], False, "", EFFECT_RUN if opaque_command_executes(command) else "", "", False
    rel_base = ""
    targets: list[str] = []
    whole_project = False
    tier_rank = -1
    execution = ""
    missing_head = ""
    for seg in segments:
        if seg.kind == Kind.CHDIR:
            new_dir = seg.operands[0] if seg.operands else ""
            if new_dir:
                rel_base = new_dir if os.path.isabs(new_dir) else os.path.normpath(
                    os.path.join(rel_base, new_dir) if rel_base else new_dir
                )
            continue
        # UNKNOWN is an execution too: a head the taxonomy does not place ran
        # something, and a run owes a verdict whether or not we can name it.
        if seg.kind not in (Kind.EXEC, Kind.UNKNOWN):
            continue
        if (
            not missing_head
            and seg.effect in (EFFECT_RUN, EFFECT_BUILD, EFFECT_VALIDATE)
            and seg.argv
            and not is_path_like_command(seg.argv[0])
            and not any_command_on_path((seg.argv[0],))
        ):
            missing_head = seg.argv[0]
        for op in seg.operands:
            based = op if (os.path.isabs(op) or not rel_base) else os.path.join(rel_base, op)
            path = agent.normalize_workspace_path(based)
            if path:
                targets.append(path)
        if seg.effect == EFFECT_RUN:
            execution = EFFECT_RUN
            continue
        if seg.effect == EFFECT_BUILD:
            execution = execution or EFFECT_BUILD
            continue
        if seg.effect != EFFECT_VALIDATE:
            continue  # env setup: preparation, not evidence
        seg_tier = VALIDATOR_TIER.get(seg.head)
        if seg_tier is None:
            continue
        if _rewrites_instead_of_checking(seg.head, seg.argv):
            continue
        if seg_tier == "compiled" and any(f in seg.argv for f in _SYNTAX_ONLY_FLAGS):
            seg_tier = "syntax"
        tier_rank = max(tier_rank, VALIDATION_TIERS.index(seg_tier))
        names_no_file = not any(FILE_LIKE_RE.search(op) for op in seg.operands)
        if seg.head in PROJECT_VALIDATORS and names_no_file:
            whole_project = True
    tier = VALIDATION_TIERS[tier_rank] if tier_rank >= 0 else ""
    # The shell hands back the last segment's status. It is the run's exactly when the
    # last thing in the chain is the thing that did the work.
    exit_is_the_run_s = segments[-1].effect in (EFFECT_RUN, EFFECT_BUILD, EFFECT_VALIDATE)
    return targets, whole_project, tier, execution, missing_head, exit_is_the_run_s


def _record_test_failure_signatures(
    execution_context: ExecutionContext, out: str,
) -> None:
    """Record WHICH tests a red run reported, per test file.

    The ledger the oracle nudge reads. A file whose failing SET keeps changing between
    runs is one whose test is moving under the model's feet — a different situation from
    the same failure repeating, which is already the repeated-call guard's business, and
    the one that wants a different correction (re-read the test, not the code).

    Keyed off the node ids in the runner's own report rather than off the command's file
    operands, so a bare ``pytest`` is recorded exactly as ``pytest tests/x.py`` is. A
    consecutive repeat of the same signature collapses into the existing entry, which is
    what makes ``len(entries) >= 2`` mean "the failing set changed at least once" rather
    than merely "it failed twice". A report clipped so hard that no node id survived
    records nothing: unable to tell a moving failure from a stuck one, we do not guess.
    """
    signature = test_failure_signature(out)
    if not signature:
        return
    by_file: dict[str, list[str]] = {}
    for node in signature.split():
        by_file.setdefault(node.split("::", 1)[0], []).append(node)
    ledger = execution_context.test_failure_signatures
    for path, nodes in by_file.items():
        seen = ledger.setdefault(path, [])
        sig = " ".join(nodes)
        if not seen or seen[-1] != sig:
            seen.append(sig)


def _stage_moving_test_alert(execution_context: ExecutionContext) -> None:
    """Arm the one-time "read the test itself" corrective for a test that is moving.

    Fires on a file whose failing SET has changed at least once — two entries in the
    signature ledger, which collapses consecutive repeats. An unchanging failure is
    deliberately excluded: that is the repeated-call guard's business, and it wants the
    opposite advice (change something) from this one (stop changing things and read).

    Staged rather than injected, exactly as ``_repeat_alert`` is, because this fires in
    the middle of a tool loop that a nudge from the table can never reach — the model
    debugging a red suite is by definition still calling tools. Once per file per query:
    the point is made or it is not.
    """
    ledger = execution_context.test_failure_signatures or {}
    told = execution_context.moving_test_told
    for path, seen in sorted(ledger.items()):
        if len(seen) >= 2 and path not in told:
            told.append(path)
            execution_context.moving_test_alert = path
            return


def bash_validation(call: Call) -> None:
    """Drive validation and run state from a bash command — success AND failure.

    Two axes, never mixed, because they answer different questions:

    - a **checker** (``py_compile`` → syntax; ``ruff``/``mypy``/a compiler → static)
      validates the *files it names*. Its output is a list of problems and an empty one
      is the finding, so exit 0 settles it and nobody has to read anything. Non-zero
      charges the file's retry budget.
    - an **execution** (``pytest``, ``python solver.py``, ``./solver``, ``python -c …``)
      validates no file at all. It is recorded as a *run*: a non-zero exit is a failure
      the machine already judged, and a green one owes the model a reading of what it
      printed (``guardrails/verdict.py``). A green exit is demoted when the output itself
      reports otherwise — a declared ``check=fail``, or a test runner's own summary of
      red tests, which is the only signal left once ``| tail`` has eaten the exit status.
      A red *test* settles the run without charging any file's validation budget:
      the run is the subject, and no file is credited or blamed by guesswork.

    A green project-wide **checker** (``ruff check .``) covers every pending file at
    once. Test files are recorded in ``tests_run`` either way (pass or fail) so the
    regression nudge sees them. Only shell-command tools qualify.
    """
    execution_context = call.execution_context
    command_args = shell_command_args(call.tool_name, call.agent.tool_caps)
    if command_args is None:
        return
    if call.payload.get("background_job"):
        # The run is still going: this payload is a handle plus whatever it had
        # printed so far, not a verdict. Judging it would credit a whole pending
        # validation set at the instant a suite was launched.
        return
    if _call_was_refused(call.status, call.payload):
        # The server refused the call before any shell opened, so there is nothing to
        # observe in either direction: no check ran, and no run failed.
        return
    command = str(call.arguments.get(command_args[0], "") or "")
    explicit, whole_project, tier, execution, missing_head, exit_is_the_run_s = (
        bash_validation_scan(call.agent, command)
    )
    # `execution` (not just a named target) is what catches `python -c …` and a bare
    # `./solver`: they validate nothing, but they produced output somebody must judge.
    if not explicit and not whole_project and not execution:
        return
    # stdout and stderr together: every recorded invocation writes `2>&1`, which puts
    # the run's report in whichever stream the caller merged into.
    out = str(call.payload.get("stdout") or "") + "\n" + str(call.payload.get("stderr") or "")
    # One question, asked once, in the module that owns it. An `or` chain of dialects
    # assembled here is how each new way of reporting a failure arrives as a missed failure
    # first and a clause second.
    completed, reason = judge_run(exit_ok=(call.status == "ok"), output=out)
    if reason == TEST_RUNNER_FAILURE:
        _record_test_failure_signatures(execution_context, out)
        _stage_moving_test_alert(execution_context)
    dirty = execution_context.dirty_written_files
    for target in explicit:
        if is_python_test_filepath(target):
            execution_context.tests_run.add(target)
        if not tier or target not in dirty:
            continue
        if completed:
            mark_file_validated(execution_context, target, tier)
        else:
            # A specific-file failure is attributable; a whole-project failure is not.
            register_validation_failure(execution_context, target, tier)
    # A green project-wide check covers every still-pending file. A failing one is not
    # attributable to a single file, so it leaves pending untouched (the model narrows
    # down or re-runs per file).
    if tier and whole_project and completed:
        for target in list(pending_validation_paths(execution_context)):
            mark_file_validated(execution_context, target, tier)
    if execution:
        record_run_outcome(
            execution_context, command, completed, call.call_id, reason,
            effect=execution, missing_head=missing_head,
            exit_settles=exit_status_settles(exit_is_the_run_s),
        )


def command(call: Call) -> None:
    """Credit the blackboard from a shell command classified into capability kinds.

    A dual-use bash tool hides its effect inside a ``command`` string, so the other
    observers (which key off static caps + a named path arg) miss it.
    ``classify_bash_command`` maps each ``; && || |`` segment to a kind + operands;
    this mirrors the dedicated observers so a bash ``cat``/``grep``/``sed -i`` feeds
    the same discovery/edit/action state the file/search tools would.
    """
    execution_context = call.execution_context
    if call.status != "ok":
        return
    command_args = shell_command_args(call.tool_name, call.agent.tool_caps)
    if command_args is None:
        return
    if call.payload.get("background_job"):
        return  # still running — its effects have not happened yet
    command = call.arguments.get(command_args[0], "")
    segments = classify_bash_command(command)
    if not segments:
        return

    # Track the shell's cwd across the chain: a ``cd sub`` rebases the relative path
    # operands of every later segment, so they normalize to the same workspace path the
    # dedicated file/search observers record. Applied to every path-bearing kind.
    rel_base = ""

    def _resolve(op: str) -> str | None:
        if not op:
            return None
        based = op if (os.path.isabs(op) or not rel_base) else os.path.join(rel_base, op)
        return call.agent.normalize_workspace_path(based)

    for seg in segments:
        kind, operands = seg.kind, seg.operands
        if kind == Kind.CHDIR:
            new_dir = operands[0] if operands else ""
            if new_dir:
                rel_base = new_dir if os.path.isabs(new_dir) else os.path.normpath(
                    os.path.join(rel_base, new_dir) if rel_base else new_dir
                )
            continue
        if kind == Kind.READ:
            for op in operands:
                path = _resolve(op)
                if path:
                    execution_context.read_files.add(path)
                    execution_context.checked_paths.add(path)
                    register_known_path(execution_context, path)
        elif kind == Kind.SEARCH:
            execution_context.searched = True
            execution_context.search_tool_calls += 1
        elif kind == Kind.INSPECT:
            for op in operands:
                path = _resolve(op)
                if path:
                    execution_context.inspected_dirs.add(path)
        elif kind == Kind.WRITE:
            execution_context.action_op_count = int(execution_context.action_op_count) + 1
            for op in operands:
                path = _resolve(op)
                if path and call.agent.is_code_filepath(path):
                    record_code_edit(execution_context, path)
                    execution_context.last_edit_success_path = path
                    enter_post_edit_state(execution_context)
        elif kind in (Kind.EXEC, Kind.UNKNOWN):
            # Marking a dirty file validated (or failed) is handled status-agnostically
            # by bash_validation; here we only count the substantive action.
            execution_context.action_op_count = int(execution_context.action_op_count) + 1
        elif kind == Kind.ENV_DISCOVERY:
            execution_context.env_probed = True
        elif kind == Kind.ENV_MUTATE:
            mutations = execution_context.env_mutations
            record = {
                "tool": call.tool_name, "installed": list(operands),
                "python": "", "path": "", "name": "",
            }
            if record not in mutations:
                mutations.append(record)

    # Mirror discover_transition (which matches result payloads the bash
    # tool doesn't emit): a read/search/inspect that produced output leaves discover.
    if execution_context.workflow_state == "discover" and call.payload.get("stdout"):
        if any(seg.kind in (Kind.READ, Kind.SEARCH, Kind.INSPECT) for seg in segments):
            set_workflow_state(execution_context, "edit")
