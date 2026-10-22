"""What a shell command did: the checks it ran, the files it touched or read."""
from __future__ import annotations

from __future__ import annotations
import os
from typing import Any
from ...context import (
    VALIDATION_TIERS,
)
from ..verification.runner_output import (
    TEST_RUNNER_FAILURE,
    exit_status_settles,
    judge_run,
    test_failure_signature,
)
from ..verification.ladder import (
    mark_file_validated,
    record_run_outcome,
    register_validation_failure,
)
from ..workflow import (
    pending_validation_paths,
    set_workflow_state,
)
from ...tools.capabilities import shell_command_args
from ...context.paths import is_python_test_filepath
from ....shared.jobs import launched
from ....shared.shell_paths import (
    EFFECT_BUILD,
    EFFECT_RUN,
    EFFECT_VALIDATE,
    any_command_on_path,
    is_path_like_command,
)
from ...tools.bash_classify import (
    Kind,
    classify_bash_command,
    opaque_command_executes,
    FILE_LIKE_RE,
)
from ...context.execution_context import ExecutionContext
from ...core.ports import ObserverAgent
from .call import Call
from .files import enter_post_edit_state, record_code_edit, register_known_path


# Checkers that cover a whole tree in one call: a green run naming no file clears every
# pending file. Each is `validate` in `shell_paths.EXEC_EFFECTS` and in `VALIDATOR_TIER`
# (test-enforced); test runners are `run` and credit no file.
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
    """Whether the server refused this call before any shell opened (``rejected`` flag).

    A refusal is no evidence and charges nothing; a non-zero exit is a finding.
    """
    return status != "ok" and bool(payload.get("refused"))


def bash_validation_scan(
    agent: ObserverAgent, command: str,
) -> tuple[list[str], bool, str, str, str, bool]:
    """Scan a command: ``(targets, whole_project, tier, execution, missing_head, exit_is_the_run_s)``.

    Targets are the files EXEC segments name (a leading ``cd`` rebases them); *execution* is
    ``EFFECT_RUN`` or ``EFFECT_BUILD`` from ``shell_paths.EXEC_EFFECTS``; *missing_head* is
    the first command that is not installed. An unreadable command still reports execution.
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
    """Record which tests a red run reported, per test file, collapsing consecutive repeats.

    Read from the runner's node ids; a report with none records nothing.
    """
    signature = test_failure_signature(out)
    if not signature:
        return
    by_file: dict[str, list[str]] = {}
    for node in signature.split():
        by_file.setdefault(node.split("::", 1)[0], []).append(node)
    ledger = execution_context.execution.test_failure_signatures
    for path, nodes in by_file.items():
        seen = ledger.setdefault(path, [])
        sig = " ".join(nodes)
        if not seen or seen[-1] != sig:
            seen.append(sig)


def _stage_moving_test_alert(execution_context: ExecutionContext) -> None:
    """Stage the one-time "read the test itself" corrective for a test whose failures moved.

    Staged like ``_repeat_alert``, since the model is still mid-loop; once per file per query.
    """
    ledger = execution_context.execution.test_failure_signatures or {}
    told = execution_context.nudges.moving_test_told
    for path, seen in sorted(ledger.items()):
        if len(seen) >= 2 and path not in told:
            told.append(path)
            execution_context.nudges.moving_test_alert = path
            return


def bash_validation(call: Call) -> None:
    """Drive the check and run axes from a bash command, success and failure.

    A checker validates the files it names (non-zero charges their budget); an execution
    validates no file and is recorded as a run owing a verdict. A green project-wide check
    covers every pending file. See docs/POLICY.md, "Checks and runs are two axes".
    """
    execution_context = call.execution_context
    command_args = shell_command_args(call.tool_name, call.agent.tool_caps)
    if command_args is None:
        return
    if launched(call.payload):
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
    dirty = execution_context.edits.dirty_written_files
    for target in explicit:
        if is_python_test_filepath(target):
            execution_context.execution.tests_run.add(target)
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
    """Credit the blackboard from a shell command, segment by segment, like the file tools."""
    execution_context = call.execution_context
    if call.status != "ok":
        return
    command_args = shell_command_args(call.tool_name, call.agent.tool_caps)
    if command_args is None:
        return
    if launched(call.payload):
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
                    execution_context.discovery.read_files.add(path)
                    execution_context.discovery.checked_paths.add(path)
                    register_known_path(execution_context, path)
        elif kind == Kind.SEARCH:
            execution_context.discovery.searched = True
            execution_context.discovery.search_tool_calls += 1
        elif kind == Kind.INSPECT:
            for op in operands:
                path = _resolve(op)
                if path:
                    execution_context.discovery.inspected_dirs.add(path)
        elif kind == Kind.WRITE:
            execution_context.action_op_count = int(execution_context.action_op_count) + 1
            for op in operands:
                path = _resolve(op)
                if path and call.agent.is_code_filepath(path):
                    record_code_edit(execution_context, path)
                    execution_context.edits.last_edit_success_path = path
                    enter_post_edit_state(execution_context)
        elif kind in (Kind.EXEC, Kind.UNKNOWN):
            # Marking a dirty file validated (or failed) is handled status-agnostically
            # by bash_validation; here we only count the substantive action.
            execution_context.action_op_count = int(execution_context.action_op_count) + 1
        elif kind == Kind.ENV_DISCOVERY:
            execution_context.env.env_probed = True
        elif kind == Kind.ENV_MUTATE:
            mutations = execution_context.env.env_mutations
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
