"""MCP Bash Server: workspace shell access, the toolchain for building, running and testing.

The approval prompt governs whether a command runs; this validation governs the
command written (program, paths, redirections), not what it does once running. A short
denylist, wrappers unwrapped, one segmenter shared with the client; each call starts at
the workspace root. See docs/SERVERS_DETAILED.md, "workspace/server_bash.py".
"""

import os
from typing import Annotated

from pydantic import Field
import time


from mcp.server.fastmcp import FastMCP
from ...shared.capabilities import (
    tool_caps,
    PLAN_READONLY,
    PLAN_BLOCKED,
    CODE_EXEC,
    JUDGE,
    RECOVERABLE,
    BACKGROUNDABLE,
    DIVERTIBLE,
)
from ...shared.responses import err, ok
from . import bash_exec, bash_jobs
from ...shared.jobs import shell_descriptor
from ...shared.slurm_site import (
    SLURM_COMMANDS,
    slurm_host_from_ssh,
    queued_by_host as slurm_queued_by_host,
    record_host,
    unwatched_by_host as slurm_unwatched_by_host,
    unwatched_hint as slurm_unwatched_hint,
)
from ...shared.shell_paths import (
    normalize_path_arg as _normalize_path_arg,
)
from .bash_validation import WORKSPACE_ROOT, module_preamble, validate_command

mcp = FastMCP(
    "BashServer",
    debug=False,
    log_level="ERROR",
)

# The tail a progress read of a running job gets (see bash_jobs.output).
_RUNNING_OUTPUT = 4 * 1024
# A default that survives an ordinary build or test suite, and a ceiling past which a
# run belongs to the background-job route rather than to a call that blocks the turn.
# A run still going at the ceiling is stopped, and returns what it had printed, which
# keeps a low ceiling cheap.
_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 120



@mcp.tool(**tool_caps(kind="bash",
    caps=[PLAN_READONLY, CODE_EXEC, BACKGROUNDABLE, DIVERTIBLE],
    reversibility=RECOVERABLE,
    non_batch=True,
    fallbacks=['read_file_lines'],
    scope={"args": ["command"], "kind": "command_prefix"},
    risk_note="runs a shell command in the workspace",
    label="Running shell command",
    # Above the tool's own _MAX_TIMEOUT so the inner cap fires first: the caller then
    # gets the "raise the call's own 'timeout'" hint instead of a bare loop timeout.
    timeout_secs=_MAX_TIMEOUT + 30,
))
def bash_run(command: str, timeout: int = _DEFAULT_TIMEOUT,
             background: bool = False) -> dict:
    """Run a controlled bash command inside the workspace root.

    This is the primary way to search file contents (there is no separate grep tool)
    and the primary way to compile, run, validate and test code — invoke the toolchain
    directly with the exact flags the task needs.

        grep -rn "FastMCP" mimir            # search; rg works too. Always -n: a hit
                                            # without its line number costs a read
        gcc solver.c -O3 -o solver.out -lm && ./solver.out
        python -m pytest tests/test_thing.py -q     # also ruff / mypy / py_compile
        make -j8 2>&1                       # bare: a `| head` on a build stops it
                                            # early, a `| tail` drops its first error
        module load cuda && nvcc kernel.cu -o kernel.out
        pip list | grep -i numpy            # then: pip install <pkg>

    Read files with the dedicated read tool, not with `sed -n`/`cat` here: it numbers
    the lines it returns, and the edit tools are addressed by line number. Text that
    arrives through this shell carries no numbers, so acting on it means counting them
    by hand — which is how one edit becomes a series of narrowing re-reads.

    Gating: any command runs. Read-only ones (search, read, inspect, `pip list`,
    `module avail`) run unattended and are available while planning; everything else —
    a write, an execution, an install, a `git` call — is shown to the user for approval
    and is blocked while planning. There is no list of permitted commands to consult:
    reach for the tool the task needs. A refusal names why and what replaces it.

    Syntax that works:
    - chaining (';', '&&', '||', '|'), globbing ('*.py'), and multi-line commands (an
      unquoted newline chains like ';'; a newline inside quotes is literal, so a
      multi-line `python3 -c "..."` body is fine).
    - redirection: '2>&1', '2>/dev/null', '> run.log', '>> run.log', '< input.txt'.
    - 'cd sub && pytest t.py' to work in a subdirectory. Every call starts at the
      workspace root and is its own shell, so a 'cd' or a 'module load' holds for the
      rest of that command only.

    Output: a long result comes back as its beginning and its end, with the middle
    left out and marked, so the first error and the last lines are both there. DO NOT
    pipe a command into `tail` or `head` to shorten it: that is already done, and
    better — a `tail` throws away the first error, and a `head` stops the command
    early, which on a build means an incomplete one reported as finished.

    Limits worth knowing before you retry something:
    - every path — file operand, '-o' target, redirection target, 'cd' destination —
      must be inside the workspace, or the user must approve that path. Ask for it
      rather than looking for a way around it.
    - no shell interpreter (bash, sh, eval, sudo): it would run a command nothing
      here can check, and chaining covers what a script would do. Wrappers that take a
      command are fine ('timeout 60 pytest -q', 'env A=B ./run'). No heredoc — pass the
      text as a quoted argument. No substitution or subshell, and '&' stays refused:
      to detach a run, pass background=True rather than backgrounding it in the shell.
    - no job submission (sbatch/salloc): the cluster tools return a handle this session
      tracks. No 'dd'/'shred'. These are absent by design: report the limitation
      instead of trying another spelling.
    - 'sed -i' rewrites in place (prefer the file-edit tools); a bare 'pip install'
      provisions the interpreter MIMIR itself runs under.
    - an empty result from a probe ('which pdflatex') is a conclusive "absent", not an
      error to retry.

    Args:
        command: Shell command string (single command or simple pipeline).
        timeout: Seconds to wait, capped at 120. Raise it for a real build or suite.
            A run still going at the cap is stopped, but you are given whatever it had
            printed by then — read that before deciding anything.
        background: Set it true from the start for work that plausibly runs longer than
            timeout — a full build, a whole test matrix, an install, a large download,
            a long solver. Do not try a blocking call first to find out: that spends
            the cap and stops the run, and starting detached costs nothing. The command
            is validated and approved exactly as a blocking one; only the waiting
            changes. Its output goes to a log you can read while it runs, so pass the
            build command bare rather than filtering it: a `| tail` keeps that log
            empty until the very end, and there is nothing to read meanwhile. If the
            result
            comes back saying the run is being watched, end your turn on it — you are
            resumed with the results. Say that only when the result says it.
    """
    cwd = WORKSPACE_ROOT

    validation = validate_command(command, cwd)
    if validation["status"] != "ok":
        return validation

    # If a 'module' command is in the chain, define the Lmod function in the
    # wrapper so 'module load cuda && nvcc ...' works within this single shell.
    needs_modules = any(seg and seg[0] == "module" for seg in validation["segments"])
    preamble = module_preamble() if needs_modules else ""

    if background:
        return bash_exec.launch_background(command, cwd, preamble)

    requested_timeout = timeout
    timeout = max(1, min(timeout, _MAX_TIMEOUT))
    result = bash_exec.run_blocking(command, cwd, timeout, preamble=preamble, segments=validation["segments"])
    if timeout != requested_timeout:
        result["timeout_clamped"] = True
        result["requested_timeout"] = requested_timeout
    _note_scheduler(validation["segments"], result)
    return result


# When this process last asked the scheduler what the user has queued. A turn reads the
# queue several times — after a submission, then again for progress — and the reminder
# built on the answer needs it once, not once per call.
_QUEUE_ASKED_AT = 0.0
_QUEUE_EVERY = 60.0


def _note_scheduler(segments: list[list[str]], result: dict) -> None:
    """After a successful Slurm command (local or over ``ssh``), record its host and attach
    the jobs it reported that no descriptor covers.

    From the validator's parse and the exit status, never from command text.
    """
    global _QUEUE_ASKED_AT
    if result.get("status") != "ok" or result.get("returncode") != 0:
        return
    reached = any(_reaches_slurm(segment) for segment in segments)
    if not reached:
        return
    for segment in segments:
        host = slurm_host_from_ssh(segment)
        if host:
            record_host(host)
            break
    now = time.monotonic()
    if now - _QUEUE_ASKED_AT < _QUEUE_EVERY:
        return
    _QUEUE_ASKED_AT = now
    # Asked of every scheduler this workspace reaches, through the same seam the
    # cluster tools use, so one answer and not two. A cluster that does not reply
    # contributes nothing rather than having its jobs reported as unwatched.
    by_host = slurm_unwatched_by_host(slurm_queued_by_host())
    if not by_host:
        return
    result["unwatched"] = [job_id for ids in by_host.values() for job_id in ids]
    result["unwatched_hint"] = slurm_unwatched_hint(by_host)


def _reaches_slurm(argv: list[str]) -> bool:
    """Whether this one command runs a Slurm tool, here or on the far side of an ssh."""
    if not argv:
        return False
    if os.path.basename(argv[0]) in SLURM_COMMANDS:
        return True
    return bool(slurm_host_from_ssh(argv))


_JOB_OPS = ("status", "output", "list")


@mcp.tool(**tool_caps(kind="job", label="Background job: {op}"))
def bash_job(
    op: Annotated[str, Field(
        description="status (default) | output | list")] = "status",
    job_key: str = "",
    run_dir: str = "",
) -> dict:
    """Read a detached bash_run job: its state, its output, or the list of them.

    Read-only, so it stays available while planning and needs no approval.

        status  state of one job: running | done | crashed | unknown, with its
                elapsed time and, once finished, its exit code.
        output  the tail of its log, with the same state. This is what to read when
                a job finishes, and what to read for progress while it runs. While
                it runs, only the last few KB come back; the full log is at 'log'.
        list    every job this host knows about, newest first.

    You do not need to poll a job whose launch result said it is being watched: that
    watcher resumes you when it reaches a terminal state, so end your turn instead.
    Poll when it said nothing of the sort, or when you deliberately want progress
    mid-run (a build's current target, say).

    'unknown' is its own answer, not a variant of 'crashed': the process is gone and
    left no exit code, so it was killed from outside rather than having returned.

    Args:
        job_key: The handle bash_run returned. Required for 'status' and 'output'.
        run_dir: Where the job's files are, when they are not where a bare handle
            resolves to — a run a sub-agent of this session launched, handed over when
            it ran out of time. Comes from the descriptor; never worth composing by
            hand. Leave it out for a run of this session's own.
    """
    if op not in _JOB_OPS:
        return err(f"Unknown op '{op}'.", valid_ops=list(_JOB_OPS))
    if op == "list":
        jobs = bash_jobs.listing()
        return ok(bash_jobs.with_live_descriptors({"jobs": jobs}, jobs))
    if not bash_jobs.valid_key(job_key):
        return err("A job_key from bash_run(background=True) is required.",
                   hint="Use op='list' to see the jobs this host knows about.")
    resolved = ""
    if run_dir:
        resolved, refusal = bash_jobs.resolve_run_dir(run_dir, job_key)
        if refusal is not None:
            return refusal
    if op == "status":
        payload = bash_jobs.state(job_key, resolved)
        return bash_jobs.answer(bash_jobs.with_live_descriptors(payload, [payload], resolved))
    payload = bash_jobs.output(job_key, bash_exec.MAX_OUTPUT,
                                running_max_bytes=_RUNNING_OUTPUT, run_dir=resolved)
    return bash_jobs.answer(bash_jobs.with_live_descriptors(payload, [payload], resolved))


@mcp.tool(**tool_caps(kind="job",
    caps=[PLAN_BLOCKED], reversibility=RECOVERABLE, non_batch=True,
    risk_note="kills a running background command",
    label="Stopping background job",
))
def bash_job_stop(job_key: str, run_dir: str = "") -> dict:
    """Stop a running detached job (SIGTERM, then SIGKILL).

    The whole process group goes, not just the shell — otherwise the compiler it
    launched keeps the machine busy after the job reads as stopped.

    A stopped job reports state 'unknown' afterwards: it was killed rather than having
    returned a status. Whatever it had already written to disk stays written — stopping
    a build halfway leaves a half-built tree to clean up, not a clean slate.

    Args:
        job_key: The handle bash_run returned.
        run_dir: Where the job's files are, as for bash_job: a run a sub-agent of
            this session launched. Comes from the descriptor; leave it out otherwise.
    """
    if not bash_jobs.valid_key(job_key):
        return err("A job_key from bash_run(background=True) is required.")
    resolved = ""
    if run_dir:
        resolved, refusal = bash_jobs.resolve_run_dir(run_dir, job_key)
        if refusal is not None:
            return refusal
    return bash_jobs.answer(bash_jobs.stop(job_key, resolved))


@mcp.tool(**tool_caps(kind="job", label="Watch process {pid}"))
def bash_watch(pid: int, what: str = "", log: str = "") -> dict:
    """Watch a long process already running, so its end reaches you.

    The counterpart of bash_run(background=True) for work that is already going: a
    process a script of yours backgrounded, a nohup started outside this session,
    anything long that no tool call owns. Those are watched by nothing — tracking is
    the record a launch writes, and a process started elsewhere wrote none — so nobody
    is told when it ends and it does not keep this workspace's server alive while it
    runs.

    For a *Slurm* job use slurm_watch(job_id) instead: a job id is not a pid, its
    liveness lives in the scheduler, and this tool can only watch a process on this
    host.

    What it cannot give you is a status. The exit-code trap is installed by the launch,
    and nothing can add it to a process already running, so when the pid goes the state
    is 'unknown' — "it ended, and its own output says how". That is still the wake you
    need; it is not a verdict.

    Args:
        pid: The process id to watch. It must be running on this host.
        what: What this process is, for the handle and for whoever reads it later.
        log: Where it writes its output, if you know — recorded so a later read goes
            to the right file, since an adopted job has no log of its own.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return err("pid must be a number.")
    if pid <= 0:
        return err("pid must be a positive process id.")
    # Resolved for a reader, not confined: nothing here opens this path. It is
    # recorded so a later answer can name where the output went, and whoever goes to
    # read it does so through the file tools, which confine themselves. Confining it
    # here would refuse the ordinary case — a job writing to the scratch filesystem,
    # which is outside every workspace by design.
    log = _normalize_path_arg(log, WORKSPACE_ROOT) or ""
    handle = bash_jobs.adopt(pid, what or f"adopted pid {pid}", WORKSPACE_ROOT, log)
    if handle is None:
        return err(f"No process {pid} is running here.",
                   hint="Check the pid — a process that has already ended cannot be "
                        "watched, and one on another host is not visible from here.")
    return ok({
        **handle,
        "state": "running",
        # What was adopted, not what will happen next — the same rule as a launch's
        # note a few hundred lines up: a resume is the client's to promise, and only
        # once it is holding a watcher.
        "headline": f"Pid {pid} is recorded here, and handed over to be watched.",
        # Plural, as every answer about work this call did not start is: see the same
        # key in bash_job_status. Singular is a launch, and a launch is what the
        # client reads only from a tool declaring BACKGROUNDABLE.
        "background_jobs": [shell_descriptor(handle["job_key"], handle["job_dir"])],
    })


_VERDICT_VALUES = ("pass", "fail", "unknown", "blocked", "rejected")


@mcp.tool(**tool_caps(kind="verdict",
    caps=[JUDGE],
    arg_roles={
        "verdict": ["verdict"], "verdict_reason": ["reason"], "verdict_scope": ["run"],
    },
    label="Verdict: {verdict}",
))
def report_verdict(
    verdict: Annotated[str, Field(
        description=(
            "The judgement: one of 'pass', 'fail', 'unknown', 'blocked', 'rejected'. "
            "Required — a reason without a verdict judges nothing."
        ),
        json_schema_extra={"enum": list(_VERDICT_VALUES)},
    )],
    reason: Annotated[str, Field(
        description=(
            "What in the output shows it — the number, message or behaviour you read. "
            "\"it worked\" is not a reason."
        ),
    )],
    run: Annotated[str, Field(
        description=(
            "Which run is being judged, as its command; a recognisable fragment is "
            "enough. Optional: left out, a 'pass' or 'rejected' settles the most recent "
            "run, so name the run whenever you are judging an earlier one. 'fail' and "
            "'unknown' address everything outstanding, which is why a losing candidate "
            "belongs under 'rejected' and not under 'fail'; 'blocked' addresses every "
            "failed run the same way."
        ),
    )] = "",
) -> dict:
    """State what a run's output showed. Recommended after any execution you had to read.

    Exit 0 means a program reached its end, never that its answer is right, and nothing
    downstream can read the output for you. Until you report it, the run stands as
    something that happened and nothing more. A run whose output settles nothing, and a
    syntax, lint or type check, need no report — an exit code is the whole finding
    there, and all it establishes: that the file parses and lints, never that the result
    is right.

    Report as soon as you have read the output, in the same step you would move on:

        verdict="pass",    reason="l2_rel=3.1e-4 against the analytic solution, under the 1e-3 bound"
        verdict="fail",    reason="energy grows from 1.56 to 4.02 — the absorbing layer reflects"
        verdict="unknown", reason="only prints 'Simulation completed.', nothing about correctness"
        verdict="rejected", reason="32 threads: 0.0642 s against the incumbent's 0.0556 — slower, not kept"
        verdict="blocked", reason="cmake needs a configured build tree; there is none here"

    Name the number, message or behaviour you read it from; "it worked" is not a
    reason. `fail` on a green run is expected sometimes and is not a setback — fix and
    re-run.

    `unknown` is the honest answer when the output does not settle the question, and it
    is a complete one: it stands as reported and counts against nothing. When the
    question is worth settling, it is also a starting point — say what would settle it
    and get it, in proportion to what the answer is worth — a reference implementation,
    a documented value, an analytical limit, an invariant, a refinement trend. Read the
    documentation, the literature or the repository, and fetch from the
    web if you can reach it. When the standard is the user's to set rather than yours to
    find — their conventions, their acceptance criteria, their data — ask them. Then
    report again. If it stays out of reach, say so and name what is unverified: you are
    not blocked on being certain, and the user is the last judge of what you could not
    settle.

    `rejected` is for a run that worked and lost: it measured cleanly, you read the
    number, and the candidate was not an improvement. That is the ordinary outcome of a
    search, not a defect — it costs no retry budget, settles only the run it names, and
    is the honest word for most iterations of an optimisation loop. Do not spend `fail`
    on it: `fail` says the run is broken, and it withholds credit from every other run
    still awaiting one.

    `blocked` is for a run that failed on a wall that is not in your change: a build to
    configure, a package that is not installed, a dataset, an allocation. It does not make
    the run a success — it stays reported as not completed, with your reason attached. It
    says only that repairing it is not the next piece of this work, which stops it counting
    against you as an unfinished task. Use it when reaching a first run would cost a step of
    its own; never to avoid a fix you could make.
    """
    value = (verdict or "").strip().lower()
    if value not in _VERDICT_VALUES:
        return err(f"verdict must be one of {', '.join(_VERDICT_VALUES)} — got '{verdict}'.")
    if not (reason or "").strip():
        return err("reason is required — name what in the output shows the verdict.")
    return ok({"verdict": value, "reason": reason.strip(), "run": (run or "").strip()})


if __name__ == "__main__":
    mcp.run()
