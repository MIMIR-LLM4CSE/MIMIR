"""MCP Bash Server: workspace shell access, the toolchain for building, running and testing.

The approval prompt governs whether a command runs; this validation governs the
command written (program, paths, redirections), not what it does once running. A short
denylist, wrappers unwrapped, one segmenter shared with the client; each call starts at
the workspace root. See SERVERS_DETAILED.md, "workspace/server_bash.py".
"""

import os
from typing import Annotated

from pydantic import Field
import sys
import time


from mcp.server.fastmcp import FastMCP
from .._shared.capabilities import (
    tool_caps,
    PLAN_READONLY,
    PLAN_BLOCKED,
    CODE_EXEC,
    JUDGE,
    RECOVERABLE,
    BACKGROUNDABLE,
    DIVERTIBLE,
)
from .._shared.responses import err, ok
from .._shared.module_env import MODULE_ENV_PASSTHROUGH as _MODULE_ENV_PASSTHROUGH
from . import bash_jobs
from .._shared import build_progress
from .._shared import run_channel
from .._shared.slurm_site import (
    SLURM_COMMANDS,
    slurm_host_from_ssh,
    queued_by_host as slurm_queued_by_host,
    record_host,
    unwatched_by_host as slurm_unwatched_by_host,
    unwatched_hint as slurm_unwatched_hint,
)
from .._shared.shell_paths import (
    ShellParseError as _ShellParseError,
    normalize_path_arg as _normalize_path_arg,
    parse_segments as _parse_segments,
)
from .bash_validation import REAL_HOME, WORKSPACE_ROOT, module_preamble, validate_command

mcp = FastMCP(
    "BashServer",
    debug=False,
    log_level="ERROR",
)

_MAX_OUTPUT = 64 * 1024
# The tail a progress read of a running job gets (see bash_jobs.output).
_RUNNING_OUTPUT = 4 * 1024
# A default that survives an ordinary build or test suite, and a ceiling past which a
# run belongs to the background-job route rather than to a call that blocks the turn.
# A run still going at the ceiling is stopped, and returns what it had printed, which
# keeps a low ceiling cheap.
_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 120

# How often the blocking path looks at the job it is waiting on. Short commands are
# the common case and must stay as immediate as a pipe made them, so the first ticks
# are tiny and only a genuinely long run relaxes into a lazy poll.
_TICKS = ((0.1, 0.005), (1.0, 0.02), (5.0, 0.05))
_TICK_MAX = 0.2
# The divert sidecar is stat'ed at most this often, so a long wait does not turn into
# a stat storm on the shared state dir.
_DIVERT_POLL_S = 0.25
# How often a blocking run's output is read for a build count. A build's count moves
# over seconds, and each read is a seek and 8 KiB, so once a second is plenty.
_PROGRESS_POLL_S = 1.0
# Which run channel a blocking run publishes under. The channel is the blocking
# tool's own registered name, so the client can address it off the tool row it is
# diverting without either end spelling the other's name out.
_DIVERT_CHANNEL = "bash_run"


# Commands whose exit status 1 means "nothing found" rather than "failed" — a
# legitimate answer the model would otherwise re-run unchanged. A clean no-match
# (rc 1, empty stdout) is reported as success; a real failure exits 2 and stays an error.
_NO_MATCH_COMMANDS = {"grep", "rg", "which"}


# Proxy settings passed through from the user's environment as they are, so network
# commands use the same proxy (and no_proxy) as the user's shell instead of hanging.
_PROXY_ENV_PASSTHROUGH = (
    "http_proxy", "https_proxy", "ftp_proxy", "no_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "NO_PROXY",
)


def _safe_env(cwd: str) -> dict:
    """Minimal subprocess environment, PATH led by this interpreter's directory (MIMIR's Python)."""
    interp_dir = os.path.dirname(os.path.abspath(sys.executable))
    ambient_path = os.environ.get("PATH", "")
    path = interp_dir + os.pathsep + ambient_path if interp_dir else ambient_path
    env = {
        "PATH": path,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        # Inherited, not pinned to the workspace: confinement comes from the operand
        # path check, and pinning only detached tools from the user's own config and
        # caches (~/.local site-packages, ~/.condarc, ~/texmf). Startup files are kept
        # out by 'bash --noprofile --norc' plus BASH_ENV below, which is what that
        # concern actually needed.
        "HOME": REAL_HOME,
        # Prevent non-interactive bash from loading an arbitrary startup file.
        "BASH_ENV": "",
        "PWD": cwd,
        # kpathsea reads these from the env, pinning the TeX sandbox regardless of the
        # site's texmf.cnf: no \write18, and 'paranoid' file access. Backstop for the
        # shell-escape flags already denylisted at validation.
        "shell_escape": "f",
        "openout_any": "p",
        "openin_any": "p",
    }
    # Let Lmod find the site's modulefiles when a 'module' command is in play, and
    # let a network command reach the site's proxy. These are config/path vars only —
    # they do not themselves run code.
    for var in _MODULE_ENV_PASSTHROUGH + _PROXY_ENV_PASSTHROUGH:
        val = os.environ.get(var)
        if val:
            env[var] = val
    return env


def _classify_runtime_error(stderr: str) -> str | None:
    text = stderr.lower()
    if any(k in text for k in [
        "modulenotfounderror",
        "importerror",
        "no module named",
        "command not found",
        "cannot open shared object file",
        "libstdc++",
        "glibc",
        "undefined symbol",
        "qt.qpa",
        "matplotlib",
        "backend",
        "thread",
    ]):
        return "environment_mismatch_likely"
    return None


def _is_clean_no_match(segments: list[list[str]] | None, returncode: int, stdout: str) -> bool:
    """True when the last segment's non-zero exit means "nothing found" (``_NO_MATCH_COMMANDS``)."""
    if returncode < 1 or stdout.strip():
        return False
    if not segments or not segments[-1]:
        return False
    head = segments[-1][0]
    if head == "which":
        return True
    return returncode == 1 and head in _NO_MATCH_COMMANDS


def _output_targets(command: str) -> list[str]:
    """Files the command redirects its own output to, from the validator's parse.

    Recorded on the job so a later empty log is explained rather than guessed about.
    """
    try:
        return [t for seg in _parse_segments(command) for t in seg.write_targets]
    except _ShellParseError:
        return []  # it parsed at validation; a change of heart here is not fatal


def _finished_payload(command: str, cwd: str, returncode: int, stdout: str,
                      stderr: str, truncated: bool,
                      segments: list[list[str]] | None) -> dict:
    """What a run that reached its own end reports — unchanged by backgrounding."""
    if returncode == 0:
        payload = ok({
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "cwd": cwd,
        })
    elif _is_clean_no_match(segments, returncode, stdout):
        payload = ok({
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "cwd": cwd,
            "matches": 0,
            "note": "The command ran correctly and found nothing. This is a "
                    "conclusive negative answer, not a failure — treat the "
                    "result as 'absent' and do not re-run the command.",
        })
    else:
        classification = _classify_runtime_error(stderr)

        payload = err(
            "Command returned a non-zero exit code.",
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            cwd=cwd,
        )

        if classification:
            payload["diagnostic"] = classification
            payload["hint"] = (
                "The failure appears to be due to the runtime environment "
                "(missing or incompatible dependencies), not shell syntax. "
                "Consider checking the Python environment or loaded modules."
            )
        else:
            payload["hint"] = (
                "Read stderr before doing anything else: a non-zero exit is "
                "often a real finding (a failing test, an absent file, a "
                "compile error) rather than a malformed command. Act on what "
                "stderr says; do not re-run the same command unchanged."
            )

    if truncated:
        payload["truncated"] = True
    return payload


def _job_descriptor(job_key: str, job_dir: str) -> dict:
    """The handle the client watcher polls.

    Data, not a tool name in loop code: the client reads status_op / summary_op off it
    and polls them generically, which is how a run gets watched without the agent
    spending a model call per poll.
    """
    return {
        "server": "bash",
        "kind": "shell-command",
        "job_key": job_key,
        "run_dir": job_dir,
        "status_op": {"tool": "bash_job", "args": {"job_key": job_key}},
        "summary_op": {"tool": "bash_job",
                       "args": {"op": "output", "job_key": job_key}},
    }


def _tick(elapsed: float) -> float:
    for boundary, interval in _TICKS:
        if elapsed < boundary:
            return interval
    return _TICK_MAX


def _run(
    command: str,
    cwd: str,
    timeout: int,
    preamble: str = "",
    segments: list[list[str]] | None = None,
) -> dict:
    """Run *command* to completion, the cap, or a user divert; output goes to a file.

    At the cap the group is killed and the partial output returned; on a divert the run is
    left going and a background handle returned (the only promotion, always a user act).
    """
    try:
        launched = bash_jobs.launch(
            command, cwd, _safe_env(cwd), preamble=preamble,
            output_targets=_output_targets(command),
            split_stderr=True, ephemeral=True,
        )
    except OSError as exc:
        return err(f"Could not start the command: {exc}", cwd=cwd)

    job_key = launched["job_key"]
    proc = launched["proc"]
    started = time.monotonic()
    deadline = started + timeout
    run_channel.publish(_DIVERT_CHANNEL, job_key, launched["pid"], command, cwd,
                        float(timeout))

    try:
        next_divert_check = started + _DIVERT_POLL_S
        next_progress_check = started + _PROGRESS_POLL_S
        last_progress: tuple[float, str] | None = None
        while True:
            if proc.poll() is not None:
                # proc.poll(), not the exit_code file: the trap that writes that file
                # is replaced by any command installing its own EXIT trap, and a
                # blocking run must report the status its command actually returned.
                stdout, stderr, truncated = bash_jobs.streams(job_key, _MAX_OUTPUT)
                payload = _finished_payload(command, cwd, proc.returncode,
                                            stdout, stderr, truncated, segments)
                bash_jobs.discard(job_key)
                return payload

            now = time.monotonic()

            if now >= deadline:
                stopped = bash_jobs.stop(job_key)
                stdout, stderr, truncated = bash_jobs.streams(
                    job_key, _MAX_OUTPUT, keep="tail")
                payload = err(
                    f"Command timed out after {timeout}s and was stopped.",
                    returncode=stopped.get("returncode", 124),
                    stdout=stdout,
                    stderr=stderr,
                    partial=True,
                    elapsed_s=round(now - started, 1),
                    cwd=cwd,
                    hint="The output above is everything the command had printed "
                         "before it was stopped — read it before deciding anything. "
                         "If the work genuinely takes this long, start it again with "
                         "background=True rather than with a larger timeout.",
                )
                if truncated:
                    payload["truncated"] = True
                bash_jobs.discard(job_key)
                return payload

            if now >= next_progress_check:
                next_progress_check = now + _PROGRESS_POLL_S
                # From the output, not the command: a make buried in a chain still
                # prints its count into this log. Only a change is written, and a
                # count that disappears (the build ended, the next step began) is
                # retracted rather than left standing.
                progress = build_progress.from_file(
                    bash_jobs.progress_path(job_key))
                if progress != last_progress:
                    last_progress = progress
                    percent, phase = progress if progress else (None, "")
                    run_channel.update(_DIVERT_CHANNEL, job_key, phase, percent)

            if now >= next_divert_check:
                next_divert_check = now + _DIVERT_POLL_S
                if run_channel.requested(_DIVERT_CHANNEL, job_key):
                    bash_jobs.promote(job_key, "diverted")
                    bash_jobs.ABANDONED.append(proc)
                    stdout, stderr, truncated = bash_jobs.streams(
                        job_key, _MAX_OUTPUT, keep="tail")
                    payload = ok({
                        "cwd": cwd,
                        "command": command,
                        "stdout": stdout,
                        "stderr": stderr,
                        # No returncode: the run has not produced one. Inventing a
                        # status the command did not return is the one lie the caller
                        # cannot detect.
                        "partial": True,
                        "reason": "diverted",
                        "elapsed_s": round(now - started, 1),
                        "job_key": job_key,
                        "pid": launched["pid"],
                        "log": launched["log"],
                        "note": "The user moved this run to the background while it "
                                "was still going. It was not killed: the output "
                                "above is what it had produced at that moment, and "
                                "it continues as job '" + job_key + "'. Do not "
                                "run it again.",
                        "background_job": _job_descriptor(job_key,
                                                          launched["job_dir"]),
                    })
                    if truncated:
                        payload["truncated"] = True
                    return payload

            time.sleep(_tick(now - started))
    finally:
        run_channel.clear(_DIVERT_CHANNEL, job_key)


def _launch_background(command: str, cwd: str, preamble: str) -> dict:
    """Detach an already-validated command and return the watcher's handle.

    The blocking path launches the same way (see :func:`_run`); what differs is only
    that nothing here waits, and the job directory is a handle rather than a scratch
    buffer.
    """
    try:
        launched = bash_jobs.launch(command, cwd, _safe_env(cwd), preamble=preamble,
                                     output_targets=_output_targets(command))
    except OSError as exc:
        return err(f"Could not start the background job: {exc}", cwd=cwd)

    job_key = launched["job_key"]
    return ok({
        "cwd": cwd,
        "command": command,
        "job_key": job_key,
        "pid": launched["pid"],
        "log": launched["log"],
        # What was started, not what will happen next: whether anything watches this
        # run is the client's to know, and it says so itself when it registers one.
        # A server that promises a resume it does not perform produces a model that
        # ends its turn on a guarantee nobody is holding.
        "note": "Started in the background; this call returns before it finishes. "
                "Its log is readable at the path above while it runs.",
        "background_job": _job_descriptor(job_key, launched["job_dir"]),
    })


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
        return _launch_background(command, cwd, preamble)

    requested_timeout = timeout
    timeout = max(1, min(timeout, _MAX_TIMEOUT))
    result = _run(command, cwd, timeout, preamble=preamble, segments=validation["segments"])
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
        return ok(_with_live_descriptors({"jobs": jobs}, jobs))
    if not bash_jobs.valid_key(job_key):
        return err("A job_key from bash_run(background=True) is required.",
                   hint="Use op='list' to see the jobs this host knows about.")
    resolved = ""
    if run_dir:
        resolved, refusal = _resolve_run_dir(run_dir, job_key)
        if refusal is not None:
            return refusal
    if op == "status":
        payload = bash_jobs.state(job_key, resolved)
        return _job_payload(_with_live_descriptors(payload, [payload], resolved))
    payload = bash_jobs.output(job_key, _MAX_OUTPUT,
                                running_max_bytes=_RUNNING_OUTPUT, run_dir=resolved)
    return _job_payload(_with_live_descriptors(payload, [payload], resolved))


def _resolve_run_dir(run_dir: str, job_key: str) -> tuple[str, dict | None]:
    """``(run_dir, None)`` for an explicit job directory inside this session's own state dir
    (its sub-agents' runs), else ``("", error)``.
    """
    from .._shared.state_paths import session_state_dir
    resolved = os.path.realpath(os.path.expanduser(run_dir))
    root = os.path.realpath(session_state_dir())
    if os.path.commonpath([resolved, root]) != root:
        return "", err("That run_dir is outside this session.",
                       hint="A run_dir names a job of this session or of one of its "
                            "sub-agents; leave it out to read this session's own job.")
    if os.path.basename(os.path.dirname(resolved)) != "jobs":
        return "", err("That run_dir is not a job directory.",
                       hint="A job directory sits under a 'jobs/' directory and is "
                            "named like the handle bash_run returned.")
    if not bash_jobs.valid_key(os.path.basename(resolved)):
        return "", err("That run_dir is not named like a job.",
                       hint="Its last segment is the handle bash_run returned.")
    if job_key and os.path.basename(resolved) != job_key:
        return "", err("That run_dir names a different job than job_key does.",
                       hint="Pass the pair as the descriptor handed them over, or "
                            "leave run_dir out.")
    return resolved, None


def _with_live_descriptors(payload: dict, jobs: list[dict], run_dir: str = "") -> dict:
    """Attach a watcher handle (``background_jobs``, plural) for every job still running.

    Asking about a run is what lets a client that lost its watchers put them back.
    """
    live = [_job_descriptor(j["job_key"],
                            run_dir or bash_jobs.job_dir(j["job_key"]))
            for j in jobs
            if isinstance(j, dict) and j.get("state") == "running" and j.get("job_key")]
    if live:
        payload["background_jobs"] = live
    return payload


def _job_payload(payload: dict) -> dict:
    """Wrap a job-layer answer in ``ok()``, unless it carries ``error`` (a key naming no job)."""
    reason = payload.pop("error", "")
    if reason:
        return err(reason, hint="Use bash_job(op='list') to see this session's jobs.",
                   **payload)
    return ok(payload)


@mcp.tool(**tool_caps(kind="job",
    caps=[PLAN_BLOCKED], reversibility=RECOVERABLE, non_batch=True,
    risk_note="kills a running background command",
    label="Stopping background job",
))
def bash_job_stop(job_key: str) -> dict:
    """Stop a running detached job (SIGTERM, then SIGKILL).

    The whole process group goes, not just the shell — otherwise the compiler it
    launched keeps the machine busy after the job reads as stopped.

    A stopped job reports state 'unknown' afterwards: it was killed rather than having
    returned a status. Whatever it had already written to disk stays written — stopping
    a build halfway leaves a half-built tree to clean up, not a clean slate.

    Args:
        job_key: The handle bash_run returned.
    """
    if not bash_jobs.valid_key(job_key):
        return err("A job_key from bash_run(background=True) is required.")
    return _job_payload(bash_jobs.stop(job_key))


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
        "background_jobs": [_job_descriptor(handle["job_key"], handle["job_dir"])],
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
