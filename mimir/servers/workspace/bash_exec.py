"""Running a validated shell command: the environment it gets, the blocking run with its
ticks, its cap and the user's divert, what a finished run reports, and the detached
launch. Every run is a ``bash_jobs`` job, its output in a file rather than a pipe.
"""

import os
import sys
import time

from ...shared import build_progress, run_channel
from ...shared.jobs import shell_descriptor
from ...shared.module_env import MODULE_ENV_PASSTHROUGH as _MODULE_ENV_PASSTHROUGH
from ...shared.responses import err, ok
from ...shared.shell_paths import ShellParseError as _ShellParseError, parse_segments as _parse_segments
from . import bash_jobs
from .bash_validation import REAL_HOME

MAX_OUTPUT = 64 * 1024

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


def safe_env(cwd: str) -> dict:
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


def _tick(elapsed: float) -> float:
    for boundary, interval in _TICKS:
        if elapsed < boundary:
            return interval
    return _TICK_MAX


def run_blocking(
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
            command, cwd, safe_env(cwd), preamble=preamble,
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
                stdout, stderr, truncated = bash_jobs.streams(job_key, MAX_OUTPUT)
                payload = _finished_payload(command, cwd, proc.returncode,
                                            stdout, stderr, truncated, segments)
                bash_jobs.discard(job_key)
                return payload

            now = time.monotonic()

            if now >= deadline:
                stopped = bash_jobs.stop(job_key)
                stdout, stderr, truncated = bash_jobs.streams(
                    job_key, MAX_OUTPUT, keep="tail")
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
                        job_key, MAX_OUTPUT, keep="tail")
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
                        "background_job": shell_descriptor(job_key,
                                                          launched["job_dir"]),
                    })
                    if truncated:
                        payload["truncated"] = True
                    return payload

            time.sleep(_tick(now - started))
    finally:
        run_channel.clear(_DIVERT_CHANNEL, job_key)


def launch_background(command: str, cwd: str, preamble: str) -> dict:
    """Detach an already-validated command and return the watcher's handle.

    The blocking path launches the same way (see :func:`run_blocking`); what differs is only
    that nothing here waits, and the job directory is a handle rather than a scratch
    buffer.
    """
    try:
        launched = bash_jobs.launch(command, cwd, safe_env(cwd), preamble=preamble,
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
        "background_job": shell_descriptor(job_key, launched["job_dir"]),
    })

