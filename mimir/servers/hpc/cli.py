"""Talking to a Slurm scheduler, local or over ssh: the one seam every call goes through.

Every command reaches Slurm through ``run_bash`` or ``run_argv``, called by attribute
(``cli.run_bash``) so a test replaces it in one place."""
from __future__ import annotations

import os
import re
import subprocess
from .._shared.responses import err, ok
from .._shared.slurm_site import (
    LOCAL as _LOCAL_HOST,
    hosts as slurm_hosts,
    name_of as _host_name,
    resolve_host as _resolve_host,
    argv_for as slurm_argv,
    exec_argv as slurm_exec_argv,
)
from .._shared.slurm_nodes import (
    SINFO_NODES_FORMAT as _SINFO_NODES_FORMAT,
    parse_sinfo_nodes as _parse_sinfo_nodes,
)

TIMEOUT_READ = 15


TIMEOUT_ALLOC = 20


TIMEOUT_SUBMIT = 30


MAX_OUTPUT = 128 * 1024


# Where a batch submission's own bookkeeping lives — the script, the id, the state a
# poll settled it at — under the submitting session's state dir, like every other
# artefact belonging to a conversation (env-overridable for tests).
#
# Not the job's output: that goes where the job runs, which is the only directory it is
# certain to be able to write. See _shared/slurm_script.sbatch_header.
#
# Per session, resolved per call. A Slurm job outlives its session, sometimes by days,
# which is the one way these differ from detached shell jobs — so a job is never
# *unreachable* from another conversation: _find_job_dir below looks in the sibling
# sessions read-only and says where it came from. Per session buys two things: two
# conversations submitting at the same moment cannot land on each other's directory, and a
# conversation's submissions go with it when it is deleted.
def no_scheduler_hint(host: str = _LOCAL_HOST) -> str:
    """What to do when a Slurm command did not answer. Self-documenting on purpose.

    Two different failures, and the cure is only ever offered for the one it fits. A
    scheduler that was *named* and did not answer is a host to check — the inventory is
    not the problem. Nothing reachable at all is the case this learning exists for, and
    there the answer has to be a thing the user is about to do anyway: reach a
    scheduler over ``ssh`` once, which adds it.
    """
    known = slurm_hosts()
    if host:
        return (f"That went to {host}; check the host answers — "
                "slurm_schedulers() lists what this workspace reaches.")
    if known:
        return ("No Slurm command answered on this host. Reachable instead: "
                + ", ".join(_host_name(h) for h in known)
                + " — pass one as scheduler=.")
    return ("No Slurm command answered on this host, and no other scheduler has been "
            "reached yet. If one lives elsewhere, read its queue through the shell "
            "once — ssh <host> \"squeue -u $USER\" — and it joins this workspace's "
            "schedulers, or name it with slurm_schedulers(add=\"<host>\").")


def chosen_host(scheduler: str) -> tuple[str, dict | None]:
    """The scheduler a call goes to, or the refusal naming the choice left to make.

    The refusal is the useful half: with two schedulers reachable, "which one" is a
    decision about the work — where the inputs are, which has the GPUs — and this says
    so rather than picking one. See ``slurm_site.resolve_host``.
    """
    host, refusal = _resolve_host(scheduler)
    return host, (err(refusal) if refusal else None)


def run_bash(script: str, timeout: int, stdin: str | None = None, *,
              host: str = _LOCAL_HOST) -> dict:
    """Run *script* on *host*'s scheduler, optionally feeding it *stdin*.

    Without *stdin* the command gets none at all: the server's own is its MCP protocol
    pipe, and a command that reads it (``cat``, an ``ssh`` without ``-n``) eats the
    client's traffic. With it, the text is handed over on a pipe we opened — which is
    how a batch script reaches ``sbatch`` without any file having to exist on the host
    that runs it.
    """
    try:
        res = subprocess.run(
            slurm_argv(script, host=host, piped=stdin is not None),
            input=stdin,
            stdin=None if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return command_result(res)
    except subprocess.TimeoutExpired:
        return err(f"Command timed out after {timeout}s.",
                   hint="Narrow the query or increase timeout.")
    except Exception as e:
        return err(str(e))


def command_result(res: subprocess.CompletedProcess) -> dict:
    """A finished command as a tool answer; a non-zero exit is an error."""
    fields = {"returncode": res.returncode,
              "stdout": res.stdout[:MAX_OUTPUT], "stderr": res.stderr[:MAX_OUTPUT]}
    if len(res.stdout) > MAX_OUTPUT or len(res.stderr) > MAX_OUTPUT:
        fields["truncated"] = True
    if res.returncode == 0:
        return ok(fields)
    return err(res.stderr.strip() or f"Exited with code {res.returncode}.", **fields)


def run_argv(argv: list[str], timeout: int, *, host: str = _LOCAL_HOST) -> dict:
    """Run a command as argv on *host* — no shell, so no argument can inject a second one.

    With a remote controller one shell is unavoidable: ``ssh`` has no argv to hand over,
    only a command line for the far side to parse. So each element is quoted on the way
    out and the property survives in the form that matters — what the remote shell
    splits is exactly the argv built here, and an argument cannot become a command.
    """
    try:
        res = subprocess.run(
            slurm_exec_argv(argv, host=host), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout,
        )
        return command_result(res)
    except subprocess.TimeoutExpired:
        return err(f"Command timed out after {timeout}s.")
    except Exception as e:
        return err(str(e))


def current_user() -> str:
    import getpass
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "")


def validate_time(value: str) -> bool:
    # Accept Slurm-like forms: MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS
    return bool(re.fullmatch(r"\d+(-\d{1,2}(:\d{2}(:\d{2})?)?|(:\d{2}){0,2})", value))


def validate_mem(value: str) -> bool:
    # Examples: 8G, 32000M, 2T
    return bool(re.fullmatch(r"\d+[KMGTP]", value.upper()))


COMMENT_MAX = 512


def validate_comment(value: str) -> str | None:
    """Return why *value* cannot be a Slurm --comment, or None when it can.

    Free text, so the only hard rules are a length Slurm will store and a single
    line: a newline in a value that lands in a ``#SBATCH`` directive turns the rest
    of it into directives of the caller's choosing, and sacct renders the field on
    one line regardless.
    """
    if not value:
        return None
    if len(value) > COMMENT_MAX:
        return f"comment is too long ({len(value)} chars). Keep it under {COMMENT_MAX}."
    bad = [ch for ch in value if ord(ch) < 0x20 or ord(ch) == 0x7F]
    if bad:
        return ("Invalid comment: control characters are not allowed "
                f"(found {bad[0]!r}). Keep it to a single line of plain text.")
    return None


def lines_of(result: dict) -> list[str]:
    """The non-empty lines of a command's output, upper-cased. [] when it failed."""
    if result.get("status") != "ok":
        return []
    return [line.strip().upper()
            for line in (result.get("stdout") or "").splitlines() if line.strip()]


def sinfo_nodes(host: str = _LOCAL_HOST) -> list[dict]:
    """Degraded node list for a cluster whose `scontrol show node` is restricted.

    sinfo is readable everywhere but does not carry the architecture, so callers are
    told the field is unknown rather than being handed a wrong default.
    """
    result = run_bash(f"sinfo -N -h -o '{_SINFO_NODES_FORMAT}'",
                       TIMEOUT_READ, host=host)
    if result["status"] != "ok":
        return []
    return _parse_sinfo_nodes(result.get("stdout", ""))
