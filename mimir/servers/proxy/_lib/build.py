"""Building a registered proxy before it is measured.

A Python proxy needs no build step: the file the ratchet edits is the file that
runs. A compiled one breaks that identity — the edited ``solver.cpp`` is in
``optimize_paths``, but the measured program is ``./solver``, an artifact some
earlier compiler produced. The ratchet fingerprints sources only
(``tree_snapshot.fingerprint``), so nothing in it can notice the two disagreeing:
a forgotten rebuild measures the previous binary and the verdict lands on code
that never ran, and a ``reset_to_best`` restores sources while leaving the
artifact from the attempt it just rejected.

So the server builds, rather than asking anyone to remember to. ``build_cmd`` on
the registration is executed here, once per evaluation run (never once per
replicate) and before any case is measured; a build that fails ends the run with
no verdict and no ledger entry.

INCREMENTALITY IS THE PROJECT'S BUILD SYSTEM'S JOB. The build is a command the
project already owns, so ``make``/``ninja``/``cmake`` decide what to recompile —
from mtimes, which is why ``tree_snapshot`` stamps restored files as new, and
only the ones whose content actually changed. Three things here can defeat that,
so all three are settled deliberately:

* The command is argv, never a shell line: split with ``shlex`` and executed
  directly. ``make -C <dir> <target>`` covers most of it; anything needing
  several commands, a ``module load`` or a redirect belongs in a wrapper script
  named here.
* The *spelling* of the build directory. ``build_cwd`` is kept as declared,
  symlinks unresolved, and exported as ``PWD`` — the only channel that carries
  it, since the child's ``getcwd()`` comes back resolved whichever spelling
  ``Popen`` was handed. Scratch space is normally presented through a link, and
  CMake compares against the absolute path it was configured with.
* The *environment*, which is inherited — and what is inherited is the
  environment the MCP server started with, not the shell the user builds in by
  hand. When the two disagree on the compiler or on ccache, each build
  invalidates the other's artifacts. ``build_env`` pins what decides this, and
  the log records the effective toolchain and whether ccache was reachable.

"Before any case is measured" does not have to mean "on the node that measures".
On a cluster the build can be a Slurm job of its own — see ``_lib/placement.py``
and ``_ops/slurm.submit_eval`` — because a node dedicated to GPU simulation is not
a node to compile on. What this module does is unchanged either way; what changes
is which machine it runs on. The two halves meet through the run directory
(``build.json``, ``build.log``, the launch snapshot, the executable), which is why
the split assumes the build node and the run node see the same filesystem.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time

import proc_run

from _lib import procs, store

# Long enough for a real project's first build, bounded so a wedged build cannot
# hold a run for the whole 24 h measurement budget. Half an hour was not: a GPU
# project's full rebuild after one touched header reached 50% of its targets in
# exactly that, so the default cut off every clean build it was meant to cover and
# reported it as a failure. A build is incremental, so the cost of a generous
# ceiling is a wedged build noticed later, not work redone.
_DEFAULT_TIMEOUT_S = 7200.0
_MAX_TIMEOUT_S = 6 * 3600.0

# Tokens that only mean anything to a shell. Passing them to execvp() would hand
# ``make`` a literal "&&" argument and fail three minutes later with a message
# about a missing target, so they are refused up front with the remedy named.
_SHELL_METACHARS = ("&&", "||", ";", "|", "<", ">", "`", "$(", "&")

_SHELL_HINT = (
    "build_cmd is run as argv, not through a shell. Use 'make -C <dir> <target>', "
    "or put the sequence in a wrapper script and name the script here."
)

_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# The variables that decide whether a build is incremental, recorded in the log
# whatever their origin. A ccache miss and a swapped compiler both look exactly like
# "it rebuilt everything again", and neither left any trace to read afterwards.
_ENV_OF_INTEREST = ("CC", "CXX", "FC", "F77", "CCACHE_DIR", "CMAKE_PREFIX_PATH", "HOME")


def spec(entry: dict) -> dict | None:
    """Return the build spec of a registry *entry*, or None when it declares none."""
    cmd = (entry.get("build_cmd") or "").strip()
    if not cmd:
        return None
    timeout = entry.get("build_timeout_s") or 0
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        timeout = 0.0
    if timeout <= 0:
        timeout = _DEFAULT_TIMEOUT_S
    return {
        "cmd":       cmd,
        "cwd":       (entry.get("build_cwd") or "").strip(),
        "timeout_s": min(timeout, _MAX_TIMEOUT_S),
        "env":       _declared_env(entry),
    }


def _declared_env(entry: dict) -> dict[str, str]:
    """The environment variables this registration pins for its build.

    Everything else is inherited. What is inherited, though, is the environment the
    MCP *server* was started with — not the one the user builds in by hand. When the
    two disagree on the compiler, on CCACHE_DIR, or on a module-provided toolchain,
    the two builds invalidate each other's artifacts and the project recompiles from
    scratch every time it changes hands. Pinning the few variables that decide this,
    once, at registration, is what stops that alternation.
    """
    raw = entry.get("build_env") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def check_env(env: dict | None) -> str | None:
    """Return why *env* cannot be a process environment, or None when it can."""
    if not env:
        return None
    if not isinstance(env, dict):
        return "build_env must be a mapping of variable name to value."
    for key in env:
        if not _ENV_NAME_RE.fullmatch(str(key)):
            return (f"Invalid build_env variable name {key!r}. Use names like "
                    f"'CCACHE_DIR', 'CC', 'PATH'.")
        if "\0" in str(env[key]):
            return f"build_env[{key!r}] contains a NUL byte."
    return None


def check_cmd(cmd: str) -> str | None:
    """Return why *cmd* cannot be run as argv, or None when it can."""
    if not cmd.strip():
        return None
    for meta in _SHELL_METACHARS:
        if meta in cmd:
            return f"build_cmd contains the shell operator {meta!r}. {_SHELL_HINT}"
    try:
        argv = shlex.split(cmd)
    except ValueError as exc:
        return f"build_cmd could not be parsed ({exc}). {_SHELL_HINT}"
    if not argv:
        return f"build_cmd is empty after parsing. {_SHELL_HINT}"
    return None


def _declared_root() -> str:
    """The workspace root as the user spelled it, symlinks intact.

    ``store.workspace_root`` resolves them, which is right for fingerprinting — two
    spellings of one file must not read as two files. It is wrong for a build
    directory: see ``_resolve_cwd``.
    """
    return os.path.abspath(os.environ.get("MCP_FILES_ROOT") or os.getcwd())


def _resolve_cwd(raw: str) -> tuple[str, str | None]:
    """Where the build runs, in the spelling the build system was configured with.

    This used to call ``realpath``. On a cluster that is a full rebuild on every run:
    a project reached through a symlink (``/home/me/proj`` -> ``/gpfs/.../proj``, which
    is how scratch space is normally presented) gets configured by CMake with the
    logical path, and CMake bakes absolute paths into its cache, its rules and its
    depfiles. Handing the same build the resolved path makes every one of those compare
    unequal, so it reconfigures and recompiles the world — and it alternates forever
    with whatever the user runs from their own shell.

    Normalised but not resolved, therefore. ``isdir`` still follows the link, so a
    symlinked build directory is accepted; what changes is only which spelling the
    compiler is handed. Nothing here was a containment check — there is none on this
    path — so nothing is weakened by keeping the declared form.
    """
    root = _declared_root()
    if not raw:
        return root, None
    path = raw if os.path.isabs(raw) else os.path.join(root, raw)
    path = os.path.normpath(os.path.abspath(path))
    if not os.path.isdir(path):
        return root, f"build_cwd is not a directory: {raw}"
    return path, None


def _toolchain_facts(env) -> dict:
    """What the build is about to compile with, as far as it can be established.

    ``ccache`` is resolved against the build's own PATH rather than reported from a
    variable: "CCACHE_DIR is set" and "ccache is reachable" are different claims, and
    only the second one makes a rebuild cheap. A null here is the answer to a whole
    class of "why does it recompile everything" — and it was previously invisible,
    since the log recorded the command and the directory and nothing else.
    """
    facts = {k: env[k] for k in _ENV_OF_INTEREST if env.get(k)}
    facts["ccache"] = shutil.which("ccache", path=env.get("PATH"))
    return facts


def _env_log_lines(declared: dict, toolchain: dict) -> str:
    """The environment header of a build log entry; empty when there is nothing to say."""
    lines = []
    if declared:
        lines.append("    env (pinned by the registration): "
                     + " ".join(f"{k}={v}" for k, v in sorted(declared.items())))
    named = " ".join(f"{k}={v}" for k, v in sorted(toolchain.items())
                     if k != "ccache" and v)
    if named:
        lines.append(f"    toolchain: {named}")
    lines.append("    ccache: " + (toolchain.get("ccache")
                                   or "not on PATH (every build recompiles in full)"))
    return "\n".join(lines) + "\n"


def _record(proxy: str, status: str, **extra) -> dict:
    rec = {"proxy": proxy, "status": status}
    rec.update(extra)
    return rec


def run_build(entry: dict, run_dir: str, proxy: str = "") -> dict:
    """Build one registered proxy, appending its output to ``<run_dir>/build.log``.

    Never raises: every failure mode comes back as a record whose ``status`` says
    which one. ``skipped`` means the registration declares no build.
    """
    sp = spec(entry)
    if sp is None:
        return _record(proxy, "skipped", duration_s=0.0)

    bad = check_cmd(sp["cmd"])
    if bad:
        return _record(proxy, "invalid", cmd=sp["cmd"], error=bad, duration_s=0.0)

    cwd, cwd_err = _resolve_cwd(sp["cwd"])
    if cwd_err:
        return _record(proxy, "invalid", cmd=sp["cmd"], error=cwd_err, duration_s=0.0)

    bad_env = check_env(sp["env"])
    if bad_env:
        return _record(proxy, "invalid", cmd=sp["cmd"], error=bad_env, duration_s=0.0)

    argv = shlex.split(sp["cmd"])
    log_path = procs._build_log_path(run_dir)
    started = time.monotonic()
    # PWD, because the spelling is the whole point of keeping build_cwd unresolved and
    # it is the only channel that carries it. Popen chdir()s, and the child's getcwd()
    # comes back symlink-resolved whichever spelling we passed — so a build launched in
    # /home/me/proj/build saw /gpfs/.../build, disagreeing with the logical path CMake
    # was configured with. Worse, PWD was simply inherited: it named the MCP server's
    # directory, not the one the build was running in, which is a false statement to
    # every wrapper script that reads it.
    env = {**os.environ, "PWD": cwd, **sp["env"]}
    toolchain = _toolchain_facts(env)
    base = _record(proxy, "ok", cmd=sp["cmd"], argv=argv, cwd=cwd, log=log_path,
                   env_declared=sp["env"], toolchain=toolchain)

    try:
        with open(log_path, "a") as fh:
            fh.write(f"\n=== build {proxy or '?'}: {sp['cmd']} (cwd={cwd}) ===\n")
            fh.write(_env_log_lines(sp["env"], toolchain))
            fh.flush()
            proc = proc_run.run(argv, cwd=cwd, stdout=fh,
                                stderr=subprocess.STDOUT,
                                timeout=sp["timeout_s"], env=env)
    except subprocess.TimeoutExpired:
        base.update(status="timeout", returncode=None,
                    error=f"build timed out after {sp['timeout_s']:.0f}s",
                    duration_s=round(time.monotonic() - started, 2))
        return base
    except (FileNotFoundError, PermissionError, OSError) as exc:
        base.update(status="launch_error", returncode=None,
                    error=f"could not launch build: {exc}",
                    duration_s=round(time.monotonic() - started, 2))
        return base

    base["duration_s"] = round(time.monotonic() - started, 2)
    base["returncode"] = proc.returncode
    if proc.returncode != 0:
        base["status"] = "failed"
        base["error"] = f"build exited {proc.returncode}"
    return base


def builds_for(reg: dict, suite: dict, proxy_name: str,
               fallback_entry: dict) -> list[tuple[str, dict]]:
    """The registry entries a run of *suite* must build, in first-use order.

    The entry a case resolves to is computed exactly as the runner computes it,
    so what gets built is what gets measured. Two proxies sharing one build
    command and directory are built once: the key is the work, not the name.
    """
    out: list[tuple[str, dict]] = []
    seen: set[tuple[str, str]] = set()
    for case in (suite.get("cases") or []):
        case_proxy = case.get("proxy_name", proxy_name)
        entry = reg.get(case_proxy, fallback_entry)
        sp = spec(entry)
        if sp is None:
            continue
        key = (sp["cmd"], sp["cwd"])
        if key in seen:
            continue
        seen.add(key)
        out.append((case_proxy, entry))
    return out


def report_path(run_dir: str) -> str:
    return os.path.join(run_dir, "build.json")


def write_report(run_dir: str, records: list[dict]) -> dict:
    """Persist the run's build records; returns the payload written."""
    status = "ok"
    for rec in records:
        if rec.get("status") not in ("ok", "skipped"):
            status = rec["status"]
            break
    payload = {
        "status":            status,
        "builds":            records,
        "total_duration_s":  round(sum(r.get("duration_s") or 0.0
                                       for r in records), 2),
    }
    try:
        with open(report_path(run_dir), "w") as fh:
            json.dump(payload, fh, indent=2)
    except OSError:
        pass
    return payload


def read_report(run_dir: str) -> dict | None:
    """Read a run's build.json, or None when the run declared no build."""
    data = store._read_json(report_path(run_dir))
    return data if isinstance(data, dict) else None


def failure_summary(report: dict) -> str:
    """One line naming which proxy's build failed and how."""
    for rec in report.get("builds") or []:
        if rec.get("status") not in ("ok", "skipped"):
            proxy = rec.get("proxy") or "?"
            detail = rec.get("error") or rec.get("status")
            return f"Build failed for proxy '{proxy}': {detail}"
    return f"Build failed ({report.get('status')})."
