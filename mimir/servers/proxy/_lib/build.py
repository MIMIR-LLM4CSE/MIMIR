"""Building a registered proxy before it is measured, once per evaluation run.

A compiled proxy's edited sources are not the program that runs, so the server runs
``build_cmd`` (argv, never a shell line) before any case; a failed build ends the run
without a verdict. Incrementality is the project's own build system's: see
SERVERS_DETAILED.md, "proxy".
"""

from __future__ import annotations

from ....shared import proxy_store
import json
import os
import re
import shlex
import shutil
import subprocess
import time

from . import proc_run

from . import procs

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

SHELL_HINT = (
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
    """The variables this registration pins for its build (the rest is the server's environment)."""
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
            return f"build_cmd contains the shell operator {meta!r}. {SHELL_HINT}"
    try:
        argv = shlex.split(cmd)
    except ValueError as exc:
        return f"build_cmd could not be parsed ({exc}). {SHELL_HINT}"
    if not argv:
        return f"build_cmd is empty after parsing. {SHELL_HINT}"
    return None


def _declared_root() -> str:
    """The workspace root as the user spelled it, symlinks intact.

    ``store.workspace_root`` resolves them, which is right for fingerprinting — two
    spellings of one file must not read as two files. It is wrong for a build
    directory: see ``_resolve_cwd``.
    """
    return os.path.abspath(os.environ.get("MCP_FILES_ROOT") or os.getcwd())


def _resolve_cwd(raw: str) -> tuple[str, str | None]:
    """``(cwd, error)``: the build directory as declared, normalised but not symlink-resolved.

    CMake compares against the logical path it was configured with.
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
    """The effective toolchain variables and whether ``ccache`` resolves on the build's PATH."""
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
    log_path = procs.build_log_path(run_dir)
    started = time.monotonic()
    # PWD carries the declared spelling of the build directory (the child's getcwd() comes
    # back resolved); otherwise it would name the server's directory.
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
    data = proxy_store.read_json(report_path(run_dir))
    return data if isinstance(data, dict) else None


def failure_summary(report: dict) -> str:
    """One line naming which proxy's build failed and how."""
    for rec in report.get("builds") or []:
        if rec.get("status") not in ("ok", "skipped"):
            proxy = rec.get("proxy") or "?"
            detail = rec.get("error") or rec.get("status")
            return f"Build failed for proxy '{proxy}': {detail}"
    return f"Build failed ({report.get('status')})."
