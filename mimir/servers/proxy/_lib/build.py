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
the registration is executed here, once per evaluation run, before any case is
measured; a build that fails ends the run with no verdict and no ledger entry.

Two consequences worth stating, because they are what make the cost bearable:

* The build is a command the project already owns, so incrementality comes from
  ``make``/``ninja``/``cmake`` and not from us — the long first build is paid
  once, and an edit to one file costs that file. Naming a target in
  ``build_cmd`` narrows it further. Deciding *what* to rebuild is the build
  system's job, and it does it from mtimes, which is why ``tree_snapshot``
  must stamp restored files as new.
* It runs once per run, never once per replicate, so ``repeat=3`` pays for one
  build.

Like ``command._build_run_cmd``, the command is argv, never a shell line: it is
split with ``shlex`` and executed directly. ``make -C <dir> <target>`` covers
most of it; anything needing several commands, a ``module load`` or a redirect
belongs in a wrapper script named here.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time

import proc_run

from _lib import procs

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


def _workspace_root() -> str:
    """The workspace root, read at call time.

    ``store`` freezes its own copy at import; a test that repoints
    MCP_FILES_ROOT in setUp would then build in the wrong tree. ``_lib`` cannot
    import ``_ops``, so this mirrors ``eval_session._workspace_root`` rather
    than sharing it.
    """
    return os.path.realpath(os.path.abspath(
        os.environ.get("MCP_FILES_ROOT") or os.getcwd()))


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
    }


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


def _resolve_cwd(raw: str) -> tuple[str, str | None]:
    root = _workspace_root()
    if not raw:
        return root, None
    path = raw if os.path.isabs(raw) else os.path.join(root, raw)
    path = os.path.realpath(path)
    if not os.path.isdir(path):
        return root, f"build_cwd is not a directory: {raw}"
    return path, None


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

    argv = shlex.split(sp["cmd"])
    log_path = procs._build_log_path(run_dir)
    started = time.monotonic()
    base = _record(proxy, "ok", cmd=sp["cmd"], argv=argv, cwd=cwd, log=log_path)

    try:
        with open(log_path, "a") as fh:
            fh.write(f"\n=== build {proxy or '?'}: {sp['cmd']} (cwd={cwd}) ===\n")
            fh.flush()
            proc = proc_run.run(argv, cwd=cwd, stdout=fh,
                                stderr=subprocess.STDOUT,
                                timeout=sp["timeout_s"])
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
    try:
        with open(report_path(run_dir)) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def failure_summary(report: dict) -> str:
    """One line naming which proxy's build failed and how."""
    for rec in report.get("builds") or []:
        if rec.get("status") not in ("ok", "skipped"):
            proxy = rec.get("proxy") or "?"
            detail = rec.get("error") or rec.get("status")
            return f"Build failed for proxy '{proxy}': {detail}"
    return f"Build failed ({report.get('status')})."
