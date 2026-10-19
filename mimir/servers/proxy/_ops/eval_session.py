"""Act ops of an optimization session (``proxy_eval``): init, configure, run, stop,
reset, reset_to_best, rebaseline, end. Every response names the exact next call."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from ....shared.launcher import launch_argv
from . import with_next, err, ok
from .._lib.procs import (
    run_log_path,
    run_state,
    new_run_dir,
    write_run_config,
    launch_detached,
    cancel_run,
    opt_active_run_dir,
    update_opt_active_link,
)
from .._lib.ratchet import effective_repeat, load_best
from ._eval_validate import (
    check_optimize_paths,
    check_reference_requirements,
    check_requirements,
)
from ....shared import run_channel
from .._lib import tree_snapshot
from ....shared.atomic import write_json
from .._lib.store import (
    opt_git_dir, workspace_root, entry_or_err, load_suite, load_opt_config, save_opt_config,
    opt_ledger_file, opt_best_file, write_active_session, clear_active_session,
)
from .._lib.layout import cache_dir, opt_session_runs_dir, resolve_proxy_name, read_json
from .eval_common import (
    NEXT_DETACHED,
    NEXT_RUN,
    NEXT_STATUS,
    eval_background_descriptor,
)
from .eval_observe import results


_OPT_RUNNER = "mimir.servers.proxy._proxy_runner"


# A run that has reached one of these has nothing left to wait for.
_TERMINAL_STATES = ("done", "crashed")


# How often the wait re-reads the run dir. The run writes metrics.json once, at the
# end, so polling faster buys nothing and only spins the server.
_RUN_POLL_S = 2.0


# Which run channel a waiting op='run' publishes under. The channel is the blocking
# tool's own registered name, so the client can address it off the tool row it is
# diverting without either end spelling the other's name out.
_DIVERT_CHANNEL = "proxy_eval"


# How long op='run' waits before handing the run to the watcher: under the client's cap
# on tool calls (TOOL_CALL_TIMEOUT_MAX_SECS, 1200 s), with room to settle and read results.
RUN_WAIT_BUDGET_S = 1100.0


def _workspace_branch() -> str:
    """The workspace's git branch, or "": what tells parallel axes' ledgers apart."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=workspace_root(),
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    branch = (done.stdout or "").strip()
    return "" if done.returncode != 0 or branch == "HEAD" else branch


def _init_payload(*, proxy_name, benchmark_name, abs_src, abs_paths, baseline_id,
                  baseline_existed, requirements, primary_goal, primary_metric,
                  min_improvement, cfg) -> dict:
    """What init answers with: the session as recorded, and how to work in it.

    Separated from the persistence above because it is prose, and because it is the
    only place the loop's rules are stated to the caller — what to edit, what the
    first run has to be, and what an edit must beat.
    """
    repeat = effective_repeat(cfg)
    edited = ", ".join(os.path.basename(p) for p in abs_paths)
    payload = {
        "proxy_name":        proxy_name,
        "benchmark_name":    benchmark_name,
        "proxy_source_path": abs_src,
        "optimize_paths":    abs_paths,
        "baseline_id":       baseline_id,
        "baseline_existed":  baseline_existed,
        "requirements":      requirements,
        "objective":         f"{primary_goal}imize {primary_metric} subject to the requirements",
        # What a run costs and what it has to beat. Both were once silent, and the
        # threshold was a constant asserting a property of a machine nobody had measured.
        "repeat":            repeat,
        "min_improvement":   float(min_improvement),
        "margin_note": (
            f"Each case is measured {repeat}x and reduced to its median. A run must "
            f"beat the incumbent by {min_improvement:.1%} OR by the spread measured "
            f"across the baseline's own replicates, whichever is larger — so an edit "
            f"cannot be accepted on a difference this machine produces without it."
        ),
        # Names optimize_paths, never proxy_source_path: pointing a model at the
        # harness is what makes it optimise a copy of the thing instead of the thing.
        "note": ("Baseline already recorded — kept, not moved. " if baseline_existed
                 else "Baseline tree snapshotted. ")
            + f"Edit {edited} between runs — never the harness at "
            + os.path.basename(abs_src) + ", which only runs the code and prints "
            "metrics. The FIRST run must be launched with those files untouched: it is "
            "the baseline every later number is compared against, and until it exists "
            "no run can be accepted. Feasible runs that improve the objective are "
            "accepted; regressions are rejected — revert them with "
            "proxy_eval(op='reset_to_best', confirm=True).",
    }
    if baseline_existed:
        # Said on the reply rather than buried in a doc: this is the moment a model
        # discovers its harness was wrong, and the unsupported route out of it
        # (end + clean + init) moves the baseline by destroying the ledger.
        payload["baseline_note"] = (
            "This baseline was taken by an earlier init and has NOT been moved — "
            "comparisons still run against the original tree. If the harness itself "
            "was wrong and the baseline must be re-measured from the tree as it "
            "stands, use proxy_eval(op='rebaseline', confirm=True): it archives the "
            "ledger and best-so-far instead of discarding them.")
    return payload


def init(
    proxy_name: str,
    benchmark_name: str,
    requirements: list[dict],
    proxy_source_path: str,
    optimize_paths: list[str] | None = None,
    python_executable: str = "",
    max_hours: float = 0.0,
    primary_metric: str = "time_s",
    primary_goal: str = "min",
    min_improvement: float = 0.02,
    max_stall: int = 5,
    convergence: dict | None = None,
    repeat: int = 0,
) -> dict:
    entry, error = entry_or_err(proxy_name)
    if error:
        return error

    suite = load_suite(benchmark_name)
    if suite is None:
        return err(f"Benchmark suite '{benchmark_name}' not found.",
                   hint="Create one with proxy_exec(op='benchmark_create', ...) "
                        "or proxy_manage(op='suite_define', ...) first.")

    req_err = check_requirements(requirements)
    if req_err:
        return err(req_err)

    ref_err = check_reference_requirements(requirements, suite, entry)
    if ref_err:
        return err(ref_err)

    if primary_goal not in ("min", "max"):
        return err(f"primary_goal must be 'min' or 'max', got '{primary_goal}'.")
    if max_stall < 1:
        return err(f"max_stall must be >= 1, got {max_stall}.")
    if min_improvement < 0:
        return err(f"min_improvement must be >= 0, got {min_improvement}.")
    if repeat < 0:
        return err(f"repeat must be >= 0 (0 = decide from the metric), got {repeat}.")

    abs_src = os.path.abspath(proxy_source_path)
    if not os.path.isfile(abs_src):
        return err(f"proxy_source_path not found: {abs_src}")

    paths_err = check_optimize_paths(list(optimize_paths or []), abs_src)
    if paths_err:
        return err(paths_err)
    abs_paths = [os.path.abspath(p) for p in (optimize_paths or [])]
    if python_executable and not os.path.isfile(python_executable):
        return err(f"python_executable not found: {python_executable}")

    # The baseline snapshots the whole tracked tree, and a re-init never moves it: comparisons
    # stay against the original.
    os.makedirs(cache_dir(), exist_ok=True)
    prior = load_opt_config(proxy_name) or {}
    baseline_existed = bool(prior.get("baseline_id"))
    if baseline_existed:
        baseline_id  = prior["baseline_id"]
        baseline_fp  = prior.get("baseline_fingerprint", "")
        baseline_run = prior.get("baseline_run_id", "")
    else:
        baseline_id = tree_snapshot.snapshot(
            opt_git_dir(), workspace_root(), abs_paths,
            f"baseline: {proxy_name} before optimisation",
        )
        if not baseline_id:
            return err("Could not snapshot the baseline tree.",
                       hint="Check that optimize_paths are readable and the proxy store "
                            "is writable.")
        baseline_fp  = tree_snapshot.fingerprint(workspace_root(), abs_paths)
        baseline_run = ""

    cfg = {
        "proxy_name":        proxy_name,
        "benchmark_name":    benchmark_name,
        "requirements":      requirements,
        "proxy_source_path": abs_src,
        "optimize_paths":    abs_paths,
        "baseline_id":       baseline_id,
        # Content hash of the untouched tree. prepare_run refuses the first run if the
        # files have already moved away from it, so the baseline is measured, not assumed.
        "baseline_fingerprint": baseline_fp,
        "baseline_run_id":   baseline_run,
        "python_executable": python_executable,
        "max_hours":         max_hours if max_hours > 0 else 24.0,
        "primary_metric":    primary_metric,
        "primary_goal":      primary_goal,
        "min_improvement":   float(min_improvement),
        "max_stall":         int(max_stall),
        "repeat":            int(repeat),
        "stall":             0,
        "convergence":       convergence or {},
        # Where this axis is being worked. Empty outside a git repository, and equal to
        # the main branch for an ordinary session — it earns its keep when several
        # sub-agents each optimise a copy of their own.
        "branch":            _workspace_branch(),
        "initialized_at":    datetime.now(timezone.utc).isoformat(),
    }
    save_opt_config(cfg)
    write_active_session(proxy_name)

    return ok(with_next(_init_payload(
        proxy_name=proxy_name, benchmark_name=benchmark_name, abs_src=abs_src,
        abs_paths=abs_paths, baseline_id=baseline_id,
        baseline_existed=baseline_existed, requirements=requirements,
        primary_goal=primary_goal, primary_metric=primary_metric,
        min_improvement=min_improvement, cfg=cfg), NEXT_RUN))


def configure(
    proxy_name: str = "",
    requirements: list[dict] | None = None,
    benchmark_name: str = "",
    python_executable: str = "",
    max_hours: float = 0.0,
) -> dict:
    cfg = load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.",
                   hint="Call proxy_eval(op='init', ...) first.")

    if requirements is not None:
        req_err = check_requirements(requirements)
        if req_err:
            return err(req_err)
        cfg["requirements"] = requirements

    if benchmark_name:
        if load_suite(benchmark_name) is None:
            return err(f"Benchmark suite '{benchmark_name}' not found.")
        cfg["benchmark_name"] = benchmark_name

    if python_executable:
        if not os.path.isfile(python_executable):
            return err(f"python_executable not found: {python_executable}")
        cfg["python_executable"] = python_executable

    if max_hours > 0:
        cfg["max_hours"] = max_hours

    # Re-check reference-dependent requirements against the (possibly new)
    # suite and registration — same dead-end guard as init.
    final_suite = load_suite(cfg.get("benchmark_name", "")) or {}
    entry, error = entry_or_err(cfg.get("proxy_name", ""))
    if error:
        return error
    ref_err = check_reference_requirements(
        cfg.get("requirements") or [], final_suite, entry)
    if ref_err:
        return err(ref_err)

    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_opt_config(cfg)
    return ok(with_next({"config": cfg}, NEXT_RUN))


def _resume_notice(cfg: dict, name: str, paths: list[str]) -> str:
    """A notice when a resumed session's tree differs from its last run's launch tree, or "".

    Inside a live session the tree changes every iteration and nothing fires. Never a refusal.
    """
    if not paths or not cfg.get("baseline_run_id"):
        return ""
    if resolve_proxy_name("") == name:
        return ""  # a live session, not a resumption
    last_run = opt_active_run_dir(name)
    if not last_run:
        return ""
    measured = (read_json(os.path.join(last_run, "tree_at_launch.json")) or {}).get(
        "fingerprint", "")
    if not measured or measured == tree_snapshot.fingerprint(workspace_root(), paths):
        return ""
    return (
        "Resuming a session whose tracked files have changed since it was last measured. "
        "This run will be compared against the previous best, which was measured on code "
        "that is no longer on disk. If the change is part of what you are optimising, "
        "carry on. If it arrived from anywhere else, take the current state as the new "
        "reference first: proxy_eval(op='rebaseline', confirm=True)."
    )


def prepare_run(
    proxy_name: str,
    axis: str = "",
) -> tuple[dict, dict | None, str, str]:
    """The shared preamble of local and Slurm eval runs: ``(cfg, err, run_dir, notice)``.

    On failure only *err* is set. *notice* comes back separately from *cfg*, which is
    persisted.
    """
    cfg = load_opt_config(proxy_name)
    if not cfg:
        return {}, err("No optimization session found.",
                         hint="Call proxy_eval(op='init', ...) first."), "", ""

    name           = cfg.get("proxy_name", "")
    benchmark_name = cfg.get("benchmark_name", "")
    if not name or not benchmark_name:
        return {}, err("proxy_name and benchmark_name must be set.",
                         hint="Call proxy_eval(op='init', ...) first."), "", ""

    active = opt_active_run_dir(name)
    if active:
        rs = run_state(active)
        if rs["state"] in ("running", "pending"):
            pid = rs.get("pid")
            jid = rs.get("slurm_job_id")
            tag = f"slurm_job_id={jid}" if jid else f"pid={pid}"
            return {}, err(f"An optimization run is already active ({tag}).",
                             hint="Use proxy_eval(op='stop', confirm=True) first."), "", ""

    # The first run must measure the UNTOUCHED code. Checked here, before the run is
    # spent, rather than at settle time: a run that cannot be compared to anything is
    # not worth waiting for, and "restore and run" is an instruction the caller can act
    # on immediately. Without it the first *feasible* run became the best, so a session
    # whose first run was already an edit had no baseline at all and every later number
    # was an assertion rather than a comparison.
    paths = list(cfg.get("optimize_paths") or [])
    if paths and not cfg.get("baseline_run_id"):
        current = tree_snapshot.fingerprint(workspace_root(), paths)
        if current != cfg.get("baseline_fingerprint", current):
            return {}, err(
                "No baseline run on record, and the tracked files have already been "
                "edited — this run would have nothing to be compared against.",
                hint="Restore the original with proxy_eval(op='reset', confirm=True), "
                     "run once to measure it, then optimize from there."), "", ""

    notice = _resume_notice(cfg, name, paths)

    run_dir = new_run_dir(opt_session_runs_dir(name))
    write_run_config(run_dir, {
        "proxy_name":        name,
        "benchmark_name":    benchmark_name,
        "requirements":      cfg.get("requirements", []),
        "python_executable": cfg.get("python_executable", ""),
        "deadline_s":        (cfg.get("max_hours") or 24.0) * 3600,
        "convergence":       cfg.get("convergence") or {},
        "primary_metric":    cfg.get("primary_metric", "time_s"),
        "repeat":            effective_repeat(cfg),
        # What this run is trying, in the caller's words. Recorded with the run rather
        # than derived at settle time: by then the edit is measured and nothing on disk
        # says what it was for. It is the only field of the ledger the machine cannot
        # supply, and without it a session's runs are a list of numbers with no account
        # of which ideas produced them.
        "axis":              (axis or "").strip()[:80],
        "started_at":        datetime.now(timezone.utc).isoformat(),
    })
    # Freeze what is about to run. The accepted state must be the code that PRODUCED
    # the run, not whatever the files hold when it settles — the agent may edit while a
    # run is in flight, and an accepted "best" that never ran would poison every later
    # comparison. Recorded as a tree snapshot, so a multi-file edit is captured whole.
    paths = list(cfg.get("optimize_paths") or [])
    launch_id = tree_snapshot.snapshot(
        opt_git_dir(), workspace_root(), paths,
        f"launch: {name} {os.path.basename(run_dir)}",
    ) if paths else None
    if launch_id:
        write_json(os.path.join(run_dir, "tree_at_launch.json"), {
            "snapshot_id": launch_id,
            "paths":       paths,
            "fingerprint": tree_snapshot.fingerprint(workspace_root(), paths),
            "is_baseline": launch_id == cfg.get("baseline_id", ""),
        }, indent=2)
    return cfg, None, run_dir, notice


def run(proxy_name: str = "", background: bool = False, axis: str = "") -> dict:
    """Launch an optimization run without waiting; *axis* names the idea it tries (ledger).

    ``background=True`` attaches a ``background_job`` descriptor.
    """
    cfg, error, run_dir, resume_notice = prepare_run(proxy_name, axis)
    if error:
        return error
    name       = cfg["proxy_name"]
    log_file   = run_log_path(run_dir)
    python_exe = cfg.get("python_executable") or sys.executable

    pid = launch_detached(
        [python_exe, *launch_argv(_OPT_RUNNER, "--run-dir", run_dir)], run_dir, log_file=log_file,
    )
    update_opt_active_link(name, run_dir)

    payload = {
        "run_dir":        run_dir,
        "pid":            pid,
        "log":            log_file,
        "proxy_name":     name,
        "benchmark_name": cfg["benchmark_name"],
        "note": "Optimization run started in background.",
    }
    if resume_notice:
        payload["resume_notice"] = resume_notice
    if background:
        payload["background_job"] = eval_background_descriptor(name, run_dir)
        # States what was started; whether a watcher is on it is the client's to say.
        payload["note"] = "Optimization run detached; this call returns before it ends."
        return ok(with_next(payload, NEXT_DETACHED))
    return ok(with_next(payload, NEXT_STATUS))


async def await_terminal_state(
    run_dir: str, wait_s: float, channel: str = "", job_key: str = "",
    pid: int = 0, state_fn=None, label: str = "optimization run",
) -> tuple[str | None, str]:
    """Poll *run_dir* until it ends: ``(state, "")``, or ``(None, "budget" | "diverted")``.

    Async, so the server keeps answering; it owns the run channel meanwhile (phase out,
    divert requests in).
    """
    deadline = time.monotonic() + max(0.0, wait_s)
    if channel:
        # The launcher's pid, passed in rather than read back: the caller has just
        # started the process, and an extra state read here would be one more poll of
        # the run dir for a fact already in hand.
        run_channel.publish(channel, job_key, pid,
                            f"{label} {os.path.basename(run_dir)}",
                            run_dir, wait_s)
    try:
        while True:
            # One read serves both the terminal test and the phase: `run_state`
            # already folds in what the run says it is doing.
            st = (state_fn or run_state)(run_dir)
            state = st["state"]
            if state in _TERMINAL_STATES:
                return state, ""
            if channel:
                phase = st.get("phase") or ""
                if state == "pending" and st.get("slurm_job_id"):
                    # Queued, not started. Only this loop is in a position to say so:
                    # the runner that writes the phase file has not begun.
                    phase = f"queued (slurm {st['slurm_job_id']})"
                run_channel.update(channel, job_key, phase=phase,
                                   percent=st.get("percent"))
                if run_channel.requested(channel, job_key):
                    return None, "diverted"
            if time.monotonic() >= deadline:
                return None, "budget"
            await asyncio.sleep(_RUN_POLL_S)
    finally:
        if channel:
            run_channel.clear(channel, job_key)


async def run_awaited(
    proxy_name: str = "", background: bool = False,
    wait_s: float = RUN_WAIT_BUDGET_S, axis: str = "",
) -> dict:
    """Launch a run and wait for its verdict (op='run'): the results inline.

    ``background=True``, or a run outlasting *wait_s*, detaches it to the client watcher.
    """
    launched = run(proxy_name, background=background, axis=axis)
    if background or launched.get("status") != "ok":
        return launched

    run_dir = launched.get("run_dir", "")
    name    = launched.get("proxy_name", "")
    state, reason = await await_terminal_state(
        run_dir, wait_s, _DIVERT_CHANNEL, os.path.basename(run_dir),
        launched.get("pid") or 0)

    if state is None:
        # Still running. Detaching beats letting the tool call time out: a timeout
        # would abandon a run that is alive and doing the work asked of it. The user
        # asking for it and the budget running out reach the same place by design —
        # one detached run, one watcher, one resume — so only the note differs.
        detached = {k: v for k, v in launched.items() if k not in ("status", "next_step")}
        detached["note"] = (
            "The user moved this run to the background while it was still going. It "
            "was not killed and it did not fail: it continues, and you are resumed "
            "with its results when it ends."
            if reason == "diverted" else
            f"Still running after {int(wait_s)}s — handed to the background watcher."
        )
        detached["background_job"] = eval_background_descriptor(name, run_dir)
        return ok(with_next(detached, NEXT_DETACHED))

    settled = results(name)
    # Keep the launch identity on the answer: which run this was, and where its log
    # is, are what a crash diagnosis needs and `results` does not carry; and the
    # `resume_notice`, which this reply, rebuilt from `results()`, would otherwise drop.
    for key in ("pid", "log", "benchmark_name", "resume_notice"):
        settled.setdefault(key, launched.get(key))
    settled.setdefault("run_dir", run_dir)
    return settled


def stop(proxy_name: str = "") -> dict:
    resolved = resolve_proxy_name(proxy_name)
    if not resolved:
        return err("No proxy name or active session.")
    run_dir = opt_active_run_dir(resolved)
    if not run_dir:
        return err("No active optimization run found.")

    result = cancel_run(run_dir)
    if "error" in result:
        return err(result["error"])
    return ok(with_next({"run_dir": run_dir, **result}, NEXT_RUN + " to start a new run"))


def _restored_payload(paths: list[str], rewritten: list[str]) -> dict:
    """What a restore rewrote (the files the next build sees as new), not what it covered."""
    names = [os.path.basename(p) for p in rewritten]
    return {"restored": names, "unchanged": max(0, len(paths) - len(names))}


def _rebuild_note(rewritten: list[str]) -> str:
    """The sentence that tells the caller what this costs to rebuild."""
    if not rewritten:
        return (" Nothing was written: the tree was already in that state, so the "
                "next run has nothing to rebuild.")
    return (f" {len(rewritten)} file(s) were rewritten; the build system will "
            "recompile those and what depends on them, not the whole set.")


def reset(proxy_name: str = "") -> dict:
    """Restore every tracked file to the baseline tree taken at init."""
    cfg = load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.",
                   hint="Call proxy_eval(op='init', ...) first.")

    paths       = list(cfg.get("optimize_paths") or [])
    baseline_id = cfg.get("baseline_id", "")
    if not paths or not baseline_id:
        return err("No baseline tree recorded for this session.",
                   hint="Call proxy_eval(op='init', ...) again.")

    rewritten = tree_snapshot.restore(
        opt_git_dir(), workspace_root(), paths, baseline_id)
    if rewritten is None:
        return err("Could not restore the baseline tree.",
                   hint="The snapshot store may be missing or unwritable.")

    return ok(with_next({
        **_restored_payload(paths, rewritten),
        "from":      "baseline",
        "note": "Every tracked file is back to the baseline taken at init. "
                "Try a different modification approach."
                + _rebuild_note(rewritten),
    }, NEXT_RUN + " to verify the baseline"))


def reset_to_best(proxy_name: str = "") -> dict:
    """Restore the proxy source file from the best-so-far snapshot.

    Unlike ``reset`` (which restores the original canonical baseline), this reverts
    to the source of the best *accepted* run — the way to undo a regression without
    discarding the improvements found so far.
    """
    cfg = load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.",
                   hint="Call proxy_eval(op='init', ...) first.")
    name = cfg.get("proxy_name", "")
    best = load_best(name)
    if not best:
        return err("No best-so-far recorded yet — no feasible run has been accepted.",
                   hint="Run at least once until requirements pass, or use "
                        "proxy_eval(op='reset', confirm=True) to restore the baseline.")

    paths    = list(cfg.get("optimize_paths") or [])
    snapshot = best.get("tree_snapshot") or ""
    if not paths:
        return err("optimize_paths not set in config.",
                   hint="Call proxy_eval(op='init', ...) again.")
    if not snapshot:
        return err("The best run predates tree snapshots.",
                   hint="Use proxy_eval(op='reset', confirm=True) to restore the baseline.")

    # All of them or none: restoring a per-file best would assemble a combination that
    # was never measured together.
    rewritten = tree_snapshot.restore(
        opt_git_dir(), workspace_root(), paths, snapshot)
    if rewritten is None:
        return err("Could not restore the best tree.",
                   hint="Use proxy_eval(op='reset', confirm=True) to restore the baseline.")

    return ok(with_next({
        **_restored_payload(paths, rewritten),
        "from_best":    best.get("run_id"),
        "primary_value": best.get("primary_value"),
        "note": "Every tracked file is back to the state of the best accepted run. "
                "Try a different modification approach from there."
                + _rebuild_note(rewritten),
    }, NEXT_RUN + " to verify, or summarize if converged"))


def rebaseline(proxy_name: str = "") -> dict:
    """Take a new baseline from the current tree, archiving the ledger and best (never deleting)."""
    cfg = load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.",
                   hint="Call proxy_eval(op='init', ...) first.")

    name = cfg.get("proxy_name", "") or resolve_proxy_name(proxy_name) or ""
    paths = [os.path.abspath(p) for p in (cfg.get("optimize_paths") or [])]
    if not paths:
        return err("optimize_paths not set in config.",
                   hint="Call proxy_eval(op='init', ...) again.")

    active = opt_active_run_dir(name)
    if active and run_state(active).get("state") in ("running", "pending"):
        return err("A run is still active — stop it before re-baselining.",
                   hint="proxy_eval(op='stop', confirm=True), then retry.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = _archive_session_history(name, stamp)

    baseline_id = tree_snapshot.snapshot(
        opt_git_dir(), workspace_root(), paths,
        f"rebaseline: {name} at {stamp}",
    )
    if not baseline_id:
        return err("Could not snapshot the new baseline tree.",
                   hint="Check that optimize_paths are readable and the proxy store "
                        "is writable.")

    previous = cfg.get("baseline_id", "")
    cfg["baseline_id"] = baseline_id
    cfg["baseline_fingerprint"] = tree_snapshot.fingerprint(workspace_root(), paths)
    cfg["baseline_run_id"] = ""
    # The next baseline run pins the session to wherever it executes.
    cfg.pop("machine", None)
    cfg["stall"] = 0
    cfg["rebaselined_at"] = datetime.now(timezone.utc).isoformat()
    cfg["previous_baseline_id"] = previous
    save_opt_config(cfg)
    write_active_session(name)

    return ok(with_next({
        "rebaselined":         name,
        "baseline_id":         baseline_id,
        "previous_baseline_id": previous,
        "archived":            archived,
        "note": "The tree as it stands is the new baseline. Every later comparison is "
                "against THIS point, not against the original — the ledger and best-so-far "
                "that led here are archived, not deleted, so the earlier progression is "
                "still on record. Measure the new baseline before editing anything.",
    }, NEXT_RUN + " to measure the new baseline"))


def _archive_session_history(proxy_name: str, stamp: str) -> list[str]:
    """Move the ledger and best-so-far aside under *stamp*; return what moved.

    Renamed rather than removed. A ledger is the only record that a run happened at
    all, and the reason to re-baseline is never "that history was wrong".
    """
    moved: list[str] = []
    for path in (opt_ledger_file(proxy_name), opt_best_file(proxy_name)):
        if not os.path.isfile(path):
            continue
        base, ext = os.path.splitext(path)
        target = f"{base}.{stamp}{ext}"
        try:
            os.replace(path, target)
        except OSError:
            continue
        moved.append(os.path.basename(target))
    return moved


def end(proxy_name: str = "") -> dict:
    """End the session: clear the active pointer (files stay); refused while a run is in flight."""
    cfg = load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.",
                   hint="Nothing to end. Call proxy_eval(op='init', ...) to start one.")
    name = cfg.get("proxy_name", "") or resolve_proxy_name(proxy_name) or ""

    active = opt_active_run_dir(name)
    if active and run_state(active).get("state") in ("running", "pending"):
        return err("A run is still active — stop it before ending the session.",
                   hint="proxy_eval(op='stop', confirm=True), then proxy_eval(op='end', confirm=True).")

    clear_active_session()
    return ok(with_next({
        "ended":  name,
        "note": "Optimization session ended; source/snapshots/ledger kept for history. "
                "The proxy can be run directly again. Re-init to optimize further.",
    }, "proxy_eval(op='init', ...) to start a new session, or you are done."))
