"""Optimization-session ops.

Act ops (dispatched by ``proxy_eval``): init, configure, run, stop, reset,
reset_to_best, end.
Observe ops (dispatched by ``proxy_eval_status``): status, results, log,
runs, diff, config.  Every response carries a ``next_step`` hint naming the
exact next call, so the loop is self-describing.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timezone

from _ops import _PROXY_DIR, _with_next, err, ok
from _lib.procs import (
    _log_path, _read_log, _run_state, background_descriptor,
    _new_run_dir, _write_run_config, _launch_detached, _cancel_run,
    _opt_active_run_dir, _update_opt_active_link,
)
from _lib.ratchet import _effective_repeat, _load_best
from _lib.report import _diff_run_pair, parse_runner_log
from _ops._eval_ratchet import _ratchet_state
from _ops._eval_validate import (
    _check_optimize_paths, _check_reference_requirements, _check_requirements,
)
import build_progress
import run_channel
from _lib import build, procs, tree_snapshot
from _lib.store import (
    cache_dir, opt_git_dir, workspace_root,
    _entry_or_err, _load_suite,
    _load_opt_config, _save_opt_config,
    _opt_session_runs_dir, _opt_ledger_file, _opt_best_file,
    _resolve_proxy_name, _write_active_session, _clear_active_session,
    _read_json, _write_json_atomic, _run_dir_names,
)

_OPT_RUNNER = os.path.join(_PROXY_DIR, "_proxy_runner.py")

_NEXT_RUN       = "proxy_eval(op='run', confirm=True)"
_NEXT_STATUS    = "proxy_eval_status() to monitor (state 'done' means finished)"
_NEXT_RESULTS   = "proxy_eval_status(op='results')"
_NEXT_RESET_BEST = "proxy_eval(op='reset_to_best', confirm=True)"
_NEXT_DETACHED  = ("this run continues without you; if the result says it is being "
                   "watched, end your turn on it rather than polling — you are resumed "
                   "with the results")

# A run that has reached one of these has nothing left to wait for.
_TERMINAL_STATES = ("done", "crashed")
# How often the wait re-reads the run dir. The run writes metrics.json once, at the
# end, so polling faster buys nothing and only spins the server.
_RUN_POLL_S = 2.0
# Which run channel a waiting op='run' publishes under. The channel is the blocking
# tool's own registered name, so the client can address it off the tool row it is
# diverting without either end spelling the other's name out.
_DIVERT_CHANNEL = "proxy_eval"
# How long op='run' waits before handing the job to the client watcher instead.
# Sits under the tool-call budget proxy_eval declares, leaving room for the ratchet
# to settle and the results to be read in the same call. That budget is capped
# client-side at TOOL_CALL_TIMEOUT_MAX_SECS (1200 s) whatever the tool declares, so
# a larger value here means the client always cuts the call before the server detaches,
# and the orderly hand-off to the watcher never happens. With a build ahead of the
# measurement, that is the common case rather than a corner.
_RUN_WAIT_BUDGET_S = 1100.0
# Above this measured build time, a run is worth detaching rather than waiting out.
# One minute: short enough to catch any real compiled project, long enough that a
# scripted proxy with a trivial build never triggers it.
_LONG_BUILD_S = 60.0


def _opt_tail_log(run_dir: str, n: int) -> list[str]:
    lines = _read_log(run_dir).splitlines()
    return lines[-n:]


# ── act ops (confirm already checked by the dispatch tool) ────────────────────

def _init_payload(*, proxy_name, benchmark_name, abs_src, abs_paths, baseline_id,
                  baseline_existed, requirements, primary_goal, primary_metric,
                  min_improvement, cfg) -> dict:
    """What init answers with: the session as recorded, and how to work in it.

    Separated from the persistence above because it is prose, and because it is the
    only place the loop's rules are stated to the caller — what to edit, what the
    first run has to be, and what an edit must beat.
    """
    repeat = _effective_repeat(cfg)
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
    entry, error = _entry_or_err(proxy_name)
    if error:
        return error

    suite = _load_suite(benchmark_name)
    if suite is None:
        return err(f"Benchmark suite '{benchmark_name}' not found.",
                   hint="Create one with proxy_exec(op='benchmark_create', ...) "
                        "or proxy_manage(op='suite_define', ...) first.")

    req_err = _check_requirements(requirements)
    if req_err:
        return err(req_err)

    ref_err = _check_reference_requirements(requirements, suite, entry)
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

    paths_err = _check_optimize_paths(list(optimize_paths or []), abs_src)
    if paths_err:
        return err(paths_err)
    abs_paths = [os.path.abspath(p) for p in (optimize_paths or [])]
    if python_executable and not os.path.isfile(python_executable):
        return err(f"python_executable not found: {python_executable}")

    # The baseline snapshots the whole tracked TREE, not one file: the state an accepted
    # run is restored to has to be a state that was measured as a whole.
    #
    # A re-init NEVER moves an existing baseline. Re-initialising mid-optimisation is
    # how requirements get widened or a benchmark swapped, and re-snapshotting there
    # would quietly promote the current, already-optimised tree to "the original" —
    # after which every comparison is against work already done, and the honest question
    # "is this faster than what we started with?" can no longer be asked.
    os.makedirs(cache_dir(), exist_ok=True)
    prior = _load_opt_config(proxy_name) or {}
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
        # Content hash of the untouched tree. _prepare_run refuses the first run if the
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
        "initialized_at":    datetime.now(timezone.utc).isoformat(),
    }
    _save_opt_config(cfg)
    _write_active_session(proxy_name)

    return ok(_with_next(_init_payload(
        proxy_name=proxy_name, benchmark_name=benchmark_name, abs_src=abs_src,
        abs_paths=abs_paths, baseline_id=baseline_id,
        baseline_existed=baseline_existed, requirements=requirements,
        primary_goal=primary_goal, primary_metric=primary_metric,
        min_improvement=min_improvement, cfg=cfg), _NEXT_RUN))


def configure(
    proxy_name: str = "",
    requirements: list[dict] | None = None,
    benchmark_name: str = "",
    python_executable: str = "",
    max_hours: float = 0.0,
) -> dict:
    cfg = _load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.",
                   hint="Call proxy_eval(op='init', ...) first.")

    if requirements is not None:
        req_err = _check_requirements(requirements)
        if req_err:
            return err(req_err)
        cfg["requirements"] = requirements

    if benchmark_name:
        if _load_suite(benchmark_name) is None:
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
    final_suite = _load_suite(cfg.get("benchmark_name", "")) or {}
    entry, error = _entry_or_err(cfg.get("proxy_name", ""))
    if error:
        return error
    ref_err = _check_reference_requirements(
        cfg.get("requirements") or [], final_suite, entry)
    if ref_err:
        return err(ref_err)

    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_opt_config(cfg)
    return ok(_with_next({"config": cfg}, _NEXT_RUN))


def _resume_notice(cfg: dict, name: str, paths: list[str]) -> str:
    """Say when a run is resuming a session whose code moved on without it.

    ``_prepare_run`` refuses a first run on an already-edited tree, but only while no
    baseline run is on record; once one is, nothing checks again. That leaves a gap in
    the middle: coming back to a finished optimisation later measures against a best
    taken on code that is no longer there, and says nothing about it.

    The hard part is not noticing that the tree changed: during an optimisation it changes
    on every iteration, which is the loop working. What distinguishes the two cases is
    whether this is a **resumption** — a run on a session that is not the current one,
    because it was ended and is being picked up again. Inside a live session the active
    pointer names this proxy and nothing fires.

    Compared against the last run's launch tree, not against the baseline: the baseline
    differs from every candidate by construction, while "different from what we last
    measured, and we are only now coming back" is precisely "something changed that this
    loop did not do".

    A notice, never a refusal. Continuing can be exactly right — the change may be the very
    thing being measured — and the caller is the one who knows.
    """
    if not paths or not cfg.get("baseline_run_id"):
        return ""
    if _resolve_proxy_name("") == name:
        return ""  # a live session, not a resumption
    last_run = _opt_active_run_dir(name)
    if not last_run:
        return ""
    measured = (_read_json(os.path.join(last_run, "tree_at_launch.json")) or {}).get(
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


def _prepare_run(
    proxy_name: str,
) -> tuple[dict | None, dict | None, str | None, str]:
    """Shared preamble for local/Slurm eval runs.

    Returns ``(cfg, err_response, run_dir, notice)``: on failure *err_response* is set;
    on success *cfg* holds the session config and *run_dir* the fresh run dir
    (config.json + start_time already written). *notice* is a caller-facing warning about
    the run being launched (see :func:`_resume_notice`), empty when there is nothing to
    say. Returned rather than hung on *cfg*, which is persisted elsewhere by
    ``_save_opt_config`` — a warning about one run has no business on disk.
    """
    cfg = _load_opt_config(proxy_name)
    if not cfg:
        return None, err("No optimization session found.",
                         hint="Call proxy_eval(op='init', ...) first."), None, ""

    name           = cfg.get("proxy_name", "")
    benchmark_name = cfg.get("benchmark_name", "")
    if not name or not benchmark_name:
        return None, err("proxy_name and benchmark_name must be set.",
                         hint="Call proxy_eval(op='init', ...) first."), None, ""

    active = _opt_active_run_dir(name)
    if active:
        rs = _run_state(active)
        if rs["state"] in ("running", "pending"):
            pid = rs.get("pid")
            jid = rs.get("slurm_job_id")
            tag = f"slurm_job_id={jid}" if jid else f"pid={pid}"
            return None, err(f"An optimization run is already active ({tag}).",
                             hint="Use proxy_eval(op='stop', confirm=True) first."), None, ""

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
            return None, err(
                "No baseline run on record, and the tracked files have already been "
                "edited — this run would have nothing to be compared against.",
                hint="Restore the original with proxy_eval(op='reset', confirm=True), "
                     "run once to measure it, then optimize from there."), None, ""

    notice = _resume_notice(cfg, name, paths)

    run_dir = _new_run_dir(_opt_session_runs_dir(name))
    _write_run_config(run_dir, {
        "proxy_name":        name,
        "benchmark_name":    benchmark_name,
        "requirements":      cfg.get("requirements", []),
        "python_executable": cfg.get("python_executable", ""),
        "deadline_s":        (cfg.get("max_hours") or 24.0) * 3600,
        "convergence":       cfg.get("convergence") or {},
        "primary_metric":    cfg.get("primary_metric", "time_s"),
        "repeat":            _effective_repeat(cfg),
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
        _write_json_atomic(os.path.join(run_dir, "tree_at_launch.json"), {
            "snapshot_id": launch_id,
            "paths":       paths,
            "fingerprint": tree_snapshot.fingerprint(workspace_root(), paths),
            "is_baseline": launch_id == cfg.get("baseline_id", ""),
        })
    return cfg, None, run_dir, notice


def _background_descriptor(name: str, run_dir: str) -> dict:
    """The watcher handle for an optimization run: it polls the session, not the run."""
    return background_descriptor(
        run_dir, kind="proxy-optimization",
        status_op={"tool": "proxy_eval_status", "args": {"proxy_name": name}},
        summary_op={"tool": "proxy_eval_status",
                    "args": {"op": "results", "proxy_name": name}})


def run(proxy_name: str = "", background: bool = False) -> dict:
    """Launch a background optimization run (non-blocking).

    ``background=True`` attaches a ``background_job`` descriptor so a client watcher
    monitors completion off the agent's critical path and auto-resumes the agent with
    the results — the agent should end its turn instead of polling.
    """
    cfg, error, run_dir, resume_notice = _prepare_run(proxy_name)
    if error:
        return error
    name       = cfg["proxy_name"]
    log_file   = _log_path(run_dir)
    python_exe = cfg.get("python_executable") or sys.executable

    pid = _launch_detached(
        [python_exe, _OPT_RUNNER, "--run-dir", run_dir], run_dir, log_file=log_file,
    )
    _update_opt_active_link(name, run_dir)

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
        payload["background_job"] = _background_descriptor(name, run_dir)
        # States what was started; whether a watcher is on it is the client's to say.
        payload["note"] = "Optimization run detached; this call returns before it ends."
        return ok(_with_next(payload, _NEXT_DETACHED))
    return ok(_with_next(payload, _NEXT_STATUS))


async def _await_terminal_state(
    run_dir: str, wait_s: float, channel: str = "", job_key: str = "",
    pid: int = 0,
) -> tuple[str | None, str]:
    """Poll *run_dir* until the run finishes, reporting what it is doing as it goes.

    Returns ``(state, reason)``. A terminal state comes back with an empty reason;
    ``None`` comes with the reason the wait ended anyway — ``"budget"`` when *wait_s*
    ran out, ``"diverted"`` when the user asked for the run to be backgrounded. The
    two look identical to the run, which carries on either way, but they read very
    differently to the person who caused one of them.

    Async on purpose. FastMCP calls a synchronous tool directly on the server's
    event loop, so a blocking sleep here would stop this process answering
    anything for the whole wait — including the ``proxy_eval_status`` a watcher
    polls, which is exactly what the caller falls back to when the budget runs out.

    While it waits it owns the run channel (see ``_shared/run_channel``): the phase
    goes out on every tick, and a divert request left by the user is consumed on the
    same one. This is the only place either can happen — the tool call itself cannot
    answer until the run is over, which is precisely the problem.
    """
    deadline = time.monotonic() + max(0.0, wait_s)
    if channel:
        # The launcher's pid, passed in rather than read back: the caller has just
        # started the process, and an extra state read here would be one more poll of
        # the run dir for a fact already in hand.
        run_channel.publish(channel, job_key, pid,
                            f"optimization run {os.path.basename(run_dir)}",
                            run_dir, wait_s)
    try:
        while True:
            # One read serves both the terminal test and the phase: `_run_state`
            # already folds in what the run says it is doing.
            st = _run_state(run_dir)
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
    wait_s: float = _RUN_WAIT_BUDGET_S,
) -> dict:
    """Launch an optimization run and wait for its verdict (the default for op='run').

    One call, one answer: the run is launched, awaited, settled by the ratchet, and
    the ``results`` payload comes back inline. Repeatedly asking
    ``proxy_eval_status`` for a run whose only interesting moment is its end spends
    turns to learn "still running", and the ratchet verdict is the thing worth
    reading anyway.

    Two ways out of the wait, both of which detach rather than fail: ``background=True``
    asks for it up front, and a run still going after *wait_s* is handed to the client
    watcher on its own. Either way the response carries a ``background_job``
    descriptor and tells the agent to end its turn — it is resumed with the results.
    """
    launched = run(proxy_name, background=background)
    if background or launched.get("status") != "ok":
        return launched

    run_dir = launched.get("run_dir", "")
    name    = launched.get("proxy_name", "")
    state, reason = await _await_terminal_state(
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
        detached["background_job"] = _background_descriptor(name, run_dir)
        return ok(_with_next(detached, _NEXT_DETACHED))

    settled = results(name)
    # Keep the launch identity on the answer: which run this was, and where its log
    # is, are what a crash diagnosis needs and `results` does not carry. `resume_notice`
    # rides along for a blunter reason: this branch rebuilds its reply from `results()`,
    # so anything set only on the launch payload is dropped here — and this is the branch
    # a caller actually reaches, which would have made the warning invisible exactly
    # where it matters.
    for key in ("pid", "log", "benchmark_name", "resume_notice"):
        settled.setdefault(key, launched.get(key))
    settled.setdefault("run_dir", run_dir)
    return settled


def stop(proxy_name: str = "") -> dict:
    resolved = _resolve_proxy_name(proxy_name)
    if not resolved:
        return err("No proxy name or active session.")
    run_dir = _opt_active_run_dir(resolved)
    if not run_dir:
        return err("No active optimization run found.")

    result = _cancel_run(run_dir)
    if "error" in result:
        return err(result["error"])
    return ok(_with_next({"run_dir": run_dir, **result}, _NEXT_RUN + " to start a new run"))


def _restored_payload(paths: list[str], rewritten: list[str]) -> dict:
    """What a restore actually did, rather than what it was asked to cover.

    Listing every tracked file whatever happened is untrue the moment most of them were
    byte-identical and never touched. Naming only the rewritten ones makes the answer
    match the tree, and makes the next build's size predictable from it: those are
    exactly the files the build system now sees as new.
    """
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
    cfg = _load_opt_config(proxy_name)
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

    return ok(_with_next({
        **_restored_payload(paths, rewritten),
        "from":      "baseline",
        "note": "Every tracked file is back to the baseline taken at init. "
                "Try a different modification approach."
                + _rebuild_note(rewritten),
    }, _NEXT_RUN + " to verify the baseline"))


def reset_to_best(proxy_name: str = "") -> dict:
    """Restore the proxy source file from the best-so-far snapshot.

    Unlike ``reset`` (which restores the original canonical baseline), this reverts
    to the source of the best *accepted* run — the way to undo a regression without
    discarding the improvements found so far.
    """
    cfg = _load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.",
                   hint="Call proxy_eval(op='init', ...) first.")
    name = cfg.get("proxy_name", "")
    best = _load_best(name)
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

    return ok(_with_next({
        **_restored_payload(paths, rewritten),
        "from_best":    best.get("run_id"),
        "primary_value": best.get("primary_value"),
        "note": "Every tracked file is back to the state of the best accepted run. "
                "Try a different modification approach from there."
                + _rebuild_note(rewritten),
    }, _NEXT_RUN + " to verify, or summarize if converged"))


def rebaseline(proxy_name: str = "") -> dict:
    """Re-measure from the current tree, keeping the history that led here.

    ``init`` refuses to move an existing baseline, and it is right to: re-snapshotting
    mid-optimisation quietly promotes already-optimised code to "the original", after
    which "is this faster than what we started with?" can no longer be asked. But that
    invariant answered only half the question. The other half — *the instrument was
    wrong, measure again from here* — had no supported answer at all, and a harness is
    sealed before anyone has seen it produce a number.

    So it got an unsupported one::

        end -> proxy_manage(op='clean') -> init -> run

    the only sequence that moves a baseline, and it moves it by destroying the ledger to
    get there — accepted runs and their numbers gone from the record while their code
    stays on disk, uncredited.

    This op is that sequence made honest. The ledger and the best-so-far are *archived*
    under a timestamp rather than deleted, the new baseline is taken from the tree as it
    stands, and the reply says plainly that comparisons now run against this point and
    not against the original.
    """
    cfg = _load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.",
                   hint="Call proxy_eval(op='init', ...) first.")

    name = cfg.get("proxy_name", "") or _resolve_proxy_name(proxy_name) or ""
    paths = [os.path.abspath(p) for p in (cfg.get("optimize_paths") or [])]
    if not paths:
        return err("optimize_paths not set in config.",
                   hint="Call proxy_eval(op='init', ...) again.")

    active = _opt_active_run_dir(name)
    if active and _run_state(active).get("state") in ("running", "pending"):
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
    cfg["stall"] = 0
    cfg["rebaselined_at"] = datetime.now(timezone.utc).isoformat()
    cfg["previous_baseline_id"] = previous
    _save_opt_config(cfg)
    _write_active_session(name)

    return ok(_with_next({
        "rebaselined":         name,
        "baseline_id":         baseline_id,
        "previous_baseline_id": previous,
        "archived":            archived,
        "note": "The tree as it stands is the new baseline. Every later comparison is "
                "against THIS point, not against the original — the ledger and best-so-far "
                "that led here are archived, not deleted, so the earlier progression is "
                "still on record. Measure the new baseline before editing anything.",
    }, _NEXT_RUN + " to measure the new baseline"))


def _archive_session_history(proxy_name: str, stamp: str) -> list[str]:
    """Move the ledger and best-so-far aside under *stamp*; return what moved.

    Renamed rather than removed. A ledger is the only record that a run happened at
    all, and the reason to re-baseline is never "that history was wrong".
    """
    moved: list[str] = []
    for path in (_opt_ledger_file(proxy_name), _opt_best_file(proxy_name)):
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
    """End the optimization session: drop the active-session pointer.

    A clean close for the session. The proxy source, snapshots, ledger and best
    stay on disk for history — only the "current session" marker is cleared, so
    subsequent nameless ops no longer resolve to it and the client's direct-exec
    guard lifts (the proxy may be run by hand again). Refuses while a run is still
    in flight so a live job is never orphaned. Re-``init`` to optimize again.
    """
    cfg = _load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.",
                   hint="Nothing to end. Call proxy_eval(op='init', ...) to start one.")
    name = cfg.get("proxy_name", "") or _resolve_proxy_name(proxy_name)

    active = _opt_active_run_dir(name)
    if active and _run_state(active).get("state") in ("running", "pending"):
        return err("A run is still active — stop it before ending the session.",
                   hint="proxy_eval(op='stop', confirm=True), then proxy_eval(op='end', confirm=True).")

    _clear_active_session()
    return ok(_with_next({
        "ended":  name,
        "note": "Optimization session ended; source/snapshots/ledger kept for history. "
                "The proxy can be run directly again. Re-init to optimize further.",
    }, "proxy_eval(op='init', ...) to start a new session, or you are done."))


# ── observe ops ───────────────────────────────────────────────────────────────

def status(proxy_name: str = "") -> dict:
    resolved = _resolve_proxy_name(proxy_name)
    if not resolved:
        return ok(_with_next(
            {"state": "no_session", "note": "No proxy name or active session."},
            "proxy_eval(op='init', ...)"))
    run_dir = _opt_active_run_dir(resolved)
    if not run_dir:
        return ok(_with_next(
            {"state": "no_runs", "note": "No optimization runs found."}, _NEXT_RUN))
    rs         = _run_state(run_dir)
    last_lines = _opt_tail_log(run_dir, 3)
    next_step = {
        "running": "poll proxy_eval_status() again until state is 'done'",
        "pending": "poll proxy_eval_status() again until state is 'done'",
        "done":    _NEXT_RESULTS,
        "crashed": "proxy_eval_status(op='log', tail=100) to diagnose the failure",
    }.get(rs["state"], _NEXT_RESULTS)
    payload = {
        "run_dir":        run_dir,
        "state":          rs["state"],
        "pid":            rs["pid"],
        "slurm_job_id":   rs["slurm_job_id"],
        "elapsed_s":      rs["elapsed_s"],
        "last_log_lines": last_lines,
    }
    # What the run is doing, when it says. Carried here and not only on the blocking
    # wait's own channel because this op is what a *detached* run is watched through:
    # without it, a run moved to the background goes quiet the moment it is moved.
    for key in ("phase", "percent"):
        if rs.get(key) is not None:
            payload[key] = rs[key]
    # A handle for a run still going, so asking where it is at puts a watcher back on
    # it: the watchers belong to the agent that made them and do not outlive it, while
    # the run — a detached process with its own run dir — carries on regardless.
    # Plural, never ``background_job``: that key says *this call launched it*, and this
    # one only looked.
    if rs["state"] in ("running", "pending"):
        payload["background_jobs"] = [_background_descriptor(resolved, run_dir)]
    return ok(_with_next(payload, next_step))


def _recommend(r: dict, summary: dict, proxy_source: str) -> tuple[str, str]:
    """What the ratchet's verdict means for the caller, and what to do next.

    A pure function of the settled outcome: four verdicts, four answers. It sits
    apart from ``results`` because it is the loop's script — the sentences that
    decide whether the next call is an edit, a revert, or a summary — and burying
    it in a payload builder made it look like formatting.
    """
    verdict  = r["verdict"]
    best     = r.get("best")
    best_id  = best.get("run_id") if best else None
    best_val = best.get("primary_value") if best else None
    pm, pv   = r["primary_metric"], r["primary_value"]

    if verdict == "converged":
        return (
            f"Converged: all requirements pass and {pm} did not improve for "
            f"{r['max_stall']} iterations (best run {best_id}, {pm}={best_val}). "
            "Restore the best implementation and summarize.",
            _NEXT_RESET_BEST + ", then summarize the results.")
    if verdict == "accept":
        return (
            f"Accepted — new best for {pm} ({pv}) with all requirements passing. "
            f"Try to improve {pm} further, or restore the best and summarize if "
            f"satisfied.",
            f"edit '{proxy_source}' to improve {pm}, then {_NEXT_RUN} "
            f"(or {_NEXT_RESET_BEST} + summarize)")
    if verdict == "reject":
        if r["feasible"]:
            why = (f"Requirements still pass but {pm}={pv} did not beat the best "
                   f"({best_val}). This edit is not an improvement — revert to the "
                   f"best and try a different approach.")
        else:
            why = ("This change regressed: requirements no longer pass. Revert to "
                   "the best-so-far and try a different modification.")
        return why, _NEXT_RESET_BEST + f", then edit '{proxy_source}' and {_NEXT_RUN}"

    # verdict is None — not feasible yet, and no best to fall back on.
    passed = summary.get("cases_passed", 0)
    total  = summary.get("cases_total", 0)
    return (
        f"{total - passed} of {total} case(s) failed requirements. "
        f"Read and modify '{proxy_source}' with the file tools, then run again.",
        f"edit '{proxy_source}', then {_NEXT_RUN}")


def results(proxy_name: str = "") -> dict:
    """Per-case benchmark results + requirements check, parsed from the run log."""
    resolved = _resolve_proxy_name(proxy_name)
    if not resolved:
        return err("No proxy name or active session.")
    run_dir = _opt_active_run_dir(resolved)
    if not run_dir:
        return err("No optimization runs found.")

    cases, summary = parse_runner_log(_read_log(run_dir))

    final_metrics = _read_json(os.path.join(run_dir, "metrics.json"))

    cfg          = _load_opt_config(proxy_name)
    proxy_source = cfg.get("proxy_source_path", "")
    complete     = isinstance(final_metrics, dict) and "all_passed" in final_metrics

    # A run stopped by its build has no metrics.json, which without this would be
    # reported as "may still be in progress" — the one answer that hides the cause
    # and invites a wait for a run that is already over.
    bld = build.read_report(run_dir)
    if not complete and bld and bld.get("status") not in (None, "ok"):
        log_path = procs._build_log_path(run_dir)
        return err(
            build.failure_summary(bld) + " — no measurement was taken.",
            hint="Fix the build and run again; nothing was accepted or recorded. "
                 + build._SHELL_HINT,
            run_dir=run_dir,
            state="crashed",
            build=bld,
            build_log=log_path,
            build_log_tail=build_progress.read_tail(log_path, 8192),
            next_step="read build_log_tail, fix the build, then "
                      "proxy_eval(op='run', confirm=True)",
        )

    if not complete:
        # Run has not produced a summary yet — still in progress.
        return ok(_with_next({
            "run_dir":        run_dir,
            "state":          _run_state(run_dir)["state"],
            "cases":          cases,
            "summary":        summary,
            "final_metrics":  final_metrics,
            "recommendation": "No summary yet — the run may still be in progress.",
        }, "proxy_eval_status() to check the run state"))

    r        = _ratchet_state(proxy_name, run_dir, final_metrics, cfg)
    verdict  = r["verdict"]
    best     = r.get("best")
    best_id  = best.get("run_id") if best else None
    best_val = best.get("primary_value") if best else None
    pm, pv   = r["primary_metric"], r["primary_value"]

    recommendation, next_step = _recommend(r, summary, proxy_source)

    if r.get("timing_warning"):
        recommendation += " WARNING: " + r["timing_warning"]

    # Said only once the run has measured how long its own build takes: a constant
    # cannot know whether this project compiles in two seconds or half an hour, and
    # advising a detach on a two-second build would be noise on every reply.
    build_s = (final_metrics or {}).get("build_s") or 0.0
    recommendation += _long_build_note(build_s)

    return ok(_with_next({
        "run_dir":        run_dir,
        **({"build_s": build_s} if build_s else {}),
        "state":          _run_state(run_dir)["state"],
        "cases":          cases,
        "summary":        summary,
        "final_metrics":  final_metrics,
        "verdict":        verdict,
        "feasible":       r["feasible"],
        "primary_metric": pm,
        "primary_value":  pv,
        "wall_value":     r.get("wall_value"),
        "best":           {"run_id": best_id, "primary_value": best_val},
        "stall":          r["stall"],
        "recommendation": recommendation,
        **({"timing_warning": r["timing_warning"]} if r.get("timing_warning") else {}),
    }, next_step))


def _long_build_note(build_s: float) -> str:
    """Advice to detach the next run, or "" when the build is not worth it."""
    if not build_s or build_s < _LONG_BUILD_S:
        return ""
    return (f" This run spent {build_s:.0f}s building. Pass background=True to the "
            "next run: it detaches immediately and you are resumed with the verdict, "
            "so the wait is not yours to sit through.")


def log(proxy_name: str = "", tail: int = 50) -> dict:
    tail = max(1, min(tail, 500))
    resolved = _resolve_proxy_name(proxy_name)
    if not resolved:
        return err("No proxy name or active session.")
    run_dir = _opt_active_run_dir(resolved)
    if not run_dir:
        return err("No optimization runs found.")

    if not os.path.isfile(_log_path(run_dir)):
        return err("Log file not yet created — run may not have started writing output.")
    content = _read_log(run_dir)

    all_lines = content.splitlines()
    rs        = _run_state(run_dir)
    return ok(_with_next({
        "run_dir":     run_dir,
        "state":       rs["state"],
        "total_lines": len(all_lines),
        "lines":       all_lines[-tail:],
    }, "proxy_eval(op='status') for the ratchet verdict on this run."))


def runs_list(proxy_name: str = "") -> dict:
    resolved = _resolve_proxy_name(proxy_name)
    if not resolved:
        return ok(_with_next({"runs": [], "count": 0,
                              "note": "No proxy name or active session."},
                             "proxy_eval(op='init', ...) to start a session."))
    session_dir = _opt_session_runs_dir(resolved)
    if not os.path.isdir(session_dir):
        return ok(_with_next({"runs": [], "count": 0},
                             "proxy_eval(op='run', confirm=True) to produce a first run."))

    names = _run_dir_names(session_dir)
    best        = _load_best(resolved)
    best_run_id = best.get("run_id") if best else None
    runs: list[dict] = []
    for name in names:
        run_dir = os.path.join(session_dir, name)
        rs      = _run_state(run_dir)
        entry: dict = {"run_id": name, "run_dir": run_dir,
                       "state": rs["state"], "elapsed_s": rs["elapsed_s"],
                       "is_best": name == best_run_id}
        m = _read_json(os.path.join(run_dir, "metrics.json"))
        if isinstance(m, dict):
            for k in ("cases_passed", "cases_total", "all_passed",
                      "best_case", "best_time_s", "convergence_order"):
                entry[k] = m.get(k)
        runs.append(entry)
    return ok(_with_next(
        {"runs": runs, "count": len(runs), "best_run_id": best_run_id},
        "proxy_eval(op='diff') to compare the last two runs."))


def runs_diff(run_a: str = "", run_b: str = "", proxy_name: str = "") -> dict:
    resolved = _resolve_proxy_name(proxy_name)
    if not resolved:
        return err("No proxy name or active session.")
    session_dir = _opt_session_runs_dir(resolved)
    if not os.path.isdir(session_dir):
        return err("No optimization runs found.")

    all_runs = _run_dir_names(session_dir)

    def _resolve(rid: str, default_idx: int) -> str:
        rid = rid or all_runs[default_idx]
        return rid if os.path.isabs(rid) else os.path.join(session_dir, rid)

    payload, error = _diff_run_pair(all_runs, run_a, run_b, _resolve)
    if error:
        return err(error, hint="Run proxy_eval(op='run', confirm=True) more times first.")
    return ok(_with_next(payload,
                         "proxy_eval(op='results') for the ratchet view across all runs."))


def config_get(proxy_name: str = "") -> dict:
    cfg = _load_opt_config(proxy_name)
    if not cfg:
        return ok(_with_next({"config": None, "note": "No optimization session initialized."},
                             "proxy_eval(op='init', ...)"))
    return ok(_with_next({"config": cfg},
                         "proxy_eval(op='configure', ...) to change it, or op='run' to use it."))
