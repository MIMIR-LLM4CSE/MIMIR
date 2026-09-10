"""Optimization-session ops.

Act ops (dispatched by ``proxy_eval``): init, configure, run, stop, reset,
reset_to_best, end.
Observe ops (dispatched by ``proxy_eval_status``): status, results, log,
runs, diff, config.  Every response carries a ``next_step`` hint naming the
exact next call, so the loop is self-describing.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

from _ops import _PROXY_DIR, _with_next, err, ok
from _lib.metrics import _VALID_OPT_OPERATORS, _coerce
from _lib.procs import (
    _log_path, _read_log, _run_state,
    _new_run_dir, _write_run_config, _launch_detached, _cancel_run,
    _opt_active_run_dir, _update_opt_active_link,
)
from _lib.ratchet import (
    _load_best, _save_best, _append_ledger, _ratchet_verdict,
    _run_primary_value,
)
from _lib.report import _diff_run_pair
from _lib import tree_snapshot
from _lib.store import (
    cache_dir,
    _load_registry_or_err, _load_suite,
    _opt_config_file, _opt_session_runs_dir,
    _opt_ledger_file, _opt_best_file,
    _resolve_proxy_name, _write_active_session, _clear_active_session,
    _read_json, _write_json_atomic, _file_lock, _run_dir_names,
)

_OPT_RUNNER = os.path.join(_PROXY_DIR, "_proxy_runner.py")

# Computed server-side from a sealed reference, so a requirement on one of these
# can never be satisfied when a case has no reference — refuse it up front.
_REFERENCE_METRICS = ("conservation_residual", "l2_abs", "l2_rel",
                      "linf_abs", "linf_rel")

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
# How long op='run' waits before handing the job to the client watcher instead.
# Sits under the tool-call budget proxy_eval declares, leaving room for the ratchet
# to settle and the results to be read in the same call.
_RUN_WAIT_BUDGET_S = 1500.0


# ── session config helpers ────────────────────────────────────────────────────

def _load_opt_config(proxy_name: str = "") -> dict:
    name = _resolve_proxy_name(proxy_name)
    if not name:
        return {}
    return _read_json(_opt_config_file(name), {}) or {}


def _save_opt_config(cfg: dict, proxy_name: str = "") -> None:
    name = proxy_name or cfg.get("proxy_name", "")
    if not name:
        return
    _write_json_atomic(_opt_config_file(name), cfg)


def _opt_tail_log(run_dir: str, n: int) -> list[str]:
    lines = _read_log(run_dir).splitlines()
    return lines[-n:]


def _workspace_root() -> str:
    """The workspace the servers were started against (see server_bash's own copy)."""
    return os.path.realpath(os.path.abspath(os.environ.get("MCP_FILES_ROOT") or os.getcwd()))


def opt_git_dir() -> str:
    """Shadow repository for the tracked tree, inside the proxy store."""
    return os.path.join(cache_dir(), "opt.git")


def _check_optimize_paths(paths: list[str], proxy_source_path: str) -> str | None:
    """The proxy is a HARNESS; the code under optimisation is somewhere else.

    Nothing used to say so, and twice running the model answered the gap the cheapest
    way available: it wrote a self-contained script that reproduced the solver it was
    meant to accelerate — 189 lines mirroring a 307-line package — and the ratchet
    optimised the copy. The cost is not the manual port afterwards. The ratchet's whole
    guarantee (l2_rel against a sealed reference) then holds for the duplicate and for
    nothing that ships, and a standalone script does not even have the imports, the
    module-level initialisation or the memory layout of the package it mirrors: what got
    measured was not the thing.

    So the shape is refused rather than discouraged. The lesson of the tool-schema work
    is that prose is not followed and refusals are: this same skill already says "do NOT
    poll a backgrounded run", and a recorded session polled 157 times.

    Returns an error string, or None when the declared shape is sound.
    """
    root = _workspace_root()
    if not paths:
        return ("optimize_paths is required: name the file(s) the ratchet may edit. "
                "The proxy at proxy_source_path is a HARNESS — it runs the code and "
                "prints metrics — and optimize_paths is the code it exercises, which "
                "the harness should import rather than reproduce.")
    src_real = os.path.realpath(os.path.abspath(proxy_source_path))
    seen: set[str] = set()
    for raw in paths:
        p = os.path.realpath(os.path.abspath(raw))
        if not os.path.isfile(p):
            return f"optimize_paths entry not found: {raw}"
        if os.path.commonpath([p, root]) != root:
            return (f"optimize_paths entry is outside the workspace: {raw}. "
                    "The ratchet only edits code inside the workspace.")
        if p == src_real:
            return ("proxy_source_path cannot be one of optimize_paths. The harness "
                    "must not be its own subject: optimising the script that measures "
                    "means optimising a copy, and the accuracy constraints then say "
                    "nothing about the code you ship. Point optimize_paths at the real "
                    "module(s) and have the harness import them.")
        seen.add(p)
    return None


def _check_requirements(requirements: list[dict]) -> str | None:
    for i, req in enumerate(requirements):
        if not req.get("metric"):
            return f"requirements[{i}] is missing 'metric'."
        if req.get("operator") not in _VALID_OPT_OPERATORS:
            return (f"requirements[{i}] has invalid operator '{req.get('operator')}'. "
                    "Use one of: lt, gt, lte, gte, eq.")
        if req.get("threshold") is None:
            return f"requirements[{i}] is missing 'threshold'."
    return None


def _check_reference_requirements(
    requirements: list[dict], suite: dict, entry: dict,
) -> str | None:
    """Reject requirements that can never be satisfied with this setup.

    Reference-dependent metrics (see ``_REFERENCE_METRICS``) are computed
    server-side against a sealed reference; ``conservation_residual``
    additionally needs the registration to name the conserved scalar. Failing
    fast here turns a dead-end session into an actionable setup error.
    """
    needed = sorted({r.get("metric") for r in requirements}
                    & set(_REFERENCE_METRICS))
    if not needed:
        return None
    cases = suite.get("cases") or []
    no_ref = [str(c.get("case_id", "?")) for c in cases
              if not c.get("reference_name")]
    if not cases or no_ref:
        which = ", ".join(no_ref) if no_ref else "(no cases defined)"
        return (f"Requirement(s) {', '.join(needed)} are computed server-side "
                f"against a sealed reference, but benchmark case(s) {which} "
                "have no reference_name — they could never pass. Create the "
                "benchmark with proxy_exec(op='benchmark_create', "
                "reference_params=...), which seals a reference, or add "
                "reference_name to every case. Values printed by the proxy "
                "for these metrics are ignored.")
    if "conservation_residual" in needed and not entry.get("conserved_metric"):
        return ("Requirement conservation_residual needs the proxy "
                "registration to declare which scalar is conserved: "
                "proxy_manage(op='update', metadata="
                "{'conserved_metric': '<metric name>'}, confirm=True).")
    return None


# ── act ops (confirm already checked by the dispatch tool) ────────────────────

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
    reg, _reg_err = _load_registry_or_err()
    if _reg_err:
        return err(_reg_err)
    if proxy_name not in reg:
        return err(f"Proxy '{proxy_name}' not registered.",
                   hint="Call proxy_manage(op='register', ...) first.")

    suite = _load_suite(benchmark_name)
    if suite is None:
        return err(f"Benchmark suite '{benchmark_name}' not found.",
                   hint="Create one with proxy_exec(op='benchmark_create', ...) "
                        "or proxy_manage(op='suite_define', ...) first.")

    req_err = _check_requirements(requirements)
    if req_err:
        return err(req_err)

    ref_err = _check_reference_requirements(requirements, suite, reg[proxy_name])
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
            opt_git_dir(), _workspace_root(), abs_paths,
            f"baseline: {proxy_name} before optimisation",
        )
        if not baseline_id:
            return err("Could not snapshot the baseline tree.",
                       hint="Check that optimize_paths are readable and the proxy store "
                            "is writable.")
        baseline_fp  = tree_snapshot.fingerprint(_workspace_root(), abs_paths)
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

    return ok(_with_next({
        "proxy_name":        proxy_name,
        "benchmark_name":    benchmark_name,
        "proxy_source_path": abs_src,
        "optimize_paths":    abs_paths,
        "baseline_id":       baseline_id,
        "baseline_existed":  baseline_existed,
        # Said on the reply, not buried in a doc: this is exactly the moment a model
        # discovers its harness was wrong, and the only route it used to find from
        # here was end + clean + init, which moves the baseline by destroying the
        # ledger. Naming the supported one costs a line.
        **({"baseline_note":
            "This baseline was taken by an earlier init and has NOT been moved — "
            "comparisons still run against the original tree. If the harness itself "
            "was wrong and the baseline must be re-measured from the tree as it "
            "stands, use proxy_eval(op='rebaseline', confirm=True): it archives the "
            "ledger and best-so-far instead of discarding them."}
           if baseline_existed else {}),
        "requirements":      requirements,
        "objective":         f"{primary_goal}imize {primary_metric} subject to the requirements",
        # What each run will actually cost, and what the accept threshold is until
        # the machine has been measured. Both were previously silent, and the
        # second one was a constant asserting a property of a machine nobody had
        # checked — it read "2% guards timing noise" on a node whose noise was 3.1%.
        "repeat":            _effective_repeat(cfg),
        "min_improvement":   float(min_improvement),
        "margin_note": (
            f"Each case is measured {_effective_repeat(cfg)}x and reduced to its "
            f"median. A run must beat the incumbent by {min_improvement:.1%} OR by "
            f"the spread measured across the baseline's own replicates, whichever "
            f"is larger — so an edit cannot be accepted on a difference this "
            f"machine produces without it."
        ),
        # Names optimize_paths, never proxy_source_path. The old wording here said
        # "Modify '<proxy_source_path>' between runs", which pointed the model at the
        # harness and is what produced a self-contained copy of the solver twice over.
        "note": ("Baseline already recorded — kept, not moved. " if baseline_existed
                 else "Baseline tree snapshotted. ") + "Edit "
            + ", ".join(os.path.basename(p) for p in abs_paths)
            + " between runs — never the harness at "
            + os.path.basename(abs_src) + ", which only runs the code and prints "
            "metrics. The FIRST run must be launched with those files untouched: it is "
            "the baseline every later number is compared against, and until it exists "
            "no run can be accepted. Feasible runs that improve the objective are "
            "accepted; regressions are rejected — revert them with "
            "proxy_eval(op='reset_to_best', confirm=True).",
    }, _NEXT_RUN))


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
    reg, reg_err = _load_registry_or_err()
    if reg_err:
        return err(reg_err)
    entry = (reg or {}).get(cfg.get("proxy_name", ""), {})
    ref_err = _check_reference_requirements(
        cfg.get("requirements") or [], final_suite, entry)
    if ref_err:
        return err(ref_err)

    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_opt_config(cfg)
    return ok(_with_next({"config": cfg}, _NEXT_RUN))


# Metrics whose value moves between two runs of identical code. A timing metric on a
# shared node is the whole reason this file needed a notion of noise; an accuracy metric
# against a sealed reference is reproducible to the bit, and repeating it buys nothing.
_NOISY_METRICS = ("time_s", "wall_time_s", "elapsed_s", "runtime_s", "gflops_per_s",
                  "bandwidth_gbytes_per_s")

# Replicates per case when the primary metric is a noisy one and the caller expressed no
# preference. Three is the smallest number with a middle — enough for a median to mean
# something, cheap enough to be the default.
_AUTO_REPEAT = 3


def _effective_repeat(cfg: dict) -> int:
    """How many times to measure each case: what was asked, or what the metric needs.

    ``repeat=0`` means "decide for me", and the decision is made on the metric rather
    than on a constant: repeating a bit-reproducible accuracy check is pure cost, and
    not repeating a timing on a 192-core shared node is how a 2.8% edit gets accepted
    against a 3.1% noise floor.
    """
    asked = int(cfg.get("repeat", 0) or 0)
    if asked > 0:
        return asked
    return _AUTO_REPEAT if cfg.get("primary_metric", "time_s") in _NOISY_METRICS else 1


def _effective_min_improvement(cfg: dict) -> tuple[float, str]:
    """The margin a run must clear, and where that number came from.

    ``min_improvement`` was a caller-supplied constant, defaulted to 0.02 and annotated
    "guards timing noise" — an assertion about a machine nobody had measured. Where the
    real floor is higher, the guard admits exactly the changes it exists to exclude:
    observed at 2% configured against 3.1% measured, with a kernel rewrite worth nothing
    accepted on a 2.8% "gain".

    So the configured value becomes a floor, not the answer. Once the baseline has been
    measured more than once, the spread of those measurements is known, and the margin
    is whichever of the two is larger.
    """
    configured = float(cfg.get("min_improvement", 0.0) or 0.0)
    floor = cfg.get("noise_floor")
    if not isinstance(floor, (int, float)) or floor <= configured:
        return configured, "configured"
    return float(floor), "measured noise floor"


def _resume_notice(cfg: dict, name: str, paths: list[str]) -> str:
    """Say when a run is resuming a session whose code moved on without it.

    Two guards used to cover one question between them, and left a gap in the middle.
    ``_prepare_run`` refuses a first run on an already-edited tree — but only while no
    baseline run is on record. Once one is, nothing checks anything again, so coming back
    to a finished optimisation months later ran it against a best measured on code that no
    longer exists, and said nothing about it.

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
    if not measured or measured == tree_snapshot.fingerprint(_workspace_root(), paths):
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
        current = tree_snapshot.fingerprint(_workspace_root(), paths)
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
        opt_git_dir(), _workspace_root(), paths,
        f"launch: {name} {os.path.basename(run_dir)}",
    ) if paths else None
    if launch_id:
        _write_json_atomic(os.path.join(run_dir, "tree_at_launch.json"), {
            "snapshot_id": launch_id,
            "paths":       paths,
            "fingerprint": tree_snapshot.fingerprint(_workspace_root(), paths),
            "is_baseline": launch_id == cfg.get("baseline_id", ""),
        })
    return cfg, None, run_dir, notice


def _background_descriptor(name: str, run_dir: str) -> dict:
    """Data-only handle a client watcher polls to completion (no tool name in loop code).

    ``status_op``/``summary_op`` name the read-only ops the watcher calls generically;
    the client augments the model's view and, on completion, auto-resumes.
    """
    return {
        "server":     "proxy",
        "run_dir":    run_dir,
        "job_key":    name,
        "kind":       "proxy-optimization",
        "status_op":  {"tool": "proxy_eval_status", "args": {"proxy_name": name}},
        "summary_op": {"tool": "proxy_eval_status",
                       "args": {"op": "results", "proxy_name": name}},
    }


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


async def _await_terminal_state(run_dir: str, wait_s: float) -> str | None:
    """Poll *run_dir* until the run finishes; ``None`` if *wait_s* elapsed first.

    Async on purpose. FastMCP calls a synchronous tool directly on the server's
    event loop, so a blocking sleep here would stop this process answering
    anything for the whole wait — including the ``proxy_eval_status`` a watcher
    polls, which is exactly what the caller falls back to when the budget runs out.
    """
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        state = _run_state(run_dir)["state"]
        if state in _TERMINAL_STATES:
            return state
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(_RUN_POLL_S)


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
    state   = await _await_terminal_state(run_dir, wait_s)

    if state is None:
        # Still running. Detaching beats letting the tool call time out: a timeout
        # would abandon a run that is alive and doing the work asked of it.
        detached = {k: v for k, v in launched.items() if k not in ("status", "next_step")}
        detached["note"] = (
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

    if not tree_snapshot.restore(opt_git_dir(), _workspace_root(), paths, baseline_id):
        return err("Could not restore the baseline tree.",
                   hint="The snapshot store may be missing or unwritable.")

    return ok(_with_next({
        "restored":  [os.path.basename(p) for p in paths],
        "from":      "baseline",
        "note": "Every tracked file is back to the baseline taken at init. "
                "Try a different modification approach.",
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
    if not tree_snapshot.restore(opt_git_dir(), _workspace_root(), paths, snapshot):
        return err("Could not restore the best tree.",
                   hint="Use proxy_eval(op='reset', confirm=True) to restore the baseline.")

    return ok(_with_next({
        "restored":     [os.path.basename(p) for p in paths],
        "from_best":    best.get("run_id"),
        "primary_value": best.get("primary_value"),
        "note": "Every tracked file is back to the state of the best accepted run. "
                "Try a different modification approach from there.",
    }, _NEXT_RUN + " to verify, or summarize if converged"))


def rebaseline(proxy_name: str = "") -> dict:
    """Re-measure from the current tree, keeping the history that led here.

    ``init`` refuses to move an existing baseline, and it is right to: re-snapshotting
    mid-optimisation quietly promotes already-optimised code to "the original", after
    which "is this faster than what we started with?" can no longer be asked. But that
    invariant answered only half the question. The other half — *the instrument was
    wrong, measure again from here* — had no supported answer at all, and a harness is
    sealed before anyone has seen it produce a number.

    So it got an unsupported one. Observed twice in three minutes in session
    ``7d322a3b``::

        end -> proxy_manage(op='clean') -> init -> run

    which is the only sequence that moves a baseline, and it moves it by destroying the
    ledger to get there. The run it discarded the second time was a real result: a
    boundary condition measured 40% better than the incumbent, gone from the record
    while its code stayed on disk, uncredited.

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
        opt_git_dir(), _workspace_root(), paths,
        f"rebaseline: {name} at {stamp}",
    )
    if not baseline_id:
        return err("Could not snapshot the new baseline tree.",
                   hint="Check that optimize_paths are readable and the proxy store "
                        "is writable.")

    previous = cfg.get("baseline_id", "")
    cfg["baseline_id"] = baseline_id
    cfg["baseline_fingerprint"] = tree_snapshot.fingerprint(_workspace_root(), paths)
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


def _ratchet_state(proxy_name: str, run_dir: str, final_metrics: dict, cfg: dict) -> dict:
    """Compute the ratchet outcome for a completed run, persisting it once.

    Whichever of the runner or ``results`` gets here first wins: the verdict is
    frozen to ``<run_dir>/ratchet.json`` under a per-session flock and re-checked
    inside it, so best/ledger/stall never move twice for the same run.
    """
    name = cfg.get("proxy_name") or proxy_name
    outcome_path = os.path.join(run_dir, "ratchet.json")

    def _frozen() -> dict | None:
        frozen = _read_json(outcome_path)  # None when missing or unreadable
        if frozen is None:
            return None  # recompute under the lock
        frozen["best"] = _load_best(name)
        return frozen

    out = _frozen()
    if out is not None:
        return out

    with _file_lock(os.path.join(_opt_session_runs_dir(name), ".ratchet.lock")):
        out = _frozen()
        if out is not None:
            return out
        return _ratchet_settle_locked(name, run_dir, final_metrics, cfg, outcome_path)


def _ratchet_settle_locked(
    name: str, run_dir: str, final_metrics: dict, cfg: dict, outcome_path: str,
) -> dict:
    primary_metric  = cfg.get("primary_metric", "time_s")
    goal            = cfg.get("primary_goal", "min")
    max_stall       = int(cfg.get("max_stall", 5))
    # The tree as it stood when this run was LAUNCHED — what actually ran, immune to
    # edits made while it was in flight.
    launch = _read_json(os.path.join(run_dir, "tree_at_launch.json")) or {}
    launch_tree = launch.get("snapshot_id", "")

    feasible      = bool(final_metrics.get("all_passed"))
    primary_value = _run_primary_value(final_metrics, primary_metric)
    wall_value    = _run_primary_value(final_metrics, "wall_time_s")
    run_id        = os.path.basename(os.path.normpath(run_dir))
    best          = _load_best(name)

    # The baseline is the one run whose spread describes the machine rather than the
    # change, so it is where the noise floor comes from — recorded once, then applied
    # to every comparison after it.
    spread = _run_primary_value(final_metrics, "primary_spread")
    if not cfg.get("baseline_run_id") and isinstance(spread, (int, float)):
        cfg["noise_floor"] = float(spread)

    min_improvement, margin_source = _effective_min_improvement(cfg)
    verdict = _ratchet_verdict(feasible, primary_value, best, goal, min_improvement)

    # An accepted time_s improvement whose measured wall time regressed is
    # plausible I/O noise, but also the signature of a tampered timer: warn.
    timing_warning: str | None = None
    if verdict == "accept" and primary_metric == "time_s" and best is not None:
        prev_wall = best.get("wall_value")
        if (isinstance(wall_value, (int, float)) and isinstance(prev_wall, (int, float))
                and wall_value > prev_wall * 1.10):
            timing_warning = (
                "Self-reported time_s improved but the server-measured wall time "
                f"regressed ({prev_wall:.3g}s -> {wall_value:.3g}s). Verify the "
                "proxy's timing instrumentation before trusting this acceptance, "
                "or optimize primary_metric='wall_time_s' instead."
            )

    # The launch gate in _prepare_run guarantees the first run measured the untouched
    # tree; this records which run that was, so the gate opens exactly once.
    baseline_run = cfg.get("baseline_run_id", "")
    if not baseline_run:
        cfg["baseline_run_id"] = baseline_run = run_id

    stall = int(cfg.get("stall", 0))
    if verdict == "accept":
        _save_best(name, run_id, primary_value, launch_tree, wall_value=wall_value)
        best  = _load_best(name)
        stall = 0
    elif feasible:
        stall += 1  # feasible but not an improvement — a stalled iteration

    if feasible and stall >= max_stall:
        verdict = "converged"

    cfg["stall"] = stall
    _save_opt_config(cfg)

    _append_ledger(name, {
        "run_id":        run_id,
        "ts":            datetime.now(timezone.utc).isoformat(),
        "feasible":      feasible,
        "primary_value": primary_value,
        "wall_value":    wall_value,
        "verdict":       verdict,
        "stall":          stall,
        "primary_spread": spread,
        "min_improvement": min_improvement,
        "best_run_id":   best.get("run_id") if best else None,
        **({"timing_warning": timing_warning} if timing_warning else {}),
    })

    outcome = {
        "baseline_run_id": baseline_run,
        "feasible":       feasible,
        "primary_value":  primary_value,
        "wall_value":     wall_value,
        "verdict":        verdict,
        "stall":          stall,
        "max_stall":      max_stall,
        "primary_metric": primary_metric,
        "goal":           goal,
        # Said on every run, not buried in the config: the margin is what decides
        # accept from reject, and a model that cannot see it cannot tell an edit
        # that did nothing from one the threshold was too loose to catch.
        "min_improvement": min_improvement,
        "margin_source":   margin_source,
        "primary_spread":  spread,
        "noise_floor":     cfg.get("noise_floor"),
        "timing_warning": timing_warning,
    }
    try:
        with open(outcome_path, "w") as fh:
            json.dump(outcome, fh, indent=2)
    except OSError:
        pass
    outcome["best"] = best
    return outcome


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
    return ok(_with_next({
        "run_dir":        run_dir,
        "state":          rs["state"],
        "pid":            rs["pid"],
        "slurm_job_id":   rs["slurm_job_id"],
        "elapsed_s":      rs["elapsed_s"],
        "last_log_lines": last_lines,
    }, next_step))


def results(proxy_name: str = "") -> dict:
    """Per-case benchmark results + requirements check, parsed from the run log."""
    resolved = _resolve_proxy_name(proxy_name)
    if not resolved:
        return err("No proxy name or active session.")
    run_dir = _opt_active_run_dir(resolved)
    if not run_dir:
        return err("No optimization runs found.")

    content = _read_log(run_dir)

    cases:   list[dict] = []
    summary: dict       = {}
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("[proxy_runner] case="):
            row: dict = {}
            for part in line.split()[1:]:
                if "=" in part:
                    k, _, v = part.partition("=")
                    row[k] = _coerce(v)
            if row:
                cases.append(row)
        elif line.startswith("[proxy_runner] summary "):
            for part in line.split()[2:]:
                if "=" in part:
                    k, _, v = part.partition("=")
                    summary[k] = _coerce(v)

    final_metrics = _read_json(os.path.join(run_dir, "metrics.json"))

    cfg          = _load_opt_config(proxy_name)
    proxy_source = cfg.get("proxy_source_path", "")
    complete     = isinstance(final_metrics, dict) and "all_passed" in final_metrics

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

    if verdict == "converged":
        recommendation = (
            f"Converged: all requirements pass and {pm} did not improve for "
            f"{r['max_stall']} iterations (best run {best_id}, {pm}={best_val}). "
            "Restore the best implementation and summarize."
        )
        next_step = _NEXT_RESET_BEST + ", then summarize the results."
    elif verdict == "accept":
        recommendation = (
            f"Accepted — new best for {pm} ({pv}) with all requirements passing. "
            f"Try to improve {pm} further, or restore the best and summarize if satisfied."
        )
        next_step = (f"edit '{proxy_source}' to improve {pm}, then {_NEXT_RUN} "
                     f"(or {_NEXT_RESET_BEST} + summarize)")
    elif verdict == "reject":
        if r["feasible"]:
            recommendation = (
                f"Requirements still pass but {pm}={pv} did not beat the best "
                f"({best_val}). This edit is not an improvement — revert to the best "
                "and try a different approach."
            )
        else:
            recommendation = (
                "This change regressed: requirements no longer pass. Revert to the "
                "best-so-far and try a different modification."
            )
        next_step = _NEXT_RESET_BEST + f", then edit '{proxy_source}' and {_NEXT_RUN}"
    else:  # verdict is None — not feasible yet and no best to fall back on
        passed = summary.get("cases_passed", 0)
        total  = summary.get("cases_total", 0)
        recommendation = (
            f"{total - passed} of {total} case(s) failed requirements. "
            f"Read and modify '{proxy_source}' with the file tools, then run again."
        )
        next_step = f"edit '{proxy_source}', then {_NEXT_RUN}"

    if r.get("timing_warning"):
        recommendation += " WARNING: " + r["timing_warning"]

    return ok(_with_next({
        "run_dir":        run_dir,
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
