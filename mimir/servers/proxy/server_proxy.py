"""MCP Proxy Server
===================
Registration, benchmarking, execution, and iterative optimization of proxy
scientific-computing codes, behind seven op-dispatched tools.

Tool catalogue
--------------
Read-only:
  proxy_get(op, ...)          — inspect proxies, references, suites, reports
  proxy_runs(op, ...)         — run history, logs, diffs, comparisons, aggregation
  proxy_eval_status(op, ...)  — observe the optimization session

Sensitive (confirm=True required):
  proxy_manage(op, ...)       — register/update/delete proxies and suites; scaffold
  proxy_exec(op, ...)         — local runs, reference sealing, suite runs, cancel
  proxy_eval(op, ...)         — drive the optimization ratchet
                                (init/run/stop/reset/reset_to_best)

Cluster (confirm=True required; consumes allocation hours):
  proxy_slurm(op, ...)        — submit single/suite/eval runs as Slurm batch jobs

Typical flows
-------------
Benchmark a proxy:
  proxy_manage(op='register', ...) -> proxy_exec(op='benchmark_create', ...)
  -> proxy_exec(op='suite', ...) -> proxy_get(op='report', ...)

Optimize a proxy — a monotone ratchet (see also the proxy-optimize skill):
  proxy_eval(op='init', primary_metric=..., ...) -> proxy_eval(op='run')
  -> proxy_eval_status() -> proxy_eval_status(op='results') -> follow the verdict:
     accepted -> edit source to improve further -> run again
     rejected -> proxy_eval(op='reset_to_best') -> try a different edit -> run
     converged -> proxy_eval(op='reset_to_best') -> summarize
  Feasible runs that improve the objective are accepted (new best-so-far);
  regressions are rejected. Numerical invariants gate feasibility.

Solver invocation and parameter files
-------------------------------------
Registered proxies are launched from ``run_cmd_template`` with placeholders:

  {executable}    — quoted absolute path to the proxy executable
  {output_file}   — absolute path where the proxy must write its output
  {param_file}    — absolute path of the rendered parameter file
  {extra_params}  — additional CLI flags supplied at run time

Any other placeholder (e.g. ``{n}``) is filled from ``param_overrides`` at run
time, in both ``run_cmd_template`` and ``param_file_template``.

Solver output protocol
----------------------
Solvers must write a metrics block to stdout::

    PROXY_METRICS_BEGIN
    time_s=4.72
    misfit=1.23e-3
    output_file=/abs/path/to/output.npz
    PROXY_METRICS_END

Without block markers the server falls back to a ``key=value`` scan of the
last log lines.  Supported field output formats: npz (default), raw_float64,
none.

Storage layout
--------------
All state lives under ``~/.cache/proxy_bench/`` (registry.json, references/,
runs/<proxy>/<timestamp>/, suites/<name>/, opt_runs/<proxy>/, scaffolds/).
Override the root with the ``MIMIR_PROXY_BENCH_DIR`` environment variable.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps, CODE_EXEC, PLAN_BLOCKED, CLUSTER_SUBMIT, BACKGROUNDABLE, IRREVERSIBLE, RECOVERABLE
from responses import err
from _lib.execute import _DEFAULT_MAX_OUTPUT_MB, _REF_RUN_TIMEOUT
from _lib.procs import _validate_slurm_args
from _ops import eval_session, references, registry, runs, scaffold as scaffold_ops, slurm, suites

mcp = FastMCP(
    "ProxyServer",
    debug=False,
    log_level="ERROR",
)


def _unknown_op(op: str, valid_ops: tuple[str, ...]) -> dict:
    hint = f"Use one of: {', '.join(valid_ops)}."
    # The 7 tools share op vocabularies, so a misrouted op is common. When it is
    # valid on another proxy tool, name that tool instead of just "Unknown op".
    owners = _OP_OWNERS.get(op, ())
    this_tool = _TOOL_BY_OPS.get(valid_ops)
    others = [t for t in owners if t != this_tool]
    if others:
        hint += (f" Note: op '{op}' belongs to {' / '.join(others)}"
                 + (f", not {this_tool}" if this_tool else "") + ".")
    return err(f"Unknown op '{op}'.", hint=hint)


def _missing_args(op: str, **required) -> dict | None:
    """Return an err() naming the empty required args for *op*, or None."""
    missing = [k for k, v in required.items() if not v]
    if missing:
        return err(f"op '{op}' requires: {', '.join(missing)}.",
                   hint="See the ops table in the tool description for per-op required args.")
    return None


# ── read-only tools ───────────────────────────────────────────────────────────

_GET_OPS = ("proxies", "proxy", "references", "suites", "suite", "report")


@mcp.tool(**tool_caps(label="Proxy info: {op}"))
def proxy_get(op: str, name: str = "", run_timestamp: str = "") -> dict:
    """Inspect registered proxies, references, benchmark suites, and reports.

    Operations (set ``op``):
      proxies     -> list all registered proxies
      proxy       -> full registration + readme + last-5-runs (requires: name)
      references  -> list sealed reference datasets
      suites      -> list benchmark suite definitions
      suite       -> suite definition + validation status (requires: name)
      report      -> aggregated results table of a suite run (requires: name;
                     run_timestamp optional, defaults to the latest run)

    Args:
        op: The operation to perform (see list above).
        name: Proxy name (op='proxy') or suite name (op='suite'/'report').
        run_timestamp: For 'report': timestamp tag of the suite run to show.
    """
    if op == "proxies":
        return registry.list_proxies()
    if op == "proxy":
        return _missing_args(op, name=name) or registry.inspect_proxy(name)
    if op == "references":
        return references.list_references()
    if op == "suites":
        return suites.list_suites()
    if op == "suite":
        return _missing_args(op, name=name) or suites.inspect_suite(name)
    if op == "report":
        return _missing_args(op, name=name) or suites.report(name, run_timestamp)
    return _unknown_op(op, _GET_OPS)


_RUNS_OPS = ("list", "logs", "diff", "compare", "aggregate")


@mcp.tool(**tool_caps(label="Proxy runs: {op}"))
def proxy_runs(
    op: str = "list",
    proxy_name: str = "",
    run_id: str = "",
    run_a: str = "",
    run_b: str = "",
    reference_name: str = "",
    tail: int = 100,
    run_ids: list[str] | None = None,
    metrics: list[str] | None = None,
) -> dict:
    """Monitor and analyze proxy run history (read-only).

    Operations (set ``op``):
      list      -> run history, newest first (proxy_name optional filter)
      logs      -> last `tail` lines of a run's stdout log (requires: run_id)
      diff      -> config + metric diff between two runs (run_a/run_b default
                   to the two newest runs)
      compare   -> field-difference norms of a completed run vs a reference
                   (requires: run_id, reference_name)
      aggregate -> comparison table across arbitrary completed runs (requires:
                   run_ids; `metrics` adds extra columns; reference_name
                   computes missing field comparisons)

    Args:
        op: The operation to perform (default 'list').
        proxy_name: For 'list': restrict to one proxy's runs.
        run_id: Run ID (e.g. 'my_proxy/20250101T120000Z') or absolute path.
        run_a, run_b: For 'diff': run IDs or absolute paths.
        reference_name: Reference dataset name for 'compare'/'aggregate'.
        tail: For 'logs': number of lines from the end (default 100).
        run_ids: For 'aggregate': list of run IDs or absolute paths.
        metrics: For 'aggregate': extra scalar metric columns.
    """
    if op == "list":
        return runs.list_runs(proxy_name)
    if op == "logs":
        return _missing_args(op, run_id=run_id) or runs.run_logs(run_id, tail)
    if op == "diff":
        return runs.runs_diff(run_a, run_b)
    if op == "compare":
        return (_missing_args(op, run_id=run_id, reference_name=reference_name)
                or runs.compare(run_id, reference_name))
    if op == "aggregate":
        return _missing_args(op, run_ids=run_ids) or runs.aggregate(run_ids, metrics, reference_name)
    return _unknown_op(op, _RUNS_OPS)


# What this server itself saw of a run, for the client's run ledger. `state` comes from
# the run dir (crashed = the process died), `feasible` from evaluating the session's
# requirements against measured metrics. Deliberately no "passed" condition: a floor
# only ever withholds credit, and a ratchet `reject` — measured fine, just no better
# than the incumbent — is the ordinary outcome of an experiment, not a failure.
_RUN_OUTCOME = {
    "id":            "run_dir",
    "crashed_when":  {"state": ["crashed"]},
    "failed_when":   {"feasible": [False]},
    # A finished run measured the source the session names — the one place in MIMIR
    # where which file a run exercised is recorded rather than guessed.
    "measured_when": {"state": ["done"]},
}

_EVAL_STATUS_OPS = ("status", "results", "log", "runs", "diff", "config")


@mcp.tool(**tool_caps(label="Proxy eval status: {op}", run_outcome=_RUN_OUTCOME))
def proxy_eval_status(
    op: str = "status",
    proxy_name: str = "",
    tail: int = 50,
    run_a: str = "",
    run_b: str = "",
) -> dict:
    """Observe the proxy optimization session (read-only).

    Poll ``proxy_eval_status()`` while a run is active; once state is 'done',
    read ``proxy_eval_status(op='results')``.  Every response includes a
    ``next_step`` field naming the exact next call.

    Operations (set ``op``):
      status  -> state ('running'|'done'|'crashed'|...), elapsed, last log lines
      results -> per-case metrics + requirements pass/fail + recommendation
      log     -> last `tail` lines of the optimization run log
      runs    -> all optimization runs for the session, newest first
      diff    -> config + metric diff between two optimization runs (run_a/
                 run_b default to the two newest)
      config  -> the current session configuration

    Args:
        op: The operation to perform (default 'status').
        proxy_name: Session to observe. Defaults to the most-recently
            initialized session.
        tail: For 'log': number of lines from the end (1-500, default 50).
        run_a, run_b: For 'diff': run IDs or absolute paths.
    """
    if op == "status":
        return eval_session.status(proxy_name)
    if op == "results":
        return eval_session.results(proxy_name)
    if op == "log":
        return eval_session.log(proxy_name, tail)
    if op == "runs":
        return eval_session.runs_list(proxy_name)
    if op == "diff":
        return eval_session.runs_diff(run_a, run_b, proxy_name)
    if op == "config":
        return eval_session.config_get(proxy_name)
    return _unknown_op(op, _EVAL_STATUS_OPS)


# ── sensitive tools (confirm=True required) ───────────────────────────────────

_MANAGE_OPS = ("register", "update", "unregister",
               "suite_define", "suite_update", "suite_delete", "scaffold")


@mcp.tool(**tool_caps(caps=[PLAN_BLOCKED], reversibility=RECOVERABLE, non_batch=True,
                      label="Proxy manage: {op}"))
def proxy_manage(
    op: str,
    name: str = "",
    executable_path: str = "",
    run_cmd_template: str = "",
    output_format: str = "",
    param_file_template: str = "",
    param_file_path: str = "",
    param_file_format: str = "",
    description: str = "",
    metadata: dict | None = None,
    cases: list[dict] | None = None,
    proxy_path: str = "",
    component_hint: str = "",
    harness_type: str = "kernel",
    output_path: str = "",
    confirm: bool = False,
) -> dict:
    """Manage proxy registrations and benchmark suite definitions (sensitive).

    Operations (set ``op``; all require confirm=True):
      register     -> register a proxy executable (requires: name,
                      executable_path, run_cmd_template)
      update       -> patch registration fields; only provided fields change
                      (requires: name)
      unregister   -> remove a proxy from the registry; run history is kept
                      (requires: name)
      suite_define -> define a benchmark suite (requires: name, cases)
      suite_update -> patch a suite definition (requires: name)
      suite_delete -> delete a suite definition; results are kept (requires: name)
      scaffold     -> generate reference + test harness files for a source
                      component (requires: proxy_path, component_hint)

    run_cmd_template placeholders: {executable}, {output_file}, {param_file},
    {extra_params}; other placeholders (e.g. {n}) come from param_overrides at
    run time.

    Suite case schema (each element of ``cases``): case_id (unique),
    proxy_name (registered), and optionally reference_name, description,
    scope ('full'|'component'), extra_params, param_sweeps (list of
    param_override dicts, one run point each; defaults to [{}]), metrics
    (extra scalar keys for the report).

    Args:
        op: The operation to perform (see list above).
        name: Proxy name ([A-Za-z0-9_-]) or suite name, depending on op.
        executable_path: Absolute path to the proxy executable or script.
        run_cmd_template: Command template (see placeholders above).
        output_format: 'npz' (register default), 'raw_float64', or 'none'.
        param_file_template: Parameter-file content template; empty if CLI-only.
        param_file_path: Fixed path for the rendered param file, or empty for per-run.
        param_file_format: 'text' (register default), 'json', 'yaml',
            'fortran_namelist', 'ini'.
        description: Free-text description of the proxy or suite.
        metadata: Optional descriptive fields for register/update: arch,
            backend, parallelism, peak_gflops_per_s,
            peak_bandwidth_gbytes_per_s, tags, version, build_cmd, source_url,
            notes, input_description, output_description, usage_examples.
        cases: For suite_define/suite_update: list of case dicts (see schema above).
        proxy_path: For 'scaffold': absolute path to the proxy source file.
        component_hint: For 'scaffold': function/subroutine/class name to target.
        harness_type: For 'scaffold': 'kernel' or 'subsystem'.
        output_path: For 'scaffold': output directory (defaults to the scaffolds cache).
        confirm: Must be True to apply any operation.
    """
    if op not in _MANAGE_OPS:
        return _unknown_op(op, _MANAGE_OPS)
    if not confirm:
        return err("Not confirmed.", hint="Set confirm=True after reviewing the parameters.")

    if op == "register":
        return (_missing_args(op, name=name, executable_path=executable_path,
                              run_cmd_template=run_cmd_template)
                or registry.register(
                    name, executable_path, run_cmd_template,
                    description=description, output_format=output_format or "npz",
                    param_file_template=param_file_template,
                    param_file_path=param_file_path,
                    param_file_format=param_file_format or "text",
                    metadata=metadata))
    if op == "update":
        return (_missing_args(op, name=name)
                or registry.update(
                    name, executable_path=executable_path,
                    run_cmd_template=run_cmd_template, description=description,
                    output_format=output_format,
                    param_file_template=param_file_template,
                    param_file_path=param_file_path,
                    param_file_format=param_file_format,
                    metadata=metadata))
    if op == "unregister":
        return _missing_args(op, name=name) or registry.unregister(name)
    if op == "suite_define":
        return (_missing_args(op, name=name, cases=cases)
                or suites.define(name, cases, description))
    if op == "suite_update":
        return _missing_args(op, name=name) or suites.update(name, cases, description)
    if op == "suite_delete":
        return _missing_args(op, name=name) or suites.delete(name)
    # op == "scaffold"
    return (_missing_args(op, proxy_path=proxy_path, component_hint=component_hint)
            or scaffold_ops.scaffold(proxy_path, component_hint, harness_type, output_path))


_EXEC_OPS = ("run", "reference", "suite", "benchmark_create", "cancel")


@mcp.tool(**tool_caps(caps=[PLAN_BLOCKED, CODE_EXEC], reversibility=RECOVERABLE, non_batch=True,
                      label="Proxy exec: {op}",
                      run_outcome={**_RUN_OUTCOME,
                                   "rows": {"field": "rows", "id": "run_dir",
                                            "failed_when_present": ["error"]}}))
def proxy_exec(
    op: str,
    proxy_name: str = "",
    reference_name: str = "",
    suite_name: str = "",
    benchmark_name: str = "",
    extra_params: str = "",
    param_overrides: dict | None = None,
    compare_to_reference: str = "",
    param_sweeps: list[dict] | None = None,
    reference_params: dict | None = None,
    extra_metrics: list[str] | None = None,
    description: str = "",
    max_output_mb: int = _DEFAULT_MAX_OUTPUT_MB,
    timeout_s: int = 0,
    per_case_timeout_s: int = 0,
    run_id: str = "",
    confirm: bool = False,
) -> dict:
    """Execute proxies locally (sensitive).

    Operations (set ``op``; all require confirm=True):
      run              -> non-blocking background run of one proxy (requires:
                          proxy_name); monitor with proxy_runs()
      reference        -> blocking run whose output is sealed as an immutable
                          reference dataset (requires: proxy_name, reference_name)
      suite            -> blocking run of every case×sweep in a suite; writes
                          the summary table (requires: suite_name)
      benchmark_create -> one blocking call = seal a reference + define a
                          one-case suite named `benchmark_name` (requires:
                          proxy_name, benchmark_name)
      cancel           -> stop a running proxy: SIGTERM->SIGKILL local, scancel
                          Slurm (requires: run_id)

    For Slurm submission of the same work, use proxy_slurm instead.

    Args:
        op: The operation to perform (see list above).
        proxy_name: Name of a registered proxy.
        reference_name: Name for the sealed reference ('reference').
        suite_name: Suite to run ('suite').
        benchmark_name: Benchmark name ('benchmark_create'); also names the
            suite and (with '_ref') the reference.
        extra_params: Additional CLI flags forwarded to the proxy.
        param_overrides: Key/value pairs patching the param file and any
            matching placeholders in run_cmd_template.
        compare_to_reference: For 'run': compute field-difference norms
            against this reference after the run.
        param_sweeps: For 'benchmark_create': list of param_override dicts,
            one run point each (defaults to [{}]).
        reference_params: For 'benchmark_create': param_overrides for the
            reference run (defaults to the first sweep entry).
        extra_metrics: Extra scalar metric keys to track ('suite'/'benchmark_create').
        description: Free-text benchmark description ('benchmark_create').
        max_output_mb: For 'reference': max field output size to store (default 512).
        timeout_s: Wall-clock budget in seconds. For 'reference'/
            'benchmark_create' it is a *ceiling*: 3600 by default and clamped
            to it, so a larger value has no effect. 7200 for 'suite'.
        per_case_timeout_s: For 'suite': per-case cap in seconds (0 = no cap
            beyond the remaining budget).
        run_id: For 'cancel': run ID or absolute path.
        confirm: Must be True to execute.
    """
    if op not in _EXEC_OPS:
        return _unknown_op(op, _EXEC_OPS)
    if not confirm:
        return err("Not confirmed.", hint="Set confirm=True to execute.")

    if op == "run":
        return (_missing_args(op, proxy_name=proxy_name)
                or runs.launch_run(proxy_name, extra_params, param_overrides,
                                   compare_to_reference))
    if op == "reference":
        return (_missing_args(op, proxy_name=proxy_name, reference_name=reference_name)
                or references.create(proxy_name, reference_name, extra_params,
                                     param_overrides, max_output_mb,
                                     timeout_s or _REF_RUN_TIMEOUT))
    if op == "suite":
        return (_missing_args(op, suite_name=suite_name)
                or suites.run_suite(suite_name, extra_metrics,
                                    timeout_s or suites._SUITE_RUN_TIMEOUT,
                                    per_case_timeout_s))
    if op == "benchmark_create":
        return (_missing_args(op, proxy_name=proxy_name, benchmark_name=benchmark_name)
                or suites.benchmark_create(proxy_name, benchmark_name, param_sweeps,
                                           reference_params, extra_metrics, description,
                                           timeout_s or _REF_RUN_TIMEOUT))
    # op == "cancel"
    return _missing_args(op, run_id=run_id) or runs.cancel(run_id)


_EVAL_OPS = ("init", "configure", "run", "stop", "reset", "reset_to_best", "end")


@mcp.tool(**tool_caps(caps=[PLAN_BLOCKED, CODE_EXEC, BACKGROUNDABLE], reversibility=RECOVERABLE, non_batch=True,
                      label="Proxy eval: {op}", run_outcome=_RUN_OUTCOME))
def proxy_eval(
    op: str,
    proxy_name: str = "",
    benchmark_name: str = "",
    requirements: list[dict] | None = None,
    proxy_source_path: str = "",
    python_executable: str = "",
    max_hours: float = 0.0,
    primary_metric: str = "time_s",
    primary_goal: str = "min",
    min_improvement: float = 0.02,
    max_stall: int = 5,
    convergence: dict | None = None,
    background: bool = False,
    confirm: bool = False,
) -> dict:
    """Drive an iterative proxy-optimization session as a monotone ratchet (sensitive).

    Typical loop: init -> run -> poll proxy_eval_status() until 'done' ->
    proxy_eval_status(op='results') -> edit the proxy source with the file
    tools -> run again.  Every response includes a ``next_step`` field.

    The session minimizes (or maximizes) ``primary_metric`` subject to the
    ``requirements`` acting as pass/fail constraints.  A completed run that
    satisfies every constraint AND improves the primary metric is *accepted* and
    becomes the new best-so-far; a run that regresses (breaks a constraint, or is
    no better) is *rejected* and ``results`` steers you to revert with
    ``reset_to_best``.  After ``max_stall`` non-improving feasible runs the session
    reports ``converged``.

    Operations (set ``op``; all require confirm=True):
      init          -> start a session; snapshots proxy_source_path as the
                       canonical baseline, never overwritten later (requires:
                       proxy_name, benchmark_name, requirements, proxy_source_path)
      configure     -> patch requirements/benchmark_name/python_executable/
                       max_hours without re-snapshotting
      run           -> launch a background eval run (non-blocking); errors if one
                       is already active. Pass background=True for a long run to
                       detach it: end your turn instead of polling; you are
                       auto-resumed with the results when it completes.
      stop          -> stop the active run (SIGTERM local / scancel Slurm)
      reset         -> restore the proxy source from the canonical baseline
      reset_to_best -> restore the proxy source from the best *accepted* run (undo
                       a regression without discarding progress)
      end           -> end the session: clear the active-session pointer (source,
                       snapshots and ledger are kept). Lifts the direct-execution
                       guard so the proxy can be run by hand again; re-init to
                       optimize further. Refuses while a run is still active.

    Args:
        op: The operation to perform (see list above).
        proxy_name: Registered proxy to optimize. For ops other than 'init',
            defaults to the most-recently initialized session.
        benchmark_name: Name of a defined benchmark suite.
        requirements: List of {metric, operator, threshold} dicts; operator is
            one of lt, gt, lte, gte, eq. Numerical invariants (l2_rel, linf_rel,
            finite, conservation_residual, convergence_order) are ordinary metrics
            here and can gate feasibility.
        proxy_source_path: Absolute path to the proxy's own source file.
        python_executable: Python interpreter for the runner (default: server's).
        max_hours: Hard deadline per run in hours (0.0 = 24 h).
        primary_metric: Scalar objective the ratchet improves (default 'time_s').
        primary_goal: 'min' or 'max' (default 'min').
        min_improvement: Relative margin a run must beat the incumbent by to count
            as an improvement (default 0.02, i.e. 2%). A non-zero default keeps
            measurement noise on a wall-clock metric from ratcheting in spurious
            "improvements"; set 0.0 only for a metric with negligible run-to-run
            variance.
        max_stall: Consecutive non-improving feasible runs before 'converged'.
        convergence: Optional {h_param, error_metric} to fit an observed order of
            accuracy across a sweep, exposed as the 'convergence_order' metric.
        background: For op='run', detach the run so a watcher tracks completion and
            auto-resumes you with the results (end your turn; do not poll).
        confirm: Must be True to apply any operation.
    """
    if op not in _EVAL_OPS:
        return _unknown_op(op, _EVAL_OPS)
    if not confirm:
        return err("Not confirmed.", hint="Set confirm=True to proceed.")

    if op == "init":
        missing = _missing_args(op, proxy_name=proxy_name, benchmark_name=benchmark_name,
                                requirements=requirements,
                                proxy_source_path=proxy_source_path)
        if missing:
            return missing
        return eval_session.init(proxy_name, benchmark_name, requirements,
                                 proxy_source_path, python_executable, max_hours,
                                 primary_metric, primary_goal, min_improvement,
                                 max_stall, convergence)
    if op == "configure":
        return eval_session.configure(proxy_name, requirements, benchmark_name,
                                      python_executable, max_hours)
    if op == "run":
        return eval_session.run(proxy_name, background=background)
    if op == "stop":
        return eval_session.stop(proxy_name)
    if op == "reset":
        return eval_session.reset(proxy_name)
    if op == "reset_to_best":
        return eval_session.reset_to_best(proxy_name)
    # op == "end"
    return eval_session.end(proxy_name)


# ── cluster tool (confirm=True required; consumes allocation hours) ───────────

_SLURM_OPS = ("run", "suite", "eval")

# op-dispatch cross-reference, for the _unknown_op redirect hint (see above).
_TOOL_BY_OPS: dict[tuple[str, ...], str] = {
    _GET_OPS: "proxy_get",
    _RUNS_OPS: "proxy_runs",
    _EVAL_STATUS_OPS: "proxy_eval_status",
    _MANAGE_OPS: "proxy_manage",
    _EXEC_OPS: "proxy_exec",
    _EVAL_OPS: "proxy_eval",
    _SLURM_OPS: "proxy_slurm",
}
def _build_op_owners() -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for ops, tool in _TOOL_BY_OPS.items():
        for op_name in ops:
            owners.setdefault(op_name, []).append(tool)
    return owners


_OP_OWNERS: dict[str, list[str]] = _build_op_owners()


@mcp.tool(**tool_caps(caps=[PLAN_BLOCKED, CLUSTER_SUBMIT, BACKGROUNDABLE], reversibility=IRREVERSIBLE, non_batch=True,
                      label="Proxy Slurm: {op}",
                      risk_note="Submits Slurm batch jobs that consume cluster allocation hours."))
def proxy_slurm(
    op: str,
    partition: str,
    proxy_name: str = "",
    suite_name: str = "",
    extra_params: str = "",
    param_overrides: dict | None = None,
    compare_to_reference: str = "",
    gpus: int = 0,
    cpus_per_task: int = 8,
    mem: str = "32G",
    wall_time: str = "04:00:00",
    account: str = "",
    job_name: str = "",
    background: bool = False,
    confirm: bool = False,
) -> dict:
    """Submit proxy work as Slurm batch jobs (sensitive, non-blocking, expensive).

    Validate the workload locally with proxy_exec first — a failed batch job
    still consumes allocation hours.

    Operations (set ``op``; all require confirm=True):
      run   -> one job for a single proxy run (requires: proxy_name); monitor
               with proxy_runs()
      suite -> one job per case×sweep point of a suite (requires: suite_name);
               aggregate afterwards with proxy_get(op='report', ...)
      eval  -> one job for an optimization-session run; monitor with
               proxy_eval_status(). Pass background=True to detach it: end your
               turn instead of polling; you are auto-resumed when the job finishes.

    Args:
        op: The operation to perform (see list above).
        partition: Slurm partition (required for every op).
        proxy_name: For 'run' (required) and 'eval' (defaults to the
            most-recently initialized session).
        suite_name: For 'suite': suite to submit.
        extra_params: For 'run': additional CLI flags forwarded to the proxy.
        param_overrides: For 'run': key/value pairs patching the param file
            and run_cmd_template placeholders.
        compare_to_reference: For 'run': reference to compare against after the run.
        gpus: GPUs per node (0 for CPU-only).
        cpus_per_task: CPU cores to allocate (default 8).
        mem: Memory per node in Slurm format (default '32G').
        wall_time: Wall-clock limit HH:MM:SS or D-HH:MM:SS (default '04:00:00').
        account: Slurm account to charge (optional).
        job_name: Slurm job name (optional; a sensible default is derived).
        background: For 'eval': detach the job — end your turn instead of
            polling; you are auto-resumed when the job finishes.
        confirm: Must be True to submit.
    """
    if op not in _SLURM_OPS:
        return _unknown_op(op, _SLURM_OPS)
    if not confirm:
        return err("Not confirmed.", hint="Set confirm=True to submit.")
    slurm_err = _validate_slurm_args(partition, gpus, cpus_per_task, mem, wall_time)
    if slurm_err:
        return err(slurm_err)

    if op == "run":
        return (_missing_args(op, proxy_name=proxy_name)
                or slurm.submit_run(proxy_name, partition, extra_params,
                                    param_overrides, compare_to_reference,
                                    gpus, cpus_per_task, mem, wall_time,
                                    account, job_name))
    if op == "suite":
        return (_missing_args(op, suite_name=suite_name)
                or slurm.submit_suite(suite_name, partition, gpus, cpus_per_task,
                                      mem, wall_time, account))
    # op == "eval"
    return slurm.submit_eval(partition, proxy_name, background, gpus, cpus_per_task,
                             mem, wall_time, account, job_name or "proxy_opt")


if __name__ == "__main__":
    mcp.run()
