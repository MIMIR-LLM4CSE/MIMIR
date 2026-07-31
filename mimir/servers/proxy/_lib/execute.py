"""Proxy execution: synchronous case runs, reference sealing, detached finalize.

The three entry points every launch path funnels through:

* ``_run_benchmark_case`` — one synchronous case run (suite runs + the
  optimization runner), applying the reserved-metrics purge, the ``time_s``
  plausibility guard, and the non-zero-exit gate.
* ``_seal_reference``    — blocking run whose output becomes an immutable
  reference dataset.
* ``_post_run_finalize`` — settles a *detached* run (local wrapper or Slurm
  postrun.py) into metrics.json with the same invariants.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone

from responses import err

from _lib import command, metrics as metrics_mod, procs, report, store

_DEFAULT_MAX_OUTPUT_MB = 512
_REF_RUN_TIMEOUT       = 3600         # seconds


def _compare_to_reference(
    run_output_path: str | None,
    reference_name: str,
    output_format: str,
) -> dict:
    ref_output = store._ref_output_path(reference_name)
    if not ref_output:
        return {"error": f"Reference '{reference_name}' has no stored output field."}
    if not run_output_path or not os.path.isfile(run_output_path):
        return {"error": "Run did not produce an output_file — field comparison skipped."}
    a = metrics_mod._load_field(ref_output, output_format)
    b = metrics_mod._load_field(run_output_path, output_format)
    if a is None:
        return {"error": f"Could not load reference field from {ref_output}."}
    if b is None:
        return {"error": f"Could not load run output field from {run_output_path}."}
    return metrics_mod._field_norms(a, b)


# ── synchronous case run (shared by suite runs and the optimization runner) ───

def _run_benchmark_case(
    entry: dict,
    proxy_name: str,
    reference_name: str,
    extra_params: str,
    param_overrides: dict | None,
    extra_metrics: list[str],
    deadline: float,
    tag_suffix: str = "",
    per_case_timeout_s: float | None = None,
) -> tuple[str, dict]:
    """Run one proxy case synchronously and return (run_dir, row_dict).

    ``deadline`` is a monotonic timestamp; the run is capped at the remaining
    budget (deadline - now).  ``per_case_timeout_s`` further caps a single case
    when set (None = no per-case cap beyond the remaining budget).  On timeout
    or error the row contains an ``"error"`` key instead of metrics.
    """
    srd = store._proxy_runs_dir(proxy_name)
    os.makedirs(srd, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + (f"_{tag_suffix}" if tag_suffix else "")
    run_dir = os.path.join(srd, tag)
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "proxy_name":            proxy_name,
        "extra_params":         extra_params,
        "param_overrides":      param_overrides or {},
        "output_format":        entry.get("output_format", "npz"),
        "compare_to_reference": reference_name,
        "started_at":           datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    _remaining = deadline - time.monotonic()
    if _remaining <= 0:
        return run_dir, {"error": "aggregate timeout"}

    try:
        argv = command._build_run_cmd(entry, run_dir, extra_params, param_overrides)
    except Exception as exc:
        return run_dir, {"error": f"cmd build failed: {exc}"}

    cap = _remaining if per_case_timeout_s is None else min(float(per_case_timeout_s), _remaining)
    log_file = procs._log_path(run_dir)
    t0 = time.time()
    try:
        with open(log_file, "w") as log_fh:
            proc = subprocess.run(
                argv, stdout=log_fh, stderr=subprocess.STDOUT,
                timeout=cap,
            )
    except subprocess.TimeoutExpired:
        return run_dir, {"error": f"timed out after {cap:.0f}s"}
    except (FileNotFoundError, OSError) as exc:
        return run_dir, {"error": str(exc)}

    elapsed = round(time.time() - t0, 2)
    # A solver that exits non-zero fails the case even if it printed a metrics
    # block first (same rule as _seal_reference).  No metrics.json is written,
    # so the run dir reads as 'crashed' — which is what happened.
    if proc.returncode != 0:
        return run_dir, {"error": f"solver exited with code {proc.returncode}",
                         "returncode": proc.returncode}

    stdout = procs._read_log(run_dir)

    run_metrics = metrics_mod._parse_metrics_block(stdout)
    forged = metrics_mod._strip_reserved_metrics(run_metrics)
    if forged:
        run_metrics["reserved_metrics_ignored"] = forged
    metrics_mod._normalize_time_metrics(run_metrics, elapsed)
    run_metrics["returncode"] = proc.returncode

    proxy_out_key = run_metrics.pop("output_file", None)
    ext = command._output_ext(entry.get("output_format", "npz"))
    run_out = os.path.join(run_dir, f"output.{ext}")
    if proxy_out_key and os.path.isfile(str(proxy_out_key)):
        try:
            shutil.copy2(str(proxy_out_key), run_out)
        except OSError:
            pass

    output_format = entry.get("output_format", "npz")
    comparison: dict = {}
    if reference_name:
        comparison = _compare_to_reference(run_out, reference_name, output_format)
        run_metrics["comparison_to_reference"] = {"reference": reference_name, **comparison}
        # Lift error norms to top-level metrics so they are usable as requirements
        # (e.g. {"metric": "l2_rel", "operator": "lt", "threshold": 1e-3}).
        for k in ("l2_abs", "l2_rel", "linf_abs", "linf_rel"):
            if isinstance(comparison.get(k), (int, float)):
                run_metrics[k] = comparison[k]

    # Numerical invariants (server-side): finiteness of the output field, and an
    # optional conserved-quantity residual vs the reference metrics.  Both surface
    # as ordinary metrics so they can gate feasibility via requirements.
    field = metrics_mod._load_field(run_out, output_format)
    if field is not None:
        run_metrics["finite"] = 1 if metrics_mod._finite_check(field) else 0
    conserved_key = entry.get("conserved_metric")
    if conserved_key and reference_name:
        residual = metrics_mod._conservation_residual(
            run_metrics, store._load_ref_metrics(reference_name) or {}, conserved_key)
        if residual is not None:
            run_metrics["conservation_residual"] = residual

    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(run_metrics, fh, indent=2)
    procs._update_active_link(proxy_name, run_dir)

    row = report._row_with_extra(run_metrics, comparison, extra_metrics, entry, run_dir)
    return run_dir, row


# ── reference sealing ─────────────────────────────────────────────────────────

def _seal_reference(
    entry: dict,
    reference_name: str,
    *,
    extra_params: str = "",
    param_overrides: dict | None = None,
    max_output_mb: int = _DEFAULT_MAX_OUTPUT_MB,
    timeout_s: int = _REF_RUN_TIMEOUT,
) -> tuple[dict | None, dict | None]:
    """Run a proxy synchronously and store its output as an immutable reference.

    Blocking.  Returns ``(result, None)`` on success or ``(None, err_response)``
    on failure, where ``err_response`` is a ready-to-return ``err()`` dict
    (preserving hints and log tails).  ``result`` contains::

        {reference_name, ref_dir, metrics, output_stored, elapsed_s}

    The proxy ``entry`` is the registry record (must include name, executable,
    run_cmd_template, output_format).  This is the single source of truth used by
    ``proxy_exec(op='reference')`` and ``proxy_exec(op='benchmark_create')``.
    """
    proxy_name = entry.get("name", "")
    timeout_s = max(10, min(timeout_s, _REF_RUN_TIMEOUT))
    max_out_bytes = max_output_mb * 1024 * 1024

    rd = store._ref_dir(reference_name)
    os.makedirs(rd, exist_ok=True)

    os.makedirs(store.runs_dir(), exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tmp_run_dir = os.path.join(store._proxy_runs_dir(proxy_name), f"ref_{tag}")
    os.makedirs(tmp_run_dir, exist_ok=True)

    config = {
        "proxy_name":     proxy_name,
        "extra_params":   extra_params,
        "output_format":  entry.get("output_format", "npz"),
        "reference_name": reference_name,
        "created_at":     datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(rd, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    try:
        argv = command._build_run_cmd(entry, tmp_run_dir, extra_params, param_overrides)
    except Exception as exc:
        return None, err(f"Failed to expand run_cmd_template: {exc}")

    log_file = procs._log_path(tmp_run_dir)
    t0 = time.time()
    try:
        with open(log_file, "w") as log_fh:
            proc = subprocess.run(argv, stdout=log_fh, stderr=subprocess.STDOUT,
                                  timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None, err(f"Solver timed out after {timeout_s}s.",
                         hint="Increase timeout_s or use a smaller problem size "
                              "for the reference run.")
    except (FileNotFoundError, OSError) as exc:
        return None, err(f"Failed to launch proxy: {exc}")

    elapsed = round(time.time() - t0, 2)
    stdout = procs._read_log(tmp_run_dir)

    if proc.returncode != 0:
        return None, err(f"Solver exited with code {proc.returncode}.",
                         hint=f"Check stdout.log at {log_file}. Last output:\n"
                              + stdout[-500:])

    ref_metrics = metrics_mod._parse_metrics_block(stdout)
    forged = metrics_mod._strip_reserved_metrics(ref_metrics)
    if forged:
        ref_metrics["reserved_metrics_ignored"] = forged
    metrics_mod._normalize_time_metrics(ref_metrics, elapsed)
    ref_metrics["returncode"] = proc.returncode
    ref_metrics["proxy_name"] = proxy_name

    output_stored: str | None = None
    proxy_out = ref_metrics.pop("output_file", None)
    if proxy_out and os.path.isfile(str(proxy_out)):
        size = os.path.getsize(str(proxy_out))
        if size > max_out_bytes:
            ref_metrics["output_file_warning"] = (
                f"output_file ({size / 1024 / 1024:.1f} MB) exceeds "
                f"max_output_mb={max_output_mb}. Field not stored."
            )
        else:
            ext = command._output_ext(entry.get("output_format", "npz"))
            dest = os.path.join(rd, f"output.{ext}")
            try:
                shutil.copy2(str(proxy_out), dest)
                output_stored = dest
                ref_metrics["output_size_mb"] = round(size / 1024 / 1024, 3)
            except OSError as exc:
                ref_metrics["output_file_warning"] = f"Could not copy output file: {exc}"

    with open(os.path.join(rd, "metrics.json"), "w") as fh:
        json.dump(ref_metrics, fh, indent=2)

    return {
        "reference_name": reference_name,
        "ref_dir":        rd,
        "metrics":        ref_metrics,
        "output_stored":  output_stored,
        "elapsed_s":      elapsed,
    }, None


# ── detached-run finalize ─────────────────────────────────────────────────────

def _post_run_finalize(
    run_dir: str,
    proxy_name: str,
    compare_to_reference: str,
    output_format: str,
    wall_s: float | None = None,
    returncode: int | None = None,
) -> None:
    """Parse stdout.log and write metrics.json for a detached run.

    Called by the local-run wrapper (which passes ``wall_s``/``returncode``
    directly) and by Slurm postrun.py (which leaves them to be read from the
    ``wall_time_ms``/``returncode`` files the batch script wrote).  A non-zero
    exit skips metrics.json so the run reads as 'crashed', matching the
    synchronous paths.
    """
    if returncode is None:
        returncode = procs._read_int_file(os.path.join(run_dir, "returncode"))
    if returncode is not None and returncode != 0:
        return
    if wall_s is None:
        wall_ms = procs._read_int_file(os.path.join(run_dir, "wall_time_ms"))
        if wall_ms is not None:
            wall_s = round(wall_ms / 1000.0, 3)

    stdout = procs._read_log(run_dir)
    run_metrics = metrics_mod._parse_metrics_block(stdout)
    forged = metrics_mod._strip_reserved_metrics(run_metrics)
    if forged:
        run_metrics["reserved_metrics_ignored"] = forged
    if wall_s is not None:
        metrics_mod._normalize_time_metrics(run_metrics, wall_s)
    if returncode is not None:
        run_metrics["returncode"] = returncode
    if compare_to_reference:
        output_ext = command._output_ext(output_format)
        run_out    = os.path.join(run_dir, f"output.{output_ext}")
        comparison = _compare_to_reference(run_out, compare_to_reference, output_format)
        run_metrics["comparison_to_reference"] = {
            "reference": compare_to_reference,
            **comparison,
        }
    mp = os.path.join(run_dir, "metrics.json")
    try:
        with open(mp, "w") as fh:
            json.dump(run_metrics, fh, indent=2)
    except OSError:
        pass
