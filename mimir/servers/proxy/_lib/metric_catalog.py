"""The metrics a session can optimise or constrain: one catalogue the agent, the ratchet and
the window read alike.

Each metric says what it measures (label, unit), its natural direction, where its value comes
from and whether two runs of the same code give the same value. A session adds the metrics its
program reports (``declared`` in its config).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .numerics import NUMERICAL_INVARIANT_METRICS


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    unit: str
    goal: str            # "min" or "max"
    source: str          # "server", "numerics", "program"
    noisy: bool          # two runs of the same code differ: compared beyond a margin
    needs: str = ""      # what it takes to exist: "reference", "conserved", "convergence"


_SERVER = (
    Metric("time_s", "Time", "s", "min", "server", True),
    Metric("wall_time_s", "Wall time", "s", "min", "server", True),
    Metric("best_time_s", "Best case time", "s", "min", "server", True),
    Metric("total_time_s", "Total time", "s", "min", "server", True),
    # A process's peak moves by a few hundred kilobytes between two runs of the same code.
    Metric("peak_rss_mb", "Peak memory (largest process)", "MB", "min", "server", True),
    Metric("peak_rss_total_mb", "Peak memory (all processes)", "MB", "min", "server", True),
)
_NUMERICS = (
    Metric("l2_rel", "Relative L2 error to the sealed reference", "", "min", "numerics", False, "reference"),
    Metric("linf_rel", "Relative max error to the sealed reference", "", "min", "numerics", False, "reference"),
    Metric("l2_abs", "L2 error to the sealed reference", "", "min", "numerics", False, "reference"),
    Metric("linf_abs", "Max error to the sealed reference", "", "min", "numerics", False, "reference"),
    Metric("finite", "No NaN or Inf in the output", "", "max", "numerics", False, "reference"),
    Metric("conservation_residual", "Conservation residual", "", "min", "numerics", False, "conserved"),
    Metric("convergence_order", "Observed order of convergence", "", "max", "numerics", False, "convergence"),
)
# Reported by the program under these usual names: timings and throughputs move between runs.
_PROGRAM = (
    Metric("elapsed_s", "Elapsed time (reported)", "s", "min", "program", True),
    Metric("runtime_s", "Runtime (reported)", "s", "min", "program", True),
    Metric("gflops_per_s", "Throughput", "GFLOP/s", "max", "program", True),
    Metric("bandwidth_gbytes_per_s", "Bandwidth", "GB/s", "max", "program", True),
)
CATALOG: dict[str, Metric] = {m.name: m for m in (*_SERVER, *_NUMERICS, *_PROGRAM)}
assert NUMERICAL_INVARIANT_METRICS <= set(CATALOG)

NOISY_METRICS = tuple(name for name, m in CATALOG.items() if m.noisy)


def declared(cfg: dict) -> dict[str, Metric]:
    """The metrics the session's program reports, as its config declares them."""
    out = {}
    for item in cfg.get("declared") or []:
        name = str(item.get("metric") or "")
        if name and name not in CATALOG:
            out[name] = Metric(name, str(item.get("label") or name), str(item.get("unit") or ""),
                               "max" if item.get("goal") == "max" else "min",
                               "numerics" if item.get("source") == "numerics" else "program",
                               bool(item.get("noisy", False)))
    return out


def lookup(cfg: dict, name: str) -> Metric | None:
    return CATALOG.get(name) or declared(cfg).get(name)


def is_noisy(cfg: dict, name: str) -> bool:
    metric = lookup(cfg, name)
    return bool(metric and metric.noisy)


def missing(cfg: dict, metric: Metric, suite: dict | None = None, entry: dict | None = None) -> str:
    """Why *metric* cannot have a value in this session, "" when it can."""
    if metric.needs == "reference":
        cases = (suite or {}).get("cases") or []
        if not cases or any(not c.get("reference_name") for c in cases):
            return ("needs a sealed reference on every case (proxy_exec op='benchmark_create'); "
                    "the session has none")
    if metric.needs == "conserved" and not (entry or {}).get("conserved_metric"):
        return "needs the proxy's conserved_metric, which its registration does not name"
    if metric.needs == "convergence" and not (cfg.get("convergence") or {}).get("h_param"):
        return "needs a resolution sweep: init with convergence={'h_param': ...}"
    return ""


def of_session(cfg: dict, suite: dict | None = None, entry: dict | None = None) -> list[dict]:
    """Every metric this session knows, each with whether it is available and why not."""
    rows = []
    for metric in (*CATALOG.values(), *declared(cfg).values()):
        why = missing(cfg, metric, suite, entry)
        rows.append({**asdict(metric), "available": not why, **({"why": why} if why else {})})
    return rows
