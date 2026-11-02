"""The numerical invariants the proxy treats as reserved metric names.

Values a run prints under them are discarded, so code cannot satisfy its own
acceptance constraints. The client does not read this module.
"""

from __future__ import annotations


# Invariants that express *correctness*: each is a comparison against something
# the code under test does not itself define. ``finite`` is the weakest (it only
# rules out NaN/Inf) and is deliberately included — it is still an assertion
# about the solution rather than about the process exiting.
NUMERICAL_INVARIANT_METRICS = frozenset({
    "finite",
    "conservation_residual",
    "convergence_order",
    "l2_abs", "l2_rel",
    "linf_abs", "linf_rel",
})

# ``wall_time_s`` is the server's own wall-clock measurement — the tamper-proof twin of
# a self-reported ``time_s`` — and ``peak_rss_mb`` / ``peak_rss_total_mb`` its memory
# high-water marks (the largest process's; every process's summed, as MPI counts ranks),
# with ``processes``, how many there were. Reserved,
# but measurements rather than correctness invariants, hence absent from
# NUMERICAL_INVARIANT_METRICS: a run that reports only a cost has proved nothing about
# the answer.
RESERVED_METRICS = NUMERICAL_INVARIANT_METRICS | {"wall_time_s", "peak_rss_mb", "peak_rss_total_mb", "processes"}
