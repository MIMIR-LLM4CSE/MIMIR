"""Numerical correctness invariants — the shared vocabulary of "this was proved".

The **proxy optimization server** treats these names as *reserved*: values the code
under optimization prints for them are discarded before evaluation, so a solver can
never satisfy its own acceptance constraints. (Observed in the wild: an agent-edited
proxy printing ``conservation_residual=<its own drift>`` to pass a requirement whose
sealed reference was missing.)

The **client** does not read them at all. It once treated a printed ``l2_rel=…`` as
evidence and raised the validation tier for it; that rewarded a string, since the value
can never be interpreted from outside the process — the very reason the proxy seals
references server-side. What a run printed is the model's to read and report.

What the client still reads here is :func:`observed_failure_verdict`, which only ever
withholds credit.

Lives in ``_shared`` because both a server and the client import it — flat via
``sys.path`` from the servers, as ``servers._shared.numerics`` from the client.
"""

from __future__ import annotations

import re

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
# a self-reported ``time_s``. Reserved, but a *timing* invariant rather than a
# correctness one, hence absent from NUMERICAL_INVARIANT_METRICS: a run that reports
# only a duration has proved nothing about the answer.
RESERVED_METRICS = NUMERICAL_INVARIANT_METRICS | {"wall_time_s"}

# ``key=value`` alone on a line, strict single-token RHS. Whole-line by construction
# (``fullmatch``) so prose mentioning a key, verbose logs and compiler output do not
# register. Mirrors the proxy metrics parser's strict fallback pattern.
_KEY_VALUE_LINE_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>\S+)"
)

# The verdict a run declares *about itself*, for a check that computes its own
# pass/fail instead of letting an assertion raise. Client-side reading only (the
# proxy has no use for it), but it shares the line grammar above so there is one
# way to report a machine-readable result, not two.
VERDICT_KEYS = frozenset({"check", "verdict"})
_FAILING_VERDICTS = frozenset({"fail", "failed", "failure", "error", "red", "false", "no"})


def observed_failure_verdict(text: str) -> bool:
    """True when a run's own output declares that one of its checks did not pass.

    The counterweight to an exit code: a script that evaluates its own criteria,
    prints that they were not met and then returns 0 anyway is indistinguishable
    from a clean run to everything downstream. Reading the verdict *only* in this
    direction is deliberate — a ``check=fail`` line demotes a green exit, a
    ``check=pass`` line never rescues a red one, so declaring a verdict can cost
    credit but can never buy it.

    Same strict whole-line ``key=value`` grammar the proxy uses for metrics, so prose
    ("check failed to converge"), logs and compiler output do not register. Any one
    failing verdict is enough, however many checks the run reported.
    """
    if not text:
        return False
    for line in text.splitlines():
        m = _KEY_VALUE_LINE_RE.fullmatch(line.strip())
        if m and m.group("key") in VERDICT_KEYS and m.group("value").lower() in _FAILING_VERDICTS:
            return True
    return False
