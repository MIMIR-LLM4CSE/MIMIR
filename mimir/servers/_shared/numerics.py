"""Numerical correctness invariants — the shared vocabulary of "this was proved".

One name set, two very different consumers:

- **The proxy optimization server** treats these as *reserved*: values the code
  under optimization prints for them are discarded before evaluation, so a
  solver can never satisfy its own acceptance constraints. (Observed in the
  wild: an agent-edited proxy printing ``conservation_residual=<its own drift>``
  to pass a requirement whose sealed reference was missing.)
- **The client's validation observer** treats them as *evidence*: a validation
  command whose output reports one of these has compared the artifact against
  something independent of itself — an analytic or manufactured solution, a
  reference field, a conserved quantity, a refinement sweep. That is the
  difference between "the code ran" and "the code is right", and it is the only
  such signal available without reading a test's assertions.

The two readings do not conflict: the proxy distrusts the *value* (the number
could be forged), the observer only trusts the *presence* of the key (that a
comparison was performed at all, which is what raises the validation tier).
Neither ever credits a numeric claim made in prose.

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

# ``wall_time_s`` is the server's own wall-clock measurement of the solver
# process — the tamper-proof twin of a self-reported ``time_s``. It is reserved
# (the proxy strips proxy-printed values) but it is a *timing* invariant, not a
# correctness one, so it is not in NUMERICAL_INVARIANT_METRICS: a run that only
# reports a duration has proved nothing about the answer.
RESERVED_METRICS = NUMERICAL_INVARIANT_METRICS | {"wall_time_s"}

# ``key=value`` on a line of its own, with a strict single-token RHS. Anchored
# and whole-line by construction (``fullmatch``) so prose that merely mentions a
# metric name, verbose logs, and compiler output do not register. Mirrors the
# strict fallback pattern the proxy metrics parser already uses for the same
# reason.
_INVARIANT_LINE_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>\S+)"
)

# A value that is actually a number (or a bool, for ``finite``). A key whose RHS
# is not one of these was not a measurement — e.g. ``l2_rel=<computed below>``.
_BOOL_VALUES = frozenset({"true", "false", "1", "0", "yes", "no"})


def _is_measurement(value: str) -> bool:
    if value.lower() in _BOOL_VALUES:
        return True
    try:
        float(value)
    except ValueError:
        return False
    return True


def observed_invariant_metrics(text: str) -> set[str]:
    """Return the correctness invariants *reported* by a command's output.

    Scans for ``key=value`` lines whose key is in
    :data:`NUMERICAL_INVARIANT_METRICS` and whose value parses as a number or a
    boolean. Presence is the signal; the value is never interpreted, compared,
    or trusted — a forged number and an honest one are equally evidence that a
    comparison was *attempted*, and distinguishing them is impossible from
    outside the process (which is exactly why the proxy seals references
    server-side instead).

    Empty set for empty/None input, so callers need no guard.
    """
    if not text:
        return set()
    found: set[str] = set()
    for line in text.splitlines():
        m = _INVARIANT_LINE_RE.fullmatch(line.strip())
        if not m:
            continue
        key = m.group("key")
        if key in NUMERICAL_INVARIANT_METRICS and _is_measurement(m.group("value")):
            found.add(key)
    return found
