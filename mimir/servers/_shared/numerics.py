"""Numerical correctness invariants — the shared vocabulary of "this was proved".

The **proxy optimization server** treats these names as *reserved*: values the code
under optimization prints for them are discarded before evaluation, so a solver can
never satisfy its own acceptance constraints. (Observed in the wild: an agent-edited
proxy printing ``conservation_residual=<its own drift>`` to pass a requirement whose
sealed reference was missing.)

The **client** does not read this module at all. It once treated a printed ``l2_rel=…``
as evidence and raised the validation tier for it; that rewarded a string, since the
value can never be interpreted from outside the process — the very reason the proxy
seals references server-side. What a run printed is the model's to read and report.

Nothing client-only belongs here, and in particular not the ``check=fail`` verdict
grammar: that lives beside the other ways a run reports on itself, in
``client.guardrails.runner_output``, which owns the question. Keeping it here is what
drags this module across the client/server line, and it hides two failures while it does
— a grammar written to mirror the metrics parser's strictness rather than what harnesses
actually print, with no test exercising it against a real one. What belongs here is the
proxy's reserved vocabulary, which is all this module is for.

Lives in ``_shared`` for the flat ``sys.path`` import the servers use
(``from numerics import RESERVED_METRICS``).
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
# a self-reported ``time_s``. Reserved, but a *timing* invariant rather than a
# correctness one, hence absent from NUMERICAL_INVARIANT_METRICS: a run that reports
# only a duration has proved nothing about the answer.
RESERVED_METRICS = NUMERICAL_INVARIANT_METRICS | {"wall_time_s"}
