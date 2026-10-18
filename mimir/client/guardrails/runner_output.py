"""Did this run reach a good end? The one place that decides (:func:`judge_run`).

Evidence: the exit status (only when it is the run's own: :func:`exit_status_settles`),
a declared verdict (``check=fail``), a test runner's summary. Demote-only: a report can
cost a run credit, never buy it.
"""

from __future__ import annotations

import re

# ── Dialect 1: the verdict a run declares about itself ─────────────────────────
# For a check that computes its own pass/fail instead of letting an assertion raise.
# The base prompt steers self-checking scripts towards printing one of these.
VERDICT_KEYS = frozenset({"check", "verdict"})
_FAILING_VERDICTS = frozenset({"fail", "failed", "failure", "error", "red", "false", "no"})
# Keys that report a count of failures (``checks_failed=3``). ``exit`` too: a trailing
# ``echo "suite_exit=$?"`` turns the shell's status green while printing the red one.
_FAILURE_COUNT_KEYS = frozenset({"failed", "failures", "exit"})

# ``key=value`` at the start of a line, value a single token, remainder optional.
#
# The remainder is allowed: ``check=fail (2 failed: convergence_order_4, absorption)``
# declares its failure in the documented grammar. Prose is excluded by construction —
# the ``=`` must sit against the key, so "the check failed to converge" does not match.
_KEY_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>\S+)(?:\s.*)?"
)


def _verdict_key(key: str) -> str:
    """The trailing component of a possibly namespaced key: ``abc_check`` → ``check``.

    A harness checking several things names them (``abc_check=fail``,
    ``dalembert_check=pass``) rather than printing three bare ``check=`` lines it would
    have to reconcile; reading only the bare key would make every such run look silent.
    """
    return key.rsplit("_", 1)[-1].lower()


def observed_failure_verdict(text: str) -> bool:
    """True when a run's output declares a failing check, or a non-zero failure count."""
    if not text:
        return False
    for raw in text.splitlines():
        m = _KEY_VALUE_RE.fullmatch(raw.strip())
        if not m:
            continue
        key, value = _verdict_key(m.group("key")), m.group("value")
        if key in VERDICT_KEYS and value.lower() in _FAILING_VERDICTS:
            return True
        if key in _FAILURE_COUNT_KEYS:
            try:
                if float(value) != 0:
                    return True
            except ValueError:
                pass
    return False


# ── Dialect 2: what a test runner printed about itself ─────────────────────────
# pytest's short-summary rows (``FAILED path::test - reason``, ``ERROR path::test``);
# unittest's trailer ``FAILED (failures=1, errors=2)`` matches the same anchor. A colon
# is a separator too: a harness printing ``FAILED: convergence_order`` said exactly what
# the other two say, and requiring whitespace or ``(`` read it as silent. A separator is
# still required, so ``FAILED_COUNT`` and prose starting with the word do not match.
_RUNNER_FAILURE_LINE_RE = re.compile(r"^(?:FAILED|ERROR)\b\s*[\s(:]\s*\S")

# pytest's terminal counts line, decorated or bare, with or without the timing tail:
# ``=== 4 failed, 2 passed in 1.52s ===``, ``1 failed, 5 passed in 0.96s``. Whole-line
# by construction so a sentence containing "2 failed" does not register.
_COUNTS_LINE_RE = re.compile(
    r"\d+\s+[a-z]+(?:\s*,\s*\d+\s+[a-z]+)*(?:\s+in\s+[\d.]+\s*s)?"
)
_NONZERO_BAD_COUNT_RE = re.compile(r"\b[1-9]\d*\s+(?:failed|failures?|error|errors)\b")

# ``FAILED tests/test_x.py::test_name`` → ``tests/test_x.py::test_name``. The node id is
# pytest's own identifier for the test and so the stable half of the row; the
# ``- AssertionError: …`` tail is clipped by output limits and would make two reports of
# one failure look different.
_NODE_ID_RE = re.compile(r"^(?:FAILED|ERROR)\s+(?P<node>[^\s:]+(?:::\S+)?)")


def observed_test_failure(text: str) -> bool:
    """True when a test runner's own summary reports that something failed.

    False says nothing: a run with no recognisable summary (a crash before collection,
    a runner nobody here knows) is left to the other readers.
    """
    if not text:
        return False
    for raw in text.splitlines():
        line = raw.strip()
        if _RUNNER_FAILURE_LINE_RE.match(line):
            return True
        core = line.strip("=").strip()
        if _COUNTS_LINE_RE.fullmatch(core) and _NONZERO_BAD_COUNT_RE.search(core):
            return True
    return False


def test_failure_signature(text: str) -> str:
    """The failing tests a run reported, sorted and de-duplicated; "" when none can be read."""
    nodes = {
        m.group("node")
        for m in (_NODE_ID_RE.match(line.strip()) for line in text.splitlines())
        if m
    }
    return " ".join(sorted(nodes))


# ── The adjudicator ────────────────────────────────────────────────────────────

# Why a run is not being credited. Carried on the run record so the completion report
# can say what settled it rather than leaving the reader to guess.
DECLARED_FAILURE = "the run's own output declared a failing check"
TEST_RUNNER_FAILURE = "the test runner's own summary reported failures"


def judge_run(*, exit_ok: bool, output: str) -> tuple[bool, str]:
    """``(completed, reason)`` for a run, from its exit and its merged output.

    A red exit is final; the readers can only demote a green one. An unattributable green
    is not a failure: what it may not do is settle (:func:`exit_status_settles`).
    """
    if not exit_ok:
        return False, ""
    if observed_failure_verdict(output):
        return False, DECLARED_FAILURE
    if observed_test_failure(output):
        return False, TEST_RUNNER_FAILURE
    return True, ""


def exit_status_settles(exit_is_the_run_s: bool) -> bool:
    """Whether a green exit may settle a build unread: only when the status is the build's own."""
    return exit_is_the_run_s
