"""Did this run reach a good end? — the one place that decides.

Four things can say a run failed, and until this module existed each was consulted
separately by the caller, in a growing ``or`` chain, with no shared account of what
counts as evidence:

* the **exit status** — trustworthy only when it is actually the run's (see below);
* a **declared verdict** the run printed about itself (``check=fail``), for a script
  that computes its own criteria and returns 0 regardless;
* a **test runner's summary** (``1 failed, 5 passed``, ``FAILED path::test``);
* a run *server's* own report, which is judged elsewhere because it never goes
  through a shell at all.

The chain grew that way because each dialect was discovered as a missed failure and
bolted on after the fact. :func:`judge_run` owns the question instead, and the
dialects below are readers feeding it. Adding a fifth is a reader plus a row in its
test table, not another clause at a call site.

**Why the exit status needs qualifying.** A chained command reports the status of its
LAST segment, and every idiom the model actually reaches for puts something else
there::

    pytest tests/test_x.py -q 2>&1 | tail -15     # the status is tail's
    timeout 280 python3 test_x.py; echo "EXIT=$?" # the status is echo's

Across four recorded sessions of one task, every single test invocation was written
one of those two ways. All of them therefore reported ``returncode: 0`` — including
the runs with four red tests — and each was recorded as a run that had reached its end
and merely owed a reading. Nothing downstream could tell those from real passes, so
the repair ladder never engaged and the model was free to declare them passing, which
it did. The readers below are what close that: when the status cannot be trusted, the
run's own report is the only evidence there is, so it must be read.

Attribution itself is a separate question with a separate answer
(:func:`exit_status_settles`) — an unattributable green is not a failure, it is an
absence of evidence, and only the shortcut that settles a run without anyone reading it
may be denied on those grounds.

**Demote-only, throughout.** Every reader here can cost a run credit and none can buy
it: a red report demotes a green exit, and no report ever rescues a red one. Declaring
a verdict must never be a way to pass.
"""

from __future__ import annotations

import re

# ── Dialect 1: the verdict a run declares about itself ─────────────────────────
# For a check that computes its own pass/fail instead of letting an assertion raise.
# The base prompt steers self-checking scripts towards printing one of these.
VERDICT_KEYS = frozenset({"check", "verdict"})
_FAILING_VERDICTS = frozenset({"fail", "failed", "failure", "error", "red", "false", "no"})
# Keys that report a COUNT of failures rather than a verdict (``checks_failed=3``).
_FAILURE_COUNT_KEYS = frozenset({"failed", "failures"})

# ``key=value`` at the start of a line, value a single token, remainder optional.
#
# The remainder is what a strict whole-line ``fullmatch`` used to forbid, and it cost a
# whole recorded session: a harness printing ``check=fail (2 failed: convergence_order_4,
# absorption)`` declared its failure in exactly the documented grammar and was read as
# silent, because of the parenthetical. Prose is still excluded by construction — the
# ``=`` must sit against the key, so "the check failed to converge" does not match.
_KEY_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>\S+)(?:\s.*)?"
)


def _verdict_key(key: str) -> str:
    """The trailing component of a possibly namespaced key: ``abc_check`` → ``check``.

    A harness checking several things names them (``abc_check=fail``,
    ``dalembert_check=pass``) rather than printing three bare ``check=`` lines it would
    have to reconcile. Reading only the bare key made every such run look silent — the
    second half of the same recorded session that the parenthetical cost.
    """
    return key.rsplit("_", 1)[-1].lower()


def observed_failure_verdict(text: str) -> bool:
    """True when a run's own output declares that one of its checks did not pass.

    The counterweight to an exit code: a script that evaluates its own criteria, prints
    that they were not met and then returns 0 anyway is indistinguishable from a clean
    run to everything downstream.

    Any one failing verdict is enough, however many checks the run reported. A failure
    *count* counts too, and only when non-zero — ``checks_failed=0`` is a pass stated
    the long way round.
    """
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
# unittest's trailer ``FAILED (failures=1, errors=2)`` matches the same anchor.
_RUNNER_FAILURE_LINE_RE = re.compile(r"^(?:FAILED|ERROR)\b\s*[\s(]\S")

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
    """Which tests this run reported failing, as one comparable string.

    Sorted and de-duplicated so two runs that failed the same tests compare equal
    whatever order the runner listed them in and whatever ``| tail -N`` kept. Empty when
    no node id could be read — the caller treats that as "no signature", never as "the
    same signature as last time", because an unreadable report is not evidence that the
    failure stayed put.
    """
    nodes = {
        m.group("node")
        for m in (_NODE_ID_RE.match(l.strip()) for l in text.splitlines())
        if m
    }
    return " ".join(sorted(nodes))


# ── The adjudicator ────────────────────────────────────────────────────────────

# Why a run is not being credited. Carried on the run record so the completion report
# can say what settled it rather than leaving the reader to guess.
DECLARED_FAILURE = "the run's own output declared a failing check"
TEST_RUNNER_FAILURE = "the test runner's own summary reported failures"


def judge_run(*, exit_ok: bool, output: str) -> tuple[bool, str]:
    """Did this run reach a good end? Returns ``(completed, reason)``.

    ``exit_ok`` is what the machine reported and ``output`` whatever the command printed
    (stdout and stderr together — ``2>&1`` puts the report in whichever one the caller
    merged into). ``reason`` is empty for a run that completed, and names what settled
    it otherwise.

    A red exit is final: it is the one judgement the machine makes in a direction that
    can be trusted, and no report may talk it back up. The readers can only demote.

    Note what this deliberately does NOT do: it never turns a green into a failure for
    being *unattributable*. A ``pytest … | tail`` that genuinely passed reports the same
    unattributable green as one that failed silently, and charging the repair budget for
    the first in order to catch the second would punish every clean run written that
    way. Not knowing is not the same as failing; what the missing attribution must block
    is *settling*, which is :func:`exit_status_settles` below.
    """
    if not exit_ok:
        return False, ""
    if observed_failure_verdict(output):
        return False, DECLARED_FAILURE
    if observed_test_failure(output):
        return False, TEST_RUNNER_FAILURE
    return True, ""


def exit_status_settles(exit_is_the_run_s: bool) -> bool:
    """May a green exit status settle this run on its own, with nobody reading it?

    Only one kind of run is allowed that shortcut: a build, whose exit code *is* the
    finding because it produced artefacts rather than results. The shortcut assumes the
    status belongs to the thing that built, and ``make 2>&1 | tail -20`` breaks that
    assumption silently — the status is the pager's, and the build was auto-passed
    without anyone establishing it had succeeded.

    Trivial as a function; named as one because the assumption is what matters and it
    was previously nowhere written down.
    """
    return exit_is_the_run_s
