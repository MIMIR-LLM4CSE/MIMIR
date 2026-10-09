---
name: write-tests
description: Write tests that can actually fail — asserting against something independent of the code, not against what it currently prints.
disable-model-invocation: false
---

A test that cannot fail is not a test. Everything here serves that one check: before
trusting a green result, know what would have made it red.

## What to assert against

Test behaviour, not implementation detail: a test pinned to how the code is currently
written fails on every refactor and passes through every change of meaning, which is
the wrong way round. Cover the critical paths and the edge cases — empty, single
element, boundary, the value that divides by zero — rather than one more variation of
the path that already works.

Use the frameworks and patterns the project already has. A second test idiom in a
repository is a second thing to learn before anyone can add a test to it.

## Computed results: the trap

For numerical, scientific or algorithmic code, the usual assertions prove the code runs
and nothing more.

- A test that only asserts no-NaN/no-Inf, a magnitude bound, or "it did not crash"
  passes for any output that is merely not catastrophic — a completely wrong answer
  included.
- Assert against something **independent of the code under test**: an analytic or
  manufactured solution, a coarse reference implementation, an invariant that must hold,
  or the observed order of accuracy from a refinement sweep. The code's own current
  output is not a reference; locking it in records the behaviour, including the bug.
- Assert the property that defines the requirement, not a weaker proxy. Ask what a
  plausibly broken implementation would do: if it passes your test too, the test
  measures something other than the requirement.
- Print the measured quantity as `key=value` on its own line (`l2_rel=3.2e-4`,
  `convergence_order=3.98`, `conservation_residual=1.1e-12`) so the result is recorded
  rather than merely asserted.

## Order of work

1. Identify what should be tested, and say what would falsify it.
2. State the strategy — what each test holds fixed and what it varies.
3. Write the test so that it **fails** on the defect or on a deliberately wrong
   implementation. Run it and watch it fail before the fix; a test that has only ever
   been green has established nothing.
4. Fix, then watch the same test pass. The failing-then-passing pair is the evidence,
   and `report_verdict` is where you say what the output showed.

Keep tests deterministic: fix every seed, do not depend on wall-clock timing, iteration
order, or a path outside the repository. A flaky test is worse than no test — it trains
everyone to re-run until green.

## What does not count

Widening a tolerance until the assertion passes, asserting the current output as the
expected one, or testing a fragment that reimplements what the project actually
imports — a stub proves things about the stub. Do not refactor production code to make
it pass; refactoring it to make it *testable* is legitimate, and worth saying out loud
when you do.
