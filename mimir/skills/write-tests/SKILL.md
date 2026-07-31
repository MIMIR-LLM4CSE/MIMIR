---
name: write-tests
description: Add focused and meaningful tests for existing code.
disable-model-invocation: false
---

You are writing tests.

Rules:
- Test behavior, not implementation details.
- Focus on critical paths and edge cases.
- Use existing test frameworks and patterns.
- Keep tests readable and deterministic.

When the test covers computed results (numerical, scientific, or algorithmic code):
- A test that only asserts no-NaN/no-Inf, a magnitude bound, or "it did not crash"
  proves the code runs, not that it is right. It passes for any output that is
  merely not catastrophic — including a completely wrong answer.
- Assert against something independent of the code under test: an analytic or
  manufactured solution, a coarse reference implementation, an invariant that must
  hold, or the observed order of accuracy from a refinement sweep.
- Assert the property that defines the requirement, not a weaker proxy. Ask what a
  plausibly broken implementation would do: if it passes your test too, the test
  measures something other than the requirement.
- Print the measured quantity as `key=value` on its own line (`l2_rel=3.2e-4`,
  `convergence_order=3.98`, `conservation_residual=1.1e-12`) so the result is
  recorded rather than merely asserted.

Workflow:
1. Identify what should be tested.
2. Explain the testing strategy.
3. Write tests that fail before the fix (if applicable) — if a test cannot fail,
   it is not testing anything. Check that it actually fails against a deliberately
   wrong implementation before trusting a pass.
4. Ensure tests pass after.

Do NOT refactor production code unless required for testability.
