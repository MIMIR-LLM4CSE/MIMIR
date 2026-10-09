---
name: debug-numerics
description: Diagnose a computation that runs but is wrong — NaN, divergence, a plausible answer that is off, or a convergence rate that is not the expected one.
disable-model-invocation: false
---

The code does not crash. It produces numbers, and the numbers are wrong. Nothing in a
stack trace or a test name points at the cause, because the failure is in what the
program computes rather than in what it does.

## Make it small, deterministic, and serial first

A bug observed at 10^6 cells over 10^4 steps is unreadable. The same bug at four cells
over one step is a printout you can check by hand. Before diagnosing anything: reduce to
the smallest grid or input that still exhibits it, take one step rather than the full
run, fix every seed, and run serially — a wrong answer that only appears on 32 threads
is a race, which is a different investigation and the thread count is the clue. If
shrinking makes the symptom disappear, that is itself the finding: the bug scales with
something, and which something narrows it immediately.

## Read the mode of failure — it names the suspect

- **NaN or Inf at the very first step** — bad input, an uninitialized field, a division
  by a zero that is structurally zero (an empty cell, a dry layer, a vanishing
  denominator at a symmetry axis), or a `sqrt` of a quantity that went slightly
  negative at roundoff.
- **Finite values growing without bound over steps** — stability. Compare the step
  actually used against the scheme's limit; check the sign of every diffusion and
  damping term, where a flipped sign amplifies exactly what it was meant to dissipate.
- **Settles to a finite but wrong value** — boundary conditions or a missing source
  term. The interior scheme is converging; it is converging to the wrong problem.
- **Right shape, wrong magnitude by a clean factor** — units, a missing normalization,
  a factor of 2 from a double-counted face, a degree/radian confusion.
- **Right at the interior, wrong near the edges** — boundary stencil order, ghost-cell
  fill, a loop bound off by one.
- **Changes when the thread count or the decomposition changes** — a race or a halo
  exchange, not the kernel.

## Bisect the pipeline by substituting what you know

The point is to split "the formulation is wrong" from "the implementation of the
formulation is wrong", because those have disjoint fixes.

1. Feed in a solution you know exactly and take **zero** steps. If the output already
   differs from the input, the defect is in I/O, initialization, or the boundary fill —
   the kernel has not run yet.
2. Take **one** step from that exact state. The residual should be the truncation error
   at this resolution — small and of the derived order — not O(1). An O(1) residual
   after one step localizes the bug to the stencil or its coefficients.
3. Refine and look at the **slope**, not the single error. An error that does not drop
   under refinement is a formulation or boundary defect; an error that drops at the
   wrong rate is an order defect — a boundary treated one order lower than the
   interior, a limiter active where it should not be.
4. Check the invariants that must hold regardless of accuracy: a conserved quantity's
   drift, a symmetry the problem has that the output should too, positivity of a
   quantity that cannot be negative. An invariant violated at step one is a sharper
   pointer than an error norm at step one thousand.

## Never assume what a symbol denotes

Units, sign, time level, indexing, memory layout, which formulation of the model is
implemented — read the definition, the documentation, or the caller that fixes it. If
nothing does, ask. This is the one shortcut that survives every automated check: an
assumed convention runs, passes, and is wrong, and in a debugging session you are
reading unfamiliar code precisely where the conventions matter most.

## Proving the fix, and what does not count as proof

The evidence that a numerical bug is fixed is a check seen **failing before and passing
after**. Construct that check before the fix, not after: an assertion against an
analytic or manufactured solution, an invariant, or the observed order from a
refinement sweep. A check that was only ever green proves nothing happened.

Then `report_verdict(verdict=..., reason=...)` naming the number you read — `fail` with
the quantity that was wrong, `pass` with the quantity that now is, `unknown` when the
output genuinely does not settle it. Exit 0 is not a result, and "the tests pass" and
"the answer is right" are different claims.

What is not a fix: widening a tolerance until the assertion passes, clamping a NaN to
zero, adding damping to suppress a growth whose cause you did not find, or special-casing
the input that exposed it. Each of those removes the symptom and keeps the defect, and
the next person to see it will have no failing check to start from. If the real fix is
out of scope, say that plainly instead — a diagnosed bug left unfixed is a useful
result; a hidden one is not.
