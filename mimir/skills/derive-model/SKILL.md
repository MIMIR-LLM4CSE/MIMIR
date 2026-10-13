---
name: derive-model
description: Derive or verify the mathematics before implementing it — truncation error and order, stability, closed forms, non-dimensionalization — with the symbolic tool rather than from memory.
disable-model-invocation: false
---

A derivation recalled is a derivation asserted. The `symbolic` tool does the algebra
here, and what it returns is the result; what you remember about the Taylor expansion
of a stencil or the amplification factor of a scheme is a plausible-looking version of
it, and the error survives into the code, into the tests written to match the code, and
into the paper.

The available operations are `simplify`, `expand`, `factor`, `differentiate`,
`integrate`, `solve_equation`, `compute_limit`, `series_expansion`, `create_matrix`,
`matrix_determinant`, `solve_system`. Everything below is composed from those. There is
no eigenvalue operation — the route to spectra is the characteristic polynomial, see
stability.

## Consistency and order of accuracy

The order of a scheme is a statement about its truncation error, so compute the error
rather than quoting the order.

1. Write the scheme's residual with the exact solution substituted, each shifted term
   as its expansion: `series_expansion` of `u(x + h)` about `h = 0` to `n` terms, with
   `n` at least two beyond the order you expect — truncate too early and the leading
   error term is the one you cut.
2. Sum the terms with `expand`, then `simplify`. The lowest-order surviving power of
   `h` is the order; its coefficient is the error constant, and it is worth keeping —
   two second-order schemes with a factor of 12 between their constants are not
   interchangeable.
3. A residual whose `h^0` term does not vanish is an inconsistent scheme, not a
   low-order one. Stop and find the sign or index error before going further.

The order you derive is the order a refinement sweep must then measure. They are two
different claims — one about the formula, one about the implementation — and a code can
be a correct implementation of a wrong derivation, or the reverse, and pass whichever
check you ran.

## Stability

For a linear scheme, substitute the Fourier ansatz, collect the amplification factor
`g` with `simplify`, and get the condition from `|g| <= 1`: `solve_equation` on the
boundary case gives the limit, `compute_limit` on the high-wavenumber end tells you
whether it is stable in the mesh-resolution limit at all.

For a system or a multi-step scheme, build the amplification matrix with
`create_matrix`, form `A - lambda*I` as a matrix literal, take `matrix_determinant` of
it, and `solve_equation` the resulting characteristic polynomial for `lambda`. The
spectral radius over the roots is what the stability condition is about.

A stability limit is a number the implementation must then respect. Derive it, then
check what the code actually uses as its step — a scheme stable under `dt <= h^2/2`
running at `h^2` is the single most common reason a correct-looking solver blows up.

## Closed forms, and verifying them

`integrate` and `solve_equation` return an answer, not a guarantee that it answers your
question. Verify in the direction the tool did not go:

- Differentiate an integral back and `simplify` the difference against the integrand —
  zero, or you have the wrong antiderivative branch.
- Substitute a solution back into its own equation and `simplify` to zero.
- Pin one numeric point with `evaluate` on both sides. A symbolic identity that fails
  at `x = 0.3` is not an identity.
- Check the limits you know: `compute_limit` at `0`, `inf`, or at the parameter value
  where the physics degenerates to a case you can state independently.

## Non-dimensionalization and conventions

Before implementing, fix and write down: the scaling and the dimensionless groups, the
sign convention of every flux and source term, the time convention (forward or
backward, step index at which the state is valid), the index convention (cell-centred
or node-centred, 0- or 1-based, which direction is fastest in memory). The derivation
is where these are decided; the code is where they are silently assumed. Record them in
a comment beside the formula they govern and in the docstring of the routine that
implements them — an assumed convention runs, passes every check, and is wrong.

## Handing the derivation to the code

- Write the mathematics as LaTeX in your answer and in any `.md` you touch: `$...$`
  inline, `$$...$$` on its own lines. Markdown eats `\( \)` and `\[ \]`.
- A derived closed form is exactly what a numerical test needs as its independent
  reference — a manufactured solution, an analytic limit, the error constant a
  convergence sweep should recover. Carry it to the test rather than letting the test
  assert against the code's own output.
- Where the derivation ends in a result a user has to accept — a chosen scaling, a
  neglected term, a linearization — say which it is and what it costs. A neglected term
  is a modelling decision, not an algebraic step.
