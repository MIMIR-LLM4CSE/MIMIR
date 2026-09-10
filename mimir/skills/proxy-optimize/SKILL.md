---
name: proxy-optimize
description: Iteratively optimize a scientific computing proxy to meet performance and accuracy requirements using the proxy tools.
disable-model-invocation: false
---

You are running a proxy optimization **ratchet**.

The session **minimizes (or maximizes) a primary metric subject to the
requirements as pass/fail constraints**. A completed run that satisfies every
requirement *and* improves the primary metric is **accepted** and becomes the new
best-so-far; a run that regresses (breaks a constraint, or is no better than the
best) is **rejected**. Every eval response includes a `verdict` and a `next_step`
field naming the exact next call — follow it.

## Setup (first time only)

1. Confirm a proxy is registered: `proxy_get(op='proxies')`.
   - If not: `proxy_manage(op='register', ...)` with the executable path and run command template.
2. Confirm a benchmark suite exists: `proxy_get(op='suites')` or `proxy_get(op='suite', name=...)`.
   - If not: create one in a single call with `proxy_exec(op='benchmark_create', ...)`,
     or define it manually with `proxy_manage(op='suite_define', ...)`.
3. Initialize the optimization session:
   ```
   proxy_eval(
       op="init",
       proxy_name=...,
       benchmark_name=...,
       requirements=[
           {"metric": "l2_rel",  "operator": "lt",  "threshold": 1e-3},  # accuracy constraint
           {"metric": "finite",  "operator": "eq",  "threshold": 1},     # no NaN/Inf
       ],
       proxy_source_path="<workspace>/proxy_bench/harnesses/<name>.py",  # NEVER edited
       optimize_paths=["/abs/path/to/pkg/solver.py",  # the real code the harness imports
                       "/abs/path/to/pkg/kernel.py"], # several files are fine
       primary_metric="time_s",   # scalar objective the ratchet improves
       primary_goal="min",        # "min" or "max"
       min_improvement=0.02,      # FLOOR for the accept margin — the measured noise wins if it is larger
       max_stall=5,               # non-improving feasible runs before "converged"
       repeat=0,                  # measurements per case; 0 = 3 for a timing metric, 1 for a reproducible one
       confirm=True,
   )
   ```
   This snapshots the `optimize_paths` **tree** as the baseline.

   **Measure the baseline, then read the margin the reply gives you.** Each case is
   run `repeat` times and reduced to its **median**, never its best — the minimum of
   several draws improves with the number of draws whatever the code does. The spread
   across the baseline's own replicates is the noise floor of this machine, and the
   margin a run must clear is that floor or `min_improvement`, whichever is larger.
   `min_improvement` alone cannot guard noise it has not measured: on a 192-core shared
   node it stood at 2% against a real spread of 3.1%, and a kernel rewrite worth nothing
   was accepted on a 2.8% "gain" that four later runs of the same code could not
   reproduce. Every run reports `min_improvement`, `margin_source` and `primary_spread`
   — if the spread is close to your gains, raise `repeat` before believing them.

   **Write the harness under `proxy_bench/harnesses/`.** Everything else the proxy owns
   already lives in `<workspace>/proxy_bench/` — the registry, the sealed references, the
   runs, the optimisation state, the snapshot repository, and the harnesses
   `proxy_manage(op='scaffold')` generates. A harness written by hand belongs with them,
   and putting it anywhere else leaves a project with two directories for one activity:
   observed as a `benchmarks/` beside a `proxy_bench/`, with nothing to say which held
   what. `proxy_manage(op='clean')` removes runs, optimisation state and snapshots — it
   does not touch harnesses, so they survive a reset like the references do. Keeping the
   harness in the repo instead is a legitimate choice when it is a deliverable the user
   maintains; drifting there by accident is not.

   **The harness is not the subject.** `proxy_source_path` runs the code and prints
   metrics; `optimize_paths` is the code, which the harness should **import**. Writing a
   self-contained script that reproduces the code you meant to optimize means the
   accuracy constraints hold for the copy and say nothing about what ships — `init`
   refuses that shape rather than warning about it.

4. **Measure the baseline first.** Run once with `optimize_paths` untouched, before any
   edit. Until that run is on record the ratchet refuses to accept anything: with no
   measurement of the original, every later number is an assertion, not a comparison.

### Numerical invariants (correctness gates)

The proxy server computes these vs the sealed reference and exposes them as
ordinary metrics — use them as `requirements` so a fast-but-wrong run can never be
accepted:

- `l2_rel`, `linf_rel` — relative error norms of the output field vs reference.
- `finite` — 1 if the output field has no NaN/Inf, else 0.
- `conservation_residual` — relative discrepancy of a conserved scalar (requires the
  proxy to register a `conserved_metric` and emit it).
- `convergence_order` — observed order of accuracy fitted across a resolution sweep
  (pass `convergence={"h_param": "<sweep param>", "error_metric": "l2_rel"}` to `init`).
- `wall_time_s` — server-measured wall time of the solver process. `time_s` stays
  proxy-reported (a kernel may exclude startup/IO), but a claim exceeding the
  measured wall time is discarded (`time_s_ignored`) and replaced by it, and an
  accepted `time_s` improvement whose wall time regressed raises a
  `timing_warning` — never silence it by editing the timer; fix the code or
  optimize `primary_metric='wall_time_s'`.

These metric names are **reserved**: only the server computes them, and any value
the proxy itself prints for them is discarded (`reserved_metrics_ignored` lists the
drops). If a requirement targets one and it is reported missing, fix the setup —
seal a reference (`benchmark_create`) or declare `conserved_metric` — do NOT try to
emit the metric from the proxy; `init` refuses configurations that cannot satisfy
these requirements.

## Optimization loop

1. **Run**: `proxy_eval(op='run', confirm=True)`. The call **waits** for the run and
   answers with the verdict and the per-case results, so there is nothing to poll —
   do not call `proxy_eval_status()` around it.
   - For a run you expect to be long, add `background=True` to skip the wait. If the
     result says the run is being watched, **end your turn** on it: you are resumed
     with the results when it completes, leaving you and the user free meanwhile, and
     you must NOT poll it. Say that only when the result says it — a promise of a
     resume is the client's to make, never yours to assume.
   - A run still going when the wait budget expires detaches itself the same way:
     the response says so, and the rule is again to end your turn.
   - If the run crashed: `proxy_eval_status(op='log', tail=100)` to diagnose.
2. **Inspect** the payload the run returned — `verdict`, `best`, `stall`,
   `recommendation` — and follow `next_step`. (`proxy_eval_status(op='results')`
   re-reads it later if you need it again):
   - **accept** → new best. Read the source and try a further improvement, then run again.
   - **reject** → this edit regressed. `proxy_eval(op='reset_to_best', confirm=True)`,
     then try a *different* edit and run.
   - **converged** → `proxy_eval(op='reset_to_best', confirm=True)`, then summarize.
   - no verdict yet (constraints not met, no best) → edit the source and run again.
3. Editing: read the files in `optimize_paths`, understand what is slow or
   inaccurate, apply a targeted `replace_in_file(...)` to one of them, then go to
   step 1. Never edit the harness at `proxy_source_path` — it is the measuring
   instrument, not the subject.
4. **Compare runs**: `proxy_eval_status(op='diff')` or `(op='runs')` (note `is_best`).

## Rules

- Read the file you are about to change before every modification. Never edit blind.
- Make one focused change per run cycle — do not batch multiple unrelated edits.
- Trust the `verdict`: never keep an edit the ratchet rejected — reset_to_best first.
- `reset_to_best` reverts every `optimize_paths` file to the best accepted run (keeps
  progress); `reset` reverts them all to the baseline taken at init. Both restore the
  whole set or nothing: a per-file revert would assemble a combination that was never
  measured together, which can run and still mean nothing.
- Use `proxy_eval(op='configure', ...)` to change requirements or benchmark without re-initializing.
- The loop is not finished until `results` returns a `verdict` (accept/converged).
  Never declare success from `status` log tails alone — after any interruption or
  tool timeout, the detached run may still have completed: check `status`, then
  ALWAYS fetch `results` to ratify the outcome before summarizing.
- Do NOT use `bash_run` to execute the proxy directly — always go through
  `proxy_eval(op='run')`. This is **enforced**: while a session is active, a call that runs
  the proxy source/executable directly is blocked (read-only inspection like reading the file
  is fine). Executing it by hand bypasses reference sealing, the invariants and the ratchet,
  so a hand-run can never be a valid result. End the session with `proxy_eval(op='end', confirm=True)`
  to lift the guard and run the proxy directly again.
- Do NOT modify `_proxy_runner.py` — it is a stable orchestrator. Modify only the files
  listed in `optimize_paths`.

## When to stop

Stop and summarize when the session reports `verdict="converged"` (constraints met
and the primary metric stopped improving), after restoring the best with
`proxy_eval(op='reset_to_best', confirm=True)`. If no feasible run was ever found,
report what was tried and what remains. When you are fully done with the proxy,
`proxy_eval(op='end', confirm=True)` closes the session (code, snapshots and ledger are
kept) and lifts the direct-execution guard. To discard a proxy's runs, optimization state
and snapshots entirely — starting over rather than continuing — use
`proxy_manage(op='clean', name=..., confirm=True)`; it reports what it left behind
(sealed references, suites, the registry entry) and how to remove those too.
