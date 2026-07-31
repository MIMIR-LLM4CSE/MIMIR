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
       proxy_source_path="/abs/path/to/proxy_source.py",
       primary_metric="time_s",   # scalar objective the ratchet improves
       primary_goal="min",        # "min" or "max"
       min_improvement=0.02,      # relative margin required to count as an improvement (2%; guards timing noise)
       max_stall=5,               # non-improving feasible runs before "converged"
       confirm=True,
   )
   ```
   This takes a one-time canonical snapshot. It is never overwritten automatically.

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

1. **Run**: `proxy_eval(op='run', confirm=True)`. For a run you expect to be long,
   add `background=True`: **end your turn** afterward instead of polling — you are
   automatically resumed with the results when it completes, and you (and the user)
   stay free to do other work meanwhile. Do NOT poll a backgrounded run.
2. **Wait** (non-background runs only): `proxy_eval_status()` until state is 'done'
   or 'crashed'.
   - If 'crashed': `proxy_eval_status(op='log', tail=100)` to diagnose.
3. **Inspect**: `proxy_eval_status(op='results')` — read `verdict`, `best`, `stall`,
   `recommendation`, and follow `next_step`:
   - **accept** → new best. Read the source and try a further improvement, then run again.
   - **reject** → this edit regressed. `proxy_eval(op='reset_to_best', confirm=True)`,
     then try a *different* edit and run.
   - **converged** → `proxy_eval(op='reset_to_best', confirm=True)`, then summarize.
   - no verdict yet (constraints not met, no best) → edit the source and run again.
4. Editing: `read_file(path=<proxy_source_path>)`, understand what is slow or
   inaccurate, apply a targeted `replace_in_file(...)`, then go to step 1.
5. **Compare runs**: `proxy_eval_status(op='diff')` or `(op='runs')` (note `is_best`).

## Rules

- Read the proxy source before every modification. Never edit blind.
- Make one focused change per run cycle — do not batch multiple unrelated edits.
- Trust the `verdict`: never keep an edit the ratchet rejected — reset_to_best first.
- `reset_to_best` reverts to the best accepted run (keeps progress); `reset` reverts
  all the way to the original canonical baseline.
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
- Do NOT modify `_proxy_runner.py` — it is a stable orchestrator. Modify only the proxy source file.

## When to stop

Stop and summarize when the session reports `verdict="converged"` (constraints met
and the primary metric stopped improving), after restoring the best with
`proxy_eval(op='reset_to_best', confirm=True)`. If no feasible run was ever found,
report what was tried and what remains. When you are fully done with the proxy,
`proxy_eval(op='end', confirm=True)` closes the session (source/snapshots/ledger are
kept) and lifts the direct-execution guard.
