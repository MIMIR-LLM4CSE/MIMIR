---
name: proxy-optimize
description: Use whenever the goal is to make code faster or leaner — run time, memory, throughput, a speedup, a kernel, a profile to act on — even when the user does not say "proxy": sets up the proxy and its benchmark if none exists, then keeps a change only when a measured run improved the metric and broke no constraint.
disable-model-invocation: false
---

An optimization without a ratchet keeps whatever was measured last, and what was
measured last is usually noise. This is the ratchet: nothing is kept that was not shown
to be better than the best already on record.

The session **minimizes (or maximizes) a primary metric subject to the
requirements as pass/fail constraints**. A completed run that satisfies every
requirement *and* improves the primary metric is **accepted** and becomes the new
best-so-far; a run that regresses (breaks a constraint, or is no better than the
best) is **rejected**. Every eval response includes a `verdict` and a `next_step`
field naming the exact next call — follow it.

The ratchet judges a **version**, never an idea. A first version that is slower can hide a
right idea written the wrong way: 32-bit indexing that still recomputes every address is
slower, while one base pointer with constant offsets is 14 % faster. A rejected run's
reply says which kind of rejection it is, and the loop below says what each asks for.

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
       optimize_paths=["/abs/path/to/pkg/solver.py",  # the real code the harness runs
                       "/abs/path/to/pkg/kernel.py"], # several files are fine
       primary_metric="time_s",   # scalar objective the ratchet improves
       primary_goal="min",        # "min" or "max"
       min_improvement=0.02,      # FLOOR for the accept margin — the measured noise wins if it is larger
       max_stall=5,               # non-improving feasible runs before "stalled" (then "converged")
       repeat=0,                  # measurements per case; 0 = 3 for a timing metric, 1 for a reproducible one
       confirm=True,
   )
   ```
   This snapshots the `optimize_paths` **tree** as the baseline.

   **Several objectives, relative constraints, a bound: `spec=`.** In place of
   `primary_metric`/`primary_goal`/`requirements`, a `spec` says it all at once:

   ```
   spec={
       "objectives": [{"metric": "time_s"},                          # the primary
                      {"metric": "peak_rss_mb", "tolerance": 0.1}],  # may lose 10 %
       "constraints": [{"metric": "l2_rel", "operator": "lte",
                        "of_baseline": 10, "precision_floor": True}],
       "directives": ["keep the public API"],
       "bound": 0.003, "bound_basis": "3 fields x 4 B x 512^3 per step / 2000 GB/s",
   }
   ```

   `proxy_eval_status(op='metrics')` lists the metrics, what each needs, and whether
   anything checks the solution. **Write fidelity as a constraint, never as a rule to
   keep the original's order of operations.** `of_baseline` bounds an error by a factor
   of the baseline's own value, and `precision_floor` lets it reach what the field's type
   allows (1e-5 in float32). A reordered sum, or a constant folded into a weight (w·c/DX),
   changes the last bits and is not wrong. A rule like "copy the reference's order"
   forbids exactly the edits that pay. A run better on one objective and worse on another
   past its tolerance is a **tradeoff**: judge it with `report_verdict(run=<run id>,
   verdict='pass' or 'rejected', reason=...)`, and ask the user when it is close.

   **Work out the bound before the first edit.** Count the bytes each step has to move
   and the flops it has to do, divide by the machine's bandwidth and peak, and declare the
   result (the spec's `bound` and `bound_basis`), or register the peaks
   (`peak_gflops_per_s`, `peak_bandwidth_gbytes_per_s`) and have the harness print
   `flops` and `bytes_moved`. Two codes at 21 ms per step against a 3 ms memory floor
   are not memory-bound, however plausible "it needs TMA" sounds. The bound says which
   resource to work on, and the ratchet uses it to tell a stall from a convergence.

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
   runs, the optimization state, the snapshot repository, and the harnesses
   `proxy_manage(op='scaffold')` generates. A harness written by hand belongs with them,
   and putting it anywhere else leaves a project with two directories for one activity:
   observed as a `benchmarks/` beside a `proxy_bench/`, with nothing to say which held
   what. `proxy_manage(op='clean')` removes runs, optimization state and snapshots — it
   does not touch harnesses, so they survive a reset like the references do. Keeping the
   harness in the repo instead is a legitimate choice when it is a deliverable the user
   maintains; drifting there by accident is not.

   **A compiled project works the same way.** Nothing in the loop is
   Python-specific. List the sources in `optimize_paths` (`.cpp`, `.h`, `.f90`)
   and declare the build at registration:

   ```
   proxy_manage(op='register', name=..., executable_path='/abs/build/solver',
                run_cmd_template='{executable} {param_file}',
                metadata={"build_cmd": "make -C /abs/build solver"},
                confirm=True)
   ```

   The server runs `build_cmd` **once per evaluation run**, before any case is
   measured — so the binary measured is always the one the current sources
   produce, and a `reset_to_best` cannot leave you timing the rejected attempt's
   binary. A build that fails ends the run with no verdict and nothing recorded:
   read `build_log_tail` in the reply, fix it, run again.

   **Do not rebuild between edits.** The rhythm is unchanged: edit as many files
   as the change needs, then call `proxy_eval(op='run')` once. Build time sits
   outside the measurement budget and is paid once per run whatever `repeat` is,
   and incrementality comes from `make`/`ninja` — the long first build is paid
   once, an edit to one file costs that file. Name a target in `build_cmd`
   (`make -C build bench_solver`) to narrow it further. `build_cmd` is argv, not
   a shell line: a sequence or a `module load` belongs in a wrapper script.
   Registration accepts an `executable_path` that does not exist yet when a
   build is declared — the build is what produces it. Seal references *after* a
   build: a reference is immutable, so one sealed from a stale binary is wrong
   for good.

   **The harness is not the subject.** `proxy_source_path` runs the code and prints
   metrics; `optimize_paths` is the code, which the harness should **use** — import,
   link or load it, whatever the language. Writing a
   self-contained script that reproduces the code you meant to optimize means the
   accuracy constraints hold for the copy and say nothing about what ships — `init`
   refuses that shape rather than warning about it.

   **Mark the measured region.** The harness starts, makes or reads inputs, checks the
   answer: none of that is the code, and unmarked it counts in the time and the memory.
   Bracket only the call to the code, in the harness's language, with the helper copied
   beside it from `mimir/servers/proxy/region/` (`proxy_manage(op='scaffold')` does both
   for Python):
   - Python: `from mimir_region import region` … `with region(): out = solve(data)`
   - C / C++: `#include "mimir_region.h"` … `mimir_region_begin(); solve(); mimir_region_end();`
   - Fortran: compile `mimir_region.f90` with the harness, `use mimir_region` …
     `call mimir_region_begin()` … `call mimir_region_end()`
   - Any other language: write `begin` then `end` lines to the FIFO named by
     `MIMIR_REGION`, each followed by reading one byte from `$MIMIR_REGION.ack`.
   Outside MIMIR they do nothing. Several regions in turn (repetitions inside the
   harness) add their times. On a GPU, each mark first waits for the device (CUDA or
   HIP found among the process's libraries, from C, C++ or Python: CuPy, PyTorch,
   Numba, a binding; CUDA Fortran and OpenACC through their compilers), so the region
   holds the kernels it launched; under CUDA it is also Nsight's range, and the GPU
   memory its processes hold is `region_peak_device_mb`. A Fortran harness calling CUDA C
   through a binding calls `cudaDeviceSynchronize` itself before each mark. Do it before the baseline: the harness never changes
   during a session, and a `harness_counted` note in `results` says when it was not done.

4. **Measure the baseline first.** Run once with `optimize_paths` untouched, before any
   edit. Until that run is on record the ratchet refuses to accept anything: with no
   measurement of the original, every later number is an assertion, not a comparison.
   A run that gave no value of the metric is no baseline: fix the harness and run again,
   or, when the original itself cannot run, fix it and `proxy_eval(op='rebaseline', confirm=True)`.

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
- `wall_time_s` — server-measured wall time of the solver process.
- `region_time_s`, `regions` — server-measured time inside the harness's marked
  regions (summed). When the harness marks any, `time_s` **is** `region_time_s` and a
  `time_s` the harness prints is set aside (`time_s_ignored`). Without a region,
  `time_s` stays proxy-reported, but a claim exceeding the measured wall time is
  discarded (`time_s_ignored`) and replaced by it, and an accepted `time_s` improvement
  whose wall time regressed raises a `timing_warning` — never silence it by editing the
  timer; fix the code or optimize `primary_metric='wall_time_s'`.
- `region_peak_rss_mb`, `region_peak_rss_total_mb`, `region_extra_mb` — the peak memory
  inside the regions (largest process, all summed) and how much the region added to
  what the harness already held: the code's own memory. A memory requirement or
  objective is best put on these when the harness marks a region; `memory_warning`
  compares them first.
- `peak_rss_mb` and `peak_rss_total_mb` — server-measured peak memory: of the case's
  largest process, and of all its processes summed, each at its own peak, over
  `processes` of them. For MPI, the total is the ranks' memory: on one node from the
  processes the run starts, across nodes from Slurm's accounting (`sacct`) for ranks
  `srun` started. The run's is its largest case's. Use either as a ceiling
  (`{"metric": "peak_rss_total_mb", "operator": "lt", "threshold": 64000}`) or as the
  objective. An accepted run whose memory grew by more than a quarter (the total when
  both runs have it) raises a `memory_warning`: say so to the user, who may want a
  ceiling.

These metric names are **reserved**: only the server computes them, and any value
the proxy itself prints for them is discarded (`reserved_metrics_ignored` lists the
drops). If a requirement targets one and it is reported missing, fix the setup —
seal a reference (`benchmark_create`) or declare `conserved_metric` — do NOT try to
emit the metric from the proxy; `init` refuses configurations that cannot satisfy
these requirements.

## Optimizing for the compute nodes

When MIMIR runs on a login node and the code is meant for the compute nodes, the
whole session runs there: `proxy_cluster(op='eval', partition=..., constraint=...)`
instead of `proxy_eval(op='run')`. The job builds, runs the cases and times them on
the node. A baseline measured on the login node describes the login node.

- **Pick the node kind first.** Use the partition the user named, or the one the code
  calls for; ask when it is not clear. `slurm_node_profile(partition=...)` shows each
  kind of node, with its CPU model, SIMD and `-march` once a node of that kind has
  been probed (`slurm_probe_node`).
- **Stay on one kind of node.** Pass `constraint` or `nodelist` when the partition
  mixes several. `exclusive` is on by default for `eval`.
- **The session is pinned to the baseline's machine.** Each run records where it ran.
  A timing taken on another kind of machine gets the verdict `incomparable`: best and
  stall do not move. To move the session to the node, restore the original
  (`proxy_eval(op='reset')`), then `proxy_eval(op='rebaseline')`, and run on the node.
- **Inside an allocation**, MIMIR already runs on the node: `proxy_eval(op='run')` is
  the node run.
- **Under LSF** (`execution_context.scheduler` is `lsf`), `proxy_cluster` submits with
  `bsub` instead of `sbatch`. The arguments are the same: `partition` is the queue.

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
   - `where_time_goes` — the baseline and every accepted run are profiled after their
     verdict: the workspace files that take the time, and the share spent outside the
     workspace. What only the harness does (and the libraries it calls for that) is
     `harness`, a share of the whole run left out of the others: never act on it.
     `window: measured region` says the profiler recorded the marked region alone. The
     tool fits what runs: py-spy (with native frames) or cProfile for Python, perf for
     compiled code and for Python calling it, callgrind where perf is blocked (it counts
     instructions, `unit: instructions`, not time), rank 0 under an MPI launcher. Choose the next edit there, not where the code looks expensive. To go
     deeper into the costliest file, `proxy_eval(op='profile', level='function',
     focus=<that file>, confirm=True)`, then `level='line'`: both are read from the
     profile already taken, so they run nothing. For compiled code, `level='counters'`
     says why it is slow: a low `ipc` with a high `cache_miss_rate` is memory-bound, a
     high `ipc` compute-bound. When a view says `unavailable`, it says why (perf blocked
     by the kernel, no profiler for this command, cProfile keeping no lines); work from
     what there is and say what is missing.
   - `level='asm'` gives the instruction mix of a kernel's SASS (Nsight Compute) or a
     function's machine code (`perf annotate`): the classes and opcodes by how often they
     ran, calls into slow paths, spills, and notes. Integer arithmetic outweighing the
     floating point means index and address computation; a `CALL` per point means an IEEE
     division or a libm routine. A memory profile sees neither. When `kernel` says
     `compute: alu`, this is the next view.
   - `where_time_goes.gpu` — when the run uses a GPU: `busy` (kernels), `memcpy` and
     `idle` as shares of the run's time (of the region's with `window: measured
     region`: Nsight then traced it alone, the harness's kernels left out), and the
     costliest kernels. A high `idle` is the
     CPU and GPU waiting on each other (synchronisation, no overlap): fix that first. High
     `memcpy` is data moving between host and device: keep it on the device before tuning
     a kernel. Otherwise work on the costliest kernel; `op='profile', level='kernel',
     focus=<its name>` says whether it is `memory`, `compute` or `latency` bound. That run
     replays the kernel, so it is slow: ask for it once per kernel, not per step.
     `memcpy_mb` and `peak_device_mb` say how much the run moves and holds on the GPU.
   - Memory: `proxy_eval(op='profile', level='memory', confirm=True)` says which files
     allocate what is live at the heap's peak, then `focus=<file>` its functions and
     lines. It runs the case under memray, heaptrack or valgrind (2 to 100 times
     slower): ask for it when `peak_rss_mb` nears a ceiling or after a
     `memory_warning`, not every step. `level` also takes a list, e.g.
     `level=['file', 'memory']` for time and memory in one call: each is its own
     execution (an allocation profiler distorts time), and `where_time_goes` has one
     view per level. In `level='counters'`, a `bandwidth_fraction`
     above 0.7 says the DRAM is the limit: move fewer bytes (data layout, loop fusion,
     smaller types), not fewer operations.
   - Compiled code (C, C++, Fortran) needs `-g` in its build for files and lines, and
     `-fno-omit-frame-pointer` for a function's total time; neither changes the
     generated code. When a profile's note asks for them, add them to the build.
   - **reject** → read which kind:
     - `regressed` (slower than the best past twice the margin, or 5 %): the idea may be
       right and this version wrong. **Before** `reset_to_best`, while the files still
       hold this version, compare `where_time_goes` with `where_time_goes_best` and go down
       to the instructions: `op='profile', level=['kernel', 'asm'], focus=<kernel>` on a
       GPU, `level=['counters', 'asm'], focus=<function>` for compiled CPU code. When a
       cause shows (address arithmetic recomputed per point, a division routine called per
       point, spills, a register count that halved occupancy), fix it and try the idea
       again as `axis='<axis> (retry: <cause>)'`.
     - `near_miss` (better, but by less than the margin): kept on record. Two small gains
       on independent axes add up: `reset_to_best`, then `proxy_eval(op='apply_run',
       run_id=..., confirm=True)` for each, then one run `axis='combine: A + B'`.
     - otherwise: `reset_to_best`, then a *different* edit, or another version of the
       same idea if a profile says why this one did nothing.
   - **tradeoff** → judge it with `report_verdict` (above); it is not the best until then.
   - **stalled** → `max_stall` runs without a gain, far from the bound or with none known.
     It is not a convergence: the picture of what limits the code is probably wrong. Work
     out and declare the bound if there is none, profile down to `asm`, read
     `axes_tried.regressed` and `near_misses` again, then go on. One more streak follows.
   - **converged** → `proxy_eval(op='reset_to_best', confirm=True)`, then summarize,
     saying how far from the bound the best stands.
   - no verdict yet (constraints not met, no best) → edit the source and run again.
3. Editing: read the files in `optimize_paths`, understand what is slow or
   inaccurate, apply a targeted `replace_in_file(...)` to one of them, then go to
   step 1. To see where the time goes in the code as it stands now — before a first
   edit, or after reverting — `proxy_eval(op='profile', confirm=True)`: one profiled
   execution, never a measurement, so it moves nothing in the ratchet. Never edit the harness at `proxy_source_path` — it is the measuring
   instrument, not the subject.
4. **Compare runs**: `proxy_eval_status(op='diff')` or `(op='runs')` (note `is_best`).

A request from the code map (an `[R3]` axis, "Request from the code map") is this loop
with one more rule: every variant starts from the request's tree,
`proxy_eval(op='reset_to_request', request_id=...)`, not from the last one's. The
map-request skill has the protocol.

Axes optimised in parallel, each by a sub-agent on its own proxy and branch, are put
together and measured with `proxy_merge`, by you, as soon as two or more are back with
accepted gains: the proxy-merge skill has the protocol. Never assemble their files by
hand: a combination nobody measured can run and mean nothing.

## Invariants of the loop

- Read the file you are about to change before every modification. Never edit blind.
- Make one focused change per run cycle — do not batch multiple unrelated edits.
- Trust the `verdict`: never keep an edit the ratchet rejected — reset_to_best first
  (after diagnosing a regression, and `apply_run` is how a near miss comes back).
- **A correctness fix is not judged against a wrong best.** When the best turns out to be
  fast because it was wrong (a boundary not handled, a term left out), say so in the spec:
  tighten the constraint that catches it (`configure(spec={'constraints': ..., 'reason':
  ...})`). A best that fails it is retired, and the last kept run that meets it is the best
  again. Gains from there are counted against correct code. When no metric can catch it,
  take the best back with `report_verdict(run=<best run>, verdict='rejected', reason=...)`.
  Report gains against the corrected code, not against the wrong best.
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

## On a cluster

The loop is the same; only where a run happens changes. `proxy_cluster(op='eval',
partition=..., confirm=True)` submits it instead of running it here, and returns while
the job is still queued — end your turn, you are resumed when it lands. Picking the
partition and the node type, and building for the node rather than for the login host,
is the `run-on-cluster` method; what follows is only what the ratchet adds to it. Pass
`scheduler=` with the rest when this workspace reaches more than one — every submission
names its cluster there, and an unnamed one is refused rather than sent to whichever
came first.

Say where the **build** goes, separately from where the run goes. A node dedicated to
GPU simulation is not a node to compile on, and compiling there burns the allocation
you wanted the measurement from. Give the proxy a `build_partition` at registration:

    proxy_manage(op='register', name=..., metadata={
        'build_cmd': 'make -j32', 'build_partition': 'compile',
        'build_cpus_per_task': 32}, confirm=True)

From then on every `proxy_cluster(op='eval', partition='<the hardware you are measuring>')`
sends the compile to `compile` and the measurement to the partition you named, as two
chained jobs. Declare it once rather than repeating it per call — you will submit this
run many times, and that is where it goes wrong. `constraint='a100'` narrows a partition
to the nodes you actually want; without it Slurm picks any node in the queue, and a
comparison across two different nodes is not a comparison. A failed build ends the run
with no verdict, exactly as it does locally.

## When to stop

Stop and summarize when the session reports `verdict="converged"` (constraints met,
the primary metric stopped improving, and the best near the bound or the search reviewed
once), after restoring the best with `proxy_eval(op='reset_to_best', confirm=True)`.
`stalled` is not a stop. A summary says where the best stands against the bound. "We
are missing X" explains a gap only when the bound has been worked out. If no feasible run was ever found,
report what was tried and what remains. When you are fully done with the proxy,
`proxy_eval(op='end', confirm=True)` closes the session (code, snapshots and ledger are
kept) and lifts the direct-execution guard. To discard a proxy's runs, optimization state
and snapshots entirely — starting over rather than continuing — use
`proxy_manage(op='clean', name=..., confirm=True)`; it reports what it left behind
(sealed references, suites, the registry entry) and how to remove those too.
