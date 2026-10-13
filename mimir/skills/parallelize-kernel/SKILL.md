---
name: parallelize-kernel
description: Parallelize or vectorize a compute kernel — threads, SIMD, ranks or a device — keeping the result correct and proving the speedup is real.
disable-model-invocation: false
---

Two things go wrong when code is made parallel, and they go wrong quietly: the answer
changes, and the speedup is imaginary. Neither announces itself — a race usually prints
plausible numbers, and a measurement taken carelessly reports whatever the machine was
doing at the time.

## Before touching anything: a baseline you can return to

Measure the serial version, on the hardware you intend to end on, and keep the number.
Every later claim is a comparison against it, so without it there is nothing to compare
to — and a speedup quoted against your own first parallel attempt is not a speedup.

Then find where the time actually goes. The loop that looks expensive and the loop that
is expensive are different loops often enough that guessing costs more than measuring.
If a proxy is registered for this code, run the loop through `proxy-optimize` instead:
the ratchet already owns the baseline, the accept/reject margin and the snapshotting,
and doing that bookkeeping by hand is how an unreproducible 2% gain gets accepted.

Decide the correctness contract **now**, before the first edit: is the parallel result
required to be bitwise identical to the serial one, or equal within a stated tolerance?
A reduction reorders floating-point addition, so a correct parallel reduction gives a
different answer in the last bits, and a different answer every run when the schedule is
dynamic. Both answers are fine; what is not fine is discovering mid-way that you never
decided, and then treating a real divergence as rounding.

## Find the dependency before writing the directive

Read the loop and name, for each variable, whether it is private, shared read-only,
reduced, or carried across iterations. A loop-carried dependency parallelized is a race
— and a race on floating-point accumulation produces numbers that look right, vary
slightly between runs, and pass a tolerance-based test. The symptom arrives as
irreproducibility, not as a crash.

Two patterns deserve naming because they are invisible in the source: **false sharing**
(distinct threads writing adjacent entries of the same cache line, which makes a
perfectly parallel loop slower than serial) and **first-touch placement** (memory
allocated by one thread lives on that thread's NUMA node, so a serial initialization
followed by a parallel loop puts every access on one memory controller). Both show up as
"it scales to four threads and then stops", which is also what a genuine bandwidth limit
looks like — distinguish them by where the array was first written, not by staring at
the kernel.

## One axis at a time

Threads, then vectorization, then ranks, then device — in whatever order the code calls
for, but one at a time, each measured before the next. A combined change whose
measurement moved tells you nothing about which half moved it, and the half that
regressed is hidden by the half that helped.

Ask the platform tool what the machine has rather than assuming: cores, SIMD width,
NUMA topology, devices. A thread count above the cores available oversubscribes and
every measurement after that is noise.

## Measuring a speedup that is real

- **Sweep, do not spot-check.** Time at 1, 2, 4, 8, … up to the core count. One
  measurement at full width hides a plateau at four, which is the thing you most need to
  see, and the shape of the curve names the limit — flat from the start is a dependency
  or a lock, a plateau is bandwidth or sharing, a decline past a point is
  oversubscription or communication.
- **Median of repeats, never the best.** The minimum of several draws improves with the
  number of draws whatever the code does. Measure the spread first: when the run-to-run
  spread is 3%, a 2% gain is not a gain, and no amount of re-running makes it one.
- **Same input, same node, same build flags.** A comparison across two node types is two
  numbers. Pin it with a constraint if the queue is heterogeneous, and measure where you
  will run — see `run-on-cluster`.
- **Check the ceiling.** If the parallelizable fraction is 60%, no thread count gives
  more than 2.5×. A measured gain above the ceiling means the measurement is wrong —
  a warm cache, a smaller input, a timer that stopped enclosing the work, output that
  got optimized away — not that the code beat Amdahl.
- **Verify correctness at every width you measured.** Correct at 1 and 2 threads and
  wrong at 32 is the normal signature of a race, and the only way to see it is to check
  the answer at 32.

## Keeping or dropping a candidate

`report_verdict(verdict="rejected", reason=...)` with both numbers for a candidate that
measured cleanly and lost — that is the ordinary outcome of an optimization attempt, not
a failure, and recording it is what stops the same idea being tried twice. `pass` with
the speedup and the correctness check that still holds; `fail` when the parallel version
is wrong. Then revert a rejected candidate before trying the next one: layering a second
attempt on an unmeasured first produces a combination that was never measured together,
which can run and still mean nothing.

Stop when the curve has flattened and the remaining gap is explained — bandwidth,
communication, a serial fraction you can name. "No further speedup, and here is the
limit" is a complete result; a rewritten kernel with no measurement is not.
