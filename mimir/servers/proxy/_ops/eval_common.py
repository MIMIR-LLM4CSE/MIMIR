"""What the act and the observe ops of an optimization session share: the next-step
hints, the log tail, the background descriptor of a run."""

from __future__ import annotations


from .._lib.procs import (
    read_log,
    background_descriptor,
)



NEXT_RUN       = "proxy_eval(op='run', confirm=True)"


NEXT_STATUS    = "proxy_eval_status() to monitor (state 'done' means finished)"


NEXT_RESULTS   = "proxy_eval_status(op='results')"


NEXT_RESET_BEST = "proxy_eval(op='reset_to_best', confirm=True)"


NEXT_DETACHED  = ("this run continues without you; if the result says it is being "
                   "watched, end your turn on it rather than polling — you are resumed "
                   "with the results")


# Above this measured build time, a run is worth detaching rather than waiting out.
# One minute: short enough to catch any real compiled project, long enough that a
# scripted proxy with a trivial build never triggers it.
LONG_BUILD_S = 60.0


def opt_tail_log(run_dir: str, n: int) -> list[str]:
    lines = read_log(run_dir).splitlines()
    return lines[-n:]


def eval_background_descriptor(name: str, run_dir: str) -> dict:
    """The watcher handle for an optimization run: it polls the session, not the run."""
    return background_descriptor(
        run_dir, kind="proxy-optimization",
        status_op={"tool": "proxy_eval_status", "args": {"proxy_name": name}},
        summary_op={"tool": "proxy_eval_status",
                    "args": {"op": "results", "proxy_name": name}})
