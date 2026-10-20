"""What the end of a sub-agent's run says: stopped on its budget, abandoned, or finished,
and where it got to when it did not finish."""
from __future__ import annotations

import os
import sys
from typing import Any

# The fields of a reply with no answer in it: a refusal, a crash, a blank delegation.
NO_ANSWER: dict = {"answer": "", "completed": False, "files_read": [], "files_written": []}


def human_budget(secs: int) -> str:
    """The budget as a person would say it: "about 120 minutes" is a number to convert
    before it means anything."""
    if secs >= 3600:
        hours = secs / 3600
        if hours < 1.25:
            return "about an hour"
        return f"about {hours:.0f} hours" if hours >= 1.75 else "about an hour and a half"
    minutes = secs // 60
    return f"about {minutes} minutes" if minutes >= 2 else f"{secs} seconds"


def repo_relative(paths: list[str], worktree: dict | None) -> list[str]:
    """*paths* inside the child's worktree, as the repository names them; others unchanged."""
    root = (worktree or {}).get("path") or ""
    if not root:
        return paths
    root = os.path.join(os.path.realpath(root), "")
    out = []
    for path in paths:
        real = os.path.realpath(path)
        out.append(real[len(root):] if real.startswith(root) else path)
    return out


def abandon_child(child: dict) -> None:
    """Ask a child that overran its budget to stop (its ``cancel_flag``) and mark it abandoned.

    Its thread cannot be cancelled from here; the flag stops it at its next read.
    """
    child["abandoned"] = True
    agent = child.get("agent")
    flag = agent.turn.cancel_flag if agent is not None else None
    if flag is None:
        return
    try:
        flag.set()
    except Exception as exc:  # a child mid-teardown must not fail its caller
        print(f"spawn_agent: could not stop abandoned sub-agent: {exc}",
              file=sys.stderr)


def overrun_note(budget: int) -> str:
    """Why this run ended, said once and used by both paths.

    A run stopped by its clock is not a run that finished, and a caller told only
    "here is what it did" reads the handoff as a conclusion. The distinction has to be
    in the words, because ``completed: False`` alone is a flag the model may not weigh.
    """
    return (
        f"This sub-agent was stopped after {human_budget(budget)} — its time budget, "
        f"not the end of its work. What follows is where it had got to, not a "
        f"conclusion. Carry it on from here (its session holds its todo list and its "
        f"scratchpad) rather than starting the task again: a full budget has just been "
        f"spent on it and nothing about it has changed."
    )


def final_state(child: dict | None, result: dict | None) -> str:
    """The terminal state to record for a finished child.

    "abandoned" outranks the other two: a child whose caller timed out may still
    reach a clean result seconds later, and writing that as "finished" tells the
    panel a run succeeded when its answer went nowhere.
    """
    if child is not None and child.get("abandoned"):
        return "abandoned"
    return "finished" if result else "failed"


def partial_handoff(child: dict) -> dict:
    """What can be read off a child still running past its budget: where it got to."""
    agent = child.get("agent")
    worktree = child.get("worktree")
    files_read = repo_relative(agent.carry.files_read() if agent else [], worktree)
    files_written = repo_relative(agent.carry.files_written() if agent else [], worktree)
    lines = []
    if files_written:
        lines.append("Files it had already modified: " + ", ".join(files_written) + ".")
    if files_read:
        lines.append(f"It had read {len(files_read)} file(s).")
    lines.append("Its todo list and scratchpad are under its session, as it left them.")
    # Runs it launched and is no longer there to answer for. Read defensively, as the
    # agent above is: this is called from the caller's thread, and a child that never
    # got as far as a watcher has none to read. The plural key is what tells the client
    # these are runs to adopt rather than a run this call launched, and each descriptor
    # already names the directory the parent must read it through.
    watcher = child.get("watcher")
    handed_over: list[dict] = []
    try:
        handed_over = watcher.handover() if watcher is not None else []
    except Exception as exc:  # a child mid-teardown must not fail its caller
        print(f"spawn_agent: could not read a sub-agent's runs: {exc}", file=sys.stderr)
    if handed_over:
        keys = ", ".join(str(d.get("job_key") or "?") for d in handed_over)
        lines.append(f"Runs it left going, now watched here: {keys}.")
    out: dict[str, Any] = {
        "answer": " ".join(lines),
        "files_read": files_read,
        "files_written": files_written,
    }
    if handed_over:
        out["background_jobs"] = handed_over
    return out
