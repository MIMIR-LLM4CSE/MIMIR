from __future__ import annotations

import argparse
import asyncio
import os
import select
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    from mimir import __version__
    from mimir.client.agent_core import MimirAgent
    from mimir.client.ui.cli.chat_session import run_chat_session
    from mimir.client.config import DEFAULT_MODEL
    from mimir.client.extensions import all_servers
    from mimir.client.protocol import human_pause
    from mimir.client.core.human import TerminalHuman
else:
    from .... import __version__
    from ...agent_core import MimirAgent
    from .chat_session import run_chat_session
    from ...config import DEFAULT_MODEL
    from ...extensions import all_servers
    from ...protocol import human_pause
    from ...core.human import TerminalHuman


class CliHuman(TerminalHuman):
    """The terminal, which also answers the model's questions on stdin."""

    def ask(self, agent: Any, questions: list[Any], origin: dict[str, Any] | None = None,
            timeout_secs: float | None = None) -> dict[str, Any]:
        return _cli_request_question(questions, origin, timeout_secs)


async def main(model: str | None = None) -> None:
    model = model or DEFAULT_MODEL

    print(f"Using model: {model} ({os.environ.get('LLM_BACKEND', 'vllm')})")
    agent = MimirAgent(model=model)

    agent.human = CliHuman()

    await agent.connect_servers(all_servers())
    agent.seed_classification_from_caps()

    await run_chat_session(agent)
    await agent.cleanup()


class _QuestionExpired(Exception):
    """Raised inside a question prompt when its answer deadline passes."""


def _ask_line(prompt: str, deadline: float | None) -> str:
    """Read one line from stdin, giving up at *deadline* (``_QuestionExpired``).

    Without a deadline, plain ``input()``; with one, stdin is polled, since ``input()``
    cannot be interrupted. ``EOFError`` when stdin is gone.
    """
    if deadline is None:
        return input(prompt).strip()
    print(prompt, end="", flush=True)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _QuestionExpired
        try:
            # Sliced, so Ctrl-C lands within a second rather than at the deadline.
            ready, _, _ = select.select([sys.stdin], [], [], min(remaining, 1.0))
        except (OSError, ValueError):
            # No pollable stdin (a wrapped stream, a platform without it): the wall
            # is worth less than the prompt, so fall back to a blocking read.
            return input().strip()
        if ready:
            line = sys.stdin.readline()
            if not line:
                raise EOFError
            return line.strip()


def _cli_ask_one(
    question: str,
    header: str,
    options: list,
    multi_select: bool,
    *,
    progress: str = "",
    deadline: float | None = None,
) -> dict:
    """Ask one multiple-choice question: ``{"selected": [labels], "other_text": str | None}``.

    Empty selection means declined; ``_QuestionExpired`` when *deadline* passes.
    """
    labels = [str((o or {}).get("label", "")).strip() for o in options]
    labels = [lbl for lbl in labels if lbl]
    descriptions = {
        str((o or {}).get("label", "")).strip(): str((o or {}).get("description", "")).strip()
        for o in options
    }

    print(f"\n❓ {header}{progress}".rstrip())
    if question:
        print(f"   {question}")
    if deadline is not None:
        left = max(0, int(deadline - time.monotonic()))
        print(f"   (answer within {left // 60 or 1} min, or I go with [1])")
    for i, lbl in enumerate(labels, 1):
        desc = descriptions.get(lbl)
        print(f"   [{i}] {lbl}" + (f" — {desc}" if desc else ""))
    other_idx = len(labels) + 1
    print(f"   [{other_idx}] Other (type your own answer)")

    prompt = (
        "   Select options (comma-separated): "
        if multi_select
        else "   Select an option: "
    )
    # The agent asked because it needs the answer, so an empty or unparseable line
    # re-prompts instead of counting as "declined": the run stays parked until a real
    # selection arrives. EOF is the one way out — it means there is no interactive
    # stdin to block on at all (a pipe, a headless run), not a user choosing silence.
    while True:
        try:
            # `ask_user_question` is itself a tool call, so the user's thinking time
            # would otherwise burn its timeout budget (see human_pause).
            with human_pause.human_pause():
                raw = _ask_line(prompt, deadline)
        except EOFError:
            return {"selected": [], "other_text": None}

        if not raw:
            print("   An answer is needed to go on — the run is waiting.")
            continue

        picks = [p.strip() for p in raw.split(",") if p.strip()] if multi_select else [raw]
        selected: list[str] = []
        other_text: str | None = None
        for pick in picks:
            if not pick.isdigit():
                continue
            idx = int(pick)
            if idx == other_idx:
                try:
                    with human_pause.human_pause():
                        # No deadline on the free-text entry: the user is at the
                        # keyboard typing, which is the thing the wall was waiting for.
                        other_text = _ask_line("   Your answer: ", None) or None
                except EOFError:
                    other_text = None
                if other_text:
                    selected.append(other_text)
            elif 1 <= idx <= len(labels):
                selected.append(labels[idx - 1])

        if selected:
            return {"selected": selected, "other_text": other_text}
        print(f"   Enter a number between 1 and {other_idx}.")


def _cli_request_question(questions: list, origin: dict | None = None,
                          timeout_secs: float | None = None) -> dict:
    """Ask *questions* (``{header, question, multiSelect, options}``) one after another on stdin.

    Returns ``{"answers": [...]}``; all empty means declined. *timeout_secs* bounds the
    whole batch, and on expiry the result is flagged ``timed_out``.
    """
    total = len(questions)
    deadline = None if timeout_secs is None else time.monotonic() + timeout_secs
    answers: list[dict] = []
    # Who is asking, when it is not the turn the user is watching. The terminal has no
    # badge to put it in, so it goes in front of the question — which is worse than a
    # badge and much better than an unattributed "allow this?".
    who = f"[{origin.get('label') or 'sub-agent'}] " if origin else ""
    for i, q in enumerate(questions, 1):
        q = q or {}
        progress = f" ({i}/{total})" if total > 1 else ""
        try:
            answers.append(
                _cli_ask_one(
                    who + str(q.get("question", "")),
                    str(q.get("header", "")),
                    q.get("options") or [],
                    bool(q.get("multiSelect") or q.get("multi_select")),
                    progress=progress,
                    deadline=deadline,
                )
            )
        except _QuestionExpired:
            # Answers already given are dropped with the rest: the tool reports a
            # batch that nobody finished, and a half-answered batch has no shape in
            # its result — the model goes with what it recommended for all of them.
            print("\n   No answer — closing the question and going with "
                  "the recommended option.\n", flush=True)
            return {"answers": [], "timed_out": True, "timeout_secs": timeout_secs}

    if not any(a.get("selected") or a.get("other_text") for a in answers):
        return {"answers": []}
    return {"answers": answers}


def main_sync() -> None:
    """Synchronous entry point for the ``mimir`` console script."""
    parser = argparse.ArgumentParser(
        prog="mimir",
        description="MIMIR — interactive CLI agent (math, code, HPC).",
    )
    parser.add_argument(
        "--model",
        default=argparse.SUPPRESS,
        help="LLM model to use (default: $MIMIR_DEFAULT_MODEL).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    args = parser.parse_args()
    asyncio.run(main(model=getattr(args, "model", None)))


__all__ = ["main", "main_sync"]


if __name__ == "__main__":
    main_sync()