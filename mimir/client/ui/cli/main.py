from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    from mimir.client.agent_core import MimirAgent
    from mimir.client.ui.cli.chat_session import run_chat_session
    from mimir.client.config import DEFAULT_MODEL
    from mimir.client.extensions import all_servers
    from mimir.client import human_pause
else:
    from ...agent_core import MimirAgent
    from .chat_session import run_chat_session
    from ...config import DEFAULT_MODEL
    from ...extensions import all_servers
    from ... import human_pause


async def main() -> None:
    model = DEFAULT_MODEL
    if len(sys.argv) >= 3 and sys.argv[1] == "--model":
        model = sys.argv[2]

    print(f"Using model: {model} ({os.environ.get('LLM_BACKEND', 'vllm')})")
    agent = MimirAgent(model=model)

    # Interactive CLI: let a long run ask the user whether to keep going past
    # the soft step budget instead of stopping silently.
    agent.allow_continue_prompt = True
    agent._request_continue = _cli_request_continue
    agent._request_user_question = _cli_request_question

    for server_name, script_path in all_servers().items():
        await agent.connect_server(server_name, script_path)
    agent.seed_classification_from_caps()

    await run_chat_session(agent)
    await agent.cleanup()


def _cli_request_continue(summary: str) -> bool:
    """Prompt the user at a soft step-budget checkpoint. Returns True to continue."""
    print("\n⏸  Step budget reached.")
    if summary:
        print(f"   {summary}")
    try:
        choice = input("   Continue working? [y]es / [n]o: ").strip().lower()
    except EOFError:
        return False
    return choice in {"y", "yes"}


def _cli_ask_one(
    question: str,
    header: str,
    options: list,
    multi_select: bool,
    *,
    progress: str = "",
) -> dict:
    """Prompt the user with a single multiple-choice question. Returns the selection.

    Prints the question and numbered options (plus an "Other" free-text entry) and
    reads a selection from stdin. Returns ``{"selected": [<labels>], "other_text":
    <str|None>}``; an empty selection means the user declined for this question.
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
    try:
        # `ask_user_question` is itself a tool call, so the user's thinking time
        # would otherwise burn its timeout budget (see human_pause).
        with human_pause.human_pause():
            raw = input(prompt).strip()
    except EOFError:
        return {"selected": [], "other_text": None}

    if not raw:
        return {"selected": [], "other_text": None}

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
                    other_text = input("   Your answer: ").strip() or None
            except EOFError:
                other_text = None
            if other_text:
                selected.append(other_text)
        elif 1 <= idx <= len(labels):
            selected.append(labels[idx - 1])

    return {"selected": selected, "other_text": other_text}


def _cli_request_question(questions: list) -> dict:
    """Ask the user one or more clarifying questions sequentially via stdin.

    Mirrors :func:`_cli_request_continue`. Each item is a ``{header, question,
    multiSelect, options}`` spec; questions are asked one at a time and the answers
    are collected in order. Returns ``{"answers": [{"selected": [...], "other_text":
    ...}, ...]}``; an all-empty result means the user declined and the agent should
    proceed with its best judgment.
    """
    total = len(questions)
    answers: list[dict] = []
    for i, q in enumerate(questions, 1):
        q = q or {}
        progress = f" ({i}/{total})" if total > 1 else ""
        answers.append(
            _cli_ask_one(
                str(q.get("question", "")),
                str(q.get("header", "")),
                q.get("options") or [],
                bool(q.get("multiSelect") or q.get("multi_select")),
                progress=progress,
            )
        )

    if not any(a.get("selected") or a.get("other_text") for a in answers):
        return {"answers": []}
    return {"answers": answers}


def main_sync() -> None:
    """Synchronous entry point for the ``mimir`` console script."""
    asyncio.run(main())


__all__ = ["main", "main_sync"]


if __name__ == "__main__":
    main_sync()