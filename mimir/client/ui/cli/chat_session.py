from __future__ import annotations

import os
from typing import Any

from .chat_commands import CliHost, run_cli_command
from ...context.capabilities import name_with_arg_role
from ...context.resource_context import augment_query_with_resources
from ...context.todo import load_todo_items
from ...query_engine.backends.factory import get_backend
from ...query_engine.history import (
    compacted_exchanges, compaction_summary_message,
)
from ...query_engine.verification import parse_ledger_block, split_answer_ledger
from ...query_engine.context_budget import context_budget_for
from ...session.conversation import Conversation, trim_front
from ...session.store import FullSession
from ...config.paths import STATE_DIR
from ...guardrails.workflow import (
    is_incomplete_answer, parse_completion_block, split_answer_completion,
)

_TODO_FILE = os.path.join(STATE_DIR, "todo_list.md")

# Ledger status → glyph. The ledger stays a one-liner unless the user asks for it,
# so the glyph carries the whole verdict at a glance.
_LEDGER_GLYPH = {"ok": "✅", "note": "🛡", "warn": "⚠️"}

# Completion-report status → glyph, on the same principle: the headline and the glyph
# are what the terminal shows, and the sections stay folded inside the answer text.
_COMPLETION_GLYPH = {"incomplete": "⚠️", "handback": "✋", "refused-only": "✅"}

_PROCEED_SIGNALS = frozenset({
    "start implementation",
    "proceed",
    "go ahead",
    "execute",
    "implement",
    "do it",
    "apply",
    "run it",
    "build it",
})

# Auto-compact history after this many accumulated user/assistant exchange pairs.
AUTO_COMPACT_THRESHOLD: int = 10


def _plain_row(row: str) -> str:
    """Drop the row's markdown emphasis — the terminal has no use for it."""
    return row.replace("**", "").replace("`", "")


def format_ledger_summary(block: str) -> str:
    """The collapsed one-liner printed under an answer, pointing at ``/ledger``.

    Kept to the first few chips so it stays one terminal line; /ledger has the rest.
    """
    led = parse_ledger_block(block)
    glyph = _LEDGER_GLYPH.get(led["status"], "🛡")
    chips = [c.strip() for c in led["summary"].split("·") if c.strip()]
    detail = " · ".join(chips[:3]) or f"{len(led['rows'])} item(s)"
    if len(chips) > 3:
        detail += f" · +{len(chips) - 3}"
    return f"\n{glyph} Verification · {detail}   —  /ledger to expand"


def format_ledger_full(block: str) -> str:
    """The expanded ledger, one row per line."""
    led = parse_ledger_block(block)
    glyph = _LEDGER_GLYPH.get(led["status"], "🛡")
    lines = [f"\n{glyph} Verification ledger — machine-recorded, not model-authored"]
    if led["summary"]:
        lines.append(f"   {led['summary']}")
    lines.extend(f"   • {_plain_row(r)}" for r in led["rows"])
    return "\n".join(lines) + "\n"


def format_completion_summary(block: str) -> str:
    """The one-liner printed under an answer whose run ended with something open.

    The terminal never reprints the answer — it is streamed as it is produced, and the
    report is appended after — so without a line of its own the report reaches the CLI not
    at all. One line is the whole of what it gets: the headline the report already chose as
    its readable-without-unfolding line, plus the residual risk.
    """
    rep = parse_completion_block(block)
    glyph = _COMPLETION_GLYPH.get(rep["status"], "⚠️")
    risk = f" · risk: {rep['risk']}" if rep["risk"] else ""
    return f"\n{glyph} {rep['headline']}{risk}"


def _is_proceed_signal(query: str) -> bool:
    """True when the user's short message means 'execute the last plan'."""
    q = query.strip().lower()
    if len(q.split()) > 6:
        return False
    return any(sig in q for sig in _PROCEED_SIGNALS)


def _confirm_plan_execution() -> bool:
    """Display pending plan steps and ask the user to confirm before executing."""
    items = load_todo_items(_TODO_FILE)
    pending = [it for it in items if not it.get("done")]
    if not pending:
        return True  # Nothing to show; proceed silently.

    print("\n📋 Pending plan steps:")
    for it in pending:
        print(f"  [ ] {it['text']}")

    try:
        answer = input("\nExecute this plan? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    return answer in ("y", "yes")


def _offer_session_resume() -> bool:
    """If a previous plan exists on disk, show it and ask whether to resume."""
    items = load_todo_items(_TODO_FILE)
    pending = [it for it in items if not it.get("done")]
    if not pending:
        return False

    print("\n📋 Unfinished plan from previous session:")
    for it in pending:
        print(f"  [ ] {it['text']}")

    try:
        answer = input("\nResume this plan? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    return answer in ("y", "yes")


async def run_chat_session(agent: Any) -> None:
    checklist_tool = name_with_arg_role("plan_steps", agent.tool_caps)
    if checklist_tool:
        if not _offer_session_resume():
            await agent.run_tool(checklist_tool, {"steps": []})

    print("\n✅ MIMIR ready. Type 'quit' to exit.")
    print(
        "   Commands: /mode [agent|plan], "
        "/status, /think on|off, /stream on|off, /batch on|off, "
        "/approvals manual|auto|all, /help\n"
    )

    # The conversation, in memory only: the terminal keeps no session store. The
    # same object the WebSocket front-end writes, with the same commit rule.
    conversation = Conversation(FullSession(id="cli", title="", created_at="",
                                            updated_at=""), saved=False)
    # Session-local auto-compact threshold; starts from the module default.
    auto_compact_threshold = AUTO_COMPACT_THRESHOLD
    # Cooldown: number of queries to skip auto-compact after one fires.
    # Prevents compacting after every single write-heavy query.
    _compact_cooldown: int = 0
    _COMPACT_COOLDOWN_TURNS: int = 3

    async def do_compact() -> None:
        """Summarize history in-place, replacing all prior messages with one summary."""
        nonlocal _compact_cooldown
        history = conversation.history
        if not history:
            print("  ↳ Nothing to compact.")
            return

        # Two different counts. The print says what is being collapsed right now; the
        # marker says what the summary will stand for, which includes everything an
        # earlier summary in *history* already absorbed.
        n_messages = len(history)
        collapsing = n_messages // 2
        n_exchanges = compacted_exchanges(history)
        print(f"⚡ Compacting history ({collapsing} exchange{'s' if collapsing != 1 else ''})...")

        summary = await agent.compact_history(history)
        if not summary:
            print("  ↳ Compaction returned empty summary; history unchanged.")
            return

        history.clear()
        history.append(compaction_summary_message(n_exchanges, summary))
        _compact_cooldown = _COMPACT_COOLDOWN_TURNS
        # Read before the clear above, not after it.
        print(f"  ↳ History compacted: {n_messages} messages → 1 summary message "
              f"standing for {n_exchanges} exchanges")

    def set_compact_threshold(n: int) -> None:
        nonlocal auto_compact_threshold
        auto_compact_threshold = n

    # Ledger block of the last answer, kept so /ledger can expand it on demand.
    last_ledger: str | None = None

    def show_ledger() -> str | None:
        return format_ledger_full(last_ledger) if last_ledger else None

    command_host = CliHost(agent, compact=do_compact,
                           set_compact_threshold=set_compact_threshold, ledger=show_ledger)

    while True:
        try:
            query = input("🟦 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit"):
            break

        try:
            message = await run_cli_command(command_host, query)
        except Exception as exc:
            # A command that fails is one answer gone wrong, not the session: raised,
            # it ended the chat loop and took the conversation with it.
            print(f"  ✗ {query.split()[0]} failed: {type(exc).__name__}: {exc}")
            continue
        if message is not None:
            if message:
                print(message)
            continue

        was_plan_mode = agent.mode == "plan"
        # Resolve any @<uri> resource mentions: read the referenced resources and
        # prepend their content to the query the model sees. History keeps the raw
        # `query`, so the attachment is per-turn (Claude/Copilot-style).
        effective_query, attached_uris = await augment_query_with_resources(agent, query)
        if attached_uris:
            print("📎 Attached: " + ", ".join(attached_uris) + "\n")

        # Pre-query budget check: compact before we send to the LLM if the
        # history already fills the usable window (total − reserved for answer).
        ctx_total, ctx_reserved, _, _ = context_budget_for(agent.model, agent.context_mode)
        # Exact token count (vLLM /tokenize) when available, heuristic otherwise.
        # CLI runs in-process, so a blocking tokenize round-trip here is fine.
        history_tokens = get_backend().count_messages_tokens(agent.model,
                                                            conversation.history)
        if history_tokens >= ctx_total - ctx_reserved and conversation.history:
            print(f"  ⚡ Context budget reached ({history_tokens:,}/{ctx_total - ctx_reserved:,} tokens) — auto-compacting…")
            await do_compact()

        if was_plan_mode and _is_proceed_signal(query):
            if not _confirm_plan_execution():
                print("  ↳ Execution cancelled.")
                continue

            agent.apply_setting("mode", "agent")
            print("  ↳ Auto-switched to agent mode — executing plan.\n")

            if agent.last_plan_query:
                effective_query = agent.last_plan_query

        try:
            role_label = "Agent"
            print(f"\n🟩 {role_label}:\n")
            submitted = len(conversation.history)
            answer = await agent.run(
                effective_query,
                history=conversation.history,
                thinking=agent.thinking,
                streaming=agent.streaming,
            )

            if was_plan_mode and not _is_proceed_signal(query):
                agent.last_plan_query = query

            # The ledger travels inside `answer` (history keeps it for the model); the
            # terminal only gets the one-liner unless the user runs /ledger.
            # Both blocks travel inside `answer` (history keeps them for the model); the
            # terminal only gets a one-liner for each. The ledger is appended last, so it
            # comes off first and the report is the tail of what is left.
            rest, last_ledger = split_answer_ledger(answer)
            ledger = parse_ledger_block(last_ledger) if last_ledger else None
            _, last_report = split_answer_completion(rest)
            if last_report:
                print(format_completion_summary(last_report))
            if ledger:
                print(format_ledger_summary(last_ledger))

            # Written by the rule the WebSocket front-end uses: in full-context mode the
            # turn's whole transcript (tool calls and results included) becomes the
            # window, otherwise the question and the answer alone.
            full = agent._last_full_messages if agent.context_mode == "full" else None
            if full is None:
                conversation.add_user_message(query)
            conversation.commit_answer(
                {"text": answer},
                {"_full": list(full) if full else None,
                 "_turn_start": agent._last_turn_start, "_submitted_len": submitted,
                 "_context_mode": agent.context_mode})

            if full is not None:
                # Evict the oldest messages past the window the model actually has —
                # its own context budget, not a figure written here.
                total, reserved, _, _ = context_budget_for(agent.model, "full")
                history = conversation.history
                counts = get_backend().message_token_counts(agent.model, history)
                trim_front(history, counts, max(1, total - reserved))
            else:
                # Decrement cooldown each turn so write-bursts don't re-compact
                # immediately after a recent compact.
                if _compact_cooldown > 0:
                    _compact_cooldown -= 1

                wrote_files = bool(ledger and ledger["files"])
                n_exchanges = len(conversation.history) // 2
                # Compact when:
                #   - plan was executed (always worth summarising)
                #   - files were written AND we have ≥2 exchanges AND no cooldown
                #   - exchange count hit the threshold
                should_compact = (
                    was_plan_mode
                    or (wrote_files and n_exchanges >= 2 and _compact_cooldown == 0)
                    or n_exchanges >= auto_compact_threshold
                )
                if should_compact:
                    await do_compact()

            if agent.mode == "agent" and is_incomplete_answer(answer):
                try:
                    replan = input("Return to plan mode to re-plan? [y/N] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    replan = ""

                if replan in ("y", "yes"):
                    agent.apply_setting("mode", "plan")
                    print("  ↳ Switched to plan mode.\n")

        except Exception as exc:
            print(f"❌ Error: {exc}")