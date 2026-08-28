from __future__ import annotations

import os
from typing import Any

from .chat_commands import handle_chat_command
from ...context.capabilities import name_with_arg_role
from ...context.resource_context import augment_query_with_resources
from ...prompt.system_prompt import _load_todo_items
from ...query_engine.backends.factory import get_backend
from ...query_engine.verification import parse_ledger_block, split_answer_ledger
from ...config.constants import context_budget_for, STATE_DIR
from ...guardrails.workflow import is_incomplete_answer

_TODO_FILE = os.path.join(STATE_DIR, "todo_list.md")

# Ledger status → glyph. The ledger stays a one-liner unless the user asks for it,
# so the glyph carries the whole verdict at a glance.
_LEDGER_GLYPH = {"ok": "✅", "note": "🛡", "warn": "⚠️"}

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


def _is_proceed_signal(query: str) -> bool:
    """True when the user's short message means 'execute the last plan'."""
    q = query.strip().lower()
    if len(q.split()) > 6:
        return False
    return any(sig in q for sig in _PROCEED_SIGNALS)


def _confirm_plan_execution() -> bool:
    """Display pending plan steps and ask the user to confirm before executing."""
    items = _load_todo_items(_TODO_FILE)
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
    items = _load_todo_items(_TODO_FILE)
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
            await agent._run_tool(checklist_tool, {"steps": []})

    print("\n✅ MIMIR ready. Type 'quit' to exit.")
    print(
        "   Commands: /mode [agent|plan], "
        "/status, /think on|off, /stream on|off, /batch on|off, /help\n"
    )

    history: list[dict[str, str]] = []
    # Session-local auto-compact threshold; starts from the module default.
    auto_compact_threshold = AUTO_COMPACT_THRESHOLD
    # Cooldown: number of queries to skip auto-compact after one fires.
    # Prevents compacting after every single write-heavy query.
    _compact_cooldown: int = 0
    _COMPACT_COOLDOWN_TURNS: int = 3

    async def do_compact() -> None:
        """Summarize history in-place, replacing all prior messages with one summary."""
        nonlocal _compact_cooldown
        if not history:
            print("  ↳ Nothing to compact.")
            return

        n_exchanges = len(history) // 2
        print(f"⚡ Compacting history ({n_exchanges} exchange{'s' if n_exchanges != 1 else ''})...")

        summary = await agent.compact_history(history)
        if not summary:
            print("  ↳ Compaction returned empty summary; history unchanged.")
            return

        history.clear()
        # Store as assistant role so the model reads the summary as its own
        # prior memory, not as something the user typed.
        history.append({
            "role": "assistant",
            "content": (
                f"[Context Summary — {n_exchanges} prior exchange"
                f"{'s' if n_exchanges != 1 else ''} compacted]\n\n{summary}"
            ),
        })
        _compact_cooldown = _COMPACT_COOLDOWN_TURNS
        print(f"  ↳ History compacted: {n_exchanges * 2} messages → 1 summary message")

    def set_compact_threshold(n: int) -> None:
        nonlocal auto_compact_threshold
        auto_compact_threshold = n

    # Ledger block of the last answer, kept so /ledger can expand it on demand.
    last_ledger: str | None = None

    def show_ledger() -> str | None:
        return format_ledger_full(last_ledger) if last_ledger else None

    while True:
        try:
            query = input("🟦 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit"):
            break

        handled, message = await handle_chat_command(
            query=query,
            mode=agent.mode,
            thinking=agent.thinking,
            streaming=agent.streaming,
            batch_mode=agent.approvals.batch_mode,
            context_mode=agent.context_mode,
            enforcement=agent.enforcement,
            set_mode=agent.set_mode,
            set_thinking=agent.set_thinking,
            thinking_depth=agent.thinking_depth,
            set_thinking_depth=agent.set_thinking_depth,
            set_streaming=agent.set_streaming,
            set_batch_mode=agent.set_batch_mode,
            set_context_mode=agent.set_context_mode,
            set_enforcement=agent.set_enforcement,
            trust_tool=agent.approvals.trust_tool,
            untrust_tool=agent.approvals.untrust_tool,
            trusted_tools=agent.approvals.trusted_tools_view(),
            compact_history=do_compact,
            compact_threshold_setter=set_compact_threshold,
            show_ledger=show_ledger,
            approval_manager=agent.approvals,
            agent=agent,
        )
        if handled:
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
        history_tokens = get_backend().count_messages_tokens(agent.model, history)
        if history_tokens >= ctx_total - ctx_reserved and history:
            print(f"  ⚡ Context budget reached ({history_tokens:,}/{ctx_total - ctx_reserved:,} tokens) — auto-compacting…")
            await do_compact()

        if was_plan_mode and _is_proceed_signal(query):
            if not _confirm_plan_execution():
                print("  ↳ Execution cancelled.")
                continue

            agent.set_mode("agent")
            print("  ↳ Auto-switched to agent mode — executing plan.\n")

            if agent.last_plan_query:
                effective_query = agent.last_plan_query

        try:
            role_label = "Agent"
            print(f"\n🟩 {role_label}:\n")
            answer = await agent.run(
                effective_query,
                history=history,
                thinking=agent.thinking,
                streaming=agent.streaming,
            )

            if was_plan_mode and not _is_proceed_signal(query):
                agent.last_plan_query = query

            # The ledger travels inside `answer` (history keeps it for the model); the
            # terminal only gets the one-liner unless the user runs /ledger.
            _, last_ledger = split_answer_ledger(answer)
            ledger = parse_ledger_block(last_ledger) if last_ledger else None
            if ledger:
                print(format_ledger_summary(last_ledger))

            if agent.context_mode == "full":
                # Full-context mode: replace history with the complete accumulated
                # transcript from this query. agent._last_full_messages is
                # messages[1:], which was built as [old_history + new_user_msg +
                # tool_calls/results + final_assistant], so it already carries
                # everything from all prior queries. No duplication occurs.
                #
                # Intelligent trimming: evict the oldest messages when we exceed
                # the budget. Budget = 120 000 tokens (safe headroom inside a 200K
                # window; leaves room for the system message, new query, and
                # generation). Token counts are exact (vLLM) or heuristic (Ollama).
                FULL_CTX_TOKEN_BUDGET = 120_000

                history.clear()
                history.extend(agent._last_full_messages)

                # Trim from the front until we're within budget. Per-message counts
                # are computed once (cached) and decremented as we pop.
                backend = get_backend()
                counts = backend.message_token_counts(agent.model, history)
                total_tokens = sum(counts)
                idx = 0
                while total_tokens > FULL_CTX_TOKEN_BUDGET and len(history) > 1 and idx < len(counts):
                    history.pop(0)
                    total_tokens -= counts[idx]
                    idx += 1
                # Front-trimming can orphan a {"role": "tool"} whose assistant tool_call
                # was popped, which strict tokenizers (Mistral) reject — so drop leading
                # tool messages until history starts on a valid turn boundary. Mirrors
                # the same guard in the WebSocket session's pre-query trim.
                while history and history[0].get("role") == "tool":
                    history.pop(0)
            else:
                # Compact mode: only keep the final user/assistant text in history,
                # then compact when genuinely needed.
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": answer})

                # Decrement cooldown each turn so write-bursts don't re-compact
                # immediately after a recent compact.
                if _compact_cooldown > 0:
                    _compact_cooldown -= 1

                wrote_files = bool(ledger and ledger["files"])
                n_exchanges = len(history) // 2
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
                    agent.set_mode("plan")
                    print("  ↳ Switched to plan mode.\n")

        except Exception as exc:
            print(f"❌ Error: {exc}")