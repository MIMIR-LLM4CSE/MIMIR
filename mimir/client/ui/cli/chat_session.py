from __future__ import annotations

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
from ...guardrails.workflow import (
    is_incomplete_answer, parse_completion_block, split_answer_completion,
)


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
    """The one line printed under an answer whose run ended with something open: headline and risk."""
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


def _confirm_plan_execution(todo_file: str) -> bool:
    """Display pending plan steps and ask the user to confirm before executing."""
    items = load_todo_items(todo_file)
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


def _offer_session_resume(todo_file: str) -> bool:
    """If a previous plan exists on disk, show it and ask whether to resume."""
    items = load_todo_items(todo_file)
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


# Queries to skip auto-compact after one fires, so a burst of writes does not compact
# after every query.
_COMPACT_COOLDOWN_TURNS: int = 3


class CliSession:
    """The terminal conversation: in memory only, written by the same commit rule as
    the WebSocket front-end."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.conversation = Conversation(
            FullSession(id="cli", title="", created_at="", updated_at=""), saved=False)
        self.compact_threshold = AUTO_COMPACT_THRESHOLD
        self.compact_cooldown = 0
        # The last answer's ledger block, which /ledger expands.
        self.last_ledger: str | None = None
        self.commands = CliHost(agent, compact=self.compact,
                                set_compact_threshold=self.set_compact_threshold,
                                ledger=self.show_ledger)

    def set_compact_threshold(self, n: int) -> None:
        self.compact_threshold = n

    def show_ledger(self) -> str | None:
        return format_ledger_full(self.last_ledger) if self.last_ledger else None

    async def compact(self) -> None:
        """Replace the whole history with one summary message."""
        history = self.conversation.history
        if not history:
            print("  ↳ Nothing to compact.")
            return
        # What is collapsed now, and what the summary stands for (earlier summaries
        # included), read before the history is cleared.
        n_messages = len(history)
        collapsing = n_messages // 2
        n_exchanges = compacted_exchanges(history)
        print(f"⚡ Compacting history ({collapsing} exchange{'s' if collapsing != 1 else ''})...")
        summary = await self.agent.compact_history(history)
        if not summary:
            print("  ↳ Compaction returned empty summary; history unchanged.")
            return
        history.clear()
        history.append(compaction_summary_message(n_exchanges, summary))
        self.compact_cooldown = _COMPACT_COOLDOWN_TURNS
        print(f"  ↳ History compacted: {n_messages} messages → 1 summary message "
              f"standing for {n_exchanges} exchanges")

    async def run(self) -> None:
        await self._greet()
        while True:
            query = _read_query()
            if query is None:
                break
            if await self._handled_as_command(query):
                continue
            try:
                await self._answer(query)
            except Exception as exc:
                print(f"❌ Error: {exc}")

    async def _greet(self) -> None:
        checklist_tool = name_with_arg_role("plan_steps", self.agent.tool_caps)
        if checklist_tool and not _offer_session_resume(self.agent.todo_file()):
            await self.agent.run_tool(checklist_tool, {"steps": []})
        print("\n✅ MIMIR ready. Type 'quit' to exit.")
        print("   Commands: /mode [agent|plan], /status, /think on|off, /stream on|off, "
              "/batch on|off, /approvals manual|auto|all, /help\n")

    async def _handled_as_command(self, query: str) -> bool:
        try:
            message = await run_cli_command(self.commands, query)
        except Exception as exc:
            # One command gone wrong, not the session.
            print(f"  ✗ {query.split()[0]} failed: {type(exc).__name__}: {exc}")
            return True
        if message is None:
            return False
        if message:
            print(message)
        return True

    async def _answer(self, query: str) -> None:
        agent = self.agent
        was_plan_mode = agent.mode == "plan"
        effective_query = await self._prepare_query(query)
        if was_plan_mode and _is_proceed_signal(query):
            if not _confirm_plan_execution(agent.todo_file()):
                print("  ↳ Execution cancelled.")
                return
            agent.apply_setting("mode", "agent")
            print("  ↳ Auto-switched to agent mode — executing plan.\n")
            effective_query = agent.last_plan_query or effective_query

        print("\n🟩 Agent:\n")
        submitted = len(self.conversation.history)
        answer = await agent.run(effective_query, history=self.conversation.history,
                                 thinking=agent.thinking, streaming=agent.streaming)
        if was_plan_mode and not _is_proceed_signal(query):
            agent.last_plan_query = query

        ledger = self._print_summaries(answer)
        self._commit(query, answer, submitted)
        await self._after_turn(was_plan_mode, ledger)
        if agent.mode == "agent" and is_incomplete_answer(answer):
            _offer_replan(agent)

    async def _prepare_query(self, query: str) -> str:
        """Attach @-mentioned resources for this turn, and compact first when the
        history already fills the usable window."""
        effective_query, attached_uris = await augment_query_with_resources(self.agent, query)
        if attached_uris:
            print("📎 Attached: " + ", ".join(attached_uris) + "\n")
        total, reserved, _, _ = context_budget_for(self.agent.model, self.agent.context_mode)
        history = self.conversation.history
        tokens = get_backend().count_messages_tokens(self.agent.model, history)
        if history and tokens >= total - reserved:
            print(f"  ⚡ Context budget reached ({tokens:,}/{total - reserved:,} tokens)"
                  " — auto-compacting…")
            await self.compact()
        return effective_query

    def _print_summaries(self, answer: str) -> dict | None:
        """One line each for the completion report and the ledger, which both stay in
        the answer for the model. The ledger is appended last, so it comes off first."""
        rest, self.last_ledger = split_answer_ledger(answer)
        _, report = split_answer_completion(rest)
        if report:
            print(format_completion_summary(report))
        if not self.last_ledger:
            return None
        print(format_ledger_summary(self.last_ledger))
        return parse_ledger_block(self.last_ledger)

    def _commit(self, query: str, answer: str, submitted: int) -> None:
        """In full-context mode the turn's whole transcript becomes the window,
        otherwise the question and the answer alone."""
        agent = self.agent
        full = agent.turn.last_full_messages if agent.context_mode == "full" else None
        if full is None:
            self.conversation.add_user_message(query)
        self.conversation.commit_answer(
            {"text": answer},
            {"_full": list(full) if full else None,
             "_turn_start": agent.turn.last_turn_start, "_submitted_len": submitted,
             "_context_mode": agent.context_mode})

    async def _after_turn(self, was_plan_mode: bool, ledger: dict | None) -> None:
        """Full context: evict the oldest messages past the model's own budget.
        Otherwise compact after an executed plan, after writes, or past the threshold."""
        history = self.conversation.history
        if self.agent.context_mode == "full":
            total, reserved, _, _ = context_budget_for(self.agent.model, "full")
            counts = get_backend().message_token_counts(self.agent.model, history)
            trim_front(history, counts, max(1, total - reserved))
            return
        if self.compact_cooldown > 0:
            self.compact_cooldown -= 1
        wrote_files = bool(ledger and ledger["files"])
        n_exchanges = len(history) // 2
        if (was_plan_mode
                or (wrote_files and n_exchanges >= 2 and self.compact_cooldown == 0)
                or n_exchanges >= self.compact_threshold):
            await self.compact()


def _read_query() -> str | None:
    """The next query, or None when the user leaves."""
    try:
        query = input("🟦 You: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return None if not query or query.lower() in ("quit", "exit") else query


def _offer_replan(agent: Any) -> None:
    try:
        replan = input("Return to plan mode to re-plan? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        replan = ""
    if replan in ("y", "yes"):
        agent.apply_setting("mode", "plan")
        print("  ↳ Switched to plan mode.\n")


async def run_chat_session(agent: Any) -> None:
    await CliSession(agent).run()
