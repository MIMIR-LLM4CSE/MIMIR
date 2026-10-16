"""The agent as the engine, the guardrails and the tool executor may use it.

:class:`AgentPort` is the whole of what those layers reach for on an agent — nothing
else of ``MimirAgent`` is theirs to call. It is a ``Protocol``, so a test can hand in
any object with these members, and the real agent is checked against it by type, not
by inheritance.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol

from ..context.execution_context import ExecutionContext


class AgentPort(Protocol):
    # ── What the agent is ──────────────────────────────────────────────────────
    model: str
    mode: str
    session_id: str | None
    workspace_root: str
    # The live tool registry: name -> declared capabilities, and name -> its server.
    tool_caps: dict[str, Any]
    tool_owner: dict[str, str]
    tools: list[dict]
    sessions: dict[str, Any]
    skills: dict[str, dict]
    resources: dict[str, Any]
    approvals: Any                       # guardrails.policy.approval.ApprovalManager

    # ── Settings read live, at every step ──────────────────────────────────────
    thinking: bool
    thinking_depth: int
    thinking_budget: int
    temperature: float | None
    context_mode: str
    enforcement: str
    subagent_level: str
    disabled_nudges: set[str]
    # Bounds a caller may set (a sub-agent is held below the model's window).
    max_context_tokens: int
    max_answer_tokens: int

    # ── The turn in flight, as the front-ends read it ──────────────────────────
    _live_messages: list[dict] | None
    _turn_opening_message: dict | None
    _last_turn_start: int | None
    _last_full_messages: list[dict] | None
    _tool_cache: dict
    _cancel_flag: Any                    # threading.Event the user's Stop sets
    _deferred_prompts: list[dict]

    # ── Paths and tool calls ───────────────────────────────────────────────────
    def normalize_workspace_path(self, path: str | None) -> str: ...
    def is_code_filepath(self, path: str) -> bool: ...
    def is_write_tool(self, tool_name: str) -> bool: ...
    def normalize_tool_arguments(self, tool_name: str, arguments: dict) -> dict: ...
    def rewrite_tool_for_context(self, tool_name: str, arguments: dict) -> tuple[str, dict]: ...
    def get_tool_file_targets(self, tool_name: str, arguments: dict) -> list[str]: ...
    def advertised_tools(self) -> list[dict]: ...
    def skill_enabled(self, name: str) -> bool: ...
    run_tool: Callable[..., Awaitable[str]]

    # ── Approvals and the user ─────────────────────────────────────────────────
    def approval_scope(self, tool_name: str, arguments: dict) -> str: ...
    def request_tool_approval(self, tool_name: str, arguments: dict) -> tuple[bool, str]: ...
    def request_path_approval(self, *args: Any, **kwargs: Any) -> Any: ...
    def request_user_question(self, questions: list, *args: Any, **kwargs: Any) -> Any: ...
    def record_denied_tool_call(self, tool_name: str, arguments: dict,
                                execution_context: ExecutionContext | None,
                                note: str = "") -> None: ...
    def denied_tool_result(self, tool_name: str, arguments: dict, note: str = "",
                           execution_context: ExecutionContext | None = None) -> str: ...

    # ── The query's state ──────────────────────────────────────────────────────
    def new_execution_context(self, workspace_root: str = "") -> ExecutionContext: ...
    def apply_carry_context(self, execution_context: ExecutionContext) -> None: ...
    def update_carry_context(self, execution_context: ExecutionContext) -> None: ...
    def todo_file(self) -> str: ...
    async def build_system_content(self, active_mode: str) -> str: ...
