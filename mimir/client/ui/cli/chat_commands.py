from __future__ import annotations

import json

from typing import Any, Awaitable, Callable, Iterable

from ...config import THINKING_DEPTH_LABELS, thinking_depth_from_label


# What the user types → the mode name. "all" is the spoken form of ``auto_all``:
# nobody types an underscore at a prompt.
_APPROVAL_VALUES: dict[str, str] = {
    "manual": "manual", "auto": "auto",
    "all": "auto_all", "auto_all": "auto_all", "auto-all": "auto_all",
}
_APPROVAL_LABELS: dict[str, str] = {
    "manual": "manual", "auto": "auto (tools)", "auto_all": "auto (everything)",
}


async def handle_chat_command(
    *,
    query: str,
    mode: str,
    thinking: bool,
    streaming: bool,
    batch_mode: bool,
    context_mode: str = "compact",
    enforcement: str = "strict",
    approval_mode: str = "manual",
    set_mode: Callable[[str], None],
    set_thinking: Callable[[bool], None],
    thinking_depth: int | None = None,
    set_thinking_depth: Callable[[int], None] | None = None,
    set_streaming: Callable[[bool], None],
    set_batch_mode: Callable[[bool], None],
    set_context_mode: Callable[[str], None] | None = None,
    set_enforcement: Callable[[str], None] | None = None,
    set_approval_mode: Callable[[str], None] | None = None,
    trust_tool: Callable[[str], None] | None = None,
    untrust_tool: Callable[[str], None] | None = None,
    trusted_tools: Iterable[str] | None = None,
    compact_history: Callable[[], Awaitable[None]] | None = None,
    compact_threshold_setter: Callable[[int], None] | None = None,
    show_ledger: Callable[[], str | None] | None = None,
    approval_manager: Any | None = None,
    agent: Any | None = None,
) -> tuple[bool, str]:
    """Handle slash commands for the interactive loop.

    Returns a tuple (handled, message).
    """
    if not query.startswith("/"):
        return False, ""

    parts = query.split()
    cmd = parts[0].lower()

    def _thinking_label() -> str:
        """Current rung name, falling back to the on/off flag for callers that
        don't pass a depth (tests, embedders using the legacy switch)."""
        if thinking_depth is not None and 0 <= thinking_depth < len(THINKING_DEPTH_LABELS):
            return THINKING_DEPTH_LABELS[thinking_depth]
        return "on" if thinking else "off"

    def _trusted_tools_label() -> str:
        if trusted_tools is None:
            return "none"
        items = sorted(set(trusted_tools))
        return ", ".join(items) if items else "none"

    if cmd == "/help":
        return (
            True,
            "\nCommands:\n"
            "  /mode agent   -> enable autonomous tool mode\n"
            "  /mode plan    -> read-only exploration, then a plan you approve before any work\n"
            "  /mode ask     -> read-only Q&A about the codebase; no edits, no plan\n"
            "  /context compact|full -> compact: aggressive compaction (small models);\n"
            "                           full: keep all tool messages in history (200K+ models)\n"
            "  /enforcement strict|light|off -> guidance-nudge level (verification/safety always on);\n"
            "                           strict: all guidance; light: drop discovery nudge; off: no guidance\n"
            "  /status       -> show current mode\n"
            "  /think off|auto|quick|medium|deep|max -> reasoning depth; auto (default) lets the\n"
            "                           model calibrate per turn, the rest impose a fixed budget\n"
            "  /stream on|off -> enable or disable streaming mode\n"
            "  /batch on|off -> batch all write approvals until end of turn\n"
            "  /approvals manual|auto|all -> who answers the approval cards;\n"
            "                           manual: you do; auto: sensitive tools pass, leaving the\n"
            "                           workspace still asks; all: nothing asks (guardrails still apply)\n"
            "  /servers [on|off <name>] -> list/toggle MCP servers (hide their tools from the LLM)\n"
            "  /skills [on|off <name>]  -> list/toggle skills (eligible for auto-detection)\n"
            "  /nudges [on|off <name>]  -> list/toggle application nudges (extension packs)\n"
            "  /trust <tool> -> auto-approve a sensitive tool for this session\n"
            "  /untrust <tool> -> remove session auto-approval for a tool\n"
            "  /compact      -> summarize and compress conversation history\n"
            "  /compact threshold <N> -> set auto-compaction threshold (default 10)\n"
            "  /ledger       -> expand the verification ledger of the last answer\n"
            "  /undo         -> revert all file changes made in the last agent turn\n"
            "  /modules [refresh|<term>] -> environment-module catalogue: index status,\n"
            "                           force a rebuild, or search it directly\n"
            "  /resources    -> list attachable MCP resources (use @<uri> to attach one to a query)\n"
            "  @<path>[:a-b] -> attach a workspace file (or lines a-b) to a query, e.g. @src/foo.py:10-20\n"
            "  quit          -> exit\n",
        )

    if cmd == "/resources":
        resources = getattr(agent, "resources", None) or {} if agent is not None else {}
        file_hint = "You can also attach a workspace file with @<path>[:a-b], e.g. @src/foo.py:10-20."
        if not resources:
            return True, f"\nNo MCP resources are declared by connected servers.\n{file_hint}\n"
        lines = ["\nAttachable MCP resources (reference with @<uri> in your message):"]
        for uri, info in sorted(resources.items()):
            name = info.get("name", "")
            desc = info.get("description", "")
            label = f"  {uri}"
            if name and name != uri:
                label += f"  ({name})"
            if desc:
                label += f"  — {desc}"
            lines.append(label)
        lines.append(file_hint)
        return True, "\n".join(lines) + "\n"

    if cmd == "/status":
        think_status = _thinking_label()
        stream_status = "on" if streaming else "off"
        batch_status = "on" if batch_mode else "off"
        trusted_list = _trusted_tools_label()
        return (
            True,
            f"\nCurrent mode: {mode} | "
            f"thinking: {think_status} | streaming: {stream_status} | batch: {batch_status} | "
            f"context: {context_mode} | enforcement: {enforcement} | "
            f"approvals: {_APPROVAL_LABELS.get(approval_mode, approval_mode)} | "
            f"trusted: {trusted_list}\n",
        )

    if cmd == "/mode":
        if len(parts) == 1:
            return True, f"\nCurrent mode: {mode}\n"
        try:
            set_mode(parts[1])
            return True, f"\nSwitched mode to: {parts[1].strip().lower()}\n"
        except ValueError as exc:
            return True, f"\n❌ {exc}\n"

    if cmd == "/think":
        if len(parts) == 1:
            return True, f"\nThinking is currently: {_thinking_label()}\n"
        level = thinking_depth_from_label(parts[1])
        if level is None:
            return True, f"\n❌ Usage: /think {'|'.join(THINKING_DEPTH_LABELS)}\n"
        if set_thinking_depth is not None:
            set_thinking_depth(level)
        else:
            set_thinking(level > 0)
        return True, f"\nThinking depth set to {THINKING_DEPTH_LABELS[level]}.\n"

    if cmd == "/stream":
        if len(parts) == 1:
            return True, f"\nStreaming is currently: {'on' if streaming else 'off'}\n"
        val = parts[1].lower()
        if val not in ("on", "off"):
            return True, "\n❌ Usage: /stream on|off\n"
        set_streaming(val == "on")
        return True, f"\nStreaming mode turned {val}.\n"
    
    if cmd == "/batch":
        if len(parts) == 1:
            return True, f"\nBatch mode is currently: {'on' if batch_mode else 'off'}\n"
        val = parts[1].lower()
        if val not in ("on", "off"):
            return True, "\n❌ Usage: /batch on|off\n"
        set_batch_mode(val == "on")
        return True, f"\nBatch approval mode turned {val}.\n"

    if cmd == "/approvals":
        if len(parts) == 1:
            return True, (
                f"\nApprovals are currently: {_APPROVAL_LABELS.get(approval_mode, approval_mode)}  "
                "(manual=you answer every card; auto=sensitive tools pass, leaving the "
                "workspace still asks; all=nothing asks)\n"
            )
        val = _APPROVAL_VALUES.get(parts[1].lower())
        if val is None:
            return True, "\n❌ Usage: /approvals manual|auto|all\n"
        if set_approval_mode is not None:
            try:
                set_approval_mode(val)
            except ValueError as exc:
                return True, f"\n❌ {exc}\n"
        if val == "manual":
            return True, "\nApprovals switched to: manual — every card comes back to you.\n"
        return True, (
            f"\n⚠ Approvals switched to: {_APPROVAL_LABELS[val]} — "
            f"{'nothing' if val == 'auto_all' else 'no sensitive tool'} will ask you again "
            "this session. Guardrails (denylist, sandbox) still apply.\n"
        )

    if cmd == "/context":
        if len(parts) == 1:
            return True, f"\nContext mode is currently: {context_mode}  (compact=aggressive compaction, full=keep all tool messages)\n"
        val = parts[1].lower()
        if val not in ("compact", "full"):
            return True, "\n❌ Usage: /context compact|full\n"
        if set_context_mode is not None:
            try:
                set_context_mode(val)
            except ValueError as exc:
                return True, f"\n❌ {exc}\n"
        return True, f"\nContext mode switched to: {val}.\n"

    if cmd == "/enforcement":
        if len(parts) == 1:
            return True, (
                f"\nEnforcement is currently: {enforcement}  "
                "(strict=all guidance nudges, light=drop discovery nudge, off=no guidance; "
                "verification/safety always on)\n"
            )
        val = parts[1].lower()
        if val not in ("strict", "light", "off"):
            return True, "\n❌ Usage: /enforcement strict|light|off\n"
        if set_enforcement is not None:
            try:
                set_enforcement(val)
            except ValueError as exc:
                return True, f"\n❌ {exc}\n"
        return True, f"\nEnforcement level switched to: {val}.\n"

    if cmd == "/trust":
        if len(parts) < 2:
            current = _trusted_tools_label()
            return True, f"\nSession-trusted tools: {current}\nUsage: /trust <tool_name>\n"
        if trust_tool is not None:
            trust_tool(parts[1])
        return True, f"\n✅ '{parts[1]}' auto-approved for this session.\n"

    if cmd == "/untrust":
        if len(parts) < 2:
            return True, "\n❌ Usage: /untrust <tool_name>\n"
        if untrust_tool is not None:
            untrust_tool(parts[1])
        return True, f"\n🔒 '{parts[1]}' removed from session trust list.\n"

    if cmd in ("/servers", "/skills", "/nudges"):
        kind = {"/servers": "servers", "/skills": "skills", "/nudges": "nudges"}[cmd]
        if agent is None:
            return True, f"\n❌ {cmd} is not available.\n"
        rows = agent.toggles_state().get(kind, [])
        toggle = getattr(agent, {
            "servers": "set_server_enabled",
            "skills": "set_skill_enabled",
            "nudges": "set_nudge_enabled",
        }[kind])

        if len(parts) == 1:
            lines = [f"\n{kind.capitalize()} (✓ = active, ✗ = hidden):"]
            for r in rows:
                mark = "✓" if r["enabled"] else "✗"
                desc = f"  — {r['description']}" if r.get("description") else ""
                lines.append(f"  [{mark}] {r['name']}{desc}")
            lines.append(f"\nUsage: {cmd} on|off <name>")
            return True, "\n".join(lines) + "\n"

        if len(parts) >= 3 and parts[1].lower() in ("on", "off"):
            enabled = parts[1].lower() == "on"
            name = parts[2]
            known = {r["name"] for r in rows}
            if name not in known:
                return True, f"\n❌ Unknown {kind[:-1]}: {name}\n"
            toggle(name, enabled)
            return True, f"\n{'✓ enabled' if enabled else '✗ disabled'} {kind[:-1]}: {name}\n"

        return True, f"\n❌ Usage: {cmd} on|off <name>\n"

    if cmd == "/ledger":
        block = show_ledger() if show_ledger is not None else None
        if not block:
            return True, "\nNo verification ledger yet — nothing was written or left open.\n"
        return True, block

    if cmd == "/undo":
        if approval_manager is None:
            return True, "\n❌ Undo is not available (no approval manager attached).\n"
        messages: list[str] = []
        reverted = approval_manager.revert_last(output_fn=messages.append)
        if reverted:
            summary = "\n".join(messages)
            return True, f"\n{summary}\n\n↩  Reverted {len(reverted)} file(s).\n"
        return True, "\n" + (messages[0] if messages else "Nothing to undo.") + "\n"

    if cmd == "/compact":
        # Sub-command: /compact threshold N — adjust auto-compaction threshold.
        if len(parts) >= 3 and parts[1].lower() == "threshold":
            try:
                n = int(parts[2])
                if n < 1:
                    raise ValueError
            except ValueError:
                return True, "\n❌ Usage: /compact threshold <N> (N must be a positive integer)\n"
            if compact_threshold_setter is not None:
                compact_threshold_setter(n)
                return True, f"\n⚡ Auto-compact threshold set to {n} exchanges.\n"
            return True, "\n❌ Threshold setter not available.\n"

        if compact_history is None:
            return True, "\n❌ Compact not available.\n"

        await compact_history()
        return True, ""

    if cmd == "/modules":
        return True, await _modules_command(agent, parts[1:])

    return True, "\n❌ Unknown command. Type /help.\n"


async def _call_platform_tool(agent: Any, tool: str, arguments: dict) -> dict | None:
    """Call one read-only platform tool directly, bypassing the guardrail pipeline.

    Legitimate here because the user invoked it themselves and the tool cannot write:
    the approval/observation machinery exists to judge what the *model* asked for.
    Returns the decoded payload, or None when the tool is not connected.
    """
    tool_owner = getattr(agent, "tool_owner", None) or {} if agent is not None else {}
    if tool not in tool_owner:
        return None
    session = agent.sessions[tool_owner[tool]]
    result = await session.call_tool(tool, arguments)
    try:
        return json.loads(agent._normalize_tool_content(result))
    except (TypeError, ValueError):
        return None


def _format_module_hits(payload: dict) -> str:
    modules = payload.get("modules") or []
    if not modules:
        return "\nNo module matched.\n"
    lines = [f"\n{len(modules)} of {payload.get('total_indexed', '?')} indexed modules:"]
    for entry in modules:
        mark = " (default)" if entry.get("default") else ""
        line = f"  {entry.get('load', '')}{mark}"
        description = (entry.get("description") or "").strip()
        if description:
            line += f"  — {description}"
        lines.append(line)
    catalogue = payload.get("catalogue") or {}
    if catalogue.get("enriching"):
        lines.append("\n  (descriptions are still being indexed in the background)")
    lines.append("\nLoad one with: module load <name>")
    return "\n".join(lines) + "\n"


def _format_catalogue_status(payload: dict) -> str:
    if not payload.get("indexed"):
        return (f"\nModule catalogue: not indexed.\n  module system: "
                f"{payload.get('module_system', '?')}\n  {payload.get('note', '')}\n")
    digest = payload.get("digest") or {}
    lines = [
        "\nModule catalogue:",
        f"  host          : {payload.get('hostname', '?')}",
        f"  module system : {payload.get('module_system', '?')}",
        f"  entries       : {payload.get('count', 0)}",
        f"  detail level  : {payload.get('tier', '?')}"
        + (" (partial)" if payload.get("partial") else ""),
        f"  built         : {payload.get('built_at', '?')}",
        f"  up to date    : {'yes' if payload.get('signal_fresh') else 'no — rebuilds on next search'}",
    ]
    if payload.get("enriching"):
        lines.append("  enriching     : in progress")
    node_types = digest.get("node_types") or []
    if node_types:
        lines.append(f"  node types    : {len(node_types)} distinct hardware signatures")
    lines.append("\n/modules refresh to rebuild, /modules <term> to search.")
    return "\n".join(lines) + "\n"


async def _modules_command(agent: Any, args: list[str]) -> str:
    """/modules — status, /modules refresh — rebuild, /modules <term> — search."""
    if agent is None:
        return "\n❌ /modules needs a connected agent.\n"

    if not args:
        payload = await _call_platform_tool(agent, "platform_catalogue_status", {})
        if payload is None:
            return ("\n❌ The platform server is not connected, so there is no module "
                    "catalogue. Enable it with: /servers on platform\n")
        return _format_catalogue_status(payload)

    if args[0].lower() == "refresh":
        payload = await _call_platform_tool(
            agent, "platform_search", {"query": "*", "limit": 1, "refresh": True})
        if payload is None:
            return ("\n❌ The platform server is not connected, so there is no module "
                    "catalogue. Enable it with: /servers on platform\n")
        catalogue = payload.get("catalogue") or {}
        return (f"\nRebuilt: {payload.get('total_indexed', 0)} modules indexed "
                f"(detail level: {catalogue.get('tier', '?')})."
                + ("\nDescriptions are being filled in in the background.\n"
                   if catalogue.get("enriching") else "\n"))

    term = " ".join(args)
    payload = await _call_platform_tool(agent, "platform_search", {"query": term, "limit": 20})
    if payload is None:
        return ("\n❌ The platform server is not connected, so there is no module "
                "catalogue. Enable it with: /servers on platform\n")
    if payload.get("module_system") == "none":
        return f"\n{payload.get('note', 'No module system on this host.')}\n"
    return _format_module_hits(payload)
