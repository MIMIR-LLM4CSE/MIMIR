"""Per-query tool-list construction (which tools the model sees each query).

Distinct from the call-time precondition gates: this module decides the *visible*
tool set — query-stable blocking, domain pruning, the relevance cap for small
models, and the plan-mode exploration subset. Kept query-stable so the model
prompt prefix stays byte-identical across a query's steps (vLLM prefix caching).

This is prompt-construction (which tools the model sees *before* it chooses), the
counterpart to ``tool_execution`` which runs the call the model chose *after*. Its
sole consumer is ``query_engine/agent_loop.py``; it moved here out of
``policy/engine.py`` because it is not a precondition gate.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..context.signals import query_prefers_existing_file_edits
from ..context.capabilities import (
    CANDIDATE_SEARCH, CHECK_EXISTENCE, CODE_NAV, CONTENT_WRITE, EDIT, INSPECT_DIR,
    PLAN_BLOCKED, READ, SEARCH, SEARCH_WITH_PATH, names_with_cap,
)
from ..context.signals import DOMAIN_TOOL_GROUPS, query_matches_any
from ..context.execution_context import bootstrap_engine_context as _bootstrap_engine_context
from ...servers._shared import embed as _embed

logger = logging.getLogger(__name__)


def blocked_tools_for_context(
    query: str, execution_context: dict[str, Any] | None, tool_caps: Any = None,
) -> set[str]:
    execution_context = _bootstrap_engine_context(execution_context)

    if execution_context is None:
        return set()

    blocked: set[str] = set()
    if query_prefers_existing_file_edits(query):
        # Appending is effectively creating new content, not a surgical edit.
        blocked.update(names_with_cap(CONTENT_WRITE, tool_caps))

    # Returns only *query-stable* blocks, so the tool list is byte-identical across the
    # steps of one query and the prompt prefix stays cacheable (vLLM prefix caching).
    # State-dependent guards (premature writes / external fetches) are NOT enforced by
    # hiding tools here — they are rejected at call time in evaluate_tool_preconditions.
    return blocked


def inactive_domain_prefixes(
    query: str, rearmed: set[str] | None = None,
) -> tuple[str, ...]:
    """Return tool-name prefixes whose domain group is *not* signaled by *query*.

    Tools whose name starts with one of these prefixes are pruned from the per-step
    tool list (see :func:`tools_for_context`) to keep the prompt small on the common
    case of pure code/workspace tasks. A domain is active as soon as any of its
    keywords appears in the query; the keyword lists in ``DOMAIN_TOOL_GROUPS`` are
    intentionally generous so a task that genuinely needs a domain keeps its tools.
    Returns an empty tuple when every domain is active (nothing to prune).

    *rearmed* holds group keys (see :func:`domain_group_key`) unlocked mid-run by
    :func:`domains_signaled_by_text` — a domain the query never mentioned but the work
    turned out to need. Those groups are treated as active.
    """
    rearmed = rearmed or set()
    inactive: list[str] = []
    for prefixes, keywords in DOMAIN_TOOL_GROUPS:
        if domain_group_key(prefixes) in rearmed:
            continue
        if not query_matches_any(query, keywords):
            inactive.extend(prefixes)
    if inactive:
        logger.debug("domain pruning: inactive prefixes=%s", inactive)
    return tuple(inactive)


def domain_group_key(prefixes: tuple[str, ...]) -> str:
    """Stable identifier for a domain group — its first tool-name prefix."""
    return prefixes[0] if prefixes else ""


def domains_signaled_by_text(query: str, text: str, rearmed: set[str]) -> set[str]:
    """Domain groups that *text* asks for but *query* never activated.

    The per-query tool list is computed once and frozen for prefix-cache stability
    (see :func:`tools_for_context`), so a domain pruned at step 0 stays invisible for
    the rest of the query — even when the work turns out to need it. Scanning what the
    run *produced* (tool results, the model's own prose) against the same
    ``DOMAIN_TOOL_GROUPS`` keywords recovers that case without a second vocabulary.

    Note this only restores *visibility*: pruned tools were always callable, since
    dispatch validates against the full registry (``agent.tool_owner``), not this list.

    Returns the group keys to unlock, excluding those already in *rearmed*.
    """
    if not text or not text.strip():
        return set()
    signaled: set[str] = set()
    for prefixes, keywords in DOMAIN_TOOL_GROUPS:
        key = domain_group_key(prefixes)
        if key in rearmed or query_matches_any(query, keywords):
            continue  # already visible
        if query_matches_any(text, keywords):
            signaled.add(key)
    return signaled


# Capabilities the agent loop structurally depends on across discover→edit→validate.
# Tools carrying any of these (plus todo bookkeeping and spawn_agent) are always
# retained by the relevance cap so trimming the long tail never strips the plumbing.
_CORE_CAPS: tuple[str, ...] = (
    READ, SEARCH, SEARCH_WITH_PATH, CANDIDATE_SEARCH, INSPECT_DIR,
    CHECK_EXISTENCE, CONTENT_WRITE, EDIT, CODE_NAV,
)
_CORE_NAME_PREFIXES: tuple[str, ...] = ("todo_",)
_CORE_NAMES: frozenset[str] = frozenset({"spawn_agent"})


def _relevance_tokens(text: str) -> set[str]:
    """Lowercase word/identifier tokens of length ≥3 used for overlap scoring."""
    return {t for t in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(t) >= 3}


# Tool signatures (name+description) are static, so their embeddings are cached for
# the whole process, keyed by the embedding model so a model switch invalidates them.
_TOOL_EMBED_CACHE: dict[str, list[float]] = {}
_TOOL_EMBED_MODEL: str | None = None


def _tool_signature(fn: dict[str, Any]) -> str:
    return f"{fn.get('name', '')} {fn.get('description', '')}".strip()


def _rank_noncore_semantic(query: str, signatures: list[str]) -> list[int] | None:
    """Rank tool signatures by embedding similarity to *query*, best-first.

    Returns candidate indices (into *signatures*) or ``None`` to fall back to
    lexical overlap. Tool embeddings are cached across the session.
    """
    global _TOOL_EMBED_MODEL
    if not signatures or not _embed.is_available():
        return None

    model = _embed.embed_model_id()
    if model != _TOOL_EMBED_MODEL:
        _TOOL_EMBED_CACHE.clear()
        _TOOL_EMBED_MODEL = model

    missing = [s for s in signatures if s not in _TOOL_EMBED_CACHE]
    if missing:
        vecs = _embed.embed_texts(missing)
        if not vecs:
            return None
        for sig, vec in zip(missing, vecs):
            _TOOL_EMBED_CACHE[sig] = vec

    qvec = _embed.embed_one(query)
    if qvec is None:
        return None
    cand_vecs = [_TOOL_EMBED_CACHE[s] for s in signatures]
    return [pos for pos, _ in _embed.cosine_rank(qvec, cand_vecs)]


def cap_tools_by_relevance(
    tools: list[dict[str, Any]],
    *,
    query: str,
    tool_caps: Any = None,
    max_tools: int | None = None,
) -> list[dict[str, Any]]:
    """Trim the tool list to *max_tools*, keeping the most query-relevant ones.

    Small models degrade when handed too many tools (see MAX_TOOLS_PER_QUERY), so
    when the post-filter list exceeds the cap we keep (1) the agent's core
    discovery/edit/validate/todo plumbing unconditionally, then (2) fill the
    remaining budget with the non-core tools most relevant to the query. Relevance
    is scored by embedding similarity when an embedding backend is available (so it
    survives paraphrase and cross-language queries), else by name+description token
    overlap. Original order is preserved for deterministic prompts; ties break by
    original position. ``max_tools`` of 0/None disables the cap.
    """
    if not max_tools or len(tools) <= max_tools:
        return tools

    core_names: set[str] = set(_CORE_NAMES)
    for cap in _CORE_CAPS:
        core_names |= names_with_cap(cap, tool_caps)

    core_idx: list[int] = []
    noncore_idx: list[int] = []       # original indices of non-core tools
    noncore_sig: list[str] = []       # their name+description signatures (aligned)
    for i, tool in enumerate(tools):
        fn = tool.get("function", {}) or {}
        name = fn.get("name", "")
        if name in core_names or name.startswith(_CORE_NAME_PREFIXES):
            core_idx.append(i)
            continue
        noncore_idx.append(i)
        noncore_sig.append(_tool_signature(fn))

    budget = max(max_tools - len(core_idx), 0)

    # Rank non-core positions best-first: semantic when available, else lexical
    # token overlap (both deterministic on ties via original position).
    ranked = _rank_noncore_semantic(query, noncore_sig)
    if ranked is None:
        ranked = [pos for pos, _ in _embed.lexical_rank(query, noncore_sig)]

    keep_idx = set(core_idx) | {noncore_idx[pos] for pos in ranked[:budget]}
    return [tool for i, tool in enumerate(tools) if i in keep_idx]


def tools_for_context(
    *,
    query: str,
    execution_context: dict[str, Any] | None,
    tools: list[dict[str, Any]],
    tool_caps: Any = None,
    max_tools: int | None = None,
) -> list[dict[str, Any]]:
    blocked = blocked_tools_for_context(query, execution_context, tool_caps)

    # Both remaining filters are query-stable, so the list is identical across the steps
    # of one query and the prompt prefix stays cacheable. The one sanctioned exception
    # is `rearmed_domains`: it changes at most DOMAIN_REARM_MAX_PER_QUERY times, each a
    # deliberate, logged cache break buying a capability the run turned out to need.
    rearmed = (execution_context or {}).get("rearmed_domains") or set()
    inactive_prefixes = inactive_domain_prefixes(query, rearmed)

    if blocked or inactive_prefixes:
        def _keep(tool: dict[str, Any]) -> bool:
            name = tool.get("function", {}).get("name", "")
            if name in blocked:
                return False
            if inactive_prefixes and name.startswith(inactive_prefixes):
                return False
            return True

        tools = [tool for tool in tools if _keep(tool)]

    # Final safeguard: even after context filtering the list can exceed what a
    # small model tolerates, so cap it by per-query relevance.
    return cap_tools_by_relevance(
        tools, query=query, tool_caps=tool_caps, max_tools=max_tools,
    )


def tools_for_readonly_mode(tools: list[dict[str, Any]], tool_caps: Any = None) -> list[dict[str, Any]]:
    """Return the exploration-safe subset of tools allowed in a read-only mode.

    Shared by every mode in ``READONLY_MODES`` (plan, ask). Write, execution, and
    mutation tools (PLAN_BLOCKED) are stripped. Everything else — search, read,
    inspect, platform query, memory read, todo read, and the dual-use PLAN_READONLY
    exec tool (kept visible so the model can run read-only discovery commands; its
    exec use is rejected at call time by
    :func:`readonly_guard.filter_readonly_tool_calls`) — is kept so the model can
    gather the evidence its answer or plan is grounded in.
    """
    plan_blocked = names_with_cap(PLAN_BLOCKED, tool_caps)
    return [
        tool for tool in tools
        if tool.get("function", {}).get("name") not in plan_blocked
    ]


# Historical name, kept because plan_loop imports it and tests monkeypatch it there.
tools_for_plan_mode = tools_for_readonly_mode
