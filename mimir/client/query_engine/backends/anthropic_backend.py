"""Anthropic (Claude API) LLM backend.

Lets MIMIR run against Anthropic's hosted Claude models (Opus/Sonnet/Haiku/Fable)
for evaluation and comparison against the local vLLM/Ollama backends. Selected via
``LLM_BACKEND=anthropic`` (alias ``claude``); the model is the usual
``MIMIR_DEFAULT_MODEL`` (e.g. ``claude-opus-4-8``).

The backend translates MIMIR's internal Ollama-style history (see the module
docstrings in ``ollama_backend`` / ``vllm_backend``) to the Claude Messages API
and back:

  internal message                     Anthropic Messages API
  ─────────────────────────────────    ──────────────────────────────────────
  {"role": "system", ...}              -> top-level ``system`` (concatenated)
  {"role": "user", "content": str}     -> {"role": "user", content: [text]}
  {"role": "assistant", tool_calls}    -> {"role": "assistant", content: [text, tool_use…]}
  {"role": "tool", tool_call_id, ...}  -> {"role": "user", content: [tool_result]}

Thinking-block replay: when adaptive thinking is on, Claude requires the *signed*
thinking blocks be echoed back on the assistant turn that carried a ``tool_use``
(same model only). MIMIR flattens each turn to a plain dict (``thinking`` becomes a
string, losing the signature), so we cache the raw response ``content`` blocks keyed
by their tool-call ids and splice them back in during history conversion. Every
tool-use turn we replay was produced by this backend earlier in the session, so it
is always a cache hit — history stays valid without plumbing signed blocks through
the rest of the engine.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any, Callable

from .base import LLMBackend

logger = logging.getLogger(__name__)

# Cap on the raw-block replay cache. A session won't produce this many assistant
# turns; evicting the oldest entries can only affect very old tool-use turns whose
# thinking blocks the API no longer strictly requires.
_RAW_BLOCK_CACHE_CAP = 4096

# Fallback context windows by model-name substring, used only when the Models API
# lookup fails (older SDK, network blip). Current Claude models are 1M except Haiku.
_FALLBACK_WINDOWS: tuple[tuple[str, int], ...] = (
    ("haiku", 200_000),
    ("opus", 1_000_000),
    ("sonnet", 1_000_000),
    ("fable", 1_000_000),
    ("mythos", 1_000_000),
)


def _get_config() -> dict:
    """Return client kwargs from the environment.

    Credentials resolve the SDK's normal way (``ANTHROPIC_API_KEY``, then
    ``ANTHROPIC_AUTH_TOKEN``, then an ``ant auth login`` profile); a base-URL
    override is honoured via ``ANTHROPIC_BASE_URL``. Unlike the vLLM backend we do
    **not** disable ``trust_env`` — the Claude API is reached over the internet, so
    the corporate ``HTTP(S)_PROXY`` must be respected, not bypassed.
    """
    kwargs: dict = {}
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        kwargs["api_key"] = key
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def _default_max_tokens() -> int:
    env = os.environ.get("MIMIR_ANTHROPIC_MAX_TOKENS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return 16_000


def _effort() -> str | None:
    """Reasoning/output effort (low|medium|high|xhigh|max). Default high.

    ``MIMIR_ANTHROPIC_EFFORT=none`` omits ``output_config`` entirely — an escape
    hatch for legacy models that predate the effort parameter.
    """
    level = os.environ.get("MIMIR_ANTHROPIC_EFFORT", "high").strip().lower()
    if level == "none":
        return None
    return level if level in ("low", "medium", "high", "xhigh", "max") else "high"


def _thinking_enabled(flag: bool) -> bool:
    """Whether to send adaptive thinking. Env can force-disable it.

    ``MIMIR_ANTHROPIC_THINKING=off`` disables thinking regardless of the caller's
    request — an escape hatch for legacy models that reject ``{type: "adaptive"}``.
    """
    if os.environ.get("MIMIR_ANTHROPIC_THINKING", "").strip().lower() in ("off", "0", "false"):
        return False
    return bool(flag)


def _tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert Ollama function-calling tool defs to Claude tool defs.

    Ollama: ``{"type": "function", "function": {"name", "description", "parameters"}}``
    Claude: ``{"name", "description", "input_schema"}``
    """
    out: list[dict] = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        out.append({
            "name": name,
            "description": fn.get("description") or "",
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return out


def _mark_last_block_cacheable(conv: list[dict]) -> None:
    """Add a cache breakpoint to the last content block of the last turn.

    No-op unless the last block is a plain dict we constructed (text / tool_result);
    replayed assistant turns hold raw SDK block objects we must not mutate. Anthropic
    allows 4 breakpoints per request — this is the 2nd (system is the 1st).
    """
    if not conv:
        return
    content = conv[-1].get("content")
    if not isinstance(content, list) or not content:
        return
    last = content[-1]
    if isinstance(last, dict):
        last["cache_control"] = {"type": "ephemeral"}


def _log_cache_usage(final: Any) -> None:
    """Log prompt-cache effectiveness from a response's ``usage`` block.

    ``input_tokens`` is the *uncached* remainder; total prompt =
    input + cache_creation + cache_read. A hit_rate near 1.0 across a query's
    steps means the multi-turn breakpoints are biting. Zero reads across repeated
    steps points to a silent prefix invalidator (a varying system prompt, a
    reordered tool list). Best-effort — never let telemetry break a chat turn.
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    usage = getattr(final, "usage", None)
    if usage is None:
        return
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    created = getattr(usage, "cache_creation_input_tokens", 0) or 0
    fresh = getattr(usage, "input_tokens", 0) or 0
    prompt_total = read + created + fresh
    if prompt_total <= 0:
        return
    hit_rate = read / prompt_total
    logger.info(
        "anthropic cache: read=%d created=%d fresh=%d prompt_total=%d hit_rate=%.2f "
        "output=%d",
        read, created, fresh, prompt_total, hit_rate,
        getattr(usage, "output_tokens", 0) or 0,
    )


class AnthropicBackend(LLMBackend):
    def __init__(self) -> None:
        super().__init__()
        self._client: Any = None
        # tool-call-id signature -> raw response content blocks (with signed
        # thinking), so assistant tool-use turns replay faithfully. See module docs.
        self._raw_assistant_blocks: dict[tuple, list] = {}

    # ── client ───────────────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(**_get_config())
        return self._client

    # ── context window ─────────────────────────────────────────────────────────

    def _fetch_context_window(self, model: str) -> int | None:
        env = os.environ.get("MIMIR_ANTHROPIC_MAX_INPUT_TOKENS", "").strip()
        if env.isdigit() and int(env) > 0:
            return int(env)
        try:
            info = self._get_client().models.retrieve(model)
            win = getattr(info, "max_input_tokens", None)
            if isinstance(win, int) and win > 0:
                return win
        except Exception:
            pass
        low = model.lower()
        for needle, win in _FALLBACK_WINDOWS:
            if needle in low:
                return win
        return None

    # ── token counting ──────────────────────────────────────────────────────────

    def _tokenize_text(self, model: str, text: str) -> int:
        """Exact token count via the Claude count_tokens endpoint.

        Any failure propagates to LLMBackend.count_text_tokens, which falls back to
        the chars-per-token heuristic, so a network blip never breaks a budget check.
        """
        resp = self._get_client().messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
        count = getattr(resp, "input_tokens", None)
        if isinstance(count, int):
            return count
        raise ValueError("unexpected count_tokens response shape")

    # ── message conversion ──────────────────────────────────────────────────────

    @staticmethod
    def _tc_key(tool_calls: list) -> tuple:
        ids = [tc.get("id") for tc in tool_calls if isinstance(tc, dict) and tc.get("id")]
        return tuple(ids)

    def _prepare(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Split internal history into (system_prompt, anthropic_messages).

        Consecutive ``tool`` messages are merged into a single ``user`` turn of
        ``tool_result`` blocks (Claude pairs each result to its ``tool_use`` by id).
        Assistant tool-use turns are replayed from the raw-block cache when available
        so signed thinking blocks survive; otherwise they are reconstructed from the
        flattened text + tool_calls (valid whenever thinking was not used).
        """
        system_parts: list[str] = []
        out: list[dict] = []

        def _flush_tool_run(run: list[dict]) -> None:
            if run:
                out.append({"role": "user", "content": run})

        pending_tool_results: list[dict] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")

            if role == "tool":
                content = msg.get("content")
                if not isinstance(content, str):
                    import json
                    try:
                        content = json.dumps(content)
                    except (TypeError, ValueError):
                        content = str(content)
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id") or "",
                    "content": content or "",
                })
                continue

            # Any non-tool message closes an open run of tool results.
            _flush_tool_run(pending_tool_results)
            pending_tool_results = []

            if role == "system":
                text = msg.get("content")
                if isinstance(text, str) and text.strip():
                    system_parts.append(text)
                continue

            if role == "assistant":
                tool_calls = [tc for tc in (msg.get("tool_calls") or []) if isinstance(tc, dict)]
                cached = self._raw_assistant_blocks.get(self._tc_key(tool_calls)) if tool_calls else None
                if cached is not None:
                    out.append({"role": "assistant", "content": cached})
                    continue
                blocks: list[dict] = []
                text = msg.get("content")
                if isinstance(text, str) and text.strip():
                    blocks.append({"type": "text", "text": text})
                for tc in tool_calls:
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    call_id = tc.get("id")
                    if not call_id:
                        continue
                    args = fn.get("arguments")
                    if not isinstance(args, dict):
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": call_id,
                        "name": fn.get("name", ""),
                        "input": args,
                    })
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
                continue

            # user (and any unknown role) -> plain user text
            text = msg.get("content")
            if isinstance(text, str) and text != "":
                out.append({"role": "user", "content": [{"type": "text", "text": text}]})

        _flush_tool_run(pending_tool_results)
        return "\n\n".join(system_parts), out

    def _cache_blocks(self, raw_content: list) -> None:
        """Cache a response's raw content blocks keyed by its tool-use ids."""
        ids = tuple(
            getattr(b, "id", None)
            for b in raw_content
            if getattr(b, "type", None) == "tool_use" and getattr(b, "id", None)
        )
        if not ids:
            return
        if len(self._raw_assistant_blocks) >= _RAW_BLOCK_CACHE_CAP:
            self._raw_assistant_blocks.clear()
        self._raw_assistant_blocks[ids] = raw_content

    # ── request shaping ─────────────────────────────────────────────────────────

    def _build_kwargs(
        self, model: str, messages: list[dict], tools: list[dict],
        thinking: bool, options: dict,
    ) -> dict:
        system, conv = self._prepare(messages)
        max_tokens = options.get("max_tokens") or options.get("num_predict") or _default_max_tokens()
        kwargs: dict = {
            "model": model,
            "max_tokens": max(1, int(max_tokens)),
            "messages": conv,
        }
        if system:
            # Cache the stable prefix (tools + system render before messages).
            kwargs["system"] = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
        # Incremental multi-turn caching: a second breakpoint on the last content
        # block lets each agent-loop step read the prior turn's conversation prefix
        # from cache instead of re-processing the whole growing history. Only applied
        # to a dict block we built here (replayed assistant turns are raw SDK objects,
        # and are never last anyway — a tool_result/user turn always follows them).
        _mark_last_block_cacheable(conv)
        anth_tools = _tools_to_anthropic(tools)
        if anth_tools:
            kwargs["tools"] = anth_tools
        if _thinking_enabled(thinking):
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        effort = _effort()
        if effort is not None:
            kwargs["output_config"] = {"effort": effort}
        return kwargs

    # ── chat ────────────────────────────────────────────────────────────────────

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        thinking: bool,
        streaming: bool,
        options: dict,
        cancel_flag: Any = None,
        token_callback: Callable[[str], None] | None = None,
        think_token_callback: Callable[[str], None] | None = None,
        think_start_callback: Callable[[], None] | None = None,
        think_end_callback: Callable[[], None] | None = None,
    ) -> dict:
        client = self._get_client()
        create_kwargs = self._build_kwargs(model, messages, tools, thinking, options)

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls_parts: list[dict] = []
        thinking_started = False

        def _on_text(text: str) -> None:
            if not text:
                return
            content_parts.append(text)
            if token_callback is not None:
                token_callback(text)
            else:
                sys.stdout.write(text)
                sys.stdout.flush()

        def _on_thinking(text: str) -> None:
            nonlocal thinking_started
            if not text:
                return
            if not thinking_started:
                thinking_started = True
                if think_start_callback is not None:
                    think_start_callback()
            thinking_parts.append(text)
            if think_token_callback is not None:
                think_token_callback(text)

        if streaming:
            with client.messages.stream(**create_kwargs) as stream:
                for event in stream:
                    if cancel_flag is not None and cancel_flag.is_set():
                        raise asyncio.CancelledError("Cancelled by user")
                    etype = getattr(event, "type", None)
                    if etype == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        dtype = getattr(delta, "type", None)
                        if dtype == "text_delta":
                            _on_text(getattr(delta, "text", "") or "")
                        elif dtype == "thinking_delta":
                            _on_thinking(getattr(delta, "thinking", "") or "")
                final = stream.get_final_message()
            self._collect_final(final, content_parts, thinking_parts, tool_calls_parts, streamed=True)
        else:
            final = client.messages.create(**create_kwargs)
            self._collect_final(final, content_parts, thinking_parts, tool_calls_parts, streamed=False,
                                text_sink=_on_text)

        if thinking_started and think_end_callback is not None:
            think_end_callback()

        if content_parts and token_callback is None:
            sys.stdout.write("\n")
            sys.stdout.flush()

        self._cache_blocks(getattr(final, "content", []) or [])
        _log_cache_usage(final)

        result: dict = {"role": "assistant", "content": "".join(content_parts)}
        if thinking_parts:
            result["thinking"] = "".join(thinking_parts)
        if tool_calls_parts:
            result["tool_calls"] = tool_calls_parts
        return result

    @staticmethod
    def _collect_final(
        final: Any,
        content_parts: list[str],
        thinking_parts: list[str],
        tool_calls_parts: list[dict],
        *,
        streamed: bool,
        text_sink: Callable[[str], None] | None = None,
    ) -> None:
        """Read tool_use (always) and, when not already streamed, text/thinking
        from the final message's content blocks into the accumulators.

        Streaming already routed text/thinking through the delta callbacks; here we
        only need the tool calls. Non-streaming has no deltas, so we pull everything.
        """
        for block in getattr(final, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "tool_use":
                tool_calls_parts.append({
                    "id": getattr(block, "id", "") or "",
                    "function": {
                        "name": getattr(block, "name", "") or "",
                        "arguments": getattr(block, "input", {}) or {},
                    },
                })
            elif not streamed and btype == "text":
                text = getattr(block, "text", "") or ""
                if text_sink is not None:
                    text_sink(text)
                else:
                    content_parts.append(text)
            elif not streamed and btype == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
