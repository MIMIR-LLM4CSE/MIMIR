"""vLLM LLM backend — uses the OpenAI-compatible API served by vLLM."""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Callable

from .base import LLMBackend
from .tag_parser import ThinkTagParser
# The assistant↔tool pairing invariant belongs to the history, not to this provider:
# history.py owns the single implementation and applies it before every model call.
# Re-applied here because _prepare_messages_for_openai normalises ids first, which can
# turn a positional match into an id match.
from ..history import reconcile_tool_pairs


# Single source of truth in _shared so the embedding helper (which runs in the
# separate MCP server processes too) shares the exact same HPC host-resolution.
from ....servers._shared.embed import _resolve_host, verify_ssl


def _get_vllm_config() -> tuple[str, str]:
    """Return (base_url, api_key) from environment / config."""
    import os
    try:
        from ...config.models import VLLM_BASE_URL, VLLM_API_KEY
        base_url = os.environ.get("VLLM_BASE_URL", VLLM_BASE_URL)
        api_key = os.environ.get("VLLM_API_KEY", VLLM_API_KEY)
    except ImportError:
        base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
        api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    # Replace loopback with the real node hostname to bypass HPC proxies.
    base_url = _resolve_host(base_url)
    # The openai client appends /chat/completions to base_url, so it must
    # end with /v1 (e.g. http://<node>:8000/v1).
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    return base_url, api_key


def _fetch_models() -> list[dict]:
    """Return the raw model objects from GET <base_url>/v1/models, or [] on error."""
    import httpx

    base_url, api_key = _get_vllm_config()
    url = base_url.rstrip("/") + "/models"  # base_url already ends with /v1
    headers = {}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(trust_env=False, timeout=5.0, verify=verify_ssl()) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return []
    return [m for m in data.get("data", []) if isinstance(m, dict)]


def list_served_models() -> list[str]:
    """Return the model IDs currently served by the vLLM endpoint.

    Used to auto-resolve the model when the agent connects to an already-running
    server and the user did not pick one. Returns [] on any failure.
    """
    return [m["id"] for m in _fetch_models() if m.get("id")]


# Cache of served context window (max_model_len) per model id — avoids re-hitting
# /v1/models on every chat() call. The served window is fixed for a given vLLM
# process, so a process-lifetime cache is safe.
_MODEL_LEN_CACHE: dict[str, int] = {}


def served_model_len(model: str) -> int | None:
    """Return the served context window (max_model_len) for *model*, or None.

    vLLM reports max_model_len in each /v1/models entry. We use it to send an
    explicit, in-bounds max_tokens so vLLM never auto-computes a negative default
    (which surfaces as "max_tokens must be at least 1, got -N").
    """
    if model in _MODEL_LEN_CACHE:
        return _MODEL_LEN_CACHE[model]
    discovered: dict[str, Any] = {}
    for m in _fetch_models():
        mml = m.get("max_model_len")
        if m.get("id"):
            discovered[m["id"]] = mml
        if isinstance(mml, int) and m.get("id"):
            _MODEL_LEN_CACHE[m["id"]] = mml
    result = _MODEL_LEN_CACHE.get(model)
    if result is None:
        import sys
        print(
            f"[mimir] served_model_len: no window for requested model "
            f"{model!r}; /v1/models reported {discovered!r}",
            file=sys.stderr, flush=True,
        )
    return result


def _answer_max_tokens(mml: int, prompt_tokens: int) -> int:
    """max_tokens to request given the served window and our prompt estimate.

    Our estimate counts raw JSON; vLLM counts the rendered chat template + tool
    schemas, which is larger by an amount that grows with prompt/schema size — a
    fixed margin eventually loses (observed: 513-token undercount at ~17K prompt
    tokens, one over the old 512 margin → 400 "maximum context length"). Scale
    the margin with the estimate, and cap the request at the answer reserve
    instead of claiming the whole remaining window: the loop's context budgeting
    sizes prompts so the answer never legitimately needs more than the reserve.

    May return <= 0 when the prompt estimate is near the window; the call site
    clamps to >= 1 (vLLM rejects non-positive max_tokens outright).
    """
    from ...config.constants import CTX_RESERVED_RATIO
    margin = max(512, prompt_tokens // 8)
    remaining = mml - prompt_tokens - margin
    return min(remaining, int(mml * CTX_RESERVED_RATIO))


# Profile keys that configure how vLLM is *launched* (`vllm serve` flags), read
# straight from vllm_model_profiles.json by the VS Code extension. They must NOT be
# forwarded in a chat request's extra_body, where vLLM has no such parameters.
_LAUNCH_ONLY_PROFILE_KEYS: frozenset[str] = frozenset({
    "tool_call_parser", "reasoning_parser", "async-scheduling",
    # Client-only knobs consumed by the agent loop / config, never valid as vLLM
    # request sampling params — must be stripped from extra_body.
    "pin_role", "max_tools", "thinking_directive",
})


def _model_extra_body(model: str) -> dict:
    """Return the request-time ``extra_body`` settings for *model*.

    Derived from the matching ``VLLM_MODEL_PROFILES`` entry (longest case-insensitive
    prefix, so e.g. ``mistral-medium-3.5:128b`` matches ``mistral``). Launch-only
    flags (see ``_LAUNCH_ONLY_PROFILE_KEYS``) are stripped: they belong to the server
    start-up command, not to per-request bodies. Request-relevant keys such as
    ``supports_thinking`` are kept for the caller to consume.
    """
    try:
        from ...config.models import VLLM_MODEL_PROFILES
        profiles = VLLM_MODEL_PROFILES
    except ImportError:
        return {}

    model_lower = model.lower().replace("\\", "/").strip()
    parts = [p for p in model_lower.split("/") if p]
    candidates: list[str] = [model_lower]
    if parts:
        candidates.append(parts[-1])
        candidates.extend(parts)

    # Pick the longest matching prefix across all candidates.
    best_key = max(
        (
            k
            for k in profiles
            if any(c.startswith(k.lower()) for c in candidates)
        ),
        key=len,
        default=None,
    )
    if best_key is None:
        # No profile match — the parser is a launch-only concern, so there is
        # nothing request-relevant to send.
        return {}
    # Deep-copy so callers can mutate safely, then drop launch-only flags.
    import copy
    profile = copy.deepcopy(profiles[best_key])
    return {k: v for k, v in profile.items() if k not in _LAUNCH_ONLY_PROFILE_KEYS}


def _thinking_directives(model: str) -> dict:
    """Return *model*'s ``thinking_directive`` mapping (``{"on": ..., "off": ...}``), or {}.

    Most thinking-capable templates take an ``enable_thinking`` kwarg (see
    ``supports_thinking``). A few take nothing at all and are steered purely by a
    literal string in the system message — Llama-3.1-Nemotron-Ultra was trained on
    ``detailed thinking on`` / ``detailed thinking off``. Such models need this knob
    instead, and must NOT set ``supports_thinking``, whose kwarg they would ignore.
    """
    try:
        from ...config.models import profile_for_model
    except ImportError:
        return {}
    directives = profile_for_model(model).get("thinking_directive")
    return directives if isinstance(directives, dict) else {}


def _apply_thinking_directive(
    messages: list[dict], directives: dict, thinking: bool
) -> list[dict]:
    """Put the on/off directive at the head of the system message.

    Nemotron-Ultra's template only falls back to its ``detailed thinking on`` default
    when there is *no* system message; ours always replaces it, which silently turns
    reasoning off. So the directive has to be injected into the system message we send,
    as its first line — the position the model was trained on.

    Any directive left over from an earlier turn is stripped first, so toggling
    thinking on and off across a conversation cannot stack contradictory lines.
    """
    wanted = str(directives.get("on" if thinking else "off") or "")
    if not wanted:
        return messages

    variants = [str(v) for v in directives.values() if v]
    out = list(messages)
    for i, msg in enumerate(out):
        if msg.get("role") != "system":
            continue
        content = str(msg.get("content") or "")
        for variant in variants:
            if content.startswith(variant):
                content = content[len(variant):].lstrip("\n")
                break
        out[i] = {**msg, "content": f"{wanted}\n\n{content}" if content else wanted}
        return out
    return [{"role": "system", "content": wanted}, *out]


def _normalize_tool_calls(raw_tool_calls: list) -> list[dict]:
    """Convert OpenAI tool_call objects to Ollama-style dicts.

    The OpenAI tool-call ``id`` is **preserved** when present: the dispatcher keys the
    tool-result message by this id, and on replay `_prepare_messages_for_openai` echoes
    it back. Dropping it (as the non-streaming path used to) forces the dispatcher and
    the replay normalizer to invent two *different* positional ids, so the assistant
    `tool_calls` and the `tool` message no longer pair up and vLLM rejects the history.
    """
    result = []
    for tc in raw_tool_calls:
        # tc may be an object (openai SDK) or a dict (if already normalized)
        if isinstance(tc, dict):
            result.append(tc)
            continue
        fn = tc.function if hasattr(tc, "function") else None
        if fn is None:
            continue
        name = fn.name if hasattr(fn, "name") else fn.get("name", "")
        args_raw = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
        try:
            arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        entry: dict = {"function": {"name": name, "arguments": arguments}}
        tc_id = tc.id if hasattr(tc, "id") else tc.get("id") if isinstance(tc, dict) else None
        if isinstance(tc_id, str) and tc_id:
            entry["id"] = tc_id
        result.append(entry)
    return result


def _extract_reasoning(obj: Any) -> str | None:
    """Return the reasoning/thinking text from a streaming delta or message object.

    vLLM exposes the reasoning parser's output under different field names across
    versions: older builds use ``reasoning_content`` while 0.22+ (matching the newer
    OpenAI schema) uses ``reasoning``. We check both so the thinking block renders
    regardless of the server version. Returns the first non-empty value, else None.
    """
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(obj, attr, None)
        if val:
            return val
    return None


def _prepare_messages_for_openai(messages: list[dict]) -> list[dict]:
    """Convert internal/Ollama-style history to strict OpenAI chat schema.

    vLLM validates chat history strictly: assistant tool calls need
    id/type/function.arguments(string), tool messages must carry a tool_call_id that
    matches a call in the immediately preceding assistant message, and every assistant
    tool call must be answered. This does the per-message shape normalization, then
    `reconcile_tool_pairs` repairs any pairing broken by upstream trimming/compaction.
    """
    prepared: list[dict] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        m = dict(msg)
        role = m.get("role")

        if role == "assistant" and isinstance(m.get("tool_calls"), list):
            normalized_calls: list[dict] = []
            for idx, tc in enumerate(m.get("tool_calls") or []):
                tc_dict = tc if isinstance(tc, dict) else {}
                fn = tc_dict.get("function") if isinstance(tc_dict.get("function"), dict) else {}
                name = fn.get("name", "")

                raw_args = fn.get("arguments", "{}")
                if isinstance(raw_args, str):
                    args_str = raw_args
                else:
                    try:
                        args_str = json.dumps(raw_args if raw_args is not None else {})
                    except (TypeError, ValueError):
                        args_str = "{}"

                call_id = tc_dict.get("id")
                if not isinstance(call_id, str) or not call_id:
                    call_id = f"call_{len(prepared)}_{idx}"

                normalized_calls.append({
                    "id": call_id,
                    "type": tc_dict.get("type") or "function",
                    "function": {
                        "name": name,
                        "arguments": args_str,
                    },
                })

            # Never send an empty tool_calls array — vLLM treats it as malformed.
            if normalized_calls:
                m["tool_calls"] = normalized_calls
            else:
                m.pop("tool_calls", None)

        if role == "tool" and not isinstance(m.get("content"), str):
            try:
                m["content"] = json.dumps(m.get("content"))
            except (TypeError, ValueError):
                m["content"] = str(m.get("content", ""))

        prepared.append(m)

    return _merge_consecutive_user_messages(reconcile_tool_pairs(prepared))


def _merge_consecutive_user_messages(prepared: list[dict]) -> list[dict]:
    """Collapse adjacent plain ``user`` messages into one.

    The Mistral tokenizer (``--tokenizer-mode mistral``, e.g. Devstral) enforces
    strict role alternation and degenerates into token salad when it sees two
    consecutive ``user`` turns. Upstream can legitimately produce them (plan-mode
    nudges, a caller that already appended the current turn). Joining their text
    with a blank line preserves the content while keeping the sequence legal; only
    string-content user messages with no tool fields are merged, so tool pairing is
    untouched.
    """
    out: list[dict] = []
    for m in prepared:
        if (
            out
            and m.get("role") == "user"
            and out[-1].get("role") == "user"
            and isinstance(m.get("content"), str)
            and isinstance(out[-1].get("content"), str)
            and not m.get("tool_calls")
            and not out[-1].get("tool_calls")
        ):
            if m["content"] != out[-1]["content"]:
                out[-1] = {**out[-1], "content": out[-1]["content"] + "\n\n" + m["content"]}
            # identical duplicate → drop entirely
            continue
        out.append(m)
    return out


class VllmBackend(LLMBackend):
    def _fetch_context_window(self, model: str) -> int | None:
        """vLLM context window = the server's reported max_model_len.

        ``MIMIR_VLLM_MAX_MODEL_LEN`` overrides it — an escape hatch for older
        servers whose /v1/models doesn't report max_model_len.
        """
        import os
        env = os.environ.get("MIMIR_VLLM_MAX_MODEL_LEN", "").strip()
        if env.isdigit() and int(env) > 0:
            return int(env)
        return served_model_len(model)

    def _tokenize_text(self, model: str, text: str) -> int:
        """Exact token count via vLLM's /tokenize endpoint.

        /tokenize is served at the API root (sibling of /v1), so we strip the
        /v1 suffix that _get_vllm_config appends for the chat client. Any failure
        propagates to LLMBackend.count_text_tokens, which falls back to the
        chars-per-token heuristic — so a missing endpoint or network blip never
        breaks a budget check.
        """
        import httpx

        base_url, api_key = _get_vllm_config()
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        url = root + "/tokenize"
        headers = {}
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        # add_special_tokens=False: we sum per-message counts, so we don't want
        # BOS/EOS added to each fragment inflating the total.
        payload = {"model": model, "prompt": text, "add_special_tokens": False}
        with httpx.Client(trust_env=False, timeout=5.0, verify=verify_ssl()) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        count = data.get("count")
        if isinstance(count, int):
            return count
        tokens = data.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        raise ValueError("unexpected /tokenize response shape")

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
        from openai import OpenAI
        import httpx

        base_url, api_key = _get_vllm_config()
        # Use trust_env=False so the corporate Squid proxy (HTTP_PROXY/HTTPS_PROXY)
        # is bypassed — vLLM is on the local cluster network, no proxy needed.
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(trust_env=False, verify=verify_ssl()),
        )

        parser = ThinkTagParser(
            token_callback=token_callback,
            think_token_callback=think_token_callback,
            think_start_callback=think_start_callback,
            think_end_callback=think_end_callback,
        )

        # Streaming tool calls arrive as indexed fragments; accumulate by index.
        # Maps index -> {"id": ..., "name": ..., "arguments": ...}
        _streaming_tc_buf: dict[int, dict] = {}
        tool_calls_parts: list[dict] = []
        final_msg: dict = {}

        # Start from the model's profile (tool-call style, etc.) then overlay
        # any call-specific overrides (e.g. thinking_budget).
        extra_body: dict = _model_extra_body(model)
        if extra_body.pop("supports_thinking", False):
            # Send enable_thinking EXPLICITLY so "off" truly disables reasoning: it is
            # the de-facto standard kwarg across thinking-capable vLLM templates (Qwen3,
            # DeepSeek-R1, GLM, Nemotron), and most default it to True when omitted.
            ctk = extra_body.setdefault("chat_template_kwargs", {})
            ctk["enable_thinking"] = bool(thinking)
            if thinking:
                thinking_budget = options.get("thinking_budget")
                if thinking_budget is not None and thinking_budget != 0:
                    ctk["thinking_budget"] = thinking_budget
        else:
            thinking_budget = options.get("thinking_budget")
            if thinking and thinking_budget is not None and thinking_budget != 0:
                ctk = extra_body.setdefault("chat_template_kwargs", {})
                ctk["thinking_budget"] = thinking_budget

        # top_k is not an OpenAI-standard sampling param; vLLM accepts it via
        # extra_body. Forward it when callers (e.g. plan mode) request it so the
        # constraint is actually applied instead of being silently dropped.
        top_k = options.get("top_k")
        if top_k is not None:
            extra_body["top_k"] = top_k

        prepared_messages = _prepare_messages_for_openai(messages)
        # Models steered by a system-prompt directive rather than a template kwarg.
        prepared_messages = _apply_thinking_directive(
            prepared_messages, _thinking_directives(model), thinking
        )

        # vLLM raises 400 if add_generation_prompt=True (the default) when the
        # last message is from the assistant.  Signal that we want to continue
        # the partial assistant turn instead.
        if prepared_messages and prepared_messages[-1].get("role") == "assistant":
            ctk = extra_body.setdefault("chat_template_kwargs", {})
            ctk.setdefault("continue_final_message", True)
            ctk.setdefault("add_generation_prompt", False)

        create_kwargs: dict = dict(
            model=model,
            messages=prepared_messages,
            stream=streaming,
            temperature=options.get("temperature", 0.3),
        )
        if tools:
            create_kwargs["tools"] = tools
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        # Send an explicit max_tokens: vLLM otherwise defaults it to
        # (max_model_len - prompt_tokens), which goes negative near the window and 400s
        # with "max_tokens must be at least 1, got -N". Clamping to >=1 is not enough —
        # when the *prompt* itself overflows, vLLM re-caps to the negative remainder and
        # 400s anyway, so raise a clear error instead. Last-resort net; the agent loop's
        # context budgeting should prevent reaching here.
        max_tokens = options.get("max_tokens") or options.get("num_predict")
        mml = served_model_len(model)
        if mml:
            import json as _json
            prompt_text = _json.dumps(prepared_messages)
            if tools:
                prompt_text += _json.dumps(tools)
            prompt_tokens = self.count_text_tokens(model, prompt_text)
            if prompt_tokens >= mml:
                raise ValueError(
                    f"Prompt ({prompt_tokens} tokens) exceeds the model's context "
                    f"window ({mml} tokens) for {model!r}. Reduce the conversation "
                    f"(/context compact, /clear) or use a model with a larger window."
                )
            if max_tokens is None:
                max_tokens = _answer_max_tokens(mml, prompt_tokens)
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max(1, int(max_tokens))

        if streaming:
            response = client.chat.completions.create(**create_kwargs)
            for chunk in response:
                if cancel_flag is not None and cancel_flag.is_set():
                    raise asyncio.CancelledError("Cancelled by user")

                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # Capture role once
                if getattr(delta, "role", None):
                    final_msg["role"] = delta.role

                # Native reasoning field (vLLM with reasoning parser). The field is
                # named `reasoning_content` on older vLLM and `reasoning` on 0.22+.
                reasoning = _extract_reasoning(delta)
                if reasoning:
                    if think_start_callback is not None and not parser._thinking_started:
                        parser._thinking_started = True
                        think_start_callback()
                    if think_token_callback is not None:
                        think_token_callback(reasoning)
                    parser.thinking_parts.append(reasoning)

                # Regular content — may contain <think> tags for tag-based models
                content = delta.content or ""
                if content:
                    parser.feed_content(content)

                # Tool calls arrive as indexed fragments across chunks — merge by index.
                if getattr(delta, "tool_calls", None):
                    for tc_delta in delta.tool_calls:
                        idx = getattr(tc_delta, "index", 0) or 0
                        if idx not in _streaming_tc_buf:
                            _streaming_tc_buf[idx] = {"id": "", "name": "", "arguments": ""}
                        slot = _streaming_tc_buf[idx]
                        tc_id = getattr(tc_delta, "id", None)
                        if tc_id:
                            slot["id"] = tc_id
                        fn = getattr(tc_delta, "function", None)
                        if fn is not None:
                            fn_name = getattr(fn, "name", None) or ""
                            if fn_name:
                                slot["name"] = fn_name
                            fn_args = getattr(fn, "arguments", None) or ""
                            if fn_args:
                                slot["arguments"] += fn_args

            # Flush accumulated streaming tool call fragments into final list.
            for idx in sorted(_streaming_tc_buf):
                slot = _streaming_tc_buf[idx]
                if not slot.get("name"):
                    continue  # skip malformed/empty fragments
                try:
                    arguments = json.loads(slot["arguments"]) if slot["arguments"] else {}
                except json.JSONDecodeError:
                    arguments = {}
                call_id = slot["id"] or f"call_{idx}"
                tool_calls_parts.append({
                    "id": call_id,
                    "function": {"name": slot["name"], "arguments": arguments},
                })

            # Signal end of thinking if we got any native reasoning
            if parser.thinking_parts and think_end_callback is not None and not parser._in_think:
                think_end_callback()

        else:
            response = client.chat.completions.create(**create_kwargs)
            choice = response.choices[0] if response.choices else None
            if choice:
                msg = choice.message
                final_msg["role"] = getattr(msg, "role", "assistant")

                reasoning = _extract_reasoning(msg)
                if reasoning:
                    parser.thinking_parts.append(reasoning)

                # Through the tag parser, not a blind tag strip, so inline
                # <think>…</think> lands in thinking_parts instead of leaking into
                # final_msg["content"] and polluting history. The parser also echoes the
                # answer to stdout (token_callback is None here).
                content = msg.content or ""
                if content:
                    parser.feed_content(content)

                if getattr(msg, "tool_calls", None):
                    tool_calls_parts.extend(_normalize_tool_calls(msg.tool_calls))

        if parser.content_parts and token_callback is None:
            sys.stdout.write("\n")
            sys.stdout.flush()

        final_msg["content"] = "".join(parser.content_parts)

        if parser.thinking_parts:
            final_msg["thinking"] = "".join(parser.thinking_parts)

        if tool_calls_parts:
            final_msg["tool_calls"] = tool_calls_parts

        return final_msg
