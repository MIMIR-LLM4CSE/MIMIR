"""vLLM LLM backend — uses the OpenAI-compatible API served by vLLM."""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from typing import Any, Callable

from .base import LLMBackend
from .tag_parser import ThinkTagParser
# The assistant↔tool pairing invariant belongs to the history, not to this provider:
# history.py owns the single implementation and applies it before every model call.
# Re-applied here because _prepare_messages_for_openai normalises ids first, which can
# turn a positional match into an id match.
from ..history import reconcile_tool_pairs


# Single source of truth in _shared so the embedding helper (which runs in the
# separate MCP server processes too) applies the exact same TLS policy.
from ....servers._shared.embed import verify_ssl


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


# Models whose chat template rejected `chat_template_kwargs` outright, so we stop
# sending them. Keyed by model name; populated by `_create` on the one 400 it takes
# to find out. Most templates ignore a kwarg they don't know, so this stays empty.
_NO_TEMPLATE_KWARGS: set[str] = set()


# The thinking kwargs, as opposed to the structural ones (continue_final_message /
# add_generation_prompt) that a request may genuinely need to be accepted at all.
_THINKING_TEMPLATE_KWARGS = ("enable_thinking", "thinking_budget")


def _drop_thinking_kwargs(create_kwargs: dict) -> bool:
    """Remove the thinking kwargs from the request. True if anything was removed."""
    ctk = create_kwargs.get("extra_body", {}).get("chat_template_kwargs")
    if not ctk:
        return False
    dropped = [ctk.pop(k) for k in _THINKING_TEMPLATE_KWARGS if k in ctk]
    if not ctk:
        create_kwargs["extra_body"].pop("chat_template_kwargs", None)
        if not create_kwargs["extra_body"]:
            create_kwargs.pop("extra_body")
    return bool(dropped)


def _create(client, create_kwargs: dict):
    """POST the chat request, retrying once without the thinking template kwargs.

    Sending `enable_thinking` to every model is what makes reasoning work on a model
    we have no profile for, and a template that doesn't know the kwarg ignores it.
    A rare template errors instead — so absorb that one 400, remember the model, and
    never pay it again. Structural kwargs (`continue_final_message`) are kept: they
    are what makes the request valid in the first place. Any other failure
    propagates untouched.
    """
    model = create_kwargs.get("model", "")
    if model in _NO_TEMPLATE_KWARGS:
        _drop_thinking_kwargs(create_kwargs)
    try:
        return client.chat.completions.create(**create_kwargs)
    except Exception as exc:
        if getattr(exc, "status_code", None) != 400:
            raise
        msg = str(exc).lower()
        if "template" not in msg and "kwarg" not in msg:
            raise
        if not _drop_thinking_kwargs(create_kwargs):
            raise
        _NO_TEMPLATE_KWARGS.add(model)
        return client.chat.completions.create(**create_kwargs)


def _effort_level(levels: list[str], budget: int | None) -> str:
    """Pick a rung from *levels* (weakest first) for a thinking-depth budget.

    The depth ladder is expressed in tokens and an effort ladder is not, so the map
    is by position: the cheapest rung for a small budget, the dearest for a large one,
    the middle for "model-chosen"/unlimited (-1), which is the only reading an effort
    scale can give a budget it cannot express.
    """
    if not levels:
        return ""
    if budget is None or budget < 0:
        return levels[len(levels) // 2]
    if budget <= 512:
        return levels[0]
    if budget <= 4096:
        return levels[len(levels) // 2]
    return levels[-1]


def _thinking_extra_body(model: str, thinking: bool, options: dict) -> dict:
    """Build the request fields that switch *model*'s reasoning on or off.

    Nothing is forwarded blindly from the profile: profile entries are descriptors
    and client-side knobs, and would 400 or mislead if they reached vLLM as request
    params. Only the fields the family actually declares are ever sent.
    """
    from ...config.models import thinking_profile

    profile = thinking_profile(model)
    mechanism = profile["mechanism"]
    # -1 and 0 are our own sentinels for "model-chosen"/"unlimited", not token counts:
    # forwarding them would ask the template for a budget of minus one token.
    budget = options.get("thinking_budget")
    if budget is not None and budget <= 0:
        budget = None

    if mechanism == "effort":
        body: dict = {}
        toggle = profile["toggle_param"]
        if toggle:
            body[toggle] = profile["toggle_values"]["on" if thinking else "off"]
        # The rung is meaningless while reasoning is off, and a family without a
        # toggle cannot be switched off at all — it still gets its weakest rung.
        if thinking or not toggle:
            level = _effort_level(profile["levels"], budget if thinking else 0)
            if level:
                body[profile["effort_param"]] = level
        return body

    if mechanism == "directive":
        # Steered by a line in the system message (see _apply_thinking_directive);
        # its template has no kwarg to set.
        return {}

    # "kwarg" — the default. Send enable_thinking EXPLICITLY so "off" truly disables
    # reasoning: most thinking-capable templates default it to True when omitted.
    ctk: dict = {"enable_thinking": bool(thinking)}
    if thinking and budget is not None:
        ctk["thinking_budget"] = budget
    return {"chat_template_kwargs": ctk}


def _thinking_directives(model: str) -> dict:
    """Return *model*'s ``thinking_directive`` mapping (``{"on": ..., "off": ...}``).

    Most thinking-capable templates take an ``enable_thinking`` kwarg (the default
    ``"kwarg"`` mechanism). A few take nothing at all and are steered purely by a
    literal string in the system message; those declare ``"thinking": "directive"``
    plus the pair of strings, so the kwarg they would ignore is not sent instead.
    Empty when the model uses another mechanism, in which case no message is touched.
    """
    try:
        from ...config.models import thinking_profile
    except ImportError:
        return {}
    return thinking_profile(model)["directive"]


def _apply_thinking_directive(
    messages: list[dict], directives: dict, thinking: bool
) -> list[dict]:
    """Put the on/off directive at the head of the system message.

    For the ``directive`` mechanism: a chat template that takes no thinking kwarg and
    reads a trained-on line from the system message instead. Such templates typically
    apply their own default only when there is *no* system message at all — and MIMIR
    always sends one — so the line has to go inside the message we send, first, which
    is the position it was trained on.

    Both strings come from the model's profile; this function names no model and has
    no default. Any directive left from an earlier turn is stripped first, so toggling
    thinking across a conversation cannot stack contradictory lines.
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
    def __init__(self) -> None:
        super().__init__()
        # One OpenAI client (hence one httpx connection pool) per endpoint, reused
        # across calls. Built per call, each one left its socket for the GC to close:
        # a sub-agent process running dozens of turns piled up CLOSE-WAIT connections
        # to the server. Guarded because sub-agents call chat() from their own threads.
        self._clients: dict[tuple[str, str], Any] = {}
        self._clients_lock = threading.Lock()

    def _client_for(self, base_url: str, api_key: str) -> Any:
        key = (base_url, api_key)
        with self._clients_lock:
            client = self._clients.get(key)
            if client is None:
                from openai import OpenAI
                import httpx

                # trust_env=False so an HTTP proxy (HTTP_PROXY/HTTPS_PROXY)
                # is bypassed — vLLM is on the local cluster network, no proxy needed.
                client = OpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    http_client=httpx.Client(trust_env=False, verify=verify_ssl()),
                )
                self._clients[key] = client
            return client

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
        client = self._client_for(*_get_vllm_config())

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

        # How this model is told to reason (enable_thinking / reasoning_effort /
        # a system-message directive applied further down).
        extra_body: dict = _thinking_extra_body(model, thinking, options)

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
            response = _create(client, create_kwargs)
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
            response = _create(client, create_kwargs)
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
