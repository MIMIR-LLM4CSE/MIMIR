"""A backend for any endpoint speaking the OpenAI chat API: a vLLM server, or a router
in front of vLLM engines (Ray Serve). What differs between them is declared by an
:class:`Endpoint`; the request builder, the reasoning profiles, the tool-call
normalisation and the streaming read are this module's, once.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

from .base import ChatTurn, LLMBackend, PromptTooLongError, env_positive_int
from ..protocol.tool_call import normalize_arguments
# The assistant↔tool pairing invariant belongs to the history, not to this provider:
# LLMBackend.chat applies it before every call. Re-applied in
# _prepare_messages_for_openai because that normalises ids first, which can turn a
# positional match into an id match.
from ..protocol.messages import merge_consecutive_user_messages, reconcile_tool_pairs


# Single source of truth in _shared so the embedding helper (which runs in the
# separate MCP server processes too) applies the exact same TLS policy.
from ...servers._shared.embed import verify_ssl
from ..config.limits import CTX_RESERVED_RATIO, CTX_TOTAL_FULL
from ..config.models import thinking_profile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Endpoint:
    """Where an OpenAI-compatible endpoint is, and how it is configured.

    Each field names an environment variable, so the address can change between runs
    of one process (``--backend`` writes them) without a stale value being cached.
    """

    url_env: str
    key_env: str
    default_url: str
    default_key: str
    # The window to assume when /v1/models publishes none (a router that does not
    # republish vLLM's max_model_len). A fallback, never an override.
    window_env: str
    # Whether the server root answers ``/health`` while the engine is still loading its
    # weights — a local vLLM does, before ``/v1/models``; a router usually does not.
    serves_health: bool = False

    def config(self) -> tuple[str, str]:
        """``(base_url, api_key)``, the URL ending in ``/v1``.

        The openai client appends ``/chat/completions``; a Serve app is often mounted
        under a route prefix, so the suffix is ours to add.
        """
        base_url = os.environ.get(self.url_env, self.default_url)
        api_key = os.environ.get(self.key_env, self.default_key)
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        return base_url, api_key


def probe_models(config: tuple[str, str]) -> tuple[list[dict], str]:
    """``(model objects, failure reason)`` from GET <base_url>/v1/models; reason "" on success.

    Tells "not answering" from "serving nothing". *config* is ``(base_url, api_key)``.
    """
    import httpx

    base_url, api_key = config
    url = base_url.rstrip("/") + "/models"  # base_url already ends with /v1
    headers = {}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(trust_env=False, timeout=5.0, verify=verify_ssl()) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    return [m for m in data.get("data", []) if isinstance(m, dict)], ""


def _fetch_models(config: tuple[str, str]) -> list[dict]:
    """Return the raw model objects from GET <base_url>/v1/models, or [] on error."""
    return probe_models(config)[0]


def list_served_models(config: tuple[str, str]) -> list[str]:
    """Return the model IDs currently served by the endpoint in *config*.

    Used to auto-resolve the model when the agent connects to an already-running
    server and the user did not pick one. Returns [] on any failure.
    """
    return [m["id"] for m in _fetch_models(config) if m.get("id")]


# Cache of served context window (max_model_len) per (endpoint, model id) — avoids
# re-hitting /v1/models on every chat() call. The served window is fixed for a given
# vLLM process, so a process-lifetime cache is safe. Keyed by endpoint too: a vLLM
# server and a Ray Serve router can serve the same model name with different windows,
# and a cache keyed on the name alone would hand one endpoint's window to the other.
_MODEL_LEN_CACHE: dict[tuple[str, str], int] = {}


def served_model_len(model: str, config: tuple[str, str]) -> int | None:
    """Return the served context window (max_model_len) for *model*, or None.

    vLLM reports max_model_len in each /v1/models entry. We use it to send an
    explicit, in-bounds max_tokens so vLLM never auto-computes a negative default
    (which surfaces as "max_tokens must be at least 1, got -N").
    """
    cfg = config
    endpoint = cfg[0]
    if (endpoint, model) in _MODEL_LEN_CACHE:
        return _MODEL_LEN_CACHE[(endpoint, model)]
    discovered: dict[str, Any] = {}
    for m in _fetch_models(cfg):
        mml = m.get("max_model_len")
        if m.get("id"):
            discovered[m["id"]] = mml
        if isinstance(mml, int) and m.get("id"):
            _MODEL_LEN_CACHE[(endpoint, m["id"])] = mml
    result = _MODEL_LEN_CACHE.get((endpoint, model))
    if result is None:
        logger.warning("served_model_len: no window for requested model %r; "
                       "/v1/models reported %r", model, discovered)
    return result


def _answer_max_tokens(mml: int, prompt_tokens: int) -> int:
    """The ``max_tokens`` to request, from the served window and our prompt estimate.

    The margin scales with the estimate (the rendered template is larger than raw JSON)
    and the request is capped at the answer reserve. May be <= 0; the caller clamps.
    """
    margin = max(512, prompt_tokens // 8)
    remaining = mml - prompt_tokens - margin
    return min(remaining, int(mml * CTX_RESERVED_RATIO))


def _unknown_window_answer_tokens() -> int:
    """Answer allocation to request when the server reports no max_model_len.

    The reserve the context budget already sets aside for the answer on the assumed
    window — the one number here that does not pretend to know the real window.
    """
    return max(1, int(CTX_TOTAL_FULL * CTX_RESERVED_RATIO))


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


def _log_cache_usage(usage: Any) -> None:
    """Log prefix-cache hits from ``usage.prompt_tokens_details.cached_tokens`` (best-effort).

    A rate near zero across steps means a varying prefix or prefix caching not enabled
    (``--enable-prefix-caching``).
    """
    if usage is None or not logger.isEnabledFor(logging.INFO):
        return
    try:
        prompt_total = getattr(usage, "prompt_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
        if prompt_total <= 0:
            return
        logger.info(
            "vllm prefix cache: cached=%d prompt_total=%d hit_rate=%.2f completion=%d",
            cached, prompt_total, cached / prompt_total,
            getattr(usage, "completion_tokens", 0) or 0,
        )
    except Exception:
        logger.debug("_log_cache_usage: a best-effort step failed and was skipped", exc_info=True)


# Servers that 400 on `stream_options`. Like `_NO_TEMPLATE_KWARGS`: absorb the one
# rejection, remember it, and never pay it again for this process.
_NO_STREAM_OPTIONS: set[str] = set()


def _create(client, create_kwargs: dict):
    """POST the chat request, retrying once without a kwarg the server rejected.

    ``enable_thinking`` and ``stream_options`` are dropped for a model that refuses them,
    and remembered; structural kwargs are kept. Other failures propagate.
    """
    model = create_kwargs.get("model", "")
    if model in _NO_TEMPLATE_KWARGS:
        _drop_thinking_kwargs(create_kwargs)
    if model in _NO_STREAM_OPTIONS:
        create_kwargs.pop("stream_options", None)
    try:
        return client.chat.completions.create(**create_kwargs)
    except Exception as exc:
        if getattr(exc, "status_code", None) != 400:
            raise
        msg = str(exc).lower()
        if "stream_options" in msg and create_kwargs.pop("stream_options", None) is not None:
            _NO_STREAM_OPTIONS.add(model)
            return client.chat.completions.create(**create_kwargs)
        if "template" not in msg and "kwarg" not in msg:
            raise
        if not _drop_thinking_kwargs(create_kwargs):
            raise
        _NO_TEMPLATE_KWARGS.add(model)
        return client.chat.completions.create(**create_kwargs)


def _effort_level(levels: list[str], budget: int | None) -> str:
    """The rung of *levels* (weakest first) for a thinking budget, by position; unlimited takes the middle."""
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
    """*model*'s ``thinking_directive`` mapping (``{"on", "off"}``), empty unless it uses the directive mechanism."""
    return thinking_profile(model)["directive"]


def _apply_thinking_directive(
    messages: list[dict], directives: dict, thinking: bool
) -> list[dict]:
    """Put the on/off directive first in the system message, replacing any earlier one.

    For templates steered by a trained-on line rather than a kwarg; both strings come from
    the model's profile.
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
    """OpenAI tool_call objects as internal dicts, keeping the provider's ``id`` so replies pair up."""
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
        arguments = normalize_arguments(args_raw)
        entry: dict = {"function": {"name": name, "arguments": arguments}}
        tc_id = tc.id if hasattr(tc, "id") else tc.get("id") if isinstance(tc, dict) else None
        if isinstance(tc_id, str) and tc_id:
            entry["id"] = tc_id
        result.append(entry)
    return result


def _extract_reasoning(obj: Any) -> str | None:
    """The reasoning text of a delta or message (``reasoning`` or ``reasoning_content``), or None."""
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(obj, attr, None)
        if val:
            return val
    return None


def _prepare_messages_for_openai(messages: list[dict]) -> list[dict]:
    """Internal history in the strict OpenAI chat schema, pairing repaired (``reconcile_tool_pairs``)."""
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
                fn = found if isinstance(found := tc_dict.get("function"), dict) else {}
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

    return merge_consecutive_user_messages(reconcile_tool_pairs(prepared))




# /tokenize statuses that mean "try again later", not "this endpoint has none".
_TOKENIZE_RETRYABLE = frozenset({408, 425, 429, 502, 503, 504})


class OpenAICompatBackend(LLMBackend):
    """An OpenAI-compatible chat endpoint. A subclass names its :class:`Endpoint`."""

    endpoint: Endpoint

    def __init__(self) -> None:
        super().__init__()
        # One OpenAI client (hence one httpx connection pool) per endpoint, reused
        # across calls. Built per call, each one left its socket for the GC to close:
        # a sub-agent process running dozens of turns piled up CLOSE-WAIT connections
        # to the server. Guarded because sub-agents call chat() from their own threads.
        self._clients: dict[tuple[str, str], Any] = {}
        self._clients_lock = threading.Lock()
        # Plain httpx pool for /tokenize, shared by every count (see _tokenize_text).
        self._tokenize_http: Any = None
        # API roots that answered /tokenize with a definitive refusal (see
        # _tokenize_text). Keyed by root, since _config may point elsewhere later.
        self._tokenize_absent: set[str] = set()

    def _config(self) -> tuple[str, str]:
        """The (base_url, api_key) this backend talks to — everything below asks for
        the pair rather than reading an environment variable itself."""
        return self.endpoint.config()

    def readiness_urls(self) -> list[str]:
        """``/v1/models`` — the request the agent actually depends on — then, where
        the server has one, ``/health``, which answers while the weights still load."""
        base_url = self._config()[0].rstrip("/")
        urls = [base_url + "/models"]
        if self.endpoint.serves_health:
            urls.append(base_url[: -len("/v1")] + "/health")
        return urls

    def probe_models(self) -> tuple[list[dict], str]:
        """This endpoint's /v1/models answer and why it failed (see :func:`probe_models`)."""
        return probe_models(self._config())

    def served_models(self) -> list[str]:
        return list_served_models(self._config())

    def served_model_info(self) -> list[dict]:
        return [m for m in _fetch_models(self._config()) if m.get("id")]

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
        """The served window: the endpoint's published ``max_model_len``, else its ``window_env``.

        The fallback is for routers that publish none; it never overrides a published value.
        """
        return served_model_len(model, self._config()) or env_positive_int(
            self.endpoint.window_env)

    def _tokenize_text(self, model: str, text: str) -> int:
        """Exact token count through ``/tokenize`` at the API root.

        A failure falls back to the heuristic in ``count_text_tokens``. A definitive refusal is
        remembered per root so later counts skip the network; timeouts are retried.
        ``MIMIR_VLLM_TOKENIZE=0`` turns the endpoint off.
        """
        import httpx

        if os.environ.get("MIMIR_VLLM_TOKENIZE", "").strip().lower() in ("0", "false", "off", "no"):
            raise RuntimeError("/tokenize disabled by MIMIR_VLLM_TOKENIZE")
        base_url, api_key = self._config()
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        if root in self._tokenize_absent:
            raise RuntimeError(f"{root} serves no /tokenize")
        url = root + "/tokenize"
        headers = {}
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        # add_special_tokens=False: we sum per-message counts, so we don't want
        # BOS/EOS added to each fragment inflating the total.
        payload = {"model": model, "prompt": text, "add_special_tokens": False}
        with self._clients_lock:
            if self._tokenize_http is None:
                self._tokenize_http = httpx.Client(
                    trust_env=False, timeout=5.0, verify=verify_ssl()
                )
            client = self._tokenize_http
        resp = client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400 and resp.status_code not in _TOKENIZE_RETRYABLE:
            self._tokenize_absent.add(root)
        resp.raise_for_status()
        data = resp.json()
        count = data.get("count")
        if isinstance(count, int):
            return count
        tokens = data.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        raise ValueError("unexpected /tokenize response shape")

    @staticmethod
    def _prompt_tokens(usage: Any) -> int:
        """The server's reported prompt size, for the overhead calibration."""
        if usage is None:
            return 0
        return getattr(usage, "prompt_tokens", 0) or 0

    def _complete(self, model: str, messages: list[dict], tools: list[dict],
                  thinking: bool, streaming: bool, options: dict, turn: ChatTurn) -> None:
        client = self._client_for(*self._config())

        # How this model is told to reason (enable_thinking / reasoning_effort /
        # a system-message directive applied further down).
        extra_body: dict = _thinking_extra_body(model, thinking, options)

        # top_k is not an OpenAI-standard sampling param; vLLM accepts it via
        # extra_body. Forward it when a caller requests it, so the constraint is
        # actually applied instead of being silently dropped.
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
        )
        # Sent only when a caller asks for it. Left out, vLLM applies the model's
        # generation_config, i.e. its publisher's sampling. A default here would
        # override that on every agent turn.
        if options.get("temperature") is not None:
            create_kwargs["temperature"] = options["temperature"]
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
        # The window the other budgets use (detection, then the configured window), not the raw
        # /v1/models reading.
        mml = self.context_window(model)
        if mml:
            prompt_text = json.dumps(prepared_messages)
            if tools:
                prompt_text += json.dumps(tools)
            prompt_tokens = self.count_text_tokens(model, prompt_text)
            if prompt_tokens >= mml:
                raise PromptTooLongError(
                    f"Prompt ({prompt_tokens} tokens) exceeds the model's context "
                    f"window ({mml} tokens) for {model!r}. Reduce the conversation "
                    f"(/context compact, /clear) or use a model with a larger window."
                )
            # A caller's max_tokens is a ceiling, not an instruction. Callers set it
            # from what the work needs (a sub-agent caps its own steps), knowing
            # nothing about the window this endpoint serves; sent verbatim, any
            # allocation larger than the window is a 400 rather than a shorter answer.
            # Whichever of the two is smaller is the one that fits.
            sized = _answer_max_tokens(mml, prompt_tokens)
            max_tokens = sized if max_tokens is None else min(int(max_tokens), sized)
        elif max_tokens is None:
            # No window known and none requested: vLLM would default max_tokens to the rest of a
            # possibly huge window. Bound the answer by the unknown-window reserve instead.
            max_tokens = _unknown_window_answer_tokens()
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max(1, int(max_tokens))

        if streaming:
            # A streaming response carries no usage block unless it is asked for. Only
            # requested when the reader is actually listening, so a server that rejects
            # the option costs nothing to anyone who is not measuring — and `_create`
            # retries without it if one does.
            if logger.isEnabledFor(logging.INFO):
                create_kwargs["stream_options"] = {"include_usage": True}
            usage = _read_stream(_create(client, create_kwargs), turn)
        else:
            response = _create(client, create_kwargs)
            usage = getattr(response, "usage", None)
            choice = response.choices[0] if response.choices else None
            if choice:
                turn.stopped(getattr(choice, "finish_reason", None))
                msg = choice.message
                turn.fields["role"] = getattr(msg, "role", "assistant")
                turn.reasoning(_extract_reasoning(msg) or "")
                # Through the tag parser, not a blind tag strip, so inline
                # <think>…</think> lands in the thinking instead of leaking into the
                # content and polluting history.
                turn.content(msg.content or "")
                if getattr(msg, "tool_calls", None):
                    turn.tool_calls.extend(_normalize_tool_calls(msg.tool_calls))
        _log_cache_usage(usage)
        turn.prompt_tokens = self._prompt_tokens(usage)


def _read_stream(response: Any, turn: ChatTurn) -> Any:
    """Hand a streamed completion to *turn*, chunk by chunk. Returns its usage block."""
    usage = None
    # Tool calls arrive as indexed fragments across chunks — merged by index.
    fragments: dict[int, dict] = {}
    for chunk in response:
        turn.check_cancelled()
        # The usage block arrives on a final chunk of its own, with an empty `choices`
        # list — so it is read before the choice guard below.
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        choice = chunk.choices[0] if chunk.choices else None
        # Why the server stopped rides on the choice, not the delta, and arrives on the
        # last chunk (null on every earlier one).
        turn.stopped(getattr(choice, "finish_reason", None) if choice else None)
        delta = choice.delta if choice is not None else None
        if delta is None:
            continue
        if getattr(delta, "role", None):
            turn.fields["role"] = delta.role
        # Native reasoning field (vLLM with a reasoning parser), then regular content —
        # which may carry <think> tags for tag-based models.
        turn.reasoning(_extract_reasoning(delta) or "")
        turn.content(delta.content or "")
        for tc_delta in getattr(delta, "tool_calls", None) or []:
            idx = getattr(tc_delta, "index", 0) or 0
            slot = fragments.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if getattr(tc_delta, "id", None):
                slot["id"] = tc_delta.id
            fn = getattr(tc_delta, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] = fn.name
                slot["arguments"] += getattr(fn, "arguments", None) or ""
    for idx in sorted(fragments):
        slot = fragments[idx]
        if not slot["name"]:
            continue  # malformed/empty fragments
        turn.tool_calls.append({
            # No id streamed: left empty, the history assigns one that is unique across
            # the query (streaming._ensure_unique_call_ids).
            "id": slot["id"],
            # Arguments cut off mid-stream come back marked, and are refused rather
            # than run (protocol.tool_call.UNPARSED_ARGUMENTS).
            "function": {"name": slot["name"], "arguments": normalize_arguments(slot["arguments"])},
        })
    return usage
