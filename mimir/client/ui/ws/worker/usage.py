"""How much of the context window a conversation uses: the token count of its prompt
and tools (the overhead), and compaction of the middle of its history."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import Any, Callable

from ....agent_core import build_base_system_content
from ....query_engine import token_calibration
from ....query_engine.backends.factory import get_backend

logger = logging.getLogger(__name__)


class Usage:
    """How much of the window one worker's conversation fills, and compaction of it.

    *model*, *agent* and *loop* are read afresh at each call: the model can be
    switched, and the agent and its loop exist only once the worker has built them.
    """

    def __init__(self, model: Callable[[], str], agent: Callable[[], Any],
                 loop: Callable[[], asyncio.AbstractEventLoop | None]) -> None:
        self._model = model
        self._agent = agent
        self._loop = loop
        # Whether the overhead `overhead_tokens` last returned came from a server's own
        # count. Recorded as it is computed: answering it again needs the system prompt
        # and the tools schema, and the bar asks both questions once a second.
        self._overhead_measured = False

    def compact_middle(self, middle: list) -> Any:
        """Summarize *middle* into one message, off the WS event loop.

        Summarizing is a model call, which must not run on the WebSocket event loop:
        it is scheduled on this worker's loop and handed to an executor thread, since
        ``compact_messages`` blocks on the backend.

        Returns a ``concurrent.futures.Future`` resolving to the summary messages,
        or to *middle* unchanged when summarization failed (``compact_messages``
        swallows its own errors) — the caller treats that as "no compaction".
        """
        agent, loop = self._agent(), self._loop()
        if agent is None or loop is None:
            fut: Any = concurrent.futures.Future()
            fut.set_result(middle)
            return fut

        async def _run() -> list:
            return await asyncio.get_running_loop().run_in_executor(
                None, agent.compact_messages, middle)

        return asyncio.run_coroutine_threadsafe(_run(), loop)

    def count_tokens(self, text: str) -> int:
        """Token count for *text* — exact when the backend has a tokenizer.

        ``allow_network=False``: this runs inline on the streaming path, which must
        not stall on a tokenize round-trip mid-answer; the cached-or-heuristic value
        is close enough for a size readout. Never raises.
        """
        if not text:
            return 0
        try:
            return get_backend().count_text_tokens(self._model(), text, allow_network=False)
        except Exception:
            logger.debug("count_tokens: a best-effort step failed and was skipped", exc_info=True)
            return 0

    def overhead_is_measured(self) -> bool:
        """True once a server-reported prompt size has replaced the estimate.

        Surfaced to the bar's tooltip because the two numbers answer different
        questions: before the first answer the overhead is this client's reading of
        the prompt it is about to send, after it the server's reading of what it
        received. Saying which one is on screen turns a figure that quietly shifts
        after the first turn into one the user can account for.

        Reads what :meth:`overhead_tokens` recorded on its last call, so ask it
        after that one. A figure read back from the calibration cache counts as measured:
        a server did measure it, against this very system prompt and tool set.
        """
        return self._overhead_measured

    def overhead_tokens(self) -> int:
        """Fixed prompt overhead (tokens) sent on *every* LLM call besides history.

        The system prompt plus the tools schema — ~30k tokens with every tool server
        on, most of it the tools. The context bar must include it, otherwise the
        displayed usage hides the very tokens that actually overflow the window.

        Both halves are measured the way the query builds them, not approximated.
        The prompt comes from ``build_system_content_now`` in the current mode, so
        the mode-gated sections, the memory index and the checklist are counted;
        measuring the base doctrine alone under-reported the prompt by everything
        the modes add. The tools come from ``advertised_tools_for_mode``, so a
        disabled server and a read-only mode remove from the bar what they remove
        from the call; the raw registry over-reported them. What stays out of reach
        is ``tools_for_context``, which prunes against the query and therefore does
        not exist before there is one — it only ever removes, so this is a ceiling
        on the tool half and the bar errs high, never low.

        Once a call has come back, none of this is estimated any more: the server
        reports what it actually charged for the prompt, and ``measured_prompt_overhead``
        hands back that figure minus the history it came with. That number replaces the
        estimate for every later call, which is what makes the bar exact rather than
        merely close — and it is why the estimate above only has to carry the session
        as far as its first answer.

        A measured figure outlives the process too: it is written to the calibration
        cache under a fingerprint of the whole fixed part (model, mode, system prompt,
        advertised tools) and read back on the next run. A session reopened against a
        fresh server is therefore accounted for exactly from its first frame rather
        than from its first answer — which is what kept the bar reading over-full on a
        resume. The fingerprint is the safety: change a mode, a server or the prompt
        and the entry is simply not found, so the estimate answers instead of a stale
        measurement wearing its authority.

        Best-effort, cached via the backend's token cache; never raises.
        """
        agent = self._agent()
        if agent is None:
            return 0
        model = self._model()
        try:
            backend = get_backend()
            # getattr, not a direct call: a backend stub without the calibration
            # should fall through to the estimate, not lose the overhead entirely
            # to the except below.
            probe = getattr(backend, "measured_prompt_overhead", None)
            measured = probe(model) if callable(probe) else None
            mode = getattr(agent, "mode", "") or ""
            build = getattr(agent, "build_system_content_now", None)
            prompt = build(mode) if callable(build) else build_base_system_content()
            narrow = getattr(agent, "advertised_tools_for_mode", None)
            tools = narrow(mode) if callable(narrow) else getattr(agent, "tools", None)
            tools_json = json.dumps(tools) if tools else ""
            key = token_calibration.overhead_key(
                model, getattr(agent, "context_mode", "full"), prompt, tools_json
            )
            if measured is not None:
                # Write-through, so the next run of this same configuration starts
                # calibrated. A no-op when the figure is already on disk, which it is
                # for all but the first call after a turn — the bar asks every second.
                token_calibration.remember_overhead(key, measured)
                self._overhead_measured = True
                return measured
            remembered = token_calibration.recall_overhead(key)
            if remembered is not None:
                # Measured by a server too, against this very prompt and tool set.
                self._overhead_measured = True
                return remembered
            self._overhead_measured = False
            total = backend.count_text_tokens(model, prompt, allow_network=False)
            if tools_json:
                total += backend.count_text_tokens(model, tools_json, allow_network=False)
            return total
        except Exception:
            logger.debug("overhead_tokens: a best-effort step failed and was skipped", exc_info=True)
            return 0
