"""How much of the context window a conversation uses: the token count of its prompt
and tools (the overhead), and compaction of the middle of its history."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import Any, Callable

from ....llm import token_calibration
from ....llm.factory import get_backend

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
        """Summarize *middle* on the worker's loop and an executor; a future of the result.

        Resolves to *middle* unchanged when summarization failed.
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
        """True once a server-reported prompt size (or a cached one) replaced the estimate.

        Read after :meth:`overhead_tokens`, which records it.
        """
        return self._overhead_measured

    def overhead_tokens(self) -> int:
        """The fixed prompt overhead (system prompt + tools schema) of every call, in tokens.

        Measured as the query builds them (mode, advertised tools); once a server has
        reported, its figure, cached under a fingerprint of the fixed part. Never raises.
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
            mode = agent.mode or ""
            prompt = agent.build_system_content_now(mode)
            tools = agent.advertised_tools_for_mode(mode)
            tools_json = json.dumps(tools) if tools else ""
            key = token_calibration.overhead_key(
                model, agent.context_mode, prompt, tools_json
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
