"""The model calls MIMIR makes on its own behalf (history compaction).

They go through the configured backend and print nothing to the terminal. Skills are
loaded by the model itself; no internal call classifies a query.
"""

import asyncio
import io
import sys
import unittest
from contextlib import redirect_stdout
from typing import Any, Callable

from mimir.client import agent_core
from mimir.client.agent_core import MimirAgent
from mimir.client.llm.base import LLMBackend
from mimir.client.llm.tag_parser import ThinkTagParser

from mimir.tests.support.backend import ScriptedBackend
from mimir.tests.support.agents import fake_agent


class EchoingBackend(LLMBackend):
    """A backend that writes to stdout when no ``token_callback`` is set, like the real ones."""

    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content

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
        parser = ThinkTagParser(token_callback=token_callback)
        parser.feed_content(self._content)
        return {"role": "assistant", "content": "".join(parser.content_parts)}


class _BackendPatch:
    """Point ``get_backend`` at *backend* for the duration of the block."""

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend
        self._originals: list = []

    def __enter__(self) -> LLMBackend:
        from mimir.client import agent_core
        from mimir.client.llm import factory
        # Where it is defined, and where the agent looks it up.
        for module in (factory, agent_core):
            self._originals.append((module, module.get_backend))
            module.get_backend = lambda: self._backend
        return self._backend

    def __exit__(self, *exc: object) -> None:
        for module, original in self._originals:
            module.get_backend = original


_HISTORY = [
    {"role": "user", "content": "add a retry to the fetch helper"},
    {"role": "assistant", "content": "Done — three attempts with backoff."},
]


class CompactHistoryTests(unittest.TestCase):
    def _agent(self) -> Any:
        return fake_agent(model="some-model")

    def test_summary_comes_from_the_configured_backend(self) -> None:
        backend = ScriptedBackend([{"content": "HANDOFF: retry added."}])
        with _BackendPatch(backend):
            summary = asyncio.run(MimirAgent.compact_history(self._agent(), _HISTORY))
        self.assertEqual(summary, "HANDOFF: retry added.")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0]["model"], "some-model")
        # The history to summarise is in the prompt, under a system message.
        roles = [m["role"] for m in backend.calls[0]["messages"]]
        self.assertEqual(roles[0], "system")
        self.assertIn("user", roles)
        self.assertEqual(backend.calls[0]["tools"], [])

    def test_the_summary_is_not_printed(self) -> None:
        buf = io.StringIO()
        with _BackendPatch(EchoingBackend("HANDOFF: retry added.")):
            with redirect_stdout(buf):
                summary = asyncio.run(MimirAgent.compact_history(self._agent(), _HISTORY))
        self.assertEqual(summary, "HANDOFF: retry added.")
        self.assertEqual(buf.getvalue(), "")

    def test_a_backend_failure_yields_no_summary_rather_than_raising(self) -> None:
        class _Down(LLMBackend):
            def chat(self, *a: object, **k: object) -> dict:
                raise RuntimeError("endpoint down")

        with _BackendPatch(_Down()):
            self.assertEqual(
                asyncio.run(MimirAgent.compact_history(self._agent(), _HISTORY)), ""
            )

    def test_empty_history_asks_nothing(self) -> None:
        backend = ScriptedBackend([{"content": "unused"}])
        with _BackendPatch(backend):
            self.assertEqual(asyncio.run(MimirAgent.compact_history(self._agent(), [])), "")
        self.assertEqual(backend.calls, [])


class TokenSinkTests(unittest.TestCase):
    """The sink is the mechanism behind the two "not printed" tests above."""

    def test_internal_calls_pass_a_token_sink(self) -> None:
        backend = ScriptedBackend([{"content": "HANDOFF"}])
        agent = fake_agent(model="m")
        with _BackendPatch(backend):
            asyncio.run(MimirAgent.compact_history(agent, _HISTORY))
        self.assertTrue(backend.calls)
        for call in backend.calls:
            self.assertIs(call["token_callback"], agent_core._discard_token)

    def test_the_sink_swallows_the_token(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertIsNone(agent_core._discard_token("anything"))
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    sys.exit(unittest.main())
