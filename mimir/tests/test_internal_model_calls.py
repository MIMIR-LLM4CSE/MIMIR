"""The model calls MIMIR makes on its own behalf, rather than for the user.

History compaction asks the model something the user never typed and consumes the
answer in code. Two properties follow, and neither held before: the call must go
through the *configured* backend (it used to call ``ollama.chat`` directly, so it
failed under vLLM, Ray or Anthropic), and its output must not reach the terminal —
``LLMBackend.chat`` streams to stdout whenever no ``token_callback`` is given, which
is the CLI answer path's default.

A skill is not one of these calls. The model is given the index of loadable skills in
its system prompt and loads one itself with ``load_skill`` when the work turns out to
need it, so no internal round-trip is spent classifying a query that does not yet know
what the task is made of. ``test_agent_loop.SkillResolutionTests`` is what holds that.
"""

import asyncio
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from typing import Any, Callable

from mimir.client import agent_core
from mimir.client.agent_core import MimirAgent
from mimir.client.query_engine.backends.base import LLMBackend
from mimir.client.query_engine.backends.tag_parser import ThinkTagParser

from _fake_backend import ScriptedBackend


class EchoingBackend(LLMBackend):
    """A backend that reproduces the real ones' stdout contract.

    Both shipped backends push content through :class:`ThinkTagParser`, which
    writes to stdout when no ``token_callback`` is set. Mimicking that here — rather
    than asserting on a recorded argument — is what makes the "nothing is printed"
    tests fail if the sink is ever dropped again.
    """

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
        from mimir.client.query_engine.backends import factory
        self._original = factory.get_backend
        factory.get_backend = lambda: self._backend
        return self._backend

    def __exit__(self, *exc: object) -> None:
        from mimir.client.query_engine.backends import factory
        factory.get_backend = self._original


_HISTORY = [
    {"role": "user", "content": "add a retry to the fetch helper"},
    {"role": "assistant", "content": "Done — three attempts with backoff."},
]


class CompactHistoryTests(unittest.TestCase):
    def _agent(self) -> Any:
        return types.SimpleNamespace(model="some-model")

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
        agent = types.SimpleNamespace(model="m")
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
