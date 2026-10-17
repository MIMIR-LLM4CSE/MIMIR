"""Drive the step loops directly, with the keyword arguments a test has at hand."""
from __future__ import annotations

from typing import Any

from mimir.client.query_engine.loop_base import TurnContext
from mimir.client.query_engine.turn import run_loops


def _turn(*, agent: Any, query: str, messages: list[dict], execution_context: dict,
          max_steps: int = 0, thinking: bool = False, streaming: bool = False,
          logger: Any = None, cb: dict | None = None) -> TurnContext:
    return TurnContext(
        agent=agent, query=query, messages=messages, execution_context=execution_context,
        max_steps=max_steps, thinking=thinking, streaming=streaming, logger=logger,
        callbacks=dict(cb or {}),
    )


def _system_of(messages: list[dict]) -> str:
    return messages[0].get("content", "") if messages and messages[0].get("role") == "system" else ""


async def run_plan(*, resume_decision: bool = False, **kw: Any) -> str:
    """The turn run from plan mode, as far as its mode switches take it."""
    turn = _turn(**kw)
    return await run_loops(turn, "plan", _system_of(turn.messages),
                           resume_decision=resume_decision)


async def run_agent(*, active_mode: str = "agent", system_content: str | None = None,
                    **kw: Any) -> str:
    """The turn run from agent (or ask) mode, as far as its mode switches take it."""
    turn = _turn(**kw)
    system = _system_of(turn.messages) if system_content is None else system_content
    return await run_loops(turn, active_mode, system)


def as_loop_run(fake: Any):
    """A ``StepLoop.run`` that hands the loop's state to *fake*, as keyword arguments —
    for a test that checks what a mode switch handed over without running the loop."""
    async def run(self) -> Any:
        return await fake(agent=self.agent, query=self.turn.query, active_mode=self.mode,
                          messages=self.messages, system_content=self.system_content,
                          execution_context=self.ec, max_steps=self.turn.max_steps)
    return run
