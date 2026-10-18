"""How long, how far and how much a sub-agent may run: budgets, step and token limits."""
from __future__ import annotations


# Wall for one sub-agent run, by what it was given: reading and concluding, or working in
# its own copy. Chosen by the same rung test that gives it a copy.
SUBAGENT_EXPLORE_BUDGET_SECS = 600            # 10 min: read, search, conclude


SUBAGENT_WORKING_BUDGET_SECS = 7200           # 2 h:   edit, build, measure, repeat


SUBAGENT_MIN_BUDGET_SECS = 60


# The longest a BLOCKING sub-agent call can last. The client caps any timeout a tool
# declares at config.limits.TOOL_CALL_TIMEOUT_MAX_SECS (1200s, see
# context.capabilities), so a call cannot hold a child for longer whatever this server
# declares — a longer budget is detached instead (see spawn_agent). The margin below
# that ceiling is what lets this server's own cap fire first and hand back what the
# child had, rather than the dispatcher killing the call with nothing to show.
SUBAGENT_HARD_CAP_SECS = 1140


# What one child may spend in tool-call steps: nothing. A step ceiling cut a long run
# off for a reason unrelated to its difficulty, and the wall that matters is the clock.
# The agent loop reads 0 as "no ceiling" (query_engine.agent_loop).
SUBAGENT_MAX_STEPS = 0


# Ceiling on what one step of a child may generate. The backend otherwise grants the
# model's whole answer reserve, and a run that spends it is a single silent step.
SUBAGENT_ANSWER_TOKENS = 8192


# How much of the window one child may fill (a ceiling under the model's own): twice the
# compact budget for an explorer, the full-mode budget for a working child.
SUBAGENT_CONTEXT_TOKENS_EXPLORE = 64_000


SUBAGENT_CONTEXT_TOKENS_WORKING = 200_000
