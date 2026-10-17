"""How long, how far and how much a sub-agent may run: budgets, step and token limits."""
from __future__ import annotations


# Wall for one sub-agent run, by what the child was actually given to do. A child that
# only reads answers a question and stops; a child that edits, compiles and measures is
# doing the work this whole apparatus exists for, and ten minutes of it is a run cut off
# for a reason that has nothing to do with the task.
#
# The rung the user set decides which applies, through the same test that decides
# whether the child gets a copy of the repository — so "works in its own copy" and "has
# hours to do it" can never disagree.
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


# How much of the model's window one child may fill, by what the child was given.
# Not the model's whole window: a child exists to keep its reading out of the caller's
# context, and several of them on one endpoint each budgeting for the full window is
# how a fan-out becomes the thing it was meant to avoid. Both numbers are ceilings —
# a model with a smaller window still sizes the budget, and the trimming, eviction and
# compaction all follow from the total.
#
# An explorer reads a handful of files and returns a conclusion, and a sweep wide
# enough to be sure is what it is for — twice the compact budget, so breadth is not
# what makes it hand back a half-answer. A working child is the other case — it reads
# what it must to change code, runs it, and reports what came back — so it gets the
# standing full-mode budget rather than a share of it.
SUBAGENT_CONTEXT_TOKENS_EXPLORE = 64_000


SUBAGENT_CONTEXT_TOKENS_WORKING = 200_000
