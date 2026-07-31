"""Agent-behavior governance: the two guardrail systems + their shared model.

Layout::

    guardrails/
      workflow.py       shared state model (discover→edit→validate→conclude) +
                        its predicates + the agent-loop plan/loop message copy
      observations.py   the execution_context blackboard WRITER (record_tool_
                        observation + _observe_*); run by the executor after each call
      policy/           HARD guardrails — call-time preconditions that BLOCK a tool
                        (engine/write/gates/approval + bash_classify/readonly_exempt
                        + the state_machine guard)
      nudges/           SOFT guardrails — advisory, enforcement-tiered reminders
                        (engine + messages, mirror of policy's plugins registry)

Both subsystems read the foundational ``context`` layer (the ``execution_context``
blackboard, capabilities, signals) and ``config`` (enforcement levels); the
dependency is one-directional (guardrails → context/config, never the reverse).
The only symbol common to BOTH policy and nudges is ``VALIDATION_RETRY_BUDGET``
(a ``config`` constant); the genuine shared substrate is ``execution_context``.
Submodules are imported directly (e.g. ``guardrails.policy.engine``,
``guardrails.nudges``); this package root stays import-light on purpose.
"""
