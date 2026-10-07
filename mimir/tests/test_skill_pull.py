"""Loading a skill mid-run: who may, how often, and what keeps it in the context.

A skill used to enter the prompt once, before the first step, chosen by a classifier
model call from a query that does not yet know what the task is made of. Now the index
of loadable skills is in the system prompt and the model pulls one with ``load_skill``
at the step where its own reading tells it which method the work needs.

That moves the body from ``messages[0]`` — which nothing evicts, and which no query may
rewrite mid-run without costing the prompt prefix for every remaining step — to a tool
result in the tail, which three separate passes of the context budget would otherwise
carry away. So the pin is the load-bearing part of the feature, and most of this file is
about it: the model has been told the method is in its context, and a pull that can
silently disappear is worse than no pull at all.
"""
import json
import types
import unittest

from mimir.client.config.constants import SKILL_LOAD_TOOL, SKILL_PULL_MAX
from mimir.client.guardrails.policy.gates import _check_skill_pull
from mimir.client.query_engine import dispatch as dispatch_module
from mimir.client.query_engine import history as history_module
from mimir.client.query_engine.history import (
    _force_fit_to_window,
    _maybe_compact_intra_query,
    _trim_tool_history,
    reconcile_tool_pairs,
)


def _agent(disabled: set[str] = frozenset()) -> types.SimpleNamespace:
    """The three things the gate reads off an agent, and nothing else."""
    skills = {
        "fix-bug": {"description": "repair a defect", "model_invocable": True},
        "write-tests": {"description": "write tests", "model_invocable": True},
        "explore-repo": {"description": "read a codebase", "model_invocable": True},
        "refactor-code": {"description": "restructure", "model_invocable": True},
        "prepare-pr": {"description": "open a PR", "model_invocable": False},
    }
    agent = types.SimpleNamespace(
        skills=skills,
        tool_caps={},
        skill_enabled=lambda name: name not in disabled,
        _json_error_payload=lambda message, hint="", **kw: json.dumps(
            {"status": "error", "error": message, "hint": hint}
        ),
    )
    agent.model_invocable_skills = lambda: [
        (name, meta["description"]) for name, meta in sorted(skills.items())
        if agent.skill_enabled(name) and meta["model_invocable"]
    ]
    return agent


def _refusal(payload: str | None) -> dict:
    assert payload is not None, "expected a refusal, the call was allowed"
    return json.loads(payload)


class SkillPullGateTests(unittest.TestCase):
    """The five refusals, and the reservation that makes the cap exact."""

    def test_a_first_pull_is_allowed_and_reserves_its_slot(self) -> None:
        context: dict = {}
        self.assertIsNone(
            _check_skill_pull(_agent(), SKILL_LOAD_TOOL, {"name": "fix-bug"}, context)
        )
        # "pending", not the call id: the result has not come back yet. What the
        # reservation buys is a cap that two parallel pulls in one step cannot both
        # pass with one slot left.
        self.assertEqual(context["skills_loaded"], {"fix-bug": "pending"})

    def test_another_tool_is_not_this_gate_s_business(self) -> None:
        context: dict = {}
        self.assertIsNone(
            _check_skill_pull(_agent(), "read_file_lines", {"name": "fix-bug"}, context)
        )
        self.assertNotIn("skills_loaded", context)

    def test_the_same_skill_twice_is_refused_pointing_at_the_earlier_result(self) -> None:
        context: dict = {}
        _check_skill_pull(_agent(), SKILL_LOAD_TOOL, {"name": "fix-bug"}, context)
        refusal = _refusal(
            _check_skill_pull(_agent(), SKILL_LOAD_TOOL, {"name": "fix-bug"}, context)
        )
        self.assertIn("already loaded", refusal["error"])
        self.assertIn("earlier tool result", refusal["hint"])

    def test_an_unknown_skill_is_refused_with_the_ones_on_offer(self) -> None:
        refusal = _refusal(
            _check_skill_pull(_agent(), SKILL_LOAD_TOOL, {"name": "nope"}, {})
        )
        self.assertIn("No skill named 'nope'", refusal["error"])
        self.assertIn("fix-bug", refusal["hint"])

    def test_a_skill_the_operator_switched_off_answers_as_if_it_did_not_exist(self) -> None:
        """Deliberately the same answer as "unknown", not "disabled".

        A soft-hidden skill is absent from the index the model was given. Telling it
        the skill exists but is switched off describes a capability it cannot have and
        invites it to stop and ask the user to switch it back on.
        """
        refusal = _refusal(_check_skill_pull(
            _agent(disabled={"write-tests"}),
            SKILL_LOAD_TOOL, {"name": "write-tests"}, {},
        ))
        self.assertIn("No skill named 'write-tests'", refusal["error"])
        self.assertNotIn("disabled", refusal["error"].lower())
        self.assertNotIn("write-tests", refusal["hint"])

    def test_a_user_invoked_only_skill_names_its_slash_command(self) -> None:
        refusal = _refusal(
            _check_skill_pull(_agent(), SKILL_LOAD_TOOL, {"name": "prepare-pr"}, {})
        )
        self.assertIn("user-invoked only", refusal["error"])
        self.assertIn("/prepare-pr", refusal["hint"])
        # And never "ask the user to run it": a model told to stop and request a slash
        # command does exactly that, which is a handback nobody asked for.
        self.assertIn("do not", refusal["hint"].lower())

    def test_the_cap_refuses_the_next_one_and_names_what_is_loaded(self) -> None:
        context: dict = {}
        agent = _agent()
        for name in ("fix-bug", "write-tests", "explore-repo")[:SKILL_PULL_MAX]:
            self.assertIsNone(
                _check_skill_pull(agent, SKILL_LOAD_TOOL, {"name": name}, context)
            )
        refusal = _refusal(
            _check_skill_pull(agent, SKILL_LOAD_TOOL, {"name": "refactor-code"}, context)
        )
        self.assertIn(str(SKILL_PULL_MAX), refusal["error"])
        self.assertIn("fix-bug", refusal["error"])


class PinRecordingTests(unittest.TestCase):
    """What the dispatcher does with a pull's result."""

    @staticmethod
    def _pin(result: str, context: dict, skill: str = "fix-bug") -> None:
        dispatch_module._pin_loaded_skill(
            _agent(), SKILL_LOAD_TOOL, {"name": skill}, "call-1", result, context,
        )

    def test_a_successful_pull_is_promoted_and_pinned(self) -> None:
        context: dict = {"skills_loaded": {"fix-bug": "pending"}}
        self._pin(json.dumps({"status": "ok", "instructions": "method"}), context)
        self.assertEqual(context["skills_loaded"], {"fix-bug": "call-1"})
        self.assertEqual(context["pinned_call_ids"], {"call-1"})

    def test_a_refused_pull_releases_the_slot_it_reserved(self) -> None:
        """A refusal must not charge the budget of the thing it refused.

        Otherwise a mistyped name, or a skill the file declares user-invoked only,
        spends one of the three loads the model is allowed.
        """
        context: dict = {"skills_loaded": {"fix-bug": "pending"}}
        self._pin(json.dumps({"status": "error", "error": "no"}), context)
        self.assertEqual(context["skills_loaded"], {})
        self.assertNotIn("call-1", context.get("pinned_call_ids", set()))

    def test_a_result_for_any_other_tool_pins_nothing(self) -> None:
        context: dict = {}
        dispatch_module._pin_loaded_skill(
            _agent(), "read_file_lines", {"name": "fix-bug"}, "call-9", "ok", context,
        )
        self.assertNotIn("skills_loaded", context)
        self.assertNotIn("pinned_call_ids", context)


def _history_with_pinned_body(body_size: int = 400) -> list[dict]:
    """A history where one of two same-sized tool results is a pinned skill body."""
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do the work"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "pinned", "function": {"name": SKILL_LOAD_TOOL,
                                          "arguments": '{"name": "fix-bug"}'}},
            {"id": "plain", "function": {"name": "read_file_lines",
                                         "arguments": '{"path": "a.py"}'}},
        ]},
        {"role": "tool", "tool_call_id": "pinned", "content": "M" * body_size},
        {"role": "tool", "tool_call_id": "plain", "content": "P" * body_size},
        {"role": "assistant", "content": "thinking"},
        {"role": "user", "content": "carry on"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "recent", "function": {"name": "read_file_lines",
                                          "arguments": '{"path": "b.py"}'}},
        ]},
        {"role": "tool", "tool_call_id": "recent", "content": "R" * 50},
    ]


class PinSurvivesTrimTests(unittest.TestCase):
    def test_the_pinned_result_outlives_an_unpinned_sibling_of_equal_size(self) -> None:
        messages = _history_with_pinned_body()
        context = {"pinned_call_ids": {"pinned"}, "tool_msg_files": {}}
        _trim_tool_history(messages, char_budget=100, execution_context=context)
        kept = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
        self.assertIn("pinned", kept)
        self.assertNotIn("plain", kept)

    def test_without_the_pin_the_same_message_is_evicted(self) -> None:
        """The control: the pin is what saves it, not its position or its size."""
        messages = _history_with_pinned_body()
        _trim_tool_history(messages, char_budget=100,
                           execution_context={"tool_msg_files": {}})
        kept = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
        self.assertNotIn("pinned", kept)


class PinSurvivesCompactionTests(unittest.TestCase):
    """The pass that would carry the body off without the eviction path ever running."""

    @staticmethod
    def _compact(_middle: list[dict]) -> list[dict]:
        return [{"role": "assistant", "content": "[Context summary of 3 prior exchanges]"}]

    def test_the_body_survives_the_summary_and_stays_paired(self) -> None:
        messages = _history_with_pinned_body(body_size=2000)
        context = {"pinned_call_ids": {"pinned"}}
        _maybe_compact_intra_query(messages, "sys", context, self._compact,
                                   token_counter=len, token_budget=100)
        bodies = [m for m in messages if m.get("tool_call_id") == "pinned"]
        self.assertEqual(len(bodies), 1)
        self.assertTrue(bodies[0]["content"].startswith("M"))

        # Pairing, not mere presence: reconcile_tool_pairs drops any tool message not
        # immediately preceded by the assistant turn declaring its id, so a body that
        # survived the summary alone would be deleted by the very next repair pass —
        # protection that looks like it works and does nothing.
        repaired = reconcile_tool_pairs(messages)
        kept = [m for m in repaired if m.get("tool_call_id") == "pinned"]
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0]["content"].startswith("M"))

    def test_an_unpinned_sibling_of_the_same_turn_is_summarised_away(self) -> None:
        messages = _history_with_pinned_body(body_size=2000)
        _maybe_compact_intra_query(messages, "sys", {"pinned_call_ids": {"pinned"}},
                                   self._compact, token_counter=len, token_budget=100)
        self.assertEqual(
            [m for m in messages
             if m.get("tool_call_id") == "plain" and m["content"].startswith("P")],
            [],
        )


class PinIsNeverHarderThanTheWindowTests(unittest.TestCase):
    """The last resort: ordered last, never protected."""

    # Sizes are measured the way the backstop measures them — the wire form of each
    # message, envelope included — so a target here means what it means in production.
    @staticmethod
    def _total(messages: list[dict]) -> int:
        return sum(history_module._message_tokens(m, len) for m in messages)

    def test_the_pinned_body_is_cut_only_after_everything_else(self) -> None:
        messages = _history_with_pinned_body(body_size=400)
        before = messages[3]["content"]
        # A target the unpinned content alone can satisfy: cutting the sibling result
        # frees more than the overshoot, so nothing should reach the pinned body.
        target = self._total(messages) - 400
        fitted = _force_fit_to_window(messages, target, len, pinned_call_ids={"pinned"})
        self.assertTrue(fitted)
        self.assertEqual(messages[3]["content"], before)
        self.assertLess(len(messages[4]["content"]), 400)

    def test_largest_first_would_have_cut_it_first(self) -> None:
        """The control, and the reason this is not simply left alone.

        The reduction order is largest-first, and a skill body is the largest single
        message in the list — so doing nothing here would have truncated the one thing
        the model was promised, before anything else.
        """
        messages = _history_with_pinned_body(body_size=400)
        _force_fit_to_window(messages, self._total(messages) - 400, len)
        self.assertLess(len(messages[3]["content"]), 400)

    def test_a_pin_never_manufactures_an_overflow(self) -> None:
        """Protecting it outright could only turn a tight window into a hard failure.

        Here the pinned body is the only thing left that can close the gap. The
        backstop must cut it and report a fit, not preserve it and raise.
        """
        messages = _history_with_pinned_body(body_size=4000)
        target = 800  # below everything but the irreducible core
        fitted = _force_fit_to_window(messages, target, len, pinned_call_ids={"pinned"})
        self.assertTrue(fitted)
        self.assertLessEqual(self._total(messages), target)
        self.assertLess(len(messages[3]["content"]), 4000)


if __name__ == "__main__":
    unittest.main()
