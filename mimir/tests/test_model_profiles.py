"""Model-profile-driven knobs: the tool-count cap and the thinking mechanism.

The weak-model accommodations (the ~40-tool cap) are per-model settings, not hard
globals, so the same codebase scales from Devstral-24B to a 400B-class model on the
B300s.

Tool-call and reasoning parsers are deliberately absent: they are flags on the user's
own ``vllm serve`` command, not something MIMIR sends or needs to know.
"""
import os
import unittest

from mimir.client.config import models, constants
from mimir.client.query_engine.backends import vllm_backend


class ModelKnobsTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop("MIMIR_MAX_TOOLS", None)
        self._added: list[str] = []

    def tearDown(self):
        for k in self._added:
            models.VLLM_MODEL_PROFILES.pop(k, None)
        os.environ.pop("MIMIR_MAX_TOOLS", None)

    def test_max_tools_uncapped_via_profile(self):
        models.VLLM_MODEL_PROFILES["bigmodel-x"] = {"max_tools": 0}
        self._added.append("bigmodel-x")
        self.assertEqual(constants.max_tools_for("bigmodel-x"), 0)  # 0 = uncapped

    def test_max_tools_custom_via_profile(self):
        models.VLLM_MODEL_PROFILES["midmodel-y"] = {"max_tools": 120}
        self._added.append("midmodel-y")
        self.assertEqual(constants.max_tools_for("midmodel-y"), 120)

    def test_max_tools_default_when_no_profile(self):
        self.assertEqual(constants.max_tools_for("qwen3:8b"), constants.MAX_TOOLS_PER_QUERY)

    def test_env_override_beats_profile(self):
        os.environ["MIMIR_MAX_TOOLS"] = "7"
        models.VLLM_MODEL_PROFILES["bigmodel-z"] = {"max_tools": 0}
        self._added.append("bigmodel-z")
        # Explicit env override takes precedence over the profile's max_tools.
        self.assertEqual(constants.max_tools_for("bigmodel-z"), constants.MAX_TOOLS_PER_QUERY)


class ThinkingMechanismTest(unittest.TestCase):
    """Which switch each served model gets for its reasoning.

    ``profile_for_model`` matches the longest key that is a *prefix* of the path, its
    last segment, or any segment -- not a substring. The default matters more than any
    single entry: the served model name is whatever ``--served-model-name`` says, so an
    unlisted name is the normal case and must still get ``enable_thinking`` rather than
    silently losing its reasoning.
    """

    def test_unlisted_models_still_get_the_thinking_kwarg(self):
        for model in ("qwen3:30b", "mistral-small-3.2", "some-finetune-nobody-listed", ""):
            with self.subTest(model=model):
                self.assertEqual(models.thinking_mechanism(model), "kwarg")

    def test_gpt_oss_uses_reasoning_effort(self):
        for model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
            with self.subTest(model=model):
                self.assertEqual(models.thinking_mechanism(model), "effort")

    def test_nemotron_ultra_uses_a_system_directive(self):
        # Its template has no thinking kwarg at all -- only the trained-on string.
        self.assertEqual(
            models.thinking_mechanism("nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"),
            "directive",
        )

    def test_unknown_mechanism_falls_back_to_the_default(self):
        models.VLLM_MODEL_PROFILES["weird-model"] = {"thinking": "telepathy"}
        self.addCleanup(models.VLLM_MODEL_PROFILES.pop, "weird-model", None)
        self.assertEqual(models.thinking_mechanism("weird-model"), "kwarg")

    def test_shipped_profiles_only_declare_a_thinking_mechanism(self):
        """The file is a thinking table, not a per-model tuning dump.

        max_tools / enforcement stay supported for anyone who needs them, but nothing
        ships with them: a shipped opinion about one model's tool budget outlives the
        model it was measured on.
        """
        for key, profile in models.VLLM_MODEL_PROFILES.items():
            with self.subTest(key=key):
                self.assertLessEqual(set(profile), {
                    "thinking", "thinking_directive", "effort_param", "effort_levels",
                    "toggle_param", "toggle_values", "note",
                })

    def test_every_shipped_entry_explains_itself(self):
        """An entry is a documented exception, so it has to carry the documentation.

        A family added on a hunch is worse than no entry at all: the default already
        works, and a wrong mechanism costs that model its reasoning silently.
        """
        for key, profile in models.VLLM_MODEL_PROFILES.items():
            with self.subTest(key=key):
                self.assertTrue(str(profile.get("note", "")).strip(), "entry has no note")

    def test_a_family_key_covers_both_version_spellings(self):
        """Publishers write the same family as Llama-3_1-... and Llama-3.1-...

        Matching the raw spelling silently missed half a family, which is the exact
        failure the profile exists to prevent.
        """
        for name in (
            "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
            "nvidia/Llama-3.1-Nemotron-Ultra-253B-v1",
            "nvidia/Llama-3_1-Nemotron-Nano-8B-v1",
            "nvidia/Llama-3.3-Nemotron-Super-49B-v1",
        ):
            with self.subTest(model=name):
                self.assertEqual(models.thinking_mechanism(name), "directive")

    def test_plain_llama3_is_not_swept_into_the_nemotron_family(self):
        # The family keys must stay narrow: stock Llama 3 takes the default kwarg.
        self.assertEqual(models.thinking_mechanism("meta/llama3-70b"), "kwarg")
        self.assertEqual(models.thinking_mechanism("meta-llama/Llama-3.1-8B-Instruct"), "kwarg")

    def test_directive_models_declare_their_strings(self):
        """`thinking: "directive"` without the strings would be a silent no-op."""
        for key, profile in models.VLLM_MODEL_PROFILES.items():
            if profile.get("thinking") == "directive":
                with self.subTest(key=key):
                    directive = profile.get("thinking_directive")
                    self.assertIsInstance(directive, dict)
                    self.assertTrue(directive.get("on"))
                    self.assertTrue(directive.get("off"))


class ThinkingExtraBodyTest(unittest.TestCase):
    """What actually reaches vLLM for each mechanism.

    Nothing is forwarded blindly from the profile: ``max_tools`` / ``enforcement`` are
    client knobs and would be bogus sampling params.
    """

    def test_kwarg_mechanism_sends_enable_thinking_both_ways(self):
        on = vllm_backend._thinking_extra_body("qwen3:30b", True, {})
        off = vllm_backend._thinking_extra_body("qwen3:30b", False, {})
        self.assertIs(on["chat_template_kwargs"]["enable_thinking"], True)
        # Explicitly False, not omitted: most templates default the kwarg to True.
        self.assertIs(off["chat_template_kwargs"]["enable_thinking"], False)

    def test_budget_rides_along_only_when_thinking(self):
        on = vllm_backend._thinking_extra_body("qwen3:30b", True, {"thinking_budget": 4096})
        self.assertEqual(on["chat_template_kwargs"]["thinking_budget"], 4096)
        off = vllm_backend._thinking_extra_body("qwen3:30b", False, {"thinking_budget": 4096})
        self.assertNotIn("thinking_budget", off["chat_template_kwargs"])

    def test_effort_mechanism_sends_reasoning_effort_and_no_kwargs(self):
        body = vllm_backend._thinking_extra_body("openai/gpt-oss-120b", True, {})
        self.assertNotIn("chat_template_kwargs", body)
        self.assertIn(body["reasoning_effort"], ("low", "medium", "high"))

    def test_effort_ladder_climbs_with_the_budget(self):
        levels = ["low", "medium", "high"]
        self.assertEqual(vllm_backend._effort_level(levels, 512), "low")
        self.assertEqual(vllm_backend._effort_level(levels, 4096), "medium")
        self.assertEqual(vllm_backend._effort_level(levels, 16384), "high")
        # -1 is "model-chosen"/unlimited, which an effort scale can only call middling.
        self.assertEqual(vllm_backend._effort_level(levels, -1), "medium")

    def test_a_family_gets_its_own_ladder_not_openai_s(self):
        """The rung sent must be one the template accepts.

        DeepSeek-V4 and GLM take low/high/max; sending them OpenAI's "medium" lands on
        the template's fallback, silently ignoring the level the user picked.
        """
        for model in ("deepseek-v4", "GLM-5.3-Flash"):
            with self.subTest(model=model):
                levels = models.thinking_profile(model)["levels"]
                self.assertNotIn("medium", levels)
                for budget in (512, 4096, 16384, -1):
                    body = vllm_backend._thinking_extra_body(
                        model, True, {"thinking_budget": budget})
                    self.assertIn(body["reasoning_effort"], levels)

    def test_a_family_with_a_toggle_switches_reasoning_right_off(self):
        off = vllm_backend._thinking_extra_body("deepseek-v4", False, {})
        self.assertEqual(off["thinking_mode"], "chat")
        # The rung is meaningless with reasoning off, so it is not sent.
        self.assertNotIn("reasoning_effort", off)
        on = vllm_backend._thinking_extra_body("deepseek-v4", True, {})
        self.assertEqual(on["thinking_mode"], "thinking")

    def test_a_family_that_always_reasons_reports_no_off(self):
        # GLM's template always emits its Reasoning Effort line; an "off" rung in the
        # panel would be a control that does nothing.
        self.assertFalse(models.thinking_can_disable("GLM-5.3-Flash"))
        self.assertTrue(models.thinking_can_disable("deepseek-v4"))
        self.assertTrue(models.thinking_can_disable("qwen3:30b"))

    def test_sentinel_budgets_never_reach_the_template(self):
        """-1 and 0 mean "model-chosen"/unlimited, not a token count."""
        for sentinel in (-1, 0):
            with self.subTest(budget=sentinel):
                ctk = vllm_backend._thinking_extra_body(
                    "qwen3:30b", True, {"thinking_budget": sentinel})["chat_template_kwargs"]
                self.assertNotIn("thinking_budget", ctk)

    def test_directive_mechanism_sends_nothing_in_the_body(self):
        # The switch is a line in the system message; the template has no kwarg.
        body = vllm_backend._thinking_extra_body(
            "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1", True, {}
        )
        self.assertEqual(body, {})

    def test_client_only_knobs_never_reach_the_request(self):
        models.VLLM_MODEL_PROFILES["knobby"] = {
            "max_tools": 12, "enforcement": "strict", "thinking": "kwarg",
        }
        self.addCleanup(models.VLLM_MODEL_PROFILES.pop, "knobby", None)
        body = vllm_backend._thinking_extra_body("knobby", True, {})
        flat = {**body, **body.get("chat_template_kwargs", {})}
        for key in ("max_tools", "enforcement", "thinking", "thinking_directive"):
            self.assertNotIn(key, flat)


class TemplateKwargFallbackTest(unittest.TestCase):
    """A template that rejects the kwarg must cost one 400, not every turn."""

    class _Boom(Exception):
        status_code = 400

        def __str__(self) -> str:
            return "400: chat template does not accept enable_thinking"

    def setUp(self):
        vllm_backend._NO_TEMPLATE_KWARGS.clear()
        self.addCleanup(vllm_backend._NO_TEMPLATE_KWARGS.clear)

    def _client(self, fail_with_kwargs: bool):
        calls: list[dict] = []

        class _Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        calls.append(kwargs)
                        ctk = kwargs.get("extra_body", {}).get("chat_template_kwargs", {})
                        if fail_with_kwargs and "enable_thinking" in ctk:
                            raise TemplateKwargFallbackTest._Boom()
                        return "response"

        return _Client(), calls

    def _kwargs(self):
        return {"model": "picky", "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}

    def test_retries_without_the_kwarg_and_remembers(self):
        client, calls = self._client(fail_with_kwargs=True)
        self.assertEqual(vllm_backend._create(client, self._kwargs()), "response")
        self.assertEqual(len(calls), 2)  # one rejected, one retried
        self.assertIn("picky", vllm_backend._NO_TEMPLATE_KWARGS)

        # Second turn: the kwarg is dropped up front, so no wasted round trip.
        client, calls = self._client(fail_with_kwargs=True)
        self.assertEqual(vllm_backend._create(client, self._kwargs()), "response")
        self.assertEqual(len(calls), 1)

    def test_structural_kwargs_survive_the_retry(self):
        client, calls = self._client(fail_with_kwargs=True)
        kwargs = self._kwargs()
        kwargs["extra_body"]["chat_template_kwargs"]["continue_final_message"] = True
        vllm_backend._create(client, kwargs)
        # Dropping continue_final_message would 400 the request it was there to fix.
        retried = calls[-1]["extra_body"]["chat_template_kwargs"]
        self.assertEqual(retried, {"continue_final_message": True})

    def test_a_healthy_model_is_never_flagged(self):
        client, calls = self._client(fail_with_kwargs=False)
        vllm_backend._create(client, self._kwargs())
        self.assertEqual(len(calls), 1)
        self.assertEqual(vllm_backend._NO_TEMPLATE_KWARGS, set())

    def test_unrelated_errors_propagate(self):
        class _Other(Exception):
            status_code = 500

        class _Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise _Other("upstream exploded")

        with self.assertRaises(_Other):
            vllm_backend._create(_Client(), self._kwargs())
        self.assertEqual(vllm_backend._NO_TEMPLATE_KWARGS, set())


class ThinkingDirectiveTest(unittest.TestCase):
    """Llama-3.1-Nemotron-Ultra has no thinking kwarg -- only a system-prompt string.

    Its chat template falls back to ``detailed thinking on`` only when there is *no*
    system message, so mimir's own system prompt silently disabled reasoning until the
    directive was injected into it.
    """

    ULTRA = "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"

    def test_ultra_matches_its_own_key_not_plain_llama3(self):
        profile = models.profile_for_model(self.ULTRA)
        self.assertEqual(
            profile.get("thinking_directive"),
            {"on": "detailed thinking on", "off": "detailed thinking off"},
        )
        # And it must NOT fall back to the kwarg mechanism, whose kwarg it ignores.
        self.assertEqual(models.thinking_mechanism(self.ULTRA), "directive")

    def test_plain_llama3_keeps_no_directive(self):
        self.assertEqual(models.profile_for_model("meta/llama3-70b").get("thinking_directive"), None)

    def test_directive_prepended_to_system_message(self):
        msgs = [{"role": "system", "content": "# Identity"}, {"role": "user", "content": "hi"}]
        out = vllm_backend._apply_thinking_directive(
            msgs, models.profile_for_model(self.ULTRA)["thinking_directive"], True
        )
        self.assertEqual(out[0]["content"], "detailed thinking on\n\n# Identity")
        self.assertEqual(out[1], msgs[1])
        self.assertEqual(msgs[0]["content"], "# Identity")  # input not mutated

    def test_toggling_does_not_stack_directives(self):
        directives = models.profile_for_model(self.ULTRA)["thinking_directive"]
        msgs = [{"role": "system", "content": "# Identity"}]
        on = vllm_backend._apply_thinking_directive(msgs, directives, True)
        off = vllm_backend._apply_thinking_directive(on, directives, False)
        self.assertEqual(off[0]["content"], "detailed thinking off\n\n# Identity")

    def test_system_message_synthesised_when_absent(self):
        out = vllm_backend._apply_thinking_directive(
            [{"role": "user", "content": "hi"}],
            models.profile_for_model(self.ULTRA)["thinking_directive"],
            True,
        )
        self.assertEqual(out[0], {"role": "system", "content": "detailed thinking on"})

    def test_models_without_the_knob_are_untouched(self):
        msgs = [{"role": "system", "content": "# Identity"}]
        self.assertIs(vllm_backend._apply_thinking_directive(msgs, {}, True), msgs)


if __name__ == "__main__":
    unittest.main()
