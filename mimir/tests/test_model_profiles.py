"""Model-profile-driven knobs: pin_role and the B300-ready tool-count cap.

The weak-model accommodations (the ~40-tool cap, the strict-template pin role)
are per-model settings, not hard globals, so the same codebase scales from
Devstral-24B to a 400B-class model on the B300s.
"""
import os
import unittest

from mimir.client.config import models, constants
from mimir.client.query_engine.backends import vllm_backend


class ModelKnobsTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop("MIMIR_MAX_TOOLS", None)
        os.environ.pop("MIMIR_PIN_ROLE", None)
        self._added: list[str] = []

    def tearDown(self):
        for k in self._added:
            models.VLLM_MODEL_PROFILES.pop(k, None)
        os.environ.pop("MIMIR_MAX_TOOLS", None)

    def test_pin_role_resolution(self):
        self.assertEqual(models.resolve_pin_role("Devstral-Small-2507"), "append_user")
        self.assertEqual(models.resolve_pin_role("qwen3:8b"), "system")
        self.assertEqual(models.resolve_pin_role("qwen3:8b", "user"), "user")  # override wins

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


class ParserResolutionTest(unittest.TestCase):
    """Guards which parser each served model actually gets.

    ``profile_for_model`` matches the longest key that is a *prefix* of the path,
    its last segment, or any segment -- not a substring. Nemotron is the case that
    makes this matter: ``Llama-3_1-Nemotron-Ultra-253B-v1`` must fall through to
    ``llama-3``, because its tokenizer is stock Llama 3.1 and has no ``<think>``
    token, so any BaseThinkingReasoningParser subclass would abort at startup.
    Nemotron 3, by contrast, has real ``<think>``/``</think>`` tokens and emits
    Qwen-style ``<tool_call><function=...>`` XML.
    """

    CASES = {
        "openai/gpt-oss-20b":                     ("openai",     "openai_gptoss"),
        "openai/gpt-oss-120b":                    ("openai",     "openai_gptoss"),
        # gpt-oss reasons via harmony channels, surfaced by the openai_gptoss
        # reasoning parser -- but its template knows `reasoning_effort`, not
        # `enable_thinking`, so it must NOT claim supports_thinking (see below).
        "meta/llama3-70b":                        ("llama3_json", None),
        "mistralai/devstral-small-2-24b":         ("mistral",    None),
        "nvidia/nemotron-3-super:120b":           ("qwen3_coder", "nemotron_v3"),
        "nvidia/nemotron-3-super-bf16:120b":      ("qwen3_coder", "nemotron_v3"),
        "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1": ("llama3_json", None),
        "qwen3-coder:30b-a3b-fp16":               ("qwen3_coder", None),
        "qwen3:30b":                              ("hermes",     "qwen3"),
        "gemma4:31b":                             ("gemma4",     "gemma4"),
    }

    def test_served_models_resolve_to_expected_parsers(self):
        for model, (tool, reasoning) in self.CASES.items():
            with self.subTest(model=model):
                profile = models.profile_for_model(model)
                self.assertEqual(profile.get("tool_call_parser"), tool)
                self.assertEqual(profile.get("reasoning_parser"), reasoning)

    def test_supports_thinking_implies_reasoning_parser(self):
        """``supports_thinking`` is not a label -- it makes the backend send
        ``chat_template_kwargs.enable_thinking`` (vllm_backend ``_stream_chat``).

        Only set it on models whose chat template actually reads that kwarg,
        otherwise the UI offers a thinking toggle that silently does nothing.
        gpt-oss is the counter-example: it reasons (hence ``openai_gptoss``) but
        is steered by ``reasoning_effort``, so the reverse implication is false.
        """
        for key, profile in models.VLLM_MODEL_PROFILES.items():
            if profile.get("supports_thinking"):
                with self.subTest(key=key):
                    self.assertTrue(profile.get("reasoning_parser"))

    def test_gpt_oss_does_not_claim_enable_thinking(self):
        profile = models.profile_for_model("openai/gpt-oss-120b")
        self.assertEqual(profile.get("reasoning_parser"), "openai_gptoss")
        self.assertNotIn("supports_thinking", profile)


class ThinkingDirectiveTest(unittest.TestCase):
    """Llama-3.1-Nemotron-Ultra has no thinking kwarg -- only a system-prompt string.

    Its chat template falls back to ``detailed thinking on`` only when there is *no*
    system message, so mimir's own system prompt silently disabled reasoning until the
    directive was injected into it.
    """

    ULTRA = "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"

    def test_ultra_matches_its_own_key_not_plain_llama3(self):
        profile = models.profile_for_model(self.ULTRA)
        self.assertEqual(profile.get("tool_call_parser"), "llama3_json")
        self.assertEqual(
            profile.get("thinking_directive"),
            {"on": "detailed thinking on", "off": "detailed thinking off"},
        )
        # It must NOT claim supports_thinking: enable_thinking is ignored here.
        self.assertNotIn("supports_thinking", profile)

    def test_plain_llama3_keeps_no_directive(self):
        self.assertEqual(models.profile_for_model("meta/llama3-70b").get("thinking_directive"), None)

    def test_directive_is_stripped_from_extra_body(self):
        # thinking_directive is client-only; vLLM has no such sampling param.
        self.assertNotIn("thinking_directive", vllm_backend._model_extra_body(self.ULTRA))

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
