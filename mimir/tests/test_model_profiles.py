"""Model-profile-driven knobs: pin_role and the B300-ready tool-count cap.

The weak-model accommodations (the ~40-tool cap, the strict-template pin role)
are per-model settings, not hard globals, so the same codebase scales from
Devstral-24B to a 400B-class model on the B300s.
"""
import os
import unittest

from mimir.client.config import models, constants


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


if __name__ == "__main__":
    unittest.main()
