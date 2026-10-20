"""A prose variant rewords the prompt's sections and the constant messages for a run, and
leaves the code's texts in force everywhere else."""
from __future__ import annotations

import json
import os
import re
import tempfile
import unittest

from mimir.client.guardrails.nudges import messages
from mimir.client.prompt import system_prompt
from mimir.client.prompt.variants import load_variant, use_variant


class PromptVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def _file(self, payload: dict) -> str:
        path = os.path.join(self.dir, "shorter.json")
        with open(path, "w") as fh:
            json.dump(payload, fh)
        return path

    def test_every_text_has_a_stable_id(self) -> None:
        self.assertIn("validation", system_prompt.SECTIONS)
        self.assertIn("step_limit_nudge", messages.MESSAGES)
        for ids in (system_prompt.SECTIONS, messages.MESSAGES):
            self.assertTrue(all(re.fullmatch(r"[a-z_]+", i) for i in ids))

    def test_a_variant_rewords_the_prompt_and_then_gives_it_back(self) -> None:
        before = system_prompt.build_base_system_content()
        variant = load_variant(self._file({"sections": {"validation": "## Validation\nCheck."}}))
        with use_variant(variant) as name:
            during = system_prompt.build_base_system_content()
        self.assertEqual(name, "shorter")
        self.assertIn("## Validation\nCheck.", during)
        self.assertNotIn(system_prompt.SECTIONS["validation"], during)
        self.assertEqual(system_prompt.build_base_system_content(), before)

    def test_a_variant_rewords_a_constant_message(self) -> None:
        variant = load_variant(self._file({"messages": {"step_limit_nudge": "Stop now."}}))
        with use_variant(variant):
            self.assertEqual(messages.message("step_limit_nudge"), "Stop now.")
        self.assertEqual(messages.message("step_limit_nudge"), messages.STEP_LIMIT_NUDGE)

    def test_an_unknown_id_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown sections validaton"):
            load_variant(self._file({"sections": {"validaton": "typo"}}))

    def test_no_variant_is_the_code_s_texts(self) -> None:
        with use_variant(None) as name:
            self.assertEqual(name, "")
            self.assertEqual(system_prompt.build_base_system_content(),
                             system_prompt._DEFAULT_BASE_SYSTEM_CONTENT)


if __name__ == "__main__":
    unittest.main()
