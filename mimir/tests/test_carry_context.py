"""Tests for carry_context: JSON round-trip, read staleness, deleted-path purge.

carry_context is the session-level state that survives between queries. It had no
schema, so nothing checked that its keys survived being written to a session file and
read back — which is how a set came to round-trip as the *string* "{'a.py'}" and get
exploded into single characters by the next merge. These tests assert the types, not
just the values.
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from mimir.client.agent_core import MimirAgent
from mimir.client.context.execution_context import (
    carry_context_from_json,
    carry_context_to_json,
    carry_path_fields,
    carry_set_fields,
    execution_context_template,
)


def _bare_agent():
    """An agent with only the carry machinery — no servers, no model, no init."""
    agent = MimirAgent.__new__(MimirAgent)
    agent._carry_context = {}
    return agent


class CarrySchemaTests(unittest.TestCase):
    def test_set_fields_include_the_carry_only_key(self) -> None:
        fields = carry_set_fields()
        self.assertIn("read_files", fields)          # mirrored from the context
        self.assertIn("last_query_written_files", fields)  # exists only in carry

    def test_path_fields_include_the_carry_only_key(self) -> None:
        self.assertIn("last_query_written_files", carry_path_fields())


class CarryJsonRoundTripTests(unittest.TestCase):
    """The regression: a set must not survive as a string."""

    def _carry(self):
        return {
            "read_files": {"a.py", "b.py"},
            "existing_paths": {"a.py"},
            "searched": True,
            "last_query_written_files": {"solver.py", "util.py"},
            "_read_mtimes": {"a.py": 1.0},
        }

    def test_every_value_is_json_native_after_export(self) -> None:
        exported = carry_context_to_json(self._carry())
        # default=str is what the session store uses; it hides a miss instead of
        # raising, so assert the strict encoder accepts the payload untouched.
        json.dumps(exported)  # must not raise
        for key, value in exported.items():
            self.assertNotIsInstance(value, (set, frozenset), f"{key} left as a set")

    def test_round_trip_preserves_types_and_values(self) -> None:
        original = self._carry()
        restored = carry_context_from_json(json.loads(json.dumps(carry_context_to_json(original))))
        for key in carry_set_fields():
            if key in original:
                self.assertIsInstance(restored[key], set, f"{key} is not a set again")
                self.assertEqual(restored[key], original[key])
        self.assertEqual(restored["searched"], True)
        self.assertEqual(restored["_read_mtimes"], {"a.py": 1.0})

    def test_written_files_do_not_explode_into_characters(self) -> None:
        # The exact defect: set -> str -> set(str) == set of characters.
        agent = _bare_agent()
        agent._carry_context = self._carry()
        blob = json.dumps(agent.export_state(), default=str)

        reloaded = _bare_agent()
        reloaded.load_state(json.loads(blob))

        self.assertEqual(
            reloaded._carry_context["last_query_written_files"], {"solver.py", "util.py"}
        )

    def test_undeclared_set_is_still_converted(self) -> None:
        # Backstop: a key nobody declared must not corrupt the session file.
        exported = carry_context_to_json({"future_key": {"x", "y"}})
        self.assertEqual(exported["future_key"], ["x", "y"])
        json.dumps(exported)  # must not raise


class ReadStalenessTests(unittest.TestCase):
    """A re-read after an external edit must re-establish the carried evidence."""

    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "solver.py")
        with open(self.path, "w") as fh:
            fh.write("x = 1\n")
        self.agent = _bare_agent()
        self._patch = patch(
            "mimir.client.agent_core.absolute_workspace_path", side_effect=lambda p: p
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _query_that_reads_the_file(self) -> bool:
        """Run one query's carry cycle; return whether the carried read was accepted."""
        ctx = execution_context_template()
        self.agent._apply_carry_context(ctx)
        accepted = self.path in ctx["read_files"]
        ctx["read_files"].add(self.path)
        self.agent._update_carry_context(ctx)
        return accepted

    def _touch(self) -> None:
        time.sleep(0.02)
        with open(self.path, "w") as fh:
            fh.write("x = 2\n")

    def test_unmodified_file_stays_carried(self) -> None:
        self._query_that_reads_the_file()
        self.assertTrue(self._query_that_reads_the_file())

    def test_external_edit_evicts_the_carried_read(self) -> None:
        self._query_that_reads_the_file()
        self._touch()
        self.assertFalse(self._query_that_reads_the_file())

    def test_the_re_read_after_an_external_edit_is_accepted_again(self) -> None:
        # The regression: the recorded mtime was written once and never refreshed, so
        # a single external edit evicted the path on EVERY later query, forever.
        self._query_that_reads_the_file()
        self._touch()
        self._query_that_reads_the_file()          # evicted, and re-read
        self.assertTrue(self._query_that_reads_the_file())
        self.assertTrue(self._query_that_reads_the_file())

    def test_a_file_written_this_query_is_dropped_from_carry(self) -> None:
        ctx = execution_context_template()
        self.agent._apply_carry_context(ctx)
        ctx["read_files"].add(self.path)
        ctx["dirty_written_files"].add(self.path)
        self.agent._update_carry_context(ctx)
        self.assertNotIn(self.path, self.agent._carry_context["read_files"])
        self.assertNotIn(self.path, self.agent._carry_context["_read_mtimes"])

    def test_mtimes_do_not_accumulate_for_uncarried_paths(self) -> None:
        self._query_that_reads_the_file()
        self.agent._carry_context["_read_mtimes"]["gone.py"] = 1.0
        self._query_that_reads_the_file()
        self.assertNotIn("gone.py", self.agent._carry_context["_read_mtimes"])


class DiscardCarryPathTests(unittest.TestCase):
    def test_deleting_a_file_purges_it_from_every_carried_path_field(self) -> None:
        agent = _bare_agent()
        agent._carry_context = {
            "read_files": {"a.py", "b.py"},
            "existing_paths": {"a.py"},
            "checked_paths": {"a.py"},
            "last_query_written_files": {"a.py", "c.py"},
            "_read_mtimes": {"a.py": 1.0, "b.py": 2.0},
        }
        agent._discard_carry_path("a.py")
        for field in carry_path_fields():
            self.assertNotIn("a.py", agent._carry_context.get(field, set()), field)
        self.assertNotIn("a.py", agent._carry_context["_read_mtimes"])
        # Untouched neighbours survive.
        self.assertEqual(agent._carry_context["read_files"], {"b.py"})
        self.assertEqual(agent._carry_context["last_query_written_files"], {"c.py"})


if __name__ == "__main__":
    unittest.main()
