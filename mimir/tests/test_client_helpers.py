import contextlib
import io
import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import asyncio

import mimir.client.event_sink as event_sink_module
import mimir.client.config.constants as constants_module
from mimir.client.query_engine.backends.vllm_backend import VllmBackend
from mimir.tests._fake_backend import ScriptedBackend

import mimir.client.guardrails.policy.state_machine as policy_state_machine_module
import mimir.client.guardrails.policy.write as policy_runtime_module
import mimir.client.config as client_config_module
import mimir.client.query_engine.agent_loop as agent_loop_module
import mimir.client.query_engine.streaming as streaming_module
import mimir.client.query_engine.history as history_module
import mimir.client.query_engine.finalize as finalize_module
import mimir.client.query_engine.dispatch as dispatch_module
import mimir.client.human_pause as human_pause_module
import mimir.client.prompt.system_prompt as context_builder_module
import mimir.client.context.execution_context as execution_context_module
import mimir.client.ui.cli.chat_session as chat_session_module
from mimir.client.agent_core import MimirAgent
from mimir.client.tool_execution.validation import auto_validate_written_file
from mimir.client.context.capabilities import path_args
from mimir.tests._golden_caps import build_declared_registry

# Provide a module-compatible namespace so existing test references work.
import types
client_module = types.SimpleNamespace(
    MimirAgent=MimirAgent,
    SERVERS=__import__('mimir.client.config', fromlist=['SERVERS']).SERVERS,
    _BASE=__import__('mimir.client.agent_core', fromlist=['_BASE'])._BASE,
)


class ClientHelperTests(unittest.TestCase):
    def test_client_config_default_model_matches_client_constructor_default(self) -> None:
        defaults = client_module.MimirAgent.__init__.__defaults__
        self.assertIsNotNone(defaults)
        self.assertEqual(defaults[0], client_config_module.DEFAULT_MODEL)

    def test_client_config_path_args_cover_symbol_outline_path(self) -> None:
        registry = build_declared_registry()
        self.assertEqual(path_args("symbol_outline", registry), ("path",))

    def test_client_config_servers_match_client_registry(self) -> None:
        self.assertEqual(client_module.SERVERS, client_config_module.SERVERS)
        self.assertEqual(client_module._BASE, client_config_module.SERVER_BASE)

    def test_new_execution_context_matches_runtime_contract(self) -> None:
        context = client_module.MimirAgent._new_execution_context()
        execution_context_module.validate_execution_context(context)
        self.assertIn("validation_fail_count_by_file", context)
        self.assertIn("planned_edit_targets", context)

    def test_ensure_execution_context_backfills_missing_keys(self) -> None:
        partial = {"searched": False}
        ensured = execution_context_module.ensure_execution_context(partial)
        self.assertIsNotNone(ensured)
        execution_context_module.validate_execution_context(ensured)

    def test_field_specs_match_typeddict_annotations(self) -> None:
        # _FIELD_SPECS is the single source of truth from which the template and the
        # validator are derived; the TypedDict is the static-typing surface. Guard
        # against the two drifting apart when a field is added or removed.
        spec_keys = {name for name, _f, _t, _traits in execution_context_module._FIELD_SPECS}
        annotation_keys = set(execution_context_module.ExecutionContext.__annotations__)
        self.assertEqual(spec_keys, annotation_keys)

    def test_declared_traits_are_from_the_known_vocabulary(self) -> None:
        m = execution_context_module
        for name, _f, _t, traits in m._FIELD_SPECS:
            with self.subTest(field=name):
                self.assertTrue(traits <= m.FIELD_TRAITS,
                                f"{name} declares unknown trait(s): {traits - m.FIELD_TRAITS}")

    def test_every_path_valued_field_answers_the_trait_questions(self) -> None:
        """A path-like field must say what it is, or say explicitly that it is nothing.

        This is the clause that stops the defect from reforming. The per-field
        properties ("carried across queries?", "purged when a file is deleted?",
        "counts as discovery?") used to live in eight hand-maintained name lists spread
        over four modules; nothing made anyone classify a *new* field, so the lists went
        stale silently — two of them still named a field that is never in carry_context
        at all. Adding a `*_files` / `*_paths` / `*_dirs` field now fails here until the
        question has been answered one way or the other.
        """
        m = execution_context_module
        # Fields whose name looks path-like but which genuinely carry no trait. Each
        # entry is a decision, not an oversight, so it is spelled out here.
        TRAITLESS = {
            # Planning intent, not evidence: these hold paths the model *said* it would
            # touch. They are per-query and must not survive a delete purge, since a
            # plan naming a file that was then deleted is exactly what the
            # "declared but never written" ledger line needs to still see.
            "planned_edit_targets", "declared_edit_set",
            # Files passed to a test runner: a history of what ran, not path evidence.
            "tests_run",
            # Per-directory candidate names, a dict keyed by directory.
            "similar_candidates_by_dir",
            # Keyed by tool_call_id, not by path: a dict recording which files each
            # tool *message* concerns, so history trimming can match messages to files
            # structurally. Purging a deleted path from it would strand the message
            # association it exists to provide.
            "tool_msg_files",
        }
        for name, _f, _t, traits in m._FIELD_SPECS:
            if not name.endswith(("_files", "_paths", "_dirs")):
                continue
            with self.subTest(field=name):
                self.assertTrue(
                    traits or name in TRAITLESS,
                    f"{name} looks path-valued but declares no trait. Give it one "
                    f"(CARRY / FILE_PATH / KNOWN_FILE / DISCOVERY) or add it to "
                    f"TRAITLESS here with the reason.",
                )

    def test_derived_field_lists_match_their_hand_written_originals(self) -> None:
        """The derivations must reproduce the lists they replaced, membership-exactly.

        Written as literals rather than recomputed so the test states what the
        behaviour *was*: a refactor meant to preserve semantics has to be checkable
        against the semantics it claims to preserve.
        """
        m = execution_context_module
        self.assertEqual(set(m.fields_with(m.CARRY)), {
            "read_files", "delegated_read_files", "existing_paths", "inspected_dirs",
            "checked_paths"})
        self.assertEqual(set(m.fields_with(m.FILE_PATH)), {
            "existing_paths", "read_files", "delegated_read_files", "checked_paths",
            "dirty_written_files", "validated_files", "unverifiable_files"})
        self.assertEqual(set(m.fields_with(m.KNOWN_FILE)), {
            "existing_paths", "read_files", "delegated_read_files", "checked_paths"})
        # `delegated_read_files` is a discovery signal in its own right: a sub-agent's
        # reading is the model's own evidence, gathered at arm's length.
        self.assertEqual(set(m.DISCOVERY_EVIDENCE_SIGNALS), {
            "searched", "read_files", "delegated_read_files", "checked_paths",
            "inspected_dirs"})

    def test_carried_path_purge_drops_the_inert_entry_and_keeps_dirs_out(self) -> None:
        """`_discard_carry_path`'s field set, now derived — two deliberate absences.

        `dirty_written_files` is gone: it was in the hand-written list but is never
        placed in carry_context, so purging it was theatre. `inspected_dirs` stays out
        because it holds *directories* — deleting a file does not un-inspect the
        directory that contained it.
        """
        m = execution_context_module
        carried_paths = set(m.fields_with(m.CARRY, m.FILE_PATH))
        self.assertEqual(carried_paths, {
            "read_files", "delegated_read_files", "existing_paths", "checked_paths"})
        self.assertNotIn("dirty_written_files", carried_paths)
        self.assertNotIn("inspected_dirs", carried_paths)

    def test_known_existing_files_has_one_implementation(self) -> None:
        """Policy and nudges asked the same question with two separate walks."""
        from mimir.client.guardrails.nudges import engine as nudge_engine
        ec = execution_context_module.build_execution_context()
        ec["read_files"].add("a.py")
        ec["checked_paths"].add("c.py")
        ec["existing_paths"].add("d.py")
        ec["dirty_written_files"].add("written.py")  # written ≠ "encountered"
        shared = execution_context_module.known_existing_files(ec)
        self.assertEqual(shared, {"a.py", "c.py", "d.py"})
        self.assertEqual(nudge_engine._known_existing_files(ec), shared)

    def test_a_read_says_nothing_about_how_much_was_read(self) -> None:
        """The predicate answers "was it read", and deliberately not "was it read whole".

        Reads are capped and targeted, so a window that stopped at the cap is the normal
        case. A gate demanding the whole file would be asking for the one thing the read
        policy tells the model not to do.
        """
        m = execution_context_module
        ec = m.build_execution_context()
        ec["read_files"].add("partial.py")
        self.assertTrue(m.was_read(ec, "partial.py"))
        # A read also proves the file exists.
        self.assertIn("partial.py", m.known_existing_files(ec))

        self.assertFalse(m.was_read(ec, "never-read.py"))

    def test_a_checked_path_is_not_proof_of_existence(self) -> None:
        """`checked_paths` records that a look happened, including a negative one."""
        m = execution_context_module
        ec = m.build_execution_context()
        ec["checked_paths"].add("maybe.py")
        self.assertTrue(m.was_checked_for(ec, "maybe.py"))
        self.assertFalse(m.is_known_to_exist(ec, "maybe.py"))
        self.assertFalse(m.was_read(ec, "maybe.py"))

    def test_overwrite_gate_refuses_a_file_nobody_read(self) -> None:
        """The predicate pair, exercised through the guard that depends on it.

        A file known to exist but never read must not be rewritten in full: the failure
        mode the read-before-overwrite rule exists for.
        """
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        ec = execution_context_module.build_execution_context()
        ec["existing_paths"].add("src/module.py")

        violation = policy_runtime_module.check_write_policy(
            agent, "write_file", {"path": "src/module.py", "content": "x"}, ec,
        )

        self.assertIsNotNone(violation)
        self.assertIn("requires reading it first", violation)

    def test_overwrite_gate_accepts_a_file_read_in_part(self) -> None:
        """A capped read clears the gate: it asks for a read, not for the whole file."""
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        ec = execution_context_module.build_execution_context()
        ec["existing_paths"].add("src/module.py")
        ec["read_files"].add("src/module.py")

        violation = policy_runtime_module.check_write_policy(
            agent, "write_file", {"path": "src/module.py", "content": "x"}, ec,
        )

        self.assertIsNone(violation)

    def test_deleting_a_file_purges_it_from_every_path_field(self) -> None:
        """The purge actually runs — not just that the derived field set is right.

        Both purge loops (query context and carry context) had **no** behavioural
        coverage: an execution trace of the whole suite reached neither. Asserting the
        derived list without ever running the loop would have re-created exactly the
        gap this refactor is about — a rule that looks maintained and is not.
        """
        import types
        from mimir.client.guardrails import observations
        from mimir.client.context.capabilities import REMOVE, ToolCaps

        m = execution_context_module
        ec = m.build_execution_context()
        for field in m.fields_with(m.FILE_PATH):
            ec[field].add("doomed.py")
            ec[field].add("kept.py")

        agent = types.SimpleNamespace(
            tool_caps={"delete_file": ToolCaps(
                name="delete_file", capabilities=frozenset({REMOVE}))},
            _normalize_workspace_path=lambda p: p or "",
            _discard_carry_path=lambda p: None,
        )
        observations._observe_delete(
            agent, "delete_file", {"path": "doomed.py"}, "ok", ec,
        )

        for field in m.fields_with(m.FILE_PATH):
            with self.subTest(field=field):
                self.assertNotIn("doomed.py", ec[field])
                self.assertIn("kept.py", ec[field], "an unrelated path was purged too")

    def test_deleting_a_file_purges_it_from_the_carried_path_fields(self) -> None:
        agent = client_module.MimirAgent()
        m = execution_context_module
        for field in m.fields_with(m.CARRY):
            agent._carry_context[field] = {"doomed.py", "kept.py"}

        agent._discard_carry_path("doomed.py")

        for field in m.fields_with(m.CARRY, m.FILE_PATH):
            with self.subTest(field=field):
                self.assertNotIn("doomed.py", agent._carry_context[field])
        # inspected_dirs is carried but holds directories: deleting a file must not
        # un-inspect the directory it lived in.
        self.assertIn("doomed.py", agent._carry_context["inspected_dirs"])

    def test_backfill_gives_a_bare_dict_every_declared_field(self) -> None:
        """One seeder, derived from the schema, replacing four hand-picked subsets."""
        m = execution_context_module
        ec: dict = {}
        m.backfill_execution_context(ec)
        self.assertEqual(set(ec), set(m.ExecutionContext.__annotations__))
        # The nudge seeder used to default this to 99 where the schema says 0, so the
        # same absent field meant "maximally idle" to the idle gates and "just edited"
        # to the message builder. One seeder, one answer.
        self.assertEqual(ec["steps_since_last_edit"], 0)
        self.assertIsNone(m.backfill_execution_context(None))

    def test_template_satisfies_validator(self) -> None:
        template = execution_context_module.execution_context_template()
        # Round-trips: the generated template must pass the generated validator.
        execution_context_module.validate_execution_context(template)

    def test_json_error_payload_is_structured(self) -> None:
        payload = json.loads(
            client_module.MimirAgent._json_error_payload(
                "Boom",
                hint="Retry",
                tool="demo",
            )
        )
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"], "Boom")
        self.assertEqual(payload["hint"], "Retry")
        self.assertEqual(payload["tool"], "demo")

    def test_server_base_path_is_relative_to_client_file(self) -> None:
        expected = str((Path(__file__).resolve().parents[1] / "servers").resolve())
        self.assertEqual(client_module._BASE.rstrip("/"), expected)

    def test_normalize_tool_path_argument_empty_to_dot(self) -> None:
        self.assertEqual(client_module.MimirAgent._normalize_tool_path_argument(""), ".")

    def test_normalize_tool_path_argument_collapses_workspace_basename(self) -> None:
        cwd_base = Path(__file__).resolve().parents[2].name
        normalized = client_module.MimirAgent._normalize_tool_path_argument(f"{cwd_base}/mimir")
        self.assertEqual(normalized, "mimir")

    def test_rewrite_read_file_to_read_file_lines_for_code(self) -> None:
        agent = client_module.MimirAgent()
        agent.tool_owner = {"read_file_lines": "search"}

        tool_name, rewritten = agent._rewrite_tool_for_context(
            "read_file",
            {"path": "src/module.py"},
        )

        self.assertEqual(tool_name, "read_file_lines")
        self.assertEqual(rewritten["start_line"], 1)
        self.assertEqual(rewritten["end_line"], 120)

    def test_rewrite_read_file_to_read_file_lines_for_docs(self) -> None:
        agent = client_module.MimirAgent()
        agent.tool_owner = {"read_file_lines": "search"}

        tool_name, rewritten = agent._rewrite_tool_for_context(
            "read_file",
            {"path": "docs/guide.md"},
        )

        self.assertEqual(tool_name, "read_file_lines")
        self.assertEqual(rewritten["start_line"], 1)
        self.assertEqual(rewritten["end_line"], 160)

    def test_rewrite_read_file_reads_whole_file_for_non_targeted_paths(self) -> None:
        # read_file is retired; non-targeted/binary-like paths rewrite to
        # read_file_lines with end_line=0 (whole file) rather than a leading window.
        agent = client_module.MimirAgent()
        agent.tool_owner = {"read_file_lines": "search"}

        tool_name, rewritten = agent._rewrite_tool_for_context(
            "read_file",
            {"path": "models/model.gguf"},
        )

        self.assertEqual(tool_name, "read_file_lines")
        self.assertEqual(rewritten["start_line"], 1)
        self.assertEqual(rewritten["end_line"], 0)

    def test_carry_context_populates_existing_paths(self) -> None:
        agent = client_module.MimirAgent()
        agent._carry_context["existing_paths"] = {"mimir/servers/utilities/server_math.py"}
        execution_context = {
            "searched": False,
            "inspected_dirs": set(),
            "checked_paths": set(),
            "read_files": set(),
            "existing_paths": set(),
            "similar_candidates_by_dir": {},
        }

        agent._apply_carry_context(execution_context)
        self.assertIn("mimir/servers/utilities/server_math.py", execution_context["existing_paths"])

    def test_write_policy_allows_new_non_code_file_with_direct_target_context(self) -> None:
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        execution_context = {
            "searched": False,
            "inspected_dirs": {"docs"},
            "checked_paths": {"docs/notes.md"},
            "read_files": set(),
            "existing_paths": set(),
            "similar_candidates_by_dir": {},
            "search_tool_calls": 0,
        }

        violation = policy_runtime_module.check_write_policy(
            agent,
            "write_file",
            {"path": "docs/notes.md", "content": "hello"},
            execution_context,
        )

        self.assertIsNone(violation)

    def test_replace_in_file_allows_single_line_anchor_for_non_code_files(self) -> None:
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        execution_context = {
            "searched": True,
            "inspected_dirs": {"docs"},
            "checked_paths": {"docs/guide.md"},
            "read_files": {"docs/guide.md"},
            "existing_paths": {"docs/guide.md"},
            "similar_candidates_by_dir": {},
            "search_tool_calls": 1,
        }

        violation = policy_runtime_module.check_write_policy(
            agent,
            "replace_in_file",
            {"path": "docs/guide.md", "old_text": "Old title", "new_text": "New title"},
            execution_context,
        )

        self.assertIsNone(violation)

    def test_replace_in_file_allows_two_line_anchor_for_code_with_direct_target_context(self) -> None:
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        execution_context = {
            "searched": False,
            "inspected_dirs": {"src"},
            "checked_paths": {"src/module.py"},
            "read_files": {"src/module.py"},
            "existing_paths": {"src/module.py"},
            "similar_candidates_by_dir": {},
            "search_tool_calls": 0,
        }

        violation = policy_runtime_module.check_write_policy(
            agent,
            "replace_in_file",
            {
                "path": "src/module.py",
                "old_text": "def compute(x):\n    return x",
                "new_text": "def compute(x):\n    return x + 1",
            },
            execution_context,
        )

        self.assertIsNone(violation)

    def test_delete_file_ignores_similar_candidates_once_target_is_confirmed(self) -> None:
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        execution_context = {
            "searched": False,
            "inspected_dirs": {"docs"},
            "checked_paths": {"docs/obsolete.md"},
            "read_files": set(),
            "existing_paths": {"docs/obsolete.md"},
            "similar_candidates_by_dir": {"docs": {"docs/reference.md"}},
            "search_tool_calls": 0,
        }

        violation = policy_runtime_module.check_write_policy(
            agent,
            "delete_file",
            {"path": "docs/obsolete.md"},
            execution_context,
        )

        self.assertIsNone(violation)

    def test_write_policy_allows_code_creation_with_direct_target_context(self) -> None:
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        execution_context = {
            "searched": False,
            "inspected_dirs": {"src"},
            "checked_paths": {"src/new_feature.py"},
            "read_files": set(),
            "existing_paths": set(),
            "similar_candidates_by_dir": {"src": {"src/old_feature.py"}},
            "search_tool_calls": 0,
        }

        violation = policy_runtime_module.check_write_policy(
            agent,
            "write_file",
            {"path": "src/new_feature.py", "content": "def run():\n    return 1\n"},
            execution_context,
        )

        self.assertIsNone(violation)

    def test_state_machine_allows_code_edit_with_direct_target_context_before_global_discovery(self) -> None:
        agent = client_module.MimirAgent()
        execution_context = {
            "workflow_state": "discover",
            "searched": False,
            "inspected_dirs": {"src"},
            "checked_paths": {"src/module.py"},
            "read_files": set(),
            "existing_paths": {"src/module.py"},
            "similar_candidates_by_dir": {},
            "search_tool_calls": 0,
            "dirty_written_files": set(),
            "validated_files": set(),
            "denied_tool_calls": [],
            "code_mutation_started": False,
            "denial_nudges": 0,
            "state_nudges": 0,
            "creation_nudges": 0,
        }

        violation = policy_state_machine_module.check_state_machine_guard(
            agent,
            "replace_in_file",
            {"path": "src/module.py"},
            execution_context,
        )

        self.assertIsNone(violation)

    def test_replace_in_file_allows_previewed_code_target_with_explicit_anchor(self) -> None:
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        execution_context = {
            "searched": False,
            "inspected_dirs": {"src"},
            "checked_paths": {"src/module.py"},
            "read_files": set(),
            "existing_paths": {"src/module.py"},
            "similar_candidates_by_dir": {},
            "search_tool_calls": 0,
        }

        violation = policy_runtime_module.check_write_policy(
            agent,
            "replace_in_file",
            {
                "path": "src/module.py",
                "old_text": "def helper():\n    return 0",
                "new_text": "def helper():\n    return 1",
            },
            execution_context,
        )

        self.assertIsNone(violation)

    def test_request_tool_approval_denies_after_retry_budget(self) -> None:
        agent = client_module.MimirAgent()
        agent.approvals.batch_mode = False
        with patch("builtins.input", side_effect=["?", "invalid", "maybe"]):
            approved, note = agent._request_tool_approval("write_file", {"path": "x.txt"}, max_attempts=3)

        self.assertFalse(approved)
        self.assertIn("invalid approval responses", note)

    def test_denied_tool_result_offers_the_three_readings(self) -> None:
        agent = client_module.MimirAgent()
        # fallback_tools is seeded from the live registry post-connect; emulate that here.
        agent.approvals.fallback_tools["bash_run"] = ("code_run", "read_file_lines", "grep")
        payload = json.loads(
            agent._denied_tool_result("bash_run", {"command": "echo hi"}, "denied by user"))
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["denial_stage"], "reconsider")
        self.assertEqual(payload["denial_reason"], "denied by user")
        self.assertEqual(payload["denial_kind"], "denied")
        # All three readings are offered on a first refusal, alternatives included.
        for fragment in ("Not this way", "Unnecessary", "Stop", "code_run"):
            self.assertIn(fragment, payload["hint"])
        # Re-asking is the behaviour the ladder removes; the copy must not invite it.
        self.assertNotIn("Ask the user for confirmation", payload["hint"])

    def test_denied_tool_result_narrows_as_refusals_accumulate(self) -> None:
        agent = client_module.MimirAgent()
        execution_context = {"denial_history": []}
        scope = agent.approval_scope("bash_run", {"command": "pip install numpy"})

        def refuse() -> dict:
            execution_context["denial_history"].append(
                {"tool": "bash_run", "scope": scope, "kind": "denied", "reason": "denied by user"})
            return json.loads(agent._denied_tool_result(
                "bash_run", {"command": "pip install numpy"}, "denied by user", execution_context))

        first = refuse()
        self.assertEqual(first["denial_stage"], "reconsider")

        second = refuse()
        self.assertEqual(second["denial_stage"], "drop_or_stop")
        self.assertIn("Do not try another route", second["hint"])

        third = refuse()
        self.assertEqual(third["denial_stage"], "handback")
        self.assertEqual(third["prior_refusals_for_this_goal"], 3)
        self.assertIn("stop here and hand back", third["hint"])

    def test_record_denied_tool_call_feeds_two_independent_ledgers(self) -> None:
        # denied_tool_calls is the open set and gets cleared when the action later
        # succeeds; denial_history is what the ladder counts and must survive that,
        # or a refusal followed by an unrelated success would reset the escalation.
        from mimir.client.guardrails.observations import _observe_denial_clearing
        from mimir.client.guardrails.workflow import denial_stage

        agent = client_module.MimirAgent()
        execution_context = client_module.MimirAgent._new_execution_context()
        args = {"command": "pip install numpy"}
        scope = agent.approval_scope("bash_run", args)

        agent._record_denied_tool_call("bash_run", args, execution_context, "denied by user")
        agent._record_denied_tool_call("bash_run", args, execution_context, "denied by user")
        self.assertEqual(len(execution_context["denied_tool_calls"]), 2)
        self.assertEqual(execution_context["denial_history"][0]["scope"], scope)
        self.assertEqual(execution_context["denial_history"][0]["kind"], "denied")
        self.assertEqual(denial_stage(execution_context, scope), "drop_or_stop")

        _observe_denial_clearing(agent, "bash_run", args, "ok", execution_context)

        self.assertEqual(execution_context["denied_tool_calls"], [])
        self.assertEqual(len(execution_context["denial_history"]), 2)
        self.assertEqual(denial_stage(execution_context, scope), "drop_or_stop")

    def test_handback_nudge_fires_past_its_frequency_cap(self) -> None:
        # The other denial messages ask the model to act, so they are rationed. This
        # one tells it to stop — a reminder to stop that is itself suppressed leaves
        # the model going.
        import importlib
        from mimir.client.config.constants import NUDGE_MAX_DENIAL
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        execution_context = {
            "workflow_state": "edit",
            "denied_tool_calls": [{"tool": "bash_run", "scope": "bash:bash_run:pip install"}],
            "denial_history": [
                {"tool": "bash_run", "scope": "bash:bash_run:pip install", "kind": "denied"}
            ] * 3,
            "dirty_written_files": set(),
            "validated_files": set(),
            "nudge_counts": {"denial": NUDGE_MAX_DENIAL},
        }
        messages: list = []

        class _FakeAgent:
            model = "big-model"
            enforcement = "off"

        fired = nudge_logic.maybe_append_nudge(
            agent=_FakeAgent(),
            query="please add the new feature",
            active_mode="agent",
            execution_context=execution_context,
            messages=messages,
        )
        self.assertTrue(fired)
        self.assertIn("Stop here", messages[-1]["content"])
        self.assertNotIn("reach it another way", messages[-1]["content"])

    def test_state_machine_allows_code_edit_without_prior_read_in_edit_state(self) -> None:
        agent = client_module.MimirAgent()
        execution_context = {
            "workflow_state": "edit",
            "searched": True,
            "inspected_dirs": {"src"},
            "checked_paths": {"src/module.py"},
            "read_files": set(),
            "existing_paths": {"src/module.py"},
            "similar_candidates_by_dir": {},
            "search_tool_calls": 2,
            "dirty_written_files": set(),
            "validated_files": set(),
            "validation_fail_count_by_file": {},
            "denied_tool_calls": [],
            "code_mutation_started": False,
            "denial_nudges": 0,
            "state_nudges": 0,
            "creation_nudges": 0,
            "planned_edit_targets": set(),
        }

        violation = policy_state_machine_module.check_state_machine_guard(
            agent,
            "replace_in_file",
            {"path": "src/module.py"},
            execution_context,
        )

        self.assertIsNone(violation)

    def test_state_machine_blocks_code_edit_after_retry_budget_exceeded(self) -> None:
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        execution_context = {
            "workflow_state": "edit",
            "searched": True,
            "inspected_dirs": {"src"},
            "checked_paths": {"src/module.py"},
            "read_files": {"src/module.py"},
            "existing_paths": {"src/module.py"},
            "similar_candidates_by_dir": {},
            "search_tool_calls": 2,
            "dirty_written_files": {"src/module.py"},
            "validated_files": set(),
            # Meets VALIDATION_RETRY_BUDGET (5) so the hard stop fires.
            "validation_fail_count_by_file": {"src/module.py": 5},
            "denied_tool_calls": [],
            "code_mutation_started": True,
            "denial_nudges": 0,
            "state_nudges": 0,
            "creation_nudges": 0,
            "planned_edit_targets": {"src/module.py"},
        }

        violation = policy_state_machine_module.check_state_machine_guard(
            agent,
            "replace_in_file",
            {"path": "src/module.py"},
            execution_context,
        )

        self.assertIsNotNone(violation)
        self.assertIn("repeated validation failures", violation)

    # ── What the write path deliberately stopped blocking ─────────────────────
    #
    # Two hard guards were removed because they enforced a working *order* rather
    # than guarding a loss: a pre-edit planning gate (no plan/todo recorded → first
    # code edit refused) and the state guard's `validate` branch (edit to a file
    # unrelated to the pending set → refused). Both blocked reversible work, and
    # being hard guards they sat outside the `/enforcement` dial that exists to tune
    # exactly that. Neither was covered by a test, so these pin the new behaviour —
    # a silent reintroduction is the regression worth catching.

    def test_write_policy_does_not_demand_a_plan_before_the_first_edit(self) -> None:
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        execution_context = {
            "searched": True,
            "inspected_dirs": {"src"},
            "checked_paths": {"src/module.py"},
            "read_files": {"src/module.py", "src/other.py", "src/third.py"},
            "existing_paths": {"src/module.py"},
            "similar_candidates_by_dir": {},
            "search_tool_calls": 1,
            # No plan, no todo, nothing written yet — what the old gate refused on.
            "code_mutation_started": False,
            "plan_written": False,
            "todo_written": False,
            "edit_loop_state": {},
        }

        violation = policy_runtime_module.check_write_policy(
            agent, "replace_in_file",
            {"path": "src/module.py", "old_text": "a", "new_text": "b"},
            execution_context,
        )

        self.assertIsNone(violation)

    def test_state_machine_allows_unrelated_edit_while_validation_is_pending(self) -> None:
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        execution_context = {
            "workflow_state": "validate",
            "searched": True,
            "inspected_dirs": {"src"},
            "checked_paths": set(),
            # Target is NOT in read_files and NOT in the pending set — the one shape
            # the removed branch actually reached, every other being carved out.
            "read_files": set(),
            "existing_paths": set(),
            "similar_candidates_by_dir": {},
            "search_tool_calls": 1,
            "dirty_written_files": {"src/pending.py"},
            "validated_files": set(),
            "validation_fail_count_by_file": {},
            "declared_edit_set": {"src/pending.py"},
            "denied_tool_calls": [],
            "code_mutation_started": True,
        }

        violation = policy_state_machine_module.check_state_machine_guard(
            agent, "replace_in_file", {"path": "src/unrelated.py"}, execution_context,
        )

        self.assertIsNone(violation)

    def test_write_policy_still_refuses_overwriting_an_unread_existing_file(self) -> None:
        """The protection that must survive the two removals above.

        Read-before-overwrite guards an irreversible loss (content replaced by a
        rewrite that never saw it), which is a different class from the working-order
        rules that were dropped — so removing those must not have loosened this.
        """
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        execution_context = {
            "searched": True,
            "inspected_dirs": {"src"},
            "checked_paths": {"src/module.py"},
            "read_files": set(),
            "existing_paths": {"src/module.py"},
            "similar_candidates_by_dir": {},
            "search_tool_calls": 1,
            "edit_loop_state": {},
        }

        violation = policy_runtime_module.check_write_policy(
            agent, "write_file", {"path": "src/module.py", "content": "wiped"},
            execution_context,
        )

        self.assertIsNotNone(violation)
        self.assertIn("requires reading it first", violation)

    def test_write_policy_refuses_a_delete_without_existence_evidence(self) -> None:
        """The delete guard's *blocking* branch, which nothing exercised.

        Every pre-existing write-policy test asserted a permissive outcome, and each
        did so from the early `_is_write_tool` exit (the bare agent had an empty
        capability registry), so no test had ever reached a rule at all. Removing two
        neighbouring guards is only defensible if the ones kept actually block — that
        claim needs the negative case, not just the positive one.
        """
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        execution_context = {
            "searched": True,
            # A pre-check happened, but `checked_paths` never proves existence, and the
            # parent directory was never inspected — the two things a delete needs.
            "checked_paths": {"src/gone.py"},
            "inspected_dirs": set(),
            "read_files": set(),
            "existing_paths": set(),
            "similar_candidates_by_dir": {},
            "search_tool_calls": 1,
            "edit_loop_state": {},
        }

        violation = policy_runtime_module.check_write_policy(
            agent, "delete_file", {"path": "src/gone.py"}, execution_context,
        )

        self.assertIsNotNone(violation)
        self.assertIn("before deleting it", violation)

    def test_write_policy_blocks_a_patch_retried_identically_to_the_limit(self) -> None:
        """The anti-thrashing branch — also never exercised before.

        `edit_loop_state` holds (last patch signature, identical consecutive failures);
        at `REPEATED_EDIT_FAILURE_LIMIT` the write is refused so the model stops
        re-sending a patch that has already failed unchanged.
        """
        from mimir.client.config.constants import REPEATED_EDIT_FAILURE_LIMIT
        agent = client_module.MimirAgent()
        agent.tool_caps = build_declared_registry()
        execution_context = {
            "searched": True,
            "inspected_dirs": {"src"},
            "checked_paths": {"src/module.py"},
            "read_files": {"src/module.py"},
            "existing_paths": {"src/module.py"},
            "similar_candidates_by_dir": {},
            "search_tool_calls": 1,
            "edit_loop_state": {"src/module.py": ("sig-abc", REPEATED_EDIT_FAILURE_LIMIT)},
        }

        violation = policy_runtime_module.check_write_policy(
            agent, "replace_in_file",
            {"path": "src/module.py", "old_text": "a", "new_text": "b"},
            execution_context,
        )

        self.assertIsNotNone(violation)
        self.assertIn("repeated identical failed edit attempts", violation)

    def test_run_agent_query_returns_final_answer_when_no_tool_calls(self) -> None:
        captured_context = {}

        class _QueryAgent:
            mode = "agent"
            model = "dummy"
            tools = []
            tool_owner = {}
            tool_caps = {}

            @staticmethod
            def _new_execution_context():
                return client_module.MimirAgent._new_execution_context()

            @staticmethod
            def _get_todo_file():
                return ""

            @staticmethod
            def _normalize_mode(mode):
                return "agent"

            async def _build_system_content(self, **kwargs):
                return "system"

            async def _run_tool(self, tool, args, execution_context=None, run_auto_validation=True, call_id=""):
                return "{}"

            @staticmethod
            def _normalize_arguments(args):
                return args

            @staticmethod
            def _truncate_text(text, limit=600):
                return text[:limit]

            approvals = types.SimpleNamespace(flush_pending_review=lambda: None)

            def _apply_carry_context(self, execution_context):
                captured_context.update(execution_context)

            def _update_carry_context(self, execution_context):
                pass

        fake_agent = _QueryAgent()

        # The LLM backend is reached via agent_loop._stream_chat, which returns a
        # single response dict ({"content": ..., "tool_calls": ...}).  No tool calls
        # means the content becomes the final answer.
        with patch.object(finalize_module, "auto_store_memory", return_value=None):
            with patch.object(agent_loop_module, "_stream_chat", return_value={"content": "done"}):
                result = asyncio.run(
                    agent_loop_module.run_agent_query(
                        agent=fake_agent,
                        query="  tighten policy gates  ",
                        max_steps=1,
                    )
                )

        self.assertEqual(result, "done")

    def test_query_starts_with_no_discovery_evidence(self) -> None:
        """Nothing may pre-fill a discovery signal on the model's behalf.

        The repo-structure snapshot used to seed ``inspected_dirs`` at query start,
        which silently satisfied any gate reading that field raw. The invariant is
        cheaper to keep than the discount that used to compensate for it: a fresh
        context carries zero evidence, so every gate measures the model's own work.
        """
        ec = execution_context_module
        execution_context = client_module.MimirAgent._new_execution_context()
        self.assertEqual(ec.discovery_signal_count(execution_context), 0)
        for signal in ec.DISCOVERY_EVIDENCE_SIGNALS:
            with self.subTest(signal=signal):
                self.assertFalse(execution_context.get(signal))

    # ── soft step budget + continue checkpoint ─────────────────────────────────

    def _run_loop_with_budget(self, *, allow_continue, continue_returns, max_steps,
                              soft=2, ext=2, ceiling=6):
        """Drive _run_agent_loop with a _stream_chat that always emits a tool call.

        Returns (stream_call_count, continue_call_count). Heavy per-step helpers
        are patched to no-ops so the loop's budget/checkpoint arithmetic is the
        only thing under test.
        """
        stream_calls = {"n": 0}
        continue_calls = {"n": 0}

        def _fake_stream(*a, **k):
            stream_calls["n"] += 1
            return {"content": "working", "tool_calls": [
                {"id": "1", "function": {"name": "x", "arguments": "{}"}}
            ]}

        def _fake_continue(summary):
            i = continue_calls["n"]
            continue_calls["n"] += 1
            return continue_returns[i] if i < len(continue_returns) else False

        async def _noop_async(*a, **k):
            return None

        async def _fake_finalize(agent, query, answer, execution_context, messages, logger):
            return answer

        fake_agent = types.SimpleNamespace(
            model="dummy",
            tools=[],
            tool_caps={},
            thinking_budget=-1,
            allow_continue_prompt=allow_continue,
            _request_continue=_fake_continue,
            _update_carry_context=lambda ec: None,
            approvals=types.SimpleNamespace(flush_pending_review=lambda: None),
        )

        m = agent_loop_module
        with patch.object(m, "AGENT_STEP_SOFT_BUDGET", soft), \
             patch.object(m, "AGENT_STEP_EXTENSION", ext), \
             patch.object(m, "AGENT_STEP_HARD_CEILING", ceiling), \
             patch.object(m, "_stream_chat", _fake_stream), \
             patch.object(m, "_process_response", lambda *a, **k: None), \
             patch.object(agent_loop_module, "_dispatch_tool_calls", _noop_async), \
             patch.object(agent_loop_module, "_post_dispatch_inject", _noop_async), \
             patch.object(history_module, "_trim_tool_history", lambda *a, **k: None), \
             patch.object(history_module, "_maybe_compact_intra_query", lambda *a, **k: None), \
             patch.object(m, "_inject_pin", lambda *a, **k: None), \
             patch.object(m, "tools_for_context", lambda **k: []), \
             patch.object(m, "needs_incomplete_finalization", lambda ec: False), \
             patch.object(m, "_finalize_answer", _fake_finalize):
            asyncio.run(m._run_agent_loop(
                agent=fake_agent,
                query="q",
                active_mode="agent",
                messages=[{"role": "system", "content": "s"}],
                system_content="s",
                execution_context={},
                max_steps=max_steps,
                thinking=False,
                streaming=False,
                logger=None,
                cb={"think_token_callback": None},
            ))
        return stream_calls["n"], continue_calls["n"]

    def test_non_interactive_stops_at_max_steps_without_prompt(self) -> None:
        # allow_continue_prompt False → budget == max_steps, never asks the user.
        steps, prompts = self._run_loop_with_budget(
            allow_continue=False, continue_returns=[], max_steps=3)
        self.assertEqual(steps, 3)
        self.assertEqual(prompts, 0)

    def test_interactive_stops_when_user_declines(self) -> None:
        # Soft budget 2; user says no at the first checkpoint → run halts at 2.
        steps, prompts = self._run_loop_with_budget(
            allow_continue=True, continue_returns=[False], max_steps=50)
        self.assertEqual(steps, 2)
        self.assertEqual(prompts, 1)

    def test_interactive_extends_until_hard_ceiling(self) -> None:
        # User keeps saying yes: budget 2→4→6 (ceiling), prompted twice.
        steps, prompts = self._run_loop_with_budget(
            allow_continue=True, continue_returns=[True, True, True], max_steps=50)
        self.assertEqual(steps, 6)
        self.assertEqual(prompts, 2)



    def _counting_backend(self):
        calls = {"n": 0}

        def _tok(model, text):
            calls["n"] += 1
            return len(text)  # exact: 1 token per char

        return ScriptedBackend(tokenize=_tok), calls

    def test_count_text_tokens_caches_per_content(self) -> None:
        backend, calls = self._counting_backend()
        self.assertEqual(backend.count_text_tokens("m", "abcde"), 5)
        self.assertEqual(backend.count_text_tokens("m", "abcde"), 5)  # cache hit
        self.assertEqual(calls["n"], 1)
        self.assertEqual(backend.count_text_tokens("m", ""), 0)  # empty never tokenized
        self.assertEqual(calls["n"], 1)

    def test_count_messages_tokens_sums_content(self) -> None:
        backend, _ = self._counting_backend()
        total = backend.count_messages_tokens("m", [{"content": "abc"}, {"content": "de"}, {"role": "x"}])
        self.assertEqual(total, 5)

    def test_count_messages_tokens_includes_tool_call_payload(self) -> None:
        """An assistant turn is mostly its tool call; content-only scored it ~0."""
        backend, _ = self._counting_backend()
        call = [{"id": "c1", "function": {"name": "write", "arguments": {"text": "x" * 400}}}]
        total = backend.count_messages_tokens("m", [{"role": "assistant", "content": "", "tool_calls": call}])
        self.assertGreater(total, 400)

    def test_count_messages_tokens_includes_thinking(self) -> None:
        backend, _ = self._counting_backend()
        counts = backend.message_token_counts("m", [{"content": "ab", "thinking": "cde"}])
        self.assertEqual(counts, [6])  # "ab" + "\n" + "cde"

    def test_allow_network_false_uses_heuristic_when_uncached(self) -> None:
        backend, calls = self._counting_backend()
        # 8 chars / 4 (default ratio) = 2; tokenizer (network) must NOT be called.
        self.assertEqual(backend.count_text_tokens("m", "abcdefgh", allow_network=False), 2)
        self.assertEqual(calls["n"], 0)

    def test_allow_network_false_returns_cached_exact_value(self) -> None:
        backend, _ = self._counting_backend()
        backend.count_text_tokens("m", "abcdefgh")  # exact = 8, cached
        self.assertEqual(backend.count_text_tokens("m", "abcdefgh", allow_network=False), 8)

    def test_tokenize_failure_falls_back_to_heuristic_and_does_not_cache(self) -> None:
        calls = {"n": 0}

        def _tok(model, text):
            calls["n"] += 1
            raise RuntimeError("tokenize endpoint down")

        backend = ScriptedBackend(tokenize=_tok)
        self.assertEqual(backend.count_text_tokens("m", "abcdefgh"), 2)  # heuristic 8/4
        self.assertEqual(backend.count_text_tokens("m", "abcdefgh"), 2)  # still not cached
        self.assertEqual(calls["n"], 2)

    def test_chars_per_token_for_longest_match(self) -> None:
        with patch.dict(constants_module.CHARS_PER_TOKEN_BY_MODEL,
                        {"qwen": 3.5, "qwen3-coder": 3.0}, clear=True):
            self.assertEqual(constants_module.chars_per_token_for("Qwen3-Coder-30B"), 3.0)
            self.assertEqual(constants_module.chars_per_token_for("qwen2.5"), 3.5)
            self.assertEqual(constants_module.chars_per_token_for("llama-3"), 4.0)

    def test_vllm_tokenize_exact_then_fallback_on_error(self) -> None:
        backend = VllmBackend()

        # Exact path: /tokenize returns a count.
        with patch("httpx.Client") as MockClient:
            client = MockClient.return_value.__enter__.return_value
            client.post.return_value.raise_for_status.return_value = None
            client.post.return_value.json.return_value = {"count": 42}
            self.assertEqual(backend.count_text_tokens("m", "hello world"), 42)

        # Failure path: a fresh backend (clean cache) falls back to the heuristic.
        backend2 = VllmBackend()
        with patch("httpx.Client", side_effect=RuntimeError("no endpoint")):
            self.assertEqual(
                backend2.count_text_tokens("m", "x" * 12),  # 12 / 4 = 3
                3,
            )

    def test_trim_tool_history_token_mode_evicts_by_token_budget(self) -> None:
        messages = [{"role": "system", "content": "s"}]
        messages += [
            {"role": "tool", "tool_call_id": f"t{i}", "content": "x" * 100}
            for i in range(5)
        ]
        execution_context = {"read_files": set(), "tool_msg_files": {}}
        # token_counter=len -> 100 "tokens" each, total 500; budget 250 evicts the
        # 3 oldest (500 -> 200) and keeps the 2 newest.
        history_module._trim_tool_history(
            messages, execution_context=execution_context,
            token_counter=len, token_budget=250,
        )
        remaining = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
        self.assertEqual(remaining, ["t3", "t4"])

    # ── event_sink: injectable structured-event transport ──────────────────────

    def test_emit_prints_json_when_no_sink_bound(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            event_sink_module.emit({"type": "status", "text": "hi"})
        self.assertEqual(json.loads(buf.getvalue().strip()), {"type": "status", "text": "hi"})

    def test_event_sink_routes_to_bound_sink_then_resets(self) -> None:
        captured: list = []
        with event_sink_module.event_sink(captured.append):
            event_sink_module.emit({"type": "tool_call", "id": "c1"})
            event_sink_module.emit({"type": "diff", "file": "a.py"})
        self.assertEqual(captured, [{"type": "tool_call", "id": "c1"}, {"type": "diff", "file": "a.py"}])

        # After the context exits the sink is reset -> emit prints again.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            event_sink_module.emit({"type": "status", "text": "after"})
        self.assertEqual(json.loads(buf.getvalue().strip()), {"type": "status", "text": "after"})

    def test_event_sink_propagates_into_gathered_tasks(self) -> None:
        # The riskiest behaviour: a sink bound in the parent context must be seen
        # by emit() calls inside concurrently-gathered coroutines (mirrors how the
        # agent dispatches parallel tool calls). ContextVars copy into tasks.
        captured: list = []

        async def _worker(n: int) -> None:
            await asyncio.sleep(0)
            event_sink_module.emit({"type": "tool_result", "id": n})

        async def _run() -> None:
            with event_sink_module.event_sink(captured.append):
                await asyncio.gather(*[_worker(i) for i in range(5)])

        asyncio.run(_run())
        self.assertEqual(
            sorted(e["id"] for e in captured), [0, 1, 2, 3, 4]
        )

    def test_emit_falls_back_to_print_when_sink_raises(self) -> None:
        def _bad_sink(_ev: dict) -> None:
            raise RuntimeError("sink boom")

        buf = io.StringIO()
        with event_sink_module.event_sink(_bad_sink):
            with contextlib.redirect_stdout(buf):
                event_sink_module.emit({"type": "status", "text": "survive"})  # must not raise
        self.assertEqual(json.loads(buf.getvalue().strip()), {"type": "status", "text": "survive"})

    # ── _trim_tool_history structural file association ──────────────────────────

    def test_trim_history_does_not_falsely_invalidate_read_on_substring_match(self) -> None:
        # A search result that merely mentions a path must NOT invalidate that
        # path's actual read when it is evicted. The legacy substring scan got
        # this wrong; structural association (tool_msg_files) fixes it.
        big = "x" * 50_000
        messages = [
            {"role": "system", "content": "sys"},
            # Oldest tool message: a grep hit that *mentions* src/app.py but reads
            # no file. tool_msg_files records it as touching nothing ([]).
            {"role": "tool", "tool_call_id": "g1",
             "content": "match in src/app.py: foo\n" + big},
            # The actual read of src/app.py (kept; not evicted at this budget).
            {"role": "tool", "tool_call_id": "r1", "content": "file body"},
        ]
        execution_context = {
            "read_files": {"src/app.py"},
            "tool_msg_files": {"g1": [], "r1": ["src/app.py"]},
        }
        # Budget below the grep message size -> it gets evicted first.
        history_module._trim_tool_history(
            messages, char_budget=1_000, execution_context=execution_context
        )

        # grep message evicted, but src/app.py read stays valid.
        self.assertNotIn("g1", [m.get("tool_call_id") for m in messages])
        self.assertIn("src/app.py", execution_context["read_files"])

    def test_trim_history_invalidates_read_files_for_evicted_read(self) -> None:
        # When the actual read message is evicted, its file IS invalidated so the
        # policy forces a fresh re-read before the next edit.
        big = "y" * 50_000
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "tool_call_id": "r1", "content": "old body " + big},
            {"role": "tool", "tool_call_id": "r2", "content": "recent body"},
        ]
        execution_context = {
            "read_files": {"src/old.py", "src/recent.py"},
            "tool_msg_files": {"r1": ["src/old.py"], "r2": ["src/recent.py"]},
        }
        history_module._trim_tool_history(
            messages, char_budget=1_000, execution_context=execution_context
        )

        self.assertNotIn("src/old.py", execution_context["read_files"])
        self.assertIn("src/recent.py", execution_context["read_files"])

    # ── _stream_chat retry/backoff (transient-failure resilience) ───────────────

    def test_stream_chat_retries_transient_failure_then_succeeds(self) -> None:
        calls = {"n": 0}

        def flaky_chat(**kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return {"content": "ok"}

        fake_backend = types.SimpleNamespace(chat=flaky_chat)
        # Patch sleep so backoff doesn't actually wait during the test.
        with patch.object(streaming_module, "get_backend", return_value=fake_backend), \
             patch.object(streaming_module.time, "sleep", return_value=None):
            result = agent_loop_module._stream_chat(
                model="m", messages=[], tools=[],
                thinking=False, streaming=False, options={},
            )
        self.assertEqual(result, {"content": "ok"})
        self.assertEqual(calls["n"], 3)  # failed twice, succeeded on the third

    def test_stream_chat_reraises_after_exhausting_retries(self) -> None:
        calls = {"n": 0}

        def always_fail(**kwargs):
            calls["n"] += 1
            raise ConnectionError("down")

        fake_backend = types.SimpleNamespace(chat=always_fail)
        with patch.object(streaming_module, "get_backend", return_value=fake_backend), \
             patch.object(streaming_module.time, "sleep", return_value=None):
            with self.assertRaises(ConnectionError):
                agent_loop_module._stream_chat(
                    model="m", messages=[], tools=[],
                    thinking=False, streaming=False, options={},
                )
        # 1 initial attempt + LLM_RETRY_ATTEMPTS retries, all of which ran.
        self.assertEqual(calls["n"], streaming_module.LLM_RETRY_ATTEMPTS + 1)

    def test_stream_chat_does_not_retry_when_already_cancelled(self) -> None:
        calls = {"n": 0}

        def chat(**kwargs):
            calls["n"] += 1
            raise RuntimeError("backend.chat should not be reached when cancelled")

        class _Flag:
            def is_set(self):
                return True

        fake_backend = types.SimpleNamespace(chat=chat)
        with patch.object(streaming_module, "get_backend", return_value=fake_backend):
            with self.assertRaises(asyncio.CancelledError):
                agent_loop_module._stream_chat(
                    model="m", messages=[], tools=[],
                    thinking=False, streaming=False, options={},
                    cancel_flag=_Flag(),
                )
        self.assertEqual(calls["n"], 0)

    # ── _dispatch_tool_calls: writes serialized, reads parallel ─────────────────

    def test_dispatch_serializes_writes_and_parallelizes_reads(self) -> None:
        state = {"active_w": 0, "max_w": 0, "active_r": 0, "max_r": 0}

        class _DispatchAgent:
            tool_caps = {}

            @staticmethod
            def _normalize_arguments(args):
                return args

            @staticmethod
            def _is_write_tool(name):
                return name == "write_file"

            @staticmethod
            def _rewrite_tool_for_context(name, args):
                return name, args

            @staticmethod
            def get_tool_file_targets(name, args):
                # No file targets -> skip the snapshot/diff block (no real IO),
                # so the test isolates the read/write scheduling behaviour.
                return []

            async def _run_tool(self, name, args, execution_context=None, run_auto_validation=True, call_id=""):
                if name == "write_file":
                    state["active_w"] += 1
                    state["max_w"] = max(state["max_w"], state["active_w"])
                    await asyncio.sleep(0.02)
                    state["active_w"] -= 1
                else:
                    state["active_r"] += 1
                    state["max_r"] = max(state["max_r"], state["active_r"])
                    await asyncio.sleep(0.02)
                    state["active_r"] -= 1
                return "{}"

        agent = _DispatchAgent()
        # approvals is accessed as agent.approvals.record_snapshot(path) and
        # agent.approvals._file_snapshots — provide both on a namespace.
        agent.approvals = types.SimpleNamespace(
            record_snapshot=lambda path: None,
            _file_snapshots={},
        )

        tool_calls = [
            {"id": "c1", "function": {"name": "grep", "arguments": {"q": "a"}}},
            {"id": "c2", "function": {"name": "grep", "arguments": {"q": "b"}}},
            {"id": "c3", "function": {"name": "write_file", "arguments": {"path": "x", "content": "1"}}},
            {"id": "c4", "function": {"name": "write_file", "arguments": {"path": "y", "content": "2"}}},
        ]
        messages: list = []
        execution_context = {}

        with patch.object(dispatch_module, "summarize_tool_result", return_value=(True, "")):
            asyncio.run(
                dispatch_module._dispatch_tool_calls(
                    tool_calls, agent, messages, execution_context
                )
            )

        # Writes never overlapped; the two independent reads did.
        self.assertEqual(state["max_w"], 1)
        self.assertEqual(state["max_r"], 2)
        # Tool results are appended in the original call order, not execution order.
        self.assertEqual(
            [m["tool_call_id"] for m in messages],
            ["c1", "c2", "c3", "c4"],
        )

    # ── tool timeout excludes time spent waiting on the user ────────────────────

    def test_timeout_budget_is_not_burned_by_an_approval_wait(self) -> None:
        """A slow *approval* must not be reported as a slow *tool*.

        The approval prompt is raised from inside the tool call, so the naive
        wait_for charged the user's thinking time to the tool: a command approved
        after two minutes came back "timed out after 120s" without having run.
        """
        async def _tool():
            # The real prompt blocks the loop thread; mimic that exactly.
            with human_pause_module.human_pause():
                time.sleep(0.15)
            return "done"

        out = asyncio.run(dispatch_module._await_tool(_tool(), 0.05))
        self.assertEqual(out, "done")

    def test_a_genuinely_slow_tool_still_times_out(self) -> None:
        async def _tool():
            await asyncio.sleep(5)
            return "done"

        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(dispatch_module._await_tool(_tool(), 0.05))

    def test_reported_duration_excludes_the_approval_wait(self) -> None:
        """The row's elapsed time measures the tool, not the user."""
        events: list = []

        class _SlowApprovalAgent:
            tool_caps = {}
            approvals = types.SimpleNamespace(
                record_snapshot=lambda path: None, _file_snapshots={})

            @staticmethod
            def _normalize_arguments(args): return args

            @staticmethod
            def _is_write_tool(name): return False

            @staticmethod
            def _rewrite_tool_for_context(name, args): return name, args

            @staticmethod
            def get_tool_file_targets(name, args): return []

            async def _run_tool(self, name, args, execution_context=None,
                                run_auto_validation=True, call_id=""):
                with human_pause_module.human_pause():
                    time.sleep(0.15)   # user staring at the approval card
                return "{}"

        tool_calls = [{"id": "c1", "function": {"name": "bash_run",
                                                "arguments": {"command": "ls"}}}]
        with patch.object(dispatch_module, "timeout_for", lambda *_a, **_k: 0.05), \
             patch.object(dispatch_module, "emit", events.append), \
             patch.object(dispatch_module, "summarize_tool_result", return_value=(True, "")):
            asyncio.run(dispatch_module._dispatch_tool_calls(
                tool_calls, _SlowApprovalAgent(), [], {}))

        results = [e for e in events if e.get("type") == "tool_result"]
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["ok"])            # not a timeout
        self.assertLess(results[0]["duration_ms"], 100)   # the 150ms wait was the user's

    def test_auto_validate_written_file_is_quiet_without_completeness_issue(self) -> None:
        # The deterministic validator ladder was removed: with no pending replacement
        # to check, the post-write hook stays silent (validation is now steered to the
        # shell via the guidance nudge, not run automatically here).
        result = asyncio.run(
            auto_validate_written_file(
                path="src/module.py",
                execution_context={"workflow_state": "validate"},
                tool_owner={},
                run_tool=lambda tool, args, context: None,
                is_code_filepath=lambda path: True,
                absolute_workspace_path_fn=lambda path: path,
            )
        )
        self.assertEqual(result, "")

    def test_creation_nudge_fires_when_query_signals_creation_but_nothing_written(self) -> None:
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        class _FakeAgent:
            # These nudges live in the guidance layer, which `light` (the default)
            # drops — the test is about the nudge, so it states the level it needs.
            enforcement = "strict"

        execution_context = {
            "workflow_state": "edit",
            "code_mutation_started": False,
            "searched": True,
            "inspected_dirs": {"mimir/servers/utilities"},
            "read_files": {"mimir/servers/utilities/server_math.py"},
            "existing_paths": set(),
            "search_tool_calls": 2,
            "denial_nudges": 0,
            "state_nudges": 0,
            "creation_nudges": 0,
            "denied_tool_calls": [],
            "dirty_written_files": set(),
            "validated_files": set(),
            # The model named what it intends to write — the evidence the creation
            # nudge now requires on top of the (keyword-inferred) create intent.
            "planned_edit_targets": {"mimir/servers/server_science.py"},
        }
        messages: list = []

        fired = nudge_logic.maybe_append_nudge(
            agent=_FakeAgent(),
            query="propose me an implementation for a new scientific computing server",
            active_mode="agent",
            execution_context=execution_context,
            messages=messages,
        )

        self.assertTrue(fired)
        self.assertEqual(len(messages), 1)
        self.assertIn("write target was declared", messages[0]["content"])
        self.assertIn("server_science.py", messages[0]["content"])
        self.assertEqual(execution_context["nudge_counts"]["creation"], 1)

    def _pending_case(self, *, query, execution_context):
        """(probe, fired) for one context — the probe read first, then the real call."""
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        agent = types.SimpleNamespace(enforcement="strict", tool_caps={})
        events: list = []
        with event_sink_module.event_sink(events.append):
            probe = nudge_logic.nudge_pending(
                agent=agent, query=query, active_mode="agent",
                execution_context=execution_context,
            )
        # Probing is read-only: it must not spend a budget or announce anything, or
        # the real call right after it would behave differently for having been asked.
        self.assertEqual(events, [])
        self.assertEqual(execution_context["nudge_counts"], {})

        messages: list = []
        fired = nudge_logic.maybe_append_nudge(
            agent=agent, query=query, active_mode="agent",
            execution_context=execution_context, messages=messages,
        )
        return probe, fired

    def test_the_probe_answers_what_the_real_call_will_do(self) -> None:
        """``nudge_pending`` is read before the model call to decide whether the turn's
        prose can be streamed. If it disagreed with ``maybe_append_nudge``, a refused
        answer would reach the screen anyway — the symptom it exists to prevent."""
        probe, fired = self._pending_case(
            query="propose me an implementation for a new scientific computing server",
            execution_context={
                "workflow_state": "edit",
                "code_mutation_started": False,
                "searched": True,
                "read_files": {"ref.py"},
                "search_tool_calls": 2,
                "denied_tool_calls": [],
                "dirty_written_files": set(),
                "validated_files": set(),
                "planned_edit_targets": {"mimir/servers/server_science.py"},
            },
        )
        self.assertTrue(fired)
        self.assertTrue(probe)

    def test_the_probe_stays_silent_when_nothing_would_fire(self) -> None:
        probe, fired = self._pending_case(
            query="what does this function return?",
            execution_context={
                "workflow_state": "discover",
                "code_mutation_started": False,
                "searched": True,
                "read_files": {"ref.py"},
                "search_tool_calls": 2,
                "denied_tool_calls": [],
                "dirty_written_files": set(),
                "validated_files": set(),
                "planned_edit_targets": set(),
            },
        )
        self.assertFalse(fired)
        self.assertFalse(probe)

    def test_nudge_injection_is_surfaced_as_an_event(self) -> None:
        """Every injected nudge emits ``nudge_injected`` so the run stays readable."""
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        class _FakeAgent:
            # These nudges live in the guidance layer, which `light` (the default)
            # drops — the test is about the nudge, so it states the level it needs.
            enforcement = "strict"

        execution_context = {
            "workflow_state": "edit",
            "code_mutation_started": False,
            "searched": True,
            "read_files": {"ref.py"},
            "search_tool_calls": 2,
            "denied_tool_calls": [],
            "dirty_written_files": set(),
            "validated_files": set(),
            "planned_edit_targets": {"helper.py"},
        }
        events: list = []
        with event_sink_module.event_sink(events.append):
            fired = nudge_logic.maybe_append_nudge(
                agent=_FakeAgent(),
                query="create a new helper module",
                active_mode="agent",
                execution_context=execution_context,
                messages=[],
            )

        self.assertTrue(fired)
        self.assertEqual([e["type"] for e in events], ["nudge_injected"])
        self.assertEqual(events[0]["category"], "creation")
        self.assertIn("write target was declared", events[0]["text"])

    def test_every_injection_point_announces_itself_not_just_the_table(self) -> None:
        """The loop-control reminders must emit the event too.

        The webview holds the turn in flight outside the transcript and drops it on
        ``nudge_injected``. A reminder injected silently leaves the rejected prose on
        screen as if it were the answer, then swaps it for a different one when the
        real answer lands — which is the display bug that survived the draft fix,
        because only the nudge table went through the emitting path.
        """
        import importlib
        nudges = importlib.import_module("mimir.client.guardrails.nudges")

        messages: list = []
        events: list = []
        with event_sink_module.event_sink(events.append):
            nudges.inject_reminder(
                messages, "wrap it up", category="step_limit", tagged=False,
            )

        self.assertEqual([e["type"] for e in events], ["nudge_injected"])
        self.assertEqual(events[0]["category"], "step_limit")
        # Untagged: the plan/loop-control reminders are protocol, not advice, and the
        # "apply judgment" banner invites the model to skip a step the loop requires.
        self.assertEqual(messages, [{"role": "user", "content": "wrap it up"}])

    def test_creation_nudge_silent_without_a_declared_write_target(self) -> None:
        """Reading files while answering a question must not look like create intent.

        Regression: a query mentioning "a new machine" matched the create vocabulary,
        and `read_files` alone satisfied the old predicate — so the nudge fired on a
        finished answer and pushed the model into writing an unrequested script.
        """
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        class _FakeAgent:
            # These nudges live in the guidance layer, which `light` (the default)
            # drops — the test is about the nudge, so it states the level it needs.
            enforcement = "strict"

        execution_context = {
            "workflow_state": "discover",
            "code_mutation_started": False,
            "searched": True,
            "inspected_dirs": {"mimir/scripts"},
            "read_files": {"pyproject.toml", "mimir/requirements.txt"},
            "existing_paths": set(),
            "search_tool_calls": 2,
            "denied_tool_calls": [],
            "dirty_written_files": set(),
            "validated_files": set(),
            "planned_edit_targets": set(),
        }
        messages: list = []

        fired = nudge_logic.maybe_append_nudge(
            agent=_FakeAgent(),
            query="what are the pip packages required to install the kb on a new machine?",
            active_mode="agent",
            execution_context=execution_context,
            messages=messages,
        )

        self.assertFalse(fired)
        self.assertEqual(messages, [])

        # Second call: budget exhausted, nudge must not fire again.
        messages.clear()
        fired_again = nudge_logic.maybe_append_nudge(
            agent=_FakeAgent(),
            query="propose me an implementation for a new scientific computing server",
            active_mode="agent",
            execution_context=execution_context,
            messages=messages,
        )
        self.assertFalse(fired_again)
        self.assertEqual(len(messages), 0)

    def test_todo_nudge_op_count_trigger_single_file(self) -> None:
        # A task that is many substantive operations but only ONE touched file still
        # warrants a checklist — the action_op_count trigger fires it.
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        base = {
            "workflow_state": "edit",
            "code_mutation_started": True,
            "dirty_written_files": {"module.py"},   # single file → multi-file trigger off
            "planned_edit_targets": set(),
            "todo_written": False,
            "nudge_counts": {},
        }
        # Below the op threshold with one file: nothing warrants a checklist yet.
        below = {**base, "action_op_count": 4}
        self.assertFalse(
            nudge_logic._should_nudge_todo(below, level="strict", active_mode="agent")
        )
        # At the op threshold: the many-ops trigger fires even with a single file.
        at = {**base, "action_op_count": 5}
        self.assertTrue(
            nudge_logic._should_nudge_todo(at, level="strict", active_mode="agent")
        )

    def test_todo_nudge_suppressed_at_conclude(self) -> None:
        # A checklist emitted after the work is done guides nothing: the many-ops
        # trigger must not fire once the workflow has reached 'conclude'.
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        concluding = {
            "workflow_state": "conclude",
            "code_mutation_started": True,
            "dirty_written_files": {"module.py"},
            "planned_edit_targets": set(),
            "action_op_count": 50,          # well past the op threshold
            "todo_written": False,
            "nudge_counts": {},
        }
        self.assertFalse(
            nudge_logic._should_nudge_todo(concluding, level="strict", active_mode="agent")
        )

    def test_todo_nudge_multifile_trigger_unchanged(self) -> None:
        # The classic multi-file trigger keeps working with zero action ops recorded.
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        ctx = {
            "workflow_state": "edit",
            "code_mutation_started": True,
            "dirty_written_files": {"a.py", "b.py", "c.py"},
            "planned_edit_targets": set(),
            "action_op_count": 0,
            "todo_written": False,
            "nudge_counts": {},
        }
        self.assertTrue(
            nudge_logic._should_nudge_todo(ctx, level="strict", active_mode="agent")
        )

    def test_todo_nudge_disabled_at_light(self) -> None:
        # "todo" guidance is a strict-only category; the op trigger must not leak into light.
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        ctx = {
            "workflow_state": "edit",
            "code_mutation_started": True,
            "dirty_written_files": {"module.py"},
            "planned_edit_targets": set(),
            "action_op_count": 20,
            "todo_written": False,
            "nudge_counts": {},
        }
        self.assertFalse(
            nudge_logic._should_nudge_todo(ctx, level="light", active_mode="agent")
        )

    def test_an_unjudged_run_alone_nudges_nothing(self) -> None:
        # There is deliberately no reminder asking for a verdict: it fired on the happy
        # path (edit, run green, answer), discarded a finished answer, and bought a
        # label the ledger already prints. The run is reported unjudged instead.
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        agent = types.SimpleNamespace(enforcement="strict", tool_caps={})
        ctx = nudge_logic._bootstrap_nudge_context({
            "runs": {"python solver.py": {"completed": True, "verdict": ""}},
        })
        messages: list[dict] = []
        fired = nudge_logic.maybe_append_nudge(
            agent=agent, query="run it", active_mode="agent",
            execution_context=ctx, messages=messages,
        )
        self.assertFalse(fired)
        self.assertEqual(messages, [])

    def test_a_run_never_makes_a_file_pending_a_check(self) -> None:
        # The two axes do not interact: an unjudged run leaves the file exactly as
        # un-checked as it was, so the validation nudge has nothing to defer to.
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        ctx = {
            "workflow_state": "validate",
            "code_mutation_started": True,
            "dirty_written_files": {"solver.py"},
            "validated_files": set(),
            "declared_edit_set": set(),
            "steps_since_last_edit": 5,
            "runs": {"python solver.py": {"completed": True, "verdict": ""}},
            "builtin_check_findings": {"solver.py": "line 3: '{' is never closed"},
            "nudge_counts": {},
        }
        self.assertTrue(
            nudge_logic._should_nudge_validation(ctx, level="strict", active_mode="agent")
        )
        ctx["validated_files"] = {"solver.py"}
        self.assertFalse(
            nudge_logic._should_nudge_validation(ctx, level="strict", active_mode="agent")
        )

    def test_discovery_evidence_requires_two_distinct_signals(self) -> None:
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")
        # A single weak signal no longer clears the gate (so the nudge still drives).
        self.assertFalse(nudge_logic._has_local_discovery_evidence({"searched": True}))
        # Two distinct signals count as real exploration.
        self.assertTrue(nudge_logic._has_local_discovery_evidence(
            {"searched": True, "read_files": {"a.py"}}
        ))
        # A directory the model inspected itself counts, as a second signal —
        # structural discovery is real exploration, not a lesser kind of it.
        self.assertTrue(nudge_logic._has_local_discovery_evidence(
            {"inspected_dirs": {"mimir/servers"}, "searched": True}
        ))

    def test_discovery_nudge_fires_with_only_one_signal(self) -> None:
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")
        execution_context = {
            "workflow_state": "discover",
            "searched": True,  # only one signal → gate not cleared
            "denied_tool_calls": [],
            "dirty_written_files": set(),
            "validated_files": set(),
        }
        messages: list = []

        class _FakeAgent:
            # These nudges live in the guidance layer, which `light` (the default)
            # drops — the test is about the nudge, so it states the level it needs.
            enforcement = "strict"

        fired = nudge_logic.maybe_append_nudge(
            agent=_FakeAgent(),
            query="refactor the solver module in the repository",
            active_mode="agent",
            execution_context=execution_context,
            messages=messages,
        )
        self.assertTrue(fired)
        self.assertEqual(execution_context["nudge_counts"]["discovery"], 1)

    # ── single discovery-evidence owner (Move A) ──────────────────────────────

    def test_has_discovery_evidence_shared_predicate(self) -> None:
        import importlib
        ec = importlib.import_module("mimir.client.context.execution_context")
        self.assertEqual(ec.discovery_signal_count({"searched": True}), 1)
        self.assertEqual(ec.discovery_signal_count({"searched": True, "read_files": {"a.py"}}), 2)
        self.assertTrue(ec.has_discovery_evidence({"searched": True}, min_distinct=1))
        self.assertFalse(ec.has_discovery_evidence({"searched": True}, min_distinct=2))
        # inspected_dirs counts, but only past the baseline seed: a snapshot must not
        # pre-satisfy a gate, while a subtree the model actually inspected is real
        # exploration (excluding the field outright made structural discovery worth
        # nothing, so only a grep or a file read could ever clear a gate).
        self.assertIn("inspected_dirs", ec.DISCOVERY_EVIDENCE_SIGNALS)
        self.assertEqual(ec.discovery_signal_count({}), 0)
        self.assertEqual(ec.discovery_signal_count({"inspected_dirs": {"mimir/servers"}}), 1)
        self.assertEqual(ec.discovery_signal_count({"inspected_dirs": {".", "mimir/servers"}}), 1)

    # ── model-tiered enforcement (Move B) ─────────────────────────────────────

    def test_enforcement_level_resolution(self) -> None:
        import importlib
        from unittest.mock import patch
        models = importlib.import_module("mimir.client.config.models")
        # Unknown model → light default. The `light` set is the one defined by the
        # right criterion (costly, hard to detect, non-self-correcting), so it is the
        # default and `strict` is the opt-in for models that need the rails.
        self.assertEqual(models.enforcement_level("no-such-model"), "light")
        # Valid profile values resolve; bad values fall back to the default.
        with patch.object(models, "profile_for_model", return_value={"enforcement": "off"}):
            self.assertEqual(models.enforcement_level("m"), "off")
        with patch.object(models, "profile_for_model", return_value={"enforcement": "strict"}):
            self.assertEqual(models.enforcement_level("m"), "strict")
        with patch.object(models, "profile_for_model", return_value={"enforcement": "bogus"}):
            self.assertEqual(models.enforcement_level("m"), "light")

    def test_a_model_known_to_need_rails_opts_back_into_strict(self) -> None:
        """The opt-in must be exercised by a real profile, not just be possible.

        Flipping the default is only safe if the models that were empirically found
        fragile still get the full guidance layer. Devstral is the one carrying a
        recorded workaround in its profile (a tool-count cap), so it is the marker
        case: if this drops out of the profiles, the flip has silently un-railed it.
        """
        import importlib
        models = importlib.import_module("mimir.client.config.models")
        self.assertEqual(models.enforcement_level("Devstral-Small-24B"), "strict")

    def test_discovery_nudge_suppressed_when_not_strict(self) -> None:
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")
        execution_context = {
            "workflow_state": "discover",
            "denied_tool_calls": [],
            "dirty_written_files": set(),
            "validated_files": set(),
        }

        class _FakeAgent:
            model = "big-model"

        for level in ("light", "off"):
            execution_context["nudge_counts"] = {}
            messages: list = []
            agent = _FakeAgent()
            agent.enforcement = level
            fired = nudge_logic.maybe_append_nudge(
                agent=agent,
                query="refactor the solver module in the repository",
                active_mode="agent",
                execution_context=execution_context,
                messages=messages,
            )
            self.assertFalse(fired, f"discovery nudge should be suppressed at level={level}")
            self.assertEqual(messages, [])

    def test_verification_nudge_fires_even_when_enforcement_off(self) -> None:
        # Verification reminders check reality (here: a blocking denial) and must run
        # regardless of model strength — they are NOT gated by enforcement level.
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")
        execution_context = {
            "workflow_state": "edit",
            "denied_tool_calls": [{"tool": "write_file"}],
            "dirty_written_files": set(),
            "validated_files": set(),
            "nudge_counts": {},
        }
        messages: list = []

        class _FakeAgent:
            model = "big-model"
            enforcement = "off"

        fired = nudge_logic.maybe_append_nudge(
            agent=_FakeAgent(),
            query="please add the new feature",
            active_mode="agent",
            execution_context=execution_context,
            messages=messages,
        )
        self.assertTrue(fired, "denial (verification) nudge must fire even at enforcement=off")
        self.assertEqual(execution_context["nudge_counts"]["denial"], 1)

    def test_guidance_nudge_gated_by_enforcement_off(self) -> None:
        # A guidance nudge (creation) fires at strict but is fully suppressed at off.
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        class _FakeAgent:
            def __init__(self, enforcement: str) -> None:
                self.model = "big-model"
                self.enforcement = enforcement

        def _fresh_ctx() -> dict:
            # Two discovery signals (searched + read_files) so the discovery branch is
            # cleared and the creation branch is the one under test.
            return {
                "workflow_state": "edit",
                "searched": True,
                "read_files": {"ref.py"},
                "denied_tool_calls": [],
                "dirty_written_files": set(),
                "validated_files": set(),
                "planned_edit_targets": {"helper.py"},
                "code_mutation_started": False,
                "nudge_counts": {},
            }

        # strict → creation guidance fires
        ctx = _fresh_ctx()
        messages: list = []
        fired = nudge_logic.maybe_append_nudge(
            agent=_FakeAgent("strict"),
            query="create a new helper module",
            active_mode="agent",
            execution_context=ctx,
            messages=messages,
        )
        self.assertTrue(fired, "creation guidance nudge should fire at enforcement=strict")
        self.assertEqual(ctx["nudge_counts"]["creation"], 1)

        # off → guidance layer skipped entirely
        ctx = _fresh_ctx()
        messages = []
        fired = nudge_logic.maybe_append_nudge(
            agent=_FakeAgent("off"),
            query="create a new helper module",
            active_mode="agent",
            execution_context=ctx,
            messages=messages,
        )
        self.assertFalse(fired, "guidance nudge must be suppressed at enforcement=off")
        self.assertEqual(messages, [])

    def test_light_subset_drops_procedural_guidance_but_keeps_blast_radius(self) -> None:
        # The deliberate "light" line (agent mode): procedural nudges (creation/state/
        # todo/doc) are dropped, while blast_radius (and env_cleanup) survive.
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        class _Agent:
            model = "big-model"
            enforcement = "light"

        # A context+query that fires the creation nudge at strict (cf. the test above).
        creation_ctx = {
            "workflow_state": "edit",
            "searched": True,
            "read_files": {"ref.py"},
            "denied_tool_calls": [],
            "dirty_written_files": set(),
            "validated_files": set(),
            "planned_edit_targets": {"helper.py"},
            "code_mutation_started": False,
            "nudge_counts": {},
        }
        fired = nudge_logic.maybe_append_nudge(
            agent=_Agent(),
            query="create a new helper module",
            active_mode="agent",
            execution_context=creation_ctx,
            messages=[],
        )
        self.assertFalse(fired, "creation (procedural) must NOT fire at light")
        self.assertIsNone(creation_ctx["nudge_counts"].get("creation"))

        # blast_radius IS in the light subset: edit intent, a declared target, files
        # read, no search yet.
        blast_ctx = {
            "workflow_state": "edit",
            "read_files": {"mod.py"},
            "search_tool_calls": 0,
            "denied_tool_calls": [],
            "dirty_written_files": set(),
            "validated_files": set(),
            "planned_edit_targets": {"mod.py"},
            "code_mutation_started": False,
            "nudge_counts": {},
        }
        fired = nudge_logic.maybe_append_nudge(
            agent=_Agent(),
            query="refactor the existing parser",
            active_mode="agent",
            execution_context=blast_ctx,
            messages=[],
        )
        self.assertTrue(fired, "blast_radius must fire at light")
        self.assertEqual(blast_ctx["nudge_counts"].get("blast_radius"), 1)

    def test_light_plan_fires_no_guidance(self) -> None:
        # light-plan = no guidance nudges (plan relies on its explore phase).
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        class _Agent:
            model = "big-model"
            enforcement = "light"

        blast_ctx = {
            "workflow_state": "edit",
            "read_files": {"mod.py"},
            "search_tool_calls": 0,
            "denied_tool_calls": [],
            "dirty_written_files": set(),
            "validated_files": set(),
            "code_mutation_started": False,
            "nudge_counts": {},
        }
        fired = nudge_logic.maybe_append_nudge(
            agent=_Agent(),
            query="refactor the existing parser",
            active_mode="plan",
            execution_context=blast_ctx,
            messages=[],
        )
        self.assertFalse(fired, "no guidance nudge should fire in plan mode at light")

    def test_regression_nudge_fires_for_untested_edited_source(self) -> None:
        # Edited source with an existing, un-run test → verification nudge, even at off.
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        class _FakeAgent:
            model = "big-model"
            enforcement = "off"

        # source validated (no pending validation) but its existing test was never run.
        ctx = {
            "workflow_state": "validate",
            "dirty_written_files": {"foo.py"},
            "validated_files": {"foo.py"},
            "existing_paths": {"tests/test_foo.py"},
            "tests_run": set(),
            "denied_tool_calls": [],
            "nudge_counts": {},
        }
        messages: list = []
        fired = nudge_logic.maybe_append_nudge(
            agent=_FakeAgent(),
            query="fix the bug in foo",
            active_mode="agent",
            execution_context=ctx,
            messages=messages,
        )
        self.assertTrue(fired, "regression nudge should fire for an un-run existing test")
        self.assertEqual(ctx["nudge_counts"]["exercise"], 1)
        self.assertIn("test_foo.py", messages[0]["content"])

    def test_regression_nudge_silent_when_test_was_run(self) -> None:
        import importlib
        nudge_logic = importlib.import_module("mimir.client.guardrails.nudges.engine")

        class _FakeAgent:
            model = "big-model"
            enforcement = "off"

        ctx = {
            "workflow_state": "validate",
            "dirty_written_files": {"foo.py"},
            "validated_files": {"foo.py"},
            "existing_paths": {"tests/test_foo.py"},
            "tests_run": {"tests/test_foo.py"},  # already executed this session
            "denied_tool_calls": [],
            "nudge_counts": {},
        }
        messages: list = []
        fired = nudge_logic.maybe_append_nudge(
            agent=_FakeAgent(),
            query="fix the bug in foo",
            active_mode="agent",
            execution_context=ctx,
            messages=messages,
        )
        self.assertFalse(fired, "regression nudge must stay silent once the test was run")
        self.assertEqual(messages, [])

    def test_auto_store_memory_skips_readonly_turns(self) -> None:
        stored: list[dict] = []

        async def fake_run_tool(tool, args, ctx=None):
            stored.append({"tool": tool, "args": args})
            return '{"status": "ok"}'

        execution_context = client_module.MimirAgent._new_execution_context()
        # No files written — pure read-only turn.
        asyncio.run(
            context_builder_module.auto_store_memory(
                query="what does the policy module do?",
                answer="It enforces write and state guards.",
                tool_owner={"memory_add": "MemoryServer"},
                run_tool=fake_run_tool,
                truncate_text=lambda t, n=600: t[:n],
                execution_context=execution_context,
            )
        )
        self.assertEqual(stored, [], "should not store on read-only turn")

    def test_auto_store_memory_skips_code_write_without_recall_signal(self) -> None:
        # Writing code files is the normal case and must NOT trigger auto-storage —
        # only an explicit recall signal does. See auto_store_memory docstring.
        stored: list[dict] = []

        async def fake_run_tool(tool, args, ctx=None):
            stored.append({"tool": tool, "args": args})
            return '{"status": "ok"}'

        execution_context = client_module.MimirAgent._new_execution_context()
        execution_context["dirty_written_files"].add("mimir/servers/utilities/server_math.py")

        asyncio.run(
            context_builder_module.auto_store_memory(
                query="add a square_root tool",
                answer="Done — wrote server_math.py.",
                tool_owner={"memory_add": "MemoryServer"},
                run_tool=fake_run_tool,
                truncate_text=lambda t, n=600: t[:n],
                execution_context=execution_context,
            )
        )
        self.assertEqual(stored, [], "writing code files alone must not trigger auto-storage")

    def test_auto_store_memory_skips_common_task_verbs(self) -> None:
        # Ordinary task verbs that used to be recall signals ("save"/"keep") must
        # no longer trip auto-storage.
        stored: list[dict] = []

        async def fake_run_tool(tool, args, ctx=None):
            stored.append({"tool": tool, "args": args})
            return '{"status": "ok"}'

        execution_context = client_module.MimirAgent._new_execution_context()
        execution_context["dirty_written_files"].add("waveSolver/snap.py")

        asyncio.run(
            context_builder_module.auto_store_memory(
                query="save the snapshot to png since I'm on hpc",
                answer="Done.",
                tool_owner={"memory_add": "MemoryServer"},
                run_tool=fake_run_tool,
                truncate_text=lambda t, n=600: t[:n],
                execution_context=execution_context,
            )
        )
        self.assertEqual(stored, [], "'save' is a task verb, not a recall signal")

    def test_auto_store_memory_persists_on_explicit_recall_signal(self) -> None:
        stored: list[dict] = []

        async def fake_run_tool(tool, args, ctx=None):
            stored.append({"tool": tool, "args": args})
            return '{"status": "ok"}'

        execution_context = client_module.MimirAgent._new_execution_context()
        # No files written but query has a recall signal.
        asyncio.run(
            context_builder_module.auto_store_memory(
                query="remember that we use qwen3 as default model",
                answer="Noted.",
                tool_owner={"memory_add": "MemoryServer"},
                run_tool=fake_run_tool,
                truncate_text=lambda t, n=600: t[:n],
                execution_context=execution_context,
            )
        )
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["tool"], "memory_add")

    def test_build_memory_summary_is_compact_fact(self) -> None:
        execution_context = client_module.MimirAgent._new_execution_context()
        execution_context["dirty_written_files"].add("add.py")
        execution_context["read_files"].add("README.md")

        answer = (
            "I implemented add.py with an add(a, b) function. "
            "We need to also consider edge cases.\n"
            "Then we should output a final answer summarizing what we did."
        )
        summary = context_builder_module._build_memory_summary(
            "implement a+b",
            answer,
            execution_context,
            truncate_text=lambda t, n: t[:n],
        )
        self.assertIn("Task: implement a+b", summary)
        self.assertIn("Files written: add.py", summary)
        # Outcome is the first sentence only — no deliberation, no process noise.
        self.assertIn("Outcome: I implemented add.py with an add(a, b) function.", summary)
        self.assertNotIn("we should output", summary)
        self.assertNotIn("Files read", summary)
        self.assertNotIn("Searches run", summary)

    def test_load_recent_memories_returns_last_n_entries(self) -> None:
        import tempfile
        # MEMORY.md index format, newest first.
        lines = ["# Memory Index", ""]
        for i in range(15, 0, -1):
            lines.append(f"- [fact {i}](fact-{i}.md) — 2026-01-{i:02d}")
        md_content = "\n".join(lines) + "\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(md_content)
            tmp_path = f.name
        try:
            result = context_builder_module._load_recent_memories(tmp_path, max_entries=5)
            self.assertEqual(len(result), 5)
            self.assertEqual(result[0]["text"], "fact 15")
            self.assertEqual(result[-1]["text"], "fact 11")
        finally:
            import os; os.unlink(tmp_path)

    def test_load_recent_memories_returns_empty_for_missing_file(self) -> None:
        result = context_builder_module._load_recent_memories("/nonexistent/path/memory.md")
        self.assertEqual(result, [])

    def test_build_system_content_injects_memories_when_present(self) -> None:
        import tempfile
        md_content = (
            "# Memory Index\n\n"
            "- [user prefers qwen3 model](user-prefers-qwen3-model.md) — 2026-03-15\n"
            "- [project uses pytest for validation](project-uses-pytest.md) — 2026-03-01\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(md_content)
            tmp_path = f.name
        try:
            content = context_builder_module.build_system_content(
                active_mode="agent",
                tool_owner={},
                sensitive_tools=set(),
                memory_context_file=tmp_path,
            )
            self.assertIn("Memory index", content)
            self.assertIn("user prefers qwen3 model", content)
            self.assertIn("project uses pytest for validation", content)
            self.assertIn("Index at:", content)
        finally:
            import os; os.unlink(tmp_path)

    def test_build_system_content_falls_back_when_no_memories(self) -> None:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")  # empty file — no entries
            tmp_path = f.name
        try:
            content = context_builder_module.build_system_content(
                active_mode="agent",
                tool_owner={},
                sensitive_tools=set(),
                memory_context_file=tmp_path,
            )
            self.assertIn("Persistent memories are stored under", content)
            self.assertNotIn("Memory index", content)
        finally:
            import os; os.unlink(tmp_path)

    # ── foundational context injection ────────────────────────────────────────
    #
    # Two absolute paths, and nothing describing the repo's contents or the machine.
    # The repo-structure snapshot and the hardware probe that used to be injected here
    # are gone: the snapshot pre-filled a discovery-evidence field, so a gate could be
    # satisfied before the model had done anything, and neither block told the model
    # what its task actually touches.

    def test_no_repo_or_hardware_description_is_injected(self) -> None:
        for mode in ("agent", "plan", "ask"):
            content = context_builder_module.build_system_content(
                active_mode=mode, tool_owner={}, sensitive_tools=set(),
            )
            with self.subTest(mode=mode):
                self.assertNotIn("Repository structure", content)
                self.assertNotIn("Target platform", content)

    def test_absolute_paths_are_injected_in_every_mode(self) -> None:
        for mode in ("agent", "plan", "ask"):
            content = context_builder_module.build_system_content(
                active_mode=mode, tool_owner={}, sensitive_tools=set(),
            )
            with self.subTest(mode=mode):
                self.assertIn("Workspace root (absolute):", content)
                self.assertIn("Scratchpad", content)

    def test_build_system_content_injects_checklist_in_agent_mode(self) -> None:
        content = context_builder_module.build_system_content(
            active_mode="agent",
            tool_owner={},
            sensitive_tools=set(),
            plan_todos=["Read file", "Apply patch", "Run tests"],
        )
        self.assertIn("Task checklist", content)
        self.assertIn("[ ] Read file", content)
        self.assertIn("[ ] Apply patch", content)
        self.assertIn("[ ] Run tests", content)

    def test_build_system_content_no_checklist_in_plan_mode(self) -> None:
        content = context_builder_module.build_system_content(
            active_mode="plan",
            tool_owner={},
            sensitive_tools=set(),
            plan_todos=["Read file", "Apply patch"],
        )
        self.assertNotIn("Task checklist", content)

    def test_build_system_content_plan_mode_requires_own_exploration(self) -> None:
        content = context_builder_module.build_system_content(
            active_mode="plan",
            tool_owner={},
            sensitive_tools=set(),
        )
        self.assertIn("Current mode: PLAN", content)
        # The mode used to say the injected repo/hardware blocks were "orientation
        # only"; with no such block left, it states the same duty positively.
        self.assertIn("Nothing in your context tells you what has to change", content)
        self.assertIn("exploration tools", content)

    def test_build_system_content_ask_mode_is_readonly_qa(self) -> None:
        content = context_builder_module.build_system_content(
            active_mode="ask",
            tool_owner={"write_file": "fs"},
            sensitive_tools=set(),
            plan_todos=["Read file", "Apply patch"],
        )
        self.assertIn("Current mode: ASK", content)
        self.assertIn("exploration tools", content)
        # No checklist (agent-only) and no planning tool catalog (plan-only).
        self.assertNotIn("Task checklist", content)
        self.assertNotIn("Available tools by server", content)
        self.assertNotIn("Current mode: PLAN", content)

    def test_is_proceed_signal_detects_start_implementation(self) -> None:
        self.assertTrue(chat_session_module._is_proceed_signal("Start implementation"))
        self.assertTrue(chat_session_module._is_proceed_signal("proceed"))
        self.assertTrue(chat_session_module._is_proceed_signal("go ahead"))

    def test_is_proceed_signal_rejects_long_queries(self) -> None:
        self.assertFalse(chat_session_module._is_proceed_signal(
            "start implementation but only for the memory module"
        ))

    def test_is_proceed_signal_rejects_unrelated_queries(self) -> None:
        self.assertFalse(chat_session_module._is_proceed_signal("what does the policy module do?"))

    # ── verification ledger in the terminal ───────────────────────────────────

    def _ledger_block(self) -> str:
        from mimir.client.query_engine.finalize import _annotate_answer_with_changes
        from mimir.client.query_engine.verification import split_answer_ledger

        ec = client_module.MimirAgent._new_execution_context()
        ec["dirty_written_files"].add("solver.py")
        return split_answer_ledger(_annotate_answer_with_changes("Done.", ec))[1]

    def test_cli_ledger_collapses_to_one_line_pointing_at_the_command(self) -> None:
        line = chat_session_module.format_ledger_summary(self._ledger_block())
        self.assertEqual(len(line.strip().splitlines()), 1)
        self.assertIn("1 file", line)
        self.assertIn("/ledger", line)

    def test_ledger_command_expands_the_last_ledger(self) -> None:
        from mimir.client.ui.cli.chat_commands import handle_chat_command

        block = self._ledger_block()

        async def _run(show):
            return await handle_chat_command(
                query="/ledger", mode="agent", thinking=False, streaming=False,
                batch_mode=False, set_mode=lambda v: None, set_thinking=lambda v: None,
                set_streaming=lambda v: None, set_batch_mode=lambda v: None,
                show_ledger=show,
            )

        handled, message = asyncio.run(_run(
            lambda: chat_session_module.format_ledger_full(block)))
        self.assertTrue(handled)
        self.assertIn("solver.py", message)

        # No ledger yet (a read-only turn): the command says so instead of erroring.
        handled, message = asyncio.run(_run(lambda: None))
        self.assertTrue(handled)
        self.assertIn("No verification ledger", message)

    def test_cli_ledger_expands_without_markdown_noise(self) -> None:
        # The terminal gets the rows plain — no backticks, no bold markers.
        full = chat_session_module.format_ledger_full(self._ledger_block())
        self.assertIn("solver.py — not checked", full)
        self.assertNotIn("`", full)
        self.assertNotIn("**", full)

    # ── live todo checklist ───────────────────────────────────────────────────

    def test_load_todo_items_returns_items_from_file(self) -> None:
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("- [ ] Read file\n- [x] Apply patch\n")
            tmp_path = f.name
        try:
            result = context_builder_module._load_todo_items(tmp_path)
            self.assertEqual(len(result), 2)
            self.assertFalse(result[0]["done"])
            self.assertTrue(result[1]["done"])
        finally:
            os.unlink(tmp_path)

    def test_load_todo_items_returns_empty_for_missing_file(self) -> None:
        result = context_builder_module._load_todo_items("/nonexistent/todo_list.md")
        self.assertEqual(result, [])

    def test_build_system_content_injects_live_checklist_from_todo_file(self) -> None:
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("- [x] Read file\n- [ ] Apply patch\n")
            tmp_path = f.name
        try:
            content = context_builder_module.build_system_content(
                active_mode="agent",
                tool_owner={},
                sensitive_tools=set(),
                todo_file=tmp_path,
            )
            self.assertIn("Task checklist", content)
            self.assertIn("[x] Read file", content)
            self.assertIn("[ ] Apply patch", content)
            self.assertIn("1 pending", content)
        finally:
            os.unlink(tmp_path)

    def test_build_system_content_live_checklist_takes_priority_over_plan_todos(self) -> None:
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("- [ ] live step\n")
            tmp_path = f.name
        try:
            content = context_builder_module.build_system_content(
                active_mode="agent",
                tool_owner={},
                sensitive_tools=set(),
                todo_file=tmp_path,
                plan_todos=["static step"],
            )
            self.assertIn("live step", content)
            self.assertNotIn("static step", content)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()