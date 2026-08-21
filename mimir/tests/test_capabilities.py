"""Unit tests for the tool-capability registry (context/capabilities.py).

The *expected classification* now lives in ``_golden_caps.py`` and is verified
against the live server declarations by ``test_phase_b_servers.py``. This file
covers the registry's pure logic: ``infer_tool_caps`` precedence, the server-side
``tool_caps`` helper round-trip, the unannotated-tool report, and ``readonly_servers``.
"""

import types
import unittest

from mimir.client.context import capabilities as caps


class InferencePrecedenceTest(unittest.TestCase):
    """infer_tool_caps resolves meta > annotations > default (no static fallback)."""

    @staticmethod
    def _tool(name, *, meta=None, annotations=None, schema=None):
        return types.SimpleNamespace(
            name=name, meta=meta, annotations=annotations, inputSchema=schema,
        )

    def test_meta_wins(self):
        t = self._tool(
            "custom_tool",
            meta={"mimir": {
                "capabilities": ["read", "search"],
                "arg_roles": {"path": ["path"]},
                "approval": {"sensitive": True, "non_batch": True, "fallbacks": ["grep"]},
                "label": "Doing {path}",
            }},
            annotations={"readOnlyHint": True},  # ignored when meta present
        )
        c = caps.infer_tool_caps(t)
        self.assertEqual(c.capabilities, frozenset({"read", "search", "sensitive", "non_batch"}))
        self.assertEqual(c.arg_roles, {"path": ("path",)})
        self.assertEqual(c.fallbacks, ("grep",))
        self.assertEqual(c.label, "Doing {path}")

    def test_annotations_layer(self):
        ro = caps.infer_tool_caps(self._tool("foreign_read", annotations={"readOnlyHint": True}))
        self.assertIn(caps.CACHEABLE, ro.capabilities)
        self.assertNotIn(caps.PLAN_BLOCKED, ro.capabilities)

        destr = caps.infer_tool_caps(self._tool(
            "foreign_write", annotations={"readOnlyHint": False, "destructiveHint": True},
        ))
        self.assertIn(caps.PLAN_BLOCKED, destr.capabilities)
        self.assertIn(caps.SENSITIVE, destr.capabilities)
        self.assertIn(caps.NON_BATCH, destr.capabilities)

    def test_default_layer_is_empty(self):
        # No meta and no annotations -> no caps (there is no static fallback table).
        c = caps.infer_tool_caps(self._tool("write_file"))
        self.assertEqual(c.capabilities, frozenset())

    def test_default_layer_schema_paths(self):
        c = caps.infer_tool_caps(self._tool(
            "brand_new_tool",
            schema={"properties": {"path": {"type": "string"}, "n": {"type": "integer"}}},
        ))
        self.assertEqual(c.capabilities, frozenset())
        self.assertEqual(c.arg_roles.get("path"), ("path",))


class RegistryHelpersTest(unittest.TestCase):
    def test_empty_registry_default(self):
        # With no registry passed, everything is unclassified (no static fallback).
        self.assertEqual(caps.names_with_cap(caps.SENSITIVE), set())
        self.assertFalse(caps.has_cap("write_file", caps.EDIT))
        self.assertEqual(caps.path_args("read_file"), ())

    def test_unannotated_live_tools(self):
        reg = {
            "read_file": caps.ToolCaps(name="read_file", capabilities=frozenset({caps.READ})),
            "list_files": caps.ToolCaps(name="list_files", arg_roles={"path": ("subdir",)}),
            "add": caps.ToolCaps(name="add"),          # pure tool -> flagged
            "mystery": caps.ToolCaps(name="mystery"),  # forgot to declare -> flagged
        }
        self.assertEqual(caps.unannotated_live_tools(reg), ["add", "mystery"])

    def test_readonly_servers(self):
        self.assertEqual(
            caps.readonly_servers(),
            frozenset({"files", "search", "code", "math", "strings",
                       "datetime", "memory", "system", "platform"}),
        )


class ServerDeclarationTest(unittest.TestCase):
    """The server-side tool_caps() helper round-trips through infer_tool_caps."""

    def test_vocab_in_sync(self):
        from mimir.servers._shared import capabilities as srv
        client_vocab = {
            caps.READ, caps.CACHEABLE, caps.SEARCH, caps.SEARCH_WITH_PATH,
            caps.CANDIDATE_SEARCH, caps.INSPECT_DIR, caps.CHECK_EXISTENCE,
            caps.CONTENT_WRITE, caps.EDIT,
            caps.REPLACEMENT_TRACK, caps.VALIDATE,
            caps.PLAN_BLOCKED, caps.PLAN_READONLY, caps.CODE_EXEC, caps.SENSITIVE, caps.NON_BATCH,
            caps.CODE_NAV, caps.ENV_DISCOVERY, caps.EXTERNAL_FETCH, caps.CLUSTER_SUBMIT,
            caps.ENV_MUTATE, caps.BACKGROUNDABLE,
            caps.REMOVE, caps.OVERWRITE, caps.TASK_PLANNING, caps.JUDGE,
        }
        server_vocab = {
            srv.READ, srv.CACHEABLE, srv.SEARCH, srv.SEARCH_WITH_PATH,
            srv.CANDIDATE_SEARCH, srv.INSPECT_DIR, srv.CHECK_EXISTENCE,
            srv.CONTENT_WRITE, srv.EDIT,
            srv.REPLACEMENT_TRACK, srv.VALIDATE,
            srv.PLAN_BLOCKED, srv.PLAN_READONLY, srv.CODE_EXEC, srv.SENSITIVE, srv.NON_BATCH,
            srv.CODE_NAV, srv.ENV_DISCOVERY, srv.EXTERNAL_FETCH, srv.CLUSTER_SUBMIT,
            srv.ENV_MUTATE, srv.BACKGROUNDABLE,
            srv.REMOVE, srv.OVERWRITE, srv.TASK_PLANNING, srv.JUDGE,
        }
        self.assertEqual(client_vocab, server_vocab)

    def test_reversibility_vocab_in_sync(self):
        # Declared on one side, consumed on the other — the two spellings must match
        # exactly, or a level a server declares silently falls back to the derivation.
        from mimir.servers._shared import capabilities as srv
        self.assertEqual(caps.REVERSIBILITY_LEVELS, srv.REVERSIBILITY_LEVELS)

    def test_descriptor_round_trip(self):
        from mimir.servers._shared import capabilities as srv
        desc = srv.build_descriptor(
            caps=[srv.READ, srv.SEARCH],
            path_args=["path"],
            reversibility=srv.RECOVERABLE,
            non_batch=True,
            fallbacks=["grep"],
            label="Reading {path}",
            scope={"args": ["command"], "kind": "command_prefix"},
            risk_note="does a sensitive thing",
            preview={"kind": "content", "args": ["content"]},
        )
        tool = types.SimpleNamespace(
            name="declared_tool", meta={"mimir": desc},
            annotations=None, inputSchema=None,
        )
        c = caps.infer_tool_caps(tool)
        # SENSITIVE is derived from the declared reversibility, not declared alongside it.
        self.assertEqual(c.capabilities, frozenset({"read", "search", "sensitive", "non_batch"}))
        self.assertEqual(c.reversibility, srv.RECOVERABLE)
        self.assertEqual(c.arg_roles, {"path": ("path",)})
        self.assertEqual(c.fallbacks, ("grep",))
        self.assertEqual(caps.label_for("declared_tool", {"path": "a.py"}, {"declared_tool": c}), "Reading a.py")
        reg = {"declared_tool": c}
        self.assertEqual(caps.scope_spec("declared_tool", reg), {"args": ["command"], "kind": "command_prefix"})
        self.assertEqual(caps.risk_note_of("declared_tool", reg), "does a sensitive thing")
        self.assertEqual(caps.preview_spec("declared_tool", reg), {"args": ["content"], "kind": "content"})

    def test_label_for_missing_arg_falls_back(self):
        # A template referencing an absent optional arg degrades to None (the
        # generic humanizer takes over) rather than leaking raw {braces}.
        c = caps.ToolCaps(name="ranged", label="Lines {start}-{end}: {path}")
        reg = {"ranged": c}
        self.assertEqual(
            caps.label_for("ranged", {"start": 1, "end": 9, "path": "a.py"}, reg),
            "Lines 1-9: a.py",
        )
        self.assertIsNone(caps.label_for("ranged", {"path": "a.py"}, reg))
        self.assertIsNone(caps.label_for("ranged", {}, reg))

    def test_scope_and_risk_note_default_absent(self):
        # A descriptor without scope/risk_note resolves them to None (no fabrication).
        from mimir.servers._shared import capabilities as srv
        desc = srv.build_descriptor(caps=[srv.READ])
        c = caps.infer_tool_caps(types.SimpleNamespace(
            name="plain", meta={"mimir": desc}, annotations=None, inputSchema=None))
        self.assertIsNone(c.scope)
        self.assertIsNone(c.risk_note)
        self.assertIsNone(c.preview)
        self.assertIsNone(caps.scope_spec("plain", {"plain": c}))
        self.assertIsNone(caps.risk_note_of("plain", {"plain": c}))
        self.assertIsNone(caps.preview_spec("plain", {"plain": c}))

    def test_argless_scope_dropped(self):
        # A scope spec with a kind but no args is dropped (nothing to narrow on).
        from mimir.servers._shared import capabilities as srv
        desc = srv.build_descriptor(caps=[srv.READ], scope={"kind": "host"})
        self.assertNotIn("scope", desc)

    def test_preview_normalisation(self):
        # An argless preview spec is kept (a deletion has a shape but reads no
        # content args); a kindless one has nothing to dispatch on and is dropped.
        from mimir.servers._shared import capabilities as srv
        desc = srv.build_descriptor(caps=[srv.REMOVE], preview={"kind": "delete"})
        self.assertEqual(desc["preview"], {"kind": "delete"})
        desc = srv.build_descriptor(caps=[srv.EDIT], preview={"args": ["content"]})
        self.assertNotIn("preview", desc)


if __name__ == "__main__":
    unittest.main()


class ReversibilityDerivationTest(unittest.TestCase):
    """`SENSITIVE` is derived from reversibility, so the derivation *is* the gate.

    Getting it wrong is not a cosmetic mislabel: too permissive and a mutating tool
    runs unasked; too conservative and every read in the catalog raises a prompt (the
    first draft of `_derive_reversibility` defaulted everything to `recoverable` and
    did exactly that).
    """

    def _resolve(self, **descriptor_kwargs):
        from mimir.servers._shared import capabilities as srv
        desc = srv.build_descriptor(**descriptor_kwargs)
        tool = types.SimpleNamespace(
            name="t", meta={"mimir": desc}, annotations=None, inputSchema=None,
        )
        return caps.infer_tool_caps(tool)

    def test_a_read_only_tool_is_reversible_and_ungated(self):
        c = self._resolve(caps=[caps.READ, caps.SEARCH])
        self.assertEqual(c.reversibility, caps.REVERSIBLE)
        self.assertNotIn(caps.SENSITIVE, c.capabilities)

    def test_reaching_outside_the_workspace_is_not_by_itself_irreversible(self):
        # EXTERNAL_FETCH covers a read-only GitHub query as much as a POST. What makes
        # an outbound call irreversible is that it *sends*, which no capability states.
        c = self._resolve(caps=[caps.READ, caps.EXTERNAL_FETCH])
        self.assertEqual(c.reversibility, caps.REVERSIBLE)

    def test_an_in_workspace_write_is_reversible_because_mimir_snapshots_it(self):
        c = self._resolve(caps=[caps.EDIT, caps.CONTENT_WRITE])
        self.assertEqual(c.reversibility, caps.REVERSIBLE)

    def test_delete_and_env_mutation_are_recoverable_hence_gated(self):
        for cap in (caps.REMOVE, caps.ENV_MUTATE, caps.CODE_EXEC):
            with self.subTest(cap=cap):
                c = self._resolve(caps=[cap])
                self.assertEqual(c.reversibility, caps.RECOVERABLE)
                self.assertIn(caps.SENSITIVE, c.capabilities)

    def test_a_cluster_submit_is_irreversible(self):
        c = self._resolve(caps=[caps.CLUSTER_SUBMIT])
        self.assertEqual(c.reversibility, caps.IRREVERSIBLE)
        self.assertIn(caps.SENSITIVE, c.capabilities)
        self.assertEqual(caps.reversibility_of("t", {"t": c}), caps.IRREVERSIBLE)

    def test_a_declared_level_beats_the_derivation(self):
        from mimir.servers._shared import capabilities as srv
        c = self._resolve(caps=[caps.READ], reversibility=srv.IRREVERSIBLE)
        self.assertEqual(c.reversibility, caps.IRREVERSIBLE)
        self.assertIn(caps.SENSITIVE, c.capabilities)

    def test_a_legacy_sensitive_descriptor_still_prompts(self):
        # Third-party servers written against the pre-reversibility descriptor must
        # keep their approval prompt rather than silently become auto-approved.
        tool = types.SimpleNamespace(
            name="t", meta={"mimir": {"approval": {"sensitive": True}}},
            annotations=None, inputSchema=None,
        )
        c = caps.infer_tool_caps(tool)
        self.assertEqual(c.reversibility, caps.RECOVERABLE)
        self.assertIn(caps.SENSITIVE, c.capabilities)

    def test_an_unknown_tool_resolves_conservatively(self):
        self.assertEqual(caps.reversibility_of("never-connected", {}), caps.RECOVERABLE)

    def test_no_enforcement_level_can_soften_an_irreversible_action(self):
        """The invariant the dial must never be able to cross.

        `/enforcement` tunes how much the model is *reminded*; it has no say over what
        the model is *allowed* to do. Stated as an assertion rather than as prose in
        POLICY.md, because prose is what let the pre-edit planning gate end up on the
        hard-guard side of the same line.
        """
        import importlib
        gates = importlib.import_module("mimir.client.guardrails.policy.gates")

        class _Agent:
            model = "any-model"
            tool_caps = {"submit": caps.ToolCaps(
                name="submit", capabilities=frozenset({caps.CLUSTER_SUBMIT}),
                reversibility=caps.IRREVERSIBLE,
            )}

            @staticmethod
            def _json_error_payload(msg, **kw):
                return msg

        for level in ("strict", "light", "off"):
            with self.subTest(enforcement=level):
                agent = _Agent()
                agent.enforcement = level
                ctx: dict = {}
                self.assertIsNotNone(
                    gates._check_cluster_submit(agent, "submit", ctx),
                    "an irreversible action was softened by the enforcement dial",
                )
