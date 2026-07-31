import types
import unittest

from mimir.client.guardrails.policy.approval import ApprovalManager as _ApprovalManager
from mimir.client.context.capabilities import (
    NON_BATCH, SENSITIVE, ToolCaps, fallbacks, names_with_cap,
)
from mimir.tests import _golden_caps as golden

# Provide a module-compatible namespace so existing test references work.
approval_module = types.SimpleNamespace(ApprovalManager=_ApprovalManager)


# The live {name: ToolCaps} registry, parsed from server source exactly as the agent
# builds it at connect — scope specs, confirm_gate roles and risk_notes included.
_DECLARED = golden.build_declared_registry()


def _manager(sensitive=None):
    """A manager wired to the real declared registry, like a connected agent."""
    m = _ApprovalManager(set(sensitive or ()), tool_caps=_DECLARED)
    m.sensitive_tools.update(names_with_cap(SENSITIVE, _DECLARED))
    return m


class ApprovalManagerTests(unittest.TestCase):
    def test_tracks_always_approval(self) -> None:
        manager = approval_module.ApprovalManager({"write_file"})
        manager.batch_mode = False
        approved, note = manager.request(
            tool_name="write_file",
            server_name="files",
            arguments={"path": "x.txt"},
            input_func=lambda _: "a",
            output_func=lambda _: None,
        )
        self.assertTrue(approved)
        self.assertEqual(note, "approved for this session")
        self.assertIn("write_file", manager.trusted_tool_names())

    def test_render_prompt_includes_server_and_risk(self) -> None:
        manager = _manager({"write_file"})
        prompt = manager.render_prompt(
            tool_name="write_file",
            server_name="files",
            arguments={"path": "demo.txt"},
        )
        self.assertIn("Server : files", prompt)
        self.assertIn("creates or overwrites files", prompt)

    def test_fallback_suggestions_returns_configured_alternatives(self) -> None:
        # fallback_tools is seeded from the live registry post-connect; emulate that.
        manager = approval_module.ApprovalManager(
            {"bash_run"},
            fallback_tools={"bash_run": ("code_run", "read_file_lines", "grep")},
        )
        suggestions = manager.fallback_suggestions("bash_run")
        self.assertEqual(suggestions, ("code_run", "read_file_lines", "grep"))

    def test_fallback_suggestions_empty_for_unconfigured_tool(self) -> None:
        manager = approval_module.ApprovalManager({"write_file"})
        self.assertEqual(manager.fallback_suggestions("memory_delete"), tuple())

    def test_request_denies_after_invalid_responses(self) -> None:
        manager = approval_module.ApprovalManager({"write_file"})
        manager.batch_mode = False
        answers = iter(["?", "maybe", "what"])
        outputs: list[str] = []
        approved, note = manager.request(
            tool_name="write_file",
            server_name="files",
            arguments={"path": "demo.txt"},
            input_func=lambda _: next(answers),
            output_func=outputs.append,
            max_attempts=3,
        )
        self.assertFalse(approved)
        self.assertEqual(note, "denied after invalid approval responses")
        self.assertEqual(len(outputs), 3)


class SensitivityGateTests(unittest.TestCase):
    """is_sensitive is fully registry-driven (no literal tool names)."""

    def test_confirm_gate_tracks_confirm_arg(self) -> None:
        manager = _manager()
        # apply_edits / replace_all_in_file are preview when confirm is falsy, real
        # mutation (sensitive) when truthy.
        self.assertFalse(manager.is_sensitive("apply_edits", {"confirm": False}))
        self.assertTrue(manager.is_sensitive("apply_edits", {"confirm": True}))
        self.assertFalse(manager.is_sensitive("replace_all_in_file", {"confirm": False}))
        self.assertTrue(manager.is_sensitive("replace_all_in_file", {"confirm": True}))

    def test_confirm_gate_never_downgrades_declared_sensitive(self) -> None:
        # A tool that is BOTH declared SENSITIVE and confirm-gated must stay sensitive
        # even with confirm falsy — the gate may only add sensitivity, never remove it.
        reg = {
            "danger": ToolCaps(
                name="danger",
                capabilities=frozenset({SENSITIVE}),
                arg_roles={"confirm_gate": ("confirm",)},
            )
        }
        manager = _ApprovalManager({"danger"}, tool_caps=reg)
        self.assertTrue(manager.is_sensitive("danger", {"confirm": False}))

    def test_http_get_auth_url_is_sensitive(self) -> None:
        # The fail-OPEN surface: http_get is sensitive only via its declared `host`
        # scope. This exercises the REAL declaration, so a missing/renamed scope spec
        # would flip these to non-sensitive and fail here.
        manager = _manager()
        for url in (
            "https://example.com/api/token",
            "https://host/oauth/authorize",
            "https://h/v1/secret/x",
        ):
            self.assertTrue(manager.is_sensitive("http_get", {"url": url}), url)

    def test_http_get_plain_url_not_sensitive(self) -> None:
        manager = _manager()
        self.assertFalse(manager.is_sensitive("http_get", {"url": "https://example.com/index.html"}))

    def test_http_post_always_sensitive(self) -> None:
        manager = _manager()
        self.assertTrue(manager.is_sensitive("http_post", {"url": "https://example.com/x"}))


class ScopeNarrowingTests(unittest.TestCase):
    """_approval_scope / _approval_scope_label resolve via the declared scope spec.

    These pin behavior equivalence to the pre-refactor per-tool branches.
    """

    def setUp(self) -> None:
        self.m = _manager()

    def _scope(self, tool, args):
        return self.m._approval_scope(tool, "srv", args)

    def _label(self, tool, args):
        return self.m._approval_scope_label(tool, "srv", args)

    def test_bash_command_family(self) -> None:
        args = {"command": "cd /work && python3 -c \"import x\""}
        self.assertEqual(self._scope("bash_run", args), "srv:bash_run:python3 -c")
        self.assertEqual(self._label("bash_run", args), "this command family (python3 -c)")

    def test_bash_unscopable_falls_back_to_base(self) -> None:
        self.assertEqual(self._scope("bash_run", {"command": ""}), "srv:bash_run")

    def test_http_post_host(self) -> None:
        args = {"url": "https://api.example.com/v1/x"}
        self.assertEqual(self._scope("http_post", args), "srv:http_post:api.example.com")
        self.assertEqual(self._label("http_post", args), "this destination host (api.example.com)")

    def test_env_pip_install_packages(self) -> None:
        args = {"packages": ["numpy", "abc"]}
        self.assertEqual(self._scope("env_pip_install", args), "srv:env_pip_install:abc numpy")
        self.assertEqual(self._label("env_pip_install", args), "these packages (abc numpy)")

    def test_env_create_basename_with_env_noun(self) -> None:
        args = {"name": "/envs/myproj"}
        self.assertEqual(self._scope("env_create", args), "srv:env_create:myproj")
        self.assertEqual(self._label("env_create", args), "this environment (myproj)")

    def test_unscoped_tool_uses_base_and_generic_label(self) -> None:
        self.assertEqual(self._scope("write_file", {"path": "a.txt"}), "srv:write_file")
        self.assertEqual(self._label("write_file", {"path": "a.txt"}), "this tool")


class ReseedClassificationTests(unittest.TestCase):
    """Guards MimirAgent.seed_classification_from_caps's in-place mutation pattern.

    The manager is constructed before servers connect, then re-seeded from the
    live per-agent registry. The mutation must (a) preserve the identity of
    `approved_scopes` (aliased as `agent.session_approved_scopes`) and the three
    classification containers, and (b) keep non-batch gating correct.
    """

    @staticmethod
    def _reseed(manager, tool_caps):
        # Mirrors MimirAgent.seed_classification_from_caps verbatim.
        manager.sensitive_tools.clear()
        manager.sensitive_tools.update(names_with_cap(SENSITIVE, tool_caps))
        manager.non_batch_tools.clear()
        manager.non_batch_tools.update(names_with_cap(NON_BATCH, tool_caps))
        manager.fallback_tools.clear()
        manager.fallback_tools.update(
            {n: fallbacks(n, tool_caps) for n in tool_caps if fallbacks(n, tool_caps)}
        )
        manager.tool_caps = tool_caps

    def _registry(self):
        # A small live-style registry: bash_run is sensitive+non_batch with a fallback.
        from mimir.client.context.capabilities import PLAN_BLOCKED
        return {
            "bash_run": ToolCaps(
                name="bash_run",
                capabilities=frozenset({SENSITIVE, NON_BATCH, PLAN_BLOCKED}),
                fallbacks=("code_run", "read_file_lines"),
            ),
            "read_file_lines": ToolCaps(name="read_file_lines", capabilities=frozenset({"read", "cacheable"})),
        }

    def test_reseed_preserves_aliases_and_non_batch(self):
        manager = approval_module.ApprovalManager(set())
        # Capture aliases the way agent_core.py does.
        scopes_alias = manager.approved_scopes
        sensitive_alias = manager.sensitive_tools
        non_batch_alias = manager.non_batch_tools

        self._reseed(manager, self._registry())

        # Object identity preserved (mutated in place, not reassigned).
        self.assertIs(manager.approved_scopes, scopes_alias)
        self.assertIs(manager.sensitive_tools, sensitive_alias)
        self.assertIs(manager.non_batch_tools, non_batch_alias)

        # Classification now reflects the live registry.
        self.assertEqual(manager.sensitive_tools, {"bash_run"})
        self.assertEqual(manager.non_batch_tools, {"bash_run"})
        self.assertEqual(manager.fallback_tools, {"bash_run": ("code_run", "read_file_lines")})

        # Non-batch gating works: a non-batch tool is not auto-batchable.
        self.assertTrue(manager.is_sensitive("bash_run", {}))
        self.assertIn("bash_run", manager.non_batch_tools)
        self.assertNotIn("read_file_lines", manager.non_batch_tools)


if __name__ == "__main__":
    unittest.main()
