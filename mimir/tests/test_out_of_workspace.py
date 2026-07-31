"""Out-of-workspace access approval (read/write/run), allow/deny/always.

Covers the client decision + sidecar propagation and the engine gate that prompts.
The server-honoring half (guards consulting the sidecar) is in test_read_roots.py
and the smoke checks here.

Run:
    python -m unittest mimir.tests.test_out_of_workspace -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

from mimir.client.context.capabilities import ToolCaps, CODE_EXEC, READ, EDIT
from mimir.client.guardrails.policy import approval as approval_mod
from mimir.client.guardrails.policy import engine
import mimir.client.config.constants as constants


class _TmpStateDir(unittest.TestCase):
    """Point the client sidecar (and the server reader) at a temp state dir."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._old_state = constants.STATE_DIR
        constants.STATE_DIR = self._tmp.name
        self._old_env = os.environ.get("MIMIR_STATE_DIR")
        os.environ["MIMIR_STATE_DIR"] = self._tmp.name

    def tearDown(self) -> None:
        constants.STATE_DIR = self._old_state
        if self._old_env is None:
            os.environ.pop("MIMIR_STATE_DIR", None)
        else:
            os.environ["MIMIR_STATE_DIR"] = self._old_env
        self._tmp.cleanup()

    def _sidecar(self) -> str:
        return os.path.join(self._tmp.name, "approved_paths.json")


class GrantPathTests(_TmpStateDir):
    def _mgr(self):
        return approval_mod.ApprovalManager(sensitive_tools=set())

    def test_allow_once_writes_sidecar_but_not_scope(self) -> None:
        m = self._mgr()
        m.grant_path("/tmp/data/run.log", always=False)
        self.assertFalse(m.is_path_approved("/tmp/data/run.log"))   # re-prompts next time
        self.assertTrue(os.path.isfile(self._sidecar()))
        import json
        self.assertIn(os.path.realpath("/tmp/data/run.log"),
                      json.load(open(self._sidecar())))

    def test_always_records_scope_and_sidecar(self) -> None:
        m = self._mgr()
        m.grant_path("/tmp/data/run.log", always=True)
        self.assertTrue(m.is_path_approved("/tmp/data/run.log"))    # no re-prompt
        import json
        self.assertIn(os.path.realpath("/tmp/data/run.log"),
                      json.load(open(self._sidecar())))

    def test_reset_clears_paths_and_sidecar(self) -> None:
        m = self._mgr()
        m.grant_path("/tmp/data/run.log", always=True)
        m.reset_allowed_paths()
        self.assertFalse(m.is_path_approved("/tmp/data/run.log"))
        import json
        self.assertEqual(json.load(open(self._sidecar())), [])

    def test_server_reads_client_sidecar(self) -> None:
        # End-to-end propagation: what the client writes, the server helper reads.
        for p in (Path(__file__).resolve().parents[1] / "servers" / "_shared",):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        import importlib
        import approved_roots as ar
        importlib.reload(ar)  # ensure it resolves state_dir() fresh
        m = self._mgr()
        m.grant_path("/tmp/data/run.log", always=True)
        self.assertIn(os.path.realpath("/tmp/data/run.log"), ar.approved_roots())


class _FakeAgent:
    def __init__(self, mgr, *, cap, script) -> None:
        self.approvals = mgr
        self.tool_caps = {"read_file_lines": ToolCaps(name="read_file_lines",
                                                       capabilities=frozenset({cap}))}
        self._script = script          # (approved, always) returned by the prompt
        self.prompts: list[str] = []
        self.prompt_args: list[dict | None] = []

    def get_tool_file_targets(self, tool_name, arguments):
        p = arguments.get("path")
        return [p] if isinstance(p, str) and p else []

    def _is_write_tool(self, tool_name):
        return False

    def _json_error_payload(self, msg, hint="", tool=""):
        import json
        return json.dumps({"status": "error", "error": msg, "hint": hint})

    def _request_path_approval(self, abspath, tool_name, arguments=None):
        # The call's own arguments travel with the path so the prompt can describe
        # what the tool is doing, not just which path it touches.
        self.prompts.append(abspath)
        self.prompt_args.append(arguments)
        return self._script


class OutOfWorkspaceGateTests(_TmpStateDir):
    def _agent(self, cap=READ, script=(True, False)):
        return _FakeAgent(approval_mod.ApprovalManager(sensitive_tools=set()),
                          cap=cap, script=script)

    def test_in_workspace_path_never_prompts(self) -> None:
        agent = self._agent()
        # A path under the workspace root (cwd) — no prompt, no violation.
        rel = "some/in_workspace_file.py"
        out = engine._check_out_of_workspace_access(
            agent, "read_file_lines", {"path": rel}, {})
        self.assertIsNone(out)
        self.assertEqual(agent.prompts, [])

    def test_out_of_workspace_deny_returns_violation(self) -> None:
        agent = self._agent(script=(False, False))
        out = engine._check_out_of_workspace_access(
            agent, "read_file_lines", {"path": "/etc/passwd"}, {})
        self.assertIsNotNone(out)
        self.assertIn("outside the workspace", out)
        self.assertEqual(agent.prompts, [os.path.realpath("/etc/passwd")])

    def test_prompt_receives_the_calls_own_arguments(self) -> None:
        """The prompt must be able to say what the tool is doing, not just where.

        A card built from the path alone ("Out-of-workspace access: /a/b") tells the
        user nothing about the operation being approved, so the front-ends need the
        real arguments to render the usual tool description alongside it.
        """
        agent = self._agent(script=(True, False))
        args = {"path": "/etc/passwd", "start_line": 1}
        engine._check_out_of_workspace_access(agent, "read_file_lines", args, {})
        self.assertEqual(agent.prompt_args, [args])

    def test_out_of_workspace_allow_once_grants_then_reprompts(self) -> None:
        agent = self._agent(script=(True, False))
        out1 = engine._check_out_of_workspace_access(
            agent, "read_file_lines", {"path": "/tmp/x/f.log"}, {})
        self.assertIsNone(out1)
        # allow-once did NOT record the scope → a second call prompts again.
        out2 = engine._check_out_of_workspace_access(
            agent, "read_file_lines", {"path": "/tmp/x/f.log"}, {})
        self.assertIsNone(out2)
        self.assertEqual(len(agent.prompts), 2)

    def test_out_of_workspace_always_skips_second_prompt(self) -> None:
        agent = self._agent(script=(True, True))
        engine._check_out_of_workspace_access(
            agent, "read_file_lines", {"path": "/tmp/x/f.log"}, {})
        engine._check_out_of_workspace_access(
            agent, "read_file_lines", {"path": "/tmp/x/f.log"}, {})
        self.assertEqual(len(agent.prompts), 1)   # "always" → no re-prompt

    def test_trusted_read_root_not_prompted_for_reads(self) -> None:
        agent = self._agent(cap=READ)
        log = os.path.expanduser("~/.cache/proxy_bench/opt_runs/x/stdout.log")
        out = engine._check_out_of_workspace_access(
            agent, "read_file_lines", {"path": log}, {})
        self.assertIsNone(out)
        self.assertEqual(agent.prompts, [])   # item 1: silent read of proxy cache

    def test_state_dir_is_trusted_without_the_env_var(self) -> None:
        """The client must trust STATE_DIR from its own config, not from the env.

        ``MIMIR_STATE_DIR`` is only ever placed in the *server* subprocesses' env
        (server_manager), so the shared helper — which resolves the state dir from
        that variable — returns nothing for it in the client process. The gate then
        prompted for every read of the agent's own plans and sessions, while the
        servers, which do see the variable, would have allowed it. This class's setUp
        happens to export the variable, so it is removed here: with it set, the bug
        is invisible.
        """
        old = os.environ.pop("MIMIR_STATE_DIR", None)
        try:
            agent = self._agent(cap=READ)
            plan = os.path.join(constants.STATE_DIR, "sessions", "s1", "plans", "p.md")
            out = engine._check_out_of_workspace_access(
                agent, "read_file_lines", {"path": plan}, {})
            self.assertIsNone(out)
            self.assertEqual(agent.prompts, [])
        finally:
            if old is not None:
                os.environ["MIMIR_STATE_DIR"] = old

    def test_scratchpad_writes_are_never_prompted(self) -> None:
        """The scratchpad is outside the workspace by design.

        Prompting for it would defeat its purpose — it exists so throwaway work has
        a home that costs no user decision. Resolved from the client's own
        STATE_DIR, for the same reason as the trusted-read roots above, so this is
        checked with MIMIR_STATE_DIR removed.
        """
        from mimir.client.tool_execution.validation import scratch_roots
        old = os.environ.pop("MIMIR_STATE_DIR", None)
        try:
            agent = self._agent(cap=EDIT, script=(False, False))  # would DENY if asked
            probe = os.path.join(scratch_roots()[0], "probe.py")
            out = engine._check_out_of_workspace_access(
                agent, "write_file", {"path": probe}, {})
            self.assertIsNone(out)
            self.assertEqual(agent.prompts, [])
        finally:
            if old is not None:
                os.environ["MIMIR_STATE_DIR"] = old

    def test_scratchpad_grant_does_not_widen_to_its_parent(self) -> None:
        # The exemption must be the scratchpad specifically, not "anything under
        # the state dir is writable".
        agent = self._agent(cap=EDIT, script=(False, False))
        sibling = os.path.join(constants.STATE_DIR, "sessions", "s1", "plans", "p.md")
        out = engine._check_out_of_workspace_access(
            agent, "write_file", {"path": sibling}, {})
        self.assertIsNotNone(out)


class _FakeExecAgent(_FakeAgent):
    """A CODE_EXEC tool taking a shell command — paths arrive inside the string.

    No `cwd` arg-role: the bash server has no working-directory argument, so every
    call starts at the workspace root and `cd` is the only way to move.
    """

    def __init__(self, mgr, *, script) -> None:
        super().__init__(mgr, cap=CODE_EXEC, script=script)
        self.tool_caps = {
            "run_shell": ToolCaps(name="run_shell",
                                  capabilities=frozenset({CODE_EXEC}))
        }

    def get_tool_file_targets(self, tool_name, arguments):
        return []


class ShellPathApprovalTests(_TmpStateDir):
    """Every out-of-workspace path a shell command names goes to the user.

    A shell command carries its paths inside a string, so the file-target extractor
    cannot see them. Without this the server refused them and no prompt was ever
    shown — leaving the user unable to grant an access they might well have wanted.
    """

    def _agent(self, script=(True, False)):
        return _FakeExecAgent(approval_mod.ApprovalManager(sensitive_tools=set()),
                              script=script)

    def test_file_operand_outside_workspace_prompts(self) -> None:
        agent = self._agent(script=(True, False))
        out = engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": "cat /etc/passwd"}, {})
        self.assertIsNone(out)
        self.assertEqual(agent.prompts, [os.path.realpath("/etc/passwd")])

    def test_exec_operand_outside_workspace_prompts(self) -> None:
        # The half that used to slip through entirely: an interpreter's file operand.
        for cmd, target in (("python /tmp/evil.py", "/tmp/evil.py"),
                            ("gcc /tmp/x.c -o out.o", "/tmp/x.c"),
                            ("cp /etc/hosts here.txt", "/etc/hosts")):
            agent = self._agent(script=(True, False))
            engine._check_out_of_workspace_access(agent, "run_shell", {"command": cmd}, {})
            self.assertIn(os.path.realpath(target), agent.prompts, cmd)

    def test_denied_path_blocks_the_call(self) -> None:
        agent = self._agent(script=(False, False))
        out = engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": "cat /etc/passwd"}, {})
        self.assertIsNotNone(out)
        self.assertIn("outside the workspace", out)

    def test_flags_are_not_mistaken_for_paths(self) -> None:
        # `-I/usr/include`, `-lm`, `-m pytest` must not raise a prompt, or every
        # ordinary compile would stop to ask about its include dirs.
        agent = self._agent(script=(False, False))
        for cmd in ("gcc -I/usr/include a.c -o a.out -lm", "python -m pytest",
                    "ls *.py | head", "make -j4"):
            out = engine._check_out_of_workspace_access(
                agent, "run_shell", {"command": cmd}, {})
            self.assertIsNone(out, cmd)
        self.assertEqual(agent.prompts, [])

    def test_in_workspace_paths_never_prompt(self) -> None:
        agent = self._agent()
        for cmd in ("cat notes.txt", "python mimir/x.py", "cp a.py b.py"):
            out = engine._check_out_of_workspace_access(
                agent, "run_shell", {"command": cmd}, {})
            self.assertIsNone(out, cmd)
        self.assertEqual(agent.prompts, [])

    def test_cd_inside_workspace_never_prompts(self) -> None:
        agent = self._agent()
        for cmd in ("cd sub && ls", "cd sub/deeper && cat f.py", "ls"):
            out = engine._check_out_of_workspace_access(
                agent, "run_shell", {"command": cmd}, {})
            self.assertIsNone(out, cmd)
        self.assertEqual(agent.prompts, [])

    def test_search_pattern_raises_no_prompt(self) -> None:
        # `grep /etc/passwd notes.txt` opens nothing outside the workspace, so
        # stopping to ask the user about /etc/passwd would be noise.
        agent = self._agent(script=(False, False))
        for cmd in ("grep /etc/passwd notes.txt", 'sed "s|/etc/passwd|x|" f.txt'):
            out = engine._check_out_of_workspace_access(
                agent, "run_shell", {"command": cmd}, {})
            self.assertIsNone(out, cmd)
        self.assertEqual(agent.prompts, [])

    def test_path_beside_a_flag_expansion_still_reaches_the_user(self) -> None:
        # A bare $VAR makes a command opaque to *classification*, which used to mean
        # no targets were extracted at all — so a real out-of-workspace path sitting
        # beside the flag was refused by the server with no way to grant it.
        agent = self._agent(script=(True, False))
        engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": "gcc -I$CUDA_HOME/include /tmp/x.c -o a.out"}, {})
        self.assertIn(os.path.realpath("/tmp/x.c"), agent.prompts)

    def test_expanded_path_operand_raises_no_prompt(self) -> None:
        # In path position an expansion is refused outright by the guard (it cannot
        # be checked), so there is nothing to ask about.
        agent = self._agent(script=(False, False))
        out = engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": "cat $CUDA_HOME/version.txt"}, {})
        self.assertIsNone(out)
        self.assertEqual(agent.prompts, [])

    def test_command_substitution_stays_opaque(self) -> None:
        # `$(...)` runs code: no path can be attributed to it, and the server
        # rejects the command outright.
        agent = self._agent(script=(False, False))
        for cmd in ("cat $(which python)", "cat `echo /etc/passwd`"):
            out = engine._check_out_of_workspace_access(
                agent, "run_shell", {"command": cmd}, {})
            self.assertIsNone(out, cmd)
        self.assertEqual(agent.prompts, [])

    def test_path_after_an_out_of_workspace_cd_is_seen(self) -> None:
        # `cd /etc && cat passwd`: the relative operand resolves under the new base,
        # so both the destination and the file it reaches are surfaced.
        agent = self._agent(script=(True, True))
        engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": "cd /etc && cat passwd"}, {})
        self.assertIn(os.path.realpath("/etc"), agent.prompts)
        self.assertIn(os.path.realpath("/etc/passwd"), agent.prompts)

    def test_cd_outside_workspace_prompts(self) -> None:
        agent = self._agent(script=(True, False))
        out = engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": "cd /tmp/elsewhere && ls"}, {})
        self.assertIsNone(out)
        self.assertEqual(agent.prompts, [os.path.realpath("/tmp/elsewhere")])

    def test_cd_outside_workspace_denied_blocks_the_call(self) -> None:
        agent = self._agent(script=(False, False))
        out = engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": "cd /etc && cat passwd"}, {})
        self.assertIsNotNone(out)
        self.assertIn("outside the workspace", out)

    def test_escape_via_relative_cd_is_caught(self) -> None:
        # `cd ..` leaves the workspace just as surely as an absolute path.
        agent = self._agent(script=(False, False))
        out = engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": "cd ../.. && ls"}, {})
        self.assertIsNotNone(out)

    def test_chained_cds_accumulate(self) -> None:
        # Each hop resolves against the previous one, mirroring the server walk:
        # the first stays inside, the second escapes and must prompt.
        agent = self._agent(script=(False, False))
        out = engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": "cd sub && cd ../../.. && ls"}, {})
        self.assertIsNotNone(out)

    def test_approval_is_recorded_for_the_server(self) -> None:
        # The grant must reach the sidecar the server reads per call, or the
        # approved cd would still be refused server-side.
        agent = self._agent(script=(True, True))
        engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": "cd /tmp/granted && ls"}, {})
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "servers" / "_shared"))
        import approved_roots
        self.assertIn(os.path.realpath("/tmp/granted"), approved_roots.approved_roots())

    def test_non_exec_tool_is_not_scanned_for_cd(self) -> None:
        # The gate is capability-scoped: a command-looking string in a READ tool's
        # argument is not a chdir.
        agent = _FakeAgent(approval_mod.ApprovalManager(sensitive_tools=set()),
                           cap=READ, script=(False, False))
        out = engine._check_out_of_workspace_access(
            agent, "read_file_lines", {"path": "notes.txt", "q": "cd /etc"}, {})
        self.assertIsNone(out)
        self.assertEqual(agent.prompts, [])


if __name__ == "__main__":
    unittest.main()
