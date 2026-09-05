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

    def test_always_on_a_dir_covers_paths_under_it(self) -> None:
        # The sidecar hands the server a *root*, so a child of a granted directory
        # is already allowed there — re-prompting for it could deny nothing.
        m = self._mgr()
        m.grant_path("/tmp/data", always=True)
        self.assertTrue(m.is_path_approved("/tmp/data/sub/run.log"))
        self.assertFalse(m.is_path_approved("/tmp/data-other/run.log"))

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
        self.prompts: list[str] = []          # every path asked about, flattened
        self.prompt_calls: int = 0            # how many cards the user actually saw
        self.prompt_args: list[dict | None] = []

    def get_tool_file_targets(self, tool_name, arguments):
        p = arguments.get("path")
        return [p] if isinstance(p, str) and p else []

    def _is_write_tool(self, tool_name):
        return False

    def _json_error_payload(self, msg, hint="", tool=""):
        import json
        return json.dumps({"status": "error", "error": msg, "hint": hint})

    def _request_path_approval(self, paths, tool_name, arguments=None):
        # Every outside path of the call arrives in one prompt; the call's own
        # arguments travel with them so the card can describe what the tool is
        # doing, not just which paths it touches.
        self.prompt_calls += 1
        self.prompts.extend(paths)
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
        a home that costs no user decision. Checked with MIMIR_STATE_DIR removed:
        the scratchpad home lives under the temp dir and must not depend on the
        state dir (only the per-session subdirectory reads its sidecar).
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
        # The exemption must be the scratchpad specifically, not "the temp dir is
        # writable now" — nor a prefix match that admits a same-named sibling.
        from mimir.client.tool_execution.validation import scratch_roots
        home = scratch_roots()[0]
        agent = self._agent(cap=EDIT, script=(False, False))
        for path in (
            os.path.join(os.path.dirname(home), "someone_else.py"),
            home + "_evil/probe.py",
            os.path.join(constants.STATE_DIR, "sessions", "s1", "plans", "p.md"),
        ):
            with self.subTest(path=path):
                out = engine._check_out_of_workspace_access(
                    agent, "write_file", {"path": path}, {})
                self.assertIsNotNone(out)


class _FakeExecAgent(_FakeAgent):
    """A shell-command tool — paths arrive inside the command string.

    No `cwd` arg-role: the bash server has no working-directory argument, so every
    call starts at the workspace root and `cd` is the only way to move.

    The `command_prefix` scope is what makes the extractor read the arguments as shell;
    CODE_EXEC alone does not, since a tool can execute through structured parameters
    that no shell parser should be pointed at.
    """

    def __init__(self, mgr, *, script) -> None:
        super().__init__(mgr, cap=CODE_EXEC, script=script)
        self.tool_caps = {
            "run_shell": ToolCaps(name="run_shell",
                                  capabilities=frozenset({CODE_EXEC}),
                                  scope={"args": ["command"], "kind": "command_prefix"})
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

    def test_path_after_an_out_of_workspace_cd_asks_once(self) -> None:
        # `cd /etc && cat passwd`: the relative operand resolves under the new base,
        # so the file *is* seen — but granting the destination admits it to the
        # server as a root, so asking again for a path under it decides nothing.
        agent = self._agent(script=(True, True))
        out = engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": "cd /etc && cat passwd"}, {})
        self.assertIsNone(out)
        self.assertEqual(agent.prompts, [os.path.realpath("/etc")])

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


class _FakeEngineAgent(_FakeExecAgent):
    """Enough of the agent surface for the full precondition pipeline."""

    def __init__(self, mgr, *, script) -> None:
        super().__init__(mgr, script=script)
        self.tool_owner = {"run_shell": "code"}
        self.tool_approvals: list[str] = []

    def _normalize_tool_arguments(self, tool_name, arguments):
        return dict(arguments)

    def _rewrite_tool_for_context(self, tool_name, arguments):
        return tool_name, dict(arguments)

    def _request_tool_approval(self, tool_name, arguments):
        self.tool_approvals.append(tool_name)
        return True, "approved once"

    def approval_scope(self, tool_name, arguments):
        return f"fake:{tool_name}"

    def _record_denied_tool_call(self, tool_name, arguments, execution_context, note="") -> None:
        pass

    def _denied_tool_result(self, tool_name, arguments, note="", execution_context=None):
        return f"denied:{tool_name}:{note}"

    def _normalize_workspace_path(self, path):
        return path or ""

    def _is_code_filepath(self, path):
        return str(path).endswith(".py")


class SingleApprovalPerCallTests(_TmpStateDir):
    """One call, one question.

    A sensitive command reaching outside the workspace used to raise two cards in a
    row — the out-of-workspace prompt and then the ordinary sensitive-tool prompt —
    for a single decision. The first already names the tool, its arguments and the
    path, and rates the call irreversible, so it subsumes the second.
    """

    def _agent(self, script=(True, False)):
        return _FakeEngineAgent(
            approval_mod.ApprovalManager(sensitive_tools={"run_shell"}), script=script)

    def _evaluate(self, agent, command):
        return engine.evaluate_tool_preconditions(
            agent=agent, tool_name="run_shell", arguments={"command": command},
            execution_context={"searched": True},
        )

    def test_outside_workspace_asks_once(self) -> None:
        agent = self._agent()
        out = self._evaluate(agent, "mkdir /tmp/outside/build")
        self.assertIsNone(out.violation)
        self.assertEqual(agent.prompts, [os.path.realpath("/tmp/outside/build")])
        self.assertEqual(agent.tool_approvals, [])   # no second card

    def test_inside_workspace_still_asks_the_sensitive_card(self) -> None:
        # Nothing outside → the ordinary approval is the only one, and must remain.
        agent = self._agent()
        out = self._evaluate(agent, "mkdir build")
        self.assertIsNone(out.violation)
        self.assertEqual(agent.prompts, [])
        self.assertEqual(agent.tool_approvals, ["run_shell"])

    def test_denied_outside_path_still_blocks(self) -> None:
        agent = self._agent(script=(False, False))
        out = self._evaluate(agent, "mkdir /tmp/outside/build")
        self.assertIsNotNone(out.violation)
        self.assertIn("outside the workspace", out.violation)
        self.assertEqual(agent.tool_approvals, [])


class OneCardPerCallTests(_TmpStateDir):
    """A command naming several outside paths raises exactly ONE approval.

    It used to raise one per path: ``cd /data && python /opt/x.py > /var/log/y.log``
    put three cards in front of the user, in sequence, for a decision they had already
    taken when they read the command.
    """

    COMMAND = "cd /tmp/one && python /opt/one/x.py > /var/tmp/one/y.log"

    def _agent(self, script=(True, False)):
        return _FakeExecAgent(approval_mod.ApprovalManager(sensitive_tools=set()),
                              script=script)

    def _expected(self) -> set[str]:
        return {os.path.realpath(p) for p in
                ("/tmp/one", "/opt/one/x.py", "/var/tmp/one/y.log")}

    def test_every_path_arrives_in_a_single_prompt(self) -> None:
        agent = self._agent()
        out = engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": self.COMMAND}, {})
        self.assertIsNone(out)
        self.assertEqual(agent.prompt_calls, 1)
        self.assertEqual(set(agent.prompts), self._expected())

    def test_one_allow_grants_all_of_them(self) -> None:
        agent = self._agent()
        engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": self.COMMAND}, {})
        # allow-once records the sidecar grant the servers read (not an "always" scope)
        for path in self._expected():
            self.assertIn(path, agent.approvals._allowed_paths)

    def test_one_deny_blocks_and_grants_nothing(self) -> None:
        agent = self._agent(script=(False, False))
        out = engine._check_out_of_workspace_access(
            agent, "run_shell", {"command": self.COMMAND}, {})
        self.assertIsNotNone(out)
        self.assertEqual(agent.prompt_calls, 1)
        # The refusal names the paths it refused, not just the first one it saw.
        for path in self._expected():
            self.assertIn(path, out)
            self.assertNotIn(path, agent.approvals._allowed_paths)


class ApprovalModeTests(_TmpStateDir):
    """manual / auto / auto_all — what each cran stops asking about."""

    def _agent(self, mode):
        agent = _FakeEngineAgent(
            approval_mod.ApprovalManager(sensitive_tools={"run_shell"}),
            script=(True, False))
        agent.approvals.approval_mode = mode
        return agent

    def _evaluate(self, agent, command):
        return engine.evaluate_tool_preconditions(
            agent=agent, tool_name="run_shell", arguments={"command": command},
            execution_context={"searched": True},
        )

    def test_manual_asks_for_both(self) -> None:
        inside = self._agent("manual")
        self._evaluate(inside, "mkdir build")
        self.assertEqual(inside.tool_approvals, ["run_shell"])
        outside = self._agent("manual")
        self._evaluate(outside, "mkdir /tmp/outside/manual")
        self.assertEqual(outside.prompt_calls, 1)

    def test_auto_passes_tools_but_still_asks_to_leave_the_workspace(self) -> None:
        inside = self._agent("auto")
        out = self._evaluate(inside, "mkdir build")
        self.assertIsNone(out.violation)
        self.assertEqual(inside.tool_approvals, [])     # no card built
        outside = self._agent("auto")
        self._evaluate(outside, "mkdir /tmp/outside/auto")
        self.assertEqual(outside.prompt_calls, 1)       # the boundary still asks

    def test_auto_all_asks_nothing_and_still_grants_the_paths(self) -> None:
        agent = self._agent("auto_all")
        out = self._evaluate(agent, "mkdir /tmp/outside/autoall")
        self.assertIsNone(out.violation)
        self.assertEqual(agent.prompt_calls, 0)
        self.assertEqual(agent.tool_approvals, [])
        self.assertIn(os.path.realpath("/tmp/outside/autoall"),
                      agent.approvals._allowed_paths)

    def test_the_mode_is_read_afresh_at_every_call(self) -> None:
        # The switch has to work in the middle of a run: the engine reads the manager
        # at each call, so flipping it between two calls changes the second one.
        agent = self._agent("manual")
        self._evaluate(agent, "mkdir build")
        self.assertEqual(agent.tool_approvals, ["run_shell"])
        agent.approvals.approval_mode = "auto"
        self._evaluate(agent, "mkdir build2")
        self.assertEqual(agent.tool_approvals, ["run_shell"])   # no second card
        agent.approvals.approval_mode = "manual"
        self._evaluate(agent, "mkdir build3")
        self.assertEqual(agent.tool_approvals, ["run_shell", "run_shell"])

    def test_auto_does_not_revive_a_settled_refusal(self) -> None:
        # A scope already refused twice stays refused: auto mode decides what is
        # *asked*, it does not overturn an answer the user has already given.
        agent = self._agent("auto")
        ctx = {"searched": True,
               "denial_history": [{"scope": "fake:run_shell", "kind": "denied"}] * 2}
        out = engine.evaluate_tool_preconditions(
            agent=agent, tool_name="run_shell", arguments={"command": "mkdir build"},
            execution_context=ctx,
        )
        self.assertIsNotNone(out.violation)
        self.assertEqual(agent.tool_approvals, [])


if __name__ == "__main__":
    unittest.main()
