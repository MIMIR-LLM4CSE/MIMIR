"""Read-only classifier + approval exemption for the dual-use bash tool.

``bash_command_is_readonly`` classifies a command as read-only discovery vs
build/execution/write. Plan mode uses it to stay side-effect-free; the approval
exemption uses it to run read-only discovery unattended in **any** mode. Writes
like ``sed -i`` are not read-only and keep the approval prompt. See
``bash_classify.bash_command_is_readonly`` / ``readonly_exempt._readonly_bash_exempt``.
"""
import unittest

from mimir.client.guardrails.policy.bash_classify import bash_command_is_readonly
from mimir.client.guardrails.policy.readonly_exempt import _readonly_bash_exempt
from mimir.client.query_engine.toollist import tools_for_plan_mode, tools_for_readonly_mode
from mimir.client.query_engine.readonly_guard import filter_readonly_tool_calls
from mimir.client.context.capabilities import (
    ToolCaps,
    PLAN_BLOCKED,
    PLAN_READONLY,
    SENSITIVE,
)


class BashCommandReadonlyTests(unittest.TestCase):
    def test_read_only_inspection_commands_allowed(self):
        for cmd in [
            "ls -la",
            "rg FastMCP mcp",
            "find . -name '*.py'",
            "cat file.py | head -20",
            "grep -n foo bar.py",
            "wc -l *.py",
            "which nvcc",
            "ls src && cat src/main.py",
        ]:
            self.assertTrue(bash_command_is_readonly(cmd), cmd)

    def test_build_and_exec_commands_rejected(self):
        for cmd in [
            "gcc solver.c -o solver.out -lm",
            "python -m pytest tests/",
            "python3 script.py",
            "make",
            "cmake --build .",
            "nvcc kernel.cu -o kernel.out",
            "node app.js",
            "pytest -q",
            "./solver.out",
            "chmod +x build.sh",        # a mode change is a write, not discovery
            "rg foo && python bar.py",  # one exec segment poisons the chain
            "pdflatex main.tex",        # a TeX engine writes its artefacts: an exec
            "cat foo > out.txt",        # redirection to a file
        ]:
            self.assertFalse(bash_command_is_readonly(cmd), cmd)

    def test_fd_redirection_stays_read_only(self):
        # Silencing or merging a stream adds no side effect, so it must not push a
        # discovery command out of plan mode. `which X 2>/dev/null [|| true]` is the
        # capability probe — rejecting it leaves no way to ask "is this available?"
        # except running commands that fail.
        for cmd in [
            "which nvcc 2>/dev/null",
            "which pdflatex 2>/dev/null || true",
            "grep -rn foo src 2>/dev/null",
            "module avail 2>&1 | grep -i tex",
        ]:
            self.assertTrue(bash_command_is_readonly(cmd), cmd)

    def test_inplace_and_output_write_flags_rejected(self):
        # sed -i / sort -o mutate the filesystem without a redirection, so they
        # are writes, not read-only discovery — even though their leading command
        # is in the inspection allowlist.
        for cmd in [
            "sed -i 's/x/y/' file.py",
            "sed -i.bak 's/x/y/' file.py",
            "sed --in-place 's/x/y/' file.py",
            "sed -ni 's/x/y/' file.py",
            "sort -o out.txt in.txt",
            "sort --output=out.txt in.txt",
            "cat a.py && sed -i s/x/y/ b.py",  # one write segment poisons the chain
        ]:
            self.assertFalse(bash_command_is_readonly(cmd), cmd)

    def test_sed_without_inplace_is_read_only(self):
        for cmd in [
            "sed -n '1,20p' file.py",
            "sed 's/x/y/' file.py",  # writes to stdout, not the file
        ]:
            self.assertTrue(bash_command_is_readonly(cmd), cmd)

    def test_separators_accepted_when_all_segments_read_only(self):
        for cmd in [
            "ls; pwd",
            "rg foo | grep bar | wc -l",
            "cat a.py && cat b.py",
            "grep x f || echo missing",
            "find . -name '*.c'; ls -la; cat main.c",
        ]:
            self.assertTrue(bash_command_is_readonly(cmd), cmd)

    def test_module_discovery_allowed_but_load_rejected(self):
        self.assertTrue(bash_command_is_readonly("module avail"))
        self.assertTrue(bash_command_is_readonly("module list"))
        self.assertTrue(bash_command_is_readonly("module show cuda"))
        self.assertFalse(bash_command_is_readonly("module load cuda"))
        self.assertFalse(bash_command_is_readonly("module load cuda && nvcc k.cu -o k"))

    def test_env_manager_query_allowed_but_install_rejected(self):
        # Asking what is installed is discovery a plan may do; installing is not
        # (it mutates the interpreter, and a plan is drafted without the network).
        for cmd in ["pip list", "pip show numpy", "pip freeze", "conda list", "conda info"]:
            self.assertTrue(bash_command_is_readonly(cmd), cmd)
        for cmd in ["pip install requests", "conda create -n e python=3.11",
                    "conda search numpy", "pip list && pip install requests"]:
            self.assertFalse(bash_command_is_readonly(cmd), cmd)

    def test_substitution_and_malformed_rejected(self):
        for cmd in [
            "cat $(which python)",
            "echo `whoami`",
            "cat ${HOME}/x",
            "diff <(ls) <(ls)",
            "",
            "   ",
            "ls\nrm -rf .",
            None,  # type: ignore[arg-type]
        ]:
            self.assertFalse(bash_command_is_readonly(cmd), repr(cmd))


class PlanModeToolVisibilityTests(unittest.TestCase):
    def test_plan_readonly_tool_stays_visible_plan_blocked_stripped(self):
        registry = {
            "bash_run": ToolCaps(name="bash_run", capabilities=frozenset({PLAN_READONLY, SENSITIVE})),
            "write_file": ToolCaps(name="write_file", capabilities=frozenset({PLAN_BLOCKED})),
            "grep": ToolCaps(name="grep", capabilities=frozenset()),
        }
        tools = [
            {"function": {"name": "bash_run"}},
            {"function": {"name": "write_file"}},
            {"function": {"name": "grep"}},
        ]
        kept = {t["function"]["name"] for t in tools_for_plan_mode(tools, registry)}
        self.assertEqual(kept, {"bash_run", "grep"})

    def test_ask_mode_gets_the_same_subset(self):
        # tools_for_plan_mode is the historical alias; both modes share one filter.
        self.assertIs(tools_for_plan_mode, tools_for_readonly_mode)


class _FakeAgent:
    def __init__(self, mode):
        self.mode = mode
        self.tool_caps = {
            "bash_run": ToolCaps(name="bash_run", capabilities=frozenset({PLAN_READONLY, SENSITIVE})),
            "code_run": ToolCaps(name="code_run", capabilities=frozenset({SENSITIVE})),
            "write_file": ToolCaps(name="write_file", capabilities=frozenset({PLAN_BLOCKED})),
        }

    @staticmethod
    def _normalize_arguments(args):
        return args


def _call(name, args=None):
    return {"id": "1", "function": {"name": name, "arguments": args or {}}}


class ReadonlyToolCallGuardTests(unittest.TestCase):
    """The call-time guard shared by plan and ask modes (readonly_guard)."""

    def test_plan_blocked_call_dropped_with_tool_error(self):
        agent = _FakeAgent("ask")
        messages: list[dict] = []
        kept = filter_readonly_tool_calls(
            [_call("write_file", {"path": "a.py"})],
            agent=agent, messages=messages, mode_label="ask",
        )
        self.assertEqual(kept, [])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "tool")
        self.assertIn("not available in ask mode", messages[0]["content"])

    def test_exec_bash_command_dropped_but_readonly_one_kept(self):
        agent = _FakeAgent("ask")
        messages: list[dict] = []
        kept = filter_readonly_tool_calls(
            [
                _call("bash_run", {"command": "rg foo src"}),
                _call("bash_run", {"command": "python train.py"}),
            ],
            agent=agent, messages=messages, mode_label="ask",
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["function"]["arguments"]["command"], "rg foo src")
        self.assertEqual(len(messages), 1)
        self.assertIn("read-only discovery commands", messages[0]["content"])

    def test_uncapped_tool_passes_through(self):
        agent = _FakeAgent("ask")
        messages: list[dict] = []
        calls = [_call("grep", {"pattern": "x"})]
        self.assertEqual(
            filter_readonly_tool_calls(calls, agent=agent, messages=messages, mode_label="ask"),
            calls,
        )
        self.assertEqual(messages, [])

    def test_mode_label_shapes_the_error_text(self):
        agent = _FakeAgent("plan")
        messages: list[dict] = []
        filter_readonly_tool_calls(
            [_call("write_file", {})], agent=agent, messages=messages, mode_label="plan",
        )
        self.assertIn("not available in plan mode", messages[0]["content"])


class ReadonlyBashApprovalExemptionTests(unittest.TestCase):
    def test_read_only_bash_exempt_from_approval_in_plan_mode(self):
        agent = _FakeAgent("plan")
        self.assertTrue(_readonly_bash_exempt(agent, "bash_run", {"command": "rg foo src"}))

    def test_exec_bash_not_exempt_in_plan_mode(self):
        agent = _FakeAgent("plan")
        self.assertFalse(_readonly_bash_exempt(agent, "bash_run", {"command": "python x.py"}))

    def test_read_only_bash_exempt_in_agent_mode(self):
        # The exemption now applies in any mode: read-only discovery runs unattended.
        agent = _FakeAgent("agent")
        self.assertTrue(_readonly_bash_exempt(agent, "bash_run", {"command": "rg foo src"}))

    def test_exec_bash_not_exempt_in_agent_mode(self):
        agent = _FakeAgent("agent")
        self.assertFalse(_readonly_bash_exempt(agent, "bash_run", {"command": "python x.py"}))

    def test_inplace_write_not_exempt_in_agent_mode(self):
        # `sed -i` mutates a file — it keeps the approval prompt in every mode.
        agent = _FakeAgent("agent")
        self.assertFalse(_readonly_bash_exempt(agent, "bash_run", {"command": "sed -i s/x/y/ f.py"}))

    def test_other_sensitive_tool_never_exempt(self):
        agent = _FakeAgent("agent")
        self.assertFalse(_readonly_bash_exempt(agent, "code_run", {"command": "rg foo"}))


if __name__ == "__main__":
    unittest.main()
