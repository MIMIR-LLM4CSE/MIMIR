"""The proxy direct-execution guard (``gates.check_proxy_exec``): during a session, running
the proxy's source directly is blocked; reading it is not.
"""

import json
import os
import tempfile
import unittest

from mimir.client.guardrails.policy import gates
from mimir.client.tools.capabilities import ToolCaps, CODE_EXEC, READ
# Same module object the guard imports via `from ...servers.proxy._lib import store`.
from mimir.servers.proxy._lib import store
from mimir.servers._shared.atomic import write_json  # noqa: E402


class _FakeAgent:
    """Minimal surface the guard touches: tool_caps + _json_error_payload."""

    def __init__(self) -> None:
        self.tool_caps = {
            "bash_run":     ToolCaps(
                name="bash_run", capabilities=frozenset({CODE_EXEC}),
                scope={"args": ["command"], "kind": "command_prefix"},
            ),
            # Executes, but through structured arguments rather than a command line:
            # its `proxy_name` is a bare name, which reading it as shell would take for
            # a program in command position.
            "proxy_exec":   ToolCaps(name="proxy_exec", capabilities=frozenset({CODE_EXEC})),
            # a non-exec tool that also names a path (fast-path abstain)
            "read_file_lines": ToolCaps(name="read_file_lines", capabilities=frozenset({READ})),
        }



class ProxyExecGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._saved_root = store._CACHE_DIR
        store._CACHE_DIR = self._tmp.name
        self.agent = _FakeAgent()

        # A proxy source and executable living outside any workspace.
        self.src = os.path.join(self._tmp.name, "proxy_src.py")
        self.exe = os.path.join(self._tmp.name, "proxy_bin")
        with open(self.src, "w") as fh:
            fh.write("print('hi')\n")

    def tearDown(self) -> None:
        store._CACHE_DIR = self._saved_root
        self._tmp.cleanup()

    # -- helpers ---------------------------------------------------------------

    def _init_session(self, name: str = "tiny") -> None:
        os.makedirs(store.opt_session_runs_dir(name), exist_ok=True)
        write_json(
            store.opt_config_file(name),
            {"proxy_name": name, "proxy_source_path": self.src},
        )
        write_json(store.registry_path(), {name: {"executable_path": self.exe}})
        store.write_active_session(name)

    def _check(self, tool: str, **args) -> str | None:
        return gates.check_proxy_exec(self.agent, tool, args, {})

    def _assert_blocked(self, res: str | None) -> None:
        self.assertIsNotNone(res)
        payload = json.loads(res)
        self.assertEqual(payload["status"], "error")
        self.assertIn("proxy_eval(op='run')", payload["hint"])

    # -- blocked: the proxy is executed in command position --------------------

    def test_bash_python_source_blocked(self) -> None:
        self._init_session()
        self._assert_blocked(self._check("bash_run", command=f"python {self.src}"))

    def test_bash_python_relative_path_blocked(self) -> None:
        self._init_session()
        # Relative to the directory the call runs in, which is the proxy's own here.
        self.agent.workspace_root = self._tmp.name
        self._assert_blocked(self._check("bash_run", command="python proxy_src.py"))

    def test_a_namesake_elsewhere_is_not_the_proxy(self) -> None:
        # Matched on basenames, every proxy_src.py anywhere was blocked as the proxy.
        self._init_session()
        with tempfile.TemporaryDirectory() as other:
            self.agent.workspace_root = other
            self.assertIsNone(self._check("bash_run", command="python proxy_src.py"))

    def test_bash_env_wrapper_blocked(self) -> None:
        self._init_session()
        self._assert_blocked(self._check("bash_run", command=f"env X=1 python {self.src}"))

    def test_bash_direct_executable_blocked(self) -> None:
        self._init_session()
        self._assert_blocked(self._check("bash_run", command=self.exe))

    def test_the_executable_with_arguments_is_blocked(self) -> None:
        # Named for what it runs: the guard keys on the command position, and every
        # exec tool reaches it through the same shell-command argument.
        self._init_session()
        self._assert_blocked(self._check("bash_run", command=self.exe + " --fast"))

    def test_chained_command_blocked(self) -> None:
        self._init_session()
        self._assert_blocked(self._check("bash_run", command=f"cd /tmp && python {self.src}"))

    def test_wrapper_arguments_are_not_taken_for_the_program(self) -> None:
        # 'timeout 300 ./bin' once resolved to the program '300', and the proxy ran
        # by hand in the middle of a session.
        self._init_session()
        for prefix in ("timeout 300", "timeout -k 5 10m", "srun -n 4",
                       "mpirun -np 8", "nice -n 5", "time", "time timeout 30",
                       "OMP_NUM_THREADS=4 timeout 300", "env A=1 srun -N 1"):
            with self.subTest(prefix=prefix):
                self._assert_blocked(self._check(
                    "bash_run",
                    command=f"cd /tmp && printf 'n=1\\n' > p && {prefix} {self.exe} p "
                            "2>&1 | tail -5"))

    def test_the_hint_names_the_op_that_lifts_the_guard(self) -> None:
        # 'reset' restores the baseline tree; only 'end' lifts the guard.
        self._init_session()
        hint = json.loads(self._check("bash_run", command=self.exe))["hint"]
        self.assertIn("proxy_eval(op='end')", hint)
        self.assertNotIn("op='reset'", hint)

    # -- allowed: inspection, unrelated targets, no session, non-exec tool -----

    def test_bash_cat_source_allowed(self) -> None:
        self._init_session()
        self.assertIsNone(self._check("bash_run", command=f"cat {self.src}"))

    def test_bash_grep_source_allowed(self) -> None:
        self._init_session()
        self.assertIsNone(self._check("bash_run", command=f"grep foo {self.src}"))

    def test_a_bare_wrapper_runs_nothing_of_interest(self) -> None:
        self._init_session()
        self.assertIsNone(self._check("bash_run", command="env"))
        self.assertIsNone(self._check("bash_run", command="timeout 5 ls /tmp"))

    def test_bash_other_program_allowed(self) -> None:
        self._init_session()
        self.assertIsNone(self._check("bash_run", command="python /somewhere/other.py"))

    def test_non_exec_tool_abstains(self) -> None:
        self._init_session()
        # read_file_lines carries no CODE_EXEC -> fast-path None even naming the source.
        self.assertIsNone(self._check("read_file_lines", path=self.src))

    def test_the_sanctioned_route_is_never_blocked(self) -> None:
        """The guard exists to push the model towards ``proxy_exec`` — so it must not
        block it. Its arguments are not a command line, and reading them as one takes
        the bare ``proxy_name`` for a program whose basename matches the target set.
        """
        self._init_session()
        self.assertIsNone(self._check("proxy_exec", op="run", proxy_name="proxy_bin"))
        self.assertIsNone(self._check("proxy_exec", op="run", proxy_name="proxy_src.py"))

    def test_no_session_allows_execution(self) -> None:
        # No active_session written -> guard abstains.
        self.assertIsNone(self._check("bash_run", command=f"python {self.src}"))

    def test_ended_session_allows_execution(self) -> None:
        self._init_session()
        self._assert_blocked(self._check("bash_run", command=f"python {self.src}"))
        store.clear_active_session()  # what proxy_eval(op='end') does
        self.assertIsNone(self._check("bash_run", command=f"python {self.src}"))

    def test_registered_executable_is_read_under_its_real_key(self) -> None:
        """The registry stores the binary under ``executable_path``.

        Reading any other key silently narrows the target set to the source file,
        letting ``./the_binary`` through while the session is active.
        """
        self._init_session()
        self.assertIn(os.path.realpath(self.exe), gates._proxy_exec_targets())


if __name__ == "__main__":
    unittest.main()
