"""Proxy direct-execution guard (client policy gate `gates._check_proxy_exec`).

While a proxy optimization session is active, running the proxy's source/executable
directly through a CODE_EXEC tool is blocked; read-only inspection of the source is
not. The guard reads the proxy store (single source of truth) read-only. It must be
the SAME `store` module object the guard imports, so we repoint
`mimir.servers.proxy._lib.store._CACHE_DIR` (not the sys.path `_lib.store` alias).

Run:
    python -m unittest mimir.tests.test_proxy_exec_guard -v
"""

import json
import os
import tempfile
import unittest

from mimir.client.guardrails.policy import gates
from mimir.client.context.capabilities import ToolCaps, CODE_EXEC, READ
# Same module object the guard imports via `from ...servers.proxy._lib import store`.
from mimir.servers.proxy._lib import store


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

    def _json_error_payload(self, msg, hint="", tool=""):
        return json.dumps({"status": "error", "error": msg, "hint": hint, "tool": tool})


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
        os.makedirs(store._opt_session_runs_dir(name), exist_ok=True)
        store._write_json_atomic(
            store._opt_config_file(name),
            {"proxy_name": name, "proxy_source_path": self.src},
        )
        store._write_json_atomic(store.registry_path(), {name: {"executable": self.exe}})
        store._write_active_session(name)

    def _check(self, tool: str, **args) -> str | None:
        return gates._check_proxy_exec(self.agent, tool, args, {})

    def _assert_blocked(self, res: str | None) -> None:
        self.assertIsNotNone(res)
        payload = json.loads(res)
        self.assertEqual(payload["status"], "error")
        self.assertIn("proxy_eval(op='run')", payload["hint"])

    # -- blocked: the proxy is executed in command position --------------------

    def test_bash_python_source_blocked(self) -> None:
        self._init_session()
        self._assert_blocked(self._check("bash_run", command=f"python {self.src}"))

    def test_bash_python_basename_blocked(self) -> None:
        self._init_session()
        # cwd-relative basename still resolves to the proxy source's basename.
        self._assert_blocked(self._check("bash_run", command="python proxy_src.py"))

    def test_bash_env_wrapper_blocked(self) -> None:
        self._init_session()
        self._assert_blocked(self._check("bash_run", command=f"env X=1 python {self.src}"))

    def test_bash_direct_executable_blocked(self) -> None:
        self._init_session()
        self._assert_blocked(self._check("bash_run", command=self.exe))

    def test_code_execute_executable_blocked(self) -> None:
        self._init_session()
        self._assert_blocked(self._check("bash_run", command=self.exe + " --fast"))

    def test_chained_command_blocked(self) -> None:
        self._init_session()
        self._assert_blocked(self._check("bash_run", command=f"cd /tmp && python {self.src}"))

    # -- allowed: inspection, unrelated targets, no session, non-exec tool -----

    def test_bash_cat_source_allowed(self) -> None:
        self._init_session()
        self.assertIsNone(self._check("bash_run", command=f"cat {self.src}"))

    def test_bash_grep_source_allowed(self) -> None:
        self._init_session()
        self.assertIsNone(self._check("bash_run", command=f"grep foo {self.src}"))

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
        store._clear_active_session()  # op='end'
        self.assertIsNone(self._check("bash_run", command=f"python {self.src}"))


if __name__ == "__main__":
    unittest.main()
