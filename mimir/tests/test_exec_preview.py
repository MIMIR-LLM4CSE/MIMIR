"""Hermetic tests for the exec-output preview (tool_execution/exec_preview.py).

The extractor turns an exec-shaped tool result (returncode + stdout/stderr —
the command-runner envelope) into the clipped ``exec`` display object attached
to ``tool_result`` events. Shape-driven: no tool names anywhere.
"""

import json
import unittest

from mimir.client.tool_execution.exec_preview import (
    _MAX_STREAM_CHARS,
    _MAX_STREAM_LINES,
    extract_exec_preview,
)


def _bash_ok(**over):
    payload = {"status": "ok", "returncode": 0, "stdout": "hello\n", "stderr": "", "cwd": "/work"}
    payload.update(over)
    return json.dumps(payload)


class ExecShapeDetection(unittest.TestCase):
    def test_ok_envelope_extracted(self):
        info = extract_exec_preview(_bash_ok(), {"command": "echo hello"})
        self.assertEqual(info["returncode"], 0)
        self.assertEqual(info["stdout"], "hello\n")
        self.assertEqual(info["stderr"], "")
        self.assertEqual(info["command"], "echo hello")
        self.assertEqual(info["cwd"], "/work")
        self.assertNotIn("truncated", info)

    def test_error_envelope_extracted(self):
        result = json.dumps({
            "status": "error", "error": "Command returned a non-zero exit code.",
            "returncode": 2, "stdout": "", "stderr": "boom\n", "cwd": "/work",
        })
        info = extract_exec_preview(result, {"command": "false"})
        self.assertEqual(info["returncode"], 2)
        self.assertEqual(info["stderr"], "boom\n")

    def test_non_exec_payload_returns_none(self):
        # A read-tool envelope has no returncode — not a command runner.
        result = json.dumps({"status": "ok", "content": "line1\nline2", "path": "a.py"})
        self.assertIsNone(extract_exec_preview(result, {"path": "a.py"}))

    def test_returncode_without_streams_returns_none(self):
        result = json.dumps({"status": "ok", "returncode": 0})
        self.assertIsNone(extract_exec_preview(result, {}))

    def test_non_json_returns_none(self):
        self.assertIsNone(extract_exec_preview("plain text result", {}))

    def test_non_numeric_returncode_returns_none(self):
        result = json.dumps({"status": "ok", "returncode": "n/a", "stdout": "x"})
        self.assertIsNone(extract_exec_preview(result, {}))

    def test_missing_stream_defaults_empty(self):
        result = json.dumps({"status": "ok", "returncode": 0, "stdout": "out\n"})
        info = extract_exec_preview(result, {})
        self.assertEqual(info["stderr"], "")


class CommandExtraction(unittest.TestCase):
    def test_command_key_priority(self):
        # `command` outranks `code` (same priority order as the detail preview).
        info = extract_exec_preview(_bash_ok(), {"code": "print(1)", "command": "ls"})
        self.assertEqual(info["command"], "ls")

    def test_full_multiline_command_kept(self):
        cmd = "for f in *.py; do\n  wc -l $f\ndone"
        info = extract_exec_preview(_bash_ok(), {"command": cmd})
        self.assertEqual(info["command"], cmd)

    def test_long_command_clipped(self):
        info = extract_exec_preview(_bash_ok(), {"command": "x" * 5000})
        self.assertTrue(info["command"].endswith("…"))
        self.assertLess(len(info["command"]), 5000)

    def test_no_command_arg(self):
        info = extract_exec_preview(_bash_ok(), {"path": "a.py"})
        self.assertNotIn("command", info)
        info = extract_exec_preview(_bash_ok(), None)
        self.assertNotIn("command", info)


class StreamClipping(unittest.TestCase):
    def test_ansi_stripped(self):
        info = extract_exec_preview(
            _bash_ok(stdout="\x1b[32mgreen\x1b[0m done\n"), {})
        self.assertEqual(info["stdout"], "green done\n")

    def test_stdout_keeps_tail(self):
        lines = "\n".join(f"line{i}" for i in range(_MAX_STREAM_LINES + 50))
        info = extract_exec_preview(_bash_ok(stdout=lines), {})
        self.assertTrue(info["truncated"])
        self.assertIn(f"line{_MAX_STREAM_LINES + 49}", info["stdout"])  # last line kept
        self.assertNotIn("line0\n", info["stdout"])                     # head dropped
        self.assertIn("earlier lines omitted", info["stdout"].splitlines()[0])

    def test_stderr_keeps_head(self):
        lines = "\n".join(f"err{i}" for i in range(_MAX_STREAM_LINES + 50))
        info = extract_exec_preview(_bash_ok(stderr=lines), {})
        self.assertTrue(info["truncated"])
        self.assertIn("err0", info["stderr"].splitlines()[0])            # head kept
        self.assertIn("more lines omitted", info["stderr"].splitlines()[-1])

    def test_char_budget(self):
        info = extract_exec_preview(_bash_ok(stdout="x" * (_MAX_STREAM_CHARS + 100)), {})
        self.assertTrue(info["truncated"])
        self.assertLessEqual(len(info["stdout"]), _MAX_STREAM_CHARS + 1)

    def test_server_truncated_flag_passthrough(self):
        info = extract_exec_preview(_bash_ok(truncated=True), {})
        self.assertTrue(info["truncated"])


if __name__ == "__main__":
    unittest.main()
