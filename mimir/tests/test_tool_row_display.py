"""Tool-activity row display: exact line range + no intra-row duplication.

Two UI-string fixes (backend; the webview renders these verbatim):
1. read_file_lines summary shows the exact range read ("lines 207-211") instead
   of a bare "2 lines".
2. A row detail that merely repeats the label (basename under a "Reading file:
   {path}" label) is dropped, so the file name is not shown twice in one row.

Run:
    python -m unittest mimir.tests.test_tool_row_display -v
"""

import json
import unittest

from mimir.client.context.capabilities import ToolCaps, READ, SEARCH_WITH_PATH
from mimir.client.tool_execution.tool_status_messages import (
    summarize_tool_result,
    dedup_row_detail,
    error_detail,
    shorten_display_args,
    tool_arg_preview,
)

_REG = {"read_file_lines": ToolCaps(name="read_file_lines",
                                    capabilities=frozenset({READ}))}


class LineRangeSummaryTests(unittest.TestCase):
    def _sum(self, payload: dict) -> str:
        return summarize_tool_result("read_file_lines", json.dumps(payload), _REG)[1]

    def test_multi_line_range(self) -> None:
        self.assertEqual(
            self._sum({"status": "ok", "start_line": 207, "end_line": 211,
                       "content": "a\nb\nc\nd\ne\n"}),
            "lines 207-211")

    def test_single_line_range(self) -> None:
        self.assertEqual(
            self._sum({"status": "ok", "start_line": 42, "end_line": 42,
                       "content": "a\n"}),
            "line 42")

    def test_falls_back_to_count_without_range(self) -> None:
        # No start/end in the payload → the count fallback (content.count("\n")+1).
        self.assertEqual(
            self._sum({"status": "ok", "content": "a\nb\nc"}),
            "3 lines")


class ErrorRowTests(unittest.TestCase):
    def test_error_row_carries_its_message(self) -> None:
        ok, summary = summarize_tool_result(
            "read_file_lines",
            json.dumps({"status": "error", "error": "No such file"}), _REG)
        self.assertFalse(ok)
        self.assertEqual(summary, "No such file")


class SearchRowCountTests(unittest.TestCase):
    """A search row counts whatever list the tool reports its hits in.

    The count used to read only "matches", which none of the tools carrying
    SEARCH_WITH_PATH return — so every one of their rows was blank.
    """

    _CAPS = {"find_references": ToolCaps(name="find_references",
                                         capabilities=frozenset({SEARCH_WITH_PATH}))}

    def _sum(self, payload: dict) -> str:
        return summarize_tool_result(
            "find_references", json.dumps(payload), self._CAPS)[1]

    def test_references_are_counted(self) -> None:
        self.assertEqual(
            self._sum({"status": "ok", "references": [{"line": 1}, {"line": 9}]}),
            "2 references")

    def test_a_single_hit_reads_singular(self) -> None:
        self.assertEqual(
            self._sum({"status": "ok", "references": [{"line": 1}]}),
            "1 reference")

    def test_no_hits_reads_zero_not_blank(self) -> None:
        self.assertEqual(self._sum({"status": "ok", "references": []}), "0 references")


class DedupRowDetailTests(unittest.TestCase):
    def test_basename_in_path_label_dropped(self) -> None:
        label = "Reading file: /work/proj/ratchet_demo/wave2d_proxy.py"
        self.assertEqual(dedup_row_detail(label, "wave2d_proxy.py"), "")

    def test_relpath_detail_in_absolute_label_dropped(self) -> None:
        label = "Reading file: /work/proj/ratchet_demo/wave2d_proxy.py"
        self.assertEqual(
            dedup_row_detail(label, "ratchet_demo/wave2d_proxy.py"), "")

    def test_command_detail_under_generic_label_kept(self) -> None:
        # bash_run: generic label, the command adds real information → keep it.
        self.assertEqual(
            dedup_row_detail("Running shell command", "python3 wave2d_proxy.py"),
            "python3 wave2d_proxy.py")

    def test_empty_detail_stays_empty(self) -> None:
        self.assertEqual(dedup_row_detail("Reading file: x.py", ""), "")


class PolicyBlockSummaryTests(unittest.TestCase):
    """A policy-blocked call must read as a clean block, not a cropped JSON error.

    The tool never ran (preconditions block before dispatch), so the row should say
    "⛔ blocked · <reason>" rather than 100 truncated chars of the violation payload.
    """

    def _sum(self, payload: dict):
        return summarize_tool_result("replace_in_file", json.dumps(payload), _REG)

    def test_write_policy_block_is_short_and_marked_failed(self) -> None:
        ok, summary = self._sum({
            "status": "error", "policy_stage": "write_policy",
            "error": "Write blocked: read the file first before editing so the change "
                     "is grounded in the actual current content and does not clobber ...",
        })
        self.assertFalse(ok)
        self.assertEqual(summary, "⛔ blocked · read the file first")

    def test_state_guard_block_not_reported_as_success(self) -> None:
        # status="blocked" (not "error") used to fall through and render as ok=True.
        ok, summary = self._sum({"status": "blocked", "policy_stage": "state_guard",
                                 "error": "validate the pending files first"})
        self.assertFalse(ok)
        self.assertIn("validate", summary)

    def test_plain_tool_error_unchanged(self) -> None:
        ok, summary = self._sum({"status": "error", "error": "Target text was not found."})
        self.assertFalse(ok)
        self.assertEqual(summary, "Target text was not found.")


class AppendedAdvisorySummaryTests(unittest.TestCase):
    """A successful edit must stay a success even when advisory text is appended.

    The client appends AUTO_VALIDATION / MORE_CONTENT / OUTLINE text AFTER the JSON
    payload. A post-write validator (lint/typecheck/import) embedding a nested
    ``"status": "error"`` must NOT flip a successful edit's row to failed — the file
    was written; the validator finding is advisory.
    """

    _edit_ok = {
        "status": "ok", "operation": "replaced", "path": "foo.py",
        "replacements": 1, "diff": "--- foo.py\n+++ foo.py\n-old\n+new",
    }

    def test_edit_ok_with_failing_validation_stays_ok(self) -> None:
        result = (
            json.dumps(self._edit_ok)
            + "\n\nAUTO_VALIDATION\ncode_lint(foo.py):\n"
            + json.dumps({"status": "error", "error": "F401 unused import"})
            + "\n\nLINT_FAILED: fix issues."
        )
        ok, _summary = summarize_tool_result("replace_in_file", result, _REG)
        self.assertTrue(ok)

    def test_edit_ok_with_clean_validation_stays_ok(self) -> None:
        result = (
            json.dumps(self._edit_ok)
            + "\n\nAUTO_VALIDATION\ncode_lint(foo.py):\n"
            + json.dumps({"status": "ok"})
        )
        ok, _summary = summarize_tool_result("replace_in_file", result, _REG)
        self.assertTrue(ok)

    def test_genuine_edit_failure_still_failed(self) -> None:
        # An actually-failed edit (leading payload is status=error) stays failed
        # even if advisory text is appended after it.
        result = (
            json.dumps({"status": "error", "error": "Target text was not found."})
            + "\n\nAUTO_VALIDATION\ncode_lint(foo.py):\n"
            + json.dumps({"status": "ok"})
        )
        ok, summary = summarize_tool_result("replace_in_file", result, _REG)
        self.assertFalse(ok)
        self.assertEqual(summary, "Target text was not found.")


class ErrorDetailTests(unittest.TestCase):
    """The row summary is a clipped one-liner; the UI panel needs the full text."""

    def test_multiline_error_kept_whole_without_hint(self) -> None:
        payload = {
            "status": "error",
            "error": "line one\nline two " + "x" * 200,
            "hint": "try a narrower query",
        }
        detail = error_detail(json.dumps(payload))
        self.assertIn("line two", detail)
        self.assertIn("x" * 200, detail)          # not clipped at 100 like the summary
        # The hint is guidance for the model, not for the user reading the row.
        self.assertNotIn("try a narrower query", detail)

    def test_policy_block_without_error_names_the_stage(self) -> None:
        detail = error_detail(json.dumps({"status": "error", "policy_stage": "approval"}))
        self.assertIn("approval", detail)

    def test_plain_text_error_passes_through(self) -> None:
        self.assertEqual(error_detail("Error: boom"), "Error: boom")

    def test_empty_result_has_no_detail(self) -> None:
        self.assertEqual(error_detail(""), "")
        self.assertEqual(error_detail(None), "")  # type: ignore[arg-type]

    def test_very_long_error_is_bounded(self) -> None:
        detail = error_detail(json.dumps({"status": "error", "error": "y" * 10000}))
        self.assertLess(len(detail), 4100)
        self.assertTrue(detail.endswith("(truncated)"))


class RowPathsAreShortenedTests(unittest.TestCase):
    """Activity rows show the file name; approval prompts keep the absolute path.

    Tools carry absolute paths now (see ``server_files._require_abs``), which is
    right for the model and unreadable for a person — a row reading
    "Reading file: /long/absolute/path/.../observations.py" buries the one token
    the user is scanning for. The shortening is capability-driven off the declared
    ``path`` arg-role, so it needs no tool-name list.
    """

    ABS = "/work/proj/codes/mimir/client/guardrails/observations.py"

    def _reg(self, label=None):
        return {"t": ToolCaps(name="t", arg_roles={"path": ("path",)}, label=label)}

    def test_label_template_renders_the_file_name(self):
        from mimir.client.context.capabilities import label_for
        reg = self._reg(label="Reading file: {path}")
        short = shorten_display_args("t", {"path": self.ABS}, reg)
        self.assertEqual(label_for("t", short, reg), "Reading file: observations.py")

    def test_original_arguments_are_not_mutated(self):
        # Display-only: the dict actually sent to the tool must keep its absolute path.
        args = {"path": self.ABS}
        shorten_display_args("t", args, self._reg())
        self.assertEqual(args["path"], self.ABS)

    def test_non_path_arguments_are_untouched(self):
        reg = self._reg()
        short = shorten_display_args("t", {"path": self.ABS, "old_text": "a/b/c"}, reg)
        self.assertEqual(short["old_text"], "a/b/c")

    def test_tool_without_a_path_role_is_passed_through(self):
        reg = {"t": ToolCaps(name="t")}
        args = {"path": self.ABS}
        self.assertIs(shorten_display_args("t", args, reg), args)

    def test_preview_also_shows_the_file_name(self):
        self.assertEqual(tool_arg_preview("t", {"path": self.ABS}), "observations.py")

    def test_directory_path_keeps_its_last_component(self):
        self.assertEqual(
            tool_arg_preview("t", {"path": "/a/b/pkg/"}), "pkg")

    def test_out_of_workspace_card_keeps_the_absolute_path(self):
        """The location is the decision being approved — it is never shortened.

        The card carries ``oow_paths`` verbatim (rendered by the webview as an
        explicit "outside workspace" line, one row per path); only the header label
        is shortened.
        """
        import inspect
        from mimir.client.ui.ws import ws_worker
        src = inspect.getsource(ws_worker._AgentWorker._path_approval_shim)
        self.assertIn('"oow_paths": list(paths)', src)
        self.assertNotIn("os.path.basename(paths[0])}\",", src)

    def test_cli_prompt_prints_the_absolute_path(self):
        import inspect
        from mimir.client.agent_core import MimirAgent
        src = inspect.getsource(MimirAgent._request_path_approval)
        self.assertIn('f"  {p}" for p in paths', src)


if __name__ == "__main__":
    unittest.main()
