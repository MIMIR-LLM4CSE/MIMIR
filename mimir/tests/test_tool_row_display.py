"""Tool-activity row display: exact line range + no intra-row duplication.

Two UI-string fixes (backend; the webview renders these verbatim):
1. read_file_lines summary shows the exact range read ("lines 207-211") instead
   of a bare "2 lines".
2. A row detail that merely repeats the label (basename under a "Reading file:
   {path}" label) is dropped, so the file name is not shown twice in one row.

Run:
    python -m unittest mimir.tests.test_tool_row_display -v
"""

import asyncio
import json
import unittest

from mimir.client.context.capabilities import ToolCaps, READ, SEARCH_WITH_PATH
from mimir.client.tool_execution.tool_status_messages import (
    clip_doing,
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
    """The preview is dropped only when the row already says it.

    What it is deduplicated against is the model's own description of the call, since
    that is what the row puts beside it. Against the derived label — which the row no
    longer shows — it dropped previews the row needed and kept ones it did not.
    """

    def test_a_description_naming_the_file_drops_the_basename(self) -> None:
        self.assertEqual(
            dedup_row_detail("reading the loop in wave2d_proxy.py", "wave2d_proxy.py"), "")

    def test_matching_is_case_insensitive(self) -> None:
        self.assertEqual(dedup_row_detail("Checking Solver.py", "solver.py"), "")

    def test_a_preview_that_adds_something_is_kept(self) -> None:
        # The description says what the call is for; the preview says what it is for on.
        self.assertEqual(
            dedup_row_detail("running the regression suite", "python3 wave2d_proxy.py"),
            "python3 wave2d_proxy.py")
        self.assertEqual(
            dedup_row_detail("fetching the workflow file", "api.github.com/repos/foo/bar"),
            "api.github.com/repos/foo/bar")

    def test_a_short_preview_is_not_eaten_by_a_longer_word(self) -> None:
        """Matched on word boundaries, never as a substring.

        The previews are short words — a verdict is "pass", an op is "now" or "info" —
        and a substring test loses every one of them to an ordinary sentence. The
        verdict going missing from the row is why these previews exist.
        """
        for shown, detail in (
            ("recording the passing run", "pass"),
            ("knowing the time", "now"),
            ("informing the user", "info"),
            ("submitting it to the genoa partition", "gen"),
        ):
            with self.subTest(shown=shown):
                self.assertEqual(dedup_row_detail(shown, detail), detail)

    def test_the_same_word_on_its_own_still_drops(self) -> None:
        self.assertEqual(dedup_row_detail("recording what the run showed: pass", "pass"), "")

    def test_no_text_at_all_keeps_the_preview(self) -> None:
        self.assertEqual(dedup_row_detail("", "wave2d_proxy.py"), "wave2d_proxy.py")

    def test_it_is_the_label_that_is_matched_when_no_description_was_written(self) -> None:
        """Whatever the row shows beside the preview is what it is matched against.

        The dispatcher passes the description or, when the model wrote none, the label —
        which is what the row then shows. Against the description alone, a row read
        "Recording the plan: refonte des lignes" and then "refonte des lignes" again.
        """
        self.assertEqual(
            dedup_row_detail("Recording the plan: refonte des lignes",
                             "refonte des lignes"), "")

    def test_empty_detail_stays_empty(self) -> None:
        self.assertEqual(dedup_row_detail("reading the loop", ""), "")


class ObjectPreviewTests(unittest.TestCase):
    """The salient argument of a call: what it is acting on, or asking about.

    These reached the row through each server's label template ("Slurm cancel
    {job_id}", "Verdict: {verdict}") until the row stopped showing the label. Keyed on
    argument names, so a new tool needs no entry.
    """

    def test_the_verdict_a_call_records(self) -> None:
        self.assertEqual(tool_arg_preview("t", {"verdict": "pass", "reason": "14 passed"}),
                         "pass")

    def test_a_url_keeps_its_path(self) -> None:
        # The host alone said "it is reaching the network" and nothing about for what.
        self.assertEqual(
            tool_arg_preview("t", {"url": "https://api.github.com/repos/foo/bar/x.py"}),
            "api.github.com/repos/foo/bar/x.py")

    def test_a_long_url_keeps_the_end_of_its_path(self) -> None:
        # The end of a path names the thing; the middle is navigation.
        out = tool_arg_preview("t", {"url": "https://h.io/" + "deep/" * 20 + "solver.py"})
        self.assertTrue(out.startswith("h.io/…"), out)
        self.assertTrue(out.endswith("solver.py"), out)
        self.assertLessEqual(len(out), 48)

    def test_a_url_never_carries_its_credentials(self) -> None:
        """`hostname`, never `netloc`: a row is read over a shoulder and then stored.

        The port comes back, since it is what tells two local services apart.
        """
        out = tool_arg_preview("t", {"url": "http://user:secret@localhost:8080/v1/models"})
        self.assertEqual(out, "localhost:8080/v1/models")
        self.assertNotIn("secret", out)

    def test_a_url_the_parser_cannot_read_still_sheds_its_secrets(self) -> None:
        """The fallback echoes the value as written, which is where one survives.

        A schemeless `user:secret@host/x` parses to no host at all, so it took the
        branch that returns the string — and the secret went onto the row and into the
        stored transcript with it. A query goes the same way: an api key is usually in
        one, and it is not what identifies the call.
        """
        self.assertEqual(tool_arg_preview("t", {"url": "user:secret@host.io/x"}), "host.io/x")
        self.assertEqual(
            tool_arg_preview("t", {"url": "https://tok:x@api.example.com/v1?key=abcd1234"}),
            "api.example.com/v1")
        for leak in ("secret", "abcd1234"):
            for url in ("user:secret@host.io/x", "https://tok:x@api.example.com/v1?key=abcd1234"):
                self.assertNotIn(leak, tool_arg_preview("t", {"url": url}))

    def test_a_remote_file_is_named_with_its_repository(self) -> None:
        """`ci.yml` alone does not say which repository the call reached into.

        A GitHub row exists to show the remote it is touching, so the owner and the
        repository come before the path and take it with them. Clipped from the left,
        like a url: the end of the string names the thing.
        """
        self.assertEqual(
            tool_arg_preview("t", {"owner": "MIMIR-LLM4CSE", "repo": "MIMIR",
                                   "path": ".github/workflows/ci.yml"}),
            "MIMIR-LLM4CSE/MIMIR/.github/workflows/ci.yml")
        # No path: the repository is the object.
        self.assertEqual(tool_arg_preview("t", {"owner": "f", "repo": "b", "limit": 5}),
                         "f/b")
        out = tool_arg_preview("t", {"owner": "an-organisation-with-a-long-name",
                                     "repo": "a-long-repository", "path": "x/solver.cpp"})
        self.assertLessEqual(len(out), 48)
        self.assertTrue(out.endswith("solver.cpp"), out)

    def test_a_short_list_is_joined(self) -> None:
        # "numpy · scipy" is the row; "2 packages" is a row that must be expanded to
        # say anything at all.
        self.assertEqual(
            tool_arg_preview("t", {"packages": ["numpy", "scipy"],
                                   "python_executable": "/x/py"}),
            "numpy · scipy")

    def test_a_list_too_long_to_join_is_counted_by_its_own_name(self) -> None:
        # Seven steps clipped mid-sentence say less than their number does, and the
        # noun of the count is the argument's own name — no list of keys needed.
        self.assertEqual(
            tool_arg_preview("t", {"steps": [
                "read the dispatch loop", "fix the off-by-one bound",
                "run the row-display tests", "update the changelog"]}),
            "4 steps")

    def test_the_object_wins_over_the_action(self) -> None:
        # `op` selects an action, which the description already carries; the name of the
        # thing acted on is what the row cannot say otherwise.
        self.assertEqual(tool_arg_preview("t", {"op": "start", "name": "geos"}), "geos")
        self.assertEqual(tool_arg_preview("t", {"op": "solve", "equation": "x**2-4"}),
                         "x**2-4")

    def test_a_memory_operation_says_which_store_it_touched(self) -> None:
        # Two stores since 1.4.0, and one of these operations wipes one of them.
        self.assertEqual(
            tool_arg_preview("t", {"scope": "global", "text": "a fact worth keeping"}),
            "global")

    def test_the_action_is_the_fallback_when_there_is_no_object(self) -> None:
        self.assertEqual(tool_arg_preview("t", {"op": "disk_usage"}), "disk_usage")

    def test_a_numeric_identifier_is_not_dropped(self) -> None:
        self.assertEqual(tool_arg_preview("t", {"job_id": 12345}), "12345")

    def test_a_boolean_is_never_a_preview(self) -> None:
        self.assertEqual(tool_arg_preview("t", {"confirm": True, "op": "cancel"}), "cancel")

    def test_nothing_salient_reads_empty(self) -> None:
        self.assertEqual(tool_arg_preview("t", {"max_depth": 2, "use_cache": True}), "")


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

    def test_tool_without_a_path_role_still_gets_its_path_shortened(self):
        """The gap that left every write row showing an absolute path.

        The read tool declares a ``path`` role and the write, edit and delete tools
        do not, so a declared role alone shortened "Reading file:" and left "Editing
        file:" carrying the whole thing — the rows the user reads most.
        """
        reg = {"t": ToolCaps(name="t")}
        short = shorten_display_args("t", {"path": self.ABS}, reg)
        self.assertEqual(short["path"], "observations.py")

    def test_a_url_loses_its_credentials_in_the_display_copy(self):
        """Not just in the row's preview: in the label, which is the wider carrier.

        `label_for` interpolates the url verbatim ("Fetching {url}"), and that label is
        the row's tooltip, the approval card's header and a line of the stored
        transcript. The display copy of the arguments is where that is fixed for all of
        them at once. The scheme and the query stay — a consent prompt asks about a
        precise call, and only the credential is never part of it.
        """
        reg = {"t": ToolCaps(name="t")}
        short = shorten_display_args(
            "t", {"url": "https://tok:s3cr3t@api.example.com/v1?key=k"}, reg)
        self.assertEqual(short["url"], "https://api.example.com/v1?key=k")

    def test_a_url_without_credentials_is_untouched(self):
        reg = {"t": ToolCaps(name="t")}
        for url in ("https://api.example.com/v1/models?q=a@b", "host.io/a/b"):
            with self.subTest(url=url):
                self.assertEqual(shorten_display_args("t", {"url": url}, reg)["url"], url)

    def test_the_arguments_sent_keep_their_credentials(self):
        # Display-only: the call still has to be able to authenticate.
        reg = {"t": ToolCaps(name="t")}
        args = {"url": "https://tok:s3cr3t@api.example.com/v1"}
        shorten_display_args("t", args, reg)
        self.assertEqual(args["url"], "https://tok:s3cr3t@api.example.com/v1")

    def test_a_relative_path_is_left_alone_declared_or_not(self):
        """An argument named `path` is not always a filesystem path.

        A remote-fetch tool's repository path is already short, and is the one the row
        means: reduced to its basename, the GitHub row read `ci.yml` where the call was
        for `.github/workflows/ci.yml`. Only an absolute value is the problem this
        exists for, and a tool that names a workspace file is given one.
        """
        for reg in ({"t": ToolCaps(name="t")},
                    {"t": ToolCaps(name="t", arg_roles={"path": ("path",)})}):
            with self.subTest(declared="path" in (reg["t"].arg_roles or {})):
                short = shorten_display_args("t", {"path": "src/solver/core.cpp"}, reg)
                self.assertEqual(short["path"], "src/solver/core.cpp")

    def test_fallback_does_not_mutate_the_arguments_sent(self):
        reg = {"t": ToolCaps(name="t")}
        args = {"path": self.ABS}
        shorten_display_args("t", args, reg)
        self.assertEqual(args["path"], self.ABS)

    def test_declared_role_still_wins_over_the_generic_keys(self):
        """A declared role names the arguments; the key heuristic is only a fallback."""
        reg = {"t": ToolCaps(name="t", arg_roles={"path": ("target",)})}
        short = shorten_display_args("t", {"target": self.ABS, "path": self.ABS}, reg)
        self.assertEqual(short["target"], "observations.py")
        self.assertEqual(short["path"], self.ABS)

    def test_preview_also_shows_the_file_name(self):
        self.assertEqual(tool_arg_preview("t", {"path": self.ABS}), "observations.py")

    def test_preview_keeps_a_remote_path_whole(self):
        # `policy.gates` previews the raw arguments, so the rule lives here too.
        self.assertEqual(
            tool_arg_preview("t", {"path": ".github/workflows/ci.yml"}),
            ".github/workflows/ci.yml")

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


class ClipDoingTests(unittest.TestCase):
    """The model's per-call description, made fit for a one-line row.

    The 15-word limit is asked of the model, not enforced here: a sentence cut
    mid-word reads worse than a long one. This only stops a model that ignores the
    limit entirely from pushing the rest of the row off screen.
    """

    def test_nothing_in_nothing_out(self) -> None:
        for empty in (None, "", "   ", 42, {"a": 1}):
            self.assertEqual(clip_doing(empty), "")

    def test_a_sentence_passes_through_trimmed(self) -> None:
        self.assertEqual(clip_doing("  fixing the off-by-one bound  "),
                         "fixing the off-by-one bound")

    def test_only_the_first_line_survives(self) -> None:
        # A row is one line; a description with a newline would take the layout with it.
        self.assertEqual(clip_doing("\n\nreading the loop\nand then some\n"),
                         "reading the loop")

    def test_a_runaway_description_is_bounded(self) -> None:
        clipped = clip_doing("word " * 200)
        self.assertLess(len(clipped), 130)
        self.assertTrue(clipped.endswith("…"))


class _FakeAgent:
    """The slice of the agent ``_dispatch_tool_calls`` touches."""

    def __init__(self, registry):
        self.tool_caps = registry
        self.tool_owner = {name: "fake" for name in registry}
        self.model = "fake-model"
        self.approvals = None
        self.calls: list[tuple[str, dict]] = []

    def _normalize_arguments(self, args):
        # The real one hands back the dict it was given (formatter.normalize_arguments).
        # Copying here would hide exactly the mutation these tests are watching for.
        from mimir.client.tool_execution.formatter import normalize_arguments
        return normalize_arguments(args)

    def _rewrite_tool_for_context(self, name, args):
        return name, args

    def _is_write_tool(self, name):
        return False

    def get_tool_file_targets(self, *a, **k):
        return []

    async def _run_tool(self, name, args, **kwargs):
        self.calls.append((name, dict(args)))
        return json.dumps({"status": "ok"})


class DispatchRowFieldsTests(unittest.TestCase):
    """What the dispatcher puts on a row, and what it keeps off the call.

    ``doing`` is added to every tool's schema by the client (it is the user-facing
    description of the call, not an input), so the dispatcher is the one place that has
    to take it back out again — before the dedup key, and before the tool runs.
    """

    def _dispatch(self, tool_calls, registry=None, messages=None):
        from unittest.mock import patch
        from mimir.client.query_engine import dispatch as d

        agent = _FakeAgent(registry if registry is not None else dict(_REG))
        events: list[dict] = []
        with patch.object(d, "emit", events.append), \
             patch.object(d, "run_post_tool_annotations", lambda *a, **k: None):
            asyncio.run(d._dispatch_tool_calls(
                tool_calls, agent, [] if messages is None else messages, {}))
        return agent, [e for e in events if e.get("type") == "tool_call"]

    @staticmethod
    def _call(call_id, name, args):
        return {"id": call_id, "function": {"name": name, "arguments": args}}

    def test_the_row_carries_the_family_and_the_description(self) -> None:
        _agent, rows = self._dispatch([
            self._call("c1", "read_file_lines",
                       {"path": "/w/dispatch.py", "doing": "reading the dispatch loop"}),
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "read")
        self.assertEqual(rows[0]["doing"], "reading the dispatch loop")
        # The derived label stays on the wire: approval cards and the row's own
        # tooltip read it, even though the row no longer says it.
        self.assertTrue(rows[0]["label"])

    def test_the_description_never_reaches_the_tool(self) -> None:
        # No server declares the parameter — it would arrive as an unexpected kwarg.
        agent, _rows = self._dispatch([
            self._call("c1", "read_file_lines",
                       {"path": "/w/dispatch.py", "doing": "reading the loop"}),
        ])
        self.assertEqual(agent.calls, [("read_file_lines", {"path": "/w/dispatch.py"})])

    def test_two_calls_differing_only_in_description_are_one_call(self) -> None:
        # Left in the arguments, a reworded description walks straight past the dedup
        # and the repeat guard — the exact spin they exist to stop.
        agent, rows = self._dispatch([
            self._call("c1", "read_file_lines", {"path": "/w/a.py", "doing": "reading it"}),
            self._call("c2", "read_file_lines", {"path": "/w/a.py", "doing": "checking it"}),
        ])
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(len(rows), 1)

    def test_a_call_with_no_description_still_makes_a_row(self) -> None:
        # The model will forget. The family and the target carry the row alone.
        _agent, rows = self._dispatch([
            self._call("c1", "read_file_lines", {"path": "/w/a.py"}),
        ])
        self.assertEqual(rows[0]["doing"], "")
        self.assertEqual(rows[0]["kind"], "read")

    def test_a_description_is_clipped_before_it_is_sent(self) -> None:
        _agent, rows = self._dispatch([
            self._call("c1", "read_file_lines",
                       {"path": "/w/a.py", "doing": "reading it\nand explaining myself"}),
        ])
        self.assertEqual(rows[0]["doing"], "reading it")

    def test_the_model_s_own_message_is_left_as_it_issued_it(self) -> None:
        """The description is read out of the call, never popped out of it.

        ``normalize_arguments`` hands back the very dict the assistant message holds,
        and that message is the record of what the model issued — ``take_deferred_calls``
        reads the calls back out of it to resume a step that waited on the user. Popped
        in place, the description was erased from the record, and a call resumed after
        an approval came back with nothing to say.
        """
        from mimir.client.query_engine.deferral import take_deferred_calls

        call = self._call("c1", "read_file_lines",
                          {"path": "/w/a.py", "doing": "reading the loop"})
        messages = [{"role": "assistant", "tool_calls": [call]}]
        agent, rows = self._dispatch([call], messages=messages)

        self.assertEqual(rows[0]["doing"], "reading the loop")      # the row has it
        self.assertEqual(agent.calls[0][1], {"path": "/w/a.py"})    # the tool does not
        resumed = take_deferred_calls(messages, ["c1"])             # and the record kept it
        self.assertEqual(resumed[0]["function"]["arguments"],
                         {"path": "/w/a.py", "doing": "reading the loop"})

    def test_an_undeclared_tool_still_gets_a_family(self) -> None:
        _agent, rows = self._dispatch(
            [self._call("c1", "mystery_tool", {"doing": "doing something"})],
            registry={"mystery_tool": ToolCaps(name="mystery_tool")},
        )
        self.assertEqual(rows[0]["kind"], "tool")
