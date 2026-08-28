"""Annotations appended to a tool result: the fork hint, MORE_CONTENT and OUTLINE.

All live in ``tool_execution.executor`` next to VERDICT_DUE, and all say their piece at
the call they concern rather than at the end of the turn — which is the only moment the
cheap repair is still one call away.

Pure-Python + stubs (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

from mimir.client.context.capabilities import CODE_NAV, EDIT, READ, CACHEABLE, ToolCaps
from mimir.client.context.execution_context import build_execution_context
from mimir.client.tool_execution import bash_effect as be
from mimir.client.tool_execution import executor as ex
from mimir.client.tool_execution.formatter import (
    normalize_tool_content, parse_tool_payload,
)
from mimir.tests._golden_caps import build_declared_registry

_DECLARED_REGISTRY = build_declared_registry()


class ReadRenderingTests(unittest.TestCase):
    """Program text reaches the model as text, not as a JSON string literal."""

    def _rendered(self, payload: dict) -> str:
        block = types.SimpleNamespace(text=json.dumps(payload))
        return normalize_tool_content(types.SimpleNamespace(content=[block]))

    def test_the_block_leaves_the_json_and_keeps_its_real_newlines(self) -> None:
        out = self._rendered({
            "status": "ok", "path": "/w/s.py", "start_line": 1, "end_line": 3,
            "total_lines": 900, "truncated": True, "next_start_line": 4,
            "content": "1: a = 1\n2: b = 2\n3: c = 3",
        })
        self.assertIn("\n1: a = 1\n2: b = 2\n3: c = 3", out)
        self.assertNotIn("\\n", out)
        self.assertEqual(parse_tool_payload(out)["next_start_line"], 4)

    def test_quotes_and_backslashes_survive_verbatim(self) -> None:
        # An anchor is copied from what the model read. Escaped, `"\n".join(...)` reads
        # back as `\"\\n\".join(...)` and can never match the file.
        source = '1: sep = "\\n".join(parts)\n2: path = r"C:\\tmp"'
        out = self._rendered({"status": "ok", "path": "/w/s.py", "content": source})
        self.assertIn('sep = "\\n".join(parts)', out)
        self.assertIn('path = r"C:\\tmp"', out)

    def test_a_payload_without_a_text_block_is_untouched(self) -> None:
        out = self._rendered({"status": "ok", "matches": [{"path": "a.py"}]})
        self.assertEqual(parse_tool_payload(out)["matches"], [{"path": "a.py"}])
        self.assertNotIn("---", out)

    def test_a_run_hands_back_both_streams_unescaped(self) -> None:
        out = self._rendered({
            "status": "error", "returncode": 1, "cwd": "/w",
            "stdout": "collected 3 items\nF..",
            "stderr": 'Traceback:\n  File "t.py", line 4\nAssertionError',
        })
        self.assertIn("--- stdout ---\ncollected 3 items\nF..", out)
        self.assertIn('--- stderr ---\nTraceback:\n  File "t.py", line 4', out)
        self.assertNotIn("\\n", out)

    def test_every_consumer_still_reads_the_payload_it_always_read(self) -> None:
        # The split is a wire format for the model; the contract callers read is not
        # supposed to have changed.
        original = {
            "status": "ok", "returncode": 0, "cwd": "/w",
            "stdout": "one\ntwo\n", "stderr": "",
        }
        back = parse_tool_payload(self._rendered(original))
        self.assertEqual(back, original)

    def test_output_containing_a_marker_is_still_restored_exactly(self) -> None:
        # Lengths come from the manifest, so a run that prints the marker itself
        # cannot make the reader cut the block in the wrong place.
        payload = {"status": "ok", "returncode": 0,
                   "stdout": "a\n--- stderr ---\nb", "stderr": "real"}
        back = parse_tool_payload(self._rendered(payload))
        self.assertEqual(back["stdout"], "a\n--- stderr ---\nb")
        self.assertEqual(back["stderr"], "real")

    def test_trailing_annotations_never_bleed_into_the_last_block(self) -> None:
        out = self._rendered({"status": "ok", "returncode": 0, "stderr": "warn"})
        back = parse_tool_payload(out + "\n\nVERDICT_DUE: exit 0 says …")
        self.assertEqual(back["stderr"], "warn")
        self.assertEqual(back["status"], "ok")


def _agent():
    caps = {
        "write_file": ToolCaps(name="write_file", capabilities=frozenset({EDIT})),
        "read_file_lines": ToolCaps(
            name="read_file_lines", capabilities=frozenset({READ, CACHEABLE}),
        ),
    }
    return types.SimpleNamespace(
        _normalize_workspace_path=lambda p: p or "",
        tool_caps=caps,
    )


def _created(path: str) -> str:
    return json.dumps({"status": "ok", "operation": "created", "path": path})


class ForkHintTests(unittest.TestCase):
    """A second copy of a file already written this query is worth one remark."""

    def _hint(self, ec, path, result=None):
        # record_tool_observation has already run by the time the hint is built, so
        # the file just written is itself in the dirty set: the check must not read
        # it as its own fork.
        ec["dirty_written_files"].add(path)
        return ex._build_fork_hint(
            _agent(), "write_file", {"path": path},
            result if result is not None else _created(path), ec,
        )

    def test_suffixed_stem_is_read_as_a_fork(self) -> None:
        ec = build_execution_context()
        ec["dirty_written_files"].add("work/convergence_test.py")
        hint = self._hint(ec, "work/convergence_test_fixed.py")
        self.assertIn("FORK_SUSPECTED", hint)
        self.assertIn("convergence_test.py", hint)

    def test_short_stem_suffix_still_matches(self) -> None:
        ec = build_execution_context()
        ec["dirty_written_files"].add("work/solver.py")
        self.assertIn("FORK_SUSPECTED", self._hint(ec, "work/solver_ghost.py"))

    def test_unrelated_stem_is_left_alone(self) -> None:
        ec = build_execution_context()
        ec["dirty_written_files"].add("work/solver.py")
        self.assertEqual(self._hint(ec, "work/driver.py"), "")

    def test_a_sibling_in_another_directory_is_not_a_fork(self) -> None:
        ec = build_execution_context()
        ec["dirty_written_files"].add("other/solver.py")
        self.assertEqual(self._hint(ec, "work/solver_ghost.py"), "")

    def test_overwrite_of_an_existing_file_is_the_wanted_behaviour(self) -> None:
        ec = build_execution_context()
        ec["dirty_written_files"].add("work/convergence_test.py")
        updated = json.dumps({"status": "ok", "operation": "updated"})
        self.assertEqual(
            self._hint(ec, "work/convergence_test_fixed.py", updated), ""
        )

    def test_probe_name_outside_a_tests_dir_is_flagged(self) -> None:
        ec = build_execution_context()
        hint = self._hint(ec, "work/test_laplacian.py")
        self.assertIn("PROBE_PLACEMENT", hint)

    def test_the_same_name_inside_a_tests_dir_is_where_it_belongs(self) -> None:
        ec = build_execution_context()
        self.assertEqual(self._hint(ec, "pkg/tests/test_laplacian.py"), "")

    def test_it_speaks_once_per_query(self) -> None:
        ec = build_execution_context()
        ec["dirty_written_files"].add("work/solver.py")
        self.assertIn("FORK_SUSPECTED", self._hint(ec, "work/solver_ghost.py"))
        ec["dirty_written_files"].add("work/driver.py")
        self.assertEqual(self._hint(ec, "work/driver_v2.py"), "")

    def test_the_scratchpad_is_exactly_where_these_files_belong(self) -> None:
        from mimir.servers._shared.state_paths import scratch_dir
        from mimir.client.config.constants import STATE_DIR
        scratch = scratch_dir(STATE_DIR)
        ec = build_execution_context()
        ec["dirty_written_files"].add(os.path.join(scratch, "probe.py"))
        self.assertEqual(
            self._hint(ec, os.path.join(scratch, "probe_fixed.py")), ""
        )


class ContinuationHintTests(unittest.TestCase):
    """A window that stopped short says so, and says where to resume."""

    def _payload(self, **over):
        payload = {
            "path": "/w/solver.f90", "start_line": 1, "end_line": 400,
            "total_lines": 1523, "truncated": True, "next_start_line": 401,
            "line_cap": 400,
        }
        payload.update(over)
        return payload

    def test_a_truncated_read_names_the_next_call(self) -> None:
        hint = ex._build_continuation_hint(_agent(), "read_file_lines", self._payload())
        self.assertIn("MORE_CONTENT", hint)
        self.assertIn("1523 lines", hint)
        self.assertIn("start_line=401", hint)
        self.assertIn("400 lines are returned per call", hint)

    def test_a_complete_read_says_nothing(self) -> None:
        payload = self._payload(truncated=False, end_line=1523)
        self.assertEqual(ex._build_continuation_hint(_agent(), "read_file_lines", payload), "")


class OutlineHintTests(unittest.IsolatedAsyncioTestCase):
    """A page of a large file answers the wrong question; the symbol map answers it."""

    def _agent_with_outline(self, symbols, fail=False):
        agent = _agent()
        agent.tool_caps = dict(agent.tool_caps)
        agent.tool_caps["symbol_outline"] = ToolCaps(
            name="symbol_outline", capabilities=frozenset({READ, CACHEABLE, CODE_NAV})
        )
        agent._is_code_filepath = lambda p: p.endswith((".f90", ".py"))
        agent._parse_tool_payload = lambda text: json.loads(text)

        async def _run_tool(name, args, execution_context=None):
            if fail:
                raise RuntimeError("no ctags, no language server")
            return json.dumps({"status": "ok", "symbols": symbols})

        agent._run_tool = _run_tool
        return agent

    def _payload(self):
        return {"path": "/w/solver.f90", "truncated": True, "total_lines": 1523}

    async def test_a_truncated_read_of_code_gets_a_symbol_map(self) -> None:
        ec = build_execution_context()
        agent = self._agent_with_outline([
            {"name": "propagate_fwd", "line": 731},
            {"name": "propagate_bwd", "line": 1486},
        ])
        hint = await ex._build_outline_hint(agent, "read_file_lines", self._payload(), ec)
        self.assertIn("OUTLINE", hint)
        self.assertIn("propagate_bwd:1486", hint)

    async def test_it_speaks_once_per_file(self) -> None:
        ec = build_execution_context()
        agent = self._agent_with_outline([{"name": "solve", "line": 12}])
        await ex._build_outline_hint(agent, "read_file_lines", self._payload(), ec)
        again = await ex._build_outline_hint(agent, "read_file_lines", self._payload(), ec)
        self.assertEqual(again, "")

    async def test_no_outline_backend_is_not_an_error(self) -> None:
        ec = build_execution_context()
        agent = self._agent_with_outline([], fail=True)
        hint = await ex._build_outline_hint(agent, "read_file_lines", self._payload(), ec)
        self.assertEqual(hint, "")

    async def test_a_complete_read_needs_no_map(self) -> None:
        ec = build_execution_context()
        agent = self._agent_with_outline([{"name": "solve", "line": 12}])
        hint = await ex._build_outline_hint(
            agent, "read_file_lines", {"path": "/w/solver.f90"}, ec
        )
        self.assertEqual(hint, "")

    async def test_imports_and_fields_never_crowd_out_the_functions(self) -> None:
        # The budget is spent in document order, and a Python file opens with its
        # imports: unfiltered, forty slots went to names with no body and the map never
        # reached the code it was meant to locate. Kind does not separate the two — a
        # language server calls `from typing import Iterable` a class — so the body does.
        ec = build_execution_context()
        imports = [{"name": f"Imported{i}", "line": i, "end_line": i, "kind": "class"}
                   for i in range(1, 60)]
        fields = [{"name": f"field_{i}", "line": 60 + i, "end_line": 60 + i, "kind": ""}
                  for i in range(1, 20)]
        agent = self._agent_with_outline(
            imports + fields
            + [{"name": "propagate_fwd", "line": 731, "end_line": 902, "kind": "function"}]
        )
        hint = await ex._build_outline_hint(agent, "read_file_lines", self._payload(), ec)
        self.assertIn("propagate_fwd:731-902", hint)
        self.assertNotIn("Imported1:", hint)
        self.assertNotIn("field_1:", hint)
        self.assertNotIn("more)", hint)

    async def test_a_backend_reporting_no_kinds_still_gets_a_map(self) -> None:
        ec = build_execution_context()
        agent = self._agent_with_outline([{"name": "solve", "line": 12}])
        hint = await ex._build_outline_hint(agent, "read_file_lines", self._payload(), ec)
        self.assertIn("solve:12", hint)


class PathStampTests(unittest.TestCase):
    """The stamp is what makes answering from the cache safe."""

    def test_a_write_outside_the_edit_tools_changes_the_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "solver.py")
            with open(path, "w") as fh:
                fh.write("x = 1\n")
            first = ex._path_stamp(_agent(), {"path": path})
            self.assertIsNotNone(first)
            # A shell redirect or the user's own editor: no tool call to notice it.
            with open(path, "w") as fh:
                fh.write("x = 1\ny = 2\n")
            self.assertNotEqual(ex._path_stamp(_agent(), {"path": path}), first)

    def test_an_unreadable_or_absent_path_stamps_to_nothing(self) -> None:
        self.assertIsNone(ex._path_stamp(_agent(), {"path": "/nope/missing.py"}))
        self.assertIsNone(ex._path_stamp(_agent(), {}))


class ForkBaseStemTests(unittest.TestCase):
    def test_it_strips_one_trailing_word(self) -> None:
        self.assertEqual(ex._fork_base_stem("convergence_test_fixed"), "convergence_test")
        self.assertEqual(ex._fork_base_stem("solver_ghost"), "solver")

    def test_a_stem_with_no_suffix_yields_nothing(self) -> None:
        self.assertEqual(ex._fork_base_stem("solver"), "")

    def test_a_remainder_too_short_to_mean_anything_is_discarded(self) -> None:
        self.assertEqual(ex._fork_base_stem("a_b"), "")


if __name__ == "__main__":
    unittest.main()


class BashEffectTests(unittest.TestCase):
    """What a shell command changed, reported back because nothing else reports it.

    An edit through the file tools returns a diff and the prompt says to check it. A
    `sed -i` returns an empty line, so the model that just ran one has nothing to look
    at. The case this pins was real: an unanchored address inserted the same block
    eight times into a C++ header and the run never noticed.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.agent = types.SimpleNamespace(
            workspace_root=self.tmp,
            tool_caps=dict(_DECLARED_REGISTRY),
            _normalize_workspace_path=lambda p: (
                p if os.path.isabs(p) else os.path.join(self.tmp, p)),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _git(self, *args: str) -> None:
        subprocess.run(("git", *args), cwd=self.tmp, capture_output=True, check=False)

    def _make_repo(self) -> None:
        self._git("init", "-q", ".")
        self._git("config", "user.email", "t@t")
        self._git("config", "user.name", "t")

    def _write(self, name: str, text: str) -> str:
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def _run(self, command: str, execution_context=None) -> str:
        probe = be.capture(self.agent, "bash_run", {"command": command},
                           execution_context if execution_context is not None else {})
        if probe is not None:
            subprocess.run(command, shell=True, cwd=self.tmp, capture_output=True)
        return be.report(probe)

    _HEADER = "void a() {\n}\nvoid b() {\n}\nvoid c() {\n}\n"
    # The shape of the FUnTiDES command: an address matching every closing brace.
    _BAD_SED = ("sed -i '/^}$/{ i\\  .def(\"x\");\\n  .def(\"y\");\\n  .def(\"z\");\n}' hdr.h")

    def test_a_read_only_command_is_never_probed(self) -> None:
        # A grep that prints nothing has already said everything it has to say.
        self._make_repo()
        self._write("hdr.h", self._HEADER)
        self.assertIsNone(be.capture(self.agent, "bash_run", {"command": "grep -c def hdr.h"}, {}))

    def test_an_unanchored_sed_is_reported_with_its_duplication(self) -> None:
        self._make_repo()
        self._write("hdr.h", self._HEADER)
        self._git("add", "-A")
        self._git("commit", "-qm", "init")
        out = self._run(self._BAD_SED)
        self.assertIn("BASH_EFFECT", out)
        self.assertIn("hdr.h", out)
        self.assertIn("DUPLICATION_SUSPECTED", out)
        self.assertIn("3 times", out)

    def test_a_single_legitimate_write_is_reported_without_a_warning(self) -> None:
        self._make_repo()
        self._write("hdr.h", self._HEADER)
        self._git("add", "-A")
        self._git("commit", "-qm", "init")
        out = self._run("""printf '  .def("only");\\n' >> hdr.h""")
        self.assertIn("BASH_EFFECT", out)
        self.assertNotIn("DUPLICATION_SUSPECTED", out)

    def test_a_script_rewriting_a_file_it_never_names_is_still_reported(self) -> None:
        # The case classifying by Kind.WRITE would miss entirely: the command reads as
        # `exec` and credits the script, never the file it rewrites.
        self._make_repo()
        self._write("hdr.h", self._HEADER)
        self._write("fix.py", "open('hdr.h','a').write('// touched\\n')\n")
        self._git("add", "-A")
        self._git("commit", "-qm", "init")
        out = self._run(f"{sys.executable} fix.py")
        self.assertIn("BASH_EFFECT", out)
        self.assertIn("hdr.h", out)

    def test_a_created_file_is_listed(self) -> None:
        self._make_repo()
        self._write("keep.py", "x = 1\n")
        self._git("add", "-A")
        self._git("commit", "-qm", "init")
        probe = be.capture(self.agent, "bash_run", {"command": "cp keep.py keep.py.bak"}, {})
        subprocess.run("cp keep.py keep.py.bak", shell=True, cwd=self.tmp, capture_output=True)
        self.assertIn("keep.py.bak", be.report(probe))
        self.assertEqual(
            [os.path.basename(p) for p in be.created_paths(probe)], ["keep.py.bak"])

    def test_the_report_survives_outside_a_git_repo(self) -> None:
        # The fallback must not depend on parsing the command: this sed is opaque to the
        # classifier outright, and `printf … >> f` credits its format string, not the
        # redirect target. Guessing at the command is what this module exists to avoid.
        self._write("hdr.h", self._HEADER)
        out = self._run(self._BAD_SED)
        self.assertIn("BASH_EFFECT", out)
        self.assertIn("DUPLICATION_SUSPECTED", out)

    def test_a_command_that_changes_nothing_reports_nothing(self) -> None:
        self._make_repo()
        self._write("hdr.h", self._HEADER)
        self._git("add", "-A")
        self._git("commit", "-qm", "init")
        self.assertEqual(self._run("true"), "")


class RepeatedBlockTests(unittest.TestCase):
    """The duplication test is a period, not a repeated window.

    A window scan fires on ordinary code — three closing braces recur in every C++
    file — and once a block genuinely repeats, every longer window repeats too, so the
    largest match says nothing. A period says the whole inserted region is one block
    over and over, which is the signature of an address matching everywhere it looked.
    """

    def test_a_block_repeated_eight_times_is_found(self) -> None:
        self.assertEqual(be.repeated_block(["a", "b", "c"] * 8), (3, 8))

    def test_ordinary_recurring_lines_are_not_a_period(self) -> None:
        self.assertIsNone(be.repeated_block(
            ["}", "int x;", "}", "float y;", "}", "double z;", "return;"]))

    def test_a_single_block_is_not_a_repetition(self) -> None:
        self.assertIsNone(be.repeated_block(["a", "b", "c"]))

    def test_the_reported_block_is_the_real_repeating_unit(self) -> None:
        # Ten lines with period 2 also have period 4, 6 and 8. Reporting the shortest
        # describes what actually happened; the longest would say "twice" about a
        # block that was written five times.
        self.assertEqual(be.repeated_block(["a", "b"] * 5), (2, 5))

    def test_a_short_repetition_is_below_the_total_floor(self) -> None:
        self.assertIsNone(be.repeated_block(["a", "b"] * 2))

    def test_blank_lines_do_not_break_the_period(self) -> None:
        self.assertEqual(be.repeated_block(["a", "", "b", "c", "a", "b", "", "c"]), (3, 2))
