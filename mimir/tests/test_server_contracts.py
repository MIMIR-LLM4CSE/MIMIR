import importlib.util
import re
import tempfile
import unittest
from pathlib import Path
import sys

from mimir.servers._shared import shell_paths as _shell_paths


SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
_SHARED_DIR = SERVERS_DIR / "_shared"

# Add _shared/ and each group subdirectory to sys.path so modules can be
# loaded via spec_from_file_location and their imports resolve correctly.
for _p in [
    _SHARED_DIR,
    SERVERS_DIR / "workspace",
    SERVERS_DIR / "utilities",
    SERVERS_DIR / "agent_state",
    SERVERS_DIR / "external",
    SERVERS_DIR / "hpc",
    SERVERS_DIR / "proxy",
    SERVERS_DIR / "ml",
]:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)


# Mapping from logical module name to (subdirectory, filename).
_MODULE_PATHS: dict[str, tuple[str, str]] = {
    "responses":              ("_shared",      "responses.py"),
    "text_tools":             ("_shared",      "text_tools.py"),
    "server_bash":            ("workspace",    "server_bash.py"),
    "server_datetime":        ("utilities",    "server_datetime.py"),
    "server_files":           ("workspace",    "server_files.py"),
    "server_math":            ("utilities",    "server_math.py"),
    "server_platform":        ("hpc",          "server_platform.py"),
    "server_env":             ("hpc",          "server_env.py"),
    "server_search":          ("workspace",    "server_search.py"),
    "server_strings":         ("utilities",    "server_strings.py"),
    "server_symbolic_math":   ("utilities",    "server_symbolic_math.py"),
    "server_system":          ("external",     "server_system.py"),
    "server_web":             ("external",     "server_web.py"),
    "server_todo":            ("agent_state",  "server_todo.py"),
}


def _load_server_module(module_name: str):
    if module_name in _MODULE_PATHS:
        subdir, filename = _MODULE_PATHS[module_name]
        module_path = SERVERS_DIR / subdir / filename
    else:
        module_path = SERVERS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec for {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared_responses = _load_server_module("responses")
server_bash = _load_server_module("server_bash")
server_datetime = _load_server_module("server_datetime")
server_files = _load_server_module("server_files")
server_math = _load_server_module("server_math")
server_platform = _load_server_module("server_platform")
server_env = _load_server_module("server_env")
server_search = _load_server_module("server_search")
server_strings = _load_server_module("server_strings")
server_symbolic_math = _load_server_module("server_symbolic_math")
server_system = _load_server_module("server_system")

try:
    import sympy as _sympy  # noqa: F401
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False
server_web = _load_server_module("server_web")
server_todo = _load_server_module("server_todo")
shared_text_tools = _load_server_module("text_tools")


class SharedResponsesTests(unittest.TestCase):
    def test_ok_response_shape(self) -> None:
        payload = shared_responses.ok({"value": 3})
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["value"], 3)

    def test_err_response_shape(self) -> None:
        payload = shared_responses.err("boom", hint="retry", code=12)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"], "boom")
        self.assertEqual(payload["hint"], "retry")
        self.assertEqual(payload["code"], 12)

    def test_ok_supports_kwargs(self) -> None:
        payload = shared_responses.ok(result=7, count=1)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["result"], 7)
        self.assertEqual(payload["count"], 1)

    def test_reserved_keys_cannot_override_protocol(self) -> None:
        payload_ok = shared_responses.ok({"status": "error", "error": "x", "value": 3})
        self.assertEqual(payload_ok["status"], "ok")
        self.assertEqual(payload_ok["value"], 3)
        self.assertNotIn("error", payload_ok)

        payload_err = shared_responses.err("boom", status="ok", error="x", hint="h", value=4)
        self.assertEqual(payload_err["status"], "error")
        self.assertEqual(payload_err["error"], "boom")
        self.assertEqual(payload_err["hint"], "h")
        self.assertEqual(payload_err["value"], 4)

    def test_err_adds_auto_hint_when_missing(self) -> None:
        payload = shared_responses.err("File not found: demo.txt")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"], "File not found: demo.txt")
        self.assertIn("Check the path/name", payload["hint"])

    def test_err_normalizes_whitespace(self) -> None:
        payload = shared_responses.err("  Invalid   syntax   in input  ")
        self.assertEqual(payload["error"], "Invalid syntax in input")


class SharedTextToolsTests(unittest.TestCase):
    def test_tokenize_splits_and_normalizes(self) -> None:
        tokens = shared_text_tools.tokenize("Server-Files_v2.py")
        self.assertIn("server", tokens)
        self.assertIn("files", tokens)
        self.assertIn("v2", tokens)

    def test_score_filename_hint_prefers_exact_name(self) -> None:
        exact = shared_text_tools.score_filename_hint("mimir/servers/workspace/server_search.py", "server_search.py")
        partial = shared_text_tools.score_filename_hint("mimir/servers/workspace/server_search.py", "search")
        self.assertGreater(exact, partial)


class RepresentativeServerContractTests(unittest.TestCase):
    def test_system_get_env_var_invalid_is_structured_error(self) -> None:
        payload = server_system.system("env", name="NOT_ALLOWED")
        self.assertEqual(payload["status"], "error")
        self.assertIn("allowed variable list", payload["error"])

    def test_system_info_ok(self) -> None:
        payload = server_system.system("info")
        self.assertEqual(payload["status"], "ok")
        self.assertIn("os", payload)

    def test_system_unknown_op_is_structured_error(self) -> None:
        payload = server_system.system("nope")
        self.assertEqual(payload["status"], "error")
        self.assertIn("Unknown system op", payload["error"])

    def test_search_outside_root_is_structured_error(self) -> None:
        payload = server_search.list_directory("../../../etc")
        self.assertEqual(payload["status"], "error")

    def test_files_list_files_empty_path_is_treated_as_root(self) -> None:
        payload = server_files.list_files("")
        self.assertEqual(payload["status"], "ok")
        self.assertIn("entries", payload)

    def test_search_list_directory_empty_path_is_treated_as_root(self) -> None:
        payload = server_search.list_directory("")
        self.assertEqual(payload["status"], "ok")
        self.assertIn("entries", payload)

    def test_search_tree_summary_workspace_basename_collapses_to_root(self) -> None:
        root_base = Path(server_search.SEARCH_ROOT).name
        payload = server_search.tree_summary(path=root_base, max_depth=1, max_entries=20, use_cache=False)
        self.assertEqual(payload["status"], "ok")
        self.assertIn("tree", payload)

    def test_web_parse_json_returns_structured_data(self) -> None:
        payload = server_web.parse_json('{"a": 1, "b": [2, 3]}')
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["data"]["b"], [2, 3])

    def test_web_json_extract_returns_structured_value(self) -> None:
        payload = server_web.json_extract('{"results": [{"title": "x"}]}', "results.0.title")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["value"], "x")

    def test_files_missing_file_is_structured_error(self) -> None:
        payload = server_search.read_file_lines("does-not-exist.txt")
        self.assertEqual(payload["status"], "error")

    def test_search_read_is_capped_and_says_where_to_resume(self) -> None:
        # No call may return a whole large file: the cap is what keeps reading targeted.
        with tempfile.TemporaryDirectory() as d:
            old_root = server_search.SEARCH_ROOT
            server_search.SEARCH_ROOT = d
            try:
                path = Path(d) / "big.f90"
                path.write_text("".join(f"line {i}\n" for i in range(1, 1001)))
                payload = server_search.read_file_lines(str(path), 1, 0)
                self.assertEqual(payload["status"], "ok")
                self.assertEqual(payload["lines_returned"], server_search._MAX_READ_LINES)
                self.assertEqual(payload["total_lines"], 1000)
                self.assertTrue(payload["truncated"])
                self.assertEqual(
                    payload["next_start_line"], server_search._MAX_READ_LINES + 1
                )
                self.assertEqual(payload["line_cap"], server_search._MAX_READ_LINES)
            finally:
                server_search.SEARCH_ROOT = old_root

    def test_search_read_of_a_short_file_comes_back_whole(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            old_root = server_search.SEARCH_ROOT
            server_search.SEARCH_ROOT = d
            try:
                (Path(d) / "small.py").write_text("x = 1\ny = 2\n")
                payload = server_search.read_file_lines(str(Path(d) / "small.py"), 1, 0)
                self.assertEqual(payload["lines_returned"], 2)
                self.assertNotIn("truncated", payload)
                self.assertNotIn("line_cap", payload)
            finally:
                server_search.SEARCH_ROOT = old_root


class PlatformPortabilityTests(unittest.TestCase):
    """The probe must describe the host it is on, not assume an x86 one.

    MIMIR is meant to run on whatever cluster it is dropped into, and clusters mix
    architectures. Asking an aarch64 host whether it has AVX-512 always answers no,
    which a reader takes as "no vector unit" rather than "a different one".
    """

    def _probe_cpu(self, arch: str, lscpu: str) -> dict:
        server_platform._collect_cpu.cache_clear()
        orig_run, orig_exists, orig_machine = (
            server_platform._run, server_platform._cmd_exists, server_platform.platform.machine)
        server_platform._run = lambda cmd, timeout=8: {
            "ok": True, "returncode": 0, "stdout": lscpu, "stderr": ""}
        server_platform._cmd_exists = lambda name: True
        server_platform.platform.machine = lambda: arch
        try:
            return server_platform._collect_cpu()
        finally:
            (server_platform._run, server_platform._cmd_exists,
             server_platform.platform.machine) = orig_run, orig_exists, orig_machine
            server_platform._collect_cpu.cache_clear()

    def test_aarch64_reports_its_own_vector_extensions(self) -> None:
        # aarch64 lscpu prints the extension list under "Features", not "Flags".
        cpu = self._probe_cpu("aarch64", "Architecture: aarch64\nFeatures: fp asimd sve sve2 bf16\n")
        self.assertEqual(cpu["simd"], {"asimd": True, "sve": True, "sve2": True,
                                       "bf16": True, "i8mm": False})
        self.assertNotIn("avx512f", cpu["simd"])

    def test_x86_reports_avx(self) -> None:
        cpu = self._probe_cpu("x86_64", "Architecture: x86_64\nFlags: fma avx2 avx512f\n")
        self.assertTrue(cpu["simd"]["avx2"])
        self.assertTrue(cpu["simd"]["avx512f"])
        self.assertFalse(cpu["simd"]["amx_tile"])

    def test_unknown_architecture_says_so_instead_of_guessing(self) -> None:
        cpu = self._probe_cpu("riscv64", "Architecture: riscv64\nFlags: rv64imafdc\n")
        self.assertEqual(cpu["simd"], {})
        self.assertIn("riscv64", cpu["simd_note"])

    def test_non_nvidia_accelerator_is_not_reported_as_no_gpu(self) -> None:
        """Claiming "no GPU" on a host whose accelerator this probe cannot read is
        worse than saying the probe does not cover it."""
        server_platform._collect_gpu.cache_clear()
        orig = server_platform._cmd_exists
        server_platform._cmd_exists = lambda name: name == "rocm-smi"
        try:
            gpu = server_platform._collect_gpu()
        finally:
            server_platform._cmd_exists = orig
            server_platform._collect_gpu.cache_clear()
        self.assertTrue(gpu["available"])
        self.assertEqual(gpu["vendors"], ["amd"])
        self.assertIn("not enumerated", gpu["note"])


class MathServerTests(unittest.TestCase):
    """`evaluate` is now the sole math tool; it subsumes the old arithmetic wrappers."""

    def test_evaluate_addition(self) -> None:
        payload = server_math.evaluate("3 + 4")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["result"], 7.0)

    def test_evaluate_subtraction(self) -> None:
        self.assertEqual(server_math.evaluate("10 - 3")["result"], 7.0)

    def test_evaluate_multiplication(self) -> None:
        self.assertEqual(server_math.evaluate("6 * 7")["result"], 42.0)

    def test_evaluate_division(self) -> None:
        self.assertAlmostEqual(server_math.evaluate("10 / 4")["result"], 2.5)

    def test_evaluate_division_by_zero_is_structured_error(self) -> None:
        payload = server_math.evaluate("5 / 0")
        self.assertEqual(payload["status"], "error")
        self.assertIn("zero", payload["error"].lower())

    def test_evaluate_modulo(self) -> None:
        self.assertEqual(server_math.evaluate("10 % 3")["result"], 1.0)

    def test_evaluate_power(self) -> None:
        self.assertEqual(server_math.evaluate("2 ** 10")["result"], 1024.0)

    def test_evaluate_sqrt(self) -> None:
        self.assertEqual(server_math.evaluate("sqrt(144)")["result"], 12.0)

    def test_evaluate_expression_returns_result(self) -> None:
        payload = server_math.evaluate("(3 + 4) * 2")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["result"], 14.0)

    def test_evaluate_with_math_function(self) -> None:
        payload = server_math.evaluate("sqrt(64)")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["result"], 8.0)

    def test_evaluate_invalid_expression_is_structured_error(self) -> None:
        payload = server_math.evaluate("__import__('os').system('id')")
        self.assertEqual(payload["status"], "error")

    def test_response_includes_expression_string(self) -> None:
        payload = server_math.evaluate("1 + 2")
        self.assertIn("expression", payload)


class StringsServerTests(unittest.TestCase):
    """All string ops are dispatched through the single `string_op(op, ...)` tool."""

    def test_reverse_returns_reversed_string(self) -> None:
        payload = server_strings.string_op("reverse", "hello")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["result"], "olleh")

    def test_uppercase_converts_text(self) -> None:
        payload = server_strings.string_op("uppercase", "hello")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["result"], "HELLO")

    def test_lowercase_converts_text(self) -> None:
        payload = server_strings.string_op("lowercase", "HELLO")
        self.assertEqual(payload["result"], "hello")

    def test_length_returns_character_count(self) -> None:
        payload = server_strings.string_op("length", "hello")
        self.assertEqual(payload["result"], 5)

    def test_strip_removes_whitespace(self) -> None:
        payload = server_strings.string_op("strip", "  hello  ")
        self.assertEqual(payload["result"], "hello")

    def test_replace_substitutes_substring(self) -> None:
        payload = server_strings.string_op("replace", "foo bar foo", old="foo", new="baz")
        self.assertEqual(payload["result"], "baz bar baz")
        self.assertEqual(payload["replacements"], 2)

    def test_replace_with_count_limit(self) -> None:
        payload = server_strings.string_op("replace", "aaa", old="a", new="b", count=2)
        self.assertEqual(payload["result"], "bba")

    def test_split_by_separator(self) -> None:
        payload = server_strings.string_op("split", "a,b,c", sep=",")
        self.assertEqual(payload["result"], ["a", "b", "c"])
        self.assertEqual(payload["count"], 3)

    def test_contains_found(self) -> None:
        payload = server_strings.string_op("contains", "hello world", substring="world")
        self.assertTrue(payload["result"])
        self.assertIn(6, payload["positions"])

    def test_contains_not_found(self) -> None:
        payload = server_strings.string_op("contains", "hello", substring="xyz")
        self.assertFalse(payload["result"])
        self.assertEqual(payload["positions"], [])

    def test_contains_case_insensitive(self) -> None:
        payload = server_strings.string_op(
            "contains", "Hello World", substring="world", case_sensitive=False)
        self.assertTrue(payload["result"])

    def test_count_occurrences_returns_count(self) -> None:
        payload = server_strings.string_op("count_occurrences", "banana", substring="an")
        self.assertEqual(payload["result"], 2)

    def test_starts_with_true(self) -> None:
        payload = server_strings.string_op("starts_with", "hello", prefix="hel")
        self.assertTrue(payload["result"])

    def test_starts_with_false(self) -> None:
        payload = server_strings.string_op("starts_with", "hello", prefix="world")
        self.assertFalse(payload["result"])

    def test_ends_with_true(self) -> None:
        payload = server_strings.string_op("ends_with", "hello", suffix="llo")
        self.assertTrue(payload["result"])

    def test_title_case_converts_text(self) -> None:
        payload = server_strings.string_op("title_case", "hello world")
        self.assertEqual(payload["result"], "Hello World")

    def test_unknown_op_is_structured_error(self) -> None:
        payload = server_strings.string_op("nope", "hello")
        self.assertEqual(payload["status"], "error")


class DatetimeServerTests(unittest.TestCase):
    """All date ops are dispatched through the single `date_op(op, ...)` tool."""

    def test_current_datetime_utc_has_required_keys(self) -> None:
        payload = server_datetime.date_op("current_datetime", tz="UTC")
        self.assertEqual(payload["status"], "ok")
        for key in ("datetime", "date", "time", "weekday", "tz"):
            self.assertIn(key, payload)
        self.assertEqual(payload["tz"], "UTC")

    def test_current_datetime_invalid_tz_is_structured_error(self) -> None:
        payload = server_datetime.date_op("current_datetime", tz="Not/ATimezone")
        self.assertEqual(payload["status"], "error")

    def test_days_between_known_dates(self) -> None:
        payload = server_datetime.date_op("days_between", date1="2026-01-01", date2="2026-01-11")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["days"], 10)
        self.assertEqual(payload["direction"], "future")

    def test_days_between_same_date(self) -> None:
        payload = server_datetime.date_op("days_between", date1="2026-06-15", date2="2026-06-15")
        self.assertEqual(payload["days"], 0)
        self.assertEqual(payload["direction"], "same")

    def test_days_between_invalid_format_is_structured_error(self) -> None:
        payload = server_datetime.date_op("days_between", date1="not-a-date", date2="2026-01-01")
        self.assertEqual(payload["status"], "error")

    def test_add_days_forward(self) -> None:
        payload = server_datetime.date_op("add_days", date_str="2026-01-01", n=10)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["result"], "2026-01-11")

    def test_add_days_backwards(self) -> None:
        payload = server_datetime.date_op("add_days", date_str="2026-01-11", n=-10)
        self.assertEqual(payload["result"], "2026-01-01")

    def test_day_of_week_known_date(self) -> None:
        payload = server_datetime.date_op("day_of_week", date_str="2026-03-23")
        self.assertEqual(payload["weekday"], "Monday")
        self.assertFalse(payload["is_weekend"])

    def test_day_of_week_weekend(self) -> None:
        payload = server_datetime.date_op("day_of_week", date_str="2026-03-21")
        self.assertTrue(payload["is_weekend"])

    def test_unix_to_date_epoch(self) -> None:
        payload = server_datetime.date_op("unix_to_date", timestamp=0, tz="UTC")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["date"], "1970-01-01")

    def test_format_date_reformats_correctly(self) -> None:
        payload = server_datetime.date_op(
            "format_date", date_str="14/07/1990", input_fmt="%d/%m/%Y", output_fmt="%Y-%m-%d")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["result"], "1990-07-14")

    def test_format_date_invalid_format_is_structured_error(self) -> None:
        payload = server_datetime.date_op(
            "format_date", date_str="not-a-date", input_fmt="%d/%m/%Y", output_fmt="%Y-%m-%d")
        self.assertEqual(payload["status"], "error")

    def test_unknown_op_is_structured_error(self) -> None:
        payload = server_datetime.date_op("nope")
        self.assertEqual(payload["status"], "error")


@unittest.skipUnless(_HAS_SYMPY, "sympy not installed")
class SymbolicMathServerTests(unittest.TestCase):
    """All SymPy ops are dispatched through the single `symbolic(op, ...)` tool."""

    def test_simplify(self) -> None:
        payload = server_symbolic_math.symbolic("simplify", expression="sin(x)**2 + cos(x)**2")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["result"], "1")

    def test_expand(self) -> None:
        payload = server_symbolic_math.symbolic("expand", expression="(x + 1)**2")
        self.assertEqual(payload["status"], "ok")
        self.assertIn("x**2", payload["result"])

    def test_factor(self) -> None:
        payload = server_symbolic_math.symbolic("factor", expression="x**2 - 1")
        self.assertEqual(payload["status"], "ok")

    def test_differentiate(self) -> None:
        payload = server_symbolic_math.symbolic("differentiate", expression="x**2", variable="x")
        self.assertEqual(payload["result"], "2*x")

    def test_integrate(self) -> None:
        payload = server_symbolic_math.symbolic("integrate", expression="2*x", variable="x")
        self.assertEqual(payload["result"], "x**2")

    def test_solve_equation(self) -> None:
        payload = server_symbolic_math.symbolic("solve_equation", equation="x**2 - 4", variable="x")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(sorted(payload["result"]), ["-2", "2"])

    def test_solve_equation_with_equals_sign(self) -> None:
        payload = server_symbolic_math.symbolic("solve_equation", equation="x**2 = 4", variable="x")
        self.assertEqual(sorted(payload["result"]), ["-2", "2"])

    def test_compute_limit_infinity(self) -> None:
        payload = server_symbolic_math.symbolic(
            "compute_limit", expression="1/x", variable="x", point="inf")
        self.assertEqual(payload["result"], "0")

    def test_series_expansion(self) -> None:
        payload = server_symbolic_math.symbolic(
            "series_expansion", expression="exp(x)", variable="x", point="0", n=4)
        self.assertEqual(payload["status"], "ok")

    def test_matrix_determinant(self) -> None:
        payload = server_symbolic_math.symbolic("matrix_determinant", matrix_str="[[1, 2], [3, 4]]")
        self.assertEqual(payload["result"], "-2")

    def test_create_matrix(self) -> None:
        payload = server_symbolic_math.symbolic("create_matrix", matrix_str="[[1, 2], [3, 4]]")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["size"], "2x2")

    def test_solve_system(self) -> None:
        payload = server_symbolic_math.symbolic(
            "solve_system", equations=["x + y - 1", "x - y - 3"], variables=["x", "y"])
        self.assertEqual(payload["status"], "ok")

    def test_unknown_op_is_structured_error(self) -> None:
        payload = server_symbolic_math.symbolic("nope")
        self.assertEqual(payload["status"], "error")


class BashServerTests(unittest.TestCase):
    def test_an_unlisted_command_runs(self) -> None:
        """The default answer is "run it", not "not allowed".

        A refusal by name is a dead end: nothing in the approval layer can grant a
        *command*, only a path, so the agent could only retry spellings of it. The
        gate is now the denylist plus the user's prompt.
        """
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("git status", "awk '{print $1}' notes.txt", "tar -tf a.tar",
                    "rm -rf build", "clang --version", "mpicc -o a.out a.c",
                    "curl -s https://example.com"):
            self.assertEqual(
                server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)

    def test_bash_run_chaining_is_validated_per_command(self) -> None:
        # Chaining is allowed, but each command is validated on its own, so a
        # denied command anywhere in the chain rejects the call.
        payload = server_bash.bash_run("ls && sh script.sh")
        self.assertEqual(payload["status"], "error")
        self.assertIn("not available here", payload["error"])

    def test_bash_multiline_command_is_validated_per_line(self) -> None:
        # An unquoted newline chains like ';': each line must be validated on its
        # own, so a disallowed command on a later line is rejected instead of
        # being read as an extra argument of the line above.
        cwd = server_bash._WORKSPACE_ROOT
        payload = server_bash._validate_command("ls\nsh script.sh", cwd)
        self.assertEqual(payload["status"], "error")
        self.assertIn("not available here", payload["error"])
        # ... and the confinement of every line still applies.
        payload = server_bash._validate_command("ls\ncat /etc/passwd", cwd)
        self.assertEqual(payload["status"], "error")
        self.assertIn("outside workspace", payload["error"])

    def test_find_exec_allows_a_read_only_nested_command(self) -> None:
        # `-exec` used to be denied on the token alone, without ever inspecting what
        # was nested — and the rejection hint pointed at `xargs`, which is itself
        # permanently banned, so read-only fan-out had no spelling at all.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in (
            r'find . -name "*.py" -exec grep -l pattern {} \;',
            "find . -type f -exec wc -l {} +",
            r"find . -name x -exec grep q {} \; -print",  # terminator, then more find args
        ):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "ok", f"{cmd} -> {payload}")

    def test_find_exec_still_refuses_anything_not_read_only(self) -> None:
        # The grant must not widen: a nested command's operands include `{}`, whose
        # expansion cannot be resolved here, so a nested write/exec targets unknown
        # paths. `python f.py` was already directly invocable, so nothing that was
        # previously unreachable becomes reachable.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in (
            r"find . -exec rm {} \;",
            r"find . -exec python evil.py {} \;",
            r"find . -exec chmod 777 {} \;",
            r"find . -exec mv {} /tmp \;",
        ):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", f"{cmd} -> {payload}")

    def test_find_hidden_target_flags_are_still_denied(self) -> None:
        # These hide their target in flag-value position (or prompt on a tty the
        # agent does not have); they were never about nesting and stay denied.
        # ('-delete' is not among them: it deletes the matches, whose paths the
        # confinement check does see, so it runs under the approval prompt.)
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in (r"find . -ok rm {} \;",
                    "find . -fprint /tmp/out", "find . -fls /tmp/out"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", f"{cmd} -> {payload}")

    def test_nested_command_operands_are_still_confined(self) -> None:
        # The nested segment goes through the same path validation as any other, so
        # confinement is inherited rather than re-implemented.
        cwd = server_bash._WORKSPACE_ROOT
        payload = server_bash._validate_command(r"find /etc -exec cat {} \;", cwd)
        self.assertEqual(payload["status"], "error")
        self.assertIn("outside workspace", payload["error"])

    def test_unterminated_exec_is_rejected(self) -> None:
        cwd = server_bash._WORKSPACE_ROOT
        payload = server_bash._validate_command("find . -exec grep q {}", cwd)
        self.assertEqual(payload["status"], "error")

    def test_bash_run_allows_multiline_command(self) -> None:
        payload = server_bash.bash_run("pwd\necho multiline-ok")
        self.assertEqual(payload["status"], "ok", payload)
        self.assertIn("multiline-ok", payload["stdout"])
        # A newline after a connector continues the chain, and a newline inside
        # quotes is data — a multi-line `-c` body runs as written.
        payload = server_bash.bash_run('pwd &&\npython3 -c "\nprint(\'body-ok\')\n"')
        self.assertEqual(payload["status"], "ok", payload)
        self.assertIn("body-ok", payload["stdout"])

    def test_bash_write_flag_targets_are_confined_in_both_spellings(self) -> None:
        """A file a command *creates* is confined however the flag is spelled.

        The separated form always was (it parses as a positional); the glued one was
        not, so `gcc a.c -o/tmp/x.out` wrote outside the workspace with no prompt while
        `gcc a.c -o /tmp/x.out` was refused. Both must be refused.
        """
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("gcc a.c -o /tmp/x.out", "gcc a.c -o/tmp/x.out",
                    "ruff check --output-file=/tmp/r.txt f.py",
                    "pytest --junitxml=/tmp/j.xml",
                    "sort --output=/tmp/x in.txt", "cmake -B /tmp/build",
                    "javac -d /tmp X.java", "mypy --cache-dir=/tmp/c f.py",
                    "pdflatex -output-directory=/tmp main.tex"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", cmd)
            self.assertIn("outside workspace", payload["error"], cmd)
        # In-workspace targets pass, and a `cd` rebases them like any other path.
        for cmd in ("gcc a.c -o out.o", "gcc a.c -oout.o", "sort -o out.txt in.txt",
                    "pytest --junitxml=j.xml", "cd mimir && cmake -B build"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)

    def test_bash_read_flags_stay_unconfined(self) -> None:
        # Reading system headers/libraries is how any real build works, so read flags
        # are deliberately out of scope — only writes are confined.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("gcc -I/usr/include -L/usr/lib a.c -lm -o a.out",
                    "gcc -Wl,-rpath,/usr/lib a.c -o a.out",
                    "nvcc -L/usr/local/cuda/lib64 k.cu -o k.out"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)

    def test_bash_env_managers_only_lose_the_sub_command_that_nests(self) -> None:
        """pip/conda run their whole surface; 'conda run' is the one exception.

        Install and uninstall alike are reviewable from the command line, which is
        what the approval prompt is for. 'conda run' is not: it executes a nested
        command this validator never sees, exactly like a shell interpreter.
        """
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("pip install requests", "pip3 install -r requirements.txt",
                    "pip list", "conda create -n myenv python=3.11",
                    "conda env create -f env.yml", "mamba install -y numpy",
                    "conda list", "pip uninstall requests", "conda remove numpy",
                    "conda env remove -n x", "conda clean --all",
                    "pip config set global.index x", "pip wheel ."):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)
        for cmd in ("conda run -n x python a.py", "mamba run -n x ls"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", cmd)
            self.assertIn("nested command", payload["hint"], cmd)
        self.assertIn("sub-command", server_bash._validate_command("pip", cwd)["error"])

    def test_bash_env_manager_file_operands_are_confined(self) -> None:
        # A requirements file / local wheel is a file operand like any other: it can
        # be read from the workspace, not from outside it without approval.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("pip install /tmp/pkg.whl", "pip install -r /etc/req.txt",
                    "conda env create -f /etc/env.yml"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", cmd)
            self.assertIn("outside workspace", payload["error"], cmd)
        # A package name, a version pin and a VCS URL are not paths.
        for cmd in ("pip install requests", "pip install 'numpy==1.26.0'",
                    "pip install git+https://github.com/a/b"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)

    def test_bash_sed_out_of_workspace_path_is_rejected(self) -> None:
        # `sed` reads and (with -i) writes files, so its paths must be confined.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("sed -n '1,5p' /etc/passwd", "sed -i 's/a/b/' /etc/hosts"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", cmd)
            self.assertIn("outside workspace", payload["error"], cmd)

    def test_bash_wc_out_of_workspace_path_is_rejected(self) -> None:
        cwd = server_bash._WORKSPACE_ROOT
        payload = server_bash._validate_command("wc -l /etc/passwd", cwd)
        self.assertEqual(payload["status"], "error")
        self.assertIn("outside workspace", payload["error"])

    def test_bash_sed_inplace_in_workspace_is_permitted_by_validation(self) -> None:
        # `sed -i` on a workspace-relative file is a legitimate (approval-gated,
        # confined) edit — it must pass command validation, not be denied.
        cwd = server_bash._WORKSPACE_ROOT
        payload = server_bash._validate_command("sed -i 's/a/b/' notes.txt", cwd)
        self.assertEqual(payload["status"], "ok", payload)

    def test_bash_tr_character_set_is_not_treated_as_a_path(self) -> None:
        # `tr` reads stdin; its args are character SETs, so a '/' set must not be
        # rejected as an out-of-workspace path.
        cwd = server_bash._WORKSPACE_ROOT
        payload = server_bash._validate_command("tr '/' '_'", cwd)
        self.assertEqual(payload["status"], "ok", payload)

    def test_bash_run_allows_workspace_local_compiled_binary(self) -> None:
        # The code-server fallback can compile with custom flags and run the
        # resulting binary, referenced by a workspace-relative path.
        import shutil
        if shutil.which("gcc") is None:
            self.skipTest("gcc not installed in this environment")
        payload = server_bash.bash_run(
            'gcc bin_src.c -O2 -o bin_out.out && ./bin_out.out')
        self.assertEqual(payload["status"], "ok", payload)
        self.assertIn("WS-BIN-OK", payload["stdout"])

    @classmethod
    def setUpClass(cls) -> None:
        import os
        src = os.path.join(server_bash._WORKSPACE_ROOT, "bin_src.c")
        with open(src, "w") as fh:
            fh.write('#include <stdio.h>\nint main(){printf("WS-BIN-OK\\n");return 0;}')
        cls._wlocal_src = src

    @classmethod
    def tearDownClass(cls) -> None:
        import os
        for name in ("bin_src.c", "bin_out.out"):
            p = os.path.join(server_bash._WORKSPACE_ROOT, name)
            if os.path.exists(p):
                os.remove(p)

    def test_bash_run_rejects_arbitrary_system_executable_by_path(self) -> None:
        # A path-like argv0 that is NOT inside the workspace stays rejected. It is
        # refused as an unapproved path rather than an unknown command — the user can
        # grant that specific root, which is the point — but nothing runs without it.
        payload = server_bash.bash_run("/bin/ls")
        self.assertEqual(payload["status"], "error")
        self.assertIn("not approved", payload["error"])

    def test_bash_run_rejects_backgrounding(self) -> None:
        # A single '&' (background) is still blocked, unlike '&&' chaining.
        payload = server_bash.bash_run("ls & pwd")
        self.assertEqual(payload["status"], "error")
        self.assertIn("not allowed", payload["error"])

    def test_bash_run_rejects_command_substitution(self) -> None:
        for cmd in ("ls $(echo x)", "cat `whoami`"):
            payload = server_bash.bash_run(cmd)
            self.assertEqual(payload["status"], "error", cmd)

    def test_bash_run_allows_glob_and_simple_pipe(self) -> None:
        # Globbing and a single pipe must still work.
        payload = server_bash.bash_run(
            "cd mimir/servers/workspace && ls *.py | head")
        self.assertEqual(payload["status"], "ok")
        self.assertIn("server_bash.py", payload["stdout"])

    def test_bash_run_confines_path_sensitive_command(self) -> None:
        # rg reads files, so an out-of-workspace path must be rejected.
        payload = server_bash.bash_run("rg root /etc/passwd")
        self.assertEqual(payload["status"], "error")
        self.assertIn("outside workspace", payload["error"])

    def test_bash_run_rejects_a_denied_command(self) -> None:
        payload = server_bash.bash_run("shred -u notes.txt")
        self.assertEqual(payload["status"], "error")
        self.assertIn("not available here", payload["error"])
        self.assertEqual(payload["denial"], "destructive")

    def test_deletion_is_allowed_but_still_confined(self) -> None:
        # 'rm' is destructive and reviewable, which is exactly what the approval
        # prompt is for; what it may not do is reach outside the workspace.
        cwd = server_bash._WORKSPACE_ROOT
        self.assertEqual(
            server_bash._validate_command("rm -rf build", cwd)["status"], "ok")
        payload = server_bash.bash_run("rm -rf /tmp/something")
        self.assertEqual(payload["status"], "error")
        self.assertIn("outside workspace", payload["error"])

    def test_bash_run_confines_file_redirection(self) -> None:
        # Redirecting to a file is allowed, but the target is a path like any
        # other: outside the workspace it needs the user's approval.
        payload = server_bash.bash_run("ls > /tmp/out.txt")
        self.assertEqual(payload["status"], "error")
        self.assertIn("outside workspace", payload["error"])
        # Same for the input side, and for a target reached with '..'.
        for cmd in ("cat < /etc/passwd", "ls >> ../escape.log"):
            payload = server_bash._validate_command(cmd, server_bash._WORKSPACE_ROOT)
            self.assertEqual(payload["status"], "error", cmd)
            self.assertIn("outside workspace", payload["error"], cmd)
        # An expanded target cannot be checked (the shell that expands it is not
        # the one validating), so it is refused rather than guessed at.
        payload = server_bash._validate_command("ls > $HOME/x.log",
                                                server_bash._WORKSPACE_ROOT)
        self.assertEqual(payload["status"], "error")
        self.assertIn("expansion", payload["error"])

    def test_bash_run_allows_file_redirection_in_workspace(self) -> None:
        # The ordinary shell idiom for keeping a run's output: write it, read it back.
        import os
        log = os.path.join(server_bash._WORKSPACE_ROOT, "redir_test.log")
        try:
            payload = server_bash.bash_run("echo redirected > redir_test.log")
            self.assertEqual(payload["status"], "ok", payload)
            payload = server_bash.bash_run("cat redir_test.log")
            self.assertIn("redirected", payload["stdout"])
            # Append, a second stream, and a redirect after a 'cd' rebase.
            payload = server_bash.bash_run("echo more >> redir_test.log 2>&1")
            self.assertEqual(payload["status"], "ok", payload)
        finally:
            if os.path.exists(log):
                os.remove(log)

    def test_bash_run_rejects_heredoc(self) -> None:
        # A heredoc body is not an argv the validator can segment; its lines would
        # each be read as a command.
        payload = server_bash.bash_run("cat > f.txt <<EOF\nhello\nEOF")
        self.assertEqual(payload["status"], "error")

    def test_bash_run_allows_fd_redirection(self) -> None:
        # fd redirection (silencing/merging streams) is now permitted; the '2'
        # source fd must not leak into argv and bash must accept the form.
        for cmd in ("pwd 2>/dev/null", "pwd 2>&1", "pwd 1>&2", "pwd &>/dev/null"):
            payload = server_bash.bash_run(cmd)
            self.assertEqual(payload["status"], "ok", cmd)

    def test_bash_run_accepts_every_module_subcommand(self) -> None:
        # Load, unload, swap and purge all change this one subprocess's environment
        # and nothing else; the approval prompt covers that. What is still checked
        # is the shape of what reaches Lmod, which evaluates modulefiles.
        for cmd in ("module avail", "module list", "module load cuda",
                    "module unload cuda", "module purge",
                    "module load gcc/11.3.0 && gcc --version"):
            self.assertIsNone(
                server_bash._validate_module_args(cmd.split("&&")[0].split()), cmd)

    def test_bash_run_rejects_unsafe_module_argument(self) -> None:
        payload = server_bash.bash_run("module load ../../etc/passwd")
        self.assertEqual(payload["status"], "error")

    def test_every_denied_command_names_the_route_that_replaces_it(self) -> None:
        # A refusal that names no route is one the agent retries variants of. The
        # reply to a *shell* call must not name something to "call" either — that
        # reads as another shell command to try.
        cwd = server_bash._WORKSPACE_ROOT
        for name in sorted(_shell_paths.DENIED_COMMANDS):
            payload = server_bash._validate_command(f"{name} x", cwd)
            self.assertEqual(payload["status"], "error", name)
            self.assertIn(name, payload["error"], name)
            self.assertTrue(payload["hint"].strip(), name)
            self.assertNotIn("()", payload["hint"], name)

    def test_shell_interpreters_are_rejected_with_a_terminal_explanation(self) -> None:
        # 'bash'/'sh'/'eval' would nest a command this validator never sees. The
        # hint must say so, so the agent stops hunting for a wrapper.
        for cmd in ("bash -c 'ls'", "sh script.sh", "eval ls", "sudo ls"):
            payload = server_bash.bash_run(cmd)
            self.assertEqual(payload["status"], "error", cmd)
            self.assertIn("never sees", payload["hint"], cmd)

    def test_a_wrapper_that_nests_nothing_unvalidated_runs(self) -> None:
        # 'timeout'/'env'/'xargs' were refused as runners; they are unwrapped now,
        # so what is validated is the command they carry.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("timeout 5 ls", "env ls", "xargs ls", "nohup ls",
                    "env A=B python3 x.py", "timeout 60 pytest -q"):
            self.assertEqual(
                server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)

    def test_noop_commands_enable_the_capability_probe(self) -> None:
        # `which X || true` is how availability is established; without the
        # no-ops the whole chain is rejected on its last segment.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("true", "false", ":", "which pdflatex || true",
                    "which pdflatex 2>/dev/null || true"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "ok", cmd)

    def test_no_match_is_a_result_not_an_error(self) -> None:
        # grep exit 1 == "no match", which exit 1 == "not installed". Both are
        # conclusive answers; reporting them as failures makes the agent re-run
        # the same command instead of acting on the finding.
        for cmd in ("grep -r zzz-no-such-token-zzz mimir/servers/workspace",
                    "which zzz-no-such-binary-zzz",
                    "ls mimir | grep zzz-no-such-token-zzz"):
            payload = server_bash.bash_run(cmd)
            self.assertEqual(payload["status"], "ok", cmd)
            self.assertEqual(payload["returncode"], 1, cmd)
            self.assertEqual(payload["matches"], 0, cmd)

    def test_real_grep_failure_stays_an_error(self) -> None:
        # Exit 2 (unreadable file / bad usage) is a genuine failure, unlike exit 1.
        payload = server_bash.bash_run("grep foo no_such_file_here.txt")
        self.assertEqual(payload["status"], "error")

    def test_no_match_rule_only_applies_to_the_last_segment(self) -> None:
        # The chain's exit status comes from its last command; a non-matching
        # grep upstream must not relabel a downstream failure as "no match".
        payload = server_bash.bash_run("which zzz-nope-zzz || cat no_such_file.txt")
        self.assertEqual(payload["status"], "error")

    def test_file_management_is_allowed_and_confined(self) -> None:
        # mv/cp/mkdir are available (approval-gated), but both ends of a move or
        # copy must stay inside the workspace — otherwise they would be an
        # exfiltration channel out or an import channel in.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("mv old.py new.py", "cp -r src dst", "mkdir -p build/obj",
                    "mkdir build && cp main.c build/"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)
        for cmd in ("cp /etc/passwd .", "mv secrets.txt /tmp/out",
                    "cp ../../outside.txt here.txt"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", cmd)
            self.assertIn("outside workspace", payload["error"], cmd)

    def test_cd_outside_the_workspace_needs_approval(self) -> None:
        # Not forbidden — ungranted. The refusal must name approval as the way
        # forward, since that is an action the agent can actually take.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("cd /etc", "cd ..", "cd ../..", "cd /tmp && ls"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", cmd)
            self.assertIn("not approved", payload["error"], cmd)
            self.assertIn("approval", payload["hint"], cmd)

    def test_cd_outside_the_workspace_runs_once_approved(self) -> None:
        # A user grant reaches the server through the shared sidecar, so the same
        # cd that was refused above now validates — and rebases the paths after it.
        import unittest.mock as mock
        cwd = server_bash._WORKSPACE_ROOT
        with mock.patch.object(server_bash, "_is_within_workspace",
                               side_effect=lambda p: p == "/tmp" or p.startswith("/tmp/")
                               or p == cwd or p.startswith(cwd + "/")):
            for cmd in ("cd /tmp && ls", "cd /tmp && cat data.csv"):
                self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)
            # A grant on one directory does not open a different one.
            self.assertEqual(server_bash._validate_command("cd /etc", cwd)["status"], "error")

    def test_cd_rebases_the_confinement_of_later_segments(self) -> None:
        # Regression: path confinement is validated against the directory the
        # shell will actually be in. Checking every segment against the *initial*
        # cwd made each half of `cd /etc && cat passwd` look fine on its own while
        # the chain read outside the sandbox.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("cd /etc && cat passwd",
                    "cd mimir && cd ../../etc && cat passwd",
                    "cd .. && cp outside.txt here.txt"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", cmd)
        # ...and a legitimate in-workspace hop still resolves through the new base.
        # A bare 'cd' is NOT one of these any more: HOME is inherited, so it lands in
        # the user's home and is refused like any outside destination — see
        # test_home_is_reachable_only_with_approval. 'cd -' stays in, since OLDPWD can
        # only be a directory this chain already entered, and so already passed here.
        for cmd in ("cd mimir && ls", "cd mimir && cd .. && ls",
                    "cd mimir/servers && cat workspace/server_bash.py",
                    "cd -"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)

    def test_cd_then_relative_path_still_runs(self) -> None:
        payload = server_bash.bash_run("cd mimir/servers/workspace && ls server_bash.py")
        self.assertEqual(payload["status"], "ok", payload)
        self.assertIn("server_bash.py", payload["stdout"])

    def test_deletion_runs_but_only_inside_the_workspace(self) -> None:
        # 'rm' is destructive and reviewable, which is what the approval prompt is
        # for. What it may not do is reach outside the workspace.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("rm f.txt", "rm -rf build", "rmdir build"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)
        for cmd in ("rm /etc/passwd", "rm -rf ../../elsewhere"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", cmd)
            self.assertIn("outside workspace", payload["error"], cmd)

    def test_exec_commands_confine_their_file_operands(self) -> None:
        # Regression: confining the readers (`cat`) while leaving the executors
        # free was backwards — an interpreter or compiler takes a file operand the
        # same way, so `python /tmp/evil.py` reached outside the sandbox.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("python /tmp/evil.py", "python ../../evil.py",
                    "gcc /tmp/x.c -o /tmp/x.out", "pytest /etc/",
                    "mypy /etc/x.py", "pdflatex /tmp/a.tex",
                    "./solver.out /etc/passwd"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", cmd)
            self.assertIn("outside workspace", payload["error"], cmd)

    def test_exec_confinement_does_not_reject_ordinary_invocations(self) -> None:
        # Flags are skipped by _normalize_path_arg, so the toolchain's usual
        # spellings — include dirs, link flags, -m modules, sub-commands — pass.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("python -m pytest", "python -m py_compile foo.py",
                    "gcc a.c -O3 -march=native -o a.out -lm",
                    "gcc -I/usr/include a.c -o a.out", "pytest -q tests/",
                    "make -j4", "make all", "cmake -S . -B build",
                    'python -c "print(1)"', "java -cp . Main",
                    "ruff check foo.py", "./solver.out 1000 0.5",
                    "python script.py --flag value"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)

    def test_every_command_that_opens_a_file_is_path_confined(self) -> None:
        # Confinement is the default, exceptions listed — which is what keeps a
        # command nobody classified from escaping it. The exemptions are the words
        # that open nothing: `tr`'s arguments are character sets, and the metadata
        # no-ops (`echo`, `which`, `pwd`, `df`) take words, not paths.
        from mimir.servers._shared.shell_paths import (
            PATH_INSENSITIVE_COMMANDS, takes_path_operands,
        )
        self.assertEqual(
            PATH_INSENSITIVE_COMMANDS,
            {"tr", "echo", "which", "pwd", "df", "basename", "dirname",
             "true", "false", ":", "printenv", "export"},
            "a command stopped being path-confined")
        for command in ("cat", "gcc", "python3", "cd", "module", "rm", "git",
                        "a-command-nobody-classified", "./a.out"):
            self.assertTrue(takes_path_operands(command), command)

    def test_a_workspace_script_can_be_made_executable_and_run(self) -> None:
        # Running a workspace script by path is already supported, but a script
        # without the x bit (fresh checkout, or one the agent just wrote) used to be
        # a dead end: './build.sh' failed with "Permission denied", 'chmod' was
        # refused and 'bash build.sh' is refused by design, so nothing could grant it.
        # Both halves must stay available for the sequence to work.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("chmod +x ./build.sh", "chmod 755 tools/run.sh", "./build.sh"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)

    def test_chmod_stays_confined(self) -> None:
        # chmod is a write like any other: its operand is confined to the workspace.
        # '-R' is no longer refused — it re-modes a tree whose root is confined, and
        # deletion, which is worse, runs under the prompt.
        cwd = server_bash._WORKSPACE_ROOT
        self.assertEqual(
            server_bash._validate_command("chmod +x /etc/passwd", cwd)["status"], "error")
        for cmd in ("chmod -R 777 .", "chmod --recursive 700 src"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)

    def test_single_quoted_substitution_markers_are_literal(self) -> None:
        # Single quotes make these characters text, and a search pattern is where
        # they legitimately appear. The check used to run on tokens *after* shlex had
        # stripped the quotes, so a backtick bash treats as data was indistinguishable
        # from one that executes, and extracting a fenced block was impossible.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("sed -n '/```mermaid/,/```/p' ARCHITECTURE.md",
                    "grep -n '```mermaid' ARCHITECTURE.md",
                    "grep -n '${HOME}' notes.txt",
                    "grep -rn '$(x)' src"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)

    def test_substitution_the_shell_would_run_is_still_refused(self) -> None:
        # Double quotes do NOT make them literal, so those still count — as does
        # anything unquoted. This is the half that must not regress.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("ls $(echo x)", "cat `whoami`", 'grep -n "`cmd`" f.md',
                    'echo "$(id)"', "cat <(ls)", "ls ${HOME}"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "error", cmd)

    def test_which_reports_absence_as_a_conclusive_answer(self) -> None:
        # `which` returns how many names it did not find, so the standard multi-name
        # capability probe exits 4, not 1 — and was reported as a failure telling the
        # agent to re-read stderr for a problem that does not exist.
        payload = server_bash.bash_run("which nolatex nopdflatex noxelatex nolualatex")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["returncode"], 4)
        self.assertEqual(payload["matches"], 0)
        # grep keeps its stricter rule: 1 is no-match, 2 is a real error.
        self.assertEqual(server_bash.bash_run("grep -n zzzznomatch README.md")["status"], "ok")

    def test_outside_executable_is_approvable_not_flatly_refused(self) -> None:
        # An out-of-workspace *file* was refused pending the user's approval while an
        # out-of-workspace *executable* was refused as an unknown command — with a
        # hint telling the agent to stop trying. Same path, same treatment: the
        # program a call runs is judged by confinement, so the user can grant it.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("../outside/x.sh", "/opt/tools/run.sh"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", cmd)
            self.assertIn("not approved", payload["error"], cmd)
            self.assertIn("approval", payload["hint"], cmd)
        # ...while a workspace-local one still just runs.
        for cmd in ("./build.sh", "./a.out data.txt"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)

    def test_shell_interpreter_is_refused_by_path_too(self) -> None:
        # The escape hatch the rule above must not open: an interpreter spelled as a
        # path would otherwise become approvable, and one approved root would restore
        # the arbitrary-nested-command bypass that refusing 'bash' exists to prevent.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("/bin/sh -c 'rm -rf /'", "../outside/bash script.sh", "./sh -c x"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", cmd)
            self.assertIn("not available here", payload["error"], cmd)

    def test_program_path_is_an_operand_for_both_sides(self) -> None:
        # The client's gate and the server's guard walk the same extractor, so the
        # program path must appear there or only one side would see it — which is
        # exactly how it ended up unprompted by the client and unapprovable.
        import os
        from mimir.servers._shared.shell_paths import segment_path_operands
        cwd = server_bash._WORKSPACE_ROOT
        self.assertIn("/opt/tools/run.sh",
                      segment_path_operands(["/opt/tools/run.sh"], cwd))
        self.assertIn(os.path.join(cwd, "a.out"),
                      segment_path_operands(["./a.out", "data.txt"], cwd))
        # A command named plainly is not a path and must not become one.
        self.assertEqual(segment_path_operands(["ls"], cwd), [])

    def test_home_is_inherited_not_pinned_to_the_workspace(self) -> None:
        # Pinning HOME to the workspace detached every tool from the user's own
        # configuration, one silent failure per tool: user site-packages
        # (~/.local/lib) vanished from `python`, pip lost its config and cached into
        # the repo, conda lost ~/.condarc, and TeX stopped finding ~/texmf packages —
        # so a command that works in the user's shell failed under MIMIR. HOME is
        # inherited; what confines a command is the path check on its operands.
        import os
        env = server_bash._safe_env(server_bash._WORKSPACE_ROOT)
        self.assertEqual(env["HOME"], os.path.realpath(os.path.expanduser("~")))
        self.assertNotEqual(env["HOME"], server_bash._WORKSPACE_ROOT)
        # Startup files are what --noprofile/--norc and BASH_ENV are for, and that
        # must not regress now that HOME points at a directory holding real dotfiles.
        self.assertEqual(env["BASH_ENV"], "")

    def test_home_is_reachable_only_with_approval(self) -> None:
        # HOME being inherited must not become a way out of the workspace: a bare
        # 'cd' and 'cd ~' now really do land in the user's home, so they are judged
        # like any other outside destination rather than modelled as a no-op.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("cd", "cd ~", "cd ~ && cat .netrc", "cat ~/.netrc"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "error", cmd)

    def test_there_is_no_working_directory_argument(self) -> None:
        # One way to say "where", not two: every call starts at the workspace root
        # and `cd` moves within it. A second base was only ever a second thing to
        # keep confined.
        import inspect
        params = inspect.signature(server_bash.bash_run).parameters
        self.assertEqual(list(params), ["command", "timeout"])
        self.assertFalse(hasattr(server_bash, "_safe_cwd"))

    def test_search_pattern_is_not_treated_as_a_path(self) -> None:
        # `grep /etc/passwd notes.txt` searches for a *string*; nothing opens
        # /etc/passwd. Treating the pattern as a path refused a legitimate search
        # and (once the client gate existed) asked the user about a file that would
        # never be read. Same for a `sed` script.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("grep /etc/passwd notes.txt", "grep -rn /etc/passwd .",
                    "rg /etc/shadow src", 'sed "s|/etc/passwd|x|" f.txt',
                    "grep -e /etc/passwd notes.txt"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)

    def test_search_file_operands_are_still_confined(self) -> None:
        # The pattern is skipped, not the files — including the ones that arrive in
        # flag-value position, where the pattern-skipping rule stops looking.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("grep foo /etc/passwd", "rg foo /etc", "sed -n 1p /etc/passwd",
                    "sed -f /etc/s.sed f.txt", "grep -f /etc/patterns f.txt",
                    "rg -f /etc/patterns f.txt"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "error", cmd)

    def test_expanded_path_is_refused_not_guessed(self) -> None:
        # Regression: `module load cuda && cat $CUDA_HOME/version.txt` validated as a
        # harmless relative path (CUDA_HOME unset in the checking process) and then
        # read from the module tree at runtime. The command that runs must be the
        # command that was checked.
        import os
        cwd = server_bash._WORKSPACE_ROOT
        os.environ.pop("CUDA_HOME", None)
        for cmd in ("module load cuda && cat $CUDA_HOME/version.txt",
                    "cat $HOME/x", "cd $SOMEDIR && ls", "python $X/evil.py",
                    "cp $SRC/a.txt b.txt"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", cmd)
            self.assertIn("expansion", payload["error"], cmd)

    def test_expansion_outside_path_position_still_works(self) -> None:
        # The HPC idiom: a module-provided root used as an include/link flag. Nothing
        # is claimed about a flag's value, so there is no divergence to protect from.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("echo $HOME", "gcc -I$CUDA_HOME/include a.c -o a.out",
                    "module load cuda && nvcc -L$CUDA_HOME/lib64 k.cu -o k.out"):
            self.assertEqual(server_bash._validate_command(cmd, cwd)["status"], "ok", cmd)

    def test_no_classified_command_is_one_the_server_refuses(self) -> None:
        """A command cannot be both classified and denied.

        A command the classifier files under a kind but the server refuses is dead
        weight that reads as a supported capability: every consumer of the kind (plan
        mode, the approval waiver, the blackboard) treats a call that never runs as
        one that does. The taxonomy and the denylist must stay disjoint.
        """
        from mimir.client.guardrails import observations as obs
        from mimir.client.guardrails.policy.bash_classify import (
            READONLY_KINDS, classify_bash_command,
        )
        from mimir.servers._shared.shell_paths import (
            DENIED_COMMANDS, ENV_MANAGER_COMMANDS, EXEC_COMMANDS, INSPECT_COMMANDS,
            NEUTRAL_COMMANDS, READ_COMMANDS, SEARCH_COMMANDS,
            WRITE_COMMANDS, WRITE_VALUE_FLAGS_BY_CMD,
        )

        categories = {
            **{c: "read" for c in READ_COMMANDS},
            **{c: "search" for c in SEARCH_COMMANDS},
            **{c: "inspect" for c in INSPECT_COMMANDS},
            **{c: "write" for c in WRITE_COMMANDS},
            **{c: "exec" for c in EXEC_COMMANDS},
            **{c: "env" for c in ENV_MANAGER_COMMANDS},
            **{c: "neutral" for c in NEUTRAL_COMMANDS},
            "module": "env",
            "cd": "chdir",
        }
        groups = {
            "EXEC_COMMANDS": set(EXEC_COMMANDS),
            "ENV_MANAGER_COMMANDS": set(ENV_MANAGER_COMMANDS),
            "READ_COMMANDS": set(READ_COMMANDS),
            "SEARCH_COMMANDS": set(SEARCH_COMMANDS),
            "INSPECT_COMMANDS": set(INSPECT_COMMANDS),
            "WRITE_COMMANDS": set(WRITE_COMMANDS),
            "NEUTRAL_COMMANDS": set(NEUTRAL_COMMANDS),
            "WRITE_VALUE_FLAGS_BY_CMD": set(WRITE_VALUE_FLAGS_BY_CMD),
        }
        for name, group in groups.items():
            self.assertEqual(sorted(group & set(DENIED_COMMANDS)), [],
                             f"shell_paths.{name} names commands the bash server "
                             f"refuses to run")
        # The category a command is filed under must agree with how a call to it is
        # actually gated: the side-effect-free categories are exactly the plan-safe ones.
        probe = {"cd": "cd sub", "module": "module avail", "grep": "grep p f.py",
                 "rg": "rg p", "sed": "sed -n 1p f.py", "tr": "tr a b",
                 "mv": "mv a b", "cp": "cp a b", "find": "find . -name x"}
        for command, category in sorted(categories.items()):
            if category == "env":
                continue  # the sub-command decides; covered by its own tests
            cmd = probe.get(command, f"{command} x")
            segments = classify_bash_command(cmd)
            self.assertIsNotNone(segments, f"{cmd!r} classifies as opaque")
            plan_safe = segments[0].kind in READONLY_KINDS
            self.assertEqual(
                plan_safe, category in ("read", "search", "inspect", "neutral", "chdir"),
                f"{command!r} is filed as {category!r} but classifies "
                f"{segments[0].kind!r}")
        # The project-wide validators carry the same invariant, stated in their own
        # comment but never checked: a validator the server refuses (or the
        # classifier cannot tag EXEC) never clears a pending file. `py_compile` runs as
        # `python -m py_compile`, which _exec_head reports as the module name.
        self.assertEqual(
            sorted(obs._PROJECT_VALIDATORS & set(DENIED_COMMANDS)), [],
            "project validators the bash server will not run")
        self.assertEqual(
            sorted(obs._PROJECT_VALIDATORS - set(EXEC_COMMANDS) - {"py_compile"}), [],
            "project validators the classifier does not tag as exec")
        # And the invariant the split into checks-vs-runs added: a head absent from
        # _VALIDATOR_TIER is an execution, for which _bash_validation_scan never reaches
        # the whole-project test — so listing one here is dead weight that reads like a
        # rule. `pytest` and `ctest` sat here unreachable for exactly that reason.
        self.assertEqual(
            sorted(obs._PROJECT_VALIDATORS - set(obs._VALIDATOR_TIER)), [],
            "project checkers absent from _VALIDATOR_TIER are unreachable")

    def test_the_header_table_is_the_denylist(self) -> None:
        """The module docstring is the denylist's only human-readable form.

        It is also the first thing a reader trusts about this server, and prose
        drifts silently because prose is not imported. A denied command missing from
        the table reads as available; a listed one that is not denied reads as
        refused. Both are wrong facts about the one thing this file still gates.
        """
        doc = server_bash.__doc__ or ""
        denied = sorted(_shell_paths.DENIED_COMMANDS)
        table = doc.split("=============  ====================")[2]
        missing = [
            c for c in denied
            if not re.search(rf"(?<![\w.+-]){re.escape(c)}(?![\w.+-])", table)
        ]
        self.assertEqual(missing, [], "denied commands absent from the header table")

    def test_a_wrapper_is_unwrapped_never_denied(self) -> None:
        """A wrapper may not be denied, and a denied head may not be a wrapper.

        The validator reads a head. A command whose argument list *is* another
        command would otherwise clear the denylist on its own name and nothing it
        runs, so wrappers are unwrapped instead of listed — and the two sets must
        stay disjoint or a wrapper would be refused before it could be unwrapped.
        """
        self.assertEqual(
            sorted(_shell_paths.COMMAND_WRAPPERS & _shell_paths.DENIED_COMMANDS), [],
            "a wrapper is denylisted; it would be refused instead of unwrapped")
        for wrapper in sorted(_shell_paths.COMMAND_WRAPPERS):
            for denied in sorted(_shell_paths.SHELL_INTERPRETERS):
                argv, _ = _shell_paths.unwrap_argv([wrapper, denied, "x"])
                self.assertEqual(argv[0], denied, f"{wrapper} {denied}")

    def test_a_wrapper_cannot_smuggle_a_refused_command(self) -> None:
        """The behaviour the set invariant above protects, spelled out end to end."""
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("timeout 5 bash -c 'echo hi'", "nohup sudo ls",
                    "env A=B sh script.sh", "/usr/bin/timeout 5 /bin/sh -c x",
                    "timeout 5 nohup bash -c x", "xargs -n1 bash",
                    "nice -n 10 sbatch job.sh"):
            with self.subTest(command=cmd):
                self.assertEqual(server_bash._validate_command(cmd, cwd).get("status"),
                                 "error", f"{cmd!r} smuggled a denied command through")
        # …without costing the ordinary run its own timeout, which is the tool's to set.
        self.assertEqual(
            server_bash._validate_command("python solver.py", cwd).get("status"), "ok")

    def test_unwrapping_terminates(self) -> None:
        # A chain deeper than the bound stops unwrapping rather than looping, and
        # never loses the head — what is left is still validated.
        argv, wrappers = _shell_paths.unwrap_argv(["timeout", "5"] * 10 + ["ls"])
        self.assertLessEqual(len(wrappers), 4)
        self.assertTrue(argv)

    def test_every_refused_path_is_one_the_user_can_be_asked_about(self) -> None:
        """The invariant tying the guard to the gate.

        The guard refuses a path; the gate offers it to the user; a grant makes the
        guard accept. If the gate cannot see a path the guard refuses, the access is
        unreachable — refused with no way to allow it — and the user never learns
        why. So: every path this guard rejects must be one the gate would surface.
        """
        from mimir.client.guardrails.policy.bash_classify import shell_segments
        from mimir.servers._shared.shell_paths import (
            cd_destination, normalize_path_arg, segment_path_operands,
        )

        cwd = server_bash._WORKSPACE_ROOT
        corpus = [
            "ls > /tmp/out.txt",
            "cat < /etc/passwd",
            "cd sub && ls >> /tmp/out.txt",
            "cat /etc/passwd",
            "python /tmp/evil.py",
            "gcc /tmp/x.c -o out.o",
            "cp /etc/hosts here.txt",
            "mv notes.txt /tmp/out.txt",
            "cd /etc && cat passwd",
            "cd .. && ls",
            "./solver.out /etc/secrets",
            "pdflatex /tmp/a.tex",
            "grep foo /etc/passwd",
            "wc -l /etc/passwd",
            "gcc -I$CUDA_HOME/include /tmp/x.c -o a.out",
            "ls && cat /etc/passwd",
        ]
        for cmd in corpus:
            payload = server_bash._validate_command(cmd, cwd)
            if payload["status"] == "ok" or "not approved" not in payload["error"]:
                continue
            refused = payload["error"].split(": ", 1)[1]

            # What the client gate would collect for the same command.
            offered: set[str] = set()
            cursor = cwd
            for segment in shell_segments(cmd, allow_expansion=True) or []:
                argv = segment.argv
                for target in segment.redirect_targets:
                    resolved = normalize_path_arg(target, cursor)
                    if resolved is not None:
                        offered.add(resolved)
                if argv and argv[0] == "cd":
                    cursor = cd_destination(argv, cursor, cwd)
                    offered.add(cursor)
                    continue
                offered.update(segment_path_operands(argv, cursor))
            self.assertIn(refused, offered,
                          f"{cmd!r}: guard refuses {refused} but the gate never offers it")

    def test_tex_toolchain_is_classified_as_a_build(self) -> None:
        # It runs, like anything else; what matters is that it is not read-only, so
        # a document compile is approval-gated and owes a verdict.
        for tool in ("pdflatex", "xelatex", "lualatex", "latexmk", "bibtex",
                     "biber", "makeindex"):
            self.assertEqual(_shell_paths.EXEC_EFFECTS.get(tool), "build", tool)

    def test_tex_shell_escape_is_rejected(self) -> None:
        # \write18 is arbitrary execution outside the validator; every spelling
        # of the flag that re-enables it must be refused.
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("pdflatex -shell-escape main.tex",
                    "pdflatex --shell-escape main.tex",
                    "lualatex -enable-write18 main.tex",
                    "latexmk -pdf -shell-escape main.tex"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "error", cmd)

    def test_tex_normal_invocation_validates(self) -> None:
        cwd = server_bash._WORKSPACE_ROOT
        for cmd in ("pdflatex main.tex",
                    "pdflatex -interaction=nonstopmode report.tex",
                    "pdflatex main.tex && pdflatex main.tex"):
            payload = server_bash._validate_command(cmd, cwd)
            self.assertEqual(payload["status"], "ok", cmd)

    def test_tex_source_outside_workspace_is_rejected(self) -> None:
        cwd = server_bash._WORKSPACE_ROOT
        payload = server_bash._validate_command("pdflatex /etc/passwd", cwd)
        self.assertEqual(payload["status"], "error")
        self.assertIn("outside workspace", payload["error"])

    def test_tex_sandbox_is_pinned_in_the_environment(self) -> None:
        # Backstop for the denylisted flags: kpathsea reads these from the env,
        # so \write18 stays off even if the site's texmf.cnf enables it.
        env = server_bash._safe_env(server_bash._WORKSPACE_ROOT)
        self.assertEqual(env["shell_escape"], "f")
        self.assertEqual(env["openout_any"], "p")

    def test_bash_run_rejects_outside_workspace(self) -> None:
        payload = server_bash.bash_run("ls ../../..")
        self.assertEqual(payload["status"], "error")

    def test_bash_run_runs_git(self) -> None:
        # git ran through a dedicated server only because bash refused it; that
        # server is gone, and git is an ordinary approval-gated command here.
        payload = server_bash.bash_run("git status --short")
        self.assertEqual(payload["status"], "ok", payload)

    def test_bash_run_allows_python_as_code_fallback(self) -> None:
        # python is now an allowed build/exec tool so the code server can fall
        # back to running commands directly with custom arguments.
        payload = server_bash.bash_run('python3 -c "print(1)"')
        self.assertEqual(payload["status"], "ok")
        self.assertIn("1", payload["stdout"])

    def test_bash_run_allows_command_chaining(self) -> None:
        # Multiple commands joined by '&&' run in one call; each is validated.
        payload = server_bash.bash_run('python3 -c "print(1)" && python3 -c "print(2)"')
        self.assertEqual(payload["status"], "ok")
        self.assertIn("1", payload["stdout"])
        self.assertIn("2", payload["stdout"])

    def test_the_timeout_is_clamped_and_the_clamp_is_reported(self) -> None:
        # The ceiling is a real limit the model must be able to see it hit: a silently
        # shortened run reads as a hang. Below it, the call's own value is honoured.
        payload = server_bash.bash_run("pwd", timeout=server_bash._MAX_TIMEOUT + 60)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["timeout_clamped"])
        self.assertEqual(payload["requested_timeout"], server_bash._MAX_TIMEOUT + 60)
        self.assertNotIn("timeout_clamped", server_bash.bash_run("pwd", timeout=120))
        # And the default is sized for a build or a suite, not for a `ls`.
        self.assertGreaterEqual(server_bash._DEFAULT_TIMEOUT, 30)
        self.assertLessEqual(server_bash._DEFAULT_TIMEOUT, server_bash._MAX_TIMEOUT)

    def test_bash_run_chaining_validates_each_command(self) -> None:
        # A denied command anywhere in the chain rejects the call.
        payload = server_bash.bash_run("pwd && eval ls")
        self.assertEqual(payload["status"], "error")
        self.assertIn("not available here", payload["error"])

    def test_bash_run_quoted_semicolon_is_literal_not_chaining(self) -> None:
        # A ';' inside a quoted argument is literal text, so python runs normally.
        payload = server_bash.bash_run('python3 -c "print(1);print(2)"')
        self.assertEqual(payload["status"], "ok")
        self.assertIn("1", payload["stdout"])
        self.assertIn("2", payload["stdout"])

    def test_bash_run_still_blocks_command_substitution(self) -> None:
        payload = server_bash.bash_run("pwd `whoami`")
        self.assertEqual(payload["status"], "error")


class ReportVerdictTests(unittest.TestCase):
    def test_a_verdict_word_outside_the_three_is_refused(self) -> None:
        payload = server_bash.report_verdict("probably", "residual 3e-4 under the bound")
        self.assertEqual(payload["status"], "error")

    def test_an_empty_reason_is_refused(self) -> None:
        payload = server_bash.report_verdict("pass", "   ")
        self.assertEqual(payload["status"], "error")

    def test_a_reason_naming_what_was_read_is_accepted(self) -> None:
        for reason in (
            "l2_rel=3.1e-4 against the analytic solution, under the 1e-3 bound",
            "energy grows from 1.56 to 4.02 — the absorbing layer reflects",
            "only prints 'Simulation completed.', nothing about correctness",
            "the pulse leaves the domain and the field near the boundary stays flat",
        ):
            payload = server_bash.report_verdict("pass", reason)
            self.assertEqual(payload["status"], "ok", reason)
            self.assertEqual(payload["reason"], reason)

    def test_scope_is_carried_through_verbatim(self) -> None:
        payload = server_bash.report_verdict(
            "fail", "the residual plateaus at 1e-1 instead of falling", " solver.py "
        )
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["run"], "solver.py")


class TodoServerTests(unittest.TestCase):
    def setUp(self) -> None:
        # server_todo resolves its file dynamically via _get_todo_file() (session
        # aware), so redirect that rather than a module-level path constant.
        self._orig = server_todo._get_todo_file
        import tempfile, os
        self._tmpdir = tempfile.mkdtemp()
        todo_path = os.path.join(self._tmpdir, "todo_list.md")
        server_todo._get_todo_file = lambda: todo_path

    def tearDown(self) -> None:
        server_todo._get_todo_file = self._orig
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_todo_write_replaces_list(self) -> None:
        payload = server_todo.todo_write(["step one", "step two"])
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["count"], 2)

    def test_todo_read_returns_all_items_as_undone(self) -> None:
        server_todo.todo_write(["a", "b", "c"])
        payload = server_todo.todo_read()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["items"]), 3)
        self.assertEqual(payload["pending"], 3)
        self.assertEqual(payload["done"], 0)
        self.assertFalse(payload["items"][0]["done"])

    def test_todo_update_marks_item_done(self) -> None:
        server_todo.todo_write(["x", "y"])
        payload = server_todo.todo_update(0, True)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["item"]["done"])
        self.assertEqual(payload["pending"], 1)

    def test_todo_update_out_of_range_is_error(self) -> None:
        server_todo.todo_write(["only one"])
        payload = server_todo.todo_update(5, True)
        self.assertEqual(payload["status"], "error")

    def test_todo_write_empty_clears_list(self) -> None:
        server_todo.todo_write(["old step"])
        server_todo.todo_write([])
        payload = server_todo.todo_read()
        self.assertEqual(payload["items"], [])
        self.assertEqual(payload["pending"], 0)


class EnvServerTests(unittest.TestCase):
    """Input-validation contracts for the env-management server.

    These exercise only the rejection/validation paths so the suite never installs
    packages, reaches the network, or creates/destroys real environments.
    """

    def test_pip_install_rejects_illegal_package(self) -> None:
        payload = server_env.env_pip_install("numpy; rm -rf /")
        self.assertEqual(payload["status"], "error")

    def test_pip_install_rejects_empty_packages(self) -> None:
        payload = server_env.env_pip_install("   ")
        self.assertEqual(payload["status"], "error")

    def test_resolve_python_rejects_relative_path(self) -> None:
        _, error = server_env._resolve_python("./bin/python")
        self.assertIsNotNone(error)
        self.assertEqual(error["status"], "error")

    def test_resolve_python_rejects_missing_absolute(self) -> None:
        _, error = server_env._resolve_python("/no/such/python")
        self.assertIsNotNone(error)

    def test_resolve_python_bare_name_uses_server_interpreter(self) -> None:
        exe, error = server_env._resolve_python("python3")
        self.assertIsNone(error)
        self.assertEqual(exe, sys.executable)

    def test_env_create_rejects_name_with_separator(self) -> None:
        payload = server_env.env_create("../escape", kind="venv")
        self.assertEqual(payload["status"], "error")

    def test_env_create_rejects_unknown_kind(self) -> None:
        payload = server_env.env_create("good_name", kind="pipenv")
        self.assertEqual(payload["status"], "error")

    def test_env_delete_refuses_non_venv_directory(self) -> None:
        # A real, existing directory that is not a virtualenv (no pyvenv.cfg).
        payload = server_env.env_delete(str(SERVERS_DIR), kind="venv")
        self.assertEqual(payload["status"], "error")

    def test_looks_like_venv_false_for_plain_dir(self) -> None:
        self.assertFalse(server_env._looks_like_venv(str(SERVERS_DIR)))

class ToolSchemaHonestyTests(unittest.TestCase):
    """A tool's schema may only advertise policies that exist.

    ``write_file`` once carried a ``large_file_overwrite_confirmed`` flag and a
    docstring promising that ">150 lines blocks overwrite unless it is set". No policy
    ever read it, so the model paid for the field in every tool schema and was told a
    rule that would never fire. The two guards that do exist are the read-before-
    overwrite check and the server's refusal to overwrite without ``overwrite=true``.
    """

    def test_write_file_advertises_no_line_threshold_policy(self) -> None:
        import inspect
        source = (SERVERS_DIR / "workspace" / "server_files.py").read_text()
        self.assertNotIn("large_file_overwrite_confirmed", source)

        sig = inspect.signature(server_files.write_file)
        self.assertEqual(
            list(sig.parameters), ["path", "content", "overwrite"]
        )
        self.assertNotIn("150", server_files.write_file.__doc__ or "")


if __name__ == "__main__":
    unittest.main()
