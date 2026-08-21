"""Shared test oracle for tool classification.

Shipped code no longer contains any hardcoded tool-classification list — each
server declares its tools' capabilities via ``@mcp.tool(**tool_caps(...))`` and the
client reads them from the per-agent live registry. So the *expected* classification
lives here, in the tests, as two pieces:

* ``GOLDEN`` — independent literal snapshots of the intended classification (the
  regression oracle: drift fails loudly).
* ``build_declared_registry()`` — the ``{name: ToolCaps}`` registry parsed straight
  from the server source decorators via AST (no ``mcp`` import needed, so it runs on
  the x86 build host). This is what the live agent's ``connect_server`` produces.

The faithfulness test asserts the declared registry reproduces ``GOLDEN`` exactly.
"""

import ast
import inspect
import pathlib
import types

from mimir.client.context.capabilities import infer_tool_caps
from mimir.servers._shared import capabilities as srv

SERVERS_DIR = pathlib.Path(__file__).resolve().parents[1] / "servers"

# tool_caps() accepts annotation-only kwargs (read_only, destructive) that set
# ToolAnnotations and are deliberately NOT forwarded to build_descriptor (they
# don't affect the classification descriptor). The AST replay below must mirror
# that: keep only the kwargs build_descriptor actually takes.
_BUILD_DESCRIPTOR_PARAMS = frozenset(inspect.signature(srv.build_descriptor).parameters)


# --- golden expected classification (independent literal snapshots) ----------
SENSITIVE_TOOLS = {
    "delete_file", "bash_run", "http_post",
    "salloc_submit", "sbatch_submit", "benchmark_summary", "memory_delete",
    "memory_clear", "todo_delete_plan",
    "ft_config_set", "ft_run", "ft_run_slurm", "ft_stop", "ft_runner_promote",
    "proxy_manage", "proxy_exec", "proxy_eval", "proxy_slurm",
    "env_pip_install", "env_pip_uninstall", "env_create", "env_delete",
}

# bash_run is intentionally NOT here: it is PLAN_READONLY (kept available in plan
# mode for read-only discovery; its exec use is gated client-side at call time).
PLAN_BLOCKED_TOOLS = {
    "append_file", "delete_file", "replace_in_file",
    "replace_all_in_file", "replace_lines", "write_file", "salloc_submit",
    "http_post", "memory_delete", "memory_clear",
    "ft_config_set", "ft_run", "ft_run_slurm", "ft_stop",
    "ft_runner_promote", "benchmark_summary", "benchmark_memory_copy",
    "benchmark_numpy_matmul", "benchmark_python_compute",
    "proxy_manage", "proxy_exec", "proxy_eval", "proxy_slurm", "apply_edits",
    "salloc_submit", "sbatch_submit",
    "env_pip_install", "env_pip_uninstall", "env_create", "env_delete",
}

NON_BATCH_TOOLS = {
    "proxy_manage", "proxy_exec", "proxy_eval", "proxy_slurm",
    "salloc_submit", "sbatch_submit", "ft_run", "ft_run_slurm", "ft_stop",
    "ft_runner_promote",
    "bash_run", "http_post", "memory_delete",
    "memory_clear", "todo_delete_plan", "benchmark_summary",
    "env_pip_install", "env_pip_uninstall", "env_create", "env_delete",
}

CLUSTER_SUBMIT_TOOLS = {
    "salloc_submit", "sbatch_submit", "ft_run_slurm", "proxy_slurm",
}

# Launchers of long detached runs a client watcher can track to completion.
BACKGROUNDABLE_TOOLS = {
    "proxy_eval", "proxy_slurm", "sbatch_submit",
}

# Dual-use exec tools kept available in plan mode for read-only discovery only.
PLAN_READONLY_TOOLS = {"bash_run"}

SEARCH_TOOLS = set()
EDIT_TOOLS = {
    "write_file", "append_file", "replace_in_file", "replace_all_in_file",
    "replace_lines", "apply_edits",
}
CONTENT_WRITE_TOOLS = {"append_file", "write_file"}
REPLACEMENT_TRACK_TOOLS = {"replace_in_file", "replace_all_in_file"}
READ_TOOLS = {"read_file_lines"}
CANDIDATE_SEARCH_TOOLS = set()
INSPECT_DIR_TOOLS = {"list_directory", "tree_summary"}
CHECK_EXISTENCE_TOOLS = set()
CACHEABLE_TOOLS = {
    "read_file_lines", "read_files",
    "tree_summary", "list_directory",
}
SEARCH_WITH_PATH_TOOLS = set()
EXTERNAL_FETCH_TOOLS = {
    "github_repo_info", "github_list_branches", "github_list_issues",
    "github_get_file", "github_search_repositories",
}

# --- file-mutation / planning behavioral categories --------------------------
REMOVE_TOOLS = {"delete_file", "env_delete"}
OVERWRITE_TOOLS = {"write_file"}
TASK_PLANNING_TOOLS = {"todo_write", "todo_set_plan"}
JUDGE_TOOLS = {"report_verdict"}

# --- code-intelligence navigation (server_code_intel.py) ---------------------
# These nav tools join the broad read/search caps.
CODE_NAV_TOOLS = {"find_definition", "symbol_outline"}
_CODE_INTEL_NAV = {"find_definition", "find_references", "symbol_outline", "hover"}
READ_TOOLS = READ_TOOLS | _CODE_INTEL_NAV
CACHEABLE_TOOLS = CACHEABLE_TOOLS | _CODE_INTEL_NAV
SEARCH_TOOLS = SEARCH_TOOLS | {"find_definition", "find_references"}
SEARCH_WITH_PATH_TOOLS = SEARCH_WITH_PATH_TOOLS | {"find_definition", "find_references"}
CANDIDATE_SEARCH_TOOLS = CANDIDATE_SEARCH_TOOLS | {"find_definition"}

PATH_ARGS_BY_TOOL = {
    "list_directory": ("path",), "tree_summary": ("path",),
    "read_file_lines": ("path",),
    "symbol_outline": ("path",), "hover": ("path",),
}

FALLBACK_TOOLS = {
    "bash_run": ("read_file_lines",),
    "write_file": ("replace_in_file", "read_file_lines"),
    "delete_file": ("read_file_lines",),
}

# edit_sig arg-role: the args a server declares as identifying an edit (used by
# the dedup-signature builder in policy/runtime.py instead of hardcoded branches).
EDIT_SIG_ARGS = {
    "write_file": ("content",),
    "append_file": ("content",),
    "replace_in_file": ("old_text", "new_text"),
    "replace_lines": ("start_line", "end_line", "new_content"),
}

# line_range arg-role: the args a ranged read declares so policy/runtime.py can do
# its line accounting off the declared role instead of a hardcoded tool name.
LINE_RANGE_ARGS = {
    "read_file_lines": ("start_line", "end_line"),
}

# edit_batch arg-role: the arg a batch-edit tool declares as carrying its list of
# sub-edits (so the observers/targeting find the paths without naming the tool).
EDIT_BATCH_ARGS = {
    "apply_edits": ("edits_json",),
}

# plan_steps arg-role: the arg the task-checklist tool declares as carrying its ordered
# steps — distinguishes the checklist from the prose-rationale planning tool by structure.
PLAN_STEPS_ARGS = {
    "todo_write": ("steps",),
}

# verdict arg-roles: the args the judging tool declares as carrying the verdict itself,
# the reason behind it, and the run it speaks for — read by the client observer, which
# never names the tool or its parameters.
VERDICT_ARGS = {
    "report_verdict": ("verdict",),
}
VERDICT_REASON_ARGS = {
    "report_verdict": ("reason",),
}
VERDICT_SCOPE_ARGS = {
    "report_verdict": ("run",),
}

# confirm_gate arg-role: the boolean arg whose truthiness toggles a tool between a
# read-only preview and a real mutation; drives is_sensitive for these tools.
CONFIRM_GATE_ARGS = {
    "replace_all_in_file": ("confirm",),
    "apply_edits": ("confirm",),
}

# preview kind: the generic diff-shape a file-mutating tool declares so the WS UI
# reconstructs a pre-write diff (ui/file_preview.py) and auto-approves the call
# without naming tools. A tool with no preview spec gets the normal approval flow.
PREVIEW_KIND_BY_TOOL = {
    "write_file": "content",
    "append_file": "append",
    "replace_in_file": "replace",
    "replace_all_in_file": "replace_all",
    "replace_lines": "line_splice",
    "delete_file": "delete",
}

# scope kind a tool declares to narrow session "always" approvals (and, for `host`,
# to drive the conditional URL-sensitivity gate). Keyed off the live declaration so a
# missing/renamed scope spec — including http_get's, the one fail-open surface —
# fails this gate loudly.
SCOPE_KIND_BY_TOOL = {
    "bash_run": "command_prefix",
    "http_get": "host",
    "http_post": "host",
    "env_pip_install": "packages",
    "env_pip_uninstall": "packages",
    "env_create": "basename",
    "env_delete": "basename",
}

# Every tool that surfaces a custom risk sentence in the approval prompt. Guards
# against a sensitive tool silently losing its risk_note in the descriptor migration.
RISK_NOTE_TOOLS = {
    "write_file", "append_file", "delete_file", "replace_in_file",
    "replace_all_in_file", "apply_edits",
    "bash_run", "http_get", "http_post",
    "env_pip_install", "env_pip_uninstall", "env_create", "env_delete",
    "salloc_submit", "sbatch_submit", "benchmark_summary", "memory_delete",
    "memory_clear", "todo_delete_plan", "proxy_slurm",
}

#: capability constant -> golden member set
GOLDEN = {
    "sensitive": SENSITIVE_TOOLS,
    "plan_blocked": PLAN_BLOCKED_TOOLS,
    "plan_readonly": PLAN_READONLY_TOOLS,
    "non_batch": NON_BATCH_TOOLS,
    "cluster_submit": CLUSTER_SUBMIT_TOOLS,
    "backgroundable": BACKGROUNDABLE_TOOLS,
    "search": SEARCH_TOOLS,
    "edit": EDIT_TOOLS,
    "content_write": CONTENT_WRITE_TOOLS,
    "replacement_track": REPLACEMENT_TRACK_TOOLS,
    "read": READ_TOOLS,
    "candidate_search": CANDIDATE_SEARCH_TOOLS,
    "inspect_dir": INSPECT_DIR_TOOLS,
    "check_existence": CHECK_EXISTENCE_TOOLS,
    "cacheable": CACHEABLE_TOOLS,
    "search_with_path": SEARCH_WITH_PATH_TOOLS,
    "external_fetch": EXTERNAL_FETCH_TOOLS,
    "code_nav": CODE_NAV_TOOLS,
    "remove": REMOVE_TOOLS,
    "overwrite": OVERWRITE_TOOLS,
    "task_planning": TASK_PLANNING_TOOLS,
    "judge": JUDGE_TOOLS,
}


# --- the declared registry, parsed from server source via AST ----------------
def _literal(node):
    if isinstance(node, ast.List):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return getattr(srv, node.id)  # a capability constant (e.g. INSPECT_DIR)
    raise AssertionError(f"unexpected tool_caps arg node: {ast.dump(node)}")


def build_declared_registry() -> dict:
    """``{tool_name: ToolCaps}`` parsed from every ``@mcp.tool(**tool_caps(...))``.

    Mirrors what ``connect_server`` builds at runtime from the live ``meta``.
    """
    declared: dict = {}
    for path in SERVERS_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if not (isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Attribute)
                        and dec.func.attr == "tool"):
                    continue
                for kw in dec.keywords:
                    if (kw.arg is None
                            and isinstance(kw.value, ast.Call)
                            and getattr(kw.value.func, "id", None) == "tool_caps"):
                        kwargs = {k.arg: _literal(k.value) for k in kw.value.keywords
                                  if k.arg in _BUILD_DESCRIPTOR_PARAMS}
                        desc = srv.build_descriptor(**kwargs)
                        fake = types.SimpleNamespace(
                            name=node.name, meta={"mimir": desc},
                            annotations=None, inputSchema=None,
                        )
                        assert node.name not in declared, f"duplicate tool decl: {node.name}"
                        declared[node.name] = infer_tool_caps(fake)
    return declared
