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
    "salloc_submit", "sbatch_submit", "memory_delete",
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
    "ft_runner_promote",
    "proxy_manage", "proxy_exec", "proxy_eval", "proxy_slurm",
    "salloc_submit", "sbatch_submit",
    "env_pip_install", "env_pip_uninstall", "env_create", "env_delete",
}

NON_BATCH_TOOLS = {
    "proxy_manage", "proxy_exec", "proxy_eval", "proxy_slurm",
    "salloc_submit", "sbatch_submit", "ft_run", "ft_run_slurm", "ft_stop",
    "ft_runner_promote",
    "bash_run", "http_post", "memory_delete",
    "memory_clear", "todo_delete_plan",
    "env_pip_install", "env_pip_uninstall", "env_create", "env_delete",
}

CLUSTER_SUBMIT_TOOLS = {
    "salloc_submit", "sbatch_submit", "ft_run_slurm", "proxy_slurm",
}

# Launchers of long detached runs a client watcher can track to completion.
BACKGROUNDABLE_TOOLS = {
    "proxy_eval", "proxy_slurm", "sbatch_submit",
}

# Dual-use tools kept available in a read-only mode for read-only invocations only:
# the shell (judged per command), and the sub-agent spawn (judged per role).
PLAN_READONLY_TOOLS = {"bash_run", "spawn_agent"}

SEARCH_TOOLS = set()
EDIT_TOOLS = {
    "write_file", "append_file", "replace_in_file", "replace_all_in_file",
    "replace_lines",
}
CONTENT_WRITE_TOOLS = {"append_file", "write_file"}
REPLACEMENT_TRACK_TOOLS = {"replace_in_file", "replace_all_in_file"}
READ_TOOLS = {"read_file_lines"}
CANDIDATE_SEARCH_TOOLS = set()
INSPECT_DIR_TOOLS = {"list_directory", "tree_summary"}
CHECK_EXISTENCE_TOOLS = set()
CACHEABLE_TOOLS = {
    "read_file_lines",
    "tree_summary", "list_directory",
}
SEARCH_WITH_PATH_TOOLS = set()
# Every tool that opens a socket to a host. The two HTTP tools take an arbitrary
# model-supplied URL and were the ones NOT carrying the capability, while the GitHub
# tools — one fixed, well-known host — were the only ones that did.
EXTERNAL_FETCH_TOOLS = {
    "github_repo_info", "github_list_branches", "github_list_issues",
    "github_get_file", "github_search_repositories",
    "http_get", "http_post",
}

# --- file-mutation / planning behavioral categories --------------------------
REMOVE_TOOLS = {"delete_file", "env_delete"}
OVERWRITE_TOOLS = {"write_file"}
TASK_PLANNING_TOOLS = {"todo_write", "todo_set_plan"}
JUDGE_TOOLS = {"report_verdict"}
DELEGATE_TOOLS = {"spawn_agent"}

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

# edit_sig arg-role: the args a server declares as identifying an edit (used by the
# dedup-signature builder in guardrails/observations.py instead of hardcoded branches).
EDIT_SIG_ARGS = {
    "write_file": ("content",),
    "append_file": ("content",),
    "replace_in_file": ("old_text", "new_text"),
    "replace_lines": ("start_line", "end_line", "new_content"),
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
}

# preview kind: the generic diff-shape a file-mutating tool declares so the WS UI
# reconstructs a pre-write diff (ui/ws/file_preview.py) and auto-approves the call
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
    "replace_all_in_file",
    "bash_run", "http_get", "http_post",
    "env_pip_install", "env_pip_uninstall", "env_create", "env_delete",
    "salloc_submit", "sbatch_submit", "memory_delete",
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
    "delegate": DELEGATE_TOOLS,
}


# --- the declared registry, parsed from server source via AST ----------------
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b}


def _literal(node, module_consts: dict | None = None):
    consts = module_consts or {}
    if isinstance(node, ast.List):
        return [_literal(e, consts) for e in node.elts]
    if isinstance(node, ast.Dict):
        # A `**base` entry parses as a None key; merge the mapping it unpacks so a
        # declaration can extend a shared constant instead of restating it.
        out: dict = {}
        for k, v in zip(node.keys, node.values):
            if k is None:
                out.update(_literal(v, consts))
            else:
                out[_literal(k, consts)] = _literal(v, consts)
        return out
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        # A capability constant, or one the declaring server defines for itself
        # (a role name, its own time cap).
        if node.id in consts:
            return consts[node.id]
        return getattr(srv, node.id)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](
            _literal(node.left, consts), _literal(node.right, consts),
        )
    raise AssertionError(f"unexpected tool_caps arg node: {ast.dump(node)}")


def _module_constants(tree: ast.Module) -> dict:
    """Module-level ``NAME = <literal>`` bindings, so a decorator may name its own."""
    consts: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            consts[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue
    return consts


def build_declared_registry() -> dict:
    """``{tool_name: ToolCaps}`` parsed from every ``@mcp.tool(**tool_caps(...))``.

    Mirrors what ``connect_server`` builds at runtime from the live ``meta``.
    """
    declared: dict = {}
    for path in SERVERS_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        module_consts = _module_constants(tree)
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
                        kwargs = {k.arg: _literal(k.value, module_consts)
                                  for k in kw.value.keywords
                                  if k.arg in _BUILD_DESCRIPTOR_PARAMS}
                        desc = srv.build_descriptor(**kwargs)
                        fake = types.SimpleNamespace(
                            name=node.name, meta={"mimir": desc},
                            annotations=None, inputSchema=None,
                        )
                        assert node.name not in declared, f"duplicate tool decl: {node.name}"
                        declared[node.name] = infer_tool_caps(fake)
    return declared
