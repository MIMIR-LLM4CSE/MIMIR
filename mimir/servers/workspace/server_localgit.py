"""
MCP Local Git Server
====================
Read-only git inspection of the **local working tree** (the checkout the agent is
operating on), as opposed to ``server_github`` which queries the remote GitHub API.

Why this is a separate server (not part of ``server_bash``)
----------------------------------------------------------
``git`` is the single hardest command to sandbox inside a generic shell: ``git -c
alias.x='!sh' x``, ``-c core.sshCommand=...``, ``--ext-diff`` + ``diff.external``,
and ``-O``/``--open-files-in-pager`` are all config-injection code-execution
vectors that a flag denylist only partially catches. Rather than carry that surface
in ``server_bash``, local git lives here behind a strict subcommand allowlist and
runs with ``shell=False`` (no shell metacharacter interpretation at all), pinned to
the workspace root.

Safety model
------------
- Only the read-only subcommands in ``_READ_ONLY_SUBCOMMANDS`` are accepted.
- ``-c`` / ``--exec-path`` (and their ``=`` forms) are refused outright.
- Every invocation runs ``git`` with ``shell=False``, ``cwd`` pinned to the
  workspace root, a minimal environment, and an output cap.
- All tools are read-only, so none are approval-gated.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from responses import err, ok
from capabilities import tool_caps

mcp = FastMCP(
    "LocalGitServer",
    debug=False,
    log_level="ERROR",
)

_WORKSPACE_ROOT = os.path.realpath(
    os.path.abspath(os.environ.get("MCP_FILES_ROOT", os.getcwd()))
)
_MAX_OUTPUT = 64 * 1024
_DEFAULT_TIMEOUT = 15

_READ_ONLY_SUBCOMMANDS = {
    "status", "log", "show", "diff", "branch", "rev-parse", "remote",
    "ls-files", "grep", "describe", "tag", "blame",
}

# Global git options that can lead to arbitrary code execution or sandbox escape.
_REFUSED_GIT_OPTIONS = ("-c", "--exec-path")


def _safe_env(cwd: str) -> dict:
    """Minimal environment for subprocess execution (mirrors server_bash)."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        # Inherited, not pinned to the workspace: git reads identity, aliases and
        # includes from ~/.gitconfig, so pinning HOME made every command run as a
        # user with no configuration — a blame/log format the user relies on, or a
        # commit identity, silently absent. Confinement is the repo path check and
        # the refused options above, not HOME. Mirrors server_bash._safe_env.
        "HOME": os.path.realpath(os.path.abspath(os.path.expanduser("~"))),
        # git honors core.pager only on a tty; subprocess pipes have none, but be
        # explicit and avoid loading any pager program.
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _reject_unsafe_args(args: list[str]) -> dict | None:
    """Return an error payload if any arg is a refused global git option."""
    for arg in args:
        for opt in _REFUSED_GIT_OPTIONS:
            if arg == opt or arg.startswith(opt + "="):
                return err(
                    f"Refused git option: {arg}",
                    hint="Options like -c and --exec-path can inject code and are not allowed.",
                )
    return None


def _git(args: list[str], timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """Run a read-only git command with shell=False, pinned to the workspace.

    ``args`` is the full argument vector after ``git`` (e.g. ["status", "--short"]).
    The first element must be an allowlisted read-only subcommand.
    """
    if not args:
        return err("No git subcommand provided.")

    unsafe = _reject_unsafe_args(args)
    if unsafe is not None:
        return unsafe

    subcommand = args[0]
    if subcommand not in _READ_ONLY_SUBCOMMANDS:
        return err(
            f"git subcommand '{subcommand}' is not allowed.",
            hint="Allowed: " + ", ".join(sorted(_READ_ONLY_SUBCOMMANDS)) + ".",
        )

    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_WORKSPACE_ROOT,
            env=_safe_env(_WORKSPACE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
    except FileNotFoundError:
        return err("git executable not found on PATH.")
    except subprocess.TimeoutExpired:
        return err(f"git command timed out after {timeout}s.",
                   hint="Narrow the request (e.g. a smaller max_count or a specific path).")
    except Exception as exc:  # noqa: BLE001 — surface any launch failure structurally
        return err(str(exc))

    stdout = result.stdout[:_MAX_OUTPUT]
    stderr = result.stderr[:_MAX_OUTPUT]

    if result.returncode != 0:
        return err(
            f"git {subcommand} failed (exit {result.returncode}).",
            hint=stderr.strip()[-500:] or "Check the arguments and repository state.",
            returncode=result.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    payload = ok({
        "subcommand": subcommand,
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "cwd": _WORKSPACE_ROOT,
    })
    if len(result.stdout) > _MAX_OUTPUT or len(result.stderr) > _MAX_OUTPUT:
        payload["truncated"] = True
    return payload


# ── tool — single read-only dispatch ──────────────────────────────────────────

_GIT_OPS = ("status", "log", "diff", "show", "branches", "blame", "grep")


@mcp.tool(**tool_caps(label="git {op}"))
def git(
    op: str,
    ref: str = "",
    path: str = "",
    pattern: str = "",
    max_count: int = 20,
    staged: bool = False,
    start_line: int = 0,
    end_line: int = 0,
) -> dict:
    """Read-only inspection of the local git working tree, selected by ``op``.

    Operations (set ``op`` to one of these):
      status   -> working-tree status (porcelain --short --branch)
      log      -> recent commits (uses `max_count` 1-200, optional `path`)
      diff     -> uncommitted changes (optional `path`; `staged=True` for --cached)
      show     -> a commit's metadata + patch for `ref` (SHA/tag/branch)
      branches -> list local branches (current marked)
      blame    -> line authorship of `path` (optional `start_line`..`end_line`)
      grep     -> search tracked files for `pattern` (optional `path`)

    All operations are read-only (none are approval-gated). Returns the structured
    ``_git`` payload: {subcommand, returncode, stdout, stderr, cwd}.

    Args:
        op:         The operation to perform (see list above).
        ref:        Commit SHA / tag / branch for ``show``.
        path:       Optional pathspec for ``log`` / ``diff`` / ``blame`` / ``grep``.
        pattern:    Search pattern for ``grep``.
        max_count:  Commit limit for ``log`` (1-200).
        staged:     ``diff`` staged changes (--cached) when True.
        start_line, end_line: 1-based line range for ``blame``.
    """
    if op == "status":
        return _git(["status", "--short", "--branch"])
    if op == "log":
        n = max(1, min(int(max_count), 200))
        args = ["log", f"--max-count={n}", "--pretty=format:%h %ad %an %s", "--date=short"]
        if path:
            args += ["--", path]
        return _git(args)
    if op == "diff":
        args = ["diff"]
        if staged:
            args.append("--cached")
        if path:
            args += ["--", path]
        return _git(args)
    if op == "show":
        if not ref or not ref.strip():
            return err("ref is required for show.", hint="Pass a commit SHA, tag, or branch name.")
        return _git(["show", "--stat", "--patch", ref.strip()])
    if op == "branches":
        return _git(["branch", "--list", "-vv"])
    if op == "blame":
        if not path or not path.strip():
            return err("path is required for blame.")
        args = ["blame"]
        if start_line > 0 and end_line >= start_line:
            args += ["-L", f"{start_line},{end_line}"]
        args += ["--", path.strip()]
        return _git(args)
    if op == "grep":
        if not pattern or not pattern.strip():
            return err("pattern is required for grep.")
        args = ["grep", "-n", "-e", pattern]
        if path:
            args += ["--", path]
        return _git(args)
    return err(
        f"Unknown git op '{op}'.",
        hint=f"Use one of: {', '.join(_GIT_OPS)}.",
    )


if __name__ == "__main__":
    mcp.run()
