"""
MCP Environment-Management Server
=================================
Mutating counterpart to the read-only environment *discovery* in the platform
server. These tools install packages into, and create/destroy, Python runtime
environments so the agent can satisfy a missing-module dependency when no existing
environment already has it (Tiers 2 & 3 of the environment-resolution cascade).

Every tool here mutates persistent state, so each is gated:
  - PLAN_BLOCKED  — never runs in plan mode.
  - sensitive + non_batch — always prompts for explicit approval, never batched.
  - ENV_MUTATE (install / create only) — records a cleanup obligation the client
    surfaces at the conclude phase ("offer to uninstall / delete").

Inputs are validated and shell-free (argv lists, no shell=True), and destructive
operations are restricted to things that actually look like Python environments.
"""

import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps, PLAN_BLOCKED, ENV_MUTATE, REMOVE, RECOVERABLE
from responses import err, ok

mcp = FastMCP(
    "EnvServer",
    debug=False,
    log_level="ERROR",
)

# pip / env creation reach the network and can be slow; give them real headroom.
_TIMEOUT = int(os.environ.get("MCP_ENV_TIMEOUT", "900"))

# Where `env_create(kind="venv")` puts new virtualenvs when no explicit path is
# given: a managed directory under the workspace so created envs are easy to find
# (and to clean up). Mirrors MCP_FILES_ROOT used elsewhere.
_WORKSPACE_ROOT = os.environ.get("MCP_FILES_ROOT", os.getcwd())
_ENV_HOME = os.path.join(_WORKSPACE_ROOT, ".mimir_envs")

# Reject obvious shell-injection / traversal in user-supplied tokens. Tokens reach
# subprocess as argv (never a shell), so this is defense-in-depth, not the only gate.
_FORBIDDEN = set(";|&`$><!()\n\r\t")


def _run(cmd: list[str], timeout: int = _TIMEOUT) -> dict:
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": res.returncode == 0,
            "returncode": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": f"timed out after {timeout}s"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(exc)}


def _clean_token(value: str) -> str:
    return str(value or "").strip()


def _has_forbidden(value: str) -> bool:
    return any(c in _FORBIDDEN for c in value)


def _normalize_packages(packages) -> tuple[list[str], dict | None]:
    """Coerce the packages arg (str or list) into a validated argv list."""
    if isinstance(packages, str):
        items = packages.split()
    elif isinstance(packages, (list, tuple)):
        items = [str(p) for p in packages]
    else:
        return [], err("packages must be a string or a list of strings.")
    items = [_clean_token(p) for p in items if _clean_token(p)]
    if not items:
        return [], err("No package specified.", hint="Pass one or more package names, e.g. 'numpy scipy'.")
    for p in items:
        if _has_forbidden(p):
            return [], err(f"Illegal characters in package spec: {p!r}")
    return items, None


def _resolve_python(python_executable: str) -> tuple[str, dict | None]:
    """Validate the interpreter the install/uninstall targets.

    Bare ``python``/``python3`` resolves to the server's own interpreter (the same
    rule the code runtime uses, so the default is always runnable on this host);
    an absolute path must exist and be executable; everything else resolves via PATH.
    """
    exe = _clean_token(python_executable) or "python3"
    if _has_forbidden(exe):
        return "", err(f"Illegal characters in python_executable: {exe!r}")
    if os.sep in exe and not os.path.isabs(exe):
        return "", err("Relative interpreter paths are not allowed.", hint="Pass an absolute path or a bare name.")
    if os.path.isabs(exe):
        if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
            return "", err(f"Interpreter not found or not executable: {exe}",
                           hint="Check the path to your conda/venv python binary.")
        return exe, None
    if exe in ("python", "python3") and sys.executable:
        return sys.executable, None
    resolved = shutil.which(exe)
    if resolved is None:
        return "", err(f"Interpreter not found on PATH: {exe}")
    return resolved, None


def _looks_like_venv(path: str) -> bool:
    """A directory is a virtualenv iff it carries the marker config + a python binary."""
    return (
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "pyvenv.cfg"))
        and os.path.isfile(os.path.join(path, "bin", "python"))
    )


@mcp.tool(**tool_caps(
    caps=[PLAN_BLOCKED, ENV_MUTATE], reversibility=RECOVERABLE, non_batch=True,
    scope={"args": ["packages"], "kind": "packages"},
    risk_note="installs packages into a Python environment",
))
def env_pip_install(packages, python_executable: str = "python3") -> dict:
    """Install one or more packages into an existing Python environment (sensitive).

    Use this only after environment *discovery* found no existing interpreter that
    already resolves the required module(s). Prefer installing into a project/conda
    env (pass its absolute ``bin/python`` as ``python_executable``) over the default
    interpreter. Returns the installed packages so the caller can offer to uninstall
    them when the task concludes.

    Args:
        packages: package spec(s), e.g. "numpy" or ["numpy", "scipy>=1.10"].
        python_executable: interpreter whose environment to install into
            (absolute path to a conda/venv python, or a bare name).
    """
    pkgs, error = _normalize_packages(packages)
    if error:
        return error
    py, error = _resolve_python(python_executable)
    if error:
        return error
    result = _run([py, "-m", "pip", "install", *pkgs])
    if not result["ok"]:
        return err(
            result["stderr"].strip() or "pip install failed",
            hint="Check the package name/version and that the target environment is writable.",
            returncode=result["returncode"],
            stdout=result["stdout"][-2000:],
        )
    return ok({
        "installed": pkgs,
        "python": py,
        "stdout": result["stdout"][-2000:],
        "cleanup_hint": "Offer to env_pip_uninstall these from this interpreter when the task concludes.",
    })


@mcp.tool(**tool_caps(
    caps=[PLAN_BLOCKED], reversibility=RECOVERABLE, non_batch=True,
    scope={"args": ["packages"], "kind": "packages"},
    risk_note="uninstalls packages from a Python environment",
))
def env_pip_uninstall(packages, python_executable: str = "python3") -> dict:
    """Uninstall packages from an environment (sensitive) — the cleanup for env_pip_install.

    Args:
        packages: package spec(s) to remove.
        python_executable: interpreter whose environment to uninstall from.
    """
    pkgs, error = _normalize_packages(packages)
    if error:
        return error
    py, error = _resolve_python(python_executable)
    if error:
        return error
    result = _run([py, "-m", "pip", "uninstall", "-y", *pkgs])
    if not result["ok"]:
        return err(
            result["stderr"].strip() or "pip uninstall failed",
            returncode=result["returncode"],
            stdout=result["stdout"][-2000:],
        )
    return ok({"uninstalled": pkgs, "python": py, "stdout": result["stdout"][-2000:]})


@mcp.tool(**tool_caps(
    caps=[PLAN_BLOCKED, ENV_MUTATE], reversibility=RECOVERABLE, non_batch=True,
    scope={"args": ["name", "target"], "kind": "basename", "noun": "this environment"},
    risk_note="creates a new Python environment",
))
def env_create(name: str, kind: str = "venv", packages=None, python_executable: str = "python3") -> dict:
    """Create a new Python environment (sensitive).

    Use this as the last resort in the cascade, when no existing environment has the
    required module(s) and installing into one is not appropriate. Returns the new
    interpreter's path so the caller can use it as ``python_executable`` for the
    original check, and can offer to delete the env when the task concludes.

    Args:
        name: environment name (used as the venv directory name or the conda env name).
        kind: "venv" (always available) or "conda" (only if conda is installed).
        packages: optional package spec(s) to install into the new env.
        python_executable: base interpreter for a venv (ignored for conda).
    """
    name = _clean_token(name)
    if not name or _has_forbidden(name) or os.sep in name:
        return err("Invalid environment name.", hint="Use a simple name with no path separators or shell characters.")
    kind = _clean_token(kind).lower() or "venv"
    pkgs: list[str] = []
    if packages is not None:
        pkgs, error = _normalize_packages(packages)
        if error:
            return error

    if kind == "venv":
        base_py, error = _resolve_python(python_executable)
        if error:
            return error
        os.makedirs(_ENV_HOME, exist_ok=True)
        target = os.path.join(_ENV_HOME, name)
        if os.path.exists(target):
            return err(f"Path already exists: {target}", hint="Choose another name or delete the existing env first.")
        created = _run([base_py, "-m", "venv", target])
        if not created["ok"]:
            return err(created["stderr"].strip() or "venv creation failed", returncode=created["returncode"])
        new_py = os.path.join(target, "bin", "python")
        if pkgs:
            inst = _run([new_py, "-m", "pip", "install", *pkgs])
            if not inst["ok"]:
                return err(
                    inst["stderr"].strip() or "package install in new venv failed",
                    hint=f"The venv was created at {target}; you may env_delete it and retry.",
                    python=new_py, returncode=inst["returncode"],
                )
        return ok({
            "kind": "venv",
            "name": name,
            "path": target,
            "python": new_py,
            "installed": pkgs,
            "cleanup_hint": "Offer to env_delete this path when the task concludes.",
        })

    if kind == "conda":
        if not shutil.which("conda"):
            return err("conda is not available on this host.", hint="Use kind='venv' instead.")
        cmd = ["conda", "create", "-n", name, "-y", "python", *pkgs]
        created = _run(cmd)
        if not created["ok"]:
            return err(created["stderr"].strip() or "conda env creation failed", returncode=created["returncode"])
        new_py = os.path.join(os.path.expanduser("~"), ".conda", "envs", name, "bin", "python")
        return ok({
            "kind": "conda",
            "name": name,
            "python": new_py if os.path.isfile(new_py) else "",
            "installed": pkgs,
            "stdout": created["stdout"][-2000:],
            "cleanup_hint": "Offer to env_delete this conda env (by name) when the task concludes.",
        })

    return err(f"Unsupported env kind: {kind!r}", hint="Use 'venv' or 'conda'.")


@mcp.tool(**tool_caps(
    caps=[PLAN_BLOCKED, REMOVE], reversibility=RECOVERABLE, non_batch=True,
    scope={"args": ["name", "target"], "kind": "basename", "noun": "this environment"},
    risk_note="deletes a Python environment",
))
def env_delete(target: str, kind: str = "venv") -> dict:
    """Delete an environment created by env_create (sensitive) — the cleanup for it.

    Args:
        target: for kind="venv", the venv directory path; for kind="conda", the env name.
        kind: "venv" or "conda".
    """
    target = _clean_token(target)
    kind = _clean_token(kind).lower() or "venv"
    if not target or _has_forbidden(target):
        return err("Invalid target.")

    if kind == "venv":
        path = os.path.abspath(target)
        if not _looks_like_venv(path):
            return err(
                f"Refusing to delete: {path} is not a virtualenv.",
                hint="env_delete only removes directories with a pyvenv.cfg and bin/python.",
            )
        try:
            shutil.rmtree(path)
        except OSError as exc:
            return err(f"Failed to remove {path}: {exc}")
        return ok({"deleted": path, "kind": "venv"})

    if kind == "conda":
        if not shutil.which("conda"):
            return err("conda is not available on this host.")
        if os.sep in target:
            return err("For conda, pass the environment name, not a path.")
        removed = _run(["conda", "env", "remove", "-n", target, "-y"])
        if not removed["ok"]:
            return err(removed["stderr"].strip() or "conda env remove failed", returncode=removed["returncode"])
        return ok({"deleted": target, "kind": "conda", "stdout": removed["stdout"][-2000:]})

    return err(f"Unsupported env kind: {kind!r}", hint="Use 'venv' or 'conda'.")


if __name__ == "__main__":
    mcp.run()
