"""Shared helpers for Environment Modules / Lmod shell initialization."""

import os


# Env vars Lmod's init/bash needs to locate and evaluate modulefiles. Passed
# through (when present) on top of an otherwise-minimal subprocess env so that
# 'module avail'/'module load' actually resolve the site's module tree. Sourcing
# the init scripts is not enough on its own: without MODULEPATH the shell gets a
# working `module` function pointed at nothing.
MODULE_ENV_PASSTHROUGH = (
    "MODULESHOME", "MODULEPATH", "MODULEPATH_ROOT",
    "LMOD_CMD", "LMOD_DIR", "LMOD_PKG", "LMOD_ROOT",
    "LMOD_SYSTEM_DEFAULT_MODULES", "LMOD_sys", "LMOD_arch", "LMOD_SYSHOST",
)


def module_env(base: dict | None = None) -> dict:
    """*base* (default: the ambient env) plus every module var that is actually set.

    Returns a new dict; the input is never mutated. A var absent from the ambient
    environment is absent from the result rather than being set to "", because Lmod
    treats an empty MODULEPATH differently from an unset one.
    """
    env = dict(base if base is not None else os.environ)
    for name in MODULE_ENV_PASSTHROUGH:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env


def module_probe_script(module_cmd: str) -> str:
    """A shell script that makes `module` available, then runs *module_cmd*.

    The init block is silenced, not the command: output from these scripts is
    *parsed*, and a chatty site profile would otherwise prepend noise to the payload.
    The command keeps its own streams — `module -t avail 2>&1` deliberately wants
    stderr, since Lmod prints its MODULEPATH section headers there.
    """
    init = """
if ! type module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
  [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
fi >/dev/null 2>&1
""".strip()
    return f"{init}\n{module_cmd}"
