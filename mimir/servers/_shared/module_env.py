"""Shared helpers for Environment Modules / Lmod shell initialization."""


def module_shell_script(module_cmd: str) -> str:
    init = """
if ! type module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
  [ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
fi
""".strip()
    return f"{init}\n{module_cmd}"
