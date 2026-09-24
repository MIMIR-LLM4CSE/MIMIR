#!/usr/bin/env bash
# MIMIR installer (Linux / macOS).
#
# Creates a virtual environment and installs the `mimir` package with its
# console scripts (`mimir`, `mimir-server`). Re-runnable and idempotent.
#
# Then builds and installs the VS Code extension when npm is available.
#
#   ./install.sh                 # portable Python 3.10, into ./.venv-<os>-<arch>
#   PYTHON=python3.11 ./install.sh         # use this interpreter instead
#   MIMIR_VENV=~/envs/mimir ./install.sh
#   MIMIR_EXTRAS="vllm,dev" ./install.sh   # adds pytest + ruff
#   MIMIR_EXTRAS="ray" ./install.sh        # same deps, for a Ray Serve endpoint
#   MIMIR_SKIP_EXTENSION=1 ./install.sh    # Python only
set -euo pipefail

# What a venv is tied to: the OS and the CPU, not the host. With the portable
# interpreter below, one venv serves every Linux x86_64 machine whatever its
# distribution, so machines sharing a home directory share one install.
mimir_platform() {
  printf '%s-%s\n' "$(uname -s | tr '[:upper:]' '[:lower:]')" "$(uname -m)"
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
platform="$(mimir_platform)"
venv="${MIMIR_VENV:-$here/.venv-$platform}"
extras="${MIMIR_EXTRAS:-vllm}"
state_home="${MIMIR_STATE_HOME:-$HOME/.mimir}"

# --- Python -----------------------------------------------------------------
# By default, a standalone CPython fetched by uv. A Python compiled on the machine
# links that machine's libc, libcrypt and OpenSSL, and stops at the first older
# node ("libcrypt.so.2: cannot open shared object file"). The standalone build
# needs glibc >= 2.17 and carries its own libraries. PYTHON=... skips all this.
if [ -n "${PYTHON:-}" ]; then
  py="$PYTHON"
else
  uv="$(command -v uv || true)"
  if [ -z "$uv" ]; then
    uv="$state_home/bin/uv"
    if [ ! -x "$uv" ]; then
      echo "==> Installing uv into $state_home/bin"
      command -v curl >/dev/null 2>&1 \
        || { echo "error: curl is needed to fetch uv. Or set PYTHON=... (Python >= 3.10)." >&2; exit 1; }
      curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="$state_home/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null
    fi
  fi
  # System certificates: behind a TLS-inspecting proxy, uv's bundled roots reject
  # the download that curl just accepted. (UV_NATIVE_TLS is the older uv's name.)
  # No link in ~/.local/bin: the interpreter is for MIMIR, not for the user's PATH.
  export UV_PYTHON_INSTALL_DIR="$state_home/pythons" UV_PYTHON_INSTALL_BIN=0 \
         UV_SYSTEM_CERTS=1 UV_NATIVE_TLS=1
  echo "==> Fetching a portable Python 3.10"
  "$uv" python install 3.10
  py="$(UV_PYTHON_PREFERENCE=only-managed "$uv" python find 3.10)"
fi
if ! command -v "$py" >/dev/null 2>&1; then
  echo "error: '$py' not found. Install Python >= 3.10 or set PYTHON=..." >&2
  exit 1
fi
ver="$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
"$py" - <<'PYEOF' || { echo "error: Python >= 3.10 is required." >&2; exit 1; }
import sys
raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)
PYEOF

echo "==> Using Python $ver ($py)"

# --- Virtualenv -------------------------------------------------------------
# A venv only works with the interpreter it was made from. Reusing one built on
# another Python would install into it and then fail on the first import.
base_prefix() { "$1" -c 'import sys; print(sys.base_prefix)' 2>/dev/null; }
if [ -d "$venv" ] && [ "$(base_prefix "$venv/bin/python")" != "$(base_prefix "$py")" ]; then
  echo "error: $venv was made with another Python (or cannot run here)." >&2
  echo "       Remove it and run ./install.sh again." >&2
  exit 1
fi
if [ ! -d "$venv" ]; then
  echo "==> Creating virtualenv at $venv"
  "$py" -m venv "$venv"
fi
venv="$(cd "$venv" && pwd)"
# shellcheck disable=SC1091
source "$venv/bin/activate"

# --- Install ----------------------------------------------------------------
echo "==> Installing MIMIR (extras: $extras)"
pip install --upgrade pip >/dev/null
if [ -n "$extras" ]; then
  pip install "$here[$extras]"
else
  pip install "$here"
fi

# --- Record the interpreter for the VS Code extension ------------------------
# The extension spawns the WS server through `bash -c`, which sources no profile,
# so `python3` from its PATH is rarely this venv. install.sh leaves a launcher
# path in $state_home/python, and the extension finds it on its own.
#
# The home directory is often shared by machines that cannot run the same
# binaries. So each platform records its own venv in python.d/<os>-<arch>, and the
# launcher starts the one matching the machine it runs on. Written before the
# smoke test, so an interrupted check does not leave the extension on the system
# Python.
launcher="$state_home/bin/python"
if mkdir -p "$state_home/python.d" "$state_home/bin" 2>/dev/null; then
  printf '%s\n' "$venv/bin/python" > "$state_home/python.d/$platform"
  cat > "$launcher" <<EOF
#!/bin/sh
# Written by MIMIR's install.sh; run it again rather than editing this file.
# Starts the interpreter install.sh recorded for this machine's platform.
key="\$(uname -s | tr '[:upper:]' '[:lower:]')-\$(uname -m)"
target="\$(cat "$state_home/python.d/\$key" 2>/dev/null)"
if [ -z "\$target" ] || [ ! -x "\$target" ]; then
  echo "MIMIR: no interpreter installed for \$key. Run ./install.sh on this machine." >&2
  exit 127
fi
exec "\$target" "\$@"
EOF
  chmod +x "$launcher"
  printf '%s\n' "$launcher" > "$state_home/python"
  echo "==> Interpreter recorded for $platform in $state_home/python.d"
else
  echo "warning: could not write $state_home/python — set MIMIR_PYTHON or" >&2
  echo "         mimir.pythonPath if the extension picks the wrong interpreter." >&2
fi


# --- Smoke test -------------------------------------------------------------
echo "==> Verifying installation"
# `mimir` has no --help: any invocation opens a chat session that waits on the
# terminal, so only check that the console script is on PATH.
command -v mimir >/dev/null || { echo "error: the 'mimir' command was not installed." >&2; exit 1; }
python - <<'PYEOF'
import importlib, os
import mimir.client.config.constants as c
missing = [n for n, p in c.SERVERS.items() if not os.path.exists(p)]
assert not missing, f"missing server scripts: {missing}"
print("   all %d MCP server scripts resolve" % len(c.SERVERS))
PYEOF
# What the extension will actually start: the launcher, not this venv directly.
if [ -x "$launcher" ]; then
  "$launcher" -c 'import mimir' \
    || { echo "error: $launcher does not start this install." >&2; exit 1; }
  echo "   launcher starts this venv on $platform"
fi

# --- Dev tooling ------------------------------------------------------------
# Named only when the extra was requested: pointing at `pytest`/`ruff` when they
# were never installed sends the user to a command-not-found.
case ",$extras," in
  *,dev,*)
    echo "==> Dev tooling"
    echo "   $(pytest --version 2>&1 | head -1)"
    echo "   ruff $(ruff --version 2>&1 | awk '{print $2}')"
    ;;
esac

# --- VS Code extension ------------------------------------------------------
# Never fatal: the Python install above is complete and usable on its own, so a
# missing npm or a failed build costs the user the sidebar, not the agent.
ext_installed=0
ext_dir="$here/mimir/vscode-extension"
if [ "${MIMIR_SKIP_EXTENSION:-0}" = "1" ]; then
  echo "==> Skipping VS Code extension (MIMIR_SKIP_EXTENSION=1)"
elif ! command -v npm >/dev/null 2>&1; then
  echo "==> Skipping VS Code extension (npm not found)"
  echo "   Install Node.js >= 18 (it ships npm), then re-run ./install.sh."
  echo "   No root? curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash"
  echo "            exec \$SHELL -l && nvm install --lts"
  echo "   Details: SETUP.md section 6a."
elif [ ! -d "$ext_dir" ]; then
  echo "==> Skipping VS Code extension (not found at $ext_dir)"
else
  echo "==> Installing the VS Code extension"
  if (cd "$ext_dir" && npm install --silent && npm run deploy); then
    ext_installed=1
  else
    echo "   extension install failed — MIMIR itself is installed. Retry with:"
    echo "     cd $ext_dir && npm install && npm run deploy"
  fi
fi

cat <<EOF

MIMIR installed. To use it:

  source "$venv/bin/activate"
  cd /path/to/your/project
  mimir                      # interactive CLI
EOF

if [ "$ext_installed" = "1" ]; then
  cat <<'EOF'
  # or, in VS Code: reload the window, open the MIMIR panel, and enter the
  # address of your running vLLM, Ray Serve or Ollama server (e.g. http://127.0.0.1:8000).
EOF
else
  cat <<'EOF'
  # or start the WS server for the VS Code extension:
  mimir-server --host 0.0.0.0 --port 8765
EOF
fi

cat <<'EOF'

MIMIR talks to an LLM server you already run (vLLM, Ray Serve, Ollama, or the
Claude API).
Its address is entered in the MIMIR panel — see SETUP.md.
EOF

case ",$extras," in
  *,dev,*)
    cat <<'EOF'
Dev checks, from the repo root:

  pytest                     # test suite
  ruff check .               # lint
EOF
    ;;
esac
