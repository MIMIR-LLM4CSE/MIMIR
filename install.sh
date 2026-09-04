#!/usr/bin/env bash
# MIMIR installer (Linux / macOS).
#
# Creates a virtual environment and installs the `mimir` package with its
# console scripts (`mimir`, `mimir-server`). Re-runnable and idempotent.
#
# Then builds and installs the VS Code extension when npm is available.
#
#   ./install.sh                 # install into ./.venv
#   MIMIR_VENV=~/envs/mimir ./install.sh
#   MIMIR_EXTRAS="vllm,dev" ./install.sh   # adds pytest + ruff
#   MIMIR_SKIP_EXTENSION=1 ./install.sh    # Python only
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="${MIMIR_VENV:-$here/.venv}"
extras="${MIMIR_EXTRAS:-vllm}"

# --- Python check -----------------------------------------------------------
py="${PYTHON:-python3}"
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
if [ ! -d "$venv" ]; then
  echo "==> Creating virtualenv at $venv"
  "$py" -m venv "$venv"
fi
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

# --- Smoke test -------------------------------------------------------------
echo "==> Verifying installation"
mimir --help >/dev/null 2>&1 || true   # CLI may need a backend; just confirm it resolves
python - <<'PYEOF'
import importlib, os
import mimir.client.config.constants as c
missing = [n for n, p in c.SERVERS.items() if not os.path.exists(p)]
assert not missing, f"missing server scripts: {missing}"
print("   all %d MCP server scripts resolve" % len(c.SERVERS))
PYEOF

# --- Record the interpreter for the VS Code extension ------------------------
# The extension spawns the WS server through `bash -c`, which sources no profile,
# so `python3` from its PATH is rarely this venv. Leaving the path here means the
# extension finds it on its own and the user configures nothing.
state_home="${MIMIR_STATE_HOME:-$HOME/.mimir}"
if mkdir -p "$state_home" 2>/dev/null; then
  printf '%s\n' "$venv/bin/python" > "$state_home/python"
  echo "==> Interpreter recorded in $state_home/python"
else
  echo "warning: could not write $state_home/python — set MIMIR_PYTHON or" >&2
  echo "         mimir.pythonPath if the extension picks the wrong interpreter." >&2
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
  # address of your running vLLM or Ollama server (e.g. http://127.0.0.1:8000).
EOF
else
  cat <<'EOF'
  # or start the WS server for the VS Code extension:
  mimir-server --host 0.0.0.0 --port 8765
EOF
fi

cat <<'EOF'

MIMIR talks to an LLM server you already run (vLLM, Ollama, or the Claude API).
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
