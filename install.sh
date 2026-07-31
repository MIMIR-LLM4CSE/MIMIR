#!/usr/bin/env bash
# MIMIR installer (Linux / macOS).
#
# Creates a virtual environment and installs the `mimir` package with its
# console scripts (`mimir`, `mimir-server`). Re-runnable and idempotent.
#
#   ./install.sh                 # install into ./.venv
#   MIMIR_VENV=~/envs/mimir ./install.sh
#   MIMIR_EXTRAS="vllm,dev" ./install.sh
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

cat <<EOF

MIMIR installed. To use it:

  source "$venv/bin/activate"
  cd /path/to/your/project
  mimir                      # interactive CLI
  # or start the WS server for the VS Code extension:
  mimir-server --host 0.0.0.0 --port 8765

Pick an LLM backend first (Ollama or vLLM) — see SETUP.md.
EOF
