#!/usr/bin/env bash
# Install the freshly built extension into VS Code, or update it in place.
#
# Run through `npm run deploy` (which builds first). Two cases:
#   • already installed → copy the new bundle over it (fast; needs a window reload)
#   • not installed yet → package a .vsix and install it with the `code` CLI
#
# Covers both a local VS Code (~/.vscode/extensions) and a Remote-SSH session
# (~/.vscode-server/extensions), which is where MIMIR usually runs.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

# --- Locate an existing install --------------------------------------------
ext=""
for root in "$HOME/.vscode-server/extensions" "$HOME/.vscode/extensions"; do
  found="$(ls -d "$root"/mimir.mimir-* 2>/dev/null | head -1 || true)"
  if [ -n "$found" ]; then
    ext="$found"
    break
  fi
done

# --- Update in place --------------------------------------------------------
if [ -n "$ext" ]; then
  mkdir -p "$ext/dist" "$ext/images"
  cp dist/extension.js dist/webview.js "$ext/dist/"
  cp images/* "$ext/images/"
  cp package.json "$ext/package.json"
  echo "==> Updated $ext"
  echo "    Reload the VS Code window to pick it up (Ctrl+Shift+P → Developer: Reload Window)."
  exit 0
fi

# --- First install ----------------------------------------------------------
echo "==> Not installed yet — packaging a .vsix"
# vsce shells out to npm to compute the package contents, so a bare `bash
# scripts/deploy.sh` outside `npm run deploy` fails obscurely without it.
if ! command -v npm >/dev/null 2>&1; then
  echo "error: npm is required to package the extension. Run 'npm run deploy'." >&2
  exit 1
fi
# vsce is a devDependency, so npm install has already put it here; npx is the
# fallback for a checkout where node_modules was pruned.
if [ -x node_modules/.bin/vsce ]; then
  node_modules/.bin/vsce package --no-dependencies >/dev/null
else
  npx --yes @vscode/vsce package --no-dependencies >/dev/null
fi
vsix="$(ls -t ./*.vsix | head -1)"

if ! command -v code >/dev/null 2>&1; then
  echo "error: packaged $here/$vsix but the 'code' command is not on PATH." >&2
  echo "       Install it by hand: VS Code → Extensions → ··· → Install from VSIX…" >&2
  echo "       (or enable the CLI: Ctrl+Shift+P → Shell Command: Install 'code' command in PATH)" >&2
  exit 1
fi

code --install-extension "$vsix" --force
echo "==> Installed $vsix"
echo "    Reload the VS Code window to pick it up (Ctrl+Shift+P → Developer: Reload Window)."
