# MIMIR Chat — VS Code Extension

Chat with your local [MIMIR](https://github.com/MIMIR-LLM4CSE/MIMIR) agent directly
inside VS Code: streaming responses, tool-status indicators, approval dialogs,
and model / cluster selection.

## Install

**From a packaged `.vsix`:**

```bash
code --install-extension mimir-0.1.0.vsix
```

**Build and package from source:**

```bash
cd mimir/vscode-extension
npm install
npm run build          # compile + bundle (TypeScript + webview)
npm run package        # produces mimir-<version>.vsix
```

**Dev mode:** open `mimir/vscode-extension` in VS Code and press `F5` (Run Extension).

## Configure

The extension starts (or connects to) the MIMIR WebSocket server. Set the paths
and backend in your workspace `.vscode/settings.json` — see the
[Setup Guide](https://github.com/MIMIR-LLM4CSE/MIMIR/blob/master/SETUP.md) §6 for the
full list of `mimir.*` settings (Python path, backend, SLURM/cluster config,
model lists).

Minimal local setup:

```jsonc
{
  "mimir.mimirPath": "/path/to/codes",
  "mimir.pythonPath": "/path/to/python",
  "mimir.backend": "vllm",
  "mimir.availableModels": ["qwen3:8b"]
}
```
