# MIMIR Chat — VS Code Extension

Chat with your [MIMIR](https://github.com/MIMIR-LLM4CSE/MIMIR) agent directly
inside VS Code: streaming responses, tool-status indicators, approval dialogs,
and model selection.

## Install

```bash
cd mimir/vscode-extension
npm install
npm run deploy
```

`npm run deploy` builds the bundle, then installs the extension if it isn't
installed yet or updates it in place if it is. Reload the VS Code window
afterwards (`Ctrl+Shift+P` → *Developer: Reload Window*).

`./install.sh` at the repo root already runs this for you.

Other scripts: `npm run dev` (rebuild on save), `npm test` (unit tests),
`npm run package` (produce a `.vsix` without installing it).

**Dev mode:** open `mimir/vscode-extension` in VS Code and press `F5` (Run Extension).

## Configure

Nothing to configure by hand. Open the MIMIR panel, pick a backend, and enter the
address of the LLM server you already have running:

| Backend | Address |
|---|---|
| vLLM | `http://<host>:8000` |
| Ollama | `http://<host>:11434` |
| Anthropic (Claude) | no address — paste an API key, or export `ANTHROPIC_API_KEY` |

The model dropdown is filled from that address (`/v1/models` for vLLM,
`/api/tags` for Ollama), so there is no model list to maintain.

The interpreter that runs the WS server is found on its own: `install.sh` records the
venv's Python in `~/.mimir/python`. Override it with the `MIMIR_PYTHON` environment
variable or the `mimir.pythonPath` setting if you need to.

The remaining `mimir.*` settings only hold the defaults the panel starts on
(`mimir.backend`, `mimir.vllmBaseUrl`, `mimir.ollamaUrl`, `mimir.vllmVerifySsl`) — see
the [Setup Guide](https://github.com/MIMIR-LLM4CSE/MIMIR/blob/master/SETUP.md) §6.
