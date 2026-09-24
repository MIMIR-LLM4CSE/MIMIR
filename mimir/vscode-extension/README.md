# MIMIR Chat — VS Code Extension

A chat panel for the [MIMIR](https://github.com/MIMIR-LLM4CSE/MIMIR) agent. MIMIR plans,
edits, runs and verifies code in your workspace. It uses the model server you
already run: vLLM, Ray Serve, Ollama, or the hosted Claude API.

| You want to… | Where |
|---|---|
| Install or update the extension | [Install](#install) |
| Connect to your model server | [Connect](#connect) |
| Know what the panel does | [Features](#features) |
| Change a default | [Settings](#settings) |
| See what changed between versions | [CHANGELOG.md](https://github.com/MIMIR-LLM4CSE/MIMIR/blob/main/CHANGELOG.md) |

## Install

Building the extension needs **Node.js ≥ 18 and npm** (`node -v`, `npm -v`). If
they are missing: `sudo apt install nodejs npm` (Debian/Ubuntu), `sudo dnf
install nodejs npm` (RHEL/Fedora), `brew install node` (macOS), or the installer
from [nodejs.org](https://nodejs.org/en/download) (Windows). Without root — on a
cluster node, say — install [nvm](https://github.com/nvm-sh/nvm) and run `nvm
install --lts`, which puts Node and npm under your home directory. Distribution
packages are sometimes older than Node 18; check `node -v`. Full instructions,
including the VS Code terminal caveat, are in
[SETUP.md §6a](https://github.com/MIMIR-LLM4CSE/MIMIR/blob/main/SETUP.md#6a-get-nodejs-and-npm).

```bash
cd mimir/vscode-extension
npm install
npm run deploy
```

`npm run deploy` builds the bundle. It installs the extension the first time, and
updates it in place after that. Reload the window when it is done
(`Ctrl+Shift+P` → *Developer: Reload Window*).

`./install.sh` at the repo root already runs this for you.

Other scripts: `npm run dev` rebuilds on save, `npm test` runs the unit tests, and
`npm run package` builds a `.vsix` without installing it. To work on the extension
itself, open `mimir/vscode-extension` in VS Code and press `F5`.

## Connect

Open the MIMIR panel in the secondary side bar. The Connect form asks for three things:

| Backend | Address | Model list read from |
|---|---|---|
| vLLM | `http://<host>:8000` | `/v1/models` |
| Ray Serve | `http://<host>:8000`, plus the app's route prefix if it has one | `/v1/models` |
| Ollama | `http://<host>:11434` | `/api/tags` |
| Anthropic (Claude) | none: paste an API key, or export `ANTHROPIC_API_KEY` | `mimir.anthropicAvailableModels` |

The model dropdown fills itself from the address, and ⟳ reads it again. When the
address does not answer, the form says why in plain words (no server on that port,
untrusted certificate, wrong scheme, name not found…) and what to try.

Tick **Remember this address** and the next window reconnects on its own at startup.
It checks that the address answers first; if not, you land on the form, pre-filled.
An API key is never stored.

Press **Connect**. The extension starts the MIMIR server itself, one per window on a
free port, so several windows run side by side. No `.vscode/settings.json` is needed.

## Features

### Working with the agent

| Feature | What it does |
|---|---|
| Modes | **agent** explores, edits, runs and checks. **plan** only reads, then proposes a plan you approve before any change. **ask** answers questions about the code without changing it. The mode colours the chat and applies from the next step, even mid-turn. |
| Approval modes | **manual**: every sensitive call and every path outside the workspace asks you. **auto**: sensitive tools run unasked, leaving the workspace still asks. **all**: nothing asks. The shell denylist and the workspace sandbox apply in every mode. |
| Approval cards | One card per call, listing every outside path it names. Allow, always allow, or deny. A denial tells the agent what you meant; it does not retry the same thing. |
| Batch review | Collects the file changes of a turn into one review bar, with inline diffs, instead of one card per write. |
| Questions from the agent | When a choice is yours to make, the agent asks it as a short form in the chat. |
| Steering | Type while the agent works. The message reaches it at the next step. |
| Message history | `↑` on the first line of the input recalls a message you sent. |
| Reconnect | A turn keeps running if the panel loses its connection. The work it did comes back when the panel reconnects. |
| `@` mentions | Attach a workspace file, a line range, or a server resource to your message. |
| `/` commands | Run a skill by name, such as `/fix-bug`. |
| Sub-agents | Work the agent hands to a sub-agent shows as its own group of rows. |

### Seeing what happens

| Feature | What it does |
|---|---|
| Streaming | Answers and reasoning stream as they are written. Reasoning sits in a collapsible block. |
| Tool rows | One row per call. File names are links: a click opens the file on the lines that were read or edited. |
| Terminal panel | A command shows its input as soon as it starts, and its output when it ends. |
| Long runs | A build or job shows its current phase, and a progress bar when the tool counts its own progress. A run whose row scrolled away stays visible as a small card in the corner; a click brings you back to it. |
| Background runs | A running command can be sent to the background with one click. The turn goes on, and the row settles when the run ends. |
| Plan and todo panel | The current plan and its checklist, updated as steps close. An unfinished checklist is offered again when you reopen the session. |
| Verification | Under each answer, a collapsed line says what was checked and what was not. |
| Context bar | How full the model's context window is. |
| Notifications | A VS Code notification when a task ends or MIMIR needs you while the panel is hidden. |

### Sessions and tuning

| Feature | What it does |
|---|---|
| Sessions | Switch, rename or delete past conversations. Each one gets a one-sentence description. |
| Thinking depth | From off to max. The default, *auto*, lets the model spend reasoning only where the task needs it. |
| Enforcement | How much guidance the agent gets: strict, light (default) or off. Safety and verification checks stay on at every level. |
| Context memory | Compact history for small models, or full history for large-context ones. |
| Servers and skills | Hide a tool server or a skill from the model. The choice is kept across sessions. |

## Settings

Every setting has a working default. Most only set the value the Connect form starts on.

| Setting | Default | Purpose |
|---|---|---|
| `mimir.backend` | `vllm` | Backend the form starts on |
| `mimir.vllmBaseUrl` | `http://127.0.0.1:8000` | vLLM address the form starts on |
| `mimir.rayBaseUrl` | `http://127.0.0.1:8000` | Ray Serve address the form starts on |
| `mimir.ollamaUrl` | `http://127.0.0.1:11434` | Ollama address the form starts on |
| `mimir.vllmVerifySsl` | `true` | Untick for an HTTPS vLLM or Ray endpoint signed by a private CA |
| `mimir.maxModelLen` | `0` | Context window (tokens) when the endpoint does not report `max_model_len`. `0` keeps the reported window, else 200000 |
| `mimir.anthropicAvailableModels` | current Claude models | Models offered for the Anthropic backend |
| `mimir.pythonPath` | empty (auto) | Python that runs the server. See below |
| `mimir.wsUrl` | empty (auto) | Attach to a server you started yourself; the extension then starts none |
| `mimir.notifications.enabled` | `true` | Notify when a task ends off-screen |

**Which Python runs the server.** The extension looks in this order:
`mimir.pythonPath`, then the `MIMIR_PYTHON` environment variable, then
`~/.mimir/python` (written by `install.sh`), then `python3` on `PATH`.
`~/.mimir/python` points at a launcher that starts the venv installed for this
machine's platform. So one home directory shared by several kinds of machines
still starts the right Python. If this platform has no install, the *MIMIR Server*
output says so: run `install.sh` on this machine.

## Commands

| Command | What it does |
|---|---|
| `MIMIR: Focus Chat` | Opens and focuses the chat panel |
| `MIMIR: Start Server` | Starts the MIMIR server for this window by hand |

## More

- [Setup Guide](https://github.com/MIMIR-LLM4CSE/MIMIR/blob/main/SETUP.md), §6 for the extension
- [Frontend internals](https://github.com/MIMIR-LLM4CSE/MIMIR/blob/main/EXTENSION_DETAILED.md)
- [Issues](https://github.com/MIMIR-LLM4CSE/MIMIR/issues)
