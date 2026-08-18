# MIMIR — Setup Guide

> **MIMIR docs** — [Overview](README.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Policy](POLICY.md) · [Client internals](CLIENT_DETAILED.md) · [Servers](SERVERS_DETAILED.md) · [Extension](EXTENSION_DETAILED.md) · [Plugins](PLUGINS_DETAILED.md)

Step-by-step instructions to go from a fresh clone to a running agent.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.10 | Any CPython distribution |
| Git | For cloning and the GitHub MCP server |
| Node.js ≥ 18 + npm | Only for the VS Code extension |
| Docker | Optional — only for the container image (§2c) |
| An LLM backend | Ollama **or** a running vLLM endpoint (see §3) |

---

## 2. Install MIMIR

MIMIR is a normal Python package. Installing it pulls in everything the agent
**and all of its MCP servers** need and exposes two console commands: `mimir`
(interactive CLI) and `mimir-server` (WebSocket server for the VS Code extension).

### 2a. Quick install (recommended)

The convenience script creates a virtualenv, installs the package with the `vllm`
extra, and runs a smoke test:

```bash
git clone https://github.com/MIMIR-LLM4CSE/MIMIR.git
cd MIMIR
./install.sh                 # -> .venv, `mimir` + `mimir-server` commands
source .venv/bin/activate
```

Override the target venv or extras if needed:

```bash
MIMIR_VENV=~/envs/mimir MIMIR_EXTRAS="vllm,dev" ./install.sh
```

### 2b. Manual pip install

```bash
cd /path/to/mimir            # repo root (contains pyproject.toml)
python -m venv .venv && source .venv/bin/activate
pip install .                # or:  pip install ".[vllm]"
# editable dev install + test tooling:
pip install -e ".[dev]"
```

What the base install covers:

| Group | Packages | Used by |
|-------|----------|---------|
| Core client + MCP | `mcp`, `httpx` | agent loop, all server transports |
| LLM backends | `ollama`, `openai` | Ollama backend, vLLM (OpenAI-compatible) backend |
| WebSocket frontend | `websockets` | `ws_server.py` (VS Code extension) |
| Scientific / utility servers | `numpy`, `sympy`, `psutil` | `math`, `symbolic_math`, `platform`, `benchmark`, `system` servers |

Optional extras (declared in `pyproject.toml`):

- **`finetune`** — the `torch` / `transformers` / `peft` stack, only needed for the
  `ml/` servers (`server_finetune.py`). Without it that one server fails to start
  with an `ImportError`; the rest of the agent is unaffected.
  ```bash
  pip install ".[finetune]"
  ```
- **`dev`** — `pytest` for running the test suite (`pytest` from the repo root; the
  tests are plain `unittest` cases, so `python -m unittest discover mimir/tests` also
  works without this extra).
- **Code-intelligence & validation binaries.** The `code_intel` server uses external
  language servers / tools when present (`pyright`/`pylsp`, `clangd`, `ctags`) and
  degrades to a text scan when absent. Code validation runs through the `bash` server:
  install whichever validators you want on PATH — Python (`ruff`, `mypy`, `pyflakes`,
  `black`, `pytest`), the compilers (`gcc`/`g++`/`gfortran`/`nvcc`/`javac`), and CMake
  (`cmake`/`ctest`) for C/C++/CUDA projects. These are system/CLI tools, not Python
  packages — the agent states a check as unrun when its tool is absent.

> **Legacy path.** `pip install -r mimir/requirements.txt` still works and installs
> the same core dependencies, but it does **not** register the `mimir` /
> `mimir-server` commands — prefer `pip install .`.

### 2c. Docker (reproducible, no local Python setup)

```bash
docker build -t mimir:latest .

# Interactive CLI (mount the project you want to work on as the sandbox):
docker run --rm -it -v "$PWD":/workspace \
    -e LLM_BACKEND=vllm -e VLLM_BASE_URL=http://<node>:8000 \
    mimir:latest mimir

# WebSocket server for the VS Code extension:
docker compose up mimir            # or: docker run ... -p 8765:8765 mimir:latest
```

The container supports the vLLM **connect** mode (attach to an already-running
endpoint) and Ollama over the network. The SLURM **launch** mode is *not* available
inside a container — it needs SSH access to a login node, so run that on a frontal
node with the CLI / extension instead. See `docker-compose.yml` for the available
environment variables.

---

## 3. Choose and Configure an LLM Backend

Pick **one** backend: vLLM (OpenAI-compatible, GPU-served — best for HPC/SLURM, and the
default) or Ollama (local, simplest for interactive use — set `LLM_BACKEND=ollama`).

### vLLM (OpenAI-compatible endpoint)

vLLM can be used in **two ways**, selectable in the Connect panel via **vLLM mode**
(`mimir.vllmMode`):

#### Option A — Launch on compute node (mode `launch`)

The extension allocates a SLURM node and runs `vllm serve` there for you — use this
when no vLLM is running yet. The agent process runs inside the SLURM job, so the GPU is
fully available to the model.

This path is **driven from the VS Code extension** (see §6). You provide, in
`settings.json`:

```jsonc
{
  "mimir.backend": "vllm",
  "mimir.vllmMode": "launch",
  "mimir.slurmEnabled": true,
  "mimir.loginNode": "your-login-node",
  "mimir.vllmPath": "vllm",                       // vllm binary on the compute node
  "mimir.vllmSetupScript": "/path/to/vllm/setup.sh",  // sourced before launch (CUDA modules / venv)
  "mimir.vllmModelsDir": "/path/to/models",       // root for relative model paths
  "mimir.vllmAvailableModels": [
    "mistralai/devstral-small-2-24b",
    "meta/llama3-70b"
  ],
  "mimir.vllmExtraArgs": "--tensor-parallel-size 1",  // appended to the generated serve command
  "mimir.clusterConfig": [ /* node-type form — see §6c */ ]
}
```

Then pick the model in the Connect panel and launch — the extension SSHes to the login
node, runs `salloc`, sources `vllmSetupScript`, starts `vllm serve <model>` (with the
right tool-call / reasoning parser from `vllm_model_profiles.json`), and connects the
WebSocket server once vLLM is ready. Use `mimir.vllmServeCommand` to fully override
the generated command if you need to.

#### Option B — Connect to running server (mode `connect`)

vLLM is already running somewhere reachable; the agent just points at its address. No
SLURM allocation is made — the WebSocket server runs on the frontal.

First start (or reuse) vLLM on the serving node, binding to all interfaces so the
frontal can reach it:

```bash
source /path/to/vllm/setup.sh   # load CUDA modules / venv
vllm serve /path/to/model \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
    # add --reasoning-parser deepseek_r1 for thinking-capable models
```

Then point the agent at it. In the webview, pick vLLM → **Connect to running server**
and enter the address — no model selection needed; the agent uses whatever the endpoint
serves (resolved from `/v1/models`). Headless / CLI equivalent (run on the frontal):

```bash
mimir-server \
    --backend vllm \
    --vllm-base-url http://<node-hostname>:8000
    # `python -m mimir.client.ui.ws.ws_server ...` also works if not pip-installed
    # --model is optional here; omit it to auto-select the served model
```

…or via environment variables:

```bash
export LLM_BACKEND=vllm
export VLLM_BASE_URL=http://<node-hostname>:8000   # use the node hostname, NOT 127.0.0.1
export VLLM_API_KEY=EMPTY      # leave as EMPTY unless your endpoint requires auth
```

> Pass the explicit node hostname rather than `127.0.0.1`/`localhost`: the client only
> rewrites loopback addresses to the local hostname (an HPC-proxy workaround), so a real
> hostname is forwarded untouched. Verify reachability first with
> `curl http://<node-hostname>:8000/v1/models`.

Per-model tool-call parser and chat-template overrides live in
`mimir/client/config/vllm_model_profiles.json` — add a new entry there if you
need to override the inferred parser or set `chat_template_kwargs` (e.g. `enable_thinking`).

### Ollama (local, simplest for interactive use)

1. Install Ollama: https://ollama.com/
2. Pull a model that fits your VRAM:

   ```bash
   ollama pull qwen3:8b                        # ~5 GB  — lightweight
   ollama pull qwen2.5-coder:32b               # ~19 GB
   ollama pull devstral-small-2:24b-instruct-2512-fp16   # ~48 GB
   ollama pull nemotron-3-super:120b            # ~86 GB — best results
   ```

3. Confirm Ollama is running:

   ```bash
   ollama list        # should show pulled models
   curl http://127.0.0.1:11434/api/tags    # API health check
   ```

`OLLAMA_BASE_URL` can point at a remote host (or a SLURM-launched Ollama via the
extension's `mimir.ollamaSetupScript`); it defaults to `http://127.0.0.1:11434`.

---

## 4. Environment Variables

Set these in your shell profile (`.bashrc`, `.zshrc`) or in the SLURM job script.

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BACKEND` | `vllm` | `vllm` or `ollama` |
| `MIMIR_DEFAULT_MODEL` | *(empty)* | Model selected at startup; overridden by `--model` flag or the UI |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000` | vLLM API base URL |
| `VLLM_API_KEY` | `EMPTY` | API key for vLLM calls |
| `MIMIR_EMBED_MODEL` | *(empty; `nomic-embed-text` on Ollama)* | Embedding model for semantic memory search & tool ranking. **Required for vLLM** (the served model name, e.g. `BAAI/bge-m3`). Empty + vLLM ⇒ semantic path disabled, lexical fallback used. |
| `MIMIR_EMBED_BASE_URL` | *(falls back to `VLLM_BASE_URL`)* | Serve embeddings from a separate endpoint than the chat model (vLLM only) |
| `MIMIR_EMBED_TIMEOUT` | `10` | HTTP timeout (seconds) for the vLLM embeddings call |
| `MCP_FILES_ROOT` | current working directory | Workspace root. Paths a tool *names* are confined to it, and reaching outside prompts for approval — but a program the agent runs (`python`/`make`/`gcc`) is not itself constrained, so this is a guardrail on intent, not a sandbox ([scope](SERVERS_DETAILED.md#scope-of-the-sandbox-read-this-before-trusting-confined)) |
| `GITHUB_TOKEN` | *(none)* | Raises GitHub API rate limit from 60 to 5 000 req/h |
| `MIMIR_SYSTEM_PROMPT_FILE` | *(none)* | Path to a `.md` file that replaces the agent's general context (base system prompt). See below. |

### Custom general context (base system prompt)

The agent's "general context" — its base system prompt — defaults to a built-in
profile, but you can replace it with your own `.md` file (e.g. for a different
persona or domain). The first match wins:

1. **`MIMIR_SYSTEM_PROMPT_FILE`** — an absolute path to any `.md` file, anywhere on
   disk. Needs no `.mimir/` directory; best for ad-hoc use.
2. **`.mimir/system_prompt.md`** in the workspace root. Nothing is auto-created —
   create this file yourself, using
   [`mimir/examples/system_prompt.md`](mimir/examples/system_prompt.md) as a template.
3. Otherwise the built-in default is used.

When a file is found, its content **fully replaces** the base prompt; the dynamic
platform / memory / todo / plan sections are still appended automatically.

The built-in default is structured as markdown sections (`## Non-negotiables`,
`## Workflow`, `## Validation`, …) with one instruction per line, which is what
mid-size open-weight models follow most reliably — an override is worth writing the
same way rather than as a single block of prose.

---

## 5. Run the CLI Agent

```bash
cd /path/to/your/project          # becomes MCP_FILES_ROOT
mimir                             # console command (after `pip install .`)
```

> Not installed as a package? The module form still works from the repo root:
> `python -m mimir.client.ui.cli.main`.

Useful flags:

```bash
mimir --model nemotron-3-super:120b
mimir --mode plan                 # reasoning-only, no tool calls
```

Type `/help` at the prompt for the in-session commands (`/mode`, `/rescan`, `/think`,
`/enforcement`, `/servers`, `/skills`, `/context`, `/trust`, …); `quit` exits.

---

## 6. VS Code Extension (Recommended)

The extension provides a chat panel with model/cluster selection, streaming output,
approval dialogs, and tool-status indicators.

### 6a. Build the Extension

```bash
cd mimir/vscode-extension
npm install
npm run build          # compiles TypeScript + bundles the webview
```

### 6b. Install into VS Code

**Option 1 — Dev mode (no packaging needed):**

Open `mimir/vscode-extension` in VS Code and press `F5` (Run Extension).

**Option 2 — Package and install:**

```bash
npm run package                    # produces mimir-<version>.vsix (uses @vscode/vsce)
code --install-extension mimir-*.vsix
```

### 6c. Configure in `.vscode/settings.json`

These settings are user/platform-specific and must be added to the workspace
`.vscode/settings.json` (or VS Code User settings). They are **not** committed
with the extension source.

```jsonc
{
  // ── Paths (adjust to your environment) ──────────────────────────────────
  "mimir.mimirPath": "/path/to/codes",
  "mimir.pythonPath":   "/path/to/conda/envs/myenv/bin/python",

  // ── LLM backend ──────────────────────────────────────────────────────────
  // "mimir.backend": "vllm",             // "vllm" (default) or "ollama"
  // "mimir.vllmBaseUrl": "http://127.0.0.1:8000",
  // "mimir.vllmMode":   "launch",        // "launch" (start vllm serve on a SLURM node)
  //                                         //   or "connect" (attach to vllmBaseUrl, no SLURM)

  // ── Model lists (shown in the Connect dropdown) ─────────────────────────
  "mimir.availableModels": [
    "qwen3:8b",
    "qwen2.5-coder:32b",
    "devstral-small-2:24b-instruct-2512-fp16"
  ],
  // vllmModelsDir is REQUIRED when using relative model paths.
  // Relative entries in vllmAvailableModels are prefixed with this path.
  // Must be set here in settings.json — VS Code cannot read shell env vars.
  // Absolute paths in vllmAvailableModels are always used as-is.
  "mimir.vllmModelsDir": "/path/to/models",
  "mimir.vllmAvailableModels": [
    "mistralai/devstral-small-2-24b",
    "meta/llama3-70b"
  ],

  // ── VRAM hints (shown next to each model in the dropdown) ───────────────
  "mimir.modelSizes": {
    "qwen3:8b": 5,
    "qwen2.5-coder:32b": 19,
    "devstral-small-2:24b-instruct-2512-fp16": 48,
    "devstral-small-2-24b": 48
  },

  // ── SLURM / cluster (omit if running locally without SLURM) ─────────────
  "mimir.slurmEnabled": true,
  "mimir.loginNode": "your-login-node",
  "mimir.ollamaSetupScript": "/path/to/setup_ollama.sh",
  "mimir.vllmSetupScript":   "/path/to/vllm/setup.sh",

  "mimir.clusterConfig": [
    {
      "name": "my-cluster",
      "loginNode": "login-1",
      "account": "my-account",
      "nodeTypes": [
        {
          "label": "GPU node · 1 day",
          "partition": "gpu",
          "cpusPerNode": 32,
          "gpu": { "type": "a100", "memGB": 80, "maxCount": 1 },
          "memOptionsGB": [64, 128, 256]
        }
      ]
    }
  ]
}
```

> **Tip:** `clusterConfig` is required for SLURM-managed Ollama/vLLM launches.
> Without it the Connect panel shows a simplified model + URL form (local mode).

---

## 7. SLURM / HPC Workflow

When `slurmEnabled` is `true` and `clusterConfig` is set, the extension:

1. SSH-es to the `loginNode`.
2. Runs `salloc` with the SLURM args assembled from the node-type form.
3. Sources `ollamaSetupScript` (or `vllmSetupScript`) on the allocated node.
4. Starts Ollama / vLLM and connects the WebSocket server.
5. Opens the chat panel once the backend is ready.

The agent process itself runs inside the SLURM job, so GPU resources are fully
available to the model.

---

## 8. GitHub Token (Optional but Recommended)

The `github` MCP server uses the GitHub REST API. Without a token it is rate-limited
to 60 requests/hour.

```bash
export GITHUB_TOKEN=ghp_...
```

Or add it to the SLURM job script / shell profile. The server reads it from the
environment automatically — no config file needed.

---

## 9. Quick Smoke-Test

```bash
cd /tmp/test-project && mkdir -p src
mimir --model nemotron-3-super:120b   # or: python -m mimir.client.ui.cli.main --model <model>
# At the prompt:
> list the files here
```

If you see tool calls being dispatched and a final answer returned, everything is
working correctly.
