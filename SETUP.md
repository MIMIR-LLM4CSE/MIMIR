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
| An LLM server | A running vLLM or Ollama endpoint you can reach over HTTP, or an Anthropic API key (see §3) |

---

## 2. Install MIMIR

MIMIR is a normal Python package. Installing it pulls in everything the agent
**and all of its MCP servers** need and exposes two console commands: `mimir`
(interactive CLI) and `mimir-server` (WebSocket server for the VS Code extension).

### 2a. Quick install (recommended)

One script: it creates a virtualenv, installs the package with the `vllm` extra,
runs a smoke test, and — when `npm` is available — builds and installs the VS Code
extension too.

```bash
git clone https://github.com/MIMIR-LLM4CSE/MIMIR.git
cd MIMIR
./install.sh                 # -> .venv, `mimir` + `mimir-server`, VS Code extension
source .venv/bin/activate
```

Override the target venv, the extras, or skip the extension:

```bash
MIMIR_VENV=~/envs/mimir MIMIR_EXTRAS="vllm,dev" ./install.sh
MIMIR_SKIP_EXTENSION=1 ./install.sh          # Python only
```

A failing extension step never fails the install — the Python side is complete on
its own, and the script prints the command to retry.

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
| Scientific / utility servers | `numpy`, `sympy`, `psutil` | `math`, `symbolic_math`, `platform`, `system` servers |

Optional extras (declared in `pyproject.toml`):

- **`finetune`** — the `torch` / `transformers` / `peft` stack, only needed for the
  `ml/` servers (`server_finetune.py`). Without it that one server fails to start
  with an `ImportError`; the rest of the agent is unaffected.
  ```bash
  pip install ".[finetune]"
  ```
- **`dev`** — `pytest` and `ruff`, the two checks CI runs:
  ```bash
  pip install -e ".[dev]"
  pytest                       # test suite (from the repo root)
  ruff check .                 # lint; `--fix` applies the safe fixes
  ```
  The tests are plain `unittest` cases, so `python -m unittest discover mimir/tests`
  also works without this extra; `ruff` has no stdlib equivalent. The rule set is
  narrow on purpose (`[tool.ruff.lint]` in `pyproject.toml`): the families that catch
  a defect, not the ones that impose a house style.
- **Code-intelligence & validation binaries.** The `code_intel` server uses external
  language servers / tools when present (`pyright`/`pylsp`, `clangd`, `ctags`) and
  degrades to a text scan when absent. The mandatory check needs nothing installed —
  it runs in-process. What the model *additionally* runs goes through the `bash`
  server, so install whichever validators you want on PATH — Python (`ruff`, `mypy`, `pyflakes`,
  `black`, `pytest`), the compilers (`gcc`/`g++`/`gfortran`/`nvcc`/`javac`), and CMake
  (`cmake`/`ctest`) for C/C++/CUDA projects. These are system/CLI tools, not Python
  packages, and all of them are optional: the agent names an optional check as unrun when
  its tool is absent, and the required one never depends on them.

> **Legacy path.** `pip install -r mimir/requirements.txt` still works and installs
> the same core dependencies, but it does **not** register the `mimir` /
> `mimir-server` commands — prefer `pip install .`.

---

## 3. Point MIMIR at Your LLM Server

MIMIR does not start or schedule an LLM server — it talks to one that is already
running. You supply its HTTP address, and nothing else.

| Backend | What you provide | Default address |
|---|---|---|
| vLLM (default) | Address of the OpenAI-compatible endpoint | `http://127.0.0.1:8000` |
| Ollama | Address of the Ollama API | `http://127.0.0.1:11434` |
| Anthropic (Claude) | An API key — no address | — |

In the VS Code extension (§6) you type the address into the Connect panel and the
model dropdown fills itself from the endpoint. For the CLI and for headless runs,
the same information comes from flags or environment variables (§4).

### vLLM (OpenAI-compatible endpoint)

Start vLLM wherever your GPUs are, binding to all interfaces so other machines can
reach it:

```bash
source /path/to/vllm/setup.sh   # load CUDA modules / venv, if you need one
vllm serve /path/to/model \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
    # add --reasoning-parser deepseek_r1 for thinking-capable models
```

Check it from the machine that will run MIMIR — this is exactly what the extension
asks for when it fills the model dropdown:

```bash
curl http://<host>:8000/v1/models
```

Then point the agent at it:

```bash
mimir-server \
    --backend vllm \
    --vllm-base-url http://<host>:8000
    # `python -m mimir.client.ui.ws.ws_server ...` also works if not pip-installed
    # --model is optional; omit it to use whatever the endpoint serves
```

…or via environment variables:

```bash
export LLM_BACKEND=vllm
export VLLM_BASE_URL=http://<host>:8000
export VLLM_API_KEY=EMPTY      # leave as EMPTY unless your endpoint requires auth
```

The address is used exactly as given. If the endpoint is on the same machine,
`http://127.0.0.1:8000` is fine.

> **HTTPS behind a private CA.** Internal routes often carry a certificate the
> system trust store doesn't know. Certificate verification is off by default
> (`VLLM_VERIFY_SSL=0`); set `VLLM_VERIFY_SSL=1` when your endpoint has a publicly
> trusted certificate. The extension exposes this as `mimir.vllmVerifySsl`.

`--tool-call-parser` and `--reasoning-parser` are yours to choose here: MIMIR never
sends them, it only reads what your server produces. Consult vLLM's own docs for the
parser your model needs.

**Reasoning.** You choose nothing here — MIMIR works it out from the model the
endpoint reports:

1. it connects to the address you gave;
2. it reads the served model name from `/v1/models`;
3. it looks that name up in
   [`vllm_model_profiles.json`](mimir/client/config/vllm_model_profiles.json) and
   applies the reasoning mechanism of its **family**;
4. it tells the panel which mechanism won, and the thinking-depth control adapts its
   rungs — no token budget where the family has none, no "off" where the family
   always reasons.

The default is `chat_template_kwargs.enable_thinking`, which thinking-capable vLLM
templates read and other templates ignore, so a model matching no family still gets
its reasoning. Family keys match on a prefix and treat `_` and `.` as the same
separator, so one entry covers every size and both spellings of a series.

Families verified against their published chat template or model card:

| Family | Mechanism | Rungs | Can turn reasoning off |
|---|---|---|---|
| Qwen3 and most others | `enable_thinking` kwarg | token budget | yes |
| DeepSeek-V4 | `reasoning_effort` + `thinking_mode` | `low` `high` `max` | yes |
| GLM-5.3 | `reasoning_effort` | `low` `high` | no — always reasons |
| gpt-oss | `reasoning_effort` | `low` `medium` `high` | no — always reasons |
| Llama-Nemotron 3.1 / 3.3 | `detailed thinking on`/`off` in the system message | on/off | yes |

The rungs are per family on purpose: `low/medium/high` is OpenAI's ladder, not a
standard. Sending `medium` to DeepSeek or GLM lands on their template's fallback and
silently ignores what the user picked.

Add an entry only for a family whose published behaviour differs, and record in its
`note` where you verified it — the default already works, so an entry added on a hunch
can only make things worse. That file also carries the optional per-model `max_tools`
and `enforcement` knobs, unset by default.

To see reasoning rendered as a thinking block rather than inline text, start vLLM with
the matching `--reasoning-parser`.

### Ollama

1. Install Ollama: https://ollama.com/
2. Pull a model that fits your VRAM:

   ```bash
   ollama pull qwen3:8b                        # ~5 GB  — lightweight
   ollama pull qwen2.5-coder:32b               # ~19 GB
   ollama pull devstral-small-2:24b-instruct-2512-fp16   # ~48 GB
   ollama pull nemotron-3-super:120b            # ~86 GB — best results
   ```

3. Confirm it is serving:

   ```bash
   ollama list                              # pulled models
   curl http://127.0.0.1:11434/api/tags     # what MIMIR reads for its dropdown
   ```

4. Point the agent at it:

   ```bash
   mimir-server --backend ollama --ollama-base-url http://<host>:11434
   # or: export LLM_BACKEND=ollama OLLAMA_BASE_URL=http://<host>:11434
   ```

An Ollama server on another host needs `OLLAMA_HOST=0.0.0.0` in *its own*
environment to accept remote connections.

### Anthropic (hosted Claude API)

No server to run. Export a key, or paste one into the Connect panel (it is passed to
the server process in-memory and never written to disk):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
mimir-server --backend anthropic --model claude-sonnet-5
```

---

## 4. Environment Variables

Set these in your shell profile (`.bashrc`, `.zshrc`) or in the job script. The
VS Code extension sets the backend and address ones itself from the Connect form.

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BACKEND` | `vllm` | `vllm`, `ollama`, or `anthropic` |
| `MIMIR_DEFAULT_MODEL` | *(empty)* | Model selected at startup; overridden by `--model` flag or the UI |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API endpoint (`--ollama-base-url` sets this and `OLLAMA_HOST` together) |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000` | vLLM API base URL |
| `VLLM_API_KEY` | `EMPTY` | API key for vLLM calls |
| `VLLM_VERIFY_SSL` | `0` | Verify the vLLM endpoint's TLS certificate. Off by default so an internal HTTPS route behind a private CA works out of the box |
| `ANTHROPIC_API_KEY` | *(none)* | Key for the `anthropic` backend |
| `MIMIR_PYTHON` | *(none)* | Interpreter the VS Code extension starts the WS server with, when `~/.mimir/python` is absent or wrong |
| `MIMIR_EMBED_MODEL` | *(empty; `nomic-embed-text` on Ollama)* | Embedding model for semantic memory search & tool ranking. **Required for vLLM** (the served model name, e.g. `BAAI/bge-m3`). Empty + vLLM ⇒ semantic path disabled, lexical fallback used. |
| `MIMIR_EMBED_BASE_URL` | *(falls back to `VLLM_BASE_URL`)* | Serve embeddings from a separate endpoint than the chat model (vLLM only) |
| `MIMIR_EMBED_TIMEOUT` | `10` | HTTP timeout (seconds) for the vLLM embeddings call |
| `MCP_FILES_ROOT` | current working directory | Workspace root. Paths a tool *names* are confined to it, and reaching outside prompts for approval — but a program the agent runs (`python`/`make`/`gcc`) is not itself constrained, so this is a guardrail on intent, not a sandbox ([scope](SERVERS_DETAILED.md#scope-of-the-sandbox-read-this-before-trusting-confined)) |
| `GITHUB_TOKEN` | *(none)* | Raises GitHub API rate limit from 60 to 5 000 req/h |
| `MIMIR_SYSTEM_PROMPT_FILE` | *(none)* | Path to a `.md` file that replaces the doctrine half of the agent's general context (base system prompt). See below. |

### Custom general context (base system prompt)

The agent's "general context" — its base system prompt — defaults to a built-in
profile, but you can supply your own `.md` file (e.g. for a different persona or
domain). The first match wins:

1. **`MIMIR_SYSTEM_PROMPT_FILE`** — an absolute path to any `.md` file, anywhere on
   disk. Needs no `.mimir/` directory; best for ad-hoc use.
2. **`.mimir/system_prompt.md`** in the workspace root. Nothing is auto-created —
   create this file yourself, using
   [`mimir/examples/system_prompt.md`](mimir/examples/system_prompt.md) as a template.
3. Otherwise the built-in doctrine is used.

The file replaces the **doctrine** half of the base prompt (identity, style, scope,
workflow, reasoning). The **core** half — non-negotiables, latitude, tool results,
discovery, editing, validation, running code, planning & todo — is appended after it either way
and cannot be overridden: it carries facts about MIMIR's own tools and the obligations
the agent loop checks at runtime, so an override that dropped them would leave the loop
enforcing rules the model was never told (planning & todo is core for exactly that
reason: the loop refuses to conclude on an open checklist step). The dynamic memory /
todo / plan sections are still appended on top automatically. Details and the full section split:
[PLUGINS_DETAILED.md](PLUGINS_DETAILED.md#base-prompt-general-context).

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

Type `/help` at the prompt for the in-session commands (`/mode`, `/think`,
`/enforcement`, `/servers`, `/skills`, `/context`, `/trust`, …); `quit` exits.

---

## 6. VS Code Extension (Recommended)

The extension provides a chat panel with model selection, streaming output,
approval dialogs, and tool-status indicators.

### 6a. Install It

```bash
cd mimir/vscode-extension
npm install
npm run deploy
```

`npm run deploy` builds the bundle and then either installs the extension (first
run: it packages a `.vsix` and installs it with the `code` CLI) or copies the new
build over the installed one (every run after that). Either way, reload the VS Code
window afterwards: `Ctrl+Shift+P` → *Developer: Reload Window*.

`./install.sh` (§2a) already does this — run it by hand when you have rebuilt the
extension and want the change live.

Other scripts: `npm run dev` rebuilds on save, `npm test` runs the unit tests, and
`npm run package` produces a `.vsix` without installing it (for `code
--install-extension mimir-*.vsix` on another machine). For extension development,
opening `mimir/vscode-extension` in VS Code and pressing `F5` launches a second
window running it from source.

### 6b. Connect

Open the MIMIR panel, and in the Connect form:

1. **Backend** — vLLM, Ollama, or Anthropic (Claude).
2. **Address** — where that server is running, e.g. `http://10.0.0.4:8000`. The
   model dropdown fills itself from it; the ⟳ button re-reads it.
3. **Model** — one of the models the endpoint reports. For Anthropic, an API key
   field replaces the address.
4. **Remember this address** (optional) — tick it and the next VS Code window
   reconnects to this endpoint by itself, at startup, whether or not the MIMIR
   panel is open: the extension probes the address first and only connects when
   it answers, so an endpoint that is down just leaves you on this form,
   pre-filled. Untick and connect again to forget it.
   The address and model are stored; an Anthropic key never is, which is why the
   box only appears for vLLM and Ollama.
5. **Connect** — the extension starts the WS server locally and attaches to it.

That is the whole configuration: a working setup needs **no `.vscode/settings.json`**.

### 6c. Optional Settings

Everything below has a usable default; set one only to change the value the panel
starts on, or when the WS server needs a specific interpreter.

| Setting | Default | Purpose |
|---|---|---|
| `mimir.backend` | `vllm` | Backend the Connect form opens on |
| `mimir.vllmBaseUrl` | `http://127.0.0.1:8000` | Address the form opens on for vLLM |
| `mimir.ollamaUrl` | `http://127.0.0.1:11434` | Address the form opens on for Ollama |
| `mimir.vllmVerifySsl` | `true` | Uncheck for an HTTPS endpoint behind a private CA |
| `mimir.pythonPath` | *(empty → auto)* | Interpreter used to start the WS server. Leave empty — see below |
| `mimir.wsUrl` | *(empty → auto)* | Leave empty: each window starts its own server on a free port. Set it only to attach to a server you run yourself |
| `mimir.anthropicAvailableModels` | *(list of Claude ids)* | Models offered for the Anthropic backend |
| `mimir.notifications.enabled` | `true` | Native notification when a task finishes off-screen |

**Which Python runs the server.** The extension starts it with `bash -c`, which is
neither a login nor an interactive shell: it sources no `.bashrc`, no `.bash_profile`,
and no setup script. The child process inherits the VS Code server's environment and
nothing else — so `python3` on that PATH is usually the system one, not your venv.
The interpreter is therefore resolved in this order:

1. `mimir.pythonPath`, if you set it;
2. the `MIMIR_PYTHON` environment variable;
3. `~/.mimir/python` — **written by `install.sh`**, which is why a plain
   `./install.sh` needs no configuration at all;
4. `python3` from `PATH`.

A consequence of (2): a variable you export in `.bashrc` only reaches MIMIR if it was
already set when the VS Code server started. After editing your profile, close the
remote window and reconnect — reloading the window is not enough.

---

## 7. HPC / SLURM

MIMIR runs where you run it and talks to an LLM endpoint over HTTP; it does not
allocate nodes to connect. On a cluster, start your vLLM or Ollama server in a job
as usual, then give MIMIR the address of the node serving it (§3).

Scheduling is instead something the **agent** can do on your behalf: the `hpc` MCP
server exposes `salloc_submit`, `sbatch_submit`, and job-inspection tools, so you can
ask MIMIR to submit and monitor *your* jobs. See
[SERVERS_DETAILED.md](SERVERS_DETAILED.md) for that tool catalog.

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
