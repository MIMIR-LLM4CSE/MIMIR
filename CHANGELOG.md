# Changelog

One version covers all of MIMIR: the Python package (`pyproject.toml`) and the
VS Code extension (`mimir/vscode-extension/package.json`). The extension talks to
the server of the same checkout, so the two always move together.

| Part of the version | Goes up when… | Example |
|---|---|---|
| Major (`X.0.0`) | an existing setup stops working without a change on the user's side: a setting, an environment variable, a CLI flag or a stored file is renamed or dropped | a `mimir.*` setting is renamed |
| Minor (`1.X.0`) | something new appears, and existing setups keep working | a new backend, tool server, or panel |
| Patch (`1.0.X`) | a fix, with nothing new to learn | a card that did not render |

How to release is in [CONTRIBUTING.md](CONTRIBUTING.md#releasing). The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- How far a sub-agent may go is now the user's setting, not the model's choice:
  **explore** (read only, the default) or **parallel**. At *parallel* a sub-agent
  may also edit and run commands — always in a copy of the repository, on its own
  branch, so several of them can work at once without touching each other's
  files. It hands back a branch and a diff; nothing is merged for you. The copy
  is kept when it finishes — it holds the build the work just paid for — and the
  panel says where. The last three copies of a session are kept; older ones go
  when the next sub-agent needs the room, and copies left by a conversation you
  deleted go with them. Nothing is ever removed before what it holds is committed,
  so a sub-agent that stopped half-way leaves its work on its branch. A sub-agent
  that changed nothing leaves no branch behind. Set it with `/subagents`, or in
  the new panel.
- A **scientific-computing panel** in the extension: the sub-agent setting and
  what each one is doing, the optimisation in progress, the machine MIMIR
  detected, and the runs still going outside the current turn. Each section is
  filled by the server that owns those facts, so an extension server can add its
  own.
- A sub-agent's **approvals and questions now reach you**, on the usual card.
  They are queued — one at a time — and each says which sub-agent is asking.
- A button with a badge shows how many sub-agents are working; opening one shows
  what it is doing, step by step, including a detached one.
- The main agent works as the lead engineer: it takes the heaviest part itself,
  reports progress as the work proceeds, and brings you what is yours to decide.
- The main agent chooses the tools of each sub-agent, out of what it can use
  itself. A sub-agent given a writing or executing tool works; given none, it
  explores read-only. Planning, delegation, asking the user, writing memory and
  cluster submissions stay with the main agent.
- Each sub-agent keeps its own todo list and its own scratchpad, in a session of
  its own stored under the session that spawned it. Several sub-agents can
  therefore work in parallel, one per line of work, without colliding.
- A sub-agent can be started in the background: the main agent gets a handle
  right away, keeps working, and is woken with the answer when the sub-agent
  finishes — the same behaviour as a command sent to the background. The wake
  leads with the answer, and says where to read the rest when there is more of it
  than a wake can carry.
- The main agent sets how long a sub-agent may run (up to 19 minutes). A
  sub-agent is told its budget and is asked to hand over — what it established,
  what is left, where — before the time runs out, so a long piece of work
  continues instead of restarting.
- A "sub-agents" panel in the extension lists what the current session
  delegated, with each sub-agent's state, tools and checklist. Sub-agents never
  appear in the session list, and they are deleted with their session.
- On an endpoint that serves several models, a sub-agent can run on another one
  than the main agent's. The main model picks, from a table of the served models
  with their size, estimated speed and benchmark scores by category (reasoning,
  coding, agentic work, tools, discovery). The data lives in
  `mimir/client/config/model_catalog.json`, with a source for every number.
- MIMIR optimizes for the compute nodes, not only for the machine it runs on. A
  short Slurm job reads a node's full profile on the node: CPU model, SIMD, the
  `-march` gcc resolves there, caches, GPUs, OS and glibc, toolchains, Python
  environments. The profile is kept until Slurm's description of the node changes.
  `slurm_node_profile` says which of these differ from the login node. MIMIR tells
  apart three places it can run: no scheduler, a login node, or inside an allocation
  on the node itself, where no probe is needed.
- Batch jobs can aim at one kind of node. `sbatch_submit` and `proxy_slurm` take
  `constraint`, `nodelist`, `ntasks` and `exclusive`. A proxy optimization run on
  Slurm is exclusive by default, and picks a Python that starts on the node.
- An optimization session is pinned to the machine its baseline ran on. A timing
  taken on another kind of machine is reported as `incomparable` and moves neither
  the best nor the stall count.
- `mimir.maxModelLen` sets the context window (tokens) for a vLLM or Ray Serve
  endpoint that does not report `max_model_len`, so the client no longer stays on
  its static 200K default. `0` (the default) leaves the reported window in charge.
  It forwards to `MIMIR_VLLM_MAX_MODEL_LEN` and `MIMIR_RAY_MAX_MODEL_LEN`.
- MIMIR can cancel a Slurm job. `slurm_cancel` takes one job ID of yours, pending
  or running, and asks your approval first. `scancel` stays refused in the shell,
  where `scancel -u` would cancel far more than one card shows.
- `mimir.vllmTokenize` (`MIMIR_VLLM_TOKENIZE=0`) stops every call to `/tokenize`,
  sub-agents included. A router that does not serve it, or answers it slowly, then
  costs nothing: a timeout used to be retried on every count, up to 5 s each.
- A GitHub file is read in pages. `github_get_file` takes a line range, returns at
  most 400 lines a call, and says where to resume — so the first look at a long
  source file costs a quarter of what the whole file did, and a file too large to
  inline is read through its raw URL instead of being refused outright.
- A fetch can now ask for part of a page instead of all of it. `contains=` returns
  the regions that answer it — a word matching a heading gives you that whole
  section, otherwise you get windows around the occurrences, each saying where it
  sits. `offset=` resumes a long document where the last reply stopped. Headings
  survive extraction as Markdown, so a page reads as named regions rather than one
  wall of prose. Measured across twelve sites: a 20 500-token article answers a
  targeted question in 127 tokens, a 33 700-token spec in 283.
- One install now runs on every node of a mixed cluster. `install.sh` fetches a
  portable Python 3.10 through uv, which needs only glibc 2.17 and brings its own
  OpenSSL and libcrypt. A Python compiled on a RHEL 9 node stopped on every RHEL 8
  node with `libcrypt.so.2: cannot open shared object file`. The venv is now named
  after the platform (`.venv-linux-x86_64`). A launcher, `~/.mimir/bin/python`,
  starts the venv built for the machine it runs on, so a shared home directory can
  serve several kinds of machines. `PYTHON=...` still selects your own interpreter.
  An existing install keeps working until you run `install.sh` again.

### Changed
- The context window the server reports now wins over the `MIMIR_VLLM_MAX_MODEL_LEN`
  / `MIMIR_RAY_MAX_MODEL_LEN` override. Those variables are the fallback for an
  endpoint that reports no `max_model_len`, not a value that supersedes one that
  does — so pinning stock `vllm` (e.g. 32768) is no longer overridden by a stale
  escape-hatch value.

### Fixed
- On a host whose locale is not English, `platform_probe` reported no CPU model and
  no SIMD at all: `lscpu` translates its field names. Probes now run in the C locale.
- `srun` from a login node started a job without the approval and the validation
  hold that `sbatch` gets. It now gets both. Inside an allocation it is unchanged.
- The run timer of a proxy evaluation starts at zero on every run. A second run of
  the same proxy used to continue from the first run's card, which went on counting
  from the earlier start.
- The context bar splits a prompt the way the server does. Without `/tokenize`,
  the history was counted at a fixed 4 characters per token, and the error landed in
  the fixed overhead: on one DeepSeek session the bar showed ~48k of system prompt
  and tools where the server charged 36k. The history's ratio is now measured from
  the `prompt_tokens` the server reports, with no extra request.
- Every tool parameter was sent twice on every call: once in the schema, and again
  in the description, which still carried the docstring's `Args:` block. It now
  leaves the description once the schema carries it word for word — a fifth of the
  tools schema, 31k tokens down to 25k with every server on.
- A single step can no longer overrun the context window. Tool results are now
  bounded before they enter the history — per result, and per step together — so
  several calls returning at once cannot do what four fetches did to one session:
  take a 200k window to 215k with nothing given the chance to object. Small results
  are never cut to pay for large ones. This covers every tool, including servers
  MIMIR does not ship.
- A failed web request no longer costs more than a successful one. The error
  branches returned up to 512 KB of the error body raw, with no extraction and no
  ceiling: one paper host answering 403 cost 131 164 tokens for a request that
  returned nothing.
- A page whose text cannot be extracted — a script shell, or a body that never
  arrived behind 700 KB of inline CSS — now comes back as its own metadata and
  embedded data rather than as markup. One thesis record page went from 34 479
  tokens of CSS to about 1 400 tokens carrying its title, jury, keywords and full
  abstract.
- Guidance written for the model actually reaches it now. `hint` is a reserved key
  that is stripped from success payloads, so several tools' "fetch a more specific
  URL" and "call again with confirm=True" lines had never once been delivered.
- After a `/model` switch, sub-agents run on the new model. They kept the model
  the session started with.
- Behind an OpenAI-compatible router that does not serve vLLM's `/tokenize`,
  each turn no longer waits tens of seconds before the model is asked. The
  refusal is remembered per endpoint instead of retried for every message.
- A build launched just as another background job finished keeps its row and
  its progress bar. The finished job was handed to the running turn, but the
  panel was told a new turn had begun and cleared that turn's rows.

## [1.0.0] — 2026-09-18

First versioned release. Everything since the initial public release (0.1.0) is
in it. The main points:

### Agent
- Three modes: **agent**, **plan** (read-only, ends in a plan you approve), and
  **ask** (read-only questions). The mode applies from the next step, even mid-turn.
- Three approval modes: **manual**, **auto**, **all**. One approval card per call.
  A refusal is read as an instruction, not an error to retry.
- A verification ledger under each answer: what was checked, what was not.
- Sub-agents, which run in the user's approval mode.
- No step ceiling on a run.
- Builds go through the tool that owns them, and only as far as the change needs.

### Backends
- vLLM, Ray Serve, Ollama and the Anthropic API.
- The model list is read from the endpoint. Nothing to maintain.

### Long runs
- Any blocking tool can be moved to the background. A background job wakes the
  session that launched it when it ends.
- A run reports its phase, and a progress bar when the tool counts its own progress.
- A turn outlives its connection. Its work comes back when the panel reconnects.

### VS Code extension
- Connect form: backend, address, model. **Remember this address** reconnects the
  next window on its own. A connection error is explained in plain words.
- File names in tool rows open the file on the lines that were read or edited.
- A card in the corner for a run whose row scrolled away.
- `↑` recalls a sent message.
- Default Claude models updated to the current family.

### Install
- `install.sh` clones, installs and records the Python interpreter the extension uses.
- Each VS Code window starts its own server on a free port.

### Removed
- The fine-tuning server. The proxy server already covers its runs.
- The "keep going?" card, with the step ceiling it served.

## [0.1.0] — 2026-07-31

Initial public release.

[Unreleased]: https://github.com/MIMIR-LLM4CSE/MIMIR/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/MIMIR-LLM4CSE/MIMIR/releases/tag/v1.0.0
[0.1.0]: https://github.com/MIMIR-LLM4CSE/MIMIR/commits/main
