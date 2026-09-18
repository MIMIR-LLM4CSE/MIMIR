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

### Fixed
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
