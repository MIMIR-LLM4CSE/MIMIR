# Contributing to MIMIR

Thanks for your interest in improving MIMIR! This guide covers the basics for
getting a development environment running and submitting changes.

## Development setup

```bash
git clone https://github.com/MIMIR-LLM4CSE/MIMIR.git
cd MIMIR
./install.sh                        # creates .venv and installs with the 'vllm' extra
source .venv/bin/activate
pip install -e ".[dev]"             # editable install + test tooling
```

## Running the tests

```bash
pytest                              # full suite (see pyproject [tool.pytest.ini_options])
pytest mimir/tests/test_signals.py  # a single file
```

Please make sure the suite passes before opening a pull request.

## VS Code extension

```bash
cd mimir/vscode-extension
npm install
npm run build        # compile + bundle
npm run package      # build the .vsix
npm test             # vitest
```

## Project layout

- `mimir/client/` — the agent client (loop, policy, context, UI, backends).
- `mimir/servers/` — MCP tool servers, launched as stdio subprocesses.
- `mimir/skills/` — methodology prompts auto-detected from the query.
- `mimir/tests/` — pytest suite.
- `mimir/vscode-extension/` — the VS Code chat frontend (TypeScript + React).

Authoritative docs: `README.md`, `SETUP.md`, `ARCHITECTURE.md`, `POLICY.md`,
`CLIENT_DETAILED.md`, `SERVERS_DETAILED.md`, `EXTENSION_DETAILED.md`,
`PLUGINS_DETAILED.md`.

## Pull requests

- Keep changes focused; describe the motivation and the behavior change.
- Add or update tests for new behavior.
- Update the relevant docs when you change user-facing behavior or policy.
- Run `pytest` (and `npm test` for extension changes) locally first.

## Reporting issues

Open an issue with reproduction steps, the backend/model you used, and any
relevant tool output or logs.
