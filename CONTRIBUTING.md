# Contributing to MIMIR

Thanks for your interest in improving MIMIR! This guide covers the basics for
getting a development environment running and submitting changes.

## Development setup

```bash
git clone https://github.com/MIMIR-LLM4CSE/MIMIR.git
cd MIMIR
./install.sh                        # creates .venv and installs with the 'vllm' extra
source .venv/bin/activate
pip install -e ".[dev]"             # editable install + pytest and ruff
```

## Running the checks

```bash
pytest                              # full suite (see pyproject [tool.pytest.ini_options])
pytest mimir/tests/test_signals.py  # a single file
ruff check .                        # lint (see pyproject [tool.ruff.lint])
ruff check . --fix                  # apply the safe fixes
```

Both must be clean before opening a pull request — CI runs the same two commands.

The ruff rule set is deliberately narrow: `E4`/`E7`/`E9`/`F`, i.e. the checks that
catch a defect (an undefined name, a dead import, a file that would fail at import),
not the ones that impose a house style. Import sorting and formatting are **not**
enforced; please don't turn them on in a change that is about something else, and
don't reformat code you are not otherwise touching.

## VS Code extension

```bash
cd mimir/vscode-extension
npm install
npm run deploy       # build + install into VS Code (or update it in place)
npm run dev          # rebuild on every save
npm test             # vitest
npm run package      # build a .vsix without installing it
```

After `npm run deploy`, reload the VS Code window to pick up the new bundle
(`Ctrl+Shift+P` → *Developer: Reload Window*).

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
- Run `pytest` and `ruff check .` (and `npm test` for extension changes) locally first.

## Reporting issues

Open an issue with reproduction steps, the backend/model you used, and any
relevant tool output or logs.
