"""Running a session command the way the terminal does, for tests."""
from __future__ import annotations

import asyncio
from typing import Any

from mimir.client.ui.cli.chat_commands import CliHost, run_cli_command


def cli(query: str, agent: Any = None, **host: Any) -> str | None:
    """What the terminal prints for *query*; None when it is not a command."""
    return asyncio.run(run_cli_command(CliHost(agent, **host), query))
