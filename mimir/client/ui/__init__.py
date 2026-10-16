"""Frontends that drive the ``MimirAgent`` core: the CLI (``cli/``) and the
WebSocket/VS Code bridge (``ws/``). ``MimirAgent`` itself lives at the client root."""

from ..agent_core import MimirAgent
from .cli.chat_session import run_chat_session
from .cli.main import main

__all__ = ["MimirAgent", "run_chat_session", "main"]
