"""Frontends that drive the ``MimirAgent`` core: the CLI (``cli/``) and the
WebSocket/VS Code bridge (``ws/``). ``MimirAgent`` itself lives at the client root.

Nothing is imported here on purpose: ``python -m mimir.client.ui.ws.ws_server`` imports
this package first, and the server's ``--cwd`` has to be applied before anything that
resolves the workspace root at import time (``config.paths``, the backend factory).
"""
