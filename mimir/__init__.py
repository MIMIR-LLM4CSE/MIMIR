"""MIMIR agent package.

Top-level re-export of the extension-pack authoring surface so packs (and embedding
apps) can ``from mimir import PolicyCheck, NudgeRule, register_policy_check,
register_nudge``. See :mod:`mimir.client.extensions`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    # One version for the package and the VS Code extension; see CHANGELOG.md.
    __version__ = _dist_version("mimir-agent")
except PackageNotFoundError:  # a bare checkout on sys.path, never installed
    __version__ = "unknown"

# Extension-authoring symbols re-exported lazily (see __getattr__): importing
# mimir.client at package-import time would pull in the whole client stack
# (mcp/ollama/…), so we defer until one of these is actually requested.
_EXTENSION_EXPORTS = frozenset({
    "PolicyCheck",
    "NudgeRule",
    "register_policy_check",
    "register_nudge",
    "load_plugins",
})


def __getattr__(name: str):
    if name in _EXTENSION_EXPORTS:
        from .client import extensions
        return getattr(extensions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
