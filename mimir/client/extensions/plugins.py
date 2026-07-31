"""Directory-scan loader for application extension packs (custom policies + nudges).

Mirrors ``MimirAgent.load_skills``: resolve a plugins directory, import every ``*.py``
module found there, and let each module register its descriptors as an import
side-effect (by calling ``register_policy_check`` / ``register_nudge``). A pack that
fails to import is logged and skipped — one bad pack never crashes agent startup, the
same fail-open stance the server registry takes.

Resolution order for the directory:
    ``MIMIR_PLUGINS_DIR`` env var → ``<workspace>/.mimir/plugins/``.
An absent directory is a no-op (returns 0).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys

from ..config.constants import PLUGINS_DIRNAME, PLUGINS_DIR_ENV, resolve_extension_dir

logger = logging.getLogger(__name__)


def resolve_plugins_dir() -> str:
    """Return the plugins directory path (env override, else ``.mimir/plugins/``)."""
    return resolve_extension_dir(PLUGINS_DIR_ENV, PLUGINS_DIRNAME)


def load_plugins(plugins_dir: str | None = None) -> int:
    """Import every pack module in *plugins_dir*; return how many imported cleanly.

    Registration is the import side effect, so this only needs to import each module.
    Modules whose basename starts with ``_`` are skipped (private/helpers).
    """
    directory = plugins_dir or resolve_plugins_dir()
    if not os.path.isdir(directory):
        return 0

    loaded = 0
    for entry in sorted(os.listdir(directory)):
        if not entry.endswith(".py") or entry.startswith("_"):
            continue
        path = os.path.join(directory, entry)
        mod_name = f"mimir_plugin_{os.path.splitext(entry)[0]}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                logger.warning("Could not load plugin spec for %s", path)
                continue
            module = importlib.util.module_from_spec(spec)
            # Register in sys.modules so dataclasses / relative helpers resolve during exec.
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            loaded += 1
        except Exception as exc:  # a bad pack must not crash startup
            logger.warning("Skipping extension pack %s: %s", path, exc)
    if loaded:
        logger.info("Loaded %d extension pack(s) from %s", loaded, directory)
    return loaded
