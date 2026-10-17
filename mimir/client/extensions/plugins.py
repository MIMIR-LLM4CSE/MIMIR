"""Loading the extension packs of a plugins directory into one ``ExtensionSet``.

Every ``*.py`` module of the directory is imported (``_``-prefixed ones are helpers and
skipped); each registers its descriptors as it is imported, into the set being built.
A pack that fails to import is logged and skipped: one bad pack never stops an agent.

The directory is ``MIMIR_PLUGINS_DIR`` if set, else ``<workspace>/.mimir/plugins/``.
Each directory is loaded once per process, and its set is shared by every agent of
that workspace.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading

from ..config.paths import PLUGINS_DIRNAME, PLUGINS_DIR_ENV, resolve_extension_dir
from ..core.extension_set import AMBIENT, ExtensionSet, registering_into

logger = logging.getLogger(__name__)

_loaded: dict[str, ExtensionSet] = {}
_lock = threading.Lock()


def resolve_plugins_dir() -> str:
    """Return the plugins directory path (env override, else ``.mimir/plugins/``)."""
    return resolve_extension_dir(PLUGINS_DIR_ENV, PLUGINS_DIRNAME)


def load_plugins(plugins_dir: str | None = None) -> ExtensionSet:
    """The extension set of *plugins_dir*, its packs imported on first use."""
    directory = os.path.realpath(plugins_dir or resolve_plugins_dir())
    with _lock:
        if directory not in _loaded:
            _loaded[directory] = _import_packs(directory)
        return _loaded[directory]


def _import_packs(directory: str) -> ExtensionSet:
    extensions = ExtensionSet(parent=AMBIENT)
    if not os.path.isdir(directory):
        return extensions
    with registering_into(extensions):
        for entry in sorted(os.listdir(directory)):
            if not entry.endswith(".py") or entry.startswith("_"):
                continue
            path = os.path.join(directory, entry)
            # Unique per directory, so two workspaces' same-named packs stay apart.
            stem = os.path.splitext(entry)[0]
            mod_name = f"mimir_plugin_{abs(hash(directory)):x}_{stem}"
            try:
                spec = importlib.util.spec_from_file_location(mod_name, path)
                if spec is None or spec.loader is None:
                    logger.warning("Could not load plugin spec for %s", path)
                    continue
                module = importlib.util.module_from_spec(spec)
                # In sys.modules while it runs, so dataclasses and relative helpers resolve.
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)
                extensions.packs.append(path)
            except Exception as exc:
                logger.warning("Skipping extension pack %s: %s", path, exc)
    if extensions.packs:
        logger.info("Loaded %d extension pack(s) from %s", len(extensions.packs), directory)
    return extensions
