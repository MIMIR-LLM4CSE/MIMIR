"""The settings and toggles the user changed for the session, replayed onto every
worker the pool builds later."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..worker.worker import AgentWorker

logger = logging.getLogger(__name__)


class SettingsRecord:
    """The choices a newly built worker replays: settings, toggles, non-interactive.

    Recorded and applied to live workers alike; validated where applied (core.settings).
    """

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}              # core.settings name -> value
        # Keyed by (kind, name): one record per server/skill/nudge switched, not one
        # per kind — keyed by method, only the last toggle before the first agent
        # survived to be replayed.
        self.toggles: dict[tuple[str, str], bool] = {}
        self.non_interactive = False

    def replay(self, worker: AgentWorker) -> None:
        for name, value in self.values.items():
            error = worker.apply_setting(name, value)
            if error:
                logger.warning("pool: could not apply %s to a new worker: %s", name, error)
        for (kind, name), enabled in self.toggles.items():
            try:
                worker.set_enabled(kind, name, enabled)
            except Exception:
                logger.warning("pool: could not switch %s %s on a new worker", kind, name,
                               exc_info=True)
        if self.non_interactive:
            worker.set_non_interactive(True)
