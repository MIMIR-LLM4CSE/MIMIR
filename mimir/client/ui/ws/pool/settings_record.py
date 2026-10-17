"""The settings and toggles the user changed for the session, replayed onto every
worker the pool builds later."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..worker.worker import AgentWorker

logger = logging.getLogger(__name__)


class SettingsRecord:
    """The choices a newly built worker must start from.

    A worker is built long after the user makes these, so it has to arrive already
    configured: without the record, a conversation started after the user picked "plan
    mode" would quietly come up in agent mode. Every choice is therefore recorded here
    *and* applied to every live worker, and a new worker replays the record once its
    agent exists. The values are validated where they are applied (core.settings), not
    here, so the two cannot drift apart about what a mode may be.
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
