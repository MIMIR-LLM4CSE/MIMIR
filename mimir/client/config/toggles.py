"""The servers, skills and nudges the user switched off, kept in the preferences file.

Only what is off is stored, so anything new starts on. A server switched off keeps its
process, but its tools are not offered to the model; a skill switched off is hidden
from ``/<name>`` and from the model; a nudge switched off is skipped by the cascade.
Policies have no switch.
"""
from __future__ import annotations

from .preferences import load_disabled, save_disabled

TOGGLE_KINDS = ("server", "skill", "nudge")


def is_enabled(holder: object, kind: str, name: str) -> bool:
    """Whether *holder*'s toggles leave *name* on; nothing is off without toggles."""
    toggles = getattr(holder, "toggles", None)
    return toggles is None or toggles.enabled(kind, name)


class Toggles:
    def __init__(self, disabled: dict[str, set[str]] | None = None, *,
                 persist: bool = True) -> None:
        if disabled is None:
            servers, skills, nudges = load_disabled()
            disabled = {"server": servers, "skill": skills, "nudge": nudges}
        self.disabled: dict[str, set[str]] = {kind: set(disabled.get(kind) or ())
                                              for kind in TOGGLE_KINDS}
        self._persist = persist

    def enabled(self, kind: str, name: str) -> bool:
        return name not in self.disabled[kind]

    def set_enabled(self, kind: str, name: str, enabled: bool) -> None:
        if kind not in TOGGLE_KINDS:
            raise ValueError(f"unknown toggle kind {kind!r}; expected one of {TOGGLE_KINDS}")
        if enabled:
            self.disabled[kind].discard(name)
        else:
            self.disabled[kind].add(name)
        if self._persist:
            save_disabled(self.disabled["server"], self.disabled["skill"], self.disabled["nudge"])
