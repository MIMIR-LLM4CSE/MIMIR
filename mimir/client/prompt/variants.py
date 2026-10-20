"""A prose variant: the prompt's sections and the constant messages, reworded for a run.

A variant is a JSON file ``{"name", "sections": {id: text}, "messages": {id: text}}``,
the ids being those of ``system_prompt.SECTIONS`` and ``nudges.messages.MESSAGES``. It is
how a wording is measured against the code's before the code is changed.
"""
from __future__ import annotations

import contextlib
import json
import os
from typing import Any, Iterator

from ..config import prompt_variants
from ..guardrails.nudges.messages import MESSAGES
from .system_prompt import SECTIONS


def load_variant(path: str) -> dict[str, Any]:
    """Read and check a variant file; an id the code does not have is refused, not ignored."""
    with open(path, encoding="utf-8") as fh:
        variant = json.load(fh)
    if not isinstance(variant, dict):
        raise ValueError(f"{path}: a variant is a JSON object")
    variant.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    for kind, known in (("sections", SECTIONS), ("messages", MESSAGES)):
        texts = variant.get(kind) or {}
        if not isinstance(texts, dict) or not all(isinstance(t, str) for t in texts.values()):
            raise ValueError(f"{path}: '{kind}' maps ids to texts")
        unknown = sorted(set(texts) - set(known))
        if unknown:
            raise ValueError(f"{path}: unknown {kind} {', '.join(unknown)} "
                             f"(known: {', '.join(sorted(known))})")
    return variant


@contextlib.contextmanager
def use_variant(variant: dict[str, Any] | None) -> Iterator[str]:
    """Put *variant* in force for the block (none: the code's texts); yields its name."""
    previous = prompt_variants.activate(variant or {})
    try:
        yield prompt_variants.active_name()
    finally:
        prompt_variants.restore(previous)
