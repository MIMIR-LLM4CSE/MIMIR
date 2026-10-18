"""Shared text tokenization and filename/path scoring helpers for MCP servers."""

import os
import re


_YAML_RESERVED_WORDS = {
    "true", "false", "null", "yes", "no", "on", "off", "~",
}


def yaml_scalar(value: str) -> str:
    """Render a string as a YAML scalar, quoting it when plain style is unsafe."""
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    if not text:
        return '""'
    needs_quote = (
        text[0] in "-?:,[]{}#&*!|>'\"%@`"
        or ": " in text
        or text.endswith(":")
        or " #" in text
        or text.lower() in _YAML_RESERVED_WORDS
    )
    if needs_quote:
        return "'" + text.replace("'", "''") + "'"
    return text


def yaml_unquote(value: str) -> str:
    """Inverse of :func:`yaml_scalar` for the simple scalars we emit."""
    text = (value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        inner = text[1:-1]
        return inner.replace("''", "'") if text[0] == "'" else inner
    return text


def tokenize(text: str, min_len: int = 2) -> set[str]:
    """Split text into lowercase alphanumeric tokens with a minimum length."""
    return {
        token
        for token in re.split(r"[^a-zA-Z0-9]+", (text or "").lower())
        if len(token) >= min_len
    }


def score_filename_hint(rel_path: str, hint: str) -> int:
    """Score how well a relative path matches a filename/path hint."""
    path_lower = (rel_path or "").lower()
    hint_lower = (hint or "").strip().lower()
    if not hint_lower:
        return 0

    score = 0
    hint_tokens = tokenize(hint_lower)
    path_tokens = tokenize(path_lower)
    basename = os.path.basename(path_lower)
    stem, ext = os.path.splitext(basename)
    hint_stem, hint_ext = os.path.splitext(hint_lower)

    if basename == hint_lower:
        score += 12
    if stem == hint_stem and hint_stem:
        score += 8
    if hint_lower in basename:
        score += 5
    if hint_ext and ext == hint_ext:
        score += 2
    score += 3 * len(hint_tokens & path_tokens)
    return score