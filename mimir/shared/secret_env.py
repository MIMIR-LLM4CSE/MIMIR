"""The environment without its secrets: what a child process is given by default.

A key in a server's environment is a key in every command that server runs, and in
``/proc/<pid>/environ`` for any process of the same user to read. A variable is a secret
by its name: a ``KEY``, ``TOKEN``, ``SECRET``, ``PASSWORD`` or ``CREDENTIALS`` word.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping

_SECRET_NAME = re.compile(
    r"(?:^|_)(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(?:_|$)", re.I)


def is_secret(name: str) -> bool:
    """Whether the variable *name* holds a secret, judged by its name."""
    return bool(_SECRET_NAME.search(name))


def without_secrets(env: Mapping[str, str], keep: Iterable[str] = ()) -> dict[str, str]:
    """*env* with every secret removed but those named in *keep*."""
    kept = set(keep)
    return {k: v for k, v in env.items() if k in kept or not is_secret(k)}
