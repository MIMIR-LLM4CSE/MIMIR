"""Resolution of the user skills directory under ``.mimir/skills/``.

A user skill ``.mimir/skills/<name>/SKILL.md`` (override ``MIMIR_SKILLS_DIR``) is
loaded by ``MimirAgent.load_skills`` over the bundled ones, a same-named one
overriding; the file format is ``shared/skill_file``.
"""

from __future__ import annotations

from ..config.paths import SKILLS_DIR_ENV, SKILLS_DIRNAME, resolve_extension_dir


def resolve_skills_dir() -> str:
    """User skills directory: ``MIMIR_SKILLS_DIR`` env, else ``.mimir/skills``."""
    return resolve_extension_dir(SKILLS_DIR_ENV, SKILLS_DIRNAME)
