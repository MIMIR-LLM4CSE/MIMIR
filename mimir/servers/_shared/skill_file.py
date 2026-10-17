"""Reading a skill: ``<skills_dir>/<name>/SKILL.md``, front-matter then body.

The front-matter must name the skill (``name``, matching its directory) and describe
it (``description``). ``disable-model-invocation: true`` keeps the skill for the
user's ``/<name>``: the model may not pull it itself.
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)

# Values of ``disable-model-invocation`` that mean yes; anything else, absence
# included, leaves the skill to the model too.
_TRUTHY = frozenset({"true", "yes", "1"})


def model_invocable(front_matter: dict) -> bool:
    """Whether the model may load the skill itself; the user's ``/<name>`` always may."""
    return str(front_matter.get("disable-model-invocation", "")).strip().lower() not in _TRUTHY


def parse_skill_markdown(text: str) -> dict:
    """``{name, description, content, model_invocable}``; ValueError when malformed."""
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        raise ValueError("SKILL.md is missing front-matter block")
    header, body = m.groups()
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    if "name" not in meta or "description" not in meta:
        raise ValueError("SKILL.md front-matter must define name and description")
    return {
        "name": meta["name"],
        "description": meta["description"],
        "content": body.strip(),
        "model_invocable": model_invocable(meta),
    }


def read_skills_dir(skills_dir: str) -> dict[str, dict]:
    """Every well-formed skill of *skills_dir*, by name: description, content and
    whether the model may load it. A malformed or misnamed one is logged and skipped."""
    skills: dict[str, dict] = {}
    if not os.path.isdir(skills_dir):
        return skills
    for entry in sorted(os.listdir(skills_dir)):
        md_path = os.path.join(skills_dir, entry, "SKILL.md")
        if not os.path.isfile(md_path):
            continue
        try:
            with open(md_path, encoding="utf-8") as fh:
                parsed = parse_skill_markdown(fh.read())
        except Exception as exc:
            logger.warning("Failed to load skill %s: %s", entry, exc)
            continue
        if parsed["name"] != entry:
            logger.warning("Skill directory '%s' and skill name '%s' mismatch; skipping",
                           entry, parsed["name"])
            continue
        skills[entry] = {key: parsed[key] for key in ("description", "content", "model_invocable")}
    return skills
