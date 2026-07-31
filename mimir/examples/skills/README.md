# Custom skills

Skills are methodology prompts that steer how the agent approaches a class of task.

**To add one:** create `.mimir/skills/<skill-name>/SKILL.md` in your workspace (override
the location with the `MIMIR_SKILLS_DIR` env var). It is auto-detected on startup and
merges with the bundled skills; a skill whose name matches a bundled one **overrides** it.

Use [`example-skill/SKILL.md`](example-skill/SKILL.md) here as the template — YAML
front-matter (the `name` must equal the directory name) followed by the methodology body.

See [`PLUGINS_DETAILED.md`](../../../PLUGINS_DETAILED.md) for the full authoring guide.
