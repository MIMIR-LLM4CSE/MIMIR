# Custom skills

Skills are methodology prompts that steer how the agent approaches a class of task.

**To add one:** create `.mimir/skills/<skill-name>/SKILL.md` in your workspace (override
the location with the `MIMIR_SKILLS_DIR` env var). It is auto-detected on startup and
merges with the bundled skills; a skill whose name matches a bundled one **overrides** it.

Use [`example-skill/SKILL.md`](example-skill/SKILL.md) here as the template — YAML
front-matter (the `name` must equal the directory name) followed by the methodology body.

Only the `name` and `description` reach the system prompt; the body is read on demand —
by `/<skill-name>`, or by the model itself with `load_skill(<skill-name>)` when its own
reading says the method applies. Write the `description` as *when this applies*, since it
is the whole basis for that decision. Set `disable-model-invocation: true` to keep a skill
for `/<skill-name>` only.

See [`PLUGINS_DETAILED.md`](../../../PLUGINS_DETAILED.md) for the full authoring guide.
