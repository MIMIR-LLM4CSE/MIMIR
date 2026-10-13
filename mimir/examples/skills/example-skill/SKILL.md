---
name: example-skill
description: One line — when this skill applies. The model reads it to decide whether to load the body.
disable-model-invocation: false
---

Objective: describe the methodology the agent should follow for this kind of task.

Steps:
1. ...
2. ...

Notes:
- The front-matter `name` MUST equal the directory name (`example-skill`).
- The body below the front-matter is injected as a subordinate system message when the
  skill is detected; the base system instructions stay authoritative.
