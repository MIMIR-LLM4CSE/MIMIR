---
name: refactor-code
description: Improve code structure without changing its external behavior.
disable-model-invocation: false
---

You are refactoring code.

Rules:
- Preserve external behavior exactly.
- Do NOT change public APIs unless explicitly requested.
- Refactor incrementally.
- Avoid large or sweeping changes.

Workflow:
1. Explain what needs refactoring and why.
2. Apply changes in small steps.
3. Keep code readable and idiomatic.
4. If unsure, prefer simpler structure over abstraction.

If behavior might change, stop and ask the user.
