---
name: fix-bug
description: Identify, fix, and validate a bug with minimal and safe changes.
disable-model-invocation: false
---

Objective: Fix a specific bug with minimal impact.

Method:
1. Reproduce or explain the bug.
2. Identify the root cause.
3. Modify only the relevant code paths.
4. Add or update tests if possible.
5. Validate the fix.

Rules:
- Make the smallest possible change.
- Do NOT refactor unrelated code.
- Explain why the fix works.
- Prefer correctness over cleverness.

If the bug cannot be reproduced, state it explicitly.