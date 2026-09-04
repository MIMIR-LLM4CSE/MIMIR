---
name: explore-repo
description: Systematically explore and understand a code repository before acting.
disable-model-invocation: false
---

Build an accurate map of the repository before acting on it. Scale the sweep to the
question asked: a located change needs the code it touches, not a tour of the project.

Workflow:
1. Identify the project type, language, and structure.
2. Locate the entry points, configuration files, and modules the task actually touches.
3. Read the documentation that covers those, not the documentation set at large.
4. Name the concrete files and symbols involved, and what happens to each.

Rules:
- Search for the symbol first, then read the section around the match. Do not page
  through whole files, and do not read every file to find one.
- Split independent strands into parallel read-only sub-agents rather than sweeping
  every area yourself.
- Stop at step 4. Once you can name the files and symbols, exploration is finished and
  the work moves on; reading further is cost, not evidence.
- Explore to decide, not to report. A standalone summary is owed only when the user
  asked for the analysis itself, or when what you found changes the approach.
- Ask a clarification question when the intent is ambiguous — not when it is merely
  unfamiliar.