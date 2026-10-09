---
name: explore-repo
description: Build an accurate map of unfamiliar code before acting on it — and stop as soon as the files and symbols the task touches can be named.
disable-model-invocation: false
---

Exploration has two failure modes, and the expensive one is not ignorance. Acting on a
repository you have not read breaks things; reading a repository you have already
understood spends the whole task on reconnaissance. Scale the sweep to the question
asked: a located change needs the code it touches, not a tour of the project.

## Order of work

1. Identify the project type, language, and structure.
2. Locate the entry points, configuration files, and modules the task **actually**
   touches.
3. Read the documentation that covers those, not the documentation set at large.
4. Name the concrete files and symbols involved, and what happens to each.

Stop at step 4. Once you can name them, exploration is finished and the work moves on;
reading further is cost, not evidence.

## How to read

Search for the symbol first, then read the section around the match. Never page through
a whole file to find one definition, and never read every file in a tree to find one.
A read that stops short tells you the file's length, the line to resume from, and the
symbols it holds with their line numbers — use that map to jump, rather than walking the
file.

Split independent strands into parallel read-only sub-agents instead of sweeping every
area yourself. Each comes back with a conclusion; you keep the one map.

Never assume the domain semantics of what you are reading — what a symbol denotes, its
units, sign and time conventions, its indexing and memory layout, which formulation of a
model is implemented. Read the definition, the documentation, or the caller that fixes
it, and if nothing does, ask. An assumed convention survives every later check.

## What exploration is for

Explore to decide, not to report. A standalone summary is owed only when the user asked
for the analysis itself, or when what you found changes the approach — in which case say
what changed and why, not everything you read on the way.

Ask a clarification question when the intent is genuinely ambiguous, not when it is
merely unfamiliar. Unfamiliar is what the reading is for.
