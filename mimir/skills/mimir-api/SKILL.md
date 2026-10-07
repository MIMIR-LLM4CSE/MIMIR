---
name: mimir-api
description: Explain or author a MIMIR extension — a .mimir skill, MCP server/tool, policy, post-tool hook, nudge or base prompt — using MIMIR's own API.
disable-model-invocation: false
---

The question is about MIMIR itself: what it can be extended with, or a new extension to
write. Read the API from `mimir_api` before answering. The tool reports the build that
is running — its capability vocabulary, its drop-in paths, its shipped templates — and
that is the authority here. What you remember about MIMIR is a previous version.

Workflow:
1. `mimir_api(topic="index")` — the extension types, where each file goes in this
   workspace, and the rules common to all of them. Enough on its own to answer "what
   can I extend?" or "where does X go?".
2. `mimir_api(topic="<type>")` for the one that fits: `skill`, `server`, `policy`,
   `post_tool`, `nudge`, `system_prompt`. You get the contract, the authoring rules and
   the shipped template verbatim — start from that template rather than from memory.
3. `mimir_api(topic="capabilities")` before writing a tool declaration, or a policy or
   nudge that keys off one. The flag list and what each flag drives are live; a flag
   invented from memory silently classifies the tool as nothing.
4. `mimir_api(topic="loaded")` when the answer depends on what is already there — which
   extensions this workspace has, which server namespaces are taken, which bundled
   skill a new one would override.

Choosing the type is most of the work. Match the moment, not the wording of the request:

- it should change **how the agent works** on a kind of task → **skill**
- it should give the agent **new tools** → **MCP server**
- a call must be **refused** before it runs → **policy** (locked, cannot be toggled off)
- something must be **checked after** a call produced a result → **post-tool hook**
- the model should be **reminded**, not stopped → **nudge** (toggleable)
- MIMIR needs **persona or domain knowledge** → **base prompt**

A rule that must always hold is a policy, not a nudge: a nudge the user turns off is a
rule that stops holding. A nudge that must never be ignored was a policy all along.

Writing it:

- The file goes under the workspace `.mimir/` (or the `MIMIR_*_DIR` the API names), and
  nowhere else. Never edit the installed MIMIR package to extend it: an install is
  replaced on upgrade, and a core edit is not an extension. If the user asks for a
  change to MIMIR's own behaviour that no extension type can carry, say so plainly
  instead of reaching into the package.
- Follow the template's structure and the rules the topic lists. They are contracts the
  loader enforces — a skill whose front-matter `name` differs from its directory, or a
  server whose namespace collides with a bundled one, is not loaded, and nothing says so
  at the time the file is written.
- Key a policy or a nudge off **capabilities**, never literal tool names, so it survives
  a rename and applies to any server declaring the capability.
- Declare capabilities on a new tool, and declare `reversibility` rather than
  approval-sensitivity: approval is derived from it.
- Validate the file after writing it: a plugin or server is Python, so a syntax error
  means the pack is skipped at startup with only a log line. Ask the user before
  importing or running one you wrote — importing a plugin pack executes its
  registration.

Finish by telling the user what takes effect when. Extensions are discovered by
directory scan at agent start, so a new or edited file is live on the next start —
reload the VS Code window, or restart the CLI. Say which file you wrote, what it does,
and for a nudge, that `/nudges` turns it off.
