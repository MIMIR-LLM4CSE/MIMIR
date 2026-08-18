# MIMIR Plugin & Extension Guide

> **MIMIR docs** — [Overview](README.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Policy](POLICY.md) · [Client internals](CLIENT_DETAILED.md) · [Servers](SERVERS_DETAILED.md) · [Extension](EXTENSION_DETAILED.md) · [Plugins](PLUGINS_DETAILED.md)

The authoritative guide to **extending MIMIR without editing core code**. Every extension
type is a drop-in file, auto-detected by directory scan under `.mimir/` in your workspace.

A copy-to-customize example of each type ships in [`mimir/examples/`](mimir/examples/).
MIMIR **never writes into your workspace `.mimir/`** — that directory is yours alone. To
add an extension, create the file yourself under `.mimir/` (or point the matching
`MIMIR_*_DIR` env var elsewhere), using the examples as a template.

| Type | Drop-in location | Env override | On name collision with a bundled one |
|------|------------------|--------------|--------------------------------------|
| **Skill** | `.mimir/skills/<name>/SKILL.md` | `MIMIR_SKILLS_DIR` | user **overrides** bundled |
| **MCP server** | `.mimir/servers/server_<name>.py` (or `.js`) | `MIMIR_SERVERS_DIR` | user **skipped** (core protected) |
| **Policy** | `.mimir/plugins/*.py` | `MIMIR_PLUGINS_DIR` | additive (locked) |
| **Nudge** | `.mimir/plugins/*.py` | `MIMIR_PLUGINS_DIR` | additive (toggleable) |
| **Base prompt** | `.mimir/system_prompt.md` | `MIMIR_SYSTEM_PROMPT_FILE` | user **replaces** the built-in default |

> **Golden rule for policies & nudges:** key off **capabilities**
> (from `mimir.client.context.capabilities`), never literal tool names. This keeps a plugin
> robust across tool renames and makes it apply to any server that declares the capability.

---

## Skills

A skill is a reusable methodology prompt injected as a subordinate system message when the
task is detected — it guides *how* the agent works without replacing the base instructions.

`.mimir/skills/<name>/SKILL.md` — a YAML front-matter block (the `name` must equal the
directory name) followed by the methodology body:

```markdown
---
name: fix-bug
description: One-line description shown to the skill classifier.
---

Objective: describe the methodology the agent should follow for this kind of task.

Steps:
1. ...
2. ...
```

Triggered explicitly (`/fix-bug …`) or implicitly by the classifier (current query + the last
few turns). A user skill whose name matches a bundled one **overrides** it. Example:
[`mimir/examples/skills/example-skill/SKILL.md`](mimir/examples/skills/example-skill/SKILL.md).

---

## MCP servers

A custom server is any stdio MCP server; MIMIR auto-connects it at startup. The tool
namespace is the filename stem with any `server_` prefix stripped (`server_weather.py` →
`weather`). A name colliding with a bundled core server is ignored (the core is protected).

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("example")

@mcp.tool()
def greet(name: str) -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()
```

Declaring **capabilities** on your tools (see [Tool capabilities](#tool-capabilities) below)
lets MIMIR's safety policies, caching, and nudges treat them correctly. A foreign server with
no MIMIR metadata is still connected and classified from its standard MCP annotations
(`readOnlyHint` / `destructiveHint`). Example:
[`mimir/examples/servers/server_example.py`](mimir/examples/servers/server_example.py).

---

## Tool capabilities

When you declare a tool you can tag it with one or more **capabilities** — MIMIR's vocabulary
for *what a tool does*. Capabilities drive its safety policies, result caching, workflow
tracking, and nudges. Declaring them is **optional** (MIMIR runs any tool without them and
falls back to the standard MCP `readOnlyHint` / `destructiveHint` annotations), but
**recommended** — and **essential when you write policies or nudges that key off a
capability** (e.g. "block any `EXTERNAL_FETCH` call that carries a secret").

Declare them with `tool_caps(...)` on the server side; query them from a plugin with
`has_cap(tool, CAP, agent.tool_caps)` / `names_with_cap(CAP, agent.tool_caps)`:

```python
from mimir.servers._shared.capabilities import tool_caps, READ, CACHEABLE

@mcp.tool(**tool_caps(caps=[READ, CACHEABLE], path_args=["path"]))
def read_note(path: str) -> str:
    ...
```

| Group | Capability | Meaning |
|-------|------------|---------|
| **Discovery** | `READ` | Reads a file's contents (exploration evidence; a read-before-edit precondition). |
| | `SEARCH` | Searches file contents / patterns (exploration evidence). |
| | `SEARCH_WITH_PATH` | A search whose results embed file paths MIMIR can auto-follow. |
| | `CANDIDATE_SEARCH` | Finds candidate files by name / similarity (drives path clarification). |
| | `INSPECT_DIR` | Lists / inspects a directory (evidence for a safe delete). |
| | `CHECK_EXISTENCE` | Checks whether a path exists. |
| | `CODE_NAV` | Symbol navigation (definition / references / outline). |
| | `ENV_DISCOVERY` | Enumerates Python environments / interpreters (conda / venv). |
| | `CACHEABLE` | Read-only result MIMIR may cache within a query (invalidated on a write). |
| **Mutation** | `EDIT` | Edits / writes a file — enters the edit→validate workflow and dirty-tracking. |
| | `CONTENT_WRITE` | Creates or appends a whole file. |
| | `OVERWRITE` | Replaces an existing file's entire content (needs a prior read). |
| | `REMOVE` | Deletes a file (needs existence + directory context). |
| | `REPLACEMENT_TRACK` | A find / replace whose completeness MIMIR tracks across files. |
| **Validation** | `VALIDATE` | Validates code (syntax / lint / typecheck / test). The first-party stack validates through the `bash` server, but a plugin server may declare its own validator: a successful call marks its target file validated and clears the edit-loop streak. |
| **Planning** | `TASK_PLANNING` | Records a task plan / todo checklist. |
| **Execution** | `CODE_EXEC` | Runs a program / shell command / code payload. Two consequences: the run is recorded as owing a **verdict** on its output (exit 0 alone never validates an execution), and it is a scoping signal for guards. Note the shell-reading guards (proxy direct-execution, out-of-workspace shell paths) key on the `command_prefix` **scope** instead — a tool that executes through structured arguments has no command line for them to parse. |
| | `BACKGROUNDABLE` | Launches a long detached run whose result may carry a `background_job` descriptor (status + summary ops), so a front-end with a worker watches it off the critical path and auto-resumes the agent on completion instead of having the model poll. |
| **Approval & mode** | `SENSITIVE` | Requires user approval before running. **Derived — do not declare it**: see *Reversibility* below. |
| | `NON_BATCH` | Must prompt immediately; never batched. |
| | `PLAN_BLOCKED` | Never callable in a read-only mode — plan **or ask** (writes / exec / mutations). Name kept for compatibility. |
| | `PLAN_READONLY` | Dual-use exec tool allowed in a read-only mode, but **only** for its read-only calls (the classifier decides per call — e.g. a shell tool running `grep` in plan mode, but not `gcc`). |
| **Reach & cost** | `EXTERNAL_FETCH` | Reaches outside the workspace (network / remote API); held until local discovery. |
| | `CLUSTER_SUBMIT` | Expensive cluster launch (Slurm submit / batch run); held until local validation. |
| | `ENV_MUTATE` | Mutates a Python environment (pip install / create / delete). |

This table is the **authoritative list** — 25 flags, mirrored between
`mimir/servers/_shared/capabilities.py` and `mimir/client/context/capabilities.py`
(parity guarded by `test_capabilities.test_vocab_in_sync`). The other docs point here
rather than re-listing it. Note that `is_write` (= `EDIT`∪`CONTENT_WRITE`∪`REMOVE`) and
`clears_edit_loop` (= `READ`∪`VALIDATE`) are **derived helpers, not declarable flags**:
declaring `EDIT` is enough, there is no umbrella to forget.

### Reversibility (and why you never declare `SENSITIVE`)

State **how far your tool's effect can be taken back**; approval-gating follows from it.

```python
from mimir.servers._shared.capabilities import tool_caps, PLAN_BLOCKED, IRREVERSIBLE

@mcp.tool(**tool_caps(caps=[PLAN_BLOCKED], reversibility=IRREVERSIBLE,
                      risk_note="posts to an external service"))
def publish(payload: str) -> str:
    ...
```

| Level | Declare it when… | Effect |
|-------|------------------|--------|
| `REVERSIBLE` | MIMIR itself can undo it (an in-workspace file write it snapshots), or it changes nothing | no approval prompt |
| `RECOVERABLE` | undoable, but by hand — a delete, an environment change, running a payload | approval prompt |
| `IRREVERSIBLE` | it leaves the machine or spends something real — cluster hours, an outbound POST | approval prompt, and no enforcement level can soften it |

Omit the field and MIMIR derives a level from your capabilities (`CLUSTER_SUBMIT` →
irreversible; `REMOVE`/`ENV_MUTATE`/`CODE_EXEC` → recoverable; otherwise reversible).
Declare it whenever the derivation would understate your tool — the derivation cannot
know that your HTTP tool *sends* rather than reads.

A server still passing the older `sensitive=True` keeps its prompt (read as
`RECOVERABLE`), but new tools should state a level: one fact about the effect, rather
than two that can drift apart.

The [`policy`](mimir/examples/plugins/policy_example.py) and
[`nudge`](mimir/examples/plugins/nudge_example.py) examples show capabilities in use.

---

## Policies and nudges (`.mimir/plugins/*.py`)

Both are Python modules that register descriptors as an **import side effect**, via the
public authoring surface `mimir.client.extensions`. They differ in effect and control:

| | Policy (`PolicyCheck`) | Nudge (`NudgeRule`) |
|---|---|---|
| Effect | **Blocks** a tool call (returns a violation string) | Injects an advisory reminder into the conversation |
| Runs at | `pre_mutation` (after registry) or `pre_approval` (after write policy) | `verification` layer (always on) or `guidance` layer (tier-gated) |
| Toggleable | **No — locked/mandatory** | Yes — via `/nudges`, persisted in `<STATE_DIR>/preferences.json` |
| Can it relax a core gate? | **No** — a check only ADDs constraints; `None` abstains | n/a |

The two `pre_mutation` / `pre_approval` slots, and the verification vs guidance layers, are
defined in [`POLICY.md`](POLICY.md) (with zoomable flowcharts). This guide covers *authoring*.

### A policy — block a call

A policy check returns a JSON error string to block, or `None` to abstain. Example:
[`mimir/examples/plugins/policy_example.py`](mimir/examples/plugins/policy_example.py).

```python
import json, re
from mimir.client.extensions import PolicyCheck, register_policy_check
from mimir.client.context.capabilities import EXTERNAL_FETCH, has_cap

_SECRET_RE = re.compile(r"(?i)\b(api[_-]?key|secret|token|password|bearer)\b")

def _no_secret_exfil(agent, tool_name, arguments, execution_context):
    if not has_cap(tool_name, EXTERNAL_FETCH, getattr(agent, "tool_caps", {})):
        return None                       # not an outbound-fetch tool → abstain
    if not any(_SECRET_RE.search(str(k)) or (isinstance(v, str) and _SECRET_RE.search(v))
               for k, v in (arguments or {}).items()):
        return None
    return json.dumps({"status": "error",
                       "error": "Blocked by policy: outbound request carries a credential."})

def register():
    register_policy_check(PolicyCheck(
        name="no_secret_exfil", check=_no_secret_exfil,
        stage="pre_mutation",             # or "pre_approval"
        order=10,
    ))

register()
```

### A nudge — remind, don't block

A nudge fires at most once per step (subject to per-query caps and the enforcement tier). Its
`predicate` decides when to fire; `render` returns the reminder text. Example:
[`mimir/examples/plugins/nudge_example.py`](mimir/examples/plugins/nudge_example.py).

```python
from mimir.client.extensions import NudgeRule, register_nudge
from mimir.client.context.capabilities import SENSITIVE, names_with_cap

def _authz_predicate(agent, query, active_mode, execution_context):
    return bool(names_with_cap(SENSITIVE, getattr(agent, "tool_caps", {})))

def _authz_render(agent, execution_context):
    return ("Before invoking any sensitive capability, confirm the action is authorized "
            "and least-privilege; never transmit credentials in tool arguments.")

def register():
    register_nudge(NudgeRule(
        name="authz_reminder", layer="guidance",
        predicate=_authz_predicate, render=_authz_render,
        # (enforcement, mode) pairs where this fires; omit "off" to stay quiet there.
        tiers=frozenset({("strict", "agent"), ("strict", "plan"),
                         ("light", "agent"), ("light", "plan")}),
    ))

register()
```

Guidance nudges honour the enforcement tier (`strict` / `light` / `off`) via `tiers` and are
suppressed when their name is in `agent.disabled_nudges` (the toggle panel / `/nudges`,
persisted in `<STATE_DIR>/preferences.json` — agent state, not a workspace extension).
Verification-layer nudges (`layer="verification"`) run at every level.

---

## Base prompt (general context)

Replace MIMIR's built-in system prompt with your own, by ONE of:

1. set `MIMIR_SYSTEM_PROMPT_FILE=/abs/path/to/your_context.md`, or
2. drop `.mimir/system_prompt.md` in the workspace root.

Resolution order: `MIMIR_SYSTEM_PROMPT_FILE` → `.mimir/system_prompt.md` → built-in default. The
resolved file's content **replaces** the base prompt; the dynamic platform / memory / todo /
plan sections are still appended on top automatically. Keep it short — the built-in default
(in `client/prompt/system_prompt.build_base_system_content`) is the fallback and a
fuller reference. Skeleton example:
[`mimir/examples/system_prompt.md`](mimir/examples/system_prompt.md).

---

## Where examples live

All examples live solely in [`mimir/examples/`](mimir/examples/) (the single source of
truth), mirroring the `.mimir/` layout with a short `README.md` per extension type. Nothing
is copied into your workspace `.mimir/` — create the drop-in files there yourself using the
examples as templates.
