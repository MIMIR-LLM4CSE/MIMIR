"""
MCP Todo Server
===============
Live task checklist for the agent — lets the model actively manage its
execution plan rather than relying on a static system-prompt injection.

Stored per-session under the central state dir (<STATE_DIR>/sessions/<sid>/todo_list.md)
as a markdown checkbox list:

    - [ ] pending step
    - [x] completed step

Named prose plans are stored as a per-session history under plans/ (one
<date>-<slug>.md file each, indexed by PLANS.md, with a .active pointer):

    todo_set_plan(text, title) — write a named plan, making it the active one
    todo_read_plan(name=None)  — read the active (or a named) plan back
    todo_list_plans()          — browse the plan history

Tools:
  1. todo_set_plan(text, title) — write the named approach/rationale before starting
  2. todo_write(steps)          — replace the entire checklist with new steps, once the
                                  plan it comes from has been approved and work starts
  3. todo_read()                — return the current list (index, text, done)
  4. todo_read_plan(name)       — return the active or a named prose plan
  5. todo_list_plans()          — list the session's plan history
  6. todo_delete_plan(name)     — delete one plan from the history
  7. todo_update(index, done)   — mark one item done/undone
"""

import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import TASK_PLANNING, tool_caps, RECOVERABLE
from responses import err, ok
from state_paths import state_dir
from text_tools import yaml_scalar, yaml_unquote

# Central per-workspace state dir (MIMIR_STATE_DIR, set by server_manager; legacy
# <workspace>/.mimir fallback for standalone/tests). Todos and plans are per-session
# sidecars under sessions/<sid>/; the active-session pointer and legacy shared todo
# live at the state-dir root. See state_paths.
_MIMIR_DIR = state_dir()
# Legacy fallback path (used when no active session sidecar exists).
_LEGACY_TODO_FILE = os.path.abspath(os.path.join(_MIMIR_DIR, "todo_list.md"))
_ACTIVE_SESSION_FILE = os.path.abspath(os.path.join(_MIMIR_DIR, "active_session"))


def _get_todo_file() -> str:
    """Return the todo file path for the currently active session.

    Reads .mimir/active_session (written by ws_server.py on each session
    switch) to find the session-scoped path.  Falls back to the legacy shared
    todo_list.md when no sidecar exists (e.g. CLI mode).
    """
    try:
        if os.path.exists(_ACTIVE_SESSION_FILE):
            with open(_ACTIVE_SESSION_FILE, "r", encoding="utf-8") as f:
                session_id = f.read().strip()
            if session_id:
                session_dir = os.path.join(os.path.dirname(_LEGACY_TODO_FILE), "sessions", session_id)
                os.makedirs(session_dir, exist_ok=True)
                return os.path.join(session_dir, "todo_list.md")
    except OSError:
        pass
    return _LEGACY_TODO_FILE


# ── named plan history ──────────────────────────────────────────────────────
# Plans live under sessions/<sid>/plans/ as one <YYYY-MM-DD>-<slug>.md file each
# (frontmatter title/date + body), indexed by PLANS.md, with a .active pointer to
# the current one. This keeps a readable history instead of overwriting a single
# plan.md — the same model as the agent's memory.

_PLAN_FM_RE = re.compile(r'^---\n(.*?)\n---\n?(.*)$', re.DOTALL)


def _plans_dir() -> str:
    return os.path.join(os.path.dirname(_get_todo_file()), "plans")


def _plan_index_file() -> str:
    return os.path.join(_plans_dir(), "PLANS.md")


def _active_plan_pointer() -> str:
    return os.path.join(_plans_dir(), ".active")


def _slugify(text: str) -> str:
    """Kebab-case slug from a title, capped for filesystem friendliness."""
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return '-'.join(w for w in slug.split('-') if w)[:60].strip('-') or "plan"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_display() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _parse_plan(content: str) -> dict:
    """Split a plan file into frontmatter (title/date) and body."""
    m = _PLAN_FM_RE.match(content)
    if not m:
        return {"title": "", "date": "", "text": content.strip()}
    head, body = m.group(1), m.group(2)
    fields = {"title": "", "date": ""}
    for line in head.splitlines():
        key, _, val = line.partition(":")
        key = key.strip()
        if key in ("title", "date"):
            fields[key] = yaml_unquote(val)
    fields["text"] = body.strip()
    return fields


def _read_active_name() -> str | None:
    try:
        with open(_active_plan_pointer(), "r", encoding="utf-8") as f:
            name = f.read().strip()
        return name or None
    except OSError:
        return None


def _load_plans() -> list[dict]:
    """Every plan in the active session's plans/ dir, newest first."""
    pdir = _plans_dir()
    if not os.path.isdir(pdir):
        return []
    active = _read_active_name()
    plans = []
    for fname in os.listdir(pdir):
        if not fname.endswith(".md") or fname == "PLANS.md":
            continue
        try:
            with open(os.path.join(pdir, fname), "r", encoding="utf-8") as f:
                fields = _parse_plan(f.read())
        except OSError:
            continue
        name = fname[:-3]
        plans.append({
            "name": name,
            "title": fields.get("title") or name,
            "date": fields.get("date", ""),
            "text": fields.get("text", ""),
            "active": name == active,
        })
    plans.sort(key=lambda p: (p.get("date", ""), p["name"]), reverse=True)
    return plans


def _write_plan_index() -> None:
    plans = _load_plans()
    lines = ["# Plans", ""]
    for p in plans:
        mark = " ← active" if p["active"] else ""
        lines.append(f"- [{p['title']}]({p['name']}.md) — {p['date'][:10]}{mark}")
    os.makedirs(_plans_dir(), exist_ok=True)
    with open(_plan_index_file(), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

mcp = FastMCP(
    "TodoServer",
    debug=False,
    log_level="ERROR",
)

_CHECKBOX_RE = re.compile(r"^- \[([ x])\] (.+)$")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load() -> list[dict]:
    todo_file = _get_todo_file()
    if not os.path.exists(todo_file):
        return []
    try:
        with open(todo_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    items = []
    for line in lines:
        m = _CHECKBOX_RE.match(line.rstrip("\n"))
        if m:
            items.append({
                "index": len(items),
                "text": m.group(2),
                "done": m.group(1) == "x",
            })
    return items


def _save(items: list[dict]) -> None:
    todo_file = _get_todo_file()
    os.makedirs(os.path.dirname(todo_file), exist_ok=True)
    with open(todo_file, "w", encoding="utf-8") as f:
        for it in items:
            mark = "x" if it.get("done") else " "
            f.write(f"- [{mark}] {it['text']}\n")


def _deps_file() -> str:
    return os.path.join(os.path.dirname(_get_todo_file()), "todo_deps.json")


def _load_deps() -> list[list[int]]:
    """Return the per-step dependency matrix, or [] if not set."""
    import json
    path = _deps_file()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_deps(deps: list[list[int]]) -> None:
    import json
    path = _deps_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(deps, f)


def _delete_deps() -> None:
    path = _deps_file()
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ── tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(**tool_caps(
    caps=[TASK_PLANNING],
    # Two roles, because two client-side consumers need to find different arguments
    # without knowing this tool's name: the plan loop pins the title so a revision
    # overwrites in place, and the plan-shape gate reads the body.
    arg_roles={"plan_title": ["title"], "plan_document": ["text"]},
))
def todo_set_plan(text: str, title: str) -> dict:
    """Write a named prose plan (approach and rationale) for the current task.

    Call this BEFORE todo_write when tackling a multi-step task — in plan mode it is
    the only plan tool available, since the checklist is written once the user has
    approved this document. Write it only once you have finished exploring: the
    plan rests on code you have read, and exploring, surveying, examining and
    identifying gaps are what you did to write it, never steps or axes of it. This
    written plan is what the user reads to understand and approve the work, so make
    it a short design note, not a bare list of steps. Structure it with Markdown
    headings and a few sentences of prose under each:

      * ## Overview — the goal and the outcome the change produces.
      * ## Approach — broken into the main axes of the work (one subsection per
        component, layer, or concern). Each axis is a change to make: explain WHAT
        changes, WHERE (concrete files/symbols you located while exploring), and
        WHY this approach over the alternatives.
      * ## Decisions & risks — key design decisions, trade-offs, assumptions,
        and edge cases worth calling out.
      * ## Validation — how the result will be checked (tests, manual steps).

    Scale the depth to the task: a trivial change needs only a couple of
    paragraphs, a larger one warrants several axes with explanation. Each plan is
    stored as its own file (plans/<date>-<slug>.md) and becomes the active plan;
    the history is kept rather than overwritten. Re-call with the same title on
    the same day to refine that plan in place; use a new title to start a
    distinct plan.

    Args:
        text:  Free-form markdown describing the plan, approach, and reasoning,
               structured as above. No length limit, but keep it focused.
        title: Short human title for the plan (e.g. "Refactor state dir"). Used
               verbatim in the index and slugified for the filename.

    Returns:
        {"status": "ok", "name": <slug>, "bytes": <n>}
    """
    if not title or not title.strip():
        return err("title is required — give the plan a short descriptive title.")
    name = f"{_today()}-{_slugify(title)}"
    pdir = _plans_dir()
    os.makedirs(pdir, exist_ok=True)
    body = text.strip()
    content = (
        f"---\ntitle: {yaml_scalar(title)}\ndate: {yaml_scalar(_now_display())}\n"
        f"---\n\n{body}\n"
    )
    plan_path = os.path.abspath(os.path.join(pdir, f"{name}.md"))
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(content)
    with open(_active_plan_pointer(), "w", encoding="utf-8") as f:
        f.write(name)
    _write_plan_index()
    # `path` + `open_in_editor` are a UI hint: the client emits an `open_editor`
    # event on this shape so the plan .md pops open in the editor for the user to
    # read. The model can ignore both fields.
    return ok({
        "name": name,
        "bytes": len(body.encode("utf-8")),
        "path": plan_path,
        "open_in_editor": True,
    })


@mcp.tool()
def todo_read_plan(name: str = None) -> dict:
    """Return a prose plan written by todo_set_plan.

    Reads the currently active plan when no name is given, otherwise the named
    plan (as returned by todo_list_plans). Call todo_list_plans() to browse the
    history.

    Args:
        name: Optional plan slug (e.g. "2026-07-09-refactor-state-dir"). Omit for
              the active plan.

    Returns:
        {"status": "ok", "name": <slug>, "title": <title>, "date": <date>,
         "text": <plan body>}  or  {"status": "ok", "text": ""} if none exists.
    """
    target = name or _read_active_name()
    if not target:
        return ok({"text": ""})
    target = target[:-3] if target.endswith(".md") else target
    plan_file = os.path.join(_plans_dir(), f"{target}.md")
    if not os.path.exists(plan_file):
        return err(
            f"No plan named {target!r}.",
            hint="Call todo_list_plans() to see available plans.",
        )
    try:
        with open(plan_file, "r", encoding="utf-8") as f:
            fields = _parse_plan(f.read())
    except OSError as exc:
        return err(str(exc))
    return ok({
        "name": target,
        "title": fields.get("title", ""),
        "date": fields.get("date", ""),
        "text": fields.get("text", ""),
    })


@mcp.tool()
def todo_list_plans() -> dict:
    """List every plan in the active session, newest first.

    Returns:
        {"status": "ok", "plans": [{"name", "title", "date", "active"}], "count": <n>}
    """
    plans = _load_plans()
    summary = [
        {"name": p["name"], "title": p["title"], "date": p["date"], "active": p["active"]}
        for p in plans
    ]
    return ok({"plans": summary, "count": len(summary)})


@mcp.tool(**tool_caps(
    reversibility=RECOVERABLE, non_batch=True,
    risk_note="deletes a persistent plan file",
))
def todo_delete_plan(name: str) -> dict:
    """Delete one plan by its slug name (the file's <name>.md).

    Call todo_list_plans() to see available names. If the deleted plan was the
    active one, the active pointer moves to the newest remaining plan (or is
    cleared when none remain). The index is rewritten.

    Args:
        name: Slug name of the plan to remove (without the .md extension).

    Returns:
        {"status": "ok", "deleted": <name>} or {"status": "error", ...}.
    """
    target = name[:-3] if name.endswith(".md") else name
    plan_file = os.path.join(_plans_dir(), f"{target}.md")
    if not os.path.exists(plan_file):
        return err(
            f"No plan named {target!r}.",
            hint="Call todo_list_plans() to see available plans.",
        )
    was_active = _read_active_name() == target
    try:
        os.remove(plan_file)
    except OSError as exc:
        return err(f"Could not delete {target!r}: {exc}")

    if was_active:
        remaining = _load_plans()  # newest first, excludes the just-deleted file
        pointer = _active_plan_pointer()
        if remaining:
            with open(pointer, "w", encoding="utf-8") as f:
                f.write(remaining[0]["name"])
        else:
            try:
                os.remove(pointer)
            except OSError:
                pass
    _write_plan_index()
    return ok({"deleted": target})


@mcp.tool(**tool_caps(caps=[TASK_PLANNING], arg_roles={"plan_steps": ["steps"]}))
def todo_write(steps: list[str], depends_on: list[list[int]] | None = None) -> dict:
    """Replace the entire todo list with a new ordered list of steps.

    Each step starts as not done.  For multi-step tasks, call todo_set_plan
    first to record your prose rationale and — in plan mode — get it approved,
    then call this tool with the concrete ordered steps before starting the work.
    Use todo_update() to mark items complete one at a time as
    you finish each step.

    Args:
        steps:      Ordered list of task descriptions.
        depends_on: Optional per-step dependency list.  Each entry is a list of
                    step indices that must be done before this step can start.
                    Example: [[],[],[0,1]] means step 2 depends on steps 0 and 1.
                    Omit or pass null for a plain sequential list.

    Returns:
        {"status": "ok", "count": <n>}
    """
    items = [{"index": i, "text": s.strip(), "done": False} for i, s in enumerate(steps)]
    _save(items)
    if depends_on is not None:
        _save_deps(depends_on)
    else:
        _delete_deps()
    return ok({"count": len(items)})


@mcp.tool()
def todo_read() -> dict:
    """Return the current todo list.

    Returns:
        {"status": "ok", "items": [...], "pending": <n>, "done": <n>}
    """
    items = _load()
    pending = sum(1 for it in items if not it.get("done"))
    done = len(items) - pending
    return ok({"items": items, "pending": pending, "done": done})


@mcp.tool()
def todo_read_ready() -> dict:
    """Return todo items that are ready to execute right now.

    An item is ready when it is not yet done AND all its dependencies
    (as declared in todo_write depends_on) are already done.  Items with
    no dependencies are always ready (if not done).

    Use this tool to discover which steps can be executed in parallel via
    spawn_agent — call spawn_agent once per ready item within a single
    response to run them concurrently.

    Returns:
        {"status": "ok", "ready": [...], "blocked": [...], "pending": <n>}
    """
    items = _load()
    deps = _load_deps()   # list[list[int]], one entry per step index

    ready = []
    blocked = []
    for item in items:
        if item.get("done"):
            continue
        idx = item["index"]
        step_deps = deps[idx] if idx < len(deps) else []
        if all(items[d].get("done") for d in step_deps if d < len(items)):
            ready.append(item)
        else:
            blocked.append(item)

    return ok({"ready": ready, "blocked": blocked, "pending": len(ready) + len(blocked)})


@mcp.tool()
def todo_update(index: int, done: bool) -> dict:
    """Mark a todo item as done or undone.

    Only call with done=True AFTER the step is fully complete and verified on
    disk (e.g. the file was written, the directory exists, the test passed).
    Never mark a step done speculatively, based on intent, or mid-way through
    the step.  If you are unsure whether the output exists, verify first.

    Args:
        index: Zero-based index of the item to update.
        done:  True to mark complete, False to reopen.

    Returns:
        {"status": "ok", "item": <updated entry>, "pending": <remaining count>}
    """
    items = _load()
    if index < 0 or index >= len(items):
        return err(f"Index {index} out of range (list has {len(items)} items).")
    items[index]["done"] = done
    _save(items)
    pending = sum(1 for it in items if not it.get("done"))
    return ok({"item": items[index], "pending": pending})


# ── resource ──────────────────────────────────────────────────────────────────

@mcp.resource(
    "todo://current",
    name="todo",
    description="The current task checklist / plan (attach with @todo).",
)
def todo_current() -> str:
    """Render the current checklist as Markdown."""
    todo_file = _get_todo_file()
    if not os.path.exists(todo_file):
        return "(no tasks)"
    with open(todo_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content or "(no tasks)"


if __name__ == "__main__":
    mcp.run()
