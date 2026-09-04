"""
MCP Memory Server
=================
Persistent, timestamped memory for the agent — Claude-style.

Storage layout (under the central per-workspace state dir, <STATE_DIR>/memory/;
shared across all sessions of the workspace):
  MEMORY.md        — the index: one line per memory, loaded into context each
                     session. Format: ``- [<description>](<slug>.md) — <date>``
  <slug>.md        — one memory per file, human-editable Markdown with a small
                     frontmatter block (name / description / date / tags) + body.

Workflow:
  1. memory_add(text, description?, tags?)  — store a fact as its own .md file
  2. memory_search(query)                   — retrieve relevant memories (substring)
  3. memory_list_all()                      — list every stored memory
  4. memory_delete(name)                    — remove one memory file by its slug
  5. memory_clear()                         — wipe all memory (irreversible)
"""

import os
import json
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps, PLAN_BLOCKED, RECOVERABLE
from responses import err, ok
from state_paths import state_dir
from text_tools import yaml_scalar, yaml_unquote
import embed as _embed

# Memory lives at the root of the central per-workspace state dir (MIMIR_STATE_DIR,
# set by server_manager; legacy <workspace>/.mimir fallback for standalone/tests),
# so it is shared across all sessions of the workspace — not scoped to a session.
MEMORY_DIR = os.path.join(state_dir(), "memory")
INDEX_FILE = os.path.join(MEMORY_DIR, "MEMORY.md")
# Parallel embedding cache (slug -> {"model": <id>, "vec": [...]}), kept out of the
# human-readable .md files. Makes memory_search semantic; absent/stale entries fall
# back transparently to substring search.
EMBEDDINGS_FILE = os.path.join(MEMORY_DIR, "embeddings.json")

_MAX_MEMORY_TEXT_LEN = 2000  # characters
_MAX_DESCRIPTION_LEN = 120   # characters (index stays scannable)
_MAX_ENTRIES = 50            # oldest memories are pruned beyond this
_DEDUP_WINDOW = 5            # compare against the N most recent memories
_DEDUP_THRESHOLD = 0.70      # Jaccard word-overlap ratio above which entry is skipped

mcp = FastMCP(
    "MemoryServer",
    debug=False,
    log_level="ERROR",
)


# ── helpers ───────────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---\n?(.*)$', re.DOTALL)


def _now_display() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _slugify(text: str) -> str:
    """Kebab-case slug from free text, capped for filesystem friendliness."""
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    words = slug.split('-')
    slug = '-'.join(w for w in words if w)[:60].strip('-')
    return slug or "memory"


def _derive_description(text: str) -> str:
    """First sentence / line of the body, trimmed to a scannable one-liner."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    # Prefer the first sentence when it is short enough to read at a glance.
    sentence = re.split(r'(?<=[.!?])\s', first)[0]
    desc = (sentence if len(sentence) <= _MAX_DESCRIPTION_LEN else first).strip()
    if len(desc) > _MAX_DESCRIPTION_LEN:
        desc = desc[:_MAX_DESCRIPTION_LEN - 1].rstrip() + "…"
    return desc or "memory"


def _unique_slug(base: str, existing: set[str]) -> str:
    slug = base
    n = 2
    while slug in existing:
        slug = f"{base}-{n}"
        n += 1
    return slug


def _parse_frontmatter(content: str) -> dict:
    """Split a memory file into its frontmatter fields and body text."""
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return {"text": content.strip(), "description": "", "tags": [], "date": ""}
    head, body = m.group(1), m.group(2)
    fields: dict = {"tags": []}
    for line in head.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key == "tags":
            val = val.strip("[]")
            fields["tags"] = [yaml_unquote(t) for t in val.split(",") if t.strip()]
        else:
            fields[key] = yaml_unquote(val)
    fields["text"] = body.strip()
    return fields


def _serialize(entry: dict) -> str:
    tags = entry.get("tags", [])
    tags_line = "[" + ", ".join(yaml_scalar(t) for t in tags) + "]"
    return (
        "---\n"
        f"name: {yaml_scalar(entry['name'])}\n"
        f"description: {yaml_scalar(entry.get('description', ''))}\n"
        f"date: {yaml_scalar(entry.get('date', _now_display()))}\n"
        f"tags: {tags_line}\n"
        "---\n\n"
        f"{entry.get('text', '').strip()}\n"
    )


def _load() -> list:
    """Read every memory file, newest last (sorted by date then slug)."""
    if not os.path.isdir(MEMORY_DIR):
        return []
    entries = []
    for fname in os.listdir(MEMORY_DIR):
        if not fname.endswith(".md") or fname == "MEMORY.md":
            continue
        path = os.path.join(MEMORY_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        fields = _parse_frontmatter(content)
        entries.append({
            "name": fields.get("name") or fname[:-3],
            "description": fields.get("description", ""),
            "date": fields.get("date", ""),
            "tags": fields.get("tags", []),
            "text": fields.get("text", ""),
        })
    entries.sort(key=lambda e: (e.get("date", ""), e.get("name", "")))
    return entries


def _write_index(entries: list) -> None:
    """Rewrite MEMORY.md — the human/agent-scannable index, newest first."""
    lines = ["# Memory Index", ""]
    for e in sorted(entries, key=lambda x: (x.get("date", ""), x.get("name", "")), reverse=True):
        desc = e.get("description") or e.get("name", "")
        date = (e.get("date") or "")[:10]
        lines.append(f"- [{desc}]({e['name']}.md) — {date}")
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _word_set(text: str) -> set[str]:
    return set(re.findall(r'[a-zA-Z0-9_]+', text.lower()))


def _is_near_duplicate(text: str, recent_entries: list) -> str | None:
    """Return the slug of a near-duplicate recent memory, or None."""
    words = _word_set(text)
    if not words:
        return None
    for entry in recent_entries[-_DEDUP_WINDOW:]:
        other = _word_set(entry.get("text", ""))
        if not other:
            continue
        intersection = len(words & other)
        union = len(words | other)
        if union > 0 and intersection / union >= _DEDUP_THRESHOLD:
            return entry.get("name")
    return None


# ── embedding cache ─────────────────────────────────────────────────────────────

def _embed_input(entry: dict) -> str:
    """Text embedded for a memory: its one-line description plus the full body."""
    return f"{entry.get('description', '')}\n{entry.get('text', '')}".strip()


def _load_embeddings() -> dict:
    """Read the parallel embedding cache, or {} when absent/corrupt."""
    try:
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_embeddings(store: dict) -> None:
    try:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f)
    except OSError:
        pass


def _upsert_embedding(name: str, entry: dict) -> None:
    """Compute and persist the embedding for one memory. No-op when the embedding
    backend is unavailable — search then falls back to substring matching."""
    if not _embed.is_available():
        return
    vec = _embed.embed_one(_embed_input(entry))
    if vec is None:
        return
    store = _load_embeddings()
    store[name] = {"model": _embed.embed_model_id(), "vec": vec}
    _save_embeddings(store)


def _prune_embeddings(names) -> None:
    """Drop cached vectors for the given slugs (after delete / aging trim)."""
    names = set(names)
    if not names:
        return
    store = _load_embeddings()
    if any(n in store for n in names):
        for n in names:
            store.pop(n, None)
        _save_embeddings(store)


def _semantic_search(query: str, candidates: list, limit: int) -> list | None:
    """Rank *candidates* by embedding similarity to *query*.

    Returns a list of memories (each with a "score") or ``None`` to signal the
    caller to fall back to substring search. Vectors missing from the cache (or
    embedded under a different model) are backfilled and persisted on the fly.
    """
    if not _embed.is_available():
        return None
    store = _load_embeddings()
    model = _embed.embed_model_id()

    vecs: list = []
    idx_map: list = []
    missing: list = []
    for i, m in enumerate(candidates):
        rec = store.get(m["name"])
        if rec and rec.get("model") == model and rec.get("vec"):
            vecs.append(rec["vec"])
            idx_map.append(i)
        else:
            missing.append(i)

    if missing:
        new_vecs = _embed.embed_texts([_embed_input(candidates[i]) for i in missing])
        if new_vecs:
            for j, i in enumerate(missing):
                store[candidates[i]["name"]] = {"model": model, "vec": new_vecs[j]}
                vecs.append(new_vecs[j])
                idx_map.append(i)
            _save_embeddings(store)

    if not vecs:
        return None
    qvec = _embed.embed_one(query)
    if qvec is None:
        return None

    results = []
    for pos, score in _embed.cosine_rank(qvec, vecs)[:limit]:
        m = candidates[idx_map[pos]]
        results.append({**m, "score": round(score, 4)})
    return results


# ── tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def memory_add(text: str, description: str = None, tags: list = None) -> dict:
    """Store a fact as its own timestamped Markdown memory file.

    Each memory is written to .mimir/memory/<slug>.md with a short frontmatter
    (name, description, date, tags) and indexed in MEMORY.md. Use this to remember
    user preferences, facts discovered during a task, or decisions worth recalling.

    Keep entries concise (under 2000 characters) — store key facts, file paths,
    and decisions, not raw conversation text.

    Returns {"status": "ok", "name": <slug>, "stored": <text>}.

    Args:
        text: The fact or note to remember (max 2000 characters).
        description: Optional one-line summary for the index. Derived from the
            text's first sentence when omitted.
        tags: Optional list of tag strings, e.g. ["user", "preference"].
    """
    if len(text) > _MAX_MEMORY_TEXT_LEN:
        return err(
            f"text is too long ({len(text)} chars, max {_MAX_MEMORY_TEXT_LEN}). "
            "Summarise to key facts, file paths, and decisions before storing.",
            hint=(
                "Break the content into multiple short memories, one per distinct fact "
                "(e.g. one for file paths, one for user preferences, one for conclusions). "
                "Never store raw conversation text verbatim."
            ),
        )
    memory = _load()

    dup = _is_near_duplicate(text, memory)
    if dup is not None:
        return ok({
            "name": None,
            "stored": text,
            "note": "skipped: near-duplicate of a recent memory",
            "similar_memory": dup,
        })

    desc = (description or "").strip() or _derive_description(text)
    if len(desc) > _MAX_DESCRIPTION_LEN:
        desc = desc[:_MAX_DESCRIPTION_LEN - 1].rstrip() + "…"

    existing = {e["name"] for e in memory}
    name = _unique_slug(_slugify(desc), existing)

    entry = {
        "name":        name,
        "description": desc,
        "date":        _now_display(),
        "tags":        tags or [],
        "text":        text,
    }

    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(os.path.join(MEMORY_DIR, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(_serialize(entry))
    memory.append(entry)

    # Aging: trim oldest memories beyond the cap.
    if len(memory) > _MAX_ENTRIES:
        stale_names = []
        for stale in memory[:len(memory) - _MAX_ENTRIES]:
            try:
                os.remove(os.path.join(MEMORY_DIR, f"{stale['name']}.md"))
            except OSError:
                pass
            stale_names.append(stale["name"])
        memory = memory[len(memory) - _MAX_ENTRIES:]
        _prune_embeddings(stale_names)

    _write_index(memory)
    _upsert_embedding(name, entry)
    return ok({"name": name, "stored": text})


@mcp.tool()
def memory_search(query: str, tag: str = None, limit: int = 5) -> dict:
    """Search memories by meaning, ranked most-relevant first.

    When an embedding backend is available, memories are ranked by semantic
    similarity to the query (so reworded, synonymous, or other-language queries
    still match). Otherwise it falls back to a case-insensitive substring match.
    Optionally filter by a specific tag.

    Returns {"status": "ok", "results": [<matching memories>], "count": <n>}.
    Each result carries a "score" (semantic similarity, or None for substring
    matches). If no results, "count" will be 0 — try a shorter query or
    memory_list_all().

    Args:
        query: What to look for (natural language; not just a substring).
        tag:   If provided, only memories with this tag are searched.
        limit: Maximum number of results to return (default 5).
    """
    memory = _load()
    candidates = [m for m in memory if tag is None or tag in m.get("tags", [])]
    if not candidates:
        return ok({"results": [], "count": 0})

    results = _semantic_search(query, candidates, limit)
    if results is None:
        # Fallback: original case-insensitive substring behaviour.
        q = query.lower()
        results = [
            {**m, "score": None}
            for m in candidates
            if q in m["text"].lower() or q in m.get("description", "").lower()
        ][:limit]
    return ok({"results": results, "count": len(results)})


@mcp.tool()
def memory_list_all() -> dict:
    """Return every stored memory with its name, description, date, tags, and text.

    Use this to get full context before deciding what to recall or delete.
    Returns {"status": "ok", "memory": [<all memories>], "count": <n>}.
    """
    memory = _load()
    return ok({"memory": memory, "count": len(memory)})


@mcp.tool()
def memory_update(
    name: str,
    text: str = None,
    description: str = None,
    tags: list = None,
) -> dict:
    """Edit an existing memory in place, keyed by its slug name.

    Only the provided fields are changed; omitted fields are preserved. The slug
    (filename) stays stable so the index link remains valid, and the date is
    refreshed to now. Call memory_list_all() to see available names.

    Returns {"status": "ok", "name": <slug>, "updated": <memory>} or an error.

    Args:
        name: Slug name of the memory to edit (without the .md extension).
        text: New body text (max 2000 characters). Unchanged when omitted.
        description: New one-line index summary. Unchanged when omitted.
        tags: New list of tags, replacing the old ones. Unchanged when omitted.
    """
    if text is not None and len(text) > _MAX_MEMORY_TEXT_LEN:
        return err(
            f"text is too long ({len(text)} chars, max {_MAX_MEMORY_TEXT_LEN}). "
            "Summarise to key facts, file paths, and decisions before storing.",
        )
    memory = _load()
    entry = next((e for e in memory if e["name"] == name), None)
    if entry is None:
        return err(
            f"No memory named {name!r}.",
            hint="Call memory_list_all() to see valid names, or memory_add() to create one.",
        )

    if text is not None:
        entry["text"] = text
    if description is not None:
        desc = description.strip()
        if len(desc) > _MAX_DESCRIPTION_LEN:
            desc = desc[:_MAX_DESCRIPTION_LEN - 1].rstrip() + "…"
        entry["description"] = desc
    if tags is not None:
        entry["tags"] = tags
    entry["date"] = _now_display()

    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(os.path.join(MEMORY_DIR, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(_serialize(entry))
    _write_index(memory)
    _upsert_embedding(name, entry)
    return ok({"name": name, "updated": entry})


@mcp.tool(**tool_caps(
    caps=[PLAN_BLOCKED], reversibility=RECOVERABLE, non_batch=True,
    risk_note="deletes a persistent memory file",
))
def memory_delete(name: str) -> dict:
    """Delete one memory by its slug name (the file's <name>.md).

    Call memory_list_all() to see available names. To remove everything use
    memory_clear().

    Returns {"status": "ok", "deleted": <memory>} or {"status": "error", ...}.

    Args:
        name: Slug name of the memory to remove (without the .md extension).
    """
    memory = _load()
    match = next((e for e in memory if e["name"] == name), None)
    if match is None:
        return err(
            f"No memory named {name!r}.",
            hint="Call memory_list_all() to see valid names.",
        )
    try:
        os.remove(os.path.join(MEMORY_DIR, f"{name}.md"))
    except OSError as exc:
        return err(f"Could not delete {name!r}: {exc}")
    memory = [e for e in memory if e["name"] != name]
    _write_index(memory)
    _prune_embeddings({name})
    return ok({"deleted": match})


@mcp.tool(**tool_caps(
    caps=[PLAN_BLOCKED], reversibility=RECOVERABLE, non_batch=True,
    risk_note="wipes all persistent memory files",
))
def memory_clear() -> dict:
    """Wipe all stored memories. This action is irreversible.

    Returns {"status": "ok", "cleared": <number of memories removed>}.
    """
    memory = _load()
    count = len(memory)
    for e in memory:
        try:
            os.remove(os.path.join(MEMORY_DIR, f"{e['name']}.md"))
        except OSError:
            pass
    _write_index([])
    try:
        os.remove(EMBEDDINGS_FILE)
    except OSError:
        pass
    return ok({"cleared": count})


@mcp.resource(
    "memory://all",
    name="memory",
    description="The memory index — one line per stored memory (attach with @memory).",
)
def memory_all() -> str:
    if not os.path.exists(INDEX_FILE):
        return "(no memories stored)"
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return f.read().strip() or "(no memories stored)"
    except OSError:
        return "(no memories stored)"


if __name__ == "__main__":
    mcp.run()
