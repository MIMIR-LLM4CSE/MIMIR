"""
MCP Memory Server
=================
Persistent, timestamped memory for the agent — Claude-style.

Two scopes, and which one a fact belongs to is the question every write answers:

  workspace — <STATE_DIR>/memory/, shared by every session of THIS workspace. What is
              true of this project: its decisions and their reason, its conventions
              and constraints.
  global    — <GLOBAL_STATE_DIR>/memory/ (``~/.mimir/global/memory/``), shared by every
              workspace on the machine. What is true of the user whichever repository
              they are in: their preferences, their corrections about how to work.

Storage layout, identical in both:
  MEMORY.md        — the index: one line per memory, loaded into context each
                     session. Format: ``- [<description>](<slug>.md) — <date>``
  <slug>.md        — one memory per file, human-editable Markdown with a small
                     frontmatter block (name / description / date / tags) + body.
  embeddings.json  — the parallel vector cache for that scope's memories.

Slugs are unique within a scope, never across them: the global store is written by
workspaces that never see each other, so cross-scope uniqueness is an invariant
nothing here could enforce. Tools that take a name therefore resolve a (scope, name)
pair, and a name living in both scopes is an error asking which one was meant.

Workflow:
  1. memory_add(text, scope, description?, tags?)  — store a fact as its own .md file
  2. memory_search(query)                          — retrieve relevant memories
  3. memory_list_all()                             — list every stored memory
  4. memory_delete(name)                           — remove one memory file by its slug
  5. memory_clear()                                — wipe one scope (irreversible)
"""

import os
import json
import re
import sys
from datetime import datetime, timezone
from typing import NamedTuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps, PLAN_BLOCKED, RECOVERABLE
from responses import err, ok
from state_paths import global_state_dir, state_dir
from text_tools import yaml_scalar, yaml_unquote
import embed as _embed

WORKSPACE = "workspace"
GLOBAL = "global"
_SCOPES = (WORKSPACE, GLOBAL)

# Workspace memory lives at the root of the central per-workspace state dir
# (MIMIR_STATE_DIR, set by server_manager; legacy <workspace>/.mimir fallback for
# standalone/tests), so it is shared across all sessions of the workspace — not
# scoped to a session.
MEMORY_DIR = os.path.join(state_dir(), "memory")
INDEX_FILE = os.path.join(MEMORY_DIR, "MEMORY.md")
# Parallel embedding cache (slug -> {"model": <id>, "vec": [...]}), kept out of the
# human-readable .md files. Makes memory_search semantic; absent/stale entries fall
# back transparently to substring search.
EMBEDDINGS_FILE = os.path.join(MEMORY_DIR, "embeddings.json")

# Global memory: the same three files under the machine-wide state tier
# (MIMIR_GLOBAL_STATE_DIR, resolved by the client and published to the servers).
# One cache file per scope rather than one shared: agents in different workspaces
# read-modify-write it concurrently, and _write_text_atomic gives atomic replacement,
# not lost-update protection.
GLOBAL_MEMORY_DIR = os.path.join(global_state_dir(), "memory")
GLOBAL_INDEX_FILE = os.path.join(GLOBAL_MEMORY_DIR, "MEMORY.md")
GLOBAL_EMBEDDINGS_FILE = os.path.join(GLOBAL_MEMORY_DIR, "embeddings.json")

_MAX_MEMORY_TEXT_LEN = 2000  # characters
_MAX_DESCRIPTION_LEN = 120   # characters (index stays scannable)
_MAX_ENTRIES = 50            # workspace: oldest memories are pruned beyond this
# Global: a lower cap, and a REFUSAL rather than a prune. Aging out the oldest note of
# a project is tolerable; silently deleting the preference the user asked to be
# remembered is the worst failure this server has. It also bounds what the whole
# prompt can carry across both indexes.
_MAX_GLOBAL_ENTRIES = 25
_DEDUP_THRESHOLD = 0.70      # Jaccard word-overlap ratio above which entry is skipped

mcp = FastMCP(
    "MemoryServer",
    debug=False,
    log_level="ERROR",
)


# ── stores ────────────────────────────────────────────────────────────────────

class _Store(NamedTuple):
    """Everything one scope's files and policy amount to."""
    scope: str
    dir: str
    index: str
    embeddings: str
    max_entries: int
    prunes: bool  # True: age out the oldest past the cap. False: refuse the write.


def _store(scope: str) -> _Store:
    """The store for *scope*, reading the module paths at call time.

    At call time, not frozen into a table at import, so a caller that repoints
    ``MEMORY_DIR`` and friends — the test suite does exactly this — repoints every
    tool with them.
    """
    if scope == GLOBAL:
        return _Store(GLOBAL, GLOBAL_MEMORY_DIR, GLOBAL_INDEX_FILE,
                      GLOBAL_EMBEDDINGS_FILE, _MAX_GLOBAL_ENTRIES, False)
    return _Store(WORKSPACE, MEMORY_DIR, INDEX_FILE,
                  EMBEDDINGS_FILE, _MAX_ENTRIES, True)


def _bad_scope(scope: str, *, allow_all: bool = False, allow_any: bool = False) -> dict:
    """The error for an unusable *scope* argument, naming what would work."""
    valid = list(_SCOPES) + (["all"] if allow_all else [])
    if not (scope or "").strip():
        return err(
            "scope is required: say which memory this belongs in.",
            hint=(
                f"{WORKSPACE!r} for what is true of this project — its decisions, "
                f"conventions, constraints. {GLOBAL!r} for what is true of the user "
                "whichever repository they are in — their preferences and their "
                "corrections about how to work. Ask the user if the fact could be "
                "either." + (" Omit it to resolve the name in whichever scope holds "
                             "it." if allow_any else "")
            ),
        )
    return err(
        f"Unknown scope {scope!r}.",
        hint="Valid values: " + ", ".join(repr(v) for v in valid) + ".",
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


def _load(store: _Store) -> list:
    """Read every memory file of one scope, newest last (sorted by date then slug)."""
    if not os.path.isdir(store.dir):
        return []
    entries = []
    for fname in os.listdir(store.dir):
        if not fname.endswith(".md") or fname == "MEMORY.md":
            continue
        path = os.path.join(store.dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        fields = _parse_frontmatter(content)
        entries.append({
            "name": fields.get("name") or fname[:-3],
            "scope": store.scope,
            "description": fields.get("description", ""),
            "date": fields.get("date", ""),
            "tags": fields.get("tags", []),
            "text": fields.get("text", ""),
        })
    entries.sort(key=lambda e: (e.get("date", ""), e.get("name", "")))
    return entries


def _load_scopes(scopes) -> list:
    """Every memory of the given scopes, each carrying the scope it came from."""
    entries: list = []
    for scope in scopes:
        entries.extend(_load(_store(scope)))
    return entries


def _write_text_atomic(path: str, text: str) -> None:
    """Write *text* to *path* via a temp file and one rename.

    The memory stores are shared — the workspace one by every session of the workspace,
    the global one by every session on the machine — and several write to them at the
    same time. A plain ``open(..., "w")`` truncates first, so a concurrent reader would
    see a half-written index or note, and two writers could interleave into one file.
    The rename makes a reader see either the old file or the new one; the temp name
    carries the pid so two writers do not share it.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _write_index(store: _Store) -> None:
    """Rewrite one scope's MEMORY.md — the human/agent-scannable index, newest first.

    Reads the directory back rather than taking the caller's list: a caller holds the
    entries it loaded *before* its own write, so an add concurrent with it would lose
    its index line. The files are the source of truth, and for the global store the
    concurrent writer can be any session on the machine.
    """
    lines = ["# Memory Index", ""]
    entries = _load(store)
    for e in sorted(entries, key=lambda x: (x.get("date", ""), x.get("name", "")), reverse=True):
        desc = e.get("description") or e.get("name", "")
        date = (e.get("date") or "")[:10]
        lines.append(f"- [{desc}]({e['name']}.md) — {date}")
    _write_text_atomic(store.index, "\n".join(lines) + "\n")


def _word_set(text: str) -> set[str]:
    return set(re.findall(r'[a-zA-Z0-9_]+', text.lower()))


def _is_near_duplicate(text: str, entries: list) -> dict | None:
    """Return the stored memory *text* nearly repeats, or None.

    Every memory is compared, not only the recent ones: an old fact is the one most
    likely to be written again, because it is the one least likely to be in mind.
    """
    words = _word_set(text)
    if not words:
        return None
    for entry in entries:
        other = _word_set(entry.get("text", ""))
        if not other:
            continue
        intersection = len(words & other)
        union = len(words | other)
        if union > 0 and intersection / union >= _DEDUP_THRESHOLD:
            return entry
    return None


def _find(name: str, scope: str | None) -> tuple[dict | None, dict | None]:
    """Resolve a slug to one memory, or to the error explaining why it cannot be.

    Returns ``(entry, error)`` with exactly one of the two set. *scope* names the store
    outright; ``None`` searches both, and a name living in both is an error rather than
    a guess — deleting or rewriting the wrong one of two same-named memories is silent
    and unrecoverable.
    """
    scopes = (scope,) if scope else _SCOPES
    matches = [e for e in _load_scopes(scopes) if e["name"] == name]
    if not matches:
        where = f"in the {scope} memory" if scope else "in either memory"
        return None, err(
            f"No memory named {name!r} {where}.",
            hint="Call memory_list_all() to see valid names and their scopes.",
        )
    if len(matches) > 1:
        return None, err(
            f"{name!r} exists in both the workspace and the global memory.",
            hint=f"Pass scope={WORKSPACE!r} or scope={GLOBAL!r} to say which one you mean.",
            scopes=[m["scope"] for m in matches],
        )
    return matches[0], None


# ── embedding cache ─────────────────────────────────────────────────────────────

def _embed_input(entry: dict) -> str:
    """Text embedded for a memory: its one-line description plus the full body."""
    return f"{entry.get('description', '')}\n{entry.get('text', '')}".strip()


def _load_embeddings(store: _Store) -> dict:
    """Read one scope's parallel embedding cache, or {} when absent/corrupt."""
    try:
        with open(store.embeddings, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_embeddings(store: _Store, cache: dict) -> None:
    try:
        _write_text_atomic(store.embeddings, json.dumps(cache))
    except OSError:
        pass


def _upsert_embedding(store: _Store, name: str, entry: dict) -> None:
    """Compute and persist the embedding for one memory. No-op when the embedding
    backend is unavailable — search then falls back to substring matching."""
    if not _embed.is_available():
        return
    vec = _embed.embed_one(_embed_input(entry))
    if vec is None:
        return
    cache = _load_embeddings(store)
    cache[name] = {"model": _embed.embed_model_id(), "vec": vec}
    _save_embeddings(store, cache)


def _prune_embeddings(store: _Store, names) -> None:
    """Drop cached vectors for the given slugs (after delete / aging trim)."""
    names = set(names)
    if not names:
        return
    cache = _load_embeddings(store)
    if any(n in cache for n in names):
        for n in names:
            cache.pop(n, None)
        _save_embeddings(store, cache)


def _semantic_search(query: str, candidates: list, limit: int) -> list | None:
    """Rank *candidates* by embedding similarity to *query*, across scopes.

    Returns a list of memories (each with a "score") or ``None`` to signal the
    caller to fall back to substring search. Vectors missing from a cache (or
    embedded under a different model) are backfilled and persisted on the fly —
    each into its own scope's file, keyed off the candidate's "scope", so a search
    spanning both stores cannot write one scope's vectors into the other's cache.
    Scores stay comparable across scopes: one model, one metric.
    """
    if not _embed.is_available():
        return None
    model = _embed.embed_model_id()
    caches = {scope: (_store(scope), _load_embeddings(_store(scope)))
              for scope in {m["scope"] for m in candidates}}

    vecs: list = []
    idx_map: list = []
    missing: list = []
    for i, m in enumerate(candidates):
        rec = caches[m["scope"]][1].get(m["name"])
        if rec and rec.get("model") == model and rec.get("vec"):
            vecs.append(rec["vec"])
            idx_map.append(i)
        else:
            missing.append(i)

    if missing:
        # One backend round trip for every missing vector of either scope — that is
        # what the batch call is for; the write-back is what splits by scope.
        new_vecs = _embed.embed_texts([_embed_input(candidates[i]) for i in missing])
        if new_vecs:
            dirty = set()
            for j, i in enumerate(missing):
                m = candidates[i]
                caches[m["scope"]][1][m["name"]] = {"model": model, "vec": new_vecs[j]}
                dirty.add(m["scope"])
                vecs.append(new_vecs[j])
                idx_map.append(i)
            for scope in dirty:
                _save_embeddings(*caches[scope])

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

@mcp.tool(**tool_caps(kind="memory"))
def memory_add(text: str, scope: str = "", description: str = None, tags: list = None) -> dict:
    """Store a fact as its own timestamped Markdown memory file.

    Each memory is written to <scope memory dir>/<slug>.md with a short frontmatter
    (name, description, date, tags) and indexed in MEMORY.md. Use this to remember
    user preferences, facts discovered during a task, or decisions worth recalling.

    Check the index first: if a memory already covers the subject, update it
    instead of adding one. Keep entries concise (under 2000 characters) — store key
    facts, file paths, and decisions, not raw conversation text.

    Returns {"status": "ok", "name": <slug>, "scope": <scope>, "path": <file>,
    "stored": <text>} — say which memory you wrote to when you report back, since the
    two have different reach. A text that nearly repeats a stored memory is refused
    with an error naming that memory and its scope.

    Args:
        text: The fact or note to remember (max 2000 characters).
        scope: Which memory this belongs in, and it is yours to decide:
            "workspace" for what is true of THIS project — its decisions and their
            reason, its conventions and constraints, what you could not find in the
            repository; "global" for what is true of the user whichever repository
            they are in — their preferences, and their corrections about how you
            should work. A fact that only holds for this project never goes global.
            Ask the user when it could honestly be either.
        description: Optional one-line summary for the index. Derived from the
            text's first sentence when omitted.
        tags: Optional list of tag strings, e.g. ["user", "preference"].
    """
    if scope not in _SCOPES:
        return _bad_scope(scope)
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
    store = _store(scope)
    memory = _load(store)

    # Deduplicate within the target scope always, and against the global store when
    # the target is the workspace: a fact that holds everywhere should block a local
    # copy of itself, while one project's note must never block a user preference.
    against = memory if scope == GLOBAL else memory + _load(_store(GLOBAL))
    dup = _is_near_duplicate(text, against)
    if dup is not None:
        return err(
            f"Not stored: this nearly repeats the {dup['scope']} memory {dup['name']!r}.",
            hint=f"Update {dup['name']!r} in place if the fact has changed; "
                 "otherwise it is already stored.",
            similar_memory=dup["name"],
            similar_scope=dup["scope"],
        )

    if not store.prunes and len(memory) >= store.max_entries:
        return err(
            f"The {scope} memory is full ({len(memory)}/{store.max_entries}).",
            hint="Update the memory this fact belongs with, or delete one that has gone "
                 "stale (memory_list_all shows them). Nothing is aged out automatically "
                 "here — a preference you were asked to keep must not disappear on its own.",
        )

    desc = (description or "").strip() or _derive_description(text)
    if len(desc) > _MAX_DESCRIPTION_LEN:
        desc = desc[:_MAX_DESCRIPTION_LEN - 1].rstrip() + "…"

    # Slug uniqueness is per store. Across scopes it is not enforceable: the global
    # store is written by workspaces that never see each other.
    name = _unique_slug(_slugify(desc), {e["name"] for e in memory})

    entry = {
        "name":        name,
        "scope":       scope,
        "description": desc,
        "date":        _now_display(),
        "tags":        tags or [],
        "text":        text,
    }

    path = os.path.join(store.dir, f"{name}.md")
    _write_text_atomic(path, _serialize(entry))
    memory.append(entry)

    # Aging: trim oldest memories beyond the cap, where the scope allows it.
    if store.prunes and len(memory) > store.max_entries:
        stale_names = []
        for stale in memory[:len(memory) - store.max_entries]:
            try:
                os.remove(os.path.join(store.dir, f"{stale['name']}.md"))
            except OSError:
                pass
            stale_names.append(stale["name"])
        _prune_embeddings(store, stale_names)

    _write_index(store)
    _upsert_embedding(store, name, entry)
    return ok({"name": name, "scope": scope, "path": path, "stored": text})


@mcp.tool(**tool_caps(kind="memory"))
def memory_search(query: str, tag: str = None, limit: int = 5, scope: str = "all") -> dict:
    """Search memories by meaning, ranked most-relevant first, across both scopes.

    When an embedding backend is available, memories are ranked by semantic
    similarity to the query (so reworded, synonymous, or other-language queries
    still match). Otherwise it falls back to a case-insensitive substring match.
    Optionally filter by a specific tag.

    Returns {"status": "ok", "results": [<matching memories>], "count": <n>}.
    Each result carries its "scope" and a "score" (semantic similarity, or None for
    substring matches). If no results, "count" will be 0 — try a shorter query or
    memory_list_all().

    Args:
        query: What to look for (natural language; not just a substring).
        tag:   If provided, only memories with this tag are searched.
        limit: Maximum number of results to return (default 5).
        scope: "all" (default) searches this workspace's memory and the global one
            together; "workspace" or "global" restricts it to that store.
    """
    if scope == "all":
        scopes = _SCOPES
    elif scope in _SCOPES:
        scopes = (scope,)
    else:
        return _bad_scope(scope, allow_all=True)

    memory = _load_scopes(scopes)
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


@mcp.tool(**tool_caps(kind="memory"))
def memory_list_all(scope: str = "all") -> dict:
    """Return every stored memory with its name, scope, description, date, tags, and text.

    Use this to get full context before deciding what to recall or delete.
    Returns {"status": "ok", "memory": [<all memories>], "count": <n>}.

    Args:
        scope: "all" (default) lists both this workspace's memory and the global one;
            "workspace" or "global" lists only that store.
    """
    if scope == "all":
        scopes = _SCOPES
    elif scope in _SCOPES:
        scopes = (scope,)
    else:
        return _bad_scope(scope, allow_all=True)
    memory = _load_scopes(scopes)
    return ok({"memory": memory, "count": len(memory)})


@mcp.tool(**tool_caps(kind="memory"))
def memory_update(
    name: str,
    text: str = None,
    description: str = None,
    tags: list = None,
    scope: str = None,
) -> dict:
    """Edit an existing memory in place, keyed by its slug name.

    Only the provided fields are changed; omitted fields are preserved. The slug
    (filename) stays stable so the index link remains valid, and the date is
    refreshed to now. Call memory_list_all() to see available names and their scopes.

    Returns {"status": "ok", "name": <slug>, "scope": <scope>, "path": <file>,
    "updated": <memory>} or an error.

    Args:
        name: Slug name of the memory to edit (without the .md extension).
        text: New body text (max 2000 characters). Unchanged when omitted.
        description: New one-line index summary. Unchanged when omitted.
        tags: New list of tags, replacing the old ones. Unchanged when omitted.
        scope: Which memory holds it. Omit to resolve the name in whichever one does;
            pass "workspace" or "global" when the same name exists in both. Note this
            does not MOVE a memory between scopes — to do that, add it to the other
            scope and delete it here.
    """
    if scope is not None and scope not in _SCOPES:
        return _bad_scope(scope, allow_any=True)
    if text is not None and len(text) > _MAX_MEMORY_TEXT_LEN:
        return err(
            f"text is too long ({len(text)} chars, max {_MAX_MEMORY_TEXT_LEN}). "
            "Summarise to key facts, file paths, and decisions before storing.",
        )
    entry, error = _find(name, scope)
    if error is not None:
        return error

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

    store = _store(entry["scope"])
    path = os.path.join(store.dir, f"{name}.md")
    _write_text_atomic(path, _serialize(entry))
    _write_index(store)
    _upsert_embedding(store, name, entry)
    return ok({"name": name, "scope": entry["scope"], "path": path, "updated": entry})


@mcp.tool(**tool_caps(kind="memory",
    caps=[PLAN_BLOCKED], reversibility=RECOVERABLE, non_batch=True,
    risk_note="deletes a persistent memory file",
))
def memory_delete(name: str, scope: str = None) -> dict:
    """Delete one memory by its slug name (the file's <name>.md).

    Call memory_list_all() to see available names and their scopes. To remove
    everything in one scope use memory_clear().

    Returns {"status": "ok", "deleted": <memory>, "scope": <scope>} or
    {"status": "error", ...}. Say which memory it came out of when you report back:
    a global one was reaching every workspace.

    Args:
        name: Slug name of the memory to remove (without the .md extension).
        scope: Which memory holds it. Omit to resolve the name in whichever one does;
            pass "workspace" or "global" when the same name exists in both.
    """
    if scope is not None and scope not in _SCOPES:
        return _bad_scope(scope, allow_any=True)
    match, error = _find(name, scope)
    if error is not None:
        return error
    store = _store(match["scope"])
    try:
        os.remove(os.path.join(store.dir, f"{name}.md"))
    except OSError as exc:
        return err(f"Could not delete {name!r}: {exc}")
    _write_index(store)
    _prune_embeddings(store, {name})
    return ok({"deleted": match, "scope": match["scope"]})


@mcp.tool(**tool_caps(kind="memory",
    caps=[PLAN_BLOCKED], reversibility=RECOVERABLE, non_batch=True,
    risk_note="wipes all persistent memory files of one scope; a global wipe "
              "affects every workspace on the machine",
))
def memory_clear(scope: str = WORKSPACE) -> dict:
    """Wipe every stored memory of one scope. This action is irreversible.

    One scope per call, and no "all": clearing both stores is two deliberate
    decisions, not one, because a global wipe reaches every workspace on the machine
    while a workspace wipe reaches only this project.

    Returns {"status": "ok", "cleared": <number of memories removed>, "scope": <scope>}.

    Args:
        scope: "workspace" (default) wipes this project's memory; "global" wipes the
            memory shared by every workspace — only on an explicit request for that.
    """
    if scope not in _SCOPES:
        return _bad_scope(scope)
    store = _store(scope)
    memory = _load(store)
    count = len(memory)
    for e in memory:
        try:
            os.remove(os.path.join(store.dir, f"{e['name']}.md"))
        except OSError:
            pass
    _write_index(store)
    try:
        os.remove(store.embeddings)
    except OSError:
        pass
    return ok({"cleared": count, "scope": scope})


@mcp.resource(
    "memory://all",
    name="memory",
    description="The memory index — one line per stored memory, both scopes (attach with @memory).",
)
def memory_all() -> str:
    sections = []
    for scope, heading in ((GLOBAL, "## Global memory (every workspace)"),
                           (WORKSPACE, "## Workspace memory (this project)")):
        try:
            with open(_store(scope).index, "r", encoding="utf-8") as f:
                body = f.read().strip()
        except OSError:
            continue
        # Each index file opens with its own "# Memory Index" title. Composed under a
        # per-scope heading that title is a second heading saying less, so drop it and
        # keep the entry lines.
        lines = [ln for ln in body.splitlines() if ln.strip() and not ln.startswith("# ")]
        if lines:
            sections.append(heading + "\n\n" + "\n".join(lines))
    return "\n\n".join(sections) or "(no memories stored)"


if __name__ == "__main__":
    mcp.run()
