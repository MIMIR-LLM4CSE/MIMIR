"""
MCP MIMIR API Server
====================
MIMIR's self-knowledge: the extension API a user or a developer writes against when
they add a skill, an MCP server, a policy, a post-tool hook, a nudge or a base prompt
under the workspace ``.mimir/``.

Every answer is derived from the installed package at call time — the capability
vocabulary is parsed out of ``client/context/capabilities.py``, the templates are the
shipped ``mimir/examples/`` files read verbatim, the drop-in locations come from the
same ``_shared/extension_paths`` the loaders use, and the bundled names are scanned off
disk. Nothing here is a prose copy of the documentation, because a copy is what goes
stale: a capability added to the vocabulary shows up in the next call, and the
``.mimir/`` path reported is the one that will actually be scanned on restart.

The companion methodology prompt is the bundled ``mimir-api`` skill, which drives this
tool; this server only answers questions.

Tools:
  1. mimir_api(topic, name) — one topic of the extension API
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import CACHEABLE, tool_caps
from extension_paths import (
    MIMIR_DIRNAME,
    PLUGINS_DIR_ENV,
    PLUGINS_DIRNAME,
    SERVERS_DIR_ENV,
    SERVERS_DIRNAME,
    SKILLS_DIR_ENV,
    SKILLS_DIRNAME,
    SYSTEM_PROMPT_ENV,
    SYSTEM_PROMPT_FILENAME,
    mimir_dir,
    resolve_extension_dir,
    workspace_root,
)
from responses import err, ok
from text_tools import yaml_unquote

# The installed package root (``<...>/mimir/``): this file is mimir/servers/agent_state/.
# Everything reported below is read from under it, so the answers describe the build
# that is running rather than a checkout that may not be present at all.
_PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXAMPLES = os.path.join(_PKG, "examples")
_BUNDLED_SKILLS = os.path.join(_PKG, "skills")
_BUNDLED_SERVERS = os.path.join(_PKG, "servers")
# The vocabulary is defined twice on purpose — once client-side, once in
# servers/_shared/capabilities.py, which is what a server imports — and a parity test
# keeps them level. The client copy is the one read here: it carries the section
# headers and the per-flag note that say what each flag is *for*.
_CAP_SOURCE = os.path.join(_PKG, "client", "context", "capabilities.py")

# A template is a file the user is meant to copy; none of the shipped ones is anywhere
# near this, so the cap only bounds a pathological workspace, never a real answer.
_MAX_TEMPLATE_BYTES = 64_000

mcp = FastMCP(
    "MimirApiServer",
    debug=False,
    log_level="ERROR",
)


# ── reading the package ───────────────────────────────────────────────────────

def _read_text(path: str, limit: int = _MAX_TEMPLATE_BYTES) -> str:
    """File contents, or an explanatory line in their place.

    A missing example or a trimmed install must not fail the call: the rest of the
    topic (where the file goes, when it is loaded) is still the answer the caller
    needs, and a line saying what could not be read is more use than an error.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read(limit + 1)
    except OSError as exc:
        return f"[unavailable: {type(exc).__name__}] {path}"
    if len(text) > limit:
        return text[:limit] + f"\n[truncated at {limit} bytes] {path}"
    return text


_FLAG_RE = re.compile(r'^([A-Z][A-Z0-9_]*)\s*=\s*"([a-z_]+)"\s*(?:#\s*(.*?))?\s*$')
_CONT_RE = re.compile(r'^\s{4,}#\s*(.*?)\s*$')
_SECTION_RE = re.compile(r'^#\s*---+\s*(.*?)\s*-*$')
_PROSE_RE = re.compile(r'^#\s*(.*?)\s*$')

# The three reversibility levels are declared the same way a capability is, but they
# are one ordered dimension rather than a flag set, so they are reported apart.
_REVERSIBILITY = ("reversible", "recoverable", "irreversible")
# Derived client-side from the declared reversibility: a tool that declares this is
# saying something the client works out for itself. See the note the parser lifts.
_DERIVED = frozenset({"sensitive"})


def _capability_vocabulary() -> dict:
    """The live capability vocabulary, parsed from the capability module's source.

    Parsed rather than imported: the *notes* and the section grouping are what make the
    list usable by someone choosing a flag, and those live in the comments. A flag's
    note is its trailing comment plus any indented continuation lines, falling back to
    the prose paragraph introducing its section when it has none.
    """
    section = ""
    prose: list[str] = []
    flags: list[dict] = []
    levels: list[dict] = []
    last: dict | None = None

    for line in _read_text(_CAP_SOURCE, limit=200_000).splitlines():
        flag = _FLAG_RE.match(line)
        if flag:
            name, value, note = flag.group(1), flag.group(2), (flag.group(3) or "")
            entry = {"name": name, "value": value,
                     "note": note or " ".join(prose), "group": section}
            (levels if value in _REVERSIBILITY else flags).append(entry)
            last = entry
            continue
        cont = _CONT_RE.match(line)
        if cont and last is not None:
            last["note"] = (last["note"] + " " + cont.group(1)).strip()
            continue
        last = None
        heading = _SECTION_RE.match(line)
        if heading and heading.group(1):
            section, prose = heading.group(1), []
            continue
        body = _PROSE_RE.match(line)
        if body and body.group(1):
            prose.append(body.group(1))
        elif not line.strip():
            prose = []

    for entry in flags:
        if entry["value"] in _DERIVED:
            entry["declarable"] = False
    return {"flags": flags, "reversibility": levels}


def _front_matter(path: str) -> dict:
    """``name`` / ``description`` from a SKILL.md front-matter block (best effort)."""
    text = _read_text(path, limit=8_000)
    fields: dict = {}
    if not text.startswith("---"):
        return fields
    for line in text.split("---", 2)[1].splitlines():
        key, _, value = line.partition(":")
        if key.strip() in ("name", "description") and value.strip():
            fields[key.strip()] = yaml_unquote(value.strip())
    return fields


def _bundled_skill_names() -> list[dict]:
    """The shipped skills, which a same-named user skill overrides."""
    out: list[dict] = []
    for entry in sorted(_listdir(_BUNDLED_SKILLS)):
        md = os.path.join(_BUNDLED_SKILLS, entry, "SKILL.md")
        if os.path.isfile(md):
            out.append({"name": entry, **_front_matter(md)})
    return out


def _bundled_server_names() -> dict:
    """The namespaces a user server may not take (the core is protected).

    Read from the client's bundled registry, because that is what the collision check
    tests against and a filename is not always the namespace: ``server_spawn_agent.py``
    is registered as ``agent``. The import is guarded and the answer says where it came
    from — the registry module is a stdlib-only leaf, but a server must still run with
    no client on the path (standalone, tests), and there the shipped filenames are the
    closest honest answer rather than no answer.
    """
    try:
        from mimir.client.config.constants import SERVERS
        return {"names": sorted(SERVERS), "source": "the client's bundled server registry"}
    except Exception:
        names = set()
        for _root, _dirs, files in os.walk(_BUNDLED_SERVERS):
            for filename in files:
                stem, ext = os.path.splitext(filename)
                if ext == ".py" and stem.startswith("server_"):
                    names.add(stem[len("server_"):])
        return {"names": sorted(names),
                "source": "the shipped server filenames — the registry was not "
                          "importable here, so a name registered under a different "
                          "namespace may be reported by its filename"}


def _listdir(path: str) -> list[str]:
    try:
        return os.listdir(path)
    except OSError:
        return []


# ── the extension types ───────────────────────────────────────────────────────
# One entry per drop-in kind: where it goes, when MIMIR loads it, what a name
# collision with a bundled one does, and the topic that answers it in full. The paths
# are resolved live, so a workspace whose MIMIR_*_DIR points elsewhere is told the
# truth about where its extensions are read from.

def _skills_dir() -> str:
    return resolve_extension_dir(SKILLS_DIR_ENV, SKILLS_DIRNAME)


def _servers_dir() -> str:
    return resolve_extension_dir(SERVERS_DIR_ENV, SERVERS_DIRNAME)


def _plugins_dir() -> str:
    return resolve_extension_dir(PLUGINS_DIR_ENV, PLUGINS_DIRNAME)


def _system_prompt_file() -> str:
    env = os.environ.get(SYSTEM_PROMPT_ENV)
    return os.path.abspath(env) if env else os.path.join(mimir_dir(), SYSTEM_PROMPT_FILENAME)


_TYPES: dict[str, dict] = {
    "skill": {
        "what": "A methodology prompt that steers *how* the agent works on a class of task.",
        "drop_in": lambda: os.path.join(_skills_dir(), "<name>", "SKILL.md"),
        "env_override": SKILLS_DIR_ENV,
        "loaded": "At agent start. Triggered explicitly (`/<name> …`) or by the skill "
                  "classifier, which reads the current query plus the last few turns.",
        "collision": "A user skill overrides the bundled skill of the same name.",
        "template": os.path.join(_EXAMPLES, "skills", "example-skill", "SKILL.md"),
        "rules": [
            "The front-matter `name` MUST equal the directory name.",
            "`description` is the one line the classifier reads — write what the task "
            "looks like, not what the skill contains.",
            "The body is injected as a subordinate system message; the base system "
            "instructions stay authoritative, so never restate or contradict them.",
            "Methodology only. Validation tiers, approval handling and edit-tool "
            "mechanics are already in context — a copy there goes stale.",
        ],
    },
    "server": {
        "what": "An stdio MCP server: extra tools, auto-connected at startup.",
        "drop_in": lambda: os.path.join(_servers_dir(), "server_<name>.py"),
        "env_override": SERVERS_DIR_ENV,
        "loaded": "At agent start, as a stdio child process. The tool namespace is the "
                  "filename stem with any `server_` prefix stripped.",
        "collision": "A name colliding with a bundled server is skipped — the core "
                     "cannot be shadowed. Ask topic `loaded` for the reserved names.",
        "template": os.path.join(_EXAMPLES, "servers", "server_example.py"),
        "rules": [
            "Declaring capabilities with `tool_caps(...)` is optional — an undeclared "
            "tool still runs, classified from its standard MCP annotations — but it is "
            "what makes MIMIR's policies, caching and nudges treat the tool correctly.",
            "Declare `reversibility` instead of `sensitive`: approval-gating is derived "
            "from it, so the tool states one fact about its effect rather than two.",
            "A tool that names a file takes an absolute path (see _shared/root_paths).",
            "Answer with the `ok()` / `err()` payload shape from _shared/responses.",
            "Document each parameter in an `Args:` block — those lines are lifted into "
            "the JSON schema, which is the part a model actually obeys.",
        ],
    },
    "policy": {
        "what": "A check that BLOCKS a call before it runs (locked — not toggleable).",
        "drop_in": lambda: os.path.join(_plugins_dir(), "<name>.py"),
        "env_override": PLUGINS_DIR_ENV,
        "loaded": "Each *.py in the plugins dir is imported once at agent init and "
                  "registers its descriptors as an import side effect.",
        "collision": "Additive: every registered check runs. Registration is idempotent "
                     "by name.",
        "template": os.path.join(_EXAMPLES, "plugins", "policy_example.py"),
        "rules": [
            "Return a JSON error string to block, or None to abstain.",
            "Stage `pre_mutation` (after the registry) or `pre_approval` (after the "
            "write policy); `order` breaks ties within a stage.",
            "A check only ADDs constraints — it can never relax a core gate.",
            "Key off capabilities, never literal tool names.",
        ],
    },
    "post_tool": {
        "what": "A hook that runs AFTER a successful call and appends to its result.",
        "drop_in": lambda: os.path.join(_plugins_dir(), "<name>.py"),
        "env_override": PLUGINS_DIR_ENV,
        "loaded": "Same import-time registration as a policy; runs after the built-in "
                  "post-write ladder, under its own 60 s budget.",
        "collision": "Additive: every registered hook runs.",
        "template": os.path.join(_EXAMPLES, "plugins", "post_tool_example.py"),
        "rules": [
            "Return the text to append, or \"\" to abstain. `run` may be sync or async.",
            "It cannot block — the call already happened — but it is the one seam that "
            "turns a machine observation into blackboard state the turn-end gates read.",
            "Running a command goes through the ordinary tool path, so the approval gate "
            "still applies: a hook cannot execute shell the user never agreed to.",
            "Keep it cheap: it runs after every matching call, and one that raises is "
            "skipped rather than allowed to cost the others their turn.",
        ],
    },
    "nudge": {
        "what": "An advisory reminder injected at the end of a turn with no tool call.",
        "drop_in": lambda: os.path.join(_plugins_dir(), "<name>.py"),
        "env_override": PLUGINS_DIR_ENV,
        "loaded": "Same import-time registration as a policy. Fires at most once per "
                  "step, subject to the per-query caps.",
        "collision": "Additive, and toggleable per name via `/nudges`.",
        "template": os.path.join(_EXAMPLES, "plugins", "nudge_example.py"),
        "rules": [
            "`predicate` decides when it fires; `render` returns the reminder text.",
            "Layer `verification` runs at every enforcement level; layer `guidance` is "
            "tier-gated through `tiers` — (enforcement, mode) pairs, omit \"off\".",
            "It reminds, it never blocks. A rule that must hold is a policy.",
            "Suppressed while its name is in the user's disabled set.",
        ],
    },
    "system_prompt": {
        "what": "The base prompt: MIMIR's persona and your domain knowledge.",
        "drop_in": _system_prompt_file,
        "env_override": SYSTEM_PROMPT_ENV,
        "loaded": "At agent start. Resolution order: the env var, then "
                  f"`{MIMIR_DIRNAME}/{SYSTEM_PROMPT_FILENAME}`, then the built-in doctrine.",
        "collision": "The file replaces the *doctrine* half of the built-in prompt "
                     "(identity, style, scope, workflow, reasoning). The *core* half — "
                     "non-negotiables, tool mechanics, discovery, editing, validation, "
                     "running code, planning — is appended after it, always.",
        "template": os.path.join(_EXAMPLES, SYSTEM_PROMPT_FILENAME),
        "rules": [
            "Write persona, codebase map, house rules and vocabulary.",
            "Do NOT restate validation tiers, approval handling, checklist rules or "
            "edit-tool mechanics: that text is already in context, as a copy that "
            "goes stale and that the core half will contradict.",
            "There is no opt-out of the core half, for the same reason policy checks "
            "have none.",
        ],
    },
}

_TOPICS = ("index", "capabilities", "loaded", *sorted(_TYPES))


def _type_entry(kind: str) -> dict:
    spec = _TYPES[kind]
    drop_in = spec["drop_in"]
    path = drop_in() if callable(drop_in) else drop_in
    env = spec["env_override"]
    return {
        "type": kind,
        "what": spec["what"],
        "drop_in": path,
        "env_override": {"var": env, "set_to": os.environ.get(env) or None},
        "loaded": spec["loaded"],
        "on_name_collision": spec["collision"],
    }


# ── the tool ──────────────────────────────────────────────────────────────────

@mcp.tool(**tool_caps(
    caps=[CACHEABLE],
    label="Reading MIMIR's extension API: {topic}",
))
def mimir_api(topic: str = "index", name: str = "") -> dict:
    """Read MIMIR's own extension API — what a `.mimir/` extension may do and where it goes.

    Answers from the installed package, so the capability list, the templates and the
    drop-in paths are this build's, not a remembered version's. Start at `index`; it
    names every extension type and the topic that covers it.

    Args:
        topic: Which part of the API to read. `index` — the extension types, their
            drop-in paths and the rules that apply to all of them. `skill`, `server`,
            `policy`, `post_tool`, `nudge`, `system_prompt` — one type in full: where
            the file goes, when MIMIR loads it, the authoring rules, and the shipped
            template verbatim. `capabilities` — the capability vocabulary a tool
            declares and a policy or nudge keys off, with what each flag drives.
            `loaded` — what is installed and present right now: resolved extension
            directories, the user extensions already on disk, the bundled skills, and
            the server namespaces a user server may not take.
        name: Unused for now; reserved so a topic can be narrowed to one extension.
    """
    key = (topic or "index").strip().lower().replace("-", "_")
    if key in ("", "index"):
        return ok({
            "mimir_dir": mimir_dir(),
            "workspace_root": workspace_root(),
            "extension_types": [_type_entry(kind) for kind in sorted(_TYPES)],
            "topics": list(_TOPICS),
            "rules": [
                "Author under the workspace .mimir/ (or the MIMIR_*_DIR the entry "
                "names). Never edit the installed package to extend MIMIR: an install "
                "is replaced on upgrade, and a core edit is not an extension.",
                "Nothing is auto-created there: the file is written because the user "
                "asked for it, through the ordinary approval-gated write tools.",
                "A policy or a nudge keys off capabilities, never literal tool names, "
                "so it survives a rename and applies to any server declaring the "
                "capability. Ask topic `capabilities` for the vocabulary.",
                "An extension is discovered by directory scan at agent start, so a new "
                "or edited file takes effect on the next start — reload the VS Code "
                "window, or restart the CLI.",
            ],
        })

    if key == "capabilities":
        vocab = _capability_vocabulary()
        return ok({
            "declare_with": "from mimir.servers._shared.capabilities import tool_caps, "
                            "READ, CACHEABLE   # server side",
            "read_with": "from mimir.client.context.capabilities import has_cap, "
                         "names_with_cap   # policy / nudge side",
            "flags": vocab["flags"],
            "count": len(vocab["flags"]),
            "reversibility": vocab["reversibility"],
            "notes": [
                "A flag marked `declarable: false` is derived by the client from what "
                "you did declare — declaring it yourself says nothing new.",
                "Declare `reversibility` rather than approval: `reversible` prompts for "
                "nothing, `recoverable` prompts, `irreversible` prompts and no "
                "enforcement level may soften it. Omitted, a level is derived from the "
                "capabilities — declare one whenever that derivation would understate "
                "your tool, as it cannot know an HTTP tool *sends* rather than reads.",
                "`is_write` (edit ∪ content_write ∪ remove) and `clears_edit_loop` "
                "(read ∪ validate) are derived helpers, not declarable flags.",
                "Beyond the flags, a declaration carries arg-roles (which argument is a "
                "path, which carries the steps of a plan, which the verdict), an "
                "approval scope, a risk note, a preview shape and a timeout — see the "
                "`server` topic's template and _shared/capabilities.py.",
            ],
        })

    if key == "loaded":
        skills_dir, servers_dir, plugins_dir = _skills_dir(), _servers_dir(), _plugins_dir()
        prompt_file = _system_prompt_file()
        return ok({
            "mimir_dir": mimir_dir(),
            "user": {
                "skills": {"dir": skills_dir, "present": sorted(
                    entry for entry in _listdir(skills_dir)
                    if os.path.isfile(os.path.join(skills_dir, entry, "SKILL.md"))
                )},
                "servers": {"dir": servers_dir, "present": sorted(
                    entry for entry in _listdir(servers_dir)
                    if entry.endswith((".py", ".js")) and not entry.startswith("_")
                )},
                "plugins": {"dir": plugins_dir, "present": sorted(
                    entry for entry in _listdir(plugins_dir) if entry.endswith(".py")
                )},
                "system_prompt": {"path": prompt_file,
                                  "present": os.path.isfile(prompt_file)},
            },
            "bundled_skills": _bundled_skill_names(),
            "reserved_server_namespaces": _bundled_server_names(),
            "note": "`present` is what is on disk now. A plugin pack registers its "
                    "policies, hooks and nudges on import, so what a module contributes "
                    "is read from the module, not from its filename.",
        })

    if key in _TYPES:
        spec = _TYPES[key]
        entry = _type_entry(key)
        entry["authoring_rules"] = spec["rules"]
        entry["template"] = {
            "path": spec["template"],
            "source": _read_text(spec["template"]),
        }
        if key == "skill":
            entry["bundled_skills"] = _bundled_skill_names()
        if key == "server":
            entry["reserved_server_namespaces"] = _bundled_server_names()
        if key in ("server", "policy", "nudge", "post_tool"):
            entry["see_also"] = "topic `capabilities` — what a tool declares and what a " \
                                "policy or nudge keys off"
        return ok(entry)

    return err(f"Unknown topic '{topic}'.",
               hint=f"Use one of: {', '.join(_TOPICS)}. Start at 'index'.")


if __name__ == "__main__":
    mcp.run()
