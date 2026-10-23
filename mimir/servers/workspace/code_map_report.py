"""The workspace's code map as this server reports it: whole for the editor's map tab,
summed up for the model, and as the Architecture section of the drawer."""
from __future__ import annotations

import posixpath
import time
from collections import Counter

from ...shared import code_map

# The drawer asks every few seconds while it is open; a walk of a large tree is not free,
# and the code it reports moves at the pace of edits.
_PANEL_TTL_SECS = 30.0
_LAST: dict = {"at": 0.0, "root": "", "map": None}


def current(root: str, *, refresh: bool = False) -> dict:
    """The map of *root*, rescanned at most once per TTL unless *refresh*."""
    now = time.monotonic()
    if refresh or _LAST["map"] is None or _LAST["root"] != root or now - _LAST["at"] > _PANEL_TTL_SECS:
        _LAST.update({"at": now, "root": root, "map": code_map.scan(root)})
    return _LAST["map"]


def panel(root: str, *, refresh: bool = False) -> dict:
    """The Architecture section: what the code is made of and what to look at in it."""
    m = current(root, refresh=refresh)
    if not m["files"]:
        return {"title": "Architecture", "lines": [],
                "detail": "No Python, C/C++ or Fortran code in this workspace."}
    langs = Counter(f["lang"] for f in m["files"])
    breaks = [e for e in m["edges"] if e["breaks"]]
    bindings = sum(1 for e in m["edges"] if e["kind"] == "binding")
    lines = [
        {"label": "code", "value": f"{len(m['files'])} files · {len(m['modules'])} modules"},
        {"label": "languages", "value": ", ".join(f"{k} {v}" for k, v in langs.most_common())},
        {"label": "rule breaks",
         "value": f"{len(breaks)} · {posixpath.basename(breaks[0]['from'])}" if breaks else "none",
         "state": "warn" if breaks else ""},
        {"label": "cycles", "value": f"{len(m['cycles'])} between modules" if m["cycles"] else "none",
         "state": "warn" if m["cycles"] else ""},
    ]
    if bindings:
        lines.append({"label": "across languages", "value": f"{bindings} bindings"})
    unresolved = sum(m["unresolved"].values())
    if unresolved:
        lines.append({"label": "not placed", "value": f"{unresolved} references"})
    if m["truncated"]:
        lines.append({"label": "scan", "value": f"stopped at {code_map.MAX_FILES} files", "state": "warn"})
    return {"title": "Architecture", "lines": lines, "open": "code_map", "map": components(m)}


def components(m: dict) -> dict:
    """The components and the links between them, tests aside: the drawer's small map."""
    shown = {c["id"] for c in m["components"] if not c["tests"]}
    comp_of = {f["id"]: f["component"] for f in m["files"]}
    mod_comp = {mod["id"]: mod["component"] for mod in m["modules"]}
    files = Counter(f["component"] for f in m["files"])
    breaks = Counter(comp_of[e["from"]] for e in m["edges"] if e["breaks"])
    cycles = Counter(mod_comp[c[0]] for c in m["cycles"] if c)
    links: dict[tuple[str, str], dict] = {}
    for e in m["edges"]:
        a, b = comp_of[e["from"]], comp_of[e["to"]]
        if a != b and a in shown and b in shown and e["sure"] == "exact":
            link = links.setdefault((a, b), {"from": a, "to": b, "n": 0, "breaks": 0})
            link["n"] += 1
            link["breaks"] += bool(e["breaks"])
    return {
        "components": [{"id": c["id"], "name": c["name"], "tier": c["tier"], "files": files[c["id"]],
                        "breaks": breaks[c["id"]], "cycles": cycles[c["id"]]}
                       for c in sorted(m["components"], key=lambda c: (-c["tier"], -files[c["id"]]))
                       if c["id"] in shown],
        "links": sorted(links.values(), key=lambda link: (link["from"], link["to"])),
    }
