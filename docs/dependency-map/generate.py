"""MIMIR's dependency map, read from the code into ``index.html`` and ``map.svg``.

Usage: python docs/dependency-map/generate.py [--check]
Without a flag, rewrites the page's data block and the image. With ``--check``, exits 1
when either no longer matches the code: the page's structure (units, files, imports,
rule breaks, cycles) or the image itself. The layout and the script of the page are
hand-written and left as they are; the image is drawn here, with the page's layout, for
GitHub and editors, which show an SVG but not a page.
"""
from __future__ import annotations

import ast
import html
import json
import math
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE = REPO / "mimir"
PAGE = HERE / "index.html"
IMAGE = HERE / "map.svg"
DATA_BLOCK = re.compile(r'(<script type="application/json" id="map-data">)(.*?)(</script>)', re.S)

# Trees that are not MIMIR's own code: the extension is TypeScript, the tests and the
# examples import everything, and a benchmark's task fixtures are the agent's workspaces.
SKIPPED = {"__pycache__", "tests", "vscode-extension", "examples", "skills"}
# Client packages drawn as one box per sub-package rather than as one box.
SPLIT = {"query_engine": {"dispatch", "jobs"},
         "guardrails": {"policy", "nudges", "observations", "workflow", "verification"},
         "ui": {"cli", "ws"}}
WS_PARTS = {"connection", "pool", "process", "server", "worker"}
# The rank the measuring packages are drawn at: above every client unit, which never
# imports them.
FRONT_RANK = 11


def module_name(path: pathlib.Path) -> str:
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def unit_of(module: str) -> str | None:
    """The box a module belongs to on the map, or None when it is drawn nowhere."""
    parts = module.split(".")
    if parts[:1] != ["mimir"] or len(parts) < 2:
        return None
    area = parts[1]
    if area == "client" and len(parts) > 2:
        unit = parts[2]
        if unit in SPLIT and len(parts) > 3 and parts[3] in SPLIT[unit]:
            if unit == "ui" and parts[3] == "ws" and len(parts) > 4 and parts[4] in WS_PARTS:
                return f"client.ui.ws.{parts[4]}"
            return f"client.{unit}.{parts[3]}"
        return f"client.{unit}"
    if area == "servers" and len(parts) > 2 and parts[2] != "_launch":
        return f"servers.{parts[2]}"
    if area == "shared" and len(parts) > 2:
        return f"shared.{parts[2]}"
    if area == "runner":
        return "runner"
    if area == "evals":
        return None if "tasks" in parts else "evals"
    return None


def resolve(path: pathlib.Path, node: ast.ImportFrom) -> str:
    """The absolute module an ``ImportFrom`` names."""
    if node.level == 0:
        return node.module or ""
    package = module_name(path).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    base = package[:len(package) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def first_line(doc: str | None) -> str:
    """A docstring's first sentence-line, or ""."""
    return (doc or "").strip().split("\n", 1)[0].strip()


def architecture_ranks() -> dict[str, tuple[int, str]]:
    """docs/ARCHITECTURE.md's rank table: client unit -> (rank, what it is)."""
    out: dict[str, tuple[int, str]] = {}
    for line in (REPO / "docs/ARCHITECTURE.md").read_text().splitlines():
        match = re.match(r"\|\s*(\d+)\s*\|([^|]*)\|([^|]*)\|", line)
        if match:
            rank, units, what = int(match[1]), match[2], match[3].strip()
            for unit in re.findall(r"`(\w+)`", units):
                out[unit] = (rank, what)
    return out


def shared_readers() -> dict[str, str]:
    """docs/ARCHITECTURE.md's two readers of shared: module -> "contract" or "servers"."""
    out: dict[str, str] = {}
    for line in (REPO / "docs/ARCHITECTURE.md").read_text().splitlines():
        if not line.startswith("| **The "):
            continue
        reader = "contract" if "client and the servers" in line else "servers"
        for name in re.findall(r"`(\w+)`", line.rsplit("|", 2)[1]):
            out[f"shared.{name}"] = reader
    return out


def _load_time_imports(tree: ast.Module):
    """Every import with whether it runs at load time; ``TYPE_CHECKING`` ones are dropped."""
    typing_only = {id(n) for top in tree.body if isinstance(top, ast.If)
                   and "TYPE_CHECKING" in ast.unparse(top.test) for n in ast.walk(top)}

    def walk(node: ast.AST, deferred: bool):
        for child in ast.iter_child_nodes(node):
            inside = deferred or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                    ast.Lambda))
            if isinstance(child, (ast.Import, ast.ImportFrom)) and id(child) not in typing_only:
                yield child, not deferred
            yield from walk(child, inside)
    yield from walk(tree, False)


def extract() -> dict:
    """The map's data: units, their files, and the imports between units with their sites."""
    ranks = architecture_ranks()
    readers = shared_readers()
    units: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if SKIPPED & set(path.relative_to(PACKAGE).parts):
            continue
        source = unit_of(module_name(path))
        if source is None:
            continue
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
        unit = units.setdefault(source, {"id": source, "files": [], "doc": ""})
        rel = str(path.relative_to(REPO))
        doc = first_line(ast.get_docstring(tree))
        unit["files"].append({"path": rel, "lines": text.count("\n"), "doc": doc})
        # A unit's id is its path under mimir: that package (or module) speaks for it.
        if module_name(path) == f"mimir.{source}":
            unit["doc"] = doc
        for node, at_load in _load_time_imports(tree):
            if isinstance(node, ast.ImportFrom):
                base = resolve(path, node)
                candidates = [f"{base}.{a.name}" for a in node.names] + [base]
                names = [a.name for a in node.names]
            else:
                candidates = [a.name for a in node.names]
                base, names = "", []
            hit = next(((m, u) for m, u in ((m, unit_of(m)) for m in candidates)
                        if u and u != source), None)
            if hit is None:
                continue
            target = hit[1]
            module = base or hit[0]
            edge = edges.setdefault((source, target), {"from": source, "to": target,
                                                       "sites": []})
            edge["sites"].append({"file": rel, "line": node.lineno, "module": module,
                                  "names": names, "load": at_load})
    for unit in units.values():
        unit.update(_place(unit["id"], ranks, readers))
        unit["lines"] = sum(f["lines"] for f in unit["files"])
    data = {"units": [units[k] for k in sorted(units)],
            "edges": [_edge_summary(edges[k]) for k in sorted(edges)]}
    by_id = {u["id"]: u for u in data["units"]}
    for edge in data["edges"]:
        edge["bad"] = breaks(edge, by_id)
    data["cycles"], data["local_cycles"] = cycles(data)
    return data


# Servers that may import the client (ARCHITECTURE.md, "The servers").
SERVERS_USING_CLIENT = {"servers.agent_state", "servers.interaction"}


def breaks(edge: dict, units: dict[str, dict]) -> str:
    """Why an edge breaks one of ARCHITECTURE.md's rules, or "" when it keeps them all."""
    a, b = units[edge["from"]], units[edge["to"]]
    if b["area"] == "measure" and a["area"] != "measure":
        return "nothing the agent runs on imports runner or evals"
    if a["area"] == "shared" and b["area"] != "shared":
        return "shared imports neither the client nor the servers"
    if a["area"] == "client" and b["area"] == "servers":
        return "the client imports no server"
    if a["area"] == "servers" and b["area"] == "client" and a["id"] not in SERVERS_USING_CLIENT:
        return "only agent_state and interaction import the client"
    if (a["area"] == "client" and b["area"] == "client" and edge["load"]
            and b["rank"] > a["rank"]):
        return f"rank {a['rank']} imports rank {b['rank']} at load time"
    return ""


def cycles(data: dict) -> tuple[list[list[str]], list[list[str]]]:
    """The load-time cycles across units, and those inside one unit of ARCHITECTURE.md's
    table (a package and its own sub-packages), which break no rule."""
    adj: dict[str, list[str]] = {u["id"]: [] for u in data["units"]}
    for edge in data["edges"]:
        if edge["load"]:
            adj[edge["from"]].append(edge["to"])
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    found: list[list[str]] = []

    def visit(v: str) -> None:  # Tarjan's strongly connected components
        index[v] = low[v] = len(index)
        stack.append(v)
        for w in adj[v]:
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                found.append(sorted(component))

    for v in adj:
        if v not in index:
            visit(v)
    found.sort()
    arch_unit = lambda u: ".".join(u.split(".")[:2]) if u.startswith("client.") else u
    across = [c for c in found if len({arch_unit(u) for u in c}) > 1]
    return across, [c for c in found if c not in across]


def _place(unit: str, ranks: dict[str, tuple[int, str]], readers: dict[str, str]) -> dict:
    """Where a unit is drawn and what the architecture says it is."""
    if unit.startswith("client."):
        top = unit.split(".")[1]
        rank, what = ranks.get(top, (None, ""))
        return {"area": "client", "rank": rank, "role": what}
    if unit.startswith("servers."):
        return {"area": "servers", "rank": None, "role": ""}
    if unit.startswith("shared."):
        return {"area": "shared", "rank": None, "role": readers.get(unit, "unlisted")}
    return {"area": "measure", "rank": FRONT_RANK, "role": "drives the agent without a person"}


def _edge_summary(edge: dict) -> dict:
    """An edge with its kind: load time when any of its imports runs at load time."""
    edge["load"] = any(s["load"] for s in edge["sites"])
    edge["n"] = len(edge["sites"])
    return edge


def structure(data: dict) -> dict:
    """What ``--check`` compares: the data less sizes and line numbers, which every edit moves."""
    return {
        "units": [{"id": u["id"], "rank": u["rank"], "area": u["area"], "role": u["role"],
                   "doc": u["doc"], "files": [(f["path"], f["doc"]) for f in u["files"]]}
                  for u in data["units"]],
        "edges": [{"from": e["from"], "to": e["to"], "load": e["load"], "bad": e.get("bad"),
                   "sites": sorted((s["file"], s["module"], tuple(s["names"]), s["load"])
                                   for s in e["sites"])}
                  for e in data["edges"]],
        "cycles": data.get("cycles"),
        "local_cycles": data.get("local_cycles"),
    }


def page_data(text: str) -> dict:
    match = DATA_BLOCK.search(text)
    if match is None:
        raise SystemExit(f"{PAGE} has no map-data block")
    return json.loads(match[2])


def render(data: dict) -> str:
    """The data as the page embeds it: one unit or edge per line, so a diff reads."""
    def rows(items: list) -> str:
        return ",\n".join("  " + json.dumps(i, sort_keys=True, ensure_ascii=False)
                          for i in items)
    body = ('{"units": [\n' + rows(data["units"]) + '\n], "edges": [\n' + rows(data["edges"])
            + '\n], "cycles": ' + json.dumps(data["cycles"])
            + ', "local_cycles": ' + json.dumps(data["local_cycles"]) + "}")
    return "\n" + body.replace("</", "<\\/") + "\n"


# ── the image ────────────────────────────────────────────────────────────────
# The page's layout, drawn once: client units in one row per rank, servers in a column on
# the right, shared below, one band per reader. Sizes and line numbers are left out, so the
# image changes only when the structure does.
RANK_NAMES = {
    11: "front-ends", 10: "session", 9: "agent_core", 8: "query_engine", 7: "prompt",
    6: "extensions · integration", 5: "tool_execution", 4: "guardrails", 3: "core",
    2: "context", 1: "tools · llm", 0: "config · protocol",
}
GROUPS = {
    "front": "Front-ends and measurement", "engine": "Conversation and engine (ranks 6–10)",
    "gov": "Guardrails and execution (4–5)", "found": "Contracts and foundations (0–3)",
    "servers": "MCP servers", "shared": "shared",
}
SHARED_BANDS = [("shared · client ↔ servers contract", "contract"),
                ("shared · servers only", "servers"),
                ("shared · not listed in ARCHITECTURE.md", "unlisted")]
LEFT, ROW_H, NODE_H, GAP, CHAR_W, PAD, SRV_GAP, TOP = 150, 54, 26, 12, 7.1, 18, 60, 72
STYLE = """
svg { --bg: #ffffff; --ink: #18212b; --muted: #5d6b79; --edge: #9aa8b6; --bad: #c0392b;
  --band: #eef1f4; --g-front: #dfe8f7; --b-front: #5b7fb8; --g-engine: #e3efe6;
  --b-engine: #4d8a5e; --g-gov: #efe4f2; --b-gov: #8a5b9a; --g-found: #f3ecd9;
  --b-found: #a8873a; --g-servers: #dcecef; --b-servers: #3f8794; --g-shared: #e9e6dc;
  --b-shared: #8a8162; }
@media (prefers-color-scheme: dark) { svg { --bg: #18202a; --ink: #e3e9ef; --muted: #93a1af;
  --edge: #4a5868; --bad: #ff6b5b; --band: #151c24; --g-front: #1d2a40; --b-front: #7fa2dc;
  --g-engine: #1d3226; --b-engine: #74b787; --g-gov: #2e2236; --b-gov: #b588c5;
  --g-found: #332b18; --b-found: #d0ad5c; --g-servers: #183238; --b-servers: #6cb6c3;
  --g-shared: #2c2a22; --b-shared: #b4aa86; } }
text { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; fill: var(--ink); }
.bg { fill: var(--bg); } .band { fill: var(--band); }
.head { font-size: 15px; font-weight: 600; } .meta, .label { font-size: 11px; fill: var(--muted); }
.edge { fill: none; stroke: var(--edge); opacity: .5; }
.deferred { stroke-dasharray: 4 3; } .bad { stroke: var(--bad); opacity: .9; }
.node rect { stroke-width: 1.4; } .node text { font-size: 11.5px; }
.cycle rect { stroke: var(--bad); stroke-width: 2.4; }
""" + "".join(f".{g} rect {{ fill: var(--g-{g}); stroke: var(--b-{g}); }}\n" for g in GROUPS)


def _group(unit: dict) -> str:
    if unit["area"] in ("servers", "shared"):
        return unit["area"]
    rank = unit["rank"]
    return "front" if rank >= 11 else "engine" if rank >= 6 else "gov" if rank >= 4 else "found"


def _label(unit_id: str) -> str:
    """A unit's name on the map: a sub-package by its leaf, its parent beside it in the row."""
    short = re.sub(r"^(client|servers|shared)\.", "", unit_id)
    if re.match(r"^(guardrails|query_engine)\.", short):
        return short.split(".")[-1]
    if short.startswith("ui.ws."):
        return "ws." + short.split(".")[-1]
    return short


def _box_w(unit_id: str) -> float:
    return max(54, math.ceil(len(_label(unit_id)) * CHAR_W) + PAD)


def _row_w(ids: list[str]) -> float:
    return sum(_box_w(i) for i in ids) + GAP * max(0, len(ids) - 1)


def _num(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def draw(data: dict) -> str:
    """The map as a standalone SVG: boxes, edges, rule breaks in red, cycles outlined."""
    units = {u["id"]: u for u in data["units"]}
    ranked = [u for u in data["units"] if u["area"] in ("client", "measure")]
    rows = [(r, [u["id"] for u in ranked if u["rank"] == r])
            for r in sorted({u["rank"] for u in ranked}, reverse=True)]
    servers = [u["id"] for u in data["units"] if u["area"] == "servers"]
    bands = [(name, [u["id"] for u in data["units"]
                     if u["area"] == "shared" and u["role"] == role])
             for name, role in SHARED_BANDS]
    bands = [b for b in bands if b[1]]

    client_w = max(_row_w(ids) for _r, ids in rows)
    pos: dict[str, tuple[float, float, float]] = {}
    for i, (_r, ids) in enumerate(rows):
        x = LEFT + (client_w - _row_w(ids)) / 2
        for unit_id in ids:
            pos[unit_id] = (x, TOP + i * ROW_H, _box_w(unit_id))
            x += _box_w(unit_id) + GAP
    srv_x = LEFT + client_w + SRV_GAP
    srv_w = max(_box_w(s) for s in servers)
    step = (len(rows) - 1) * ROW_H / (len(servers) - 1) if len(servers) > 1 else 0
    for i, unit_id in enumerate(servers):
        pos[unit_id] = (srv_x, TOP + i * step, srv_w)
    width = srv_x + srv_w + 24
    heads = []
    y = TOP + len(rows) * ROW_H + 34
    for i, (_name, ids) in enumerate(bands):
        if i:
            y += 40 + 28
        heads.append(y)
        x = LEFT
        for unit_id in ids:
            if x + _box_w(unit_id) > width - 24:
                x, y = LEFT, y + 40
            pos[unit_id] = (x, y, _box_w(unit_id))
            x += _box_w(unit_id) + GAP
    # The legend wraps at the map's width: a swatch or a line, then its name.
    legend = []
    x, ly = 14.0, y + NODE_H + 30
    for kind, name in [*(("node " + g, n) for g, n in GROUPS.items()),
                       ("edge", "load time"), ("edge deferred", "deferred"),
                       ("edge bad", "breaks a rule")]:
        span = 20 + len(name) * 6.7
        if x + span > width - 14:
            x, ly = 14.0, ly + 20
        legend.append((kind, name, x, ly))
        x += span + 18
    height = ly + 20

    def path(a: str, b: str) -> str:
        (ax, ay, aw), (bx, by, bw) = pos[a], pos[b]
        sx, tx = ax + aw / 2, bx + bw / 2
        if abs(by - ay) <= 2:
            mid = min(ay, by) - 20
            return f"M{_num(sx)},{_num(ay)} C{_num(sx)},{_num(mid)} {_num(tx)},{_num(mid)} {_num(tx)},{_num(by)}"
        down = by > ay
        sy, ty = (ay + NODE_H, by) if down else (ay, by + NODE_H)
        dy = (ty - sy) / 2
        return (f"M{_num(sx)},{_num(sy)} C{_num(sx)},{_num(sy + dy)} "
                f"{_num(tx)},{_num(ty - dy)} {_num(tx)},{_num(ty)}")

    esc = html.escape
    bad = [e for e in data["edges"] if e["bad"]]
    in_cycle = {u for c in data["cycles"] for u in c}
    load = sum(1 for e in data["edges"] if e["load"])
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_num(width)} {_num(height)}" '
           f'width="{_num(width)}" height="{_num(height)}" role="img" '
           f'aria-label="Imports between MIMIR\'s units">',
           f"<style>{STYLE}</style>",
           f'<rect class="bg" width="{_num(width)}" height="{_num(height)}"/>',
           '<text class="head" x="14" y="26">MIMIR dependency map</text>',
           f'<text class="meta" x="14" y="44">{len(data["units"])} units · {load} load-time edges · '
           f'{len(data["edges"]) - load} deferred-only · {len(bad)} rule break'
           f'{"" if len(bad) == 1 else "s"} · {len(data["cycles"])} cycle'
           f'{"" if len(data["cycles"]) == 1 else "s"} across units</text>']
    for i, (rank, _ids) in enumerate(rows):
        ry = TOP + i * ROW_H
        out.append(f'<rect class="band" x="8" y="{ry - 12}" width="{_num(LEFT + client_w - 8)}" '
                   f'height="{ROW_H - 6}" rx="4"/>')
        out.append(f'<text class="label" x="14" y="{ry + 10}">rank {rank}</text>')
        out.append(f'<text class="label" x="14" y="{ry + 24}">{esc(RANK_NAMES.get(rank, ""))}</text>')
    out.append(f'<text class="label" x="{_num(srv_x)}" y="{TOP - 16}">MCP servers</text>')
    for (name, _ids), head in zip(bands, heads):
        out.append(f'<text class="label" x="14" y="{head - 10}">{esc(name)}</text>')
    out.append("<g>")
    for e in data["edges"]:
        if e["from"] in pos and e["to"] in pos:
            cls = "edge" + ("" if e["load"] else " deferred") + (" bad" if e["bad"] else "")
            title = f'{e["from"]} → {e["to"]} · {e["n"]} import(s)' + (f' · {e["bad"]}' if e["bad"] else "")
            out.append(f'<path class="{cls}" d="{path(e["from"], e["to"])}" '
                       f'stroke-width="{_num(min(1 + math.log2(e["n"] + 1), 4.5))}">'
                       f"<title>{esc(title)}</title></path>")
    out.append("</g>")
    for u in data["units"]:
        if u["id"] not in pos:
            continue
        x, y, w = pos[u["id"]]
        cls = "node " + _group(u) + (" cycle" if u["id"] in in_cycle else "")
        out.append(f'<g class="{cls}"><title>{esc(u["id"] + (" — " + u["doc"] if u["doc"] else ""))}</title>'
                   f'<rect x="{_num(x)}" y="{_num(y)}" width="{_num(w)}" height="{NODE_H}" rx="4"/>'
                   f'<text x="{_num(x + w / 2)}" y="{_num(y + NODE_H / 2 + 4)}" text-anchor="middle">'
                   f'{esc(_label(u["id"]))}</text></g>')
    for kind, name, x, ly in legend:
        if kind.startswith("node"):
            out.append(f'<g class="{kind}"><rect x="{_num(x)}" y="{ly - 10}" width="14" '
                       f'height="12" rx="3"/></g>')
        else:
            out.append(f'<path class="{kind}" d="M{_num(x)},{ly - 4} h14" stroke-width="2"/>')
        out.append(f'<text class="label" x="{_num(x + 20)}" y="{ly}">{esc(name)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    data = extract()
    text = PAGE.read_text()
    image = draw(data)
    if "--check" in argv:
        stale = [p.name for p, fresh in ((PAGE, structure(page_data(text)) == structure(data)),
                                         (IMAGE, IMAGE.exists() and IMAGE.read_text() == image))
                 if not fresh]
        if not stale:
            return 0
        print(f"docs/dependency-map ({', '.join(stale)}) is out of date with the code; "
              "run: python docs/dependency-map/generate.py", file=sys.stderr)
        return 1
    PAGE.write_text(DATA_BLOCK.sub(lambda m: m[1] + render(data) + m[3], text, count=1))
    IMAGE.write_text(image)
    load = sum(1 for e in data["edges"] if e["load"])
    print(f"{len(data['units'])} units, {load} load-time edges, "
          f"{len(data['edges']) - load} deferred-only -> {PAGE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
