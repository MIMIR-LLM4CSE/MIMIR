"""MIMIR's dependency map, read from the code into the data block of ``index.html``.

Usage: python docs/dependency-map/generate.py [--check]
Without a flag, rewrites the page's data. With ``--check``, exits 1 when the page's
structure (units, files, imports) no longer matches the code. The layout and the script
of the page are hand-written and left as they are.
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE = REPO / "mimir"
PAGE = HERE / "index.html"
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
    """ARCHITECTURE.md's rank table: client unit -> (rank, what it is)."""
    out: dict[str, tuple[int, str]] = {}
    for line in (REPO / "ARCHITECTURE.md").read_text().splitlines():
        match = re.match(r"\|\s*(\d+)\s*\|([^|]*)\|([^|]*)\|", line)
        if match:
            rank, units, what = int(match[1]), match[2], match[3].strip()
            for unit in re.findall(r"`(\w+)`", units):
                out[unit] = (rank, what)
    return out


def shared_readers() -> dict[str, str]:
    """ARCHITECTURE.md's two readers of shared: module -> "contract" or "servers"."""
    out: dict[str, str] = {}
    for line in (REPO / "ARCHITECTURE.md").read_text().splitlines():
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
    return {"units": [units[k] for k in sorted(units)],
            "edges": [_edge_summary(edges[k]) for k in sorted(edges)]}


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
        "edges": [{"from": e["from"], "to": e["to"], "load": e["load"],
                   "sites": sorted((s["file"], s["module"], tuple(s["names"]), s["load"])
                                   for s in e["sites"])}
                  for e in data["edges"]],
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
    body = '{"units": [\n' + rows(data["units"]) + '\n], "edges": [\n' + rows(data["edges"]) + "\n]}"
    return "\n" + body.replace("</", "<\\/") + "\n"


def main(argv: list[str]) -> int:
    data = extract()
    text = PAGE.read_text()
    if "--check" in argv:
        if structure(page_data(text)) == structure(data):
            return 0
        print("docs/dependency-map/index.html is out of date with the code; "
              "run: python docs/dependency-map/generate.py", file=sys.stderr)
        return 1
    PAGE.write_text(DATA_BLOCK.sub(lambda m: m[1] + render(data) + m[3], text, count=1))
    load = sum(1 for e in data["edges"] if e["load"])
    print(f"{len(data['units'])} units, {load} load-time edges, "
          f"{len(data['edges']) - load} deferred-only -> {PAGE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
