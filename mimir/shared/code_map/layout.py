"""How the files group into components and modules, the order the components stack in,
the cycles between modules, and the edges that break the architecture's rules."""
from __future__ import annotations

import posixpath
from collections import Counter, defaultdict

from .build_files import Build

# A directory holding at least this share of the code is where the components start.
_DOMINANT = 0.8
# The most modules one component is cut into, so a band of the map stays readable.
MAX_MODULES = 15
_FOLDED = frozenset({"include", "inc", "src", "source", "sources"})
# Code around the code: drawn as components, but it does not decide where the code starts.
_AROUND = frozenset({"test", "tests", "testing", "example", "examples", "bench", "benchmarks",
                     "doc", "docs", "script", "scripts", "tools"})
_TESTS = frozenset({"test", "tests", "testing"})


def is_tests(component: str) -> bool:
    """Whether a component holds the code's tests, which the map folds away by default."""
    return bool(_TESTS & set(component.lower().split("/")))


def source_root(files: list[str]) -> str:
    """The deepest directory holding most of the code, tests, examples and docs aside: its
    children are the components."""
    files = [f for f in files if not _AROUND & set(f.lower().split("/")[:-1])] or files
    root = ""
    while True:
        under = [f[len(root) + 1:] if root else f for f in files if not root or f.startswith(root + "/")]
        heads = Counter(p.split("/", 1)[0] if "/" in p else "." for p in under)
        top, n = heads.most_common(1)[0] if heads else (".", 0)
        if top == "." or n < _DOMINANT * len(under):
            return root
        root = posixpath.join(root, top) if root else top


def components(files: list[str], root: str) -> dict[str, str]:
    """Each file's component: a child of *root*, or the top directory of a file outside it."""
    out = {}
    for f in files:
        rest = f[len(root) + 1:] if root and f.startswith(root + "/") else None
        if rest is not None:
            out[f] = posixpath.join(root, rest.split("/", 1)[0]) if "/" in rest else root
        else:
            out[f] = f.split("/", 1)[0] if "/" in f else "."
    return out


def modules(files: list[str], comp_of: dict[str, str]) -> dict[str, str]:
    """Each file's module: its directory under its component, as deep as the component
    can be cut without passing ``MAX_MODULES``; include/src leaves fold into their parent."""
    parts_of: dict[str, list[str]] = {}
    for f in files:
        comp = comp_of[f]
        rel = posixpath.dirname(f)[len(comp):].strip("/") if comp != "." else posixpath.dirname(f)
        parts = [p for p in rel.split("/") if p]
        while parts and parts[-1].lower() in _FOLDED:
            parts.pop()
        parts_of[f] = parts
    by_comp: dict[str, list[str]] = defaultdict(list)
    for f in files:
        by_comp[comp_of[f]].append(f)
    out = {}
    for comp, members in by_comp.items():
        depth = 1
        for d in range(4, 0, -1):
            if len({tuple(parts_of[f][:d]) for f in members}) <= MAX_MODULES:
                depth = d
                break
        for f in members:
            out[f] = "/".join([comp] + parts_of[f][:depth])
    return out


def declared_links(build: Build, comps: set[str]) -> dict[str, set[str]]:
    """The links the build declares between components: each target folded onto the
    component its directory sits in."""
    def comp_of_dir(d: str) -> str | None:
        hits = [c for c in comps if c != "." and (d == c or d.startswith(c + "/"))]
        return max(hits, key=len) if hits else None
    target_comp = {t.name: comp_of_dir(t.dir) for t in build.targets.values()}
    target_comp.update({alias: target_comp.get(real) for alias, real in build.aliases.items()})
    out: dict[str, set[str]] = defaultdict(set)
    for t in build.targets.values():
        a = target_comp.get(t.name)
        if a is None:
            continue
        out.setdefault(a, set())
        for link in t.links:
            b = target_comp.get(link)
            if b and b != a:
                out[a].add(b)
    return dict(out)


def reachable(graph: dict[str, set[str]], a: str) -> set[str]:
    seen: set[str] = set()
    todo = [a]
    while todo:
        for b in graph.get(todo.pop(), ()):
            if b not in seen:
                seen.add(b)
                todo.append(b)
    return seen


def strongly_connected(nodes: list[str], succ: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's components of more than one node, iteratively so a deep code cannot
    exhaust the stack."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    counter = 0
    for start in nodes:
        if start in index:
            continue
        work = [(start, iter(sorted(succ.get(start, ()))))]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on.add(start)
        while work:
            v, it = work[-1]
            w = next(it, None)
            if w is not None:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(sorted(succ.get(w, ())))))
                elif w in on:
                    low[v] = min(low[v], index[w])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1:
                    out.append(sorted(comp))
    return out


def tiers(comps: list[str], graph: dict[str, set[str]]) -> dict[str, int]:
    """Each component's tier: the longest chain of links below it, a cycle counting as one."""
    groups = strongly_connected(comps, graph)
    rep = {c: c for c in comps}
    for g in groups:
        for c in g:
            rep[c] = g[0]
    dag: dict[str, set[str]] = defaultdict(set)
    for a in comps:
        for b in graph.get(a, ()):
            if b in rep and rep[a] != rep[b]:
                dag[rep[a]].add(rep[b])
    memo: dict[str, int] = {}

    def depth(c: str) -> int:
        if c not in memo:
            memo[c] = 0
            memo[c] = max((depth(d) + 1 for d in dag.get(c, ())), default=0)
        return memo[c]
    return {c: depth(rep[c]) for c in comps}


def rule_break(edge: dict, comp_of: dict[str, str], declared: dict[str, set[str]],
               langs: dict[str, str], root: str) -> str:
    """Which rule an edge breaks, or "": an include from the code under *root* into a
    component its build does not link, or an api/ file reaching its own impl/."""
    a, b = comp_of[edge["from"]], comp_of[edge["to"]]
    c_family = langs[edge["from"]] in ("c", "cpp") and langs[edge["to"]] in ("c", "cpp")
    inner = not root or a.startswith(root + "/")
    if (edge["kind"] == "include" and c_family and inner and a != b and a in declared
            and b in declared and b not in reachable(declared, a)):
        return f"{posixpath.basename(a)} does not link {posixpath.basename(b)}"
    if a == b and "/api/" in "/" + edge["from"] and "/impl/" in "/" + edge["to"]:
        return "an api reaches its own implementation"
    return ""
