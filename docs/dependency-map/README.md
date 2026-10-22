# Dependency map

An interactive map of MIMIR's Python units and the imports between them. Open
[`index.html`](index.html) in a browser; it is one self-contained file and works offline.

- **Units**: a client package (one box per sub-package for `guardrails`, `query_engine`
  and `ui`), a server package, a module of `shared`, and `runner` / `evals`.
- **Rows**: the client's ranks from [ARCHITECTURE.md](../ARCHITECTURE.md), servers on
  the right, `shared` below, split by reader as ARCHITECTURE.md lists them.
- **Edges**: solid when the import runs at load time, dashed when it runs only inside a
  function. Imports under `TYPE_CHECKING` are not drawn. A red edge breaks one of
  ARCHITECTURE.md's rules (an upward import, the client importing a server, `shared`
  importing either side…); the header counts those and the load-time cycles across units.
- **Detail**: click a unit for its role, its files (with their docstring and size) and
  every import statement behind each edge, linked to its line. The selection is kept in
  the URL (`index.html#client.query_engine`), so a link can point at a unit.

## Keeping it current

The page's layout and script are hand-written; its data block is generated from the code:

```bash
python docs/dependency-map/generate.py          # rewrite the data
python docs/dependency-map/generate.py --check  # exit 1 if the page is stale
```

`mimir/tests/test_dependency_map.py` runs the same check in the test suite, so a change
that adds, moves or removes a module or an import between units fails until the page is
regenerated. Sizes and line numbers are refreshed by the generator but not checked, since
every edit moves them.
