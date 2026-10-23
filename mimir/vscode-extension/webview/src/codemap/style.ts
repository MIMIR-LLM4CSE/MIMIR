// The code map's look: the MIMIR skin follows the VS Code theme (its own variables, with
// Dark Modern's values when one is missing); the light skin is Light Modern, whatever the
// editor's theme. One hue per kind of edge, so no two are told apart by a dash alone.

export const STYLE = `
:root {
  --bg: var(--vscode-editor-background, #1f1f1f);
  --side: var(--vscode-sideBar-background, #181818);
  --surface-1: var(--vscode-editorWidget-background, #202020);
  --surface-2: var(--vscode-sideBarSectionHeader-background, #2b2b2b);
  --hover: var(--vscode-list-hoverBackground, #2a2d2e);
  --fg: var(--vscode-foreground, #cccccc);
  --fg-dim: var(--vscode-descriptionForeground, #9d9d9d);
  --border: var(--vscode-widget-border, #313131);
  --border-soft: var(--vscode-editorGroup-border, #2b2b2b);
  --accent: var(--vscode-button-background, #0078d4);
  --accent-fg: var(--vscode-button-foreground, #ffffff);
  --focus: var(--vscode-focusBorder, #0078d4);
  --warn: var(--vscode-editorWarning-foreground, #cca700);
  --edge: #8b8b8b; --e-break: var(--vscode-errorForeground, #f85149); --e-cycle: #b180d7;
  --e-binding: #33d6f2; --e-added: #3fb950; --ok: var(--vscode-testing-iconPassed, #3fb950);
  --time: #f2663a; --pill-fg: #111111;
  --band-bar: #7a2c9e;
  --mimir-gradient: linear-gradient(90deg, #33d6f2, #1a5fd8 30%, #7a2c9e 52%, #d8203c 72%, #f2663a);
  --font: var(--vscode-font-family, -apple-system, BlinkMacSystemFont, "Segoe WPC", "Segoe UI", "Ubuntu", "Droid Sans", sans-serif);
  --mono: var(--vscode-editor-font-family, "Droid Sans Mono", "Cascadia Code", monospace);
}
:root[data-skin="light"] {
  --bg: #ffffff; --side: #f8f8f8; --surface-1: #ffffff; --surface-2: #eeeeee; --hover: #ececec;
  --fg: #3b3b3b; --fg-dim: #6a6a6a; --border: #d6d6d6; --border-soft: #e5e5e5;
  --accent: #005fb8; --accent-fg: #ffffff; --focus: #005fb8; --warn: #9a6700;
  --edge: #8c8c8c; --e-break: #cf222e; --e-cycle: #8250df; --e-binding: #0598bc; --band-bar: #8250df;
  --e-added: #1a7f37; --ok: #1a7f37; --time: #e0531f; --pill-fg: #ffffff;
  color-scheme: light;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--side); color: var(--fg); font: 13px/1.45 var(--font); }
#root { padding: 14px 16px 24px; display: grid; gap: 12px; }
header { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; }
h1 { font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; margin: 0; }
.meta { font: 11.5px var(--mono); color: var(--fg-dim); }
.push { margin-left: auto; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
.seg button { font: 12px var(--font); padding: 3px 10px; border: 0; background: var(--surface-1); color: var(--fg-dim); cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--accent); color: var(--accent-fg); }
.toggle { font: 12px var(--font); padding: 3px 10px; border: 1px solid var(--border); border-radius: 4px;
  background: var(--surface-1); color: var(--fg-dim); cursor: pointer; }
.toggle[aria-pressed="true"] { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
.icon { border: 0; background: none; color: var(--fg-dim); font-size: 15px; cursor: pointer; padding: 0 4px; }
.icon:hover { color: var(--fg); }
button:focus-visible, [tabindex]:focus-visible { outline: 1px solid var(--focus); outline-offset: 2px; }
.ribbon { height: 1px; background: var(--mimir-gradient); opacity: .7; margin-top: -6px; }
.tab { display: grid; grid-template-columns: minmax(0, 1fr) 300px; background: var(--bg);
  border: 1px solid var(--border-soft); border-radius: 6px; min-width: 0; }
@media (max-width: 820px) { .tab { grid-template-columns: minmax(0, 1fr); } }
.tab-map { padding: 14px; min-width: 0; border-right: 1px solid var(--border-soft); }
.tab-side { padding: 12px; display: grid; gap: 12px; align-content: start; min-width: 0; }
.lmap { overflow-x: auto; }
.lmap svg { display: block; }

.band-bg { fill: var(--side); }
.band-name { font: 600 11px var(--font); letter-spacing: .06em; text-transform: uppercase; fill: var(--fg); }
.band-sub { font: 10.5px var(--font); fill: var(--fg-dim); }
.band-bar { fill: var(--band-bar); }
.m { transition: opacity .15s; }
.m rect.box { fill: var(--surface-1); stroke: var(--border); cursor: pointer; }
.m:hover rect.box, .m.is-sel rect.box { stroke: var(--accent); stroke-width: 1.5; }
.m.is-dim { opacity: .3; }
.m text { pointer-events: none; }
.m .mname { font: 11.5px var(--mono); fill: var(--fg); }
.m .msub { font: 10.5px var(--font); fill: var(--fg-dim); }
.dot-cycle { fill: var(--e-cycle); }
.dot-break { fill: var(--e-break); }
.edge { fill: none; stroke: var(--edge); stroke-opacity: .25; stroke-linecap: round; transition: stroke-opacity .15s; }
.edge.is-deferred { stroke-opacity: .12; }
.edge.is-binding { stroke: var(--e-binding); stroke-opacity: .85; }
.edge.is-binding.is-byname { stroke-opacity: .4; }
.edge.is-break { stroke: var(--e-break); stroke-opacity: 1; }
.edge.is-cycle { stroke: var(--e-cycle); stroke-opacity: .9; }
.edge.is-hot { stroke: var(--fg); stroke-opacity: 1; }
.edge.is-binding.is-hot { stroke: var(--e-binding); stroke-width: 2.5; }
.edge.is-faded { stroke-opacity: .04; }
.xmark { fill: var(--e-break); font: 700 12px var(--font); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 12px; font-size: 11.5px; color: var(--fg-dim); align-items: center; }
.sw { display: inline-block; width: 16px; border-top: 2px solid; vertical-align: middle; margin-right: 5px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }

.section { background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px; padding: 10px 10px 8px; min-width: 0; }
.title { display: flex; align-items: center; gap: 7px; font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; margin-bottom: 8px; }
.title::before { content: ""; width: 3px; height: 11px; border-radius: 2px; background: var(--band-bar); flex-shrink: 0; }
.section.sel .title::before { background: #33d6f2; }
.section.flags .title::before { background: #d8203c; }
.section.bind .title::before { background: var(--e-binding); }
.line { display: flex; justify-content: space-between; gap: 10px; font-size: 12.5px; padding: 3px 0; }
.line > span:first-child { color: var(--fg-dim); max-width: 55%; }
.line > span:last-child { text-align: right; font-variant-numeric: tabular-nums; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.is-warn { color: var(--warn); } .is-cycle { color: var(--e-cycle); }
.path { font: 11.5px var(--mono); color: var(--fg-dim); word-break: break-all; margin-bottom: 4px; }
.list { display: grid; gap: 1px; max-height: 220px; overflow: auto; }
.row { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 6px; align-items: center;
  font: 11.5px var(--mono); padding: 1px 2px; border-radius: 3px; }
.row:hover { background: var(--hover); }
.row > span { color: var(--fg-dim); }
.row .go { all: unset; cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--fg); }
.row .ext { all: unset; cursor: pointer; color: var(--fg-dim); padding: 0 4px; }
.row .ext:hover, .row .go:hover { color: var(--accent); }
.cap { font-size: 10.5px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--fg-dim); margin: 8px 0 3px; }
.flag { font-size: 12px; padding: 4px 6px; border-left: 2px solid; margin-top: 4px; background: var(--bg); border-radius: 0 3px 3px 0; }
.flag.danger { border-color: var(--e-break); } .flag.cycle { border-color: var(--e-cycle); }
.empty { font-size: 12px; color: var(--fg-dim); }
.back, .link { all: unset; cursor: pointer; font-size: 12px; color: var(--accent); }
.open { margin-top: 8px; font: 12px var(--font); padding: 4px 10px; background: var(--accent); color: var(--accent-fg); border: 0; border-radius: 3px; cursor: pointer; }
/* the session: the best value as it went, one stop per step */
.session { background: var(--bg); border: 1px solid var(--border-soft); border-radius: 6px; padding: 10px 14px 12px; display: grid; gap: 8px; min-width: 0; }
.session[hidden] { display: none; }
.session-head { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; }
.session-head .title { margin: 0; }
.session .title::before { background: #f2663a; }
.sum { font-size: 12px; color: var(--fg-dim); font-variant-numeric: tabular-nums; }
.sum b { color: var(--fg); font-weight: 600; }
.steps-scroll { overflow-x: auto; }
.steps { display: grid; grid-template-rows: 40px auto; min-width: 760px; }
.spark { grid-row: 1; grid-column: 1 / -1; width: 100%; height: 40px; }
.spark .line { fill: none; stroke: var(--time); stroke-width: 2; vector-effect: non-scaling-stroke; }
.pts { grid-row: 1; grid-column: 1 / -1; position: relative; }
.pt { position: absolute; width: 10px; height: 10px; border-radius: 50%; border: 2px solid var(--time); background: var(--bg); transform: translate(-50%, -50%); }
.pt.is-cur { background: var(--time); }
.pt.is-rej { border-color: var(--fg-dim); }
.stop { all: unset; grid-row: 2; cursor: pointer; display: grid; gap: 2px; padding: 6px 8px 4px; border-radius: 4px; border-top: 2px solid transparent; min-width: 0; }
.stop:hover { background: var(--hover); }
.stop[aria-current="true"] { background: var(--surface-2); border-top-color: var(--accent); }
.stop:focus-visible { outline: 1px solid var(--focus); }
.stop .n { font: 600 11px var(--font); letter-spacing: .04em; }
.stop .v { font-weight: 400; } .stop .v.ok { color: var(--ok); } .stop .v.no { color: var(--e-break); }
.stop .ax { font-size: 11.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.stop .t { font: 11px var(--mono); color: var(--fg-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.stop.is-after .ax, .stop.is-after .t, .stop.is-after .n { opacity: .45; }
.edge.is-added { stroke: var(--e-added); stroke-opacity: 1; }
.edge.is-removed { stroke: var(--edge); stroke-opacity: .9; stroke-dasharray: 10 6; }
.m.is-touched rect.box { stroke: var(--ok); stroke-width: 2.2; }
.m.is-touched-old rect.box { stroke: color-mix(in srgb, var(--ok) 45%, var(--border)); stroke-width: 1.6; }
.pill rect.ok { fill: var(--ok); } .pill rect.no { fill: var(--e-break); }
.pill text { font: 600 10px var(--font); fill: var(--pill-fg); }
.section.step .title::before { background: #f2663a; }
.is-ok { color: var(--ok); } .is-danger { color: var(--e-break); }
.sw.dash { border-top-style: dashed; }
@media (prefers-reduced-motion: reduce) { .m, .edge { transition: none; } }
`;
