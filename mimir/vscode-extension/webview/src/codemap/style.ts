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
  --time: #f2663a; --gpu: #4c9aff; --mem: #2fbfa0; --pill-fg: #111111; --track: #3a3a3a;
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
  --e-added: #1a7f37; --ok: #1a7f37; --time: #e0531f; --gpu: #0969da; --mem: #0f8a74; --pill-fg: #ffffff; --track: #e3e3e3;
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
.icon.is-busy { display: inline-block; animation: spin 0.9s linear infinite; color: var(--accent); cursor: progress; }
@keyframes spin { to { transform: rotate(360deg); } }
.status.is-warn { color: var(--warn); }
.status.flash { animation: status-in 1.6s ease-out; }
@keyframes status-in { 0% { color: var(--accent); } 100% { color: var(--fg-dim); } }
button:focus-visible, [tabindex]:focus-visible { outline: 1px solid var(--focus); outline-offset: 2px; }
.ribbon { height: 1px; background: var(--mimir-gradient); opacity: .7; margin-top: -6px; }
.tab { display: grid; grid-template-columns: minmax(0, 1fr) 300px; background: var(--bg);
  border: 1px solid var(--border-soft); border-radius: 6px; min-width: 0; }
@media (max-width: 820px) { .tab { grid-template-columns: minmax(0, 1fr); } }
.tab-map { padding: 14px; min-width: 0; border-right: 1px solid var(--border-soft); user-select: none; }
.tab-side { padding: 12px; display: grid; gap: 12px; align-content: start; min-width: 0; }
.lmap { overflow-x: auto; }
.lmap svg { display: block; }

.band-bg { fill: var(--side); cursor: pointer; }
.band-bg.is-sel { stroke: var(--accent); stroke-width: 1.5; }
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
.row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 6px; align-items: center;
  font: 11.5px var(--mono); padding: 1px 2px; border-radius: 3px; }
.row:hover { background: var(--hover); }
.row > span { color: var(--fg-dim); }
.row .go { all: unset; cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--fg);
  display: flex; gap: 6px; min-width: 0; }
.row .nm { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row .sub { flex: 0 1 auto; max-width: 45%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--fg-dim); }
.row .go:hover { color: var(--accent); }
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
.merge-go { margin: 0 0 0 auto; }
.session .title::before { background: #f2663a; }
.sum { font-size: 12px; color: var(--fg-dim); font-variant-numeric: tabular-nums; }
.sum b { color: var(--fg); font-weight: 600; }
.chart-wrap { min-width: 0; overflow: hidden; }
.chart { display: block; }
.chart .grid { stroke: var(--border-soft); stroke-width: 1; }
.chart .tick, .chart .xl { font: 10.5px var(--mono); fill: var(--fg-dim); }
.chart .base-line { stroke: var(--fg-dim); stroke-width: 1; stroke-dasharray: 4 4; opacity: .7; }
.chart .base-label { font: 10.5px var(--font); fill: var(--fg-dim); }
.chart .best { fill: none; stroke: var(--time); stroke-width: 2; stroke-linejoin: round; }
.chart .dot { cursor: pointer; }
.chart .dot .hit { fill: transparent; }
.chart .dot .mark { stroke-width: 2; }
.chart .dot.base .mark { fill: var(--bg); stroke: var(--fg); }
.chart .dot.kept .mark { fill: var(--ok); stroke: var(--ok); }
.chart .dot.rej .mark { fill: var(--bg); stroke: var(--fg-dim); }
.chart .dot .ring { fill: none; stroke: var(--accent); stroke-width: 2; }
.chart .dot.is-after { opacity: .5; }
.chart .dot:hover .mark { stroke: var(--accent); }
.chips { display: flex; flex-wrap: wrap; gap: 4px; }
.chip { font: 11.5px var(--font); padding: 2px 8px; border: 1px solid var(--border); border-radius: 10px;
  background: var(--surface-1); color: var(--fg-dim); cursor: pointer; font-variant-numeric: tabular-nums; }
.chip b { font-weight: 600; }
.chip.kept b { color: var(--ok); } .chip.rej b { color: var(--fg-dim); } .chip.base b { color: var(--fg); }
.chip:hover { background: var(--hover); color: var(--fg); }
.chip[aria-current="true"] { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 18%, var(--surface-1)); color: var(--fg); }
.chip.is-after { opacity: .55; }
.edge.is-added { stroke: var(--e-added); stroke-opacity: 1; }
.edge.is-removed { stroke: var(--edge); stroke-opacity: .9; stroke-dasharray: 10 6; }
.m.is-touched rect.box { stroke: var(--ok); stroke-width: 2.2; }
.m.is-touched-old rect.box { stroke: color-mix(in srgb, var(--ok) 45%, var(--border)); stroke-width: 1.6; }
.pill rect.ok { fill: var(--ok); } .pill rect.no { fill: var(--e-break); }
.pill text { font: 600 10px var(--font); fill: var(--pill-fg); }
.section.step .title::before { background: #f2663a; }
.is-ok { color: var(--ok); } .is-danger { color: var(--e-break); }
.sw.dash { border-top-style: dashed; }
.sw.bar { border-top-width: 4px; border-radius: 2px; border-color: var(--time); }
.m .track { fill: var(--track); }
.m .fill { fill: var(--time); }
.m .pct { font: 10px var(--font); fill: var(--fg); font-variant-numeric: tabular-nums; }
.hatch { fill: var(--time); }
.m .fill.gpu, .hatch.gpu { fill: var(--gpu); }
.m .pct.gpu { fill: var(--gpu); }
.m .fn { font: 10.5px var(--mono); fill: var(--fg); }
.m .fn-pct { font: 10px var(--font); fill: var(--fg-dim); }
.m .fill.fn-bar { fill: color-mix(in srgb, var(--time) 65%, var(--surface-1)); }
.sw.bar.gpu { border-color: var(--gpu); }
.section.time .title::before { background: #f2663a; }
.open[disabled] { opacity: .6; cursor: default; }
/* R&D requests: the form, their status, their variants */
.section.req-form .title::before, .section.reqs .title::before, .section.rq .title::before { background: var(--accent); }
/* One track that cannot grow past the pane: a long path wraps or is cut, never widens it. */
.section.req-form { display: grid; grid-template-columns: minmax(0, 1fr); gap: 6px; border-color: var(--accent); }
.section.req-form > * { min-width: 0; }
/* Just opened: one pulse of the accent, so the eye finds it. */
.section.req-form.is-new { animation: rq-open 1.4s ease-out; scroll-margin: 12px; }
@keyframes rq-open { 0% { box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 55%, transparent); } 100% { box-shadow: 0 0 0 0 transparent; } }
.section.req-form .title { margin-bottom: 2px; }
.hint { font-size: 11.5px; color: var(--fg-dim); font-style: normal; }
.cap .hint { text-transform: none; letter-spacing: 0; font-weight: 400; }
.fl { display: grid; gap: 3px; font-size: 11.5px; color: var(--fg-dim); }
.fl select, .fl input, .fl textarea { font: 12.5px var(--font); color: var(--vscode-input-foreground, var(--fg));
  background: var(--vscode-input-background, var(--bg)); border: 1px solid var(--vscode-input-border, var(--border));
  border-radius: 3px; padding: 4px 6px; width: 100%; resize: vertical; }
.fl select:focus, .fl input:focus, .fl textarea:focus { outline: 1px solid var(--focus); outline-offset: -1px; }
.tchips { display: flex; flex-wrap: wrap; gap: 4px; }
.tchip { display: inline-flex; align-items: center; gap: 2px; max-width: 100%; min-width: 0; font: 11.5px var(--mono);
  padding: 1px 2px 1px 7px; border: 1px solid var(--border); border-radius: 10px; background: var(--bg); }
.tchip .tl { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; direction: rtl; text-align: left; }
.tchip .x { flex-shrink: 0; }
.tchip .x { all: unset; cursor: pointer; padding: 0 5px; color: var(--fg-dim); }
.tchip .x:hover { color: var(--e-break); }
.finds { margin: 0; padding-left: 16px; font-size: 11.5px; color: var(--fg); display: grid; gap: 2px; overflow-wrap: anywhere; }
.actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 6px; }
.actions .open { margin-top: 0; }
.open.ghost { background: transparent; color: var(--accent); border: 1px solid var(--accent); }
.st { display: inline-block; font: 600 10px var(--font); letter-spacing: .04em; text-transform: uppercase; padding: 1px 7px;
  border-radius: 8px; color: var(--pill-fg); background: var(--fg-dim); }
.st-studying { background: var(--gpu); } .st-proposed { background: var(--e-cycle); } .st-trying { background: var(--time); }
.st-done { background: var(--ok); } .st-dropped { background: transparent; color: var(--fg-dim); border: 1px solid var(--border); }
.rq-row { grid-template-columns: minmax(0, 1fr) auto; }
.row > span.st { color: var(--pill-fg); } .row > span.st-dropped { color: var(--fg-dim); }
.rq-row .go b { color: var(--accent); font-weight: 600; }
.section.rq .idea { font-size: 12.5px; margin: 6px 0 2px; overflow-wrap: anywhere; }
.vt { width: 100%; table-layout: fixed; border-collapse: collapse; font-size: 12px; }
.vt td:first-child { width: 52%; overflow-wrap: anywhere; }
.vt td { padding: 3px 4px; border-bottom: 1px solid var(--border-soft); }
.vt td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
.vt tr[data-step-n] { cursor: pointer; }
.vt tr[data-step-n]:hover { background: var(--hover); }
.rq-pill { cursor: pointer; }
.rq-pill rect { fill: var(--fg-dim); }
.rq-pill text { font: 600 10px var(--font); fill: var(--pill-fg); }
.rq-pill.st-studying rect { fill: var(--gpu); } .rq-pill.st-proposed rect { fill: var(--e-cycle); }
.rq-pill.st-trying rect { fill: var(--time); } .rq-pill.st-done rect { fill: var(--ok); }
.chip .rq { font: 600 10px var(--font); color: var(--accent); }
/* A profile in the side pane: a bar behind each row on its list's largest, the GPU's time as
   one stacked bar, the counters as tiles, fractions of a peak as gauges, notes set apart. */
.row.meter { --bar: color-mix(in srgb, var(--time) 70%, transparent); padding-bottom: 3px;
  background: linear-gradient(var(--bar), var(--bar)) no-repeat 2px 100% / calc(var(--w) - 4px) 2px; }
.row.meter:hover { background: linear-gradient(var(--bar), var(--bar)) no-repeat 2px 100% / calc(var(--w) - 4px) 2px, var(--hover); }
.row.meter.gpu { --bar: color-mix(in srgb, var(--gpu) 75%, transparent); }
.row.meter.mem { --bar: color-mix(in srgb, var(--mem) 75%, transparent); }
.row.meter > span:last-child { color: var(--fg); font-variant-numeric: tabular-nums; }
.row > .nm { color: var(--fg); }
.row.two .go { flex-direction: column; gap: 0; }
.row.two .sub { max-width: 100%; font-size: 10.5px; }
.line.kv { font-size: 11.5px; padding: 1px 0; }
.meta-line { display: flex; flex-wrap: wrap; gap: 0 6px; font: 11px var(--mono); color: var(--fg-dim); margin-bottom: 2px; }
.meta-line span + span::before { content: "·"; margin-right: 6px; }
.note { font-size: 11.5px; line-height: 1.4; color: var(--fg-dim); border-left: 2px solid var(--border); padding: 1px 0 1px 7px; margin: 5px 0 2px; }
.stack { display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--track); margin: 2px 0 5px; }
.stack i, .keys i { display: block; }
.stack .k, .keys .k { background: var(--gpu); }
.stack .c, .keys .c { background: color-mix(in srgb, var(--gpu) 45%, var(--track)); }
.stack .i, .keys .i { background: var(--track); }
.keys { display: flex; flex-wrap: wrap; gap: 2px 12px; font-size: 11.5px; color: var(--fg-dim); margin-bottom: 4px; }
.keys span { display: inline-flex; align-items: center; gap: 5px; }
.keys i { width: 8px; height: 8px; border-radius: 2px; }
.keys b { color: var(--fg); font-weight: 600; font-variant-numeric: tabular-nums; }
.tiles { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px; margin-bottom: 6px; }
.tile { display: flex; align-items: baseline; justify-content: space-between; gap: 6px; background: var(--bg);
  border: 1px solid var(--border-soft); border-radius: 4px; padding: 3px 7px; min-width: 0; }
.tile b { order: 2; font: 600 13px var(--font); font-variant-numeric: tabular-nums; }
.tile span { font-size: 11px; color: var(--fg-dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.gauge { margin: 4px 0 6px; }
.gauge .gl { display: flex; justify-content: space-between; gap: 8px; font-size: 11.5px; color: var(--fg-dim); margin-bottom: 3px; }
.gauge .gl b { color: var(--fg); font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
.gauge .gt { height: 5px; border-radius: 3px; background: var(--track); overflow: hidden; }
.gauge .gt i { display: block; height: 100%; border-radius: 3px; background: var(--gpu); }
.gauge.is-warn .gt i { background: var(--warn); }
.gauge.is-warn .gl b { color: var(--warn); }
.kname { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 4px; }
.kname > span:first-child { font: 11.5px var(--mono); min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bound { flex-shrink: 0; font: 600 10px var(--font); letter-spacing: .04em; text-transform: uppercase; padding: 1px 7px;
  border-radius: 8px; color: var(--pill-fg); background: var(--fg-dim); }
.bound.memory { background: var(--warn); } .bound.compute { background: var(--time); } .bound.latency { background: var(--e-cycle); }
.section.time .cap:first-of-type { margin-top: 4px; }
.picker { position: relative; }
.pick-btn { display: inline-flex; align-items: center; gap: 8px; max-width: 280px; font: 12px var(--font); padding: 3px 8px 3px 10px;
  border: 1px solid var(--border); border-radius: 4px; background: var(--surface-1); color: var(--fg); cursor: pointer; }
.pick-btn:hover { background: var(--hover); }
.pick-btn[aria-expanded="true"] { border-color: var(--focus); }
.pick-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--mono); font-size: 11.5px; }
.pick-btn .caret { color: var(--fg-dim); font-size: 10px; }
.pick-panel { position: absolute; right: 0; top: calc(100% + 4px); z-index: 20; width: min(520px, 92vw); max-height: 60vh; overflow: auto;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px; box-shadow: 0 8px 24px rgba(0, 0, 0, .35); padding: 4px 0; }
.pick-group { font-size: 10.5px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--fg-dim); padding: 8px 12px 3px; }
.pick-row { display: flex; align-items: center; gap: 8px; padding: 3px 12px; font-variant-numeric: tabular-nums; cursor: pointer; }
.pick-row:hover { background: var(--hover); }
.pick-row.is-cur { background: color-mix(in srgb, var(--accent) 16%, transparent); }
.pick-row input { margin: 0; accent-color: var(--accent); cursor: pointer; }
.pick-name { all: unset; cursor: pointer; font: 12px var(--mono); color: var(--fg); min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pick-name:focus-visible { outline: 1px solid var(--focus); outline-offset: 2px; }
.pick-all { display: flex; flex-direction: column; padding: 5px 12px; width: 100%; box-sizing: border-box; font-family: var(--font); }
.pick-all:hover { background: var(--hover); }
.pick-all.is-cur { background: color-mix(in srgb, var(--accent) 16%, transparent); }
.pick-all .hint, .pick-foot .hint { font-size: 11px; color: var(--fg-dim); }
.pick-branch { font: 11px var(--mono); color: var(--fg-dim); min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pick-steps { margin-left: auto; font-size: 11px; color: var(--fg-dim); white-space: nowrap; }
.pick-gain { min-width: 76px; text-align: right; font-size: 11.5px; color: var(--fg-dim); white-space: nowrap; }
.pick-gain.good { color: var(--ok); } .pick-gain.bad { color: var(--warn); }
.badge { flex-shrink: 0; font: 600 9.5px var(--font); letter-spacing: .05em; text-transform: uppercase; padding: 1px 6px;
  border-radius: 8px; border: 1px solid var(--border); color: var(--fg-dim); }
.badge.on { border-color: var(--accent); color: var(--accent); } .badge.merge { border-color: var(--e-cycle); color: var(--e-cycle); }
.pick-foot { display: flex; align-items: center; gap: 10px; border-top: 1px solid var(--border-soft); margin-top: 4px; padding: 6px 12px 4px; }
.pick-foot .open { margin: 0 0 0 auto; }
.open[disabled] { opacity: .45; cursor: default; }
.merge { display: grid; gap: 6px; padding-bottom: 8px; border-bottom: 1px solid var(--border-soft); }
.merge .title::before { background: var(--e-cycle); }
.mgraph .box { fill: var(--surface-1); stroke: var(--border); }
.mgraph .sname, .mgraph .wname { font: 11.5px var(--mono); fill: var(--fg); }
.mgraph .wname { font-weight: 600; }
.mgraph .sgain, .mgraph .wsub { font: 10.5px var(--font); fill: var(--fg-dim); }
.mgraph .wsub.good { fill: var(--ok); } .mgraph .wsub.short { fill: var(--warn); }
.mgraph .sgain .v { fill: var(--fg); font-weight: 600; }
.mgraph .sstate { font: 10.5px var(--font); fill: var(--fg-dim); }
.mgraph .sstate.kept { fill: var(--ok); } .mgraph .sstate.conflict { fill: var(--e-break); }
.mgraph .sstate.interference, .mgraph .sstate.infeasible { fill: var(--time); }
.mgraph .src.is-out .box, .mgraph .src.is-out .sname, .mgraph .src.is-out .sgain { opacity: .55; }
.mgraph .whole .box { stroke: var(--e-cycle); stroke-width: 1.5; }
.mgraph .arc, .mgraph .mlink { fill: none; }
.mgraph .arc.shared { stroke: var(--edge); stroke-width: 1.4; }
.mgraph .arc.conflict { stroke: var(--e-break); stroke-width: 1.6; stroke-dasharray: 4 3; }
.mgraph .mlink { stroke: var(--edge); stroke-width: 1.4; stroke-opacity: .7; }
.mgraph .mlink.kept { stroke: var(--ok); stroke-opacity: 1; stroke-width: 2; }
.mgraph .mlink.kept.syn, .mgraph .mlink.syn { stroke: var(--ok); }
.mgraph .mlink.intf { stroke: var(--time); stroke-opacity: 1; }
.mgraph .mlink.conflict, .mgraph .mlink.interference, .mgraph .mlink.infeasible { stroke-dasharray: 3 4; stroke-opacity: .5; }
.mgraph .mlink.pending { stroke-dasharray: 2 3; }
.mlegend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 11.5px; color: var(--fg-dim); align-items: center; }
.mlegend .sw { display: inline-block; width: 18px; margin-right: 6px; vertical-align: middle; border-top: 2px solid var(--edge); }
.mlegend .sw.arc-conflict { border-top: 2px dashed var(--e-break); }
.mlegend .sw.link-syn { border-top-color: var(--ok); } .mlegend .sw.link-intf { border-top-color: var(--time); }
@media (prefers-reduced-motion: reduce) { .m, .edge { transition: none; } .section.req-form.is-new { animation: none; }
  .icon.is-busy { animation: none; } .status.flash { animation: none; } }
`;
