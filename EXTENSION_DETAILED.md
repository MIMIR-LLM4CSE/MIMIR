# MIMIR Extension Detailed Reference

> **MIMIR docs** — [Overview](README.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Policy](POLICY.md) · [Client internals](CLIENT_DETAILED.md) · [Servers](SERVERS_DETAILED.md) · [Extension](EXTENSION_DETAILED.md) · [Plugins](PLUGINS_DETAILED.md)

The authoritative reference for the **VS Code extension frontend** — its layers, the
WebSocket message contract with the Python client, the React file map, and step-by-step
recipes for extending the UI. For installing and configuring the extension as a user, see
the [extension README](mimir/vscode-extension/README.md) and [`SETUP.md`](SETUP.md) §6; for
the Python agent core it talks to, see [`CLIENT_DETAILED.md`](CLIENT_DETAILED.md).

## Layers

The VS Code extension lives in `mimir/vscode-extension/`. It has two independent layers:

| Layer | Path | Language | Role |
|-------|------|----------|------|
| Extension host | `src/extension.ts` | TypeScript (Node.js) | Spawns the Python WS server, reads the endpoint's model list (`src/modelList.ts`), bridges WebSocket ↔ webview postMessage |
| Webview (React) | `webview/src/` | TypeScript + React | Chat UI, approval cards, diff bars, session panel |

After changing any file under `webview/src/` run:
```bash
cd mimir/vscode-extension
npm run deploy       # production bundle + install/update in VS Code
# or
npm run dev          # incremental rebuild on every save
```
The bundle is written to `dist/webview.js` which the extension host injects into the webview HTML.
`npm run deploy` then copies it over the installed extension; reload the VS Code window to
pick it up.

---

## Message flow

```
User types query
   │
   ▼
webview (React)  ──postMessage({type:"ws_send", payload:JSON})──▶  extension host (extension.ts)
                                                                         │
                                                                         ▼
                                                                   Python ws_server.py
                                                                         │  (WebSocket)
                                                                         ▼
                                                                   MimirAgent core
                                                                         │
                                                          streams token/status/approval/answer
                                                                         │
extension host ◀──postMessage({type:"ws", payload:JSON})────────────────┘
   │
   ▼
webview handleServerMessage() → React state update → render
```

Every Python→frontend message is a JSON object with a `type` field. The full set is defined in `webview/src/types.ts` as the `ServerMessage` discriminated union.

Six message types never reach Python — the extension host answers them itself, before
any server exists:

| Message | Direction | Meaning |
|---|---|---|
| `get_config` | webview → host | Send the connect form its starting values |
| `config` | host → webview | Those values: `backend`, `vllmBaseUrl`, `rayBaseUrl`, `ollamaBaseUrl`, `anthropicModels`, `remembered` |
| `fetch_models` | webview → host | `{backend, baseUrl}` — read the model list from that endpoint |
| `models` | host → webview | `{backend, models, error?}` — the result, or why it failed |
| `connect` | webview → host | `{model, backend, baseUrl, anthropicApiKey?, remember?}` — start the WS server and attach |
| `auto_connect` | host → webview | `{backend, baseUrl, model}` — the host reconnected on its own; show "connecting" |

### Remembering an address

The connect form's **Remember this address** checkbox is what makes MIMIR come back on
its own. Ticked, `connect` carries `remember: true` and the host stores
`{backend, baseUrl, model}` under `mimir.rememberedEndpoint` in `globalState` — an
address and a model name, never the Claude API key, which is why the checkbox is hidden
for the Anthropic backend. Unticking it and connecting forgets the stored one.

The extension activates on `onStartupFinished` and probes that entry there, once per
window, with the same `/v1/models` (or `/api/tags`) request the form makes, on a 3 s
timeout. Only a reply starts the server; an address that doesn't answer — laptop off the
VPN, compute node released — leaves the user on the connect form, pre-filled with the
remembered values and nothing spawned. This is the one path that connects without a
click, and it stays deliberate: no probe, no server.

Probing at activation rather than at the first `resolveWebviewView` is what makes the
panel already connected when the user first opens it. It also means the socket can open
before any webview exists, and the `ready` frame the Python server sends on connect goes
to nobody. So the first view to resolve re-attaches — close the socket, connect again —
and is greeted with a fresh `ready`; `_connectToServer`'s close handler ignores a socket
that is no longer `this._ws`, so the replaced one drives no retry. A startup connect also
keeps the "MIMIR Server" output channel closed instead of popping it over the editor.

The host starts the server with `--port 0` and reads the address from its
`Listening on ws://…` line. With `localhost`, the server binds 127.0.0.1 and ::1 on two
different ports. The line names the literal address of one socket, never `localhost`, so
the host dials the port that socket owns. It names the IPv4 socket when there is one:
`no_proxy` rarely lists `::1`, and a client that honours the proxy variables sent
`ws://[::1]` to the corporate proxy, which closed the connection.

`fetch_models` runs in the host rather than in React because the webview's CSP allows
only `connect-src ws://localhost:*`; the fetch itself lives in `src/modelList.ts`
(`modelsUrl` / `parseModels` are pure and unit-tested in `src/modelList.test.ts`).

That request goes through `src/directHttp.ts`, which speaks HTTP/1.1 over a `net`/`tls`
socket instead of calling `http.get`. VS Code patches http/https in the extension host to
route through the proxy it resolves (`http.proxySupport`, `"override"` by default —
it replaces even an agent the caller passes). A corporate proxy has no route to a cluster
address: it accepts the CONNECT and holds it, so a proxied probe hangs until its own
deadline and the form reports "did not answer in time" about a server that answers in
under a second. `net`/`tls` are off that patch's path. It is the same posture the Python
side takes at every cluster call (`httpx.Client(trust_env=False)`, `_direct_opener()`),
and any future host-side request to an endpoint belongs on `directGet` for that reason.
The deadline is 30 s: these endpoints are often slow to *answer*, not absent.

The Python `ready` message then carries a `thinking` descriptor —
`{mechanism, levels, can_disable}` from `thinking_profile()` in
`client/config/models.py` — and `AgentSettings.buildScale()` turns it into the depth
control's rungs. The rungs are the served family's own (a token ladder for
`enable_thinking` models, the family's named effort levels otherwise), and the value
sent back is always a `THINKING_DEPTH` index, so the rest of the protocol is
unchanged.

There are two emission paths into the WS layer. Messages originated by `ws_server.py` itself (session lifecycle, errors, todos, context-usage) are sent directly via `await self.ws.send(...)` or placed on `out_q`. Structured events originated by the **engine** (`status` / `tool_call` / `tool_result` / `diff`, plus streamed `token` / `thinking`) flow through callbacks instead of stdout: `_run_query()` binds an `event_callback` (and the token callbacks) that put the event dict straight onto `out_q`, which the drain loop forwards to the WebSocket. The engine calls `emit()` (`event_sink.py`); when no callback is bound — e.g. the CLI front-end — `emit()` prints the event as a JSON line, preserving the original behaviour. The legacy `sys.stdout` router is retained only as a defensive catch-all for stray prints.

---

## Frontend file map

```
webview/src/
├── index.tsx                 ← React entry point
├── types.ts                  ← ALL shared TypeScript types (ServerMessage, ChatMessage, …)
├── hooks/
│   ├── useWebSocket.ts       ← postMessage bridge; send(), connect(), createSession(), …
│   ├── useElapsed.ts         ← ticking elapsed-time counter for a running turn
│   ├── useStickToBottom.ts   ← follow a pane's bottom while it streams, and let go when
│   │                           the reader scrolls up. Used by the transcript and by the
│   │                           live reasoning panel
│   └── useOffscreenRows.ts   ← which tool rows have scrolled out of the thread, found by
│                               `data-tool-id`; what decides when the progress dock shows
├── state/
│   └── chatReducer.ts        ← the chat state machine: every message, tool row and
│                               thinking block the transcript holds. The reducer never
│                               mutates, which is what makes identity a valid change test
├── App.tsx                   ← root component: state wiring and handleServerMessage()
└── components/
    ├── ChatThread.tsx         ← the scrolling transcript
    ├── ChatMessage.tsx        ← one message (text / thinking / approval / tools / error)
    ├── MarkdownContent.tsx    ← markdown renderer used inside a message
    ├── mathDelimiters.ts      ← normalises LaTeX delimiters for remark-math
    ├── ThinkingPanel.tsx      ← collapsible reasoning block
    ├── StreamingStatus.tsx    ← the live "working…" indicator
    ├── ToolActivityList.tsx   ← the tool rows under a turn
    ├── CommandResult.tsx      ← terminal-style in/out panel for an exec-shaped result
    ├── FileDiff.tsx           ← unified-diff renderer
    ├── diffUtils.ts           ← diff parsing helpers
    ├── VerificationLedger.tsx ← the collapsed evidence panel under an answer
    ├── ledgerUtils.ts         ← ledger split/parse helpers (unit-tested)
    ├── CompletionReport.tsx   ← the collapsed completion report
    ├── completionUtils.ts     ← its split/parse helpers — mirrors guardrails/workflow.py
    ├── GlobalApprovalBar.tsx  ← approval banner (allow / always / deny), for sensitive
    │                            tools and out-of-workspace paths: one card per call,
    │                            listing every outside path it names
    ├── BatchReviewBar.tsx     ← sticky bar of accumulated file changes after a turn
    ├── ApprovalSwitcher.tsx   ← approval-mode picker (manual / auto / all), stacked above
    │                            the send button rather than in the settings popover,
    │                            since an auto mode answers cards for the user
    ├── ModeSwitcher.tsx       ← mode button and picker; the active mode colours the chat
    ├── UserQuestion.tsx       ← a structured question the agent asked (elicitation)
    ├── PlanBar.tsx            ← plan-mode progress bar above the input
    ├── ResumePlanPrompt.tsx   ← offers to resume an unfinished checklist on reopen
    ├── TodoSidebar.tsx        ← todo / plan items panel
    ├── SessionsPanel.tsx      ← session list (switch / rename / delete); each row shows a
    │                            model-generated one-sentence description, and a hand-picked
    │                            rename wins over it
    ├── TogglesPanel.tsx       ← server / skill / nudge toggles
    ├── AgentSettings.tsx      ← the settings popover
    ├── ConnectForm.tsx        ← connection form (backend · address · model)
    ├── ContextBar.tsx         ← context-window usage
    ├── MentionAutocomplete.tsx / mentionUtils.ts   ← the `@` attach dropdown
    ├── SlashAutocomplete.tsx  / slashUtils.ts      ← the `/` command dropdown
    ├── subAgentUtils.ts       ← sub-agent row grouping
    ├── RunProgressDock.tsx    ← runs still going whose rows have scrolled away, as small
    │                            cards at the bottom-left of the thread
    ├── runDockUtils.ts        ← which runs are still in flight, across the live step and
    │                            the frozen transcript (unit-tested)
    ├── liveStreamUtils.ts     ← puts the turn in flight — prose, reasoning, tool rows,
    │                            and a message that landed mid-step such as a steer —
    │                            back in arrival order, for the live view and the freeze
    ├── transcriptUtils.ts     ← the transcript handed back to the server
    └── MimirIntro.tsx / MimirMark.tsx              ← brand assets injected by the host
```


### Chat input autocomplete ("@" and "/")

The chat textarea in `App.tsx` drives two Copilot/Claude-style dropdowns that share the
same mechanics (a pure `detect*/filter*/apply*` helper module + a presentational
component + keyboard nav with `wrapIndex`):

- **"@" mentions** — attach MCP resources or workspace files/line-ranges as context.
  `mentionUtils.detectMentionQuery` fires whenever "@" starts a token (start of input or
  after whitespace), anywhere in the message. Picks insert `@name ` at the caret.
- **"/" slash commands** — invoke a skill explicitly (e.g. `/fix-bug …`).
  `slashUtils.detectSlashQuery` fires **only** when "/" is the first non-whitespace
  character of the input, mirroring the backend rule (`agent_loop.py`:
  `query.strip().startswith("/")`). The dropdown lists skills from `skillToggles`; picks
  insert `/name ` at the start.

Both are wired identically in `App.tsx`: `syncMention`/`syncSlash` (on change/select),
`pickMention`/`pickSlash`, and a shared `handleKeyDown` (Arrow/Enter/Tab/Esc) that routes
to whichever dropdown is open. Helper logic lives outside React so it is unit-tested with
vitest (`mentionUtils.test.ts`, `slashUtils.test.ts`).


---

## Key React state in App.tsx

| State variable | Type | Purpose |
|----------------|------|---------|
| `messages` | `ChatMessage[]` | Full chat history (persisted in memory across reconnects) |
| `draft` | `string` | Prose of the turn in flight, held **out** of `messages` until the loop accepts it. A turn is only an answer once it ends with no tool call and no guardrail sends the model back to work; streaming it straight into the transcript is what made a finished-looking answer appear and then vanish. Committed on the next step's cards, on an approval/diff/error, or superseded by `answer`; dropped on `nudge_injected`. The drop is now the backstop rather than the norm: the client holds a turn's prose off the wire entirely when a guardrail could still refuse it (see `_DraftHold` in POLICY.md), so the draft mostly carries turns that will be kept. |
| `batchFiles` | `DiffEntry[]` | Files accumulated in the BatchReviewBar since last accept/revert |
| `liveThinkingBlocks` | `ThinkingBlock[]` | Currently-streaming "Working…" blocks |
| `busy` | `boolean` | True while the agent is running (input disabled) |
| `connection` | `ConnectionState` | `disconnected \| connecting \| connected \| error` |
| `todos` | `TodoItem[]` | Current todo/plan items |
| `sessions` | `SessionMeta[]` | Session list from the server |
| `activeSessionId` | `string \| null` | Currently active session |
| `endpointModels` | `string[]` | Models the endpoint reports it serves; fills the connect dropdown. Local `App.tsx` state, not a shared type |
| `thinkingProfile` | `ThinkingProfile \| undefined` | How the served model switches reasoning, from `ready`; decides which rungs the depth control offers |

---

## How to add a new server→frontend message type

1. **Define the interface** in `webview/src/types.ts`:
   ```typescript
   export interface MyCustomMessage {
     type: "my_custom";
     value: number;
     label: string;
   }
   ```

2. **Add it to the `ServerMessage` union** in `types.ts`:
   ```typescript
   export type ServerMessage =
     | ReadyMessage
     | ...
     | MyCustomMessage;   // ← add here
   ```

3. **Handle it in `App.tsx`** inside `handleServerMessage`:
   ```typescript
   case "my_custom":
     // msg is typed as MyCustomMessage here
     setMyState(msg.value);
     break;
   ```

4. **Emit it from Python** (`ws_server.py` drain loop or `_handle()`):
   ```python
   await self.ws.send(json.dumps({"type": "my_custom", "value": 42, "label": "hello"}))
   ```

---

## How to add a new client→server message type

1. **Define the interface** in `types.ts`:
   ```typescript
   export interface MyActionMessage {
     type: "my_action";
     param: string;
   }
   ```

2. **Add it to the `ClientMessage` union** in `types.ts`:
   ```typescript
   export type ClientMessage =
     | QueryMessage
     | ...
     | MyActionMessage;
   ```

3. **Send it from a React component**:
   ```typescript
   // using the send() helper from useWebSocket (already available as prop or via context)
   send({ type: "my_action", param: "foo" });
   // or for one-off calls from deep components:
   vscodePostMessage({ type: "ws_send", payload: JSON.stringify({ type: "my_action", param: "foo" }) });
   ```

4. **Handle it in Python** (`ws_server.py`): `_Session._handle()` is a dispatch table — add a handler
   method and register it in the `_MSG_HANDLERS` map (no `if/elif` chain to edit):
   ```python
   # in _Session._MSG_HANDLERS:
   "my_action": "_handle_my_action",

   async def _handle_my_action(self, msg: dict) -> None:
       param = msg.get("param", "")
       # do something
       await self.ws.send(json.dumps({"type": "output", "text": f"did: {param}\n"}))
   ```

---

## Verification ledger panel

The agent appends its machine-recorded verification ledger to the answer text, so
conversation history carries the evidence for the model's next turn (`POLICY.md` → Final
Answer Gating). Rendering it as trailing prose put a wall of bookkeeping under every
answer, so `ChatMessage` splits the answer on the ledger's `<!--mimir:ledger …-->` marker
(`ledgerUtils.splitAnswerLedger`) and hands the block to `VerificationLedger`:

- **collapsed by default** — a native `<details>` line: chevron, status glyph, "Verification",
  then the summary as chips (`2 files`, `1 not checked`, `2 steps open`). No state, no wiring.
- **status drives the colour** — `ok` (settled evidence) / `note` (it passed but discriminates
  nothing) / `warn` (needs action) set `--ledger-accent`, which the glyph, chips, body rail and
  row dots all read from.
- **rows keep their meaning** — bold marks exactly what a reader must act on, so `rowLevel`
  tints those rows and leaves settled file rows and prose notes quiet. `inlineSegments` renders
  the rows' `` `paths` `` and bold without a markdown pass.

Nothing is lost when the marker is ignored: the block is plain markdown, and a session
reloaded from disk splits again on the stored answer text. The CLI applies the same split
(`chat_session.format_ledger_summary`, `/ledger` to expand).

---

## The terminal panel of a running command

| Event | What it carries | What the row shows |
| --- | --- | --- |
| `tool_call` | `exec: {command, stdout: "", stderr: ""}` | IN, with OUT waiting |
| `tool_result` | the full `exec` | IN and OUT |

A command's output only exists once it has finished. Building the panel from the
result alone therefore put the whole thing on screen at the end, and until then the
user watched a spinner that never said what was running. So `dispatch.py` sends the
IN half on the `tool_call` event: the command, with empty streams. The reducer stores
it on the row like any other `exec`, the row opens on it, and the result replaces it
with the complete panel.

Which calls get one is read off the registry, not off a tool name: the capability is
`CODE_EXEC`. The result preview can recognise an exec by the shape of its payload,
but at call time there is no payload yet, and a non-exec tool carrying a
`command`-ish argument would otherwise grow a terminal panel of its own.

Two details follow from a panel that exists before its output. `ExecOutput` takes a
`pending` flag — empty output under a finished run is the fact "(no output)", under a
live one it is simply not in yet — and a `tool_result` with no `exec` of its own (a
failure, say) keeps the command already on the row rather than erasing it.

---

## Clickable file names in the tool rows

| Event | What it carries | What the row shows |
| --- | --- | --- |
| `tool_result`, `ok: true` | `target: {path, name, line?, end_line?}` | the file name, underlined on hover |
| `tool_result`, `ok: false` | no `target` | the file name, as plain text |

A row shows the file name alone. So the server sends the full path on the result,
next to the name the row shows. `file_target.py` finds the path by argument name
(the `path` role, or else `path` / `filepath` / `file`), never by tool name. It adds
the lines from the result: `new_start_line` / `new_end_line` for an edit, and
`start_line` / `end_line` for a read.

The target is sent only for a call that succeeded on an existing file. A failed
read or edit says nothing reliable about the file. A running row has no link yet
either, because a write in progress may not have created its file.

A click on the name posts `open_file` with `line` and `end_line`. The extension
opens the file and selects those lines, clamped to the file as it is now. A
sub-agent's rows carry the target too, under the wire key `f`.

The row head does not use `disabled`: a disabled button drops the clicks of what it
holds, so the link inside would never fire. It uses `aria-disabled` instead.

---

## Moving a running command to the background

A tool row in `ToolActivityList.tsx` carries one control that talks to the server:
while a row is `running` and the server marked it `divertible` (the `divertible`
capability, read off the tool registry in `dispatch.py` — the webview never learns
which tool is a shell), an icon appears beside the head on hover or focus. It sends
`{type: "divert_to_background", id}`; the sentence lives in the tooltip, since the row
is already a dense line.

The click deliberately does **not** reach the model. The agent thread is parked
awaiting that very tool call, and a steer is only drained at a step boundary, so an
instruction routed that way would arrive after the run it meant to divert had ended.
`_handle_divert_to_background` serves it on the WS loop: it resolves the id to the
row's tool name, which is the run channel that tool's server publishes under, and
writes a request there for the wait loop to consume on its next tick
(`SERVERS_DETAILED.md` → *Detached runs*). The name is resolved on the client for the
same reason the webview never sees one — it sends the row, not a tool.

What comes back is an ordinary `tool_result`. A shell run's carries an `exec` with
`running: true`, a `job_key` and the output so far, but no `returncode`, because the
run has not produced one. A run with nothing to preview — an optimization run prints
to no terminal pane — carries none of that, so a separate `tool_backgrounded` follows
with the `job_key`, sent only once a watcher has actually taken the job. Either way the
reducer marks the row `background`, and `job_complete` later finds it by `exec.job_key`
**or** `jobKey` and settles it with how the run really ended, live or already frozen
into a `kind:"tools"` message.

---

## Showing what a long run is doing

A call that blocks the turn for twenty minutes cannot report on itself: it does not
answer until it is over. Two events fill the gap, and they are the two halves of one
run's life. While the call blocks, `ws_session` polls the run channel once a second and
sends `tool_progress {id, phase, percent}`. Once the run is detached, the watcher's own
status poll is the only thing still asking, and it sends `job_progress {job_key, phase,
percent}`. Neither is written to the transcript — they describe a moment, and a watcher
ticking for an hour would otherwise fill the record with "still building".

The phase text is authored server-side and rendered without being interpreted, like a
tool's `label` template. `percent` is present only when the work counts itself (a
compiler's own output); absent means *this phase does not say*, never zero — so the bar
disappears at the end of the build rather than freezing at 98% for the rest of the run.

The row shows the phase beside the elapsed time, with the shared `.streaming-dots` to
say it is still going, and draws the percentage as a translucent overlay across
`.tool-row-line`. An overlay, because the head is a button that paints its own hover
background in shorthand — anything behind it would vanish under the cursor — and
because a bar on a line of its own would add height to every running row and take it
back at the moment the run settles.

When such a row scrolls out of the thread, `RunProgressDock` puts it back as a small
card at the bottom-left, over the thread and clear of the composer. `useOffscreenRows`
decides that with an `IntersectionObserver` rooted on `.chat-thread`, finding rows by
`data-tool-id` — an attribute rather than a ref, because a row moves from the live list
into a frozen message mid-run and that replaces its element. A card needs a *live*
phase, not just a `background` status: a reloaded session brings back rows whose
watchers died with the process that wrote the file, and requiring evidence means such a
row shows a card only once something reports on it again. Clicking a card scrolls back
to its row; there is no close button, because a card is there only while both facts
hold and leaves when either stops.

---

## How to add a new chat card kind

Chat messages are rendered by `ChatMessage.tsx` based on `msg.kind`. To add a new visual card:

1. **Add the kind** to the `MessageKind` union in `types.ts`:
   ```typescript
   export type MessageKind = "text" | "status" | ... | "my_card";
   ```

2. **Add any extra fields** to `ChatMessage` in `types.ts`:
   ```typescript
   export interface ChatMessage {
     ...
     myCardData?: { title: string; value: number };
   }
   ```

3. **Render it** in `ChatMessage.tsx`:
   ```tsx
   if (msg.kind === "my_card") {
     return (
       <div className="my-card">
         <strong>{msg.myCardData?.title}</strong>: {msg.myCardData?.value}
       </div>
     );
   }
   ```

4. **Push it into `messages`** from `App.tsx` when the relevant server event arrives:
   ```typescript
   case "my_custom":
     setMessages((prev) => [
       ...prev,
       { id: makeId(), role: "agent", kind: "my_card", myCardData: { title: msg.label, value: msg.value } },
     ]);
     break;
   ```

---

## Worked examples

### Example 1 — Change the default collapsed state of a component

`BatchReviewBar.tsx` — open by default instead of collapsed:
```tsx
// Before
const [expanded, setExpanded] = useState(false);
// After
const [expanded, setExpanded] = useState(true);
```

### Example 2 — Add an emoji icon for a new tool category in "Working…" blocks

`state/chatReducer.ts`, inside `iconForTool()` — the reducer assigns each tool row its
icon, so the glyph is chosen once where the row is built rather than at render time:
```typescript
// Add before the fallback return:
if (t.startsWith("benchmarking")) return "⏱️";
```

### Example 3 — Filter out a noisy status message from the thinking block

`App.tsx`, inside `case "output"` / `case "status"`:
```typescript
if (
  t.startsWith("my noisy prefix")  // ← add this line
  || t.startsWith("✓")
  || ...
) {
  break;
}
```

### Example 4 — Add a badge to the approval card showing the affected file count

`GlobalApprovalBar.tsx`, inside the header `<div>` (the per-file card it replaced,
`InlineDiffApproval.tsx`, is gone — one banner now carries the whole call):
```tsx
{diffs.length > 0 && (
  <span className="ida-file-count">{diffs.length} file{diffs.length > 1 ? "s" : ""}</span>
)}
```
Add the CSS class in `webview/src/styles/` (or in the existing inline style block of the component).

### Example 5 — Send a custom command when the user clicks a button

In any component that receives `send` as a prop:
```tsx
<button onClick={() => (send as any)({ type: "clear_todos" })}>
  Clear plan
</button>
```

### Example 6 — Persist a new piece of UI state across sessions

1. Add the field to the `session_loaded` Python payload in `ws_server.py`.
2. Read it in the `"session_loaded"` case in `App.tsx` and call the appropriate `setState`.
3. Save it back in `_autosave_session()` in `ws_server.py`.

---

## Rebuilding after frontend changes

```bash
cd mimir/vscode-extension
npm run deploy         # one-shot production build + install/update in VS Code
```
