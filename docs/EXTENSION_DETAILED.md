# MIMIR Extension Detailed Reference

> **MIMIR docs** — [Overview](../README.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Policy](POLICY.md) · [Client internals](CLIENT_DETAILED.md) · [Servers](SERVERS_DETAILED.md) · [Extension](EXTENSION_DETAILED.md) · [Plugins](PLUGINS_DETAILED.md)

The authoritative reference for the **VS Code extension frontend** — its layers, the
WebSocket message contract with the Python client, the React file map, and step-by-step
recipes for extending the UI. For installing and configuring the extension as a user, see
the [extension README](../mimir/vscode-extension/README.md) and [`SETUP.md`](SETUP.md) §6; for
the Python agent core it talks to, see [`CLIENT_DETAILED.md`](CLIENT_DETAILED.md).

## Layers

The VS Code extension lives in `mimir/vscode-extension/`. It has two independent layers:

| Layer | Path | Language | Role |
|-------|------|----------|------|
| Extension host | `src/` | TypeScript (Node.js) | Spawns the Python WS server, reads the endpoint's model list, bridges WebSocket ↔ webview postMessage |
| Webview (React) | `webview/src/` | TypeScript + React | Chat UI, approval cards, diff bars, session panel |

The host is built from components; `chatView.ts` decides when to attach, spawn or
reconnect and holds the others:

| Module | Owns |
|---|---|
| `extension.ts` | activation: the chat view, the `mimir-diff:` documents, the commands; `deactivate()` |
| `chatView.ts` | `MimirAgentViewProvider`: the webview, its message handlers, auto-connect and attach |
| `connection.ts` | `ServerConnection`: the socket, its retries by generation, what is queued before it opens |
| `localServer.ts` | `LocalServer`: the server this window is responsible for (spawned child, attached pid, detached flag), its log mirror, the start and stop commands |
| `serverEnv.ts`, `selfUpdate.ts` | the interpreter and environment the server runs with; the shared site release |
| `serverRegistry.ts` | the server a workspace already has, read from its registry entry |
| `diskFull.ts` | telling a start that died of an exceeded quota or a full disk, from its log or a probe write |
| `modelProbe.ts`, `modelList.ts`, `directHttp.ts` | the model list of an endpoint, fetched past the editor's proxy |
| `workspaceFiles.ts`, `fileIndex.ts` | the "@" autocomplete's file list and its ranking |
| `notifications.ts` | native notifications while the chat is not in front of the user; `showToast`, which every MIMIR notification goes through, hides them after `mimir.notifications.duration` |
| `editors.ts`, `patch.ts`, `planPreview.ts` | opening files, diffs and patches; the plan panel |
| `endpoints.ts` | the endpoint remembered for unattended reconnects |

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
webview (React)  ──postMessage({type:"ws_send", payload:JSON})──▶  extension host (chatView.ts)
                                                                         │
                                                                         ▼
                                                                   Python server/main.py
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

Every Python→frontend message is a JSON object with a `type` field. The full set is the `ServerMessage` discriminated union in `webview/src/types/index.ts`; each message is declared in the family file it belongs to (`types/turn.ts`, `types/tools.ts`, …).

These message types never reach Python — the extension host answers them itself, before
any server exists:

| Message | Direction | Meaning |
|---|---|---|
| `get_config` | webview → host | Send the connect form its starting values |
| `config` | host → webview | Those values: `backend`, `vllmBaseUrl`, `rayBaseUrl`, `ollamaBaseUrl`, `anthropicModels`, `remembered` |
| `fetch_models` | webview → host | `{backend, baseUrl}` — read the model list from that endpoint |
| `models` | host → webview | `{backend, models, error?}` — the result, or why it failed |
| `connect` | webview → host | `{model, backend, baseUrl, anthropicApiKey?, remember?}` — start the WS server and attach |
| `auto_connect` | host → webview | `{backend, baseUrl, model}` — the host reconnected on its own; show "connecting" |
| `file_search` | webview → host | `{query}` — the workspace files matching what has been typed after an "@" |
| `file_results` | host → webview | `{query, files}` — the ranked workspace-relative paths, with the query they answer |

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
and is greeted with a fresh `ready`; `ServerConnection`'s close handler ignores a socket
it no longer holds, so the replaced one drives no retry. A startup connect also
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
side takes at every cluster call (`httpx.Client(trust_env=False)`, an opener with an empty `ProxyHandler`),
and any future host-side request to an endpoint belongs on `directGet` for that reason.
The deadline is 30 s: these endpoints are often slow to *answer*, not absent.

The model picker does not depend on that probe getting through. Once connected, the agent
server — which holds the address and the key — reports what its endpoint serves as a
`served_models` message (`_send_served_models`, off-thread so a slow endpoint never holds
the event loop), and the webview prefers it over the host's own answer.

The Python `ready` message then carries a `thinking` descriptor —
`{mechanism, levels, can_disable}` from `thinking_profile()` in
`client/config/models.py` — and `AgentSettings.buildScale()` turns it into the depth
control's rungs. The rungs are the served family's own (a token ladder for
`enable_thinking` models, the family's named effort levels otherwise), and the value
sent back is always a `THINKING_DEPTH` index, so the rest of the protocol is
unchanged.

There are two emission paths into the WS layer. Messages originated by the connection itself (session lifecycle, errors, todos, context-usage) are sent directly through its `Sender` or placed on the worker's `out_q`. Structured events originated by the **engine** (`status` / `tool_call` / `tool_result` / `diff`, plus streamed `token` / `thinking`) flow through callbacks instead of stdout: `_run_query()` binds an `event_callback` (and the token callbacks) that put the event dict straight onto `out_q`, which the pool's `EventBus` drains and each connection's `EventRouter` forwards to its WebSocket. The engine calls `emit()` (`protocol/event_sink.py`); when no callback is bound — e.g. the CLI front-end — `emit()` prints the event as a JSON line. The `sys.stdout` router (`ui/ws/runtime.py`) is a catch-all for stray prints.

---

## Frontend file map

```
webview/src/
├── index.tsx                 ← React entry point
├── types/                    ← the shared TypeScript types by family (connection, settings, turn,
│                               tools, prompts, jobs, session, panels, editor, chat); index.ts
│                               holds the ServerMessage and ClientMessage unions and re-exports all
├── hooks/
│   ├── useWebSocket.ts       ← postMessage bridge; send(), connect(), createSession(), …
│   ├── useEndpoint.ts        ← the connection: socket state, endpoint, models, connect,
│   │                           rejoin, cancel
│   ├── useAgentSettings.ts   ← mode, reasoning depth, streaming, context, enforcement,
│   │                           temperature, approvals: each change sent as its command
│   ├── usePanels.ts          ← the drawers (sub-agents, scientific computing) and their
│   │                           polling
│   ├── useComposer.ts        ← the message box: draft, "@" and "/" dropdowns, history,
│   │                           keys, what Enter sends
│   ├── useSessions.ts        ← the conversations: the list, the one on screen, the cards
│   │                           other ones raised, replay, journal positions
│   ├── useTranscriptSync.ts  ← handing the rendered chat to the server, at turn ends,
│   │                           at checkpoints and on demand
│   ├── useWorkspaceState.ts  ← the checklist, the batch review, the toggles, the context
│   │                           usage
│   ├── useUserQuestion.ts    ← the question a turn parks on, answered or expired
│   ├── useApprovals.ts       ← answering approval cards, and the mode switch that answers
│   │                           the standing one
│   ├── useDisconnect.ts      ← leaving the server, asked first when work would end;
│   │                           detaching
│   ├── useThreadScroll.ts    ← where the chat pane is scrolled, and the run dock
│   ├── useElapsed.ts         ← ticking elapsed-time counter for a running turn
│   ├── useStickToBottom.ts   ← follow a pane's bottom while it streams, and let go when
│   │                           the reader scrolls up. Used by the transcript and by the
│   │                           live reasoning panel
│   └── useOffscreenRows.ts   ← which tool rows have scrolled out of the thread, found by
│                               `data-tool-id`; what decides when the progress dock shows
├── state/
│   ├── chatState.ts          ← the chat state: every message, tool row and thinking block
│   │                           the transcript holds, and the actions that change it
│   ├── chatReducer.ts        ← the reducer: one table of handlers gathered from reducer/.
│   │                           It never mutates, which is what makes identity a valid
│   │                           change test
│   ├── reducer/              ← the handlers by family (turn, stream, tools, jobs,
│   │                           session), and the transcript and tool-row helpers they share
│   └── serverMessages.ts     ← routing a server message to the handler tables of the
│                               hooks that hold its state
├── App.tsx                   ← root component: wires the hooks, the reducer and the layout
│                               (BottomToolbar, EmptyState, ReconnectPanel, EndpointForm)
└── components/
    ├── StatusBar.tsx          ← the bar above the chat: drawer toggles, connection, model
    ├── Composer.tsx           ← the message box and the controls beside it
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
    ├── ledgerUtils.ts         ← lifts the ledger block off an answer (unit-tested)
    ├── CompletionReport.tsx   ← the collapsed completion report
    ├── completionUtils.ts     ← its split/parse helpers — mirrors guardrails/workflow/completion.py
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
    ├── SessionsPanel.tsx      ← session list (switch / rename / delete); each row shows a
    │                            model-generated one-sentence description, and a hand-picked
    │                            rename wins over it
    ├── TogglesPanel.tsx       ← server / skill / nudge toggles
    ├── SubAgentsPanel.tsx     ← the sub-agents this session started, one open at a time;
    │                            opening a card is also what fetches its detail
    ├── subAgentLevels.ts      ← the rung a sub-agent may be given (unit-tested); read by
    │                            the settings popover, since it is an agent setting
    ├── subAgentPanelUtils.ts  ← what a sub-agent card says, by its state (unit-tested)
    ├── SciencePanel.tsx       ← the scientific-computing drawer: each server's own
    │                            section, nearest machine first
    ├── PanelGroups.tsx        ← a section's nested half, rendered without being read:
    │                            scheduler → kind of machine → what a probe found, with
    │                            the runs in flight as badges on the machine each is on
    ├── panelGroupUtils.ts     ← opening a row, and what a badge says (unit-tested)
    ├── PanelResizer.tsx       ← the grab edge of a drawer: its width is the reader's, and
    │                            remembered per drawer
    ├── panelWidthUtils.ts     ← clamping and storing that width (unit-tested)
    ├── AgentSettings.tsx      ← the settings popover
    ├── ConnectForm.tsx        ← connection form (backend · address · model)
    ├── ContextBar.tsx         ← context-window usage
    ├── MentionAutocomplete.tsx / mentionUtils.ts   ← the `@` attach dropdown: the MCP
    │                            resources, then the workspace files the host ranked
    ├── MentionText.tsx / mentionHighlight.ts       ← the `@`-attachments of a sent
    │                            message, drawn as chips; a file one opens the file
    ├── DraftHighlight.tsx     ← the mention tints drawn behind the input's own text
    ├── DraftAttachments.tsx   ← what the draft will attach, as chips above the input
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

The chat textarea (`Composer`, driven by `useComposer`) drives two Copilot/Claude-style dropdowns that share the
same mechanics (a pure `detect*/filter*/apply*` helper module + a presentational
component + keyboard nav with `wrapIndex`):

- **"@" mentions** — attach MCP resources or workspace files/line-ranges as context.
  `mentionUtils.detectMentionQuery` fires whenever "@" starts a token (start of input or
  after whitespace), anywhere in the message. Picks insert `@name ` at the caret.
  The rows are the matching MCP resources first, then the matching workspace files
  (`mentionUtils.buildMentionItems`): a handful of resources against a repository's
  thousands of files, and a name shorthand like `@memory` is otherwise indistinguishable
  from a half-typed path. The file names come from the extension host — the webview has
  no filesystem and its CSP reaches nothing else: each keystroke inside a mention posts
  `file_search`, and the host answers `file_results` with the ranked paths
  (`src/fileIndex.ts`; one cached `findFiles` scan per window, dropped when a file
  appears or disappears). The reply echoes the query it answers, so one overtaken by
  further typing is dropped rather than listed under a narrower query.
- **"/" slash commands** — invoke a skill explicitly (e.g. `/debug-numerics …`).
  `slashUtils.detectSlashQuery` fires **only** when "/" is the first non-whitespace
  character of the input, mirroring the backend rule (`agent_loop.py`:
  `query.strip().startswith("/")`). The dropdown lists skills from `skillToggles`; picks
  insert `/name ` at the start.

Both are wired identically in `hooks/useComposer.ts`: `syncMention`/`syncSlash` (on change/select),
`pickMention`/`pickSlash`, and a shared `handleKeyDown` (Arrow/Enter/Tab/Esc) that routes
to whichever dropdown is open. Helper logic lives outside React so it is unit-tested with
vitest (`mentionUtils.test.ts`, `slashUtils.test.ts`, `fileIndex.test.ts`).

A mention is marked while it is still being typed, too. A `<textarea>` holds characters
and not elements, so nothing inside it can be coloured — and the input has to stay a
textarea, because the caret arithmetic behind both dropdowns, the history browse and the
steer path is `value` + `selectionStart`, which a contenteditable does not have. So
`DraftHighlight.tsx` draws the draft a second time in a div underneath, paints the
mentions there, and the textarea sits on top with a transparent background. The mirror
takes its box from the textarea's own `clientWidth`/`clientHeight` (a `ResizeObserver`
plus the scroll event keep them in step) because those exclude the scrollbar: a width
that disagrees by a pixel wraps a line early and slides every tint below it. Nothing is
rendered when the draft carries no mention.

A textarea's text cannot be clicked through to anything either, so `DraftAttachments.tsx`
lists the same mentions as chips above the input: a file one opens the file, "×" removes
the token from the draft (`mentionHighlight.removeMention`, which takes one flanking space
with it). The chips are derived from the text on every keystroke and own no state — the
message is the only record of what it attaches, so a hand-typed path and one picked from
the dropdown behave alike.

Once sent, the mentions are marked in the bubble rather than left in the prose: what was
attached is the part of a message the user comes back to, and as plain text it is the
hardest part to find again. `mentionHighlight.splitMentionSegments` re-reads the message
with the backend's own rule (`(?<!\S)@(\S+)`, trailing sentence punctuation peeled off)
and `MentionText.tsx` renders each mention as a chip. A file mention is a button: it
posts `open_file` with the attached line range, the same host handler the edit cards use,
so clicking it opens the file in an editor tab at those lines. A resource mention
(`@memory://all`) is marked the same way but is not a link — it addresses a server, not
a path.


---

## Where the panel's state lives

`App.tsx` wires the hooks and the reducer. Each hook owns its state and answers the server
messages about it through a handler table (`state/serverMessages.ts`); what no table answers
goes to the chat reducer, whose handlers are grouped by family in `state/reducer/`.

| Hook | Holds |
|---|---|
| `useEndpoint` | `connection`, `agentReady`, `model`, the endpoint addresses, `endpointModels`, `remembered`, `runningServer` |
| `useAgentSettings` | `agentMode`, `thinkingLevel`, `thinkingProfile`, `streaming`, `contextMode`, `enforcement`, `temperature`, `approvalMode` |
| `usePanels` | the sub-agents, `panelSections`, `panelPending`, `panelAgentReady`, which drawer is open |
| `useComposer` | the draft, the `@` and `/` dropdowns, the sent-message history, the active editor chip |
| `useSessions` | `sessions`, `activeSessionId`, `sessionLoading`, the sessions drawer, the foreign cards, how far each journal has been rendered |
| `useTranscriptSync` | what the server was last sent of the transcript |
| `useWorkspaceState` | `todos`, `batchFiles`, the server and skill toggles, `contextUsage` |
| `useUserQuestion` | the question card the turn on screen is parked on |
| `useDisconnect` | `detached`, the disconnect held while the user is asked |
| `useThreadScroll` | whether the pane follows the bottom, and the runs in the dock |

The chat reducer's state (`state/chatState.ts`):

| State variable | Type | Purpose |
|----------------|------|---------|
| `messages` | `ChatMessage[]` | Full chat history (persisted in memory across reconnects) |
| `draft` | `string` | Prose of the turn in flight, held **out** of `messages` until the loop accepts it. A turn is only an answer once it ends with no tool call and no guardrail sends the model back to work; streaming it straight into the transcript is what made a finished-looking answer appear and then vanish. Committed on the next step's cards, on an approval/diff/error, or superseded by `answer`; dropped on `nudge_injected`. The drop is now the backstop rather than the norm: the client holds a turn's prose off the wire entirely when a guardrail could still refuse it (see `DraftHold` in POLICY.md), so the draft mostly carries turns that will be kept. |
| `liveThinkingBlocks` | `ThinkingBlock[]` | Currently-streaming "Working…" blocks |
| `busy` | `boolean` | True while the agent is running (input disabled) |

---

## The side drawers, and their width

Three drawers share `.sessions-panel`: the session list, the sub-agents, and the
scientific-computing panel. Each sets its own width through `usePanelWidth(key)` and
renders a `PanelResizer` at its right edge — drag, or arrow keys once focused
(`shift` for a bigger step). The width is clamped against the live window
(`clampPanelWidth`: a 180px floor, a 0.8-of-the-window ceiling, re-applied on resize)
and stored per drawer in `localStorage`, once per drag rather than once per move. During
the drag the element is sized once per animation frame, outside React; the chat's width
is frozen and its pointer events are off until release, and the release always cleans up.

The widths are the reader's because the content is longer than any fixed one: a node
kind's feature list, a CPU model, a sub-agent's task. Text is never cut mid-word: a
narrow drawer shows one line with an ellipsis and the full text on hover; past 360px,
values and titles wrap at word boundaries, clamped to two lines, while labels stay on one.

---

## A long session

`ChatThread` renders the tail of the conversation, `THREAD_WINDOW_STEP` (120) messages at
a time, and grows by that much when the reader reaches the top or asks, anchored so
nothing jumps. The last user message is found in one pass rather than a scan per message,
which was quadratic in the history and paid on every render.

The end-of-turn transcript handed back to the server (`transcriptUtils.ts`) is sent whole
once per `epoch`, then as deltas: `order` names every stored message by id and `changed`
carries only the messages new or replaced since `base_rev`. A server that does not hold
that revision answers `transcript_resync`, and the next send is whole.

Loading a session advances the reducer's `seq` counter past every stamp the stored
messages carry. Restarted at 1 after a window reload, new live rows were stamped below
the last stored message, which then sorted as part of the step in flight and stuck
below it.

---

## How to add a new server→frontend message type

1. **Define the interface** in the family file of `webview/src/types/` it belongs to:
   ```typescript
   export interface MyCustomMessage {
     type: "my_custom";
     value: number;
     label: string;
   }
   ```

2. **Add it to the `ServerMessage` union** in `types/index.ts`:
   ```typescript
   export type ServerMessage =
     | ReadyMessage
     | ...
     | MyCustomMessage;   // ← add here
   ```

3. **Handle it** in the handler table of the hook that holds the state it changes
   (`on` in `hooks/use*.ts`), typed by its `type`; a message no table answers goes to the
   chat reducer:
   ```typescript
   const on: ServerHandlers = useMemo(() => ({
     my_custom: (msg) => setMyState(msg.value),   // msg is MyCustomMessage here
   }), []);
   ```

4. **Emit it from Python**: a handler group sends it through its `Sender`, and the
   engine through `emit()`, which `EventRouter` (`connection/router.py`) forwards:
   ```python
   await self.out.send({"type": "my_custom", "value": 42, "label": "hello"})
   ```

---

## How to add a new client→server message type

1. **Define the interface** in the family file of `types/` it belongs to:
   ```typescript
   export interface MyActionMessage {
     type: "my_action";
     param: string;
   }
   ```

2. **Add it to the `ClientMessage` union** in `types/index.ts`:
   ```typescript
   export type ClientMessage =
     | QueryMessage
     | ...
     | MyActionMessage;
   ```

3. **Send it from a React component** with the typed `send()` from `useWebSocket`:
   ```typescript
   send({ type: "my_action", param: "foo" });
   ```
   A message for the extension host itself (open a file, connect) is a `HostMessage`
   (`src/shared/hostMessages.ts`, shared by both sides), posted with `vscodePostMessage`
   and answered from the host's handler table.

4. **Handle it in Python** (`mimir/client/ui/ws/connection/handlers/`): each handler
   group maps message types to its `on_<type>` methods in `MESSAGES`, and
   `Connection.handle()` dispatches on that table (no `if/elif` chain to edit):
   ```python
   # in the group's MESSAGES:
   "my_action": "on_my_action",

   async def on_my_action(self, msg: dict) -> None:
       param = msg.get("param", "")
       # do something
       await self.out.send({"type": "output", "text": f"did: {param}\n"})
   ```

---

## Verification ledger

The agent appends its machine-recorded verification ledger to the answer text, so
conversation history carries the evidence for the model's next turn (`POLICY.md` → Final
Answer Gating). The ledger is for the model, not the chat: `ChatMessage` splits the answer
on the ledger's `<!--mimir:ledger …-->` marker (`ledgerUtils.splitAnswerLedger`) and drops
the block, so the bubble shows the prose and the completion report alone. The stored answer
keeps it, and a session reloaded from disk splits again. The CLI still shows a one-line
summary (`chat_session.format_ledger_summary`, `/ledger` to expand).

---

## The scientific-computing panel

A drawer beside the session list, opened with the `⚗` button. Its sections are filled by
whichever servers declared `PANEL_REPORT`, asked by capability rather than by name. The
panel renders what a server chose to say and interprets none of it, which is what lets a
plugin server add a section with no change here. The sub-agent rung is not here: how far
delegation may go is an agent setting, and it sits in the settings popover beside
thinking depth and enforcement.

**Two asks, and they are not the same request.** Opening the drawer and the 20s timer
while it is open send `list_panel` with `build`; pressing `⟳` adds `refresh`. `refresh`
travels to each section's tool as the argument that tool declared in its `panel`
descriptor (`refresh_arg`), which is what makes the button read past the server's own
cache — the HPC section's per-host digest is otherwise memoised for the life of the
server, so a cluster unreachable at startup would stay invisible. It is the reader's
button alone because it costs an `ssh` to every cluster; a drawer that re-probed each
time it was opened took seconds to say what it already knew. `build` starts the
conversation's agent when it has none: the sections come from MCP servers, a conversation
holds none until its first query, and without this the drawer was empty until something
had been asked of the model. The build is never awaited — the report goes out saying
`agent_ready: false`, the drawer says so, and the worker's own `ready` is the cue to ask
again. Meanwhile that report carries the Cluster section as the workspace registry has it
(`connection/panel_runs.recorded_hosts_sections`, from `slurm_hosts` in the state directory): names
only, with the server's own ids, so the rows a reader opens survive the swap.

The sections are asked all at once rather than in turn: they are owned by different
processes, and the slow ones are slow because they are waiting on a scheduler. The
drawer costs the longest of those waits, not their sum — the same reason `_host_tree`
reaches every registered scheduler concurrently.

A section may answer with a tree as well as with lines. `groups` is recursive
(`{id, label, value?, lines?, nodes?, detail?, groups?, action?, badges?}`) and
`PanelGroupList` renders any depth of it, so the shape is the server's decision: the HPC
server fills it with the schedulers this workspace reaches, the kinds of node on each,
and what a probe read on one of them. A kind is labelled by its nodes (the digest's
`example_nodes` and `nodes_total`) and opens onto one spec per line — arch, cpus, gpu,
`cpu mem`, `gpu mem`, partitions, features, nodes. `gpu mem` is per GPU from a probe, or
said to be unknown until one runs, since Slurm's gres never carries it. Everything starts collapsed — a workspace reaching
three clusters would otherwise open onto a page of
machines. Which rows are open lives in `SciencePanel`, keyed on the servers' own ids so
the answer survives a refresh; native `<details>` would lose it, since the drawer
re-renders on every `panel_report`.

**Badges are the runs, on the machines they are running on.** The client joins them in
`connection/panel_runs.attach_runs` before the message goes out, because that is the one place both
halves exist. A badge names the node, and opens as a row *inside* the tree showing the
job id or pid, the command, its state and phase, how long it has been going, and which
agent launched it — a tooltip would be unreachable by keyboard and too small for a
command line.

**An action asks; it does not act.** A node kind nothing has been probed on carries the
request that would fix it, in words the *server* wrote — it is the end that knows the
scheduler and the partition, which keeps tool names out of the webview entirely. The
click confirms, then puts that sentence into the conversation as the user's own turn
(`sendPrompt`, the same path the input box uses), so whatever tool the model reaches for
goes through the approval that tool already requires. A panel button must not become a
second way to start an irreversible call.

## The terminal panel of a running command

| Event | What it carries | What the row shows |
| --- | --- | --- |
| `tool_call` | `exec: {command, stdout: "", stderr: ""}` | IN, with OUT waiting |
| `tool_result` | the full `exec` | IN and OUT |

A command's output only exists once it has finished. Building the panel from the
result alone therefore put the whole thing on screen at the end, and until then the
user watched a spinner that never said what was running. So `query_engine/dispatch/` sends the
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

## What a tool row says

| Slot | Comes from | How it reads |
| --- | --- | --- |
| icon | `kind`, via `KIND_ICONS` (`components/toolIcons.tsx`) | one glyph per family of work |
| family | `kind`, capitalised (`labelForKind`) | `--fg`, semibold, never truncated |
| description | `doing`, written by the model per call | `--fg-dim`, the elastic slot |
| detail | the call's salient argument (`tool_arg_preview`) | `--fg-dim`, mono, yields first |

Both of the first two come from **one** declared fact. A server names its tool's family
of work in its own declaration — `tool_caps(kind="edit")`, vocabulary in
`shared/capabilities.py` — and the row shows that word and draws its icon from
it, so the glyph and the label cannot disagree, and the reducer keys on no tool name. A
tool that declares no
family gets one derived from its capabilities (`kind_for`), so a third-party server
still lands on a sensible glyph without the client knowing anything about it.

The declared value is lower case — that is what the servers write, the wire carries and
the icon table is keyed on — and the row writes it with a capital: `Bash`, `Proxy eval`.
The first letter only, since a family is a phrase in sentence case and not a title, with
`KIND_LABELS` holding the one brand whose capital is not its first letter (`GitHub`).

The icon is **derived at render time** and never stored on the row. It is a function of
the family, so a second copy could only go stale — and one of them is an element rather
than a character (the GitHub mark), which would not survive the stored transcript's JSON
and would come back as a dead object React refuses to render.

The description is the model's own sentence about **this** call, under 15 words, and it
is the only part of the row that knows *why* the call is happening. It arrives as a
`doing` argument the client adds to every tool's schema (`_schema_for_model`) and the
dispatcher strips before the call runs — so no server declares it and none receives it.
Stripped *before the dedup key*, too: two identical calls with reworded descriptions are
one call, and left in the arguments they would walk past the repeat guards.

Beside it sits the call's **salient argument** — the url it is reaching, the job id it
is cancelling, the verdict it is recording, the file it is reading. The description says
what the call is for and this says what it is for *on*, so they are two facts rather than
two renderings of one. It is computed server-side by `tool_arg_preview` from argument
*names* in priority order: a command or code body, a search pattern, a url, an
`owner`+`repo` pair (which takes the path with it — `ci.yml` alone does not say which
repository the call reached into), a path, then the object keys (`verdict`, `query`,
`expression`, `name`, `job_id`, `packages`, …) and `op` last, since an op selects an
action and the description already carries that half. Of the 71 tools, 13 show nothing:
seven take no arguments at all, and the rest carry only a payload — a plan's text, a
json body — which the description describes better than an excerpt would.

It is dropped in two cases: when the model's own description already names it
(`dedup_row_detail`, server-side), and when the panel below carries it in full — the
command of an exec row, the expression under its typeset equation. The second holds
whether that panel is open or closed, because it exists from the moment the call is made:
the IN half travels on the `tool_call` event itself, so a failed exec row is collapsed
rather than empty and its command is one click away. Cropped into a share of a line, a
command was a fragment pretending to be one.

Visually the slot is subordinate on every axis: dimmer, monospaced, capped at 38% of the
line, and `flex-shrink: 3` against the description's `1`, so it is the first thing to
give way when the pane narrows. Nothing on the row can wrap to a second line — the head
sets no `flex-wrap` and every text span is `white-space: nowrap` — so an over-long value
is cut with an ellipsis, and it carries its untruncated text in its own `title`. The
head's tooltip says what was called; this one says what it was called on.

The model will sometimes omit the description, a third-party tool's row has none, and a
transcript recorded before any of this existed has neither field. Then the **derived
label takes the slot**. Leaving the family alone there was tried first and is too coarse
to read: four tools answer to `Todo`, so a bare `Todo` row gave no way to tell a
checklist being written from one being read. The label is the one thing that always says
which call it was, and it is already on the wire. It is matched for duplication the same
way a description is, so a row does not say "Recording the plan: refonte des lignes" and
then "refonte des lignes" again.

The derived label (`"Reading file: x.py"`, from the `label` template) is still on the
wire and still used: it is the row's tooltip and `aria-label`, and approval cards and
policy messages are written from it. It is not what the row says.

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
capability, read off the tool registry in `query_engine/dispatch/` — the webview never learns
which tool is a shell), an icon appears beside the head on hover or focus. It sends
`{type: "divert_to_background", id}`; the sentence lives in the tooltip, since the row
is already a dense line.

The click deliberately does **not** reach the model. The agent thread is parked
awaiting that very tool call, and a steer is only drained at a step boundary, so an
instruction routed that way would arrive after the run it meant to divert had ended.
`TurnHandlers.on_divert_to_background` serves it on the WS loop: it resolves the id to the
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
run's life. While the call blocks, the connection polls the run channel once a second and
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

1. **Add the kind** to the `MessageKind` union in `types/chat.ts`:
   ```typescript
   export type MessageKind = "text" | "status" | ... | "my_card";
   ```

2. **Add any extra fields** to `ChatMessage` in `types/chat.ts`:
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

4. **Put it into `messages`** with a reducer handler, in the family of `state/reducer/`
   it belongs to:
   ```typescript
   my_custom: (state, action, makeId) => ({
     ...state,
     messages: [
       ...state.messages,
       { id: makeId(), role: "agent", kind: "my_card", myCardData: { title: action.label, value: action.value } },
     ],
   }),
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

### Example 2 — Add an icon for a new tool category in "Working…" blocks

A tool row's glyph is picked from the *work family* its server declares, never from the
tool's name — so a new category is one entry in `components/toolIcons.tsx`:
```tsx
export const KIND_ICONS: Record<string, ReactNode> = {
  // ...
  benchmark: "⏱️",   // ← add this
};
```
Then declare the family on the tools that belong to it, server-side, and add it to
`TOOL_KINDS` in `shared/capabilities.py`:
```python
@mcp.tool(**tool_caps(kind="benchmark", caps=[CODE_EXEC], label="Benchmarking {target}"))
```
The declared word is both what the row shows and what the icon is drawn from, so the
two cannot disagree. A family with no icon still renders — its word, with the default
glyph — and `mimir/tests/test_phase_b_servers.py` fails if a tool declares none at all.

### Example 3 — Filter out a noisy status message from the thinking block

`state/reducer/stream.ts`, in `noteActivity` (which `output` and `status` share):
```typescript
if (
  t.startsWith("my noisy prefix")  // ← add this line
  || t.startsWith("✓")
  || ...
) {
  return state;
}
```

### Example 4 — Add a badge to the approval card showing the affected file count

`GlobalApprovalBar.tsx`, inside the header `<div>` (one banner carries the whole call,
every file of it):
```tsx
{diffs.length > 0 && (
  <span className="ida-file-count">{diffs.length} file{diffs.length > 1 ? "s" : ""}</span>
)}
```
Add the CSS class in `webview/src/styles/` (or in the existing inline style block of the component).

### Example 5 — Send a custom command when the user clicks a button

In any component that receives `send` as a prop:
```tsx
<button onClick={() => send({ type: "clear_todos" })}>
  Clear plan
</button>
```

### Example 6 — Persist a new piece of UI state across sessions

1. Add the field to the `session_loaded` payload in `connection/handlers/conversations.py`.
2. Read it in a `session_loaded` handler of the hook that holds that state.
3. Save it back through `Screen.save()` in `connection/screen.py`.

---

## Rebuilding after frontend changes

```bash
cd mimir/vscode-extension
npm test               # tsc --noEmit on the host and the webview (tests included), then vitest
npm run deploy         # one-shot production build + install/update in VS Code
```
