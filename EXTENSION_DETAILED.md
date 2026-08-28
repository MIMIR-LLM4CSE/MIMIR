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
| Extension host | `src/extension.ts` | TypeScript (Node.js) | Spawns the Python WS server, opens SSH/SLURM tunnels, bridges WebSocket ↔ webview postMessage |
| Webview (React) | `webview/src/` | TypeScript + React | Chat UI, approval cards, diff bars, session panel |

After changing any file under `webview/src/` run:
```bash
cd mimir/vscode-extension
npm run build        # production bundle
# or
npm run watch        # incremental rebuild on every save
```
The bundle is written to `dist/webview.js` which the extension host injects into the webview HTML.

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

There are two emission paths into the WS layer. Messages originated by `ws_server.py` itself (session lifecycle, errors, todos, context-usage) are sent directly via `await self.ws.send(...)` or placed on `out_q`. Structured events originated by the **engine** (`status` / `tool_call` / `tool_result` / `diff` / `file_access`, plus streamed `token` / `thinking`) flow through callbacks instead of stdout: `_run_query()` binds an `event_callback` (and the token callbacks) that put the event dict straight onto `out_q`, which the drain loop forwards to the WebSocket. The engine calls `emit()` (`event_sink.py`); when no callback is bound — e.g. the CLI front-end — `emit()` prints the event as a JSON line, preserving the original behaviour. The legacy `sys.stdout` router is retained only as a defensive catch-all for stray prints.

---

## Frontend file map

```
webview/src/
├── types.ts                  ← ALL shared TypeScript types (ServerMessage, ChatMessage, …)
├── hooks/
│   └── useWebSocket.ts       ← postMessage bridge; send(), connect(), createSession(), …
├── App.tsx                   ← Root component: all React state, handleServerMessage()
└── components/
    ├── ChatMessage.tsx        ← Renders one ChatMessage (text/thinking/approval/editing/tools/error)
    ├── InlineDiffApproval.tsx ← Per-file diff accept/discard card inside an approval message
    ├── BatchReviewBar.tsx     ← Sticky bar showing accumulated file changes after a turn
    ├── FileDiff.tsx           ← Unified-diff renderer (syntax-highlighted patch)
    ├── LiveThinkingBlock.tsx  ← Collapsible "Thinking…" / "Working…" block
    ├── MarkdownContent.tsx    ← Markdown renderer (used inside ChatMessage)
    ├── VerificationLedger.tsx ← Collapsed evidence panel under an answer (see below)
    ├── ledgerUtils.ts         ← Pure ledger split/parse helpers (unit-tested)
    ├── TodoSidebar.tsx        ← Todo / plan items panel
    ├── PlanBar.tsx            ← Plan-mode progress bar shown above the input
    ├── ApprovalPrompt.tsx     ← Simple yes/no approval card (non-diff tools)
    ├── GlobalApprovalBar.tsx  ← Approval banner (allow / always / deny) — sensitive-tool and out-of-workspace path prompts
    ├── SessionsPanel.tsx      ← Session list (switch / rename / delete — the ＋ lives in the status bar);
    │                            each row shows a model-generated one-sentence description of the session
    │                            (a hand-picked rename wins over it), and selecting a row opens it and
    │                            closes the panel
    ├── ConnectForm.tsx        ← Connection form (local / SLURM)
    ├── ModeSwitcher.tsx       ← Standalone mode button + picker (agent/plan/ask, each with a
    │                            description); the active mode colours the chat — blue agent,
    │                            red plan, green ask (`data-mode` on `.app` → `--mode-accent`)
    ├── AgentSettings.tsx      ← Context memory, enforcement, thinking depth, streaming
    ├── ContextBar.tsx         ← Context-window usage indicator
    ├── ResumePlanPrompt.tsx   ← "Resume previous plan?" dialog
    ├── MentionAutocomplete.tsx ← "@" dropdown to attach MCP resources / files (see below)
    ├── mentionUtils.ts        ← Pure caret/token helpers for the "@" autocomplete (unit-tested)
    ├── SlashAutocomplete.tsx  ← "/" dropdown to run a skill as a slash command (see below)
    └── slashUtils.ts          ← Pure caret/token helpers for the "/" autocomplete (unit-tested)
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

`App.tsx`, inside `toolIcon()`:
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

`InlineDiffApproval.tsx`, inside the header `<div>`:
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
npm run build          # one-shot production build
```
