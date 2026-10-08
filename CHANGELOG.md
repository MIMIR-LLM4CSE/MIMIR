# Changelog

One version covers all of MIMIR: the Python package (`pyproject.toml`) and the
VS Code extension (`mimir/vscode-extension/package.json`). The extension talks to
the server of the same checkout, so the two always move together.

| Part of the version | Goes up when… | Example |
|---|---|---|
| Major (`X.0.0`) | an existing setup stops working without a change on the user's side: a setting, an environment variable, a CLI flag or a stored file is renamed or dropped | a `mimir.*` setting is renamed |
| Minor (`1.X.0`) | something new appears, and existing setups keep working | a new backend, tool server, or panel |
| Patch (`1.0.X`) | a fix, with nothing new to learn | a card that did not render |

How to release is in [CONTRIBUTING.md](CONTRIBUTING.md#releasing). The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.3.0] — 2026-10-07

### Added
- MIMIR can explain and write its own extensions. Asking how to add a skill, a tool
  server, a policy, a post-tool hook, a nudge or a base prompt used to be answered from
  whatever the model remembered of MIMIR — a capability flag that no longer exists, a
  path from an older layout, a plugin signature that was never ours. All of it plausible,
  none of it checkable, and the file written from it fails silently: an extension that
  does not load says nothing at the moment it is written. The new `/mimir-api` skill
  answers from the build that is running instead. Its `mimir_api` tool reads the
  installed package at call time: the capability vocabulary with what each flag drives,
  the drop-in path *this* workspace resolves (env overrides included, reported with the
  value they are set to), the shipped template for the type, and what the workspace
  already has — which skills and packs are present, which server namespaces are taken.
  Nothing it answers is a prose copy, which is the point: the flag table in the docs had
  already drifted three flags behind the vocabulary it claimed to list.
- A skill can now be loaded in the middle of the work, by the model, when what it has
  read tells it which method the task needs. Before, a skill could only enter the prompt
  once — before the first step, chosen either by `/<name>` or by a classifier model call
  that read the query and the last few turns. That decision is made at the one moment
  nothing is known yet: at step 14, having just found that the slow request is a proxy
  problem, the agent could not go and get `proxy-optimize`. Now the system prompt carries
  the *index* — each skill's name and the one line saying when it applies — and
  `load_skill(<name>)` reads the body on demand. The split is what makes it affordable:
  the eight shipped skills are 22 KB together and the largest is 13 KB, so carrying every
  body would have spent thousands of tokens per query to apply, at most, one of them.
  At most three per task, never the same one twice.
- `disable-model-invocation` in a `SKILL.md`'s front-matter now does what it says. Eight
  shipped skills declared it and the parser dropped it, so `prepare-pr` — which declares
  `true` — was reachable by the classifier against its own file. It now keeps a skill for
  the user's `/<name>` and refuses the model's own pull.

### Changed
- A skill the model loads stays where it lands, as the tool result, pinned against
  eviction — not folded into the system message. Rewriting the system message mid-query
  voids the prompt prefix for every remaining step, and a pull happens at step 14 as
  readily as at step 1. An explicit `/<name>` is still folded in: the user decided before
  the first step, so the prefix is paid for once and binds the whole query.
- Switching a skill off in the toggle panel now hides it from `/<name>` **and** from what
  the model can load. A skill its own file reserves for the user reads as `(/name only)`
  on its row.

### Removed
- The implicit skill classifier, and with it one full model round-trip per query. The
  model is given the index and chooses for itself, at the step where the choice can
  actually be made. `/<name>` is unchanged.

### Fixed
- A background run that ends while no window is open now gets its turn there and
  then, instead of waiting for someone to come back and ask. Recording what the
  run reported already worked; starting the turn it asks for needed a panel,
  because the routing lived in the socket. So a job that finished three minutes
  after the last answer, with the window shut, was written down and never handed
  to the conversation waiting on it — and the run sat untouched until morning.
  The server now takes the wake in itself: it appends it to the conversation,
  queues the turn on that conversation's own agent, and writes the answer back,
  with no connection involved at any point. A run detached under `auto` or
  `auto_all` can therefore carry its work through the night, which is what
  detaching it was for. The hourly check-ins behave the same way.
- A wake is no longer lost when nothing is there to read it. A finished run was
  marked as reported the moment its event was queued, so a wake that reached an
  empty process was filed as delivered: every later scan skipped it, and
  reconnecting said nothing about a job that had finished hours earlier. The
  marker now records delivery — it is written where the wake enters a turn — so
  a run nothing took in is announced again on the next scan, and reopening the
  panel catches up on what happened while it was shut.
- A detached server no longer stops itself while it owes a conversation a turn.
  A finished run whose wake nothing has taken in counts as work in progress, the
  same as one still going.
- Detaching now behaves like being there. A run finishing with the window shut got
  its turn, and the chain stopped at that one link: a connection does more when an
  answer lands than write it down — it puts the steering the turn never read to a new
  turn, starts a turn for every run that finished while it was busy, and delivers the
  bulletin it was holding. None of that happened with nobody attached, so "when the
  training finishes, carry on with the next step" answered the training and stopped.
  The server now carries on the same way, so a chain of steps runs through the night.
  What still differs is a choice, not a gap: under `manual` or `auto` a sensitive tool
  parks its card and waits for your return, which is what the autonomy level asked at
  detach time is for.
- A window that closes at the wrong moment no longer leaves the server deaf for the
  rest of its life. Whether to act on a finished run was decided by asking if any
  socket was subscribed — which says a socket exists, not that it will do anything
  with what it is sent. A view that ended without unsubscribing (a window shut during
  the replay of a long detached run raised mid-handshake, which skipped the cleanup)
  therefore stood in for a reader that was gone, and with it went every wake, every
  check-in and every turn written back, silently, until the server was restarted. The
  cleanup now runs on every way out, and the decision no longer guesses: each such
  event is offered, an attached panel has a moment's first refusal, and the server
  acts on whatever nobody claimed.
- An agent is no longer released at the one moment it is needed. A watcher stops
  holding its job the instant it reports it finished, which left the agent eligible
  for the ten-minute idle sweep in the gap before the wake reached it — and a
  conversation waiting on an overnight run has looked idle for hours by then, since
  idleness is counted from the last time the agent was asked for, not from anything
  the run is doing. Being owed a wake, holding unread output, or holding an event
  still to be claimed now count as work in progress like the rest. An agent rebuilt
  for any reason also resumes the context the last one had carried, which previously
  only happened when a panel reopened the conversation.
- A watcher whose status op stops answering now gives up instead of waiting for ever.
  Nothing beneath it had a deadline, so a wedged tool server or a scheduler that hung
  froze the watcher in place: the job was never reported, the agent was never released
  and never woken, and the check-ins kept repeating a status frozen at the moment it
  stopped — the quietest way to lose a run, since a watcher that crashes at least says
  so. A probe now has 60 seconds, a slow one counts as unreadable like any other
  unusable answer, and a run whose status cannot be read ends as `unknown` with the
  reason after a few of those. The summary call has the same deadline, where it matters
  more: the run is already over, and better a wake carrying nothing than no wake.

## [1.2.0] — 2026-10-06

### Added
- A run keeps working when the window closes. The control is at the right-hand
  end of the toolbar that carries the mode and the settings; pressing it leaves
  the server running without this window, and reopening the workspace comes
  back to it. The autonomy it runs under while unattended is whatever the
  approval switcher above the send button already says, so there is one place
  to answer that question rather than two that can disagree — and the button's
  tooltip spells out the consequence, because at `manual` a detached run parks
  at its first sensitive call and does almost nothing overnight. Detaching does
  not disconnect: the socket stays open, the turn goes on in front of you, and
  pressing the control again gives the server back to the window.
- Coming back brings the conversation with it. The history is replayed from the
  session's journal — tool rows, diffs, and what the agent said between its
  tools — the cards a turn is parked on are put back, the level it was left
  running under is restored, and a turn still in flight carries on with a stop
  button that stops something. Reasoning is the one thing a replay does not
  keep: the deltas are hundreds per turn and are deliberately not recorded.
- Background runs that finish while nobody is attached wake their conversation
  without being asked. A run survives anything — its own process session, a
  trap that records the exit code — but the watcher that promised to report it
  does not, and re-making that promise previously needed a turn in which
  somebody asked where the job had got to.
- Disconnecting while conversations are still working asks first, naming them:
  keep them going, disconnect anyway, or neither. Dismissing the question is
  "neither", so pressing Escape cannot end a two-hour build.
- `MIMIR: Stop Detached Server` ends a server this window no longer owns. It
  signals rather than asks over the socket, because the window that wants it
  stopped may not be connected to it, and it sends `SIGTERM` — the path that
  closes each agent's MCP servers, where a `SIGKILL` would leave them behind.
- A detached server stops itself once nothing needs it: no client attached,
  every conversation having delivered a final answer, and no background run
  still going. "Not busy" is not "has finished", so the criterion is the
  answer rather than the silence. `MIMIR_SERVER_IDLE_TTL` sets the wait.
- `/diag` reports what the event chain has done — whether the pump is running
  and how much it has moved, the journal's position, each subscription's
  watermark and what it filtered, and every worker's queue. A chat that has
  gone quiet has several causes that look identical from a chat window, and
  this is what separates them.
- One server per workspace, enforced with a lock taken before anything binds. A
  second would share the sessions directory, so both would write the same
  journal and derive their sequence numbers from it — and a client attached to
  one would see nothing of the turn running in the other. A window that would
  have started one attaches to the server that already serves the workspace.
- A card nobody can answer is set aside instead of waited on for ever. The
  criterion is whether a client is *attached*, not how long the wait has been,
  so a card with somebody there still waits indefinitely — and
  `MIMIR_DETACH_GRACE` keeps a window reload from parking a question the user
  is about to answer. Answering it later resumes the turn rather than
  restarting it.

### Fixed
- Deleting a conversation that has nothing running takes one click. The
  deletion asked about work in progress based on a reading of the session's job
  directory that counted any file in it as a run still going.
- A refused deletion says so. The warning went out on the transcript's
  transient channel, which the panel drops as tool chatter, so the first click
  did nothing visible and the second deleted — the warning the refusal exists
  to give, not given.

## [1.1.2] — 2026-10-02

### Fixed
- The shared-release mode was dead for every user who had not set
  `mimir.releaseHome` by hand — which is all of them. The setting declares `""`
  as its default, so VS Code returns `""` (not *unset*) for it, and the
  resolution used `??`, which keeps an empty string as a value: the deployment's
  `site.json` was never consulted, `releaseRoot()` was always empty, and the
  interpreter fell through to `python3` from the extension host's PATH — on a
  Remote-SSH login shell, whatever python the user's profile puts there. The
  same dead root also disabled the extension's self-update, so no release ever
  reached an installed extension and reloading the window changed nothing.
  Empty and unset are now the same thing — both fall back to `site.json` — and
  an explicit setting still wins. Disabling the mode remains possible by
  pointing the setting at a path without a `current` release.
- The setting precedence lives in `resolveReleaseRoot` (release.ts), a plain
  function with regression tests — the bug sat in vscode glue that no test
  exercised, and the shipped bundle validated only against a mock whose
  `get()` returned *undefined* for an untouched setting, where the real one
  returns the declared default.

## [1.1.1] — 2026-10-02

### Fixed
- The server is started without `PYTHONHOME` and `PYTHONPATH`, in the extension
  and the CLI wrapper alike. Both travel with the user's login shell — module
  systems and setup snippets export them — and each bends the interpreter onto
  a python it was never meant to run: `PYTHONHOME` overrides the venv's own
  home, so the shared interpreter loads another installation's stdlib and
  site-packages (or refuses to start), and `PYTHONPATH` puts that
  installation's packages ahead of the venv's. On a cluster login the result
  reads as "MIMIR launched my python instead of its own", although the right
  binary was executed. Both are now dropped for the processes MIMIR starts and
  nothing else: the user's shell keeps everything it had, and commands the
  agent runs were already executing under a rebuilt minimal environment of
  their own, so nothing they do loses the variables either.
- The server says which interpreter it runs on at startup —
  `Python: <executable> (prefix <prefix>)`, on the output channel the connect
  already logs to. The binary that was *launched* was always visible there;
  these two variables made the process say otherwise at runtime, and the new
  line tells them apart from inside the process, where it cannot be mistaken.

## [1.1.0] — 2026-10-02

### Added
- A background run now reports in while it runs: 30 s, 2 min, 10 min and 30 min
  after the first one of a conversation was launched, then once an hour for as long
  as it lasts. MIMIR answers a check-in in one line when there is nothing to say, and
  says what it is doing about it when there is — so a two-hour build that went wrong
  in its third minute is found out in its fourth rather than its hundred-and-twentieth,
  and an overnight run is still answered for at hour six. The gaps widen because the
  cost of having been wrong for that long widens too; the hourly floor is there because
  a schedule that simply ran out would go quiet exactly where the stakes are highest.
  One schedule per conversation, not per job, so jobs launched together report
  together; the bulletins cost no polling of their own, being built from what the
  watcher already saw.
  - It never interrupts. While the conversation is busy or waiting on a card, the
    bulletin is held and the next one replaces it, so a long turn ends with one
    current status line instead of a backlog of stale ones — and the answer the user
    asked for is never derailed into a report about a job. A bulletin whose runs have
    since finished is dropped: their own wakes say more.
- Asking where a run is at puts a watcher back on it. Watchers live on the agent that
  made them and do not survive it being restarted — which reloading the editor window
  is enough to cause — while the run itself carries on, indifferent. The status ops
  now answer with the tracking handle as well as the state, so one question restores
  the tracking of every run still going; opening a conversation whose runs nothing is
  watching says so in a line, and leaves the asking to you.
- Conversations run at the same time, one agent each. Leaving a conversation now
  leaves its turn running instead of cancelling it (or deferring it when it was
  parked on a question), and its output goes to its own transcript, so coming back
  shows the whole turn rather than only the answer that ended it. A card says which
  conversation is asking, and an answer carries that back. `MIMIR_MAX_LIVE_SESSIONS`
  (3) caps how many agents are alive — the constraint is processes, some nineteen MCP
  servers each — and past the cap a turn waits for a slot and is told its place in
  line. An agent is built on its conversation's first query, not when the session is
  opened, and released with its servers once idle for `MIMIR_SESSION_IDLE_TTL` (600 s)
  — never one that is busy, parked on a card, watching a background job, or on screen.
- Each conversation's server state is its own. Background-job and Slurm-job
  directories are filed per session under the state dir, so `bash_job(op='list')`
  shows this conversation's jobs rather than every conversation's, and nothing is
  written to `~/.cache` any more.
- Slurm jobs can carry a `--comment`. `proxy_slurm` (ops `run`, `suite`, `eval`),
  `sbatch_submit` and `salloc_submit` take `comment`: free text Slurm stores with
  the job and `sacct -j <id> -o Comment` reads back, so a queue full of MIMIR jobs
  says what each one was for. One suite comment labels every job of the suite, and
  a split build/run eval puts the same text on both jobs. Control characters are
  refused — a newline in a value that lands in a `#SBATCH` line would append
  directives of its own.
- `mimir.maxModelLen` sets the context window (tokens) for a vLLM or Ray Serve
  endpoint that does not report `max_model_len`, so the client no longer stays on
  its static 200K default. `0` (the default) leaves the reported window in charge.
  It forwards to `MIMIR_VLLM_MAX_MODEL_LEN` and `MIMIR_RAY_MAX_MODEL_LEN`.
- MIMIR can cancel a Slurm job. `slurm_cancel` takes one job ID of yours, pending
  or running, and asks your approval first. `scancel` stays refused in the shell,
  where `scancel -u` would cancel far more than one card shows.
- `mimir.vllmTokenize` (`MIMIR_VLLM_TOKENIZE=0`) stops every call to `/tokenize`,
  sub-agents included. A router that does not serve it, or answers it slowly, then
  costs nothing: a timeout used to be retried on every count, up to 5 s each.
- A GitHub file is read in pages. `github_get_file` takes a line range, returns at
  most 400 lines a call, and says where to resume — so the first look at a long
  source file costs a quarter of what the whole file did, and a file too large to
  inline is read through its raw URL instead of being refused outright.
- A fetch can now ask for part of a page instead of all of it. `contains=` returns
  the regions that answer it — a word matching a heading gives you that whole
  section, otherwise you get windows around the occurrences, each saying where it
  sits. `offset=` resumes a long document where the last reply stopped. Headings
  survive extraction as Markdown, so a page reads as named regions rather than one
  wall of prose. Measured across twelve sites: a 20 500-token article answers a
  targeted question in 127 tokens, a 33 700-token spec in 283.
- One install now runs on every node of a mixed cluster. `install.sh` fetches a
  portable Python 3.10 through uv, which needs only glibc 2.17 and brings its own
  OpenSSL and libcrypt. A Python compiled on a RHEL 9 node stopped on every RHEL 8
  node with `libcrypt.so.2: cannot open shared object file`. The venv is now named
  after the platform (`.venv-linux-x86_64`). A launcher, `~/.mimir/bin/python`,
  starts the venv built for the machine it runs on, so a shared home directory can
  serve several kinds of machines. `PYTHON=...` still selects your own interpreter.
  An existing install keeps working until you run `install.sh` again.

### Changed
- A finished background job's record now reads as one field per line instead of a
  single line of JSON: the key and the state the wake already named are not repeated,
  and each value is clipped on its own, so an 800-character command built out of
  absolute paths no longer fills the whole message.
- The context window the server reports now wins over the `MIMIR_VLLM_MAX_MODEL_LEN`
  / `MIMIR_RAY_MAX_MODEL_LEN` override. Those variables are the fallback for an
  endpoint that reports no `max_model_len`, not a value that supersedes one that
  does — so pinning stock `vllm` (e.g. 32768) is no longer overridden by a stale
  escape-hatch value.

### Fixed
- A finished background job no longer wakes the wrong conversation, or none at all.
  Where the job's own conversation was not the one on screen, the wake was submitted
  to the agent of whichever conversation *was* — running the turn behind that one's
  work, or, where it had no agent yet, onto a queue no loop reads, which lost the run
  silently. It is now resolved through the pool by the session that launched the job,
  the way the steer path already was.
- A wake survives the ways its delivery can fail. The flush took wakes off the pending
  map before it could know whether the turn would start, so a session deleted under
  it, a store that would not write, or an agent released out from under it each turned
  a finished run into one reported to nobody. All three now put the wake back.
- Reconnecting no longer throws away a job that finished while nothing was connected.
  The purge of a previous connection's debris emptied every idle agent's queue, and a
  completion wake sits on exactly such a queue — idle being what a conversation
  waiting for a build looks like. Jobs events are kept and put back, the rest is
  counted in the log. For the same reason the drain loop now handles one before it
  sends it: the event has already left the queue, and a socket dying on the send took
  the wake with it.
- A background watcher that dies on an exception says so, instead of surfacing hours
  later as "Task exception was never retrieved" with no sign of the run it was
  holding. A watcher's start and the wake it emits are logged too — the mechanism was
  silent end to end, and the only way to audit it was to notice a job with an exit
  code on disk and no `job_wake` in the transcript.
- A finished job's summary is parsed the way every other tool result is, so an
  annotation appended after its JSON no longer empties the wake into "it recorded no
  result of its own".
- Reopening a conversation no longer costs it history. The pre-query budget check ran
  before the agent existed, so it read the context mode off a conversation that had
  none and sized the window at compact's 32k: a resumed session showed the bar
  over-full and its first query compacted and front-trimmed a history that fit the real
  window perfectly well. The check now runs once the agent is up, and the mode is saved
  with the session so a resume has its own window to measure against before then.
- The context bar is exact as soon as the agent is up after a restart, rather than only
  once the first answer lands. The server-measured prompt overhead and the history's
  chars-per-token are kept in `<state>/token_calibration.json` and read back on the next
  run — the overhead under a fingerprint of the whole fixed part (model, mode, system
  prompt, advertised tools), so enabling a server or switching mode is a cache miss
  rather than a stale figure. The bar used to come back on a default heuristic and a
  ceiling estimate, then shift on its own once the first reply re-measured it — and with
  no `/tokenize` endpoint there is no other way back to an exact number.
- The context bar no longer claims an overflow it cannot have measured. Before a
  conversation has an agent, the system prompt and tools — tens of thousands of tokens —
  are not in the figure at all, so it now reads as a floor (`≥`, neutral colour) and
  says so, instead of drawing a red overflow from a number it knows to be incomplete.
- A command run by a tool can no longer read the server's stdin, which is that
  server's MCP protocol pipe. `ssh` without `-n` drains stdin, and `cat`/`head`
  consume it, so such a command ate the client's own JSON-RPC traffic: a request
  swallowed whole was never answered (a background-job watcher polling `bash_job`
  while an `ssh` ran, waiting for ever), and a partial steal desynchronised the
  framing until every later call on that session hung or failed with
  `ClosedResourceError` — while the server itself looked healthy. Every spawn now
  reads from `/dev/null`, so a command that reads stdin sees EOF.
- Releasing a conversation's agent now closes its MCP servers. The close ran in a
  task of its own, and `stdio_client` anchors its anyio cancel scope to the task that
  entered it, so the exit stack was left half-unwound — streams closed, subprocess
  alive: each release stranded some nineteen interpreters, and a turn still running on
  that agent failed its next tool call with `ClosedResourceError`.
- Stopping the server now closes every agent's MCP servers. Each is spawned in its own
  process group and survives this process dying, so a stop used to leave up to
  3 × ~19 orphaned interpreters behind — the ordinary path, since the VS Code
  extension kills and respawns the server on every connect.
- Deleting a conversation whose turn is still running now says so instead of cutting
  it short silently; asking a second time goes through. A turn parked on a card counts
  as work waiting to continue, not as work that has stopped.
- A tool call that fails on a dead MCP transport is no longer told to retry. Nothing
  reconnects an MCP server, so every later call to any of its tools fails the same
  way; the error now names the server that died and the other tools it took down, and
  says to report what is unfinished rather than spend the turn retrying.
- A slot freed by a turn ending is admitted at once rather than at the next sweep, so a
  conversation told it was next in line no longer waits out up to thirty seconds more.
- A Slurm job submitted through `proxy_slurm` now wakes MIMIR when it finishes.
  `sbatch` returns while the job is still queued, so the answer never carried the
  run's result — but only `op='eval'` with `background=True` handed over a job
  handle, and every other submission was left with nothing watching it: the model
  was told to monitor the job by hand, ended its turn because there was nothing
  else to do, and was never resumed when the job landed. `op='run'` and `op='eval'`
  now always return one. (`op='suite'` submits many jobs at once and still needs
  `proxy_get(op='report', ...)` afterwards.) `proxy_runs(op='status', run_id=...)`
  is the new single-run state op the watcher polls.
- Two proxy runs started in the same second no longer share one run directory. The
  directory tag is second-resolution and was reused, so the two runs overwrote each
  other's log, metrics and state, and answered to one job key — which left the
  second run unwatched and its finish unreported.
- The run timer of a proxy evaluation starts at zero on every run. A second run of
  the same proxy used to continue from the first run's card, which went on counting
  from the earlier start.
- The context bar splits a prompt the way the server does. Without `/tokenize`,
  the history was counted at a fixed 4 characters per token, and the error landed in
  the fixed overhead: on one DeepSeek session the bar showed ~48k of system prompt
  and tools where the server charged 36k. The history's ratio is now measured from
  the `prompt_tokens` the server reports, with no extra request.
- Every tool parameter was sent twice on every call: once in the schema, and again
  in the description, which still carried the docstring's `Args:` block. It now
  leaves the description once the schema carries it word for word — a fifth of the
  tools schema, 31k tokens down to 25k with every server on.
- A single step can no longer overrun the context window. Tool results are now
  bounded before they enter the history — per result, and per step together — so
  several calls returning at once cannot do what four fetches did to one session:
  take a 200k window to 215k with nothing given the chance to object. Small results
  are never cut to pay for large ones. This covers every tool, including servers
  MIMIR does not ship.
- A failed web request no longer costs more than a successful one. The error
  branches returned up to 512 KB of the error body raw, with no extraction and no
  ceiling: one paper host answering 403 cost 131 164 tokens for a request that
  returned nothing.
- A page whose text cannot be extracted — a script shell, or a body that never
  arrived behind 700 KB of inline CSS — now comes back as its own metadata and
  embedded data rather than as markup. One thesis record page went from 34 479
  tokens of CSS to about 1 400 tokens carrying its title, jury, keywords and full
  abstract.
- Guidance written for the model actually reaches it now. `hint` is a reserved key
  that is stripped from success payloads, so several tools' "fetch a more specific
  URL" and "call again with confirm=True" lines had never once been delivered.
- Behind an OpenAI-compatible router that does not serve vLLM's `/tokenize`,
  each turn no longer waits tens of seconds before the model is asked. The
  refusal is remembered per endpoint instead of retried for every message.
- A build launched just as another background job finished keeps its row and
  its progress bar. The finished job was handed to the running turn, but the
  panel was told a new turn had begun and cleared that turn's rows.

## [1.0.0] — 2026-09-18

First versioned release. Everything since the initial public release (0.1.0) is
in it. The main points:

### Agent
- Three modes: **agent**, **plan** (read-only, ends in a plan you approve), and
  **ask** (read-only questions). The mode applies from the next step, even mid-turn.
- Three approval modes: **manual**, **auto**, **all**. One approval card per call.
  A refusal is read as an instruction, not an error to retry.
- A verification ledger under each answer: what was checked, what was not.
- Sub-agents, which run in the user's approval mode.
- No step ceiling on a run.
- Builds go through the tool that owns them, and only as far as the change needs.

### Backends
- vLLM, Ray Serve, Ollama and the Anthropic API.
- The model list is read from the endpoint. Nothing to maintain.

### Long runs
- Any blocking tool can be moved to the background. A background job wakes the
  session that launched it when it ends.
- A run reports its phase, and a progress bar when the tool counts its own progress.
- A turn outlives its connection. Its work comes back when the panel reconnects.

### VS Code extension
- Connect form: backend, address, model. **Remember this address** reconnects the
  next window on its own. A connection error is explained in plain words.
- File names in tool rows open the file on the lines that were read or edited.
- A card in the corner for a run whose row scrolled away.
- `↑` recalls a sent message.
- Default Claude models updated to the current family.

### Install
- `install.sh` clones, installs and records the Python interpreter the extension uses.
- Each VS Code window starts its own server on a free port.

### Removed
- The fine-tuning server. The proxy server already covers its runs.
- The "keep going?" card, with the step ceiling it served.

## [0.1.0] — 2026-07-31

Initial public release.

[Unreleased]: https://github.com/MIMIR-LLM4CSE/MIMIR/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/MIMIR-LLM4CSE/MIMIR/releases/tag/v1.0.0
[0.1.0]: https://github.com/MIMIR-LLM4CSE/MIMIR/commits/main
