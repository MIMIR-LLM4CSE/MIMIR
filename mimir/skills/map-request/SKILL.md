---
name: map-request
description: Use when a message is a request from the code map ("Request from the code map") — a numerical method, an algorithm, external code to bring in, parallelism, data layout, precision, or an analysis and refactor of the architecture (cycles, rule breaks, coupling) to study or try on named functions, modules, a component or the whole code — and when following up on one (R1, R2…); together with proxy-optimize for a request to try.
disable-model-invocation: false
---

A request from the code map is an R&D question put on a piece of the code: would this
method, this algorithm, this library do better here? The user follows it on the map,
pinned on its targets, until its outcome. Two things make it worth following: it is
recorded before anything else, and every variant it tries is measured from the same
code, so the comparison means something.

## Open it first

The message names the call: `proxy_request(op='open', token=..., mode=..., kind=...,
targets=[...], idea=..., variants=[...], source=..., constraints=...)`. Make it before
reading or editing anything. The answer gives the request its number (`R3`) and
records where the optimisation session stands: that tree is the request's **start**.
The same message opened twice is one request (the token).

## Study: read and propose, edit nothing

1. Read the targets — the functions named, their file, and what calls them and what they
   call (the map's neighbours).
2. Look at where their time goes: `proxy_eval(op='profile', level='function', focus=<file>)`
   reads it from the last profile when there is one.
3. Weigh the idea against the code as it is: what the method or algorithm changes, its
   cost and accuracy (order, stability, conditioning), what it needs from the rest of
   the code (interfaces, data layout, dependencies), the risks.
4. **Answer in the chat**: the analysis, the plan (the variants worth trying, in
   order), and what would decide between them. That answer is the study — the map's
   request links back to it, so write it whole there, not in the request.
5. `proxy_request(op='update', request_id='R3', status='proposed', summary=<one line>)`.

No file is changed in a study.

## Continued from the chat

The user may go on in the chat rather than with the map's buttons — "go ahead",
"implement the second variant". That is the same request: `status='trying'`, the
protocol below, every run's axis tagged `[R3.k]`. A run already launched without the
tag is attached with `proxy_request(op='update', request_id='R3', run_id=...)`; a run's
answer lists the open requests when its axis names none.

## Try: each variant from the start, one run each

1. `proxy_request(op='update', request_id='R3', status='trying')`. Without an
   optimisation session, set one up first as proxy-optimize says.
2. For each variant k, in the order given:
   - `proxy_eval(op='reset_to_request', request_id='R3', confirm=True)`: back to the
     start. Never build variant 2 on variant 1's edit — the comparison would be of a
     combination nobody asked for.
   - Implement the variant on the targets only.
   - `proxy_eval(op='run', axis='[R3.k] <the variant in a few words>', confirm=True)`. The
     axis prefix is what ties the run to the request on the map: always give it. A
     request with one idea and no variants uses `[R3]`.
   - The ratchet decides as usual: a variant better than the best is kept, so the
     winner among them ends up kept.
3. `proxy_eval(op='reset_to_best', confirm=True)`, then
   `proxy_request(op='update', request_id='R3', status='done', conclusion=...)`.

## Architecture: analyse, then refactor

A `refactor` request is about the code's shape — its dependency graph — on a module, a
component or the whole code (`{"all": true}`). The message carries what the map
already found there: module cycles, dependencies that break a declared rule, the
modules most depended on and most dependent.

- **Study**: for each finding, say why it is there (which import, which call, which
  shared state), and propose the target structure: what moves where, which interface
  breaks a cycle, which dependency is inverted, which module splits. Give the steps in
  an order where each one leaves the code working, and what each changes on the map
  (the cycles and breaks it removes) — as the answer in the chat. Nothing is edited.
- **Try**: one step per run, as for any variant: `reset_to_request` only between
  alternative designs, not between successive steps of one design — those build on
  each other. Each run is tagged `[R3.k]`. A refactor run is kept when it is
  **neutral**: the requirements pass and the metric is no worse than the best beyond
  the margin (the run's answer says `neutral`). It does not have to be faster.
  Behaviour must not change: when the project has tests, run them too.
- The conclusion names the findings removed and those left, from the map as it stands
  after the last run (the map's ⟳ shows it).

## Accuracy is half the answer

A numerical method changes the result, not only the time. The session's requirements
(an error norm, a residual, a conserved quantity) are pass/fail; report their values
for each variant beside the metric, in the conclusion, so a variant that is faster
but barely within tolerance reads as such. When the request names a convergence
order, measure it (`convergence=` on the session) rather than asserting it.

## External code

`source` names what to bring in: a package, a local path, a repository.
- A local path can be read freely.
- Installing a package or cloning a repository goes through the tools that do it, and
  their approval: say what will be installed, and where, before asking for it.
- Say the version used (package version, commit) in the answer: a result on a
  library is only reproducible with its version.
- Keep the integration behind the targets' interface where the request asks to keep
  the API.

## Closing

The outcome is said in the chat: the variant kept (or none), its gain on the metric
and its accuracy against the start, what each other variant gave and why it lost, what
it cost to integrate, and what to try next. `conclusion` is its one line. A request abandoned on the
user's word is `status='dropped'`, with why in `summary`.
