---
name: proxy-merge
description: Use when the gains of several optimisation proxies are to be put together — on your own as soon as two or more axes come back with accepted gains (sub-agents' copies, or one after the other in this session), or when asked: "merge the proxies", "what do they give together", "Merge" from the code map — to merge their kept optimisations with git on a branch of its own, resolve their conflicts (asking the user when it is a choice of method), and measure each kernel and the whole as a new proxy.
disable-model-invocation: false
---

Each proxy's optimisations are measured alone, against its own baseline, whether it ran
in a sub-agent's copy or in this session, beside or after the others.
Nothing says they add up: two can touch the same lines, one can take away what another
gave (both cut the same memory traffic, say). Only the merged code measured settles it.
`proxy_merge` does the merge with git and the measurement as a new proxy, and keeps the
account.

## What a proxy brings

A proxy brings its **kept** work: the patch from its baseline to its best accepted
tree, read from the snapshot store, and only the files under its own `optimize_paths`.
No branch is needed: it is not the tip of a branch (which can still hold an attempt the
ratchet rejected), and a sub-agent's copy need not stand any more. A proxy that started
from another's work merges from its own baseline, so only what it added is picked. A proxy with no accepted run beyond its baseline
brings nothing and is left out. Proxies need not share a metric or a suite: each is
measured on its own cases.

## When

Merging is your decision, not a request to wait for: as soon as two or more proxies
have accepted gains — parallel axes back from their sub-agents, or axes run one after
the other here — plan and start the merge yourself. Only what is the user's stays
theirs: a choice of method in a conflict, and a kernel that lost its gain.

## 1. Plan

`proxy_merge(op='plan', proxies=[...])`, or `proxies=['*']` for every proxy. It is
read-only. It answers with each proxy's gain alone and files, the files two of them
share and the `conflicts` git will stop on. Tell the user what it found as you start,
the conflicts especially, and every `warnings` entry: two proxies that ran at the same
time in the same tree, on a file both change, each snapshotted what the other had on
disk, so one's patch and gain alone may hold the other's work, and nothing in the merge
can tell them apart.

## 2. Merge, in your session

1. `proxy_merge(op='start', proxies=[...], confirm=True)` puts the workspace on a branch
   of its own, `mimir/merge-<stamp>`, registers `merge-<stamp>`, a suite of every
   source's cases (each with its own harness and requirements), and opens its session.
   It needs a tree with no change git tracks (the kept patches come from the snapshot
   store: ask the user whether to commit or stash what is there) and no run under way
   on these files. Tell the user their open files show the merged code until it is
   measured; the workspace then goes back to their branch by itself.
   A proxy whose work is already committed on the branch is taken out first (a revert),
   so the tree measured untouched is really without it; it is merged back like the
   others.
2. Follow each `next_step`: `proxy_merge(op='next', confirm=True)` and the
   `proxy_eval(op='run', ...)` it names. The tree is measured untouched first. While
   the merge holds the workspace, the other sessions' runs and resets are refused.
3. Then each proxy's kept patch is picked as a git commit, best alone first: a
   three-way merge whose base is that proxy's own baseline. A clean pick goes on.

### A conflict

`next` stops on it with each file's conflicting hunks (between `<<<<<<<` and `=======`
what is merged so far, after it the proxy's) and the axes each side worked on. Decide
what kind it is:

- **Trivial**: an import, neighbouring lines, a renamed variable, both sides keeping
  the same intent. Resolve it yourself in the files.
- **A choice of method**: two rewrites of the same loop, two layouts of the same data,
  two ways to tile one kernel — keeping one means losing the other. Or any doubt.
  Reason about it (what each side buys, whether they can be combined), then ask the
  user with `ask_user_question`: both options, what each costs, your recommendation.
  Resolve as they answer.

Then `proxy_merge(op='resolved', resolution=<how and why>, confirm=True)`. It refuses
while a conflict marker is left. If the user gives no answer,
`proxy_merge(op='pause', question=<the question>, confirm=True)` undoes the pending pick,
keeps the question and takes the workspace back to the user's branch. Once answered,
`proxy_merge(op='next', proxy_name='merge-<stamp>', confirm=True)` goes back to the
merge's branch and carries it on.

## 3. The measurement

Once every patch is merged, `next` names one run of the merged code on every source's
cases. The report gives, per kernel (each source's case):

- `alone`: its gain in its own proxy (best / baseline);
- `in_merge`: its gain here, against the tree untouched;
- `interaction`: their ratio, under 1 it did better merged, over 1 the merge took some
  of its gain (`flagged` beyond the noise margin);

and the **whole**: the total time of every case against the tree untouched.

Nothing is set aside on its own. A flagged kernel is a question for the user: reason
about why it lost its gain, then ask whether to keep it anyway, take that proxy back
out (`proxy_merge(op='drop', proxy_name='merge-<stamp>', proxies=[...], confirm=True)`,
which goes back to the merge's branch, makes a `git revert` of its commit and measures the
whole again), or rework that axis on the merged code.

## 4. Report

`proxy_merge(op='report', proxy_name='merge-<stamp>')` (the last `next` gives it too):
each kernel alone and in the merge, the whole, each
conflict and how it was resolved, what was dropped, the branch. Give the user those
figures, then the branch. Once they have merged it, suggest cleaning the source
proxies (`proxy_manage(op='clean', ...)`): their graph stays on the map. Never clean
them unasked.

What a merge leaves: its branch `mimir/merge-<stamp>`, and the proxy `merge-<stamp>` with
its suite and one registry entry per source (`merge-<stamp>--<proxy>`), all removed by
`proxy_manage(op='clean', name='merge-<stamp>')`. No other branch, no copy.

## Several combinations

One merge answers "what do these give together". When several combinations are worth
comparing — two proxies that are alternatives on the same kernel (a choice of method the
plan's conflicts foretell), or a pair the `warnings` flag as having run together — measure
them at once, each in a sub-agent's copy: one `plan` per combination, then
`spawn_agent(task=<its brief>, tools=<the tools the plan names>, background=True)` for
each, all in the same response. Compare their reports and say which you would keep.

The code map shows the merge under `merge-<stamp>`: every proxy's gain alone, the
conflicts as arcs (resolved or not), and the whole. When the user ticks proxies in the
map's picker, the same graph appears as a forecast, and its **Merge** button puts the
plan to you.
