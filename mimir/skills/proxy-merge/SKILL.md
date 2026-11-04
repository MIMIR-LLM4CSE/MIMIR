---
name: proxy-merge
description: Use when several optimisation proxies ran in parallel (one axis per sub-agent, each on its own branch) and their gains are to be put together — "merge the proxies", "what do they give together", "Merge" from the code map's proxy picker — to merge their kept optimisations in a copy of its own, measure the whole against the gains alone, and see which ones help or hurt each other.
disable-model-invocation: false
---

Optimisations made in parallel are each measured alone, against their own baseline.
Nothing says they add up: two can touch the same lines, one can take away what another
gave (both cut the same memory traffic, say), and a gain alone is no promise once the
code around it has changed. Only the merged code measured settles it. `proxy_merge`
does that measurement and keeps its account, so you never have to assemble the
combination by hand.

## What a proxy brings

A proxy brings its **kept** work: the patch from its baseline to its best accepted
tree, read from the snapshot store. That is not the tip of its branch, which can still
hold an attempt the ratchet rejected. Its copy need not stand any more. A proxy with no
accepted run beyond its baseline brings nothing and is left out.

## 1. Plan, in your own session

`proxy_merge(op='plan', proxies=[...])`, or `proxies=['*']` for every proxy. It is
read-only. It answers with:

- each proxy's gain alone and the files it touches;
- for each two proxies, the files both change, and the `conflicts` between their two
  patches (those two cannot both be kept);
- the gain **expected** together, which assumes the gains multiply;
- the order a one-at-a-time measurement will follow (best alone first);
- the `brief` to hand to a sub-agent.

The plan refuses proxies whose metric, goal or benchmark suite differ: one measurement
cannot judge them all. Tell the user what it found before going on, the conflicts
especially.

## 2. Measure, in a sub-agent's copy

`spawn_agent(task=<brief>, tools=['proxy_merge', 'proxy_eval', 'proxy_eval_status'],
background=True)`.

The merge is measured in a copy of its own, on a branch of its own, so the main
workspace does not change until the user merges that branch. `start` refuses to run
anywhere else.

The sub-agent does three things:

1. `proxy_merge(op='start', proxies=[...], confirm=True)` registers `merge-<stamp>` over
   its copy and opens its session.
2. It alternates `proxy_merge(op='next', confirm=True)` with the
   `proxy_eval(op='run', axis=..., confirm=True)` that `next` names, until `next` says
   the merge is done. It edits no file itself.
3. It answers with `proxy_merge(op='report')`.

`next` drives this sequence:

1. **The copy untouched.** It is measured first, and every result is judged against it.
2. **All at once.** Every proxy is applied, one file at a time with a three-way merge. A
   proxy in conflict with what is already applied writes nothing and is set aside. If
   the whole holds the expected gain within the noise floor, the merge is done.
3. **One at a time**, when the whole falls short of the expected gain. The copy goes
   back to untouched and the proxies are applied in the plan's order, best alone first.
   Each proxy is measured on top of what was kept before it:
   - it is kept when it still improves the whole;
   - it is set aside as `interference` when it does not;
   - it is set aside as `conflict` when its patch no longer applies.

At the end, the copy holds the better of the whole and the one-at-a-time chain, and
that is what gets committed to the branch.

## 3. Report

The report gives:

- the gain expected and the gain measured (`final`);
- the proxies kept, and those set aside with why;
- for each proxy measured in the chain, its gain alone beside its gain in the merge
  (`interaction`: under 1 means it did better with the others, over 1 that the merge
  took some of its gain).

Give the user those figures, then the branch. Bring the branch over with merge or
cherry-pick when they want it. A proxy set aside for a conflict is an axis to redo on
top of the merged code, not a loss; say so.

The code map shows the merge in its strip, under the proxy `merge-<stamp>`:

- every proxy's gain alone on the left;
- their shared files and conflicts as arcs between them;
- the whole on the right, with the gain expected and the gain measured.

When the user ticks proxies in the map's picker, the same graph appears without a
measurement, as a forecast. Its **Merge** button puts the plan to you.
