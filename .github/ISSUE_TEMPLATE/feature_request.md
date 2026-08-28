---
name: Feature request
about: Propose something spawnkit should do that it does not do yet
title: ""
labels: enhancement
assignees: ""
---

## The problem

<!--
Start with the failure, not the feature. What went wrong in a real run, and what did it cost —
wall clock, VRAM, a hang, a log tail you never got to read?
-->

## What you tried instead

<!-- Existing spawnkit APIs, your own workaround, another library. Why each fell short. -->

## Proposed behaviour

<!-- What the API would look like from the caller's side. A sketch of the call is ideal. -->

```python
# how you would want to write it
```

## Which tier does this belong in

- [ ] hygiene / seeding (tier 1) — stdlib + numpy only
- [ ] supervision (tier 2) — stdlib + numpy only, may use tier 1
- [ ] service (tier 3) — may need torch
- [ ] not sure

Imports run downward only, and tiers 1 and 2 must stay torch-free so `pip install spawnkit` remains a
two-second install. A proposal that needs torch below tier 3 will need a different shape.

## Scope check

spawnkit is deliberately the layer *below* the trainer: process hygiene, worker supervision, run
identity, and batched inference from one GPU process. It is not a scheduler, not an experiment
tracker, and not a training framework.

- [ ] This is useful to more than one training loop.
- [ ] This cannot reasonably live in the caller's code.

<!-- If either box stays unchecked, say why it should still be in scope. -->

## Anything else

<!-- Prior art, links, a benchmark you would expect to move. Note that any performance claim in the
     README has to come from a committed file under benchmarks/results/, so a proposal justified by
     speed will eventually need one. -->
