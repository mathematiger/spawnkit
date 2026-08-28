---
name: Bug report
about: Something in spawnkit behaves differently from what it documents
title: ""
labels: bug
assignees: ""
---

## What happened

<!-- One or two sentences. What did you observe? -->

## What you expected

<!-- What the docstring or README led you to expect. -->

## Reproduction

<!--
The smallest script that shows it. A reproduction that runs on CPU, with no GPU and no training
loop, is worth far more than a stack trace: most of what spawnkit does is reproducible with a
handful of workers that sleep.
-->

```python
# minimal reproduction
```

## Which tier

- [ ] hygiene / seeding (tier 1)
- [ ] supervision — monitor, run, oom, processes, supply, lifecycle (tier 2)
- [ ] service — batched inference (tier 3, needs the `[torch]` extra)
- [ ] not sure

## Environment

| | |
|---|---|
| spawnkit version | output of `python -c "import spawnkit; print(spawnkit.__version__)"` |
| Python version | |
| OS | |
| Start method | spawn / fork / forkserver |
| Worker count | |
| torch version | if the service tier is involved |
| GPU | model and VRAM, if any |

## If a run hung or died

<!--
Fill this in for anything about worker death, shutdown or a stalled queue. These are the details
that make the difference between a guess and a diagnosis.
-->

- Exit codes of the workers, if you saw them:
- Did the run hang, exit, or get killed by the out-of-memory killer?
- Log tail around the failure (please paste as text, not a screenshot):

```
```

## Anything else

<!-- Workarounds you found, whether it is intermittent, how often it reproduces. -->
