# AGENTS.md

## Project context

This repository contains the existing Music Critic V1 pipeline and a new
Music Critic V2 implementation.

Read before changing V2 code:

- docs/music_critic_v2/IMPLEMENTATION_PLAN.md
- docs/music_critic_v2/STATUS.md
- docs/music_critic_v2/DATA_CONTRACT.md

## Mandatory rules

1. Preserve the existing V1 pipeline unless the task explicitly says otherwise.
2. Add V2 modules alongside V1 modules.
3. Implement only the requested phase. Do not start later phases.
4. Do not use theory labels as raw encoder inputs.
5. Missing labels must be represented by masks, never as negative labels.
6. Raw-only inference must remain possible from an unlabeled MIDI file.
7. Do not introduce gold harmony, phrase, cadence, or tonal-region nodes into
   the mandatory inference graph.
8. Do not download large datasets or commit dataset files.
9. Do not add production dependencies without explaining why.
10. Every change must include tests.
11. Run the specified test commands before declaring completion.
12. Update docs/music_critic_v2/STATUS.md after completing a phase.

## Engineering rules

- Use type hints for public functions.
- Use dataclasses or typed dictionaries for canonical data.
- Reject malformed input with informative errors.
- Preserve provenance and target availability masks.
- Avoid float equality for musical timing.
- Prefer deterministic tests and fixed random seeds.
- Do not silently infer labels. Mark inferred values with source and confidence.
- Keep changes scoped and reviewable.

## Required final report

At the end of each task report:

1. files changed;
2. behavior implemented;
3. tests executed and their results;
4. unresolved ambiguities;
5. compatibility impact on V1;
6. suggested next phase.

## Standard checks

Run the relevant subset of:

```bash
python -m pytest -q
python -m pytest tests/v2 -q
python -m compileall src
```
Do not claim tests passed unless they were actually executed.
