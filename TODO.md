# TODO

## 1. Structural Analysis Cache via AnalysisGNN
- [ ] Add offline export from `song_theory` / encoded songs to `MusicXML` or another `partitura`-readable symbolic format.
- [ ] Run `analysisgnn` offline over the exported scores and cache structural predictions.
- [ ] Store compact per-song structural cache with:
  - cadence onsets / spans / types / confidence
  - phrase boundaries / spans / confidence
  - optional local harmony fields: `localkey`, `romanNumeral`, `bass`, `inversion`
- [ ] Version the cache by source-song hash, exporter version, and `analysisgnn` checkpoint version.
- [ ] Add fallback behavior for songs where export or structural analysis fails.

## 2. Cadence / Phrase-Aware Corruptions
- [ ] Add cadence-targeted corruptions that sabotage structurally important endings instead of random chord positions.
- [ ] Candidate cadence corruptions:
  - `cadential_dominant_break`
  - `final_tonic_substitution`
  - `cadential_inversion_conflict`
  - `leading_tone_resolution_break`
- [ ] Add phrase-aware corruptions that target phrase boundaries or phrase endings.
- [ ] Decide which phrase corruption family is musically stable enough to use in training:
  - break end-of-phrase closure
  - shift / blur boundary emphasis
  - create harmonic carry-over across boundary
- [ ] Use prediction confidence thresholds so weak `analysisgnn` phrase/cadence outputs do not pollute training data.

## 3. Onset-Level Structural Prediction
- [ ] Add explicit onset-level prediction tasks instead of using onset nodes only as intermediate graph structure.
- [ ] First priority onset tasks:
  - `cadence_flag`
  - `phrase_boundary`
- [ ] Optional later onset tasks:
  - `cadence_type`
  - local harmonic state at onset
- [ ] Keep note-level modeling in parallel so onset aggregation does not wash out single bad-note events.
- [ ] Add onset-level consistency / aggregation rules for harmony-sensitive predictions.

## 4. Dense Structural Supervision
- [ ] Extend supervision beyond global `graph_score` and local corruption labels.
- [ ] Treat cadence / phrase labels as train-time structural supervision, not as inference-time required inputs.
- [ ] Distinguish two supervision families explicitly:
  - local anomaly supervision: `note/chord/onset corrupted or not`
  - structural supervision: `cadence / phrase / harmonic role`
- [ ] Evaluate whether structural labels should be attached to `onset` nodes only or also summarized into `chord` nodes.

## 5. Observer Distillation Upgrade
- [ ] Keep observer inference MIDI-only, but enrich observer training with privileged structural labels from offline cache.
- [ ] Add observer auxiliary heads for:
  - `cadence_flag` on onset
  - `phrase_boundary` on onset
- [ ] Train observer with combined objective:
  - scalar teacher score regression
  - pairwise clean-vs-corrupted rank loss
  - auxiliary structural losses
- [ ] Verify that cadence / phrase auxiliary tasks improve ranking quality instead of only adding task noise.

## 6. HGT / Encoder Upgrade
- [ ] Add configurable encoder backend switch:
  - current `HeteroConv + SAGEConv`
  - `HGT`
- [ ] Benchmark `HGT` against current backbone before replacing the default.
- [ ] Test `HGT` first where heterogeneity matters most:
  - observer with onset structural heads
  - teacher with richer structural supervision
- [ ] Keep the current backbone as a stable baseline and fallback.

## 7. Evaluation / Ablations
- [ ] Add ablations to separate gains from:
  - better corruptions
  - structural labels
  - onset-level tasks
  - `HGT`
- [ ] Track metrics that reflect structural improvements, not only scalar score fit:
  - `pair_rank_acc`
  - global clean-vs-corrupted separation
  - cadence-targeted corruption success
  - phrase-targeted corruption success
- [ ] Compare observer performance with and without auxiliary cadence / phrase heads.

## 8. Practical Order of Work
- [ ] Step 1: build offline symbolic export and structural cache.
- [ ] Step 2: add cadence-aware corruptions.
- [ ] Step 3: add observer onset auxiliary heads for cadence.
- [ ] Step 4: add phrase supervision only after cadence pipeline is stable.
- [ ] Step 5: benchmark `HGT` as an optional encoder upgrade.
