# TODO

## 1. Timeline / Section Dataset Audit
- [ ] Use `data/HTCanon/HK_processed/original_songs_timeline.json` as the source of original-song section grouping.
- [ ] Validate that every `clip_song_id` in the timeline exists in `data/HTCanon/encoded_full/teacher_encoded.json`.
- [ ] Build an audit report per `ori_uid`:
  - ordered timeline segments
  - section labels
  - clip ids
  - start/end seconds
  - gap / overlap between neighboring segments
  - split membership
- [ ] Classify original songs into assembly buckets:
  - `single_section`
  - `safe_multisection`
  - `small_gap`
  - `large_gap`
  - `small_overlap`
  - `large_overlap`
  - `mixed_split`
- [ ] Decide conservative MVP filters:
  - at least 2 sections
  - single split only
  - skip large overlaps
  - either compact gaps or skip large gaps
- [ ] Save summary metrics: number of usable originals, section label distribution, transition distribution, duration distribution.

## 2. Section-Aware Song Assembly
- [ ] Build assembled encoded songs by grouping clips under the same `ori_uid`.
- [ ] Sort clips by `segment_start_seconds`.
- [ ] Start with compact concatenation:
  - append sections back-to-back
  - ignore real-time gaps in seconds
  - do not insert silence/rests yet
- [ ] Shift every event beat inside each clip into assembled-song coordinates.
- [ ] Preserve traceability metadata:
  - `ori_uid`
  - source `clip_song_id`
  - source section label(s)
  - source start/end seconds
  - assembled start/end beats
  - source split
- [ ] Add `meta.section_spans` to assembled songs.
- [ ] Save assembled dataset separately from original clips.
- [ ] Keep original short clips as the baseline dataset.

## 3. Graph Schema: Add Section Hierarchy
- [ ] Add a new graph hierarchy level:
  - `song -> section -> musical events`
- [ ] Add `section` nodes for assembled songs using `meta.section_spans`.
- [ ] Add one dummy section for every old short clip without real section spans:
  - label: `clip` or `unknown`
  - span: whole song
- [ ] Add section node features:
  - section label id
  - order index
  - duration beats
  - normalized start/end position
  - optional source clip count
- [ ] Add section edges:
  - `song -> section`
  - `section -> song`
  - `section -> section_next`
  - `section -> chord`
  - `section -> note`
  - optionally `section -> onset/beat` if those nodes exist
- [ ] Keep existing song-level readout paths so the current model remains a stable baseline.
- [ ] Add tests that old short clips still build graphs correctly with one dummy section.

## 4. Section-Level Corruptions
- [ ] Add corruption targets at section and section-transition level.
- [ ] First MVP corruption:
  - `adjacent_section_swap`
  - example: `verse -> pre-chorus -> chorus` becomes `verse -> chorus -> pre-chorus`
- [ ] Add additional section corruptions after MVP:
  - `non_adjacent_section_swap`
  - `section_duplicate`
  - `section_drop`
  - `section_boundary_blur`
  - `section_entry_chord_substitution`
  - `section_exit_chord_substitution`
- [ ] Preserve corruption metadata:
  - corruption type
  - affected section ids
  - original labels
  - original order
  - new order
  - boundary beats
- [ ] Keep local chord/melody corruptions as a baseline.
- [ ] Avoid generating section-level corruptions for songs with only one section.

## 5. Training Strategy
- [ ] Use one graph schema for all data by always including section nodes.
- [ ] Stage 1: pretrain on the main short-clip dataset:
  - dummy section per clip
  - existing local corruptions
  - no dependence on real section labels
- [ ] Stage 2: fine-tune on assembled multi-section data:
  - real section nodes
  - section-aware corruptions
  - lower learning rate
- [ ] Use mixed fine-tune batches to avoid forgetting:
  - mostly assembled section data
  - some original short clips
  - some old local corruptions
- [ ] Compare:
  - short-only baseline
  - assembled-only fine-tune
  - mixed fine-tune

## 6. Evaluation / Ablations
- [ ] Track existing ranking metrics:
  - pair rank accuracy
  - clean-vs-corrupted score margin
- [ ] Add section-specific metrics:
  - section-swap detection success
  - transition corruption success
  - metrics grouped by transition label pair (`verse->chorus`, `pre-chorus->chorus`, etc.)
- [ ] Check regression on old short-clip evaluation.
- [ ] Ablate:
  - no section nodes
  - dummy section only
  - real section nodes without section corruptions
  - real section nodes with section corruptions
  - mixed vs section-only fine-tuning

## 7. Practical Order Of Work
- [ ] Step 1: implement timeline audit script and reports.
- [ ] Step 2: implement compact assembled-song builder with `meta.section_spans`.
- [ ] Step 3: add dummy/real `section` nodes to graph construction.
- [ ] Step 4: implement `adjacent_section_swap` corruption.
- [ ] Step 5: run a small smoke training/eval with mixed short + assembled data.
- [ ] Step 6: expand corruption families only after the MVP path works.
