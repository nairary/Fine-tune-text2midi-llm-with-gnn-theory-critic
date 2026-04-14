# Observer graph schema (teacher-compatible)

Observer graph keeps the same hierarchical node set as teacher graph:
`song -> bar -> onset -> {note, chord}` plus sequence/coverage edges.

## Contract between pipeline and model

- `x_cat` contains only categorical ids in teacher-compatible id spaces.
- `x_num` contains only dense numeric features.
- `x = torch.cat([x_cat.float(), x_num], dim=1)` is backward-compatible convenience only (not source of truth).
- Reverse edges are **not** used in observer schema. `build_observer_graph()` creates forward edges only.

## Node types and features

- **song**
  - categorical-id: `main_key_tonic_pc_id`, `main_key_scale_id`, `main_num_beats_id`, `main_beat_unit_id`
  - numeric: `main_bpm`, `end_beat`
  - id-space/source:
    - `main_key_tonic_pc_id`, `main_num_beats_id`, `main_beat_unit_id` -> `spec_global.json` allowed_values
    - `main_key_scale_id` -> `vocab_key_scale.json`
- **bar**
  - categorical-id: none
  - numeric: `bar_index`, `bar_start_beat`, `bar_end_beat`, `n_notes_in_bar`, `n_chords_in_bar`, `n_onsets_in_bar`
- **onset**
  - categorical-id: none
  - numeric: `beat`, `bar_index`, `pos_in_bar`, `n_notes_starting_here`, `n_chords_starting_here`
- **note**
  - categorical-id: `sd_id`, `octave_id`
  - numeric: `beat`, `duration`, `bar_index`, `pos_in_bar`
  - id-space/source:
    - `sd_id` -> `vocab_melody_sd.json`
    - `octave_id` -> `spec_global.json` octave range
- **chord**
  - categorical-id: `root_id`, `type_id`, `inversion_id`, `borrowed_kind_id`, `borrowed_mode_name_id`
  - numeric: `adds_vec(6)`, `omits_vec(2)`, `suspensions_vec(2)`, `alterations_vec(6)`, `borrowed_pcset_vec(12)`, `beat`, `duration`, `bar_index`, `pos_in_bar`
  - id-space/source:
    - `root_id`, `type_id`, `inversion_id` -> `spec_global.json` allowed_values
    - `borrowed_kind_id` -> `vocab_borrowed_kind.json`
    - `borrowed_mode_name_id` -> `vocab_borrowed_mode_name.json`
    - `borrowed_pcset_vec` -> `spec_chord_sets.json` (`borrowed_pcset`)

## Edge types (forward only)

- `("song", "contains_bar", "bar")`
- `("bar", "next_bar", "bar")`
- `("bar", "contains_onset", "onset")`
- `("onset", "next_onset", "onset")`
- `("onset", "starts_note", "note")`
- `("onset", "starts_chord", "chord")`
- `("note", "next_note", "note")`
- `("chord", "next_chord", "chord")`
- `("chord", "covers_note", "note")`
