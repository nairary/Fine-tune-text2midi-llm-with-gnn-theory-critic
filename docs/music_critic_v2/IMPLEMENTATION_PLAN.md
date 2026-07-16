# Codex Implementation Specification: Music Critic V2

**Repository:** `nairary/Fine-tune-text2midi-llm-with-gnn-theory-critic`  
**Primary goal:** replace the current HookTheory-specific, corruption-heavy teacher/observer pipeline with a reusable multi-dataset symbolic-music critic that:

1. consumes ordinary unlabeled MIDI or score-derived symbolic data at inference;
2. is pretrained with modern masked graph self-supervision inspired by GraphMAE2, Hi-GMAE, and UGMAE;
3. learns music-theory concepts from partially labeled datasets without requiring those labels at inference;
4. models local polyphonic relations with a heterogeneous graph encoder and long-range form with a hierarchical Transformer;
5. produces interpretable theory predictions and multiple quality dimensions rather than only one noisy scalar;
6. can later provide a preference/reward signal for Text2MIDI GRPO or inference-time selection;
7. can optionally approximate an audio-domain aesthetic model such as Meta Audiobox Aesthetics without rendering every MIDI during policy optimization.

This document is an implementation plan, architectural specification, migration guide, and acceptance checklist for a Codex coding agent. Implement it incrementally. Do not attempt to replace the entire existing pipeline in one change.

---

## 0. Instructions to the Codex agent

### 0.1 General working rules

1. **Preserve the current pipeline.** The existing `TeacherGNN`, `HookTheoryDataset`, theory-aware corruptions, observer pipeline, and existing command-line interfaces must continue to work until the new pipeline has independent tests and checkpoints.
2. **Create V2 modules alongside V1 modules.** Do not rewrite `src/dataloader/utils_graph.py` or `src/models/teacher_gnn.py` into incompatible forms during the first phases.
3. **Implement one phase at a time.** Each phase below has explicit deliverables and acceptance criteria. Stop after a phase, run tests, and report the result before continuing when operating interactively.
4. **Prefer deterministic, inspectable transformations.** Every dataset conversion must write manifests, statistics, validation reports, and provenance.
5. **Never use a theory annotation as an encoder input if it will not be available for ordinary MIDI inference.** Theory annotations are targets unless explicitly marked as optional observable metadata.
6. **Missing labels are not negative labels.** Every supervised target must have an availability mask.
7. **Keep exact timing as long as possible.** Normalize to rational quarter-note beat units; do not convert score timing to floating-point seconds early.
8. **Do not silently invent harmony, voices, phrases, or track roles.** Inferred and pseudo-labeled values must have a source and confidence field.
9. **Add unit tests before large preprocessing runs.** Use tiny synthetic pieces that cover meter changes, tempo changes, sustained notes, multiple tracks, empty tracks, missing metadata, pickups, and transposition.
10. **Use schema versions.** A cached graph must identify the canonical schema version, feature registry version, and graph-builder version.
11. **Keep raw score evidence separate from labels.** The model input must be reproducible from an unlabeled MIDI whenever the task claims MIDI inference support.
12. **Do not claim that masked reconstruction loss is an aesthetic score.** SSL representation learning and preference/quality learning are separate stages.

### 0.2 Current repository state that must be considered

The existing repository currently has the following design:

- `src/data/preprocess_hooktheory.py` parses raw HookTheory.
- `src/data/canonicalize_hooktheory.py` normalizes HookTheory theory values.
- `src/data/encode_teacher_features.py` converts fields to encoded IDs.
- `src/dataloader/hooktheory_dataset.py` returns three graph variants per item:
  - `graph_real`;
  - `graph_masked`;
  - `graph_corrupted`.
- `src/dataloader/utils_graph.py` builds a fixed heterograph with:
  - `song`;
  - `bar`;
  - `onset`;
  - `note`;
  - `chord`.
- Current note inputs contain theoretical `sd_id`.
- Current chord inputs contain `root_id`, `type_id`, `inversion_id`, `applied_id`, and borrowed-chord fields.
- `src/models/teacher_gnn.py` uses type-specific MLP encoders, `HeteroConv` with `SAGEConv`, global pooling, local note/chord/onset score heads, reconstruction heads, and a graph-score head.
- `src/training/teacher_losses.py` combines masked reconstruction, clean-vs-corrupted ranking, and local corruption losses.
- `src/observer/` distills the teacher score into a MIDI-derived observer graph.

The V2 design should eventually remove the need for teacher-to-observer scalar distillation, because the V2 encoder itself must accept raw MIDI-derived graphs. However, the observer pipeline must not be deleted until V2 is demonstrably functional.

---

# Part I. Research objective and scope

## 1. What the new system should learn

The new critic should learn three distinct categories of behavior.

### 1.1 Symbolic representation learning

From large unlabeled symbolic corpora, learn representations that encode:

- pitch and interval context;
- onset and duration context;
- meter and rhythmic organization;
- local vertical sonority;
- within-track continuation;
- cross-track interaction;
- instrumentation and track behavior;
- bar-level temporal development;
- long-range repetition, variation, and form.

This stage uses masked graph self-supervision. It does **not** require quality labels.

### 1.2 Music-theory analysis

From partially labeled corpora, learn to predict:

- local key;
- chord root and quality;
- inversion or bass relation;
- Roman numeral or scale-degree function;
- applied/secondary and borrowed harmony;
- chord-tone versus non-chord-tone status;
- cadence class;
- phrase and section boundaries;
- track roles;
- possibly voice identity and voice-leading relations.

These tasks are auxiliary supervision for the shared encoder and interpretable outputs at inference.

### 1.3 Preference and quality assessment

From pairwise human judgments, generated-MIDI comparisons, weak preference labels, or audio-aesthetic teacher scores, learn:

- harmony quality;
- melody-harmony compatibility;
- voice-leading quality;
- rhythm and meter quality;
- structural coherence;
- track interaction and arrangement quality;
- audio-aesthetic surrogate scores;
- overall pairwise preference or calibrated quality.

The first deployment target should be **pairwise ranking within a prompt group**, not a universal absolute MOS across all genres.

## 2. Non-goals for the first implementation

Do not attempt all of the following in the initial version:

- perfect automatic Roman-numeral analysis for every genre;
- automatic phrase segmentation with no labels;
- universal genre-independent aesthetic scoring;
- faithful implementation of every detail from GraphMAE2, Hi-GMAE, and UGMAE at once;
- exact orchestral voice inference from arbitrary MIDI;
- end-to-end GRPO before the critic has independent evaluation;
- replacement of every current corruption rule.

The implementation must first establish a clean data and model foundation.

---

# Part II. Core design principles

## 3. Input/target separation

All information must be classified into one of three groups.

### 3.1 Always-observable raw input

These are available in an ordinary MIDI or can be deterministically derived from it:

- MIDI pitch;
- pitch class;
- octave;
- note onset;
- note duration;
- velocity;
- MIDI channel and program where valid;
- percussion flag;
- track index;
- tempo changes;
- time-signature changes;
- bar index and beat position derived from the meter map;
- simultaneous or overlapping events;
- previous/next event relations;
- basic track statistics derived from notes.

These are safe model inputs.

### 3.2 Optional observable metadata

These may be available in MusicXML or curated MIDI, but often absent in generated MIDI:

- key signature;
- track name;
- instrument name;
- staff number;
- voice number;
- lyrics;
- articulation;
- dynamic markings;
- rehearsal marks;
- section labels from notation.

Rules:

- every optional field requires an availability mask;
- use an `unknown` categorical ID or zeroed continuous value plus availability bit;
- apply metadata dropout during training;
- report metrics with and without optional metadata;
- do not require these fields in the public inference API.

### 3.3 Theoretical or subjective targets

These are normally unavailable in raw MIDI and must not be required encoder inputs:

- scale degree;
- local tonic and mode;
- chord root/quality/inversion;
- Roman numeral;
- harmonic function;
- applied and borrowed harmony;
- cadence;
- phrase boundary;
- section function;
- non-chord-tone type;
- semantic track role;
- preference and MOS labels.

They are stored as target tensors with masks, provenance, and confidence.

## 4. No gold semantic graph structure at inference

A raw MIDI does not directly contain gold chord-event spans, phrase nodes, cadence nodes, or tonal-region boundaries. The V2 model must not rely on gold semantic graph structure.

Use one of these inference-safe designs:

1. **Candidate slots:** chord predictions at every beat or relevant onset; cadence/phrase predictions at every bar boundary.
2. **Direct heads:** predict chord labels from beat/onset embeddings and phrase/cadence labels from bar-boundary embeddings.
3. **Two-pass inference later:** first detect boundaries, then construct semantic nodes and re-encode.

For V2.1, use candidate slots and direct heads. Explicit gold-derived semantic nodes can exist in the canonical data for supervision and analysis, but the base encoder path must work without them.

## 5. Graph structure must be reproducible from raw evidence

Mandatory structural nodes:

- `song`;
- `track`;
- `bar`;
- `beat`;
- `onset`;
- `note`.

Optional semantic nodes:

- `harmony`;
- `tonal_region`;
- `phrase`;
- `section`.

The model must have a `raw_only=True` inference mode that excludes optional semantic nodes and still produces all deployable outputs.

---

# Part III. Canonical data schema

## 6. New module layout

Add the following package structure without deleting old modules:

```text
src/
  data_v2/
    __init__.py
    schema.py
    feature_registry.py
    target_registry.py
    timing.py
    validation.py
    serialization.py
    manifests.py
    windowing.py
    augmentation.py
    stats.py
    adapters/
      __init__.py
      base.py
      hooktheory.py
      pop909.py
      dilemmadata.py
      pdmx.py
      midi_generic.py
    cli/
      preprocess_dataset.py
      validate_canonical.py
      build_graph_cache.py
      inspect_piece.py

  graph_v2/
    __init__.py
    builder.py
    edge_builder.py
    candidate_slots.py
    hierarchy.py
    masking.py
    negative_sampling.py
    transforms.py

  dataloader_v2/
    __init__.py
    canonical_dataset.py
    multisource_dataset.py
    samplers.py
    collate.py
    task_router.py

  models_v2/
    __init__.py
    feature_encoder.py
    local_graph_encoder.py
    hierarchy_pooling.py
    global_transformer.py
    top_down_fusion.py
    ssl_decoder.py
    theory_heads.py
    quality_heads.py
    aesthetic_head.py
    music_critic_v2.py

  training_v2/
    __init__.py
    losses_ssl.py
    losses_theory.py
    losses_preference.py
    losses_aesthetic.py
    loss_router.py
    train_ssl.py
    train_multitask.py
    train_preference.py
    evaluate.py

  inference_v2/
    __init__.py
    midi_to_graph.py
    infer_critic.py
    output_schema.py
```

The exact names may be adjusted to repository conventions, but keep the separation between raw canonical data, graph construction, model, training, and inference.

## 7. Canonical piece dataclasses

Implement serializable dataclasses or validated dictionaries. Do not couple the canonical schema to PyTorch Geometric.

Suggested high-level structure:

```python
@dataclass
class CanonicalPiece:
    schema_version: str
    piece_id: str
    dataset_name: str
    source_path: str | None
    source_group_id: str
    split: str
    resolution: int | None
    metadata: PieceMetadata
    tempo_events: list[TempoEvent]
    time_signature_events: list[TimeSignatureEvent]
    key_signature_events: list[KeySignatureEvent]
    tracks: list[CanonicalTrack]
    notes: list[CanonicalNote]
    bars: list[CanonicalBar]
    beats: list[CanonicalBeat]
    harmony_events: list[HarmonyAnnotation]
    tonal_regions: list[TonalRegionAnnotation]
    phrase_events: list[SpanAnnotation]
    section_events: list[SpanAnnotation]
    targets: dict[str, TargetArray]
    provenance: dict[str, Any]
    quality_flags: list[str]
```

### 7.1 Timing convention

Use quarter-note beats as the common musical coordinate:

```python
onset_qn: Fraction | rational pair
 duration_qn: Fraction | rational pair
```

For JSON serialization, store rational values as either:

```json
{"num": 3, "den": 2}
```

or integer ticks with a documented common resolution. Internally, preprocessing may use `fractions.Fraction`. Convert to float only when building tensors.

Also preserve:

- original ticks/divisions;
- original seconds where supplied;
- source resolution;
- conversion diagnostics.

### 7.2 Canonical note fields

Required:

```python
note_id: int
track_id: int
pitch: int
pitch_class: int
octave: int
onset_qn: Rational
duration_qn: Rational
velocity: int | None
is_drum: bool
```

Optional observable:

```python
spelling_step: str | None
spelling_alter: int | None
pitch_spelling: str | None
staff_id: int | None
voice_id: int | None
is_grace: bool | None
articulation_ids: list[str]
dynamic_id: str | None
```

Derived raw features should not be serialized as authoritative facts unless useful for caching. They may include:

- bar index;
- position in bar;
- beat strength;
- normalized duration;
- track-relative register;
- previous and next interval.

### 7.3 Canonical track fields

Required:

```python
track_id: int
source_track_index: int
program: int | None
is_drum: bool
name: str | None
```

Optional targets:

```python
role_label: melody | secondary_melody | bass | accompaniment | drums | texture | other
role_confidence: float
role_source: str
```

Derived input statistics computed by graph builder:

- mean pitch;
- min/max pitch;
- pitch standard deviation;
- note density;
- polyphony ratio;
- average duration;
- active-bar ratio;
- velocity statistics;
- onset regularity.

### 7.4 Harmony annotations

Store rich raw/canonical labels without forcing every dataset into the current five-value chord type vocabulary.

```python
@dataclass
class HarmonyAnnotation:
    annotation_id: int
    start_qn: Rational
    end_qn: Rational
    raw_label: str | None
    absolute_root_pc: int | None
    bass_pc: int | None
    coarse_quality: str | None
    extensions: list[str]
    alterations: list[str]
    inversion: int | None
    roman_numeral: str | None
    scale_degree: int | None
    local_key_tonic_pc: int | None
    local_key_mode: str | None
    harmonic_function: str | None
    applied_to_degree: int | None
    borrowed_kind: str | None
    confidence: float
    source: str
    is_human: bool
```

### 7.5 Target arrays

Create a generic target wrapper:

```python
@dataclass
class TargetArray:
    values: list[Any]
    available: list[bool]
    confidence: list[float]
    source: list[str]
    alignment_ids: list[int]
```

Examples:

- note-level scale-degree target aligned to `note_id`;
- beat-level chord-root target aligned to `beat_id`;
- bar-boundary cadence target aligned to `bar_id`;
- track-role target aligned to `track_id`.

## 8. Feature registry

Replace fixed positional layouts such as `NOTE_LAYOUT` and `CHORD_LAYOUT` for V2 with a registry that declares:

- name;
- node type;
- categorical or continuous;
- vocabulary size;
- unknown ID;
- normalization;
- availability-mask behavior;
- whether the feature is permitted during `raw_only` inference.

Example:

```python
FeatureSpec(
    name="pitch",
    node_type="note",
    kind="categorical",
    vocab_size=128,
    unknown_id=128,
    raw_inference_safe=True,
)
```

Store node attributes in `HeteroData` as separate tensors where practical:

```python
data["note"].x_cat
data["note"].x_cont
data["note"].x_cat_available
data["note"].x_cont_available
```

This is preferable to one mixed float tensor containing encoded IDs and continuous values.

## 9. Target registry

Create a task registry defining:

- task name;
- prediction level;
- output dimension;
- loss type;
- ignored/unknown values;
- class weights;
- dataset availability;
- metric set.

Example tasks:

```text
note_pitch_reconstruction
note_duration_reconstruction
note_velocity_reconstruction
track_program_reconstruction
track_role
scale_degree
chord_root
chord_quality
chord_inversion
roman_numeral
local_key
cadence
phrase_boundary
section_boundary
nct_status
harmony_quality
voice_leading_quality
rhythm_quality
structure_quality
track_interaction_quality
overall_preference
aesthetic_content_enjoyment
```

---

# Part IV. Graph schema

## 10. Mandatory node types

### 10.1 `song`

One node per graph/window.

Raw features:

- duration in quarter notes;
- number of bars;
- number of tracks;
- global tempo statistics;
- meter-change count;
- note count;
- optional dataset/domain embedding handled by the model, not necessarily as a raw musical feature.

Do not pass gold key or genre unless the experiment explicitly evaluates conditional metadata.

### 10.2 `track`

One node per MIDI/score track or logical part.

Raw features:

- program;
- instrument family;
- drum flag;
- register statistics;
- density;
- polyphony;
- activity statistics;
- optional track-name embedding with dropout.

### 10.3 `bar`

One node per measure in the window.

Raw features:

- bar index within piece and window;
- numerator and denominator;
- duration;
- pickup flag;
- note/onset/track activity counts;
- tempo statistics;
- downbeat confidence.

### 10.4 `beat`

One node per metrical beat or a configurable regular metric grid.

Raw features:

- beat index;
- position in bar;
- beat strength;
- duration to next beat;
- tempo;
- number of active and starting notes.

Chord candidate heads will operate here.

### 10.5 `onset`

One node per unique note-start time after timing normalization.

Raw features:

- absolute position;
- bar-relative position;
- number of starting notes;
- number of active notes;
- number of active tracks;
- local density.

### 10.6 `note`

One node per note event.

Raw features as defined above.

## 11. Optional semantic node types

### 11.1 `harmony`

Use for datasets with annotated harmonic spans and for analysis/debugging. The deployable model must also support predicting harmony directly at beat/onset candidate slots.

### 11.2 `tonal_region`

Use for annotated local key spans. Do not require these nodes in raw inference.

### 11.3 `phrase` and `section`

Use only when annotations are available or when an explicit predicted-segmentation pass is enabled. Base V2 should predict boundaries from bar tokens.

## 12. Edge types

Mandatory containment and temporal edges:

```text
(song, contains_track, track)
(song, contains_bar, bar)
(track, contains_note, note)
(bar, contains_beat, beat)
(bar, contains_onset, onset)
(bar, contains_note, note)
(beat, contains_onset, onset)
(onset, starts_note, note)
(note, belongs_to_track, track)
(note, belongs_to_bar, bar)
(bar, next_bar, bar)
(beat, next_beat, beat)
(onset, next_onset, onset)
(note, next_in_track, note)
```

Reverse edges should be generated explicitly or by a transform:

```text
(track, has_note, note)
(note, in_onset, onset)
(bar, previous_bar, bar)
```

Recommended additional raw-inference-safe edges:

```text
(note, overlaps_onset, onset)
(note, next_in_voice, note)          only when voice is known or inferred
(note, same_pitch_next, note)        optional
(track, coactive_with, track)        sparse, bar-level evidence
(beat, next_same_position, beat)     optional across bars
```

Semantic supervision edges:

```text
(harmony, covers_note, note)
(harmony, starts_at, beat/onset)
(tonal_region, governs_harmony, harmony)
(phrase, contains_bar, bar)
(section, contains_phrase, phrase)
```

Do not construct full pairwise `simultaneous_with` cliques unless an experiment requires them. Use onset/beat intermediary nodes to avoid quadratic growth.

## 13. Sustained notes

A note contributes to harmony after its start. Therefore, distinguish:

- `starts_note`: onset incidence;
- `overlaps_onset` or `active_at_beat`: sustained activity.

Build sustained-activity edges only to musically relevant candidate slots to control graph size. Recommended initial rule:

- connect each note to every beat node whose time lies in `[onset, offset)`;
- optionally connect to intermediate onset nodes only when needed.

---

# Part V. Dataset adapters

## 14. Common adapter interface

Implement:

```python
class DatasetAdapter(ABC):
    dataset_name: str

    @abstractmethod
    def discover(self, root: Path) -> Iterable[SourceRecord]: ...

    @abstractmethod
    def parse(self, record: SourceRecord) -> CanonicalPiece: ...

    def normalize(self, piece: CanonicalPiece) -> CanonicalPiece: ...
    def validate(self, piece: CanonicalPiece) -> ValidationReport: ...
    def group_id(self, piece: CanonicalPiece) -> str: ...
    def suggested_split(self, piece: CanonicalPiece) -> str | None: ...
```

Every adapter must emit:

- canonical file;
- manifest row;
- validation report;
- statistics row;
- source-group ID for leakage-safe splitting.

## 15. HookTheory V2 adapter

### 15.1 Purpose

HookTheory remains a theory-rich pop lead-sheet source. It is not a true multitrack accompaniment dataset.

### 15.2 Source to canonical mapping

- Create one virtual `track` node with role target `melody`.
- Convert melody events into absolute MIDI pitch where possible using tonic, mode, scale degree, accidental, and octave.
- Preserve raw scale-degree and chord annotations as targets, not default model inputs.
- Convert chord spans into `HarmonyAnnotation` objects.
- Build bars, beats, and onsets from beat units.
- Preserve section annotations when present as optional phrase/section targets.
- Keep `ori_uid` or original-song identity as `source_group_id` so clips from one original song remain in the same split.

### 15.3 Important migration rule

Do not use the already encoded `sd_id`, `root_id`, and similar IDs as raw V2 features. Prefer the canonical pre-encoding HookTheory representation. If only encoded data is available, decode through metadata vocabularies and mark the conversion source.

### 15.4 HookTheory limitations

- no performed accompaniment notes;
- chord symbols are semantic annotations, not a MIDI accompaniment track;
- limited instrumentation;
- fragments may not provide complete form.

Use HookTheory for:

- scale degree;
- melody-harmony alignment;
- functional pop harmony;
- applied and borrowed chords;
- section supervision where available.

## 16. POP909 adapter

### 16.1 Purpose

POP909 is the first priority for validating the new track-aware graph because each song has distinct melody/lead/accompaniment material and chord/key/beat annotations.

### 16.2 Expected musical roles

Preserve the documented tracks separately:

- main/vocal melody;
- secondary or lead melody;
- piano accompaniment.

Attach track-role targets with high confidence based on dataset semantics.

### 16.3 Pipeline

1. Parse MIDI and retain all three tracks.
2. Parse tempo, beat, chord, and key annotation files.
3. Build a reliable mapping among:
   - MIDI ticks;
   - musical quarter-note beats;
   - annotation times in seconds where applicable.
4. Validate monotonic alignment and report maximum alignment error.
5. Create note, onset, beat, bar, and track objects.
6. Parse chord strings into:
   - raw symbol;
   - absolute root;
   - bass;
   - coarse quality;
   - extensions and alterations.
7. Derive relative degree only when a key annotation is available and valid.
8. Store key/chord annotation source as algorithmic/pseudo-human according to dataset documentation, with configurable confidence.
9. Keep multiple versions of the same song under the same `source_group_id` and split.

### 16.4 Do not overcompress chord vocabulary

Retain both:

- a coarse quality vocabulary used for classification;
- a structured representation of extensions, alterations, suspensions, and bass;
- the original raw chord string.

### 16.5 Primary uses

- track-role prediction;
- melody/accompaniment modeling;
- pop harmony;
- cross-track SSL;
- exact two/three-track controllability evaluation;
- multitrack graph debugging.

## 17. Dilemmadata adapter

### 17.1 Purpose

Dilemmadata is the primary source for classical Roman-numeral, local-key, cadence, phrase, and note-wise theory supervision.

### 17.2 Input nature

The dataset uses note-wise TSV-like rows. Theory labels may be repeated on all notes active under one harmonic event. The adapter must not create one harmony node per TSV row.

### 17.3 Pipeline

1. Parse rows and preserve original source corpus and analysis standard.
2. Create one canonical note per note row.
3. Map onset/duration divisions into rational quarter-note units.
4. Preserve staff and voice where available.
5. Build tracks from part/staff information where possible.
6. Build `next_in_voice` edges using `(part, staff, voice)` order.
7. Compress repeated chord labels into harmonic spans using run-length grouping over:
   - local key;
   - root/quality;
   - inversion/bass;
   - Roman numeral;
   - onset continuity.
8. Create tonal-region spans when local key changes.
9. Align cadence labels to beat/bar boundaries and harmonic events.
10. Align phrase and section starts to bar or onset candidates.
11. Store pedal and non-chord information if available.
12. Preserve alternative analyses as separate annotation views, not duplicate independent songs.
13. Generate target masks because fields differ across constituent corpora.

### 17.4 Annotation disagreement

When the same score has two legitimate analyses:

- keep a shared `source_group_id`;
- add `analysis_view_id`;
- optionally create soft target distributions;
- never split alternative analyses across train and test;
- evaluate uncertainty/disagreement separately.

### 17.5 Primary uses

- local key;
- chord root/quality/inversion;
- Roman numeral;
- cadence;
- phrase boundary;
- section boundary;
- note degree;
- voice-aware analysis.

## 18. PDMX adapter

### 18.1 Purpose

PDMX is primarily a large-scale public-domain score corpus for raw symbolic SSL, instrumentation, multitrack structure, notation, and expressive features. It is not a clean Roman-numeral corpus.

### 18.2 Initial subset

Begin with a rated and deduplicated, license-safe subset. Make subset filters configurable and record them in the manifest.

### 18.3 Source format

Prefer the supplied serialized `MusicRender`/JSON representation for initial processing. Add original MXL parsing only when explicit voices/staves or richer notation are required.

### 18.4 Pipeline

1. Load score metadata, resolution, tempo, key signatures, time signatures, barlines, beats, tracks, notes, lyrics, and annotations.
2. Convert score time to rational quarter-note units.
3. Create track nodes from score parts/tracks.
4. Preserve program, drum flag, name, pitch spelling, grace status, velocity, annotations.
5. Create bars and beats from explicit score structure.
6. Create onsets from note starts.
7. Initially build `next_in_track`, not `next_in_voice`, when voice data are unavailable.
8. Parse expressive notation into optional inputs or reconstruction targets:
   - dynamics;
   - articulation;
   - tempo text;
   - rehearsal/section markers;
   - lyrics.
9. Treat notated key signature as optional observable metadata, not ground-truth local key.
10. Do not generate chord or Roman-numeral labels in the first version.
11. Window long scores by bars while retaining piece identity and global metadata.
12. Filter invalid, empty, extremely large, or timing-inconsistent scores with explicit reasons.

### 18.5 Voice handling

V2.1 options:

- use track-level sequential edges only;
- no inferred voices by default.

V2.2 options:

- parse original MusicXML voices;
- add a deterministic voice-separation algorithm;
- store `voice_source` and confidence.

### 18.6 Primary uses

- GraphMAE2-style masked feature learning;
- hierarchical bar/track masking;
- instrument and track modeling;
- multitrack generalization;
- expressive-symbolic reconstruction;
- large-scale pretraining.

Do not use user rating as an absolute musical-quality target in the first experiments. It may be a later weak-label ablation.

## 19. Generic MIDI inference adapter

Implement `GenericMidiAdapter` early. It defines the actual deployment distribution.

It must:

- parse arbitrary type-0 and type-1 MIDI;
- merge tempo and time-signature maps safely;
- preserve tracks and programs where possible;
- handle all notes in one track;
- handle missing tempo/time signature with defaults and flags;
- detect percussion;
- construct bars/beats/onsets/notes/tracks;
- derive track statistics;
- never require key, chord, phrase, or role labels;
- emit the same canonical schema as training adapters.

All labeled validation corpora must be evaluable through this raw MIDI path with labels hidden.

---

# Part VI. Timing, windowing, and preprocessing safeguards

## 20. Meter and tempo maps

Implement a reusable timing map that supports:

- tempo changes;
- time-signature changes;
- pickup measures;
- irregular meters;
- missing metadata;
- conversion among ticks, quarter-note beats, bars, and seconds.

Add tests for:

- 4/4 to 3/4 change;
- tempo change at a bar boundary;
- tempo change mid-bar;
- pickup bar;
- triplet timing;
- sustained notes across meter changes.

## 21. Quantization

Do not globally force all datasets to a sixteenth-note grid during canonicalization.

Use:

- exact rational timing in canonical data;
- optional configurable graph quantization for beat candidate slots;
- tolerance-based grouping of simultaneous onsets;
- diagnostic counts of snapped events and maximum error.

## 22. Windowing

Training graphs should use configurable windows:

```text
4 bars   debugging and local tasks
8 bars   default SSL prototype
16 bars  form and phrase context
32 bars  later long-context experiments
```

Rules:

- window boundaries must preserve source-group identity;
- include a configurable context halo around supervised target positions;
- record original bar range;
- avoid splitting notes without indicating clipping;
- use overlapping windows only in train, or group evaluation by original piece;
- prevent windows of one piece from crossing splits.

---

# Part VII. Augmentation and imbalance

## 23. Transposition augmentation

Implement on-the-fly global transposition for pitched tracks.

### 23.1 Valid transformation

For shift `k` semitones:

- shift all non-drum MIDI pitches;
- shift absolute key tonic labels;
- shift absolute chord roots and bass pitch classes;
- shift pitch spelling where supported;
- preserve relative Roman numeral, scale degree, chord quality, cadence, rhythm, and form.

### 23.2 Constraints

- choose only shifts that keep pitches in valid MIDI range;
- optionally respect conservative instrument ranges;
- do not transpose percussion;
- do not physically duplicate all files;
- apply only after group-safe train/val/test split;
- log sampled shifts.

### 23.3 Do not treat mode change as label-preserving

Major-to-minor or modal conversion is not ordinary transposition. Do not include it as a simple augmentation.

### 23.4 Consistency/equivariance losses

Invariant outputs should agree after transposition:

- chord quality;
- Roman numeral;
- cadence;
- phrase boundary;
- structure quality.

Equivariant outputs should shift correctly:

- absolute pitch;
- absolute chord root;
- local tonic.

Implement optional transposition consistency after the basic SSL pipeline works.

## 24. Class imbalance

Transposition does not solve imbalance of chord qualities, cadences, applied chords, borrowed chords, or phrase boundaries.

Implement:

1. dataset-balanced sampling;
2. task-aware sampling;
3. target-event sampling for rare labels;
4. class-balanced or logit-adjusted cross entropy;
5. focal loss as an ablation, not the default everywhere;
6. balanced masking that selects rare target fields more often;
7. macro-F1, per-class recall, and PR-AUC reporting.

Never report only accuracy for cadence or rare harmony classes.

## 25. Dataset mixture sampling

PDMX must not numerically drown out HookTheory, POP909, and Dilemmadata.

Support sampling modes:

- fixed dataset probabilities;
- temperature sampling based on dataset size;
- round-robin batches;
- task-aware batch scheduling.

Initial recommended SSL mixture:

```yaml
pdmx: 0.55
pop909: 0.20
dilemmadata: 0.15
hooktheory: 0.10
```

Treat these as configurable starting values, not fixed research conclusions.

---

# Part VIII. Multi-dataset loader and batching

## 26. Dataset output contract

Replace the V1 contract of `real/masked/corrupted` with a more general V2 item:

```python
{
    "piece_id": str,
    "dataset_name": str,
    "source_group_id": str,
    "graph_full": HeteroData,
    "view_a": HeteroData,
    "view_b": HeteroData | None,
    "mask_plan_a": MaskPlan,
    "mask_plan_b": MaskPlan | None,
    "targets": dict[str, Tensor],
    "target_masks": dict[str, Tensor],
    "target_confidence": dict[str, Tensor],
    "provenance": dict,
}
```

During SSL, `graph_full` may be used only by a stop-gradient target encoder or decoder target extraction. It must not leak masked values into the online encoder.

## 27. Empty optional node types

PyG batching must work when one dataset has no `harmony`, `phrase`, or `section` nodes.

Options:

- keep the base model metadata limited to mandatory raw node types;
- place optional semantic-node tasks in a separate analysis branch;
- or initialize empty storages consistently.

Recommended V2.1: base encoder uses mandatory raw node types only. Theory predictions are made at candidate beat/bar/note/track positions. This minimizes schema mismatch.

## 28. Target routing

Implement a loss router:

```python
for task_name, prediction in outputs["task_logits"].items():
    if task_name not in batch.targets:
        continue
    mask = batch.target_masks[task_name]
    if mask.any():
        loss += weight * task_loss(prediction[mask], target[mask])
```

Confidence may scale individual examples:

```python
loss_i *= confidence_i
```

Do not infer task availability solely from dataset name. Use actual masks.

---

# Part IX. SSL design inspired by GraphMAE2, Hi-GMAE, and UGMAE

## 29. Important terminology

The implementation should be described as **inspired by** these methods unless it faithfully reproduces their original algorithms and experimental settings.

The goal is one integrated music-specific masked graph framework, not three models chained sequentially.

## 30. SSL V2.1: GraphMAE2-style baseline

### 30.1 Online and target paths

Recommended structure:

```text
full graph --------------------------> target encoder / target extractor
   |
   +--> encoder mask --> online encoder --> decoder remask views --> reconstruction
```

Possible target representation implementations:

1. stop-gradient output of the same encoder on an unmasked or lightly augmented graph;
2. exponential-moving-average target encoder;
3. stop-gradient pre-decoder latent target.

Start with the simplest stable implementation, then add EMA as an ablation.

### 30.2 Maskable raw fields

Notes:

- pitch or pitch class;
- duration bucket;
- velocity bucket;
- onset-offset feature;
- optional spelling;
- optional articulation.

Tracks:

- program/instrument family;
- selected statistics;
- optional name embedding.

Bars/beats/onsets:

- local counts;
- tempo/meter fields only when reconstruction is nontrivial;
- structural embeddings rather than deterministic identifiers.

Avoid reconstructing fields that are exact deterministic copies of visible features.

### 30.3 Grouped masks

Prevent shortcuts. Examples:

- when masking scale-relative pitch targets, do not expose an exact encoded scale degree;
- when masking pitch, decide whether pitch class and octave are also masked;
- when masking program, optionally mask track-name metadata;
- when masking chord target, do not pass gold chord-node fields.

### 30.4 Multi-view decoder re-masking

After online encoding, create `K` decoder views. In each view, re-mask a random subset of latent node embeddings before decoding the same original targets.

Config example:

```yaml
ssl:
  decoder_views: 3
  decoder_remask_prob: 0.20
```

Average reconstruction and latent losses across views.

### 30.5 Latent prediction

Add node- or hierarchy-level latent prediction:

```math
L_latent = 1 - cosine(P(z_masked), stopgrad(z_full))
```

Use projectors/predictors to avoid forcing raw encoder spaces to be identical.

Compute at selected levels:

- note/onset local embeddings;
- bar embeddings;
- track embeddings;
- song embedding.

Do not require every level from the first commit.

## 31. SSL V2.2: Hi-GMAE-style hierarchy

### 31.1 Use explicit musical hierarchy

Unlike generic learned graph pooling, music already has meaningful membership relations:

```text
note -> onset -> beat -> bar -> song
note -> track -> song
```

Use these deterministic relations first.

### 31.2 Coarse-to-fine masks

Sample a mask at a coarse level and project it downward.

Examples:

- mask one bar and selected notes/onsets inside it;
- mask one track for several bars;
- mask a contiguous beat span;
- later mask an annotated/predicted phrase.

Represent every mask as a `MaskPlan` with:

```python
mask_kind
node_ids_by_type
feature_names_by_type
span_start_qn
span_end_qn
severity
random_seed
```

### 31.3 Gradual recovery

Implement decoders in stages:

1. recover bar/track latent targets;
2. use recovered coarse context to decode onset/beat representations;
3. recover note attributes.

A practical V2.2 implementation can use:

- coarse latent loss at bar and track levels;
- fine raw reconstruction at note level;
- no fully recursive decoder initially.

### 31.4 Hierarchical loss

```math
L_hier = λ_note L_note + λ_onset L_onset + λ_beat L_beat + λ_bar L_bar + λ_track L_track + λ_song L_song
```

Each component must log count and mean separately.

## 32. SSL V2.3: UGMAE-style extensions

### 32.1 Adaptive mask generation

Do not start with a learned mask generator. First establish uniform and span masks.

Later implement a mask policy that can use:

- node degree;
- duration;
- graph centrality;
- reconstruction uncertainty;
- class rarity;
- encoder salience;
- local density.

Blend adaptive and uniform masks to prevent collapse:

```yaml
mask_policy:
  uniform_fraction: 0.50
  span_fraction: 0.25
  adaptive_fraction: 0.25
```

Use curriculum scheduling.

### 32.2 Structure reconstruction

Predict masked or held-out relations.

Positive relation examples:

- next note in track;
- note belongs to track;
- note belongs to bar;
- note active at beat;
- next bar;
- track coactivity.

Use relation-aware scoring:

```python
score = relation_mlp([h_src, h_dst, relation_embedding, h_src * h_dst, h_src - h_dst])
```

Sample hard negatives from the same piece and compatible node types.

Use ranking or binary loss. Log relation-specific AUC and MRR where meaningful.

### 32.3 Bootstrap similarity

Two independently masked views of one piece should have compatible global representations:

```math
L_bootstrap = 1 - cosine(p(z_a), stopgrad(z_b))
```

Use asymmetric predictor plus optional symmetric direction.

### 32.4 Consistency targets

Benign transformations should preserve appropriate outputs:

- transposition;
- velocity jitter;
- metadata dropout;
- equivalent track splitting/merging only when carefully constructed;
- removal of nonessential optional metadata.

Specify which outputs are invariant and which are equivariant. Do not enforce identical local note embeddings after transposition.

## 33. Combined SSL objective

Initial full objective:

```math
L_SSL =
  λ_feat L_feature_reconstruction
+ λ_edge L_structure_reconstruction
+ λ_latent L_latent_prediction
+ λ_hier L_hierarchical_recovery
+ λ_boot L_bootstrap
+ λ_cons L_consistency
```

Every term must be individually switchable by config.

Recommended implementation order:

1. feature reconstruction;
2. multi-view decoder remasking;
3. song/bar latent prediction;
4. hierarchical span masking;
5. structure reconstruction;
6. bootstrap consistency;
7. adaptive masking.

---

# Part X. Hybrid hierarchical graph-Transformer architecture

## 34. High-level architecture

```text
raw heterogeneous graph
        |
        v
node-type feature encoders
        |
        v
local relation-aware graph encoder
        |
        +----------------------+
        |                      |
        v                      v
note/onset/beat/bar pooling    note/track pooling
        |                      |
        +----------+-----------+
                   v
       hierarchical global Transformer
          bar tokens + track tokens + song token
                   |
                   v
          top-down contextual fusion
                   |
        +----------+-----------------------------+
        |                  |                     |
        v                  v                     v
   SSL decoders       theory heads         quality/preference heads
```

## 35. Feature encoders

Use separate categorical embeddings and continuous projections.

For each node type:

```python
cat_emb = sum(embedding_i(x_cat[:, i]))
cont_emb = cont_mlp(normalized_x_cont)
avail_emb = availability_projection(mask_bits)
h0 = layer_norm(cat_emb + cont_emb + avail_emb + type_embedding)
```

Include:

- node-type embedding;
- dataset/domain embedding only as an optional ablation;
- positional/metric embeddings for bar, beat, and onset;
- track-ID embedding local to a graph only if carefully handled.

Do not use absolute song identity embeddings.

## 36. Local graph encoder

### 36.1 Baseline

First implement a V2 baseline using relation-specific `HeteroConv` and residual layers, because the repository already uses PyG `HeteroConv`.

### 36.2 Stronger option

Add configurable HGT-style or relation-aware attention layers later.

The local encoder should model:

- within-track note progression;
- note-to-onset vertical context;
- sustained-note context;
- note-to-bar and note-to-track membership;
- local cross-track evidence through shared onset/beat/bar nodes.

Recommended depth: 3–5 layers. Do not attempt to model full-song form only by adding many GNN layers.

## 37. Hierarchical pooling

Implement membership-aware attention pooling.

### 37.1 Note to onset

Aggregate starting notes and optionally sustained notes separately:

```python
onset_update = concat(
    onset_embedding,
    attn_pool(starting_notes),
    attn_pool(active_notes),
    attn_pool(per_track_summaries),
)
```

### 37.2 Onset to beat

Aggregate irregular onsets into regular metric positions.

### 37.3 Beat to bar

Use ordered pooling or a small local Transformer/GRU within a bar.

### 37.4 Note to track

Aggregate note embeddings with time-aware attention and track statistics.

## 38. Global Transformer

### 38.1 Tokens

Construct per-piece tokens:

```text
[SONG]
[BAR_0] ... [BAR_T]
[TRACK_0] ... [TRACK_K]
```

Later optional:

```text
[PHRASE_0] ...
```

### 38.2 Positional information

Bar tokens:

- absolute bar index within window;
- relative bar position;
- meter embedding;
- optional position in piece.

Track tokens:

- no arbitrary sequential positional order is required;
- use instrument and track-statistic embeddings;
- retain a stable source index only for batching/debugging.

### 38.3 Attention strategy

Start with one padded `nn.TransformerEncoder` over song+bar+track tokens with token-type embeddings and padding masks.

Later compare:

- full attention;
- bar-only temporal Transformer plus track set Transformer;
- sparse hierarchical attention;
- relation-biased attention.

## 39. Top-down fusion

Return global context to local entities.

For each note:

```python
note_final = gated_fusion(
    note_local,
    onset_context,
    beat_context,
    bar_transformer_context,
    track_transformer_context,
    song_context,
)
```

Use membership edges and batch indices for efficient scatter/gather. Avoid Python loops over every note as in current contextual score computation.

## 40. Model output contract

```python
{
    "node_embeddings_local": {...},
    "node_embeddings_contextual": {...},
    "bar_embeddings": Tensor,
    "track_embeddings": Tensor,
    "song_embedding": Tensor,
    "ssl_predictions": {...},
    "theory_logits": {...},
    "aspect_scores": {
        "harmony": Tensor,
        "melody_harmony": Tensor,
        "voice_leading": Tensor,
        "rhythm": Tensor,
        "structure": Tensor,
        "track_interaction": Tensor,
    },
    "overall_utility": Tensor,
    "aesthetic_predictions": {...},
    "uncertainty": {...},
}
```

`overall_utility` is a latent ranking utility until calibrated. Do not name it MOS in code or reports without human MOS training.

---

# Part XI. Theory heads

## 41. Inference-safe prediction positions

### 41.1 Note-level

- scale degree;
- chord-tone/NCT status;
- NCT type;
- optional voice role.

### 41.2 Beat/onset-level

- chord boundary/change;
- root;
- quality;
- bass/inversion;
- Roman numeral components;
- local key.

### 41.3 Bar-boundary-level

- cadence;
- phrase boundary;
- section boundary;
- modulation probability.

### 41.4 Track-level

- melody;
- secondary melody;
- bass;
- chordal accompaniment;
- drums;
- texture/other.

## 42. Factorized harmony representation

Avoid one enormous flat Roman-numeral class if possible. Predict factorized fields:

- local tonic;
- mode;
- scale degree;
- chord quality;
- inversion;
- secondary/applied degree;
- borrowed/modal status.

Also retain an optional flat-label head for direct benchmark comparison.

## 43. Using theory predictions in quality scoring

V2.1 safest design:

```text
shared encoder -> theory heads
shared encoder -> quality heads
```

Theory supervision helps indirectly through shared representations.

V2.2 ablation:

```text
shared encoder -> soft theory distributions -> quality head
```

If using predicted theory in the quality head:

- pass probabilities, not hard argmax labels;
- never train the quality head only on perfect ground truth;
- use theory-label dropout/noise or scheduled predicted inputs;
- detach or not detach predictions as a configurable experiment.

---

# Part XII. Quality and preference modeling

## 44. Do not rely on one absolute score initially

The current model learns clean-vs-corrupted ranking but the raw graph score is not globally calibrated across unrelated songs. V2 should explicitly distinguish:

- local utility used for ranking;
- aspect scores;
- pairwise preference probability;
- calibrated MOS if available later.

## 45. Aspect heads

Add heads for:

- harmony;
- melody-harmony compatibility;
- voice leading;
- rhythm/meter;
- structural coherence;
- track interaction/arrangement;
- optional stylistic fit;
- audio-aesthetic surrogate.

Each head may produce:

- song/window score;
- bar-level score;
- local violation logits;
- uncertainty.

## 46. Preference loss

For pair `(A, B)`:

```math
P(A > B) = sigmoid(u_A - u_B)
```

Use Bradley-Terry/logistic pairwise loss.

Support ties or uncertain judgments with soft labels.

For GRPO, normalize reward components inside each prompt group before weighted fusion.

## 47. Counterfactual degradation and fragility

Keep the concept as an evaluation and auxiliary training signal:

```math
Δ(G, c) = u(G) - u(c(G))
```

However:

- large degradation does not prove high absolute quality;
- random masking measures recoverability, not degradation;
- theory-aware corruptions may be used as an ablation or hard-negative source;
- OOD corruption and benign-transform tests are mandatory if corruptions are used.

Do not make handcrafted corruptions the sole basis of the V2 critic.

## 48. Reconstruction-based diagnostics

Possible SSL diagnostics:

- masked-note reconstruction loss;
- masked-track reconstruction loss;
- bar latent recoverability;
- track dropout stability;
- hierarchy-level reconstruction profile.

These may correlate with internal coherence but must not be presented as objective musical quality without validation.

---

# Part XIII. Audio-aesthetic integration

## 49. Teacher score generation

Add an offline pipeline:

```text
MIDI -> deterministic renderer -> audio -> Audiobox Aesthetics -> cached labels
```

Store:

- MIDI checksum;
- renderer name/version;
- soundfont/model;
- sample rate;
- normalization settings;
- windowing settings;
- all aesthetic output dimensions;
- teacher model version;
- timestamp.

Text is not required for a no-reference aesthetic model. Keep CLAP/text alignment as a separate component.

## 50. Symbolic aesthetic student

Add regression heads on the shared song/window embedding for:

- content enjoyment;
- production quality;
- production complexity;
- content usefulness;
- uncertainty.

Recommended loss:

```math
L_aesthetic = λ_reg Huber(y_hat, y)
            + λ_rank PairwiseRank(y_hat_i, y_hat_j)
            + λ_unc GaussianNLL_or_variance_loss
```

The symbolic student approximates a fixed renderer+audio-model pipeline. It is not a renderer-independent ground truth.

## 51. Teacher-in-the-loop deployment

Later GRPO strategy:

1. score all candidates with the symbolic student;
2. render a random subset, uncertain subset, and top subset;
3. query the real audio teacher;
4. add corrected labels to replay data;
5. periodically update the student;
6. monitor reward hacking and distribution shift.

Do not use only the student indefinitely during RL.

---

# Part XIV. Training stages

## 52. Stage 0: data and graph smoke tests

Datasets:

- migrated HookTheory;
- POP909 small subset;
- tiny synthetic fixtures.

Objectives:

- no training beyond graph integrity.

Exit criterion:

- same raw graph interface works for HookTheory, POP909, and generic MIDI.

## 53. Stage 1: basic masked SSL

Datasets:

- POP909;
- small PDMX subset;
- optionally HookTheory/Dilemmadata raw notes.

Model:

- feature encoders;
- local hetero-GNN;
- note/onset/beat/bar/track hierarchy;
- bar+track Transformer;
- feature reconstruction.

No theory-aware corruption required.

## 54. Stage 2: GraphMAE2-style extensions

Add:

- decoder re-masking views;
- latent prediction;
- two-view consistency.

Evaluate frozen/linear probes.

## 55. Stage 3: hierarchical masking

Add:

- bar-span masking;
- track dropout/masking;
- coarse-to-fine mask projection;
- bar/track latent reconstruction.

## 56. Stage 4: theory multitask supervision

Datasets:

- Dilemmadata;
- HookTheory;
- POP909.

Losses activated by target masks.

Train either:

- from SSL checkpoint with full fine-tuning;
- or staged freeze/unfreeze.

## 57. Stage 5: UGMAE-style structure and adaptive masking

Add:

- edge reconstruction;
- hard-negative sampling;
- bootstrap similarity;
- adaptive mask policy;
- consistency transforms.

Only after simpler baselines are stable.

## 58. Stage 6: preference critic

Collect or build comparisons from:

- multiple Text2MIDI candidates for one prompt;
- outputs from different checkpoints;
- Text2MIDI versus MuseCoco where reproducible;
- human ratings;
- optionally curated original versus model output;
- selected corruption pairs as auxiliary data, not the sole source.

Train aspect and overall utility heads.

## 59. Stage 7: aesthetic distillation

Generate cached audio-aesthetic teacher labels and train symbolic aesthetic heads.

## 60. Stage 8: GRPO integration

Use a weighted, normalized reward:

```math
R = α R_symbolic_preference
  + β R_aesthetic
  + γ R_text_alignment
  + δ R_controllability
```

Add explicit controllability checks:

- exact track count;
- requested instruments;
- track-role assignment;
- monophonic melody requirement;
- accompaniment polyphony;
- active duration of tracks.

---

# Part XV. Loss routing

## 61. Total loss

A configurable superset:

```math
L =
  λ_ssl L_ssl
+ Σ_t λ_t m_t L_theory,t
+ λ_pref L_preference
+ λ_aspect L_aspect
+ λ_aes L_aesthetic
+ λ_cal L_calibration
```

`m_t` is the per-example/per-node target availability mask.

## 62. Uncertainty and task balancing

Possible later options:

- learned homoscedastic task uncertainty;
- GradNorm;
- dynamic weight averaging.

Do not add these before fixed manually configured weights have a stable baseline.

## 63. Logging

Log for every task:

- loss;
- valid target count;
- dataset-specific metrics;
- macro and micro metrics;
- confidence-weighted and unweighted metrics where relevant.

Never average a zero-target batch as a zero loss without logging count.

---

# Part XVI. Inference on unlabeled MIDI

## 64. Public inference path

```text
ordinary MIDI
  -> GenericMidiAdapter
  -> canonical raw piece
  -> raw graph builder
  -> MusicCriticV2
  -> theory predictions + aspect scores + utility + uncertainty
```

No ground-truth theoretical fields are supplied.

## 65. Soft intermediate predictions

Return distributions for ambiguous concepts:

```json
{
  "chord_root_probs": [...],
  "chord_quality_probs": [...],
  "local_key_probs": [...],
  "cadence_probs": [...]
}
```

Do not immediately collapse all predictions to hard labels inside the quality model.

## 66. Output schema

Suggested output:

```json
{
  "model_version": "...",
  "schema_version": "...",
  "piece_summary": {
    "num_tracks": 5,
    "num_bars": 16,
    "duration_qn": 64.0
  },
  "theory": {
    "track_roles": [],
    "local_keys": [],
    "chords": [],
    "cadences": [],
    "phrase_boundaries": []
  },
  "quality": {
    "harmony": 0.0,
    "melody_harmony": 0.0,
    "voice_leading": 0.0,
    "rhythm": 0.0,
    "structure": 0.0,
    "track_interaction": 0.0,
    "overall_utility": 0.0,
    "aesthetic_content_enjoyment": 0.0
  },
  "uncertainty": {},
  "warnings": []
}
```

Name the score `overall_utility` until calibrated to human MOS.

---

# Part XVII. Evaluation plan

## 67. Data preprocessing evaluation

For every dataset report:

- discovered/processed/skipped counts;
- skip reasons;
- note/track/bar distributions;
- timing conversion error;
- missing metadata rates;
- target availability rates;
- class distributions;
- duplicate group statistics;
- window counts;
- graph size distributions.

## 68. SSL evaluation

- masked feature accuracy/MAE;
- edge reconstruction AUC/MRR;
- latent consistency;
- reconstruction by mask level;
- performance versus mask ratio;
- linear probes for theory tasks;
- cross-dataset probes;
- frozen versus fine-tuned probes.

## 69. Theory evaluation

Use:

- macro-F1;
- per-class precision/recall;
- balanced accuracy;
- PR-AUC for rare binary labels;
- sequence/span metrics for boundaries;
- chord segmentation and label metrics;
- calibration/ECE;
- accuracy by dataset and genre/domain.

## 70. Critic evaluation

- pairwise accuracy;
- ROC-AUC where appropriate;
- Spearman and Kendall ranking correlation;
- system-level ranking correlation;
- prompt-group ranking accuracy;
- human preference correlation;
- calibration;
- uncertainty-quality relationship;
- worst-dataset/worst-generator performance.

## 71. OOD and robustness evaluation

- unseen generator outputs;
- unseen corruption families;
- benign transposition;
- metadata removal;
- missing tracks;
- incorrect track names;
- type-0 merged MIDI;
- different numbers of tracks;
- genre transfer;
- classical-to-pop and pop-to-classical transfer.

## 72. Required architecture ablations

1. sequence/compound-token Transformer baseline;
2. local GNN only;
3. global Transformer only on pooled raw features;
4. hybrid GNN + Transformer;
5. hybrid without top-down fusion;
6. hybrid with and without track nodes;
7. simple random masking;
8. GraphMAE2-style additions;
9. hierarchical masking;
10. UGMAE-style structure/bootstrap;
11. theory supervision on/off;
12. predicted-theory-to-quality fusion on/off.

## 73. Required data ablations

1. HookTheory only;
2. HookTheory + POP909;
3. plus Dilemmadata;
4. plus PDMX SSL;
5. transposition off/on;
6. class balancing off/on;
7. pseudo-labels off/on.

---

# Part XVIII. Detailed implementation roadmap

## Phase 0. Baseline protection and test snapshot

### Tasks

- Run existing tests.
- Add a short `docs/v2_baseline_snapshot.md` recording current commands and expected artifacts.
- Add golden tiny HookTheory graph fixtures for current V1 behavior.
- Ensure new dependencies are not added unnecessarily.

### Acceptance criteria

- Existing V1 training smoke test still starts.
- Existing graph and loss tests pass.
- No V1 file formats changed.

## Phase 1. Canonical schema and validation

### Add

- `src/data_v2/schema.py`;
- `src/data_v2/timing.py`;
- `src/data_v2/validation.py`;
- `src/data_v2/serialization.py`;
- schema unit tests.

### Required tests

- canonical JSON round trip;
- rational timing round trip;
- invalid pitch rejection;
- negative duration rejection;
- missing optional field acceptance;
- target lengths and alignment checks.

### Acceptance criteria

- a synthetic two-track piece serializes and validates;
- validation report contains structured errors/warnings;
- schema version is written.

## Phase 2. Generic MIDI and HookTheory V2 adapters

### Add

- `GenericMidiAdapter`;
- `HookTheoryV2Adapter`;
- conversion CLI;
- manifest writer.

### Required behavior

- both emit the same canonical schema;
- HookTheory theory is stored as targets;
- generic MIDI needs no theory labels;
- one virtual melody track is created for HookTheory;
- original-song grouping is preserved.

### Acceptance criteria

- convert at least 10 HookTheory clips;
- convert synthetic MIDI with 1, 2, and 5 tracks;
- canonical validator passes;
- raw-only graph can be built from both.

## Phase 3. V2 raw graph builder

### Add

- mandatory node types;
- mandatory edges;
- reverse-edge utility;
- sustained-note-to-beat edges;
- feature registry;
- graph inspection CLI.

### Tests

- node counts;
- edge endpoint validity;
- temporal order;
- no cross-graph indices;
- empty track handling;
- sustained notes;
- meter changes.

### Acceptance criteria

- identical model-facing schema for HookTheory and generic MIDI;
- no theory target appears in raw input tensors;
- graph metadata stores schema versions.

## Phase 4. POP909 adapter

### Tasks

- discover songs and versions;
- parse MIDI tracks and annotation files;
- align time systems;
- create track-role targets;
- parse chord/key spans;
- generate alignment reports;
- group versions safely.

### Tests

- known track-role mapping;
- chord span alignment;
- key-change alignment;
- multi-version grouping;
- no split leakage.

### Acceptance criteria

- process a small POP909 subset with less than a configured alignment-error threshold;
- graph includes three tracks and correct role targets;
- raw inference mode works with annotation files hidden.

## Phase 5. Multi-source dataset and collator

### Tasks

- load canonical files from multiple datasets;
- support dataset mixture probabilities;
- collate graphs, targets, masks, confidence, and provenance;
- support deterministic worker seeds;
- implement stats by dataset.

### Acceptance criteria

- one batch can contain different datasets;
- missing tasks do not produce loss;
- PDMX-like no-harmony sample and HookTheory sample collate together.

## Phase 6. MusicCriticV2 baseline architecture

### Implement

- feature encoder;
- local relation-aware GNN baseline;
- hierarchy pooling;
- bar+track Transformer;
- song embedding;
- simple raw reconstruction heads.

### Do not yet implement

- adaptive masking;
- aesthetic head;
- preference head;
- phrase nodes.

### Acceptance criteria

- forward pass on mixed batch;
- no Python loop over every note in score-head computation;
- gradients flow through all mandatory node types;
- one-batch overfit test passes for reconstruction.

## Phase 7. GraphMAE2-style SSL

### Implement

- field-group masking;
- multiple decoder re-mask views;
- full/light target path;
- latent prediction at song and bar levels;
- switchable EMA target encoder.

### Acceptance criteria

- masked fields are not visible to online encoder;
- loss decreases on tiny dataset;
- decoder views differ deterministically by seed;
- latent target has stop-gradient;
- comparison against simple V1-like masking is logged.

## Phase 8. Hi-GMAE-style hierarchy

### Implement

- bar-span masks;
- track masks/dropout;
- coarse-to-fine `MaskPlan` projection;
- bar and track latent recovery;
- configurable mask curriculum.

### Acceptance criteria

- masked bar removes or masks all intended descendants;
- decoder receives visible hierarchy correctly;
- per-level losses are logged;
- one-batch overfit test for bar masking passes.

## Phase 9. Dilemmadata adapter and theory heads

### Implement

- TSV parser;
- note/voice mapping;
- harmony run compression;
- tonal regions;
- cadence/phrase/section targets;
- alternate-analysis provenance;
- theory heads and loss routing.

### Acceptance criteria

- no repeated one-chord-per-note graph explosion;
- alternative analyses remain grouped;
- raw-only input predicts labels;
- theory label leakage test passes;
- macro-F1 metrics are produced.

## Phase 10. PDMX adapter and scalable cache

### Implement

- subset manifest filters;
- MusicRender JSON reader;
- long-score windowing;
- track/notation fields;
- multiprocessing preprocessing;
- shard/cache format;
- skip report.

### Acceptance criteria

- process a 1k-piece subset reproducibly;
- restart/resume preprocessing;
- graph-size percentiles reported;
- invalid files isolated rather than stopping run;
- training loader streams cached windows.

## Phase 11. UGMAE-style structure and consistency

### Implement

- relation reconstruction;
- hard negative sampler;
- bootstrap two-view loss;
- transposition consistency;
- optional learned adaptive masks.

### Acceptance criteria

- relation negatives never include true positives;
- metrics per relation;
- transposition invariant/equivariant tests;
- adaptive policy cannot select zero or all nodes;
- uniform-mask baseline remains available.

## Phase 12. Preference and aspect critic

### Implement

- pair dataset schema;
- prompt-group IDs;
- aspect heads;
- Bradley-Terry loss;
- tie/soft preference support;
- uncertainty;
- group-level metrics.

### Acceptance criteria

- pair order swap changes target consistently;
- no leakage from prompt or file names;
- one-batch pair overfit;
- prompt-group evaluation report.

## Phase 13. Audio-aesthetic distillation

### Implement

- renderer manifest;
- teacher-score cache schema;
- aesthetic heads;
- regression + ranking loss;
- uncertainty-based teacher query selection.

### Acceptance criteria

- labels tied to renderer/model version;
- cached teacher scores reproducible;
- symbolic student beats constant baseline;
- renderer shift is evaluated separately.

## Phase 14. Inference and GRPO API

### Implement

- raw MIDI CLI;
- batch scoring API;
- JSON output schema;
- per-prompt reward normalization;
- explicit controllability checks;
- optional real audio-teacher fallback.

### Acceptance criteria

- score arbitrary unlabeled MIDI;
- no theory annotation required;
- exact-track-count and role diagnostics available;
- latency and memory benchmark logged;
- old observer inference still works independently.

---

# Part XIX. Proposed configuration

## 74. Example V2 Hydra config

```yaml
project:
  name: music_critic_v2
  schema_version: "2.0.0"

sources:
  hooktheory:
    enabled: true
    manifest: data/v2/hooktheory/manifest.jsonl
    weight: 0.10
  pop909:
    enabled: true
    manifest: data/v2/pop909/manifest.jsonl
    weight: 0.20
  dilemmadata:
    enabled: true
    manifest: data/v2/dilemmadata/manifest.jsonl
    weight: 0.15
  pdmx:
    enabled: true
    manifest: data/v2/pdmx/manifest.jsonl
    weight: 0.55

windowing:
  bars: 8
  stride_bars: 4
  clip_sustained_notes: true

augmentation:
  transpose:
    enabled: true
    min_semitones: -5
    max_semitones: 6
    respect_instrument_ranges: false
  metadata_dropout: 0.20
  velocity_jitter: 0.05

masking:
  encoder_mask_prob: 0.30
  mask_min_nodes: 1
  field_groups: true
  span_mask_fraction: 0.25
  track_mask_fraction: 0.10
  decoder_views: 3
  decoder_remask_prob: 0.20
  adaptive:
    enabled: false
    uniform_fraction: 0.50
    span_fraction: 0.25
    learned_fraction: 0.25

model:
  hidden_dim: 256
  local_layers: 4
  local_encoder: hetero_sage
  local_heads: 8
  transformer_layers: 4
  transformer_heads: 8
  transformer_ff_dim: 1024
  dropout: 0.10
  use_top_down_fusion: true
  use_ema_target_encoder: false

losses:
  feature_reconstruction: 1.0
  edge_reconstruction: 0.0
  latent_prediction: 0.25
  hierarchical_recovery: 0.25
  bootstrap: 0.0
  consistency: 0.0
  theory_total: 1.0
  preference: 0.0
  aesthetic: 0.0

training:
  stage: ssl
  batch_size: 8
  grad_accum_steps: 4
  lr: 0.0003
  weight_decay: 0.0001
  epochs: 100
  use_amp: true
  grad_clip: 1.0
  seed: 42
```

Do not make this the only config. Create stage-specific config groups.

---

# Part XX. Critical pseudocode

## 75. Building the raw graph

```python
def build_raw_graph(piece: CanonicalPiece, registry: FeatureRegistry) -> HeteroData:
    data = HeteroData()

    data["song"] = encode_song(piece)
    data["track"] = encode_tracks(piece.tracks)
    data["bar"] = encode_bars(piece.bars)
    data["beat"] = encode_beats(piece.beats)
    data["onset"] = encode_onsets(piece.notes, piece.beats)
    data["note"] = encode_notes(piece.notes)

    add_song_track_edges(data, piece)
    add_song_bar_edges(data, piece)
    add_bar_beat_edges(data, piece)
    add_bar_onset_edges(data, piece)
    add_onset_note_edges(data, piece)
    add_note_track_edges(data, piece)
    add_note_bar_edges(data, piece)
    add_temporal_edges(data, piece)
    add_sustained_activity_edges(data, piece)
    add_reverse_edges(data)

    attach_targets_and_masks(data, piece.targets)
    attach_schema_metadata(data, piece)
    validate_heterodata(data)
    return data
```

## 76. Creating SSL views

```python
def make_ssl_item(full_graph, rng, cfg):
    plan_a = mask_planner.sample(full_graph, rng, cfg)
    plan_b = mask_planner.sample(full_graph, rng, cfg)

    view_a = apply_encoder_mask(full_graph, plan_a)
    view_b = apply_encoder_mask(full_graph, plan_b)

    return {
        "graph_full": full_graph,
        "view_a": view_a,
        "view_b": view_b,
        "mask_plan_a": plan_a,
        "mask_plan_b": plan_b,
        "reconstruction_targets_a": extract_targets(full_graph, plan_a),
        "reconstruction_targets_b": extract_targets(full_graph, plan_b),
    }
```

## 77. Model forward

```python
def forward(self, graph, decoder_mask_plans=None):
    h0 = self.feature_encoder(graph)
    h_local = self.local_graph_encoder(h0, graph.edge_index_dict)

    hierarchy = self.hierarchy_pool(h_local, graph)
    global_ctx = self.global_transformer(
        song=hierarchy.song,
        bars=hierarchy.bars,
        tracks=hierarchy.tracks,
        masks=hierarchy.padding_masks,
    )

    h_ctx = self.top_down_fusion(h_local, hierarchy, global_ctx, graph)

    return {
        "node_embeddings": h_ctx,
        "song_embedding": global_ctx.song,
        "bar_embeddings": global_ctx.bars,
        "track_embeddings": global_ctx.tracks,
        "ssl_predictions": self.ssl_decoder(h_ctx, global_ctx, decoder_mask_plans),
        "theory_logits": self.theory_heads(h_ctx, global_ctx),
        "aspect_scores": self.aspect_heads(h_ctx, global_ctx),
        "overall_utility": self.preference_head(global_ctx.song),
        "aesthetic_predictions": self.aesthetic_head(global_ctx.song),
    }
```

## 78. Training step

```python
full = batch.graph_full.to(device)
view_a = batch.view_a.to(device)
view_b = batch.view_b.to(device)

with autocast:
    out_a = model(view_a, decoder_mask_plans=batch.decoder_plans_a)
    out_b = model(view_b, decoder_mask_plans=batch.decoder_plans_b)

    with torch.no_grad():
        target_out = target_encoder(full)

    ssl_losses = compute_ssl_losses(out_a, out_b, target_out, batch)
    theory_losses = compute_masked_task_losses(out_a, batch)
    total = loss_router(ssl_losses, theory_losses)
```

## 79. Pairwise preference step

```python
out_a = model(batch.graph_a)
out_b = model(batch.graph_b)
margin = out_a["overall_utility"] - out_b["overall_utility"]
loss = soft_bradley_terry_loss(margin, batch.preference_probability)
```

---

# Part XXI. Tests and quality gates

## 80. Mandatory unit tests

### Schema

- serialization round trip;
- target alignment;
- rational timing;
- schema version mismatch.

### Graph

- edge endpoint validity;
- reverse edges;
- temporal ordering;
- sustained activity;
- mixed dataset graph consistency;
- no target fields in raw input.

### Masking

- masked value inaccessible;
- decoder target preserved;
- deterministic seed;
- hierarchical descendant mask;
- no all-node accidental mask unless configured.

### Augmentation

- transposition round trip;
- drum preservation;
- label equivariance;
- split-group safety.

### Losses

- no target means no gradient/loss for task;
- confidence weighting;
- pair swap invariance;
- stop-gradient target encoder;
- empty optional node types.

### Inference

- unlabeled type-0 MIDI;
- unlabeled multi-track MIDI;
- missing tempo/time signature;
- invalid track names;
- one-track merged arrangement.

## 81. Integration tests

1. tiny multi-dataset preprocessing run;
2. one-batch reconstruction overfit;
3. one-batch theory-task overfit;
4. one-batch pairwise preference overfit;
5. checkpoint save/load;
6. raw MIDI inference after training;
7. V1 smoke run after V2 changes.

## 82. Data leakage tests

- same source-group ID cannot appear in multiple splits;
- transposed copies inherit source group;
- POP909 versions remain grouped;
- Dilemmadata alternative analyses remain grouped;
- PDMX duplicates use provided/hash grouping;
- generated variants of one prompt use group-aware evaluation.

---

# Part XXII. Risks and safeguards

## 83. Label leakage

Risk: gold chord or key values appear in node inputs.

Safeguard:

- `raw_inference_safe` feature flag;
- unit test comparing input feature names against target registry;
- `raw_only` graph mode used in all deployable validation.

## 84. Missing labels interpreted as negatives

Risk: absent cadence annotation becomes `no cadence`.

Safeguard:

- explicit availability mask;
- no implicit default target;
- valid-count logging.

## 85. Dataset dominance

Risk: PDMX overwhelms theory-rich corpora.

Safeguard:

- mixture sampler;
- dataset-specific metrics;
- minimum theory batches per epoch.

## 86. Overfitting to handcrafted corruptions

Risk: model recognizes corruption artifacts rather than music.

Safeguard:

- random/hierarchical masking as primary SSL;
- corruption only as ablation or hard negatives;
- leave-one-corruption-family-out testing;
- real generator outputs and human comparisons.

## 87. Reconstruction equals predictability, not quality

Risk: repetitive simple music receives excellent reconstruction score.

Safeguard:

- do not use reconstruction as final reward;
- train separate preference heads;
- test complexity and diversity bias.

## 88. Gold segmentation unavailable at inference

Risk: harmony/phrase nodes created from annotations.

Safeguard:

- beat/bar candidate slots;
- raw-only encoder;
- optional two-pass predicted segmentation.

## 89. Graph-size explosion

Risk: dense simultaneous note edges and long scores.

Safeguard:

- onset intermediary nodes;
- beat-level sustained edges;
- bar windows;
- graph size filters;
- sparse attention.

## 90. Reward hacking

Risk: GRPO exploits symbolic aesthetic student or critic weaknesses.

Safeguard:

- real audio-teacher audits;
- human evaluation;
- KL control;
- diversity metrics;
- uncertainty-based fallback;
- held-out generators and renderers.

## 91. Audio renderer dependence

Risk: aesthetic teacher labels reflect soundfont rather than composition.

Safeguard:

- renderer metadata;
- fixed reproducible baseline renderer;
- multi-renderer ablation;
- separate symbolic and audio scores.

---

# Part XXIII. Definition of done for Music Critic V2

The V2 project is not complete merely when a large model trains. It is complete when all of the following hold:

1. A generic unlabeled MIDI can be converted to the same raw graph schema used in training.
2. No hidden theory label is required at inference.
3. HookTheory, POP909, Dilemmadata, and PDMX adapters produce versioned canonical data with provenance.
4. The model handles variable track counts.
5. The SSL pipeline supports simple random masks, decoder re-masking, latent prediction, and hierarchical masks.
6. UGMAE-style structure/bootstrap components are optional and ablated.
7. Theory heads are trained using availability masks and evaluated in raw-only mode.
8. The critic produces multiple aspect scores and a pairwise utility.
9. The utility is evaluated on real generator outputs or human comparisons, not only synthetic corruptions.
10. Audio-aesthetic scores are kept separate from symbolic theory scores and may be distilled with renderer provenance.
11. V1 remains runnable until migration is explicitly approved.
12. Every major research claim is supported by an ablation.

---

# Part XXIV. Recommended first three implementation milestones

## Milestone A: data foundation

Implement:

- canonical schema;
- generic MIDI adapter;
- HookTheory V2 adapter;
- POP909 adapter;
- raw graph builder with `track`, `beat`, `onset`, `bar`, `note`, `song`;
- leakage-safe targets and masks.

Do not implement a new large model before this milestone passes.

## Milestone B: minimal hybrid SSL model

Implement:

- type-aware feature encoder;
- local hetero-GNN;
- note/onset/beat/bar and note/track pooling;
- bar+track Transformer;
- random field masking;
- GraphMAE2-style decoder re-masking and song/bar latent prediction.

Train on POP909 plus a small PDMX subset.

## Milestone C: theory transfer

Implement:

- Dilemmadata adapter;
- local key, harmony, cadence, phrase, and track-role heads;
- multi-task loss masks;
- linear probe and fine-tuning evaluation;
- raw-only inference on labeled validation MIDI.

Only after these milestones should the agent add adaptive masking, aesthetic distillation, or GRPO.

---

# Part XXV. Reference papers and resources

Use these as conceptual references. Confirm licenses, dataset versions, and implementation details before automating downloads.

- GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner  
  https://arxiv.org/abs/2304.04779
- Hi-GMAE: Hierarchical Graph Masked Autoencoders  
  https://arxiv.org/abs/2405.10642
- UGMAE: A Unified Framework for Graph Masked Autoencoders  
  https://arxiv.org/abs/2402.08023
- PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing  
  https://arxiv.org/abs/2409.10831
- POP909: A Pop-song Dataset for Music Arrangement Generation  
  https://arxiv.org/abs/2008.07142
- Dilemmadata: On the Interoperability of Heterogeneous Roman Numeral Datasets  
  https://arxiv.org/abs/2606.31595
- Roman Numeral Analysis with Graph Neural Networks / ChordGNN  
  https://arxiv.org/abs/2307.03544
- GraphMuse  
  https://arxiv.org/abs/2407.12671
- AnalysisGNN  
  https://arxiv.org/abs/2509.06654
- MusicBERT  
  https://arxiv.org/abs/2106.05630
- Meta Audiobox Aesthetics  
  https://arxiv.org/abs/2502.05139
- SMART: Tuning a Symbolic Music Generation System with an Audio Domain Aesthetic Reward  
  https://arxiv.org/abs/2504.16839
- Text2MIDI  
  https://arxiv.org/abs/2412.16526
- Text2MIDI-InferAlign  
  https://arxiv.org/abs/2505.12669
- MuseCoco  
  https://arxiv.org/abs/2306.00110

---

# Final instruction to Codex

Begin with **Phase 0**, inspect the current repository and tests, then produce a concrete file-by-file plan for **Phases 1–3 only**. Do not start by implementing the full graph Transformer. The first code changes must establish the canonical schema, raw MIDI inference parity, target masks, and the new track-aware graph. Every subsequent model component depends on those interfaces.
