"""Observer utilities."""

from .data_pipeline import (
    ONSET_EPSILON,
    build_bar_events,
    build_observer_chord_events,
    build_observer_graph,
    build_observer_song_record,
    build_onset_events,
    extract_observer_meta,
    extract_observer_note_events,
    load_observer_input_jsonl,
)

from .chord_parser import (
    ChordCandidate,
    build_body_from_tertian_row,
    build_sounding_sonority,
    build_tertian_row,
    extract_harmonic_onsets,
    generate_all_candidates,
    generate_candidates_for_mode_and_degree,
    load_midi_notes,
    predict_chords_for_midi,
    score_candidate,
    select_best_candidates,
)
from .model import ObserverGNN
from .dataset import ObserverDataset
from .schema import (
    OBSERVER_CAT_FIELDS,
    OBSERVER_EDGE_TYPES,
    OBSERVER_NODE_DIMS,
    OBSERVER_NODE_TYPES,
    OBSERVER_NUM_FIELDS,
    build_observer_vocab_sizes,
)

__all__ = [
    "ONSET_EPSILON",
    "build_bar_events",
    "build_observer_chord_events",
    "build_observer_graph",
    "build_observer_song_record",
    "build_onset_events",
    "extract_observer_meta",
    "extract_observer_note_events",
    "load_observer_input_jsonl",
    "ChordCandidate",
    "build_body_from_tertian_row",
    "build_sounding_sonority",
    "build_tertian_row",
    "extract_harmonic_onsets",
    "generate_all_candidates",
    "generate_candidates_for_mode_and_degree",
    "load_midi_notes",
    "predict_chords_for_midi",
    "score_candidate",
    "select_best_candidates",
    "ObserverGNN",
    "ObserverDataset",
    "OBSERVER_NODE_TYPES",
    "OBSERVER_EDGE_TYPES",
    "OBSERVER_CAT_FIELDS",
    "OBSERVER_NUM_FIELDS",
    "OBSERVER_NODE_DIMS",
    "build_observer_vocab_sizes",
]
