"""Observer graph schema aligned to teacher-style feature encoding.

Node types:
- song, bar, onset, note, chord

For every node type the graph stores:
- `x_num`: dense numeric float tensor
- `x_cat`: categorical id tensor (`torch.long`) aligned with teacher id-space
- `x`: backward-compatible concatenation `[x_cat.float(), x_num]`

Field provenance summary:
- Input JSON / MIDI meta: tonic/mode/meter/tempo/end_beat
- MIDI melody/chord tracks: notes/chords onsets and durations
- Chord parser: root/type/inversion/mode/added-or-omitted chord components
- Derived: bar_index/pos_in_bar/count statistics
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Teacher-compatible node/edge schema
OBSERVER_NODE_TYPES = ("song", "bar", "onset", "note", "chord")
OBSERVER_EDGE_TYPES = (
    ("song", "contains_bar", "bar"),
    ("bar", "next_bar", "bar"),
    ("bar", "contains_onset", "onset"),
    ("onset", "next_onset", "onset"),
    ("onset", "starts_note", "note"),
    ("onset", "starts_chord", "chord"),
    ("note", "next_note", "note"),
    ("chord", "next_chord", "chord"),
    ("chord", "covers_note", "note"),
)

# Per-node fields, grouped by categorical-id vs dense numeric features.
OBSERVER_CAT_FIELDS = {
    "song": ("main_key_tonic_pc_id", "main_key_scale_id", "main_num_beats_id", "main_beat_unit_id"),
    "bar": tuple(),
    "onset": tuple(),
    "note": ("sd_id", "octave_id"),
    "chord": (
        "root_id",
        "type_id",
        "inversion_id",
        "borrowed_kind_id",
        "borrowed_mode_name_id",
    ),
}

OBSERVER_NUM_FIELDS = {
    "song": ("main_bpm", "end_beat"),
    "bar": ("bar_index", "bar_start_beat", "bar_end_beat", "n_notes_in_bar", "n_chords_in_bar", "n_onsets_in_bar"),
    "onset": ("beat", "bar_index", "pos_in_bar", "n_notes_starting_here", "n_chords_starting_here"),
    "note": ("beat", "duration", "bar_index", "pos_in_bar"),
    "chord": (
        "adds_vec_0",
        "adds_vec_1",
        "adds_vec_2",
        "adds_vec_3",
        "adds_vec_4",
        "adds_vec_5",
        "omits_vec_0",
        "omits_vec_1",
        "suspensions_vec_0",
        "suspensions_vec_1",
        "alterations_vec_0",
        "alterations_vec_1",
        "alterations_vec_2",
        "alterations_vec_3",
        "alterations_vec_4",
        "alterations_vec_5",
        "borrowed_pcset_vec_0",
        "borrowed_pcset_vec_1",
        "borrowed_pcset_vec_2",
        "borrowed_pcset_vec_3",
        "borrowed_pcset_vec_4",
        "borrowed_pcset_vec_5",
        "borrowed_pcset_vec_6",
        "borrowed_pcset_vec_7",
        "borrowed_pcset_vec_8",
        "borrowed_pcset_vec_9",
        "borrowed_pcset_vec_10",
        "borrowed_pcset_vec_11",
        "beat",
        "duration",
        "bar_index",
        "pos_in_bar",
    ),
}

OBSERVER_NODE_DIMS = OrderedDict(
    (
        (node_type, len(OBSERVER_CAT_FIELDS[node_type]) + len(OBSERVER_NUM_FIELDS[node_type]))
        for node_type in OBSERVER_NODE_TYPES
    )
)


def _load_teacher_vocab_sizes() -> dict[str, int]:
    vocabs_dir = Path(__file__).resolve().parents[2] / "metadata" / "vocabs"
    def _max_id(file_name: str) -> int:
        payload = json.loads((vocabs_dir / file_name).read_text(encoding="utf-8"))
        return int(max(payload.values()) + 1)
    return {
        "melody_sd": _max_id("vocab_melody_sd.json"),
        "key_scale": _max_id("vocab_key_scale.json"),
        "borrowed_kind": _max_id("vocab_borrowed_kind.json"),
        "borrowed_mode_name": _max_id("vocab_borrowed_mode_name.json"),
    }


def build_observer_vocab_sizes(theory_ctx: dict[str, Any], spec_global: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return embedding vocab sizes per node type (max id + 1 for each categorical field)."""
    _ = theory_ctx
    vocab_sizes = _load_teacher_vocab_sizes()
    max_ids = {
        "song": (
            len(spec_global["tonic_pc"]["allowed_values"]) + 1,
            vocab_sizes["key_scale"],
            len(spec_global["num_beats"]["allowed_values"]) + 1,
            len(spec_global["beat_unit"]["allowed_values"]) + 1,
        ),
        "bar": tuple(),
        "onset": tuple(),
        "note": (
            vocab_sizes["melody_sd"],
            (spec_global["octave"]["max"] - spec_global["octave"]["min"] + 1) + 1,
        ),
        "chord": (
            len(spec_global["root"]["allowed_values"]) + 1,
            len(spec_global["type"]["allowed_values"]) + 1,
            len(spec_global["inversion"]["allowed_values"]) + 1,
            vocab_sizes["borrowed_kind"],
            vocab_sizes["borrowed_mode_name"],
        ),
    }
    return {node_type: tuple(int(v) for v in values) for node_type, values in max_ids.items()}
