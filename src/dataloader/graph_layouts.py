"""Feature layouts and constants for hierarchical teacher graphs."""

from collections import OrderedDict

DEFAULT_NUM_BEATS = 4.0
DEFAULT_BEAT_UNIT = 1.0
DEFAULT_BPM = 0.0
DEFAULT_END_BEAT = 1.0

# Explicit valid id sets aligned with the current metadata/spec mappings.
# These are intentionally defined as ids rather than raw symbolic values so
# corruption operates in the same encoded space as teacher_encoded.json.
VALID_ID_SETS = {
    "note_sd_id": tuple(range(4, 26)),
    "chord_root_id": tuple(range(1, 9)),
    "chord_type_id": tuple(range(1, 6)),
    "chord_applied_id": tuple(range(1, 9)),
    "chord_borrowed_kind_id": tuple(range(2, 6)),
}

NOTE_LAYOUT = OrderedDict([
    ("sd_id", 0),
    ("octave_id", 1),
    ("is_rest", 2),
    ("beat", 3),
    ("duration", 4),
    ("bar_index", 5),
    ("pos_in_bar", 6),
])

CHORD_COMPONENT_SIZES = OrderedDict([
    ("adds_vec", 6),
    ("omits_vec", 2),
    ("suspensions_vec", 2),
    ("alterations_vec", 6),
    ("borrowed_pcset_vec", 12),
])

CHORD_LAYOUT = OrderedDict()
_chord_cursor = 0
for _field in [
    "root_id",
    "type_id",
    "inversion_id",
    "applied_id",
    "borrowed_kind_id",
    "borrowed_mode_name_id",
]:
    CHORD_LAYOUT[_field] = _chord_cursor
    _chord_cursor += 1
for _field, _size in CHORD_COMPONENT_SIZES.items():
    CHORD_LAYOUT[_field] = slice(_chord_cursor, _chord_cursor + _size)
    _chord_cursor += _size
for _field in ["is_rest", "beat", "duration", "bar_index", "pos_in_bar"]:
    CHORD_LAYOUT[_field] = _chord_cursor
    _chord_cursor += 1

ONSET_LAYOUT = OrderedDict([
    ("beat", 0),
    ("bar_index", 1),
    ("pos_in_bar", 2),
    ("n_notes_starting_here", 3),
    ("n_chords_starting_here", 4),
])

BAR_LAYOUT = OrderedDict([
    ("bar_index", 0),
    ("bar_start_beat", 1),
    ("bar_end_beat", 2),
    ("n_notes_in_bar", 3),
    ("n_chords_in_bar", 4),
    ("n_onsets_in_bar", 5),
])

SONG_LAYOUT = OrderedDict([
    ("main_key_tonic_pc_id", 0),
    ("main_key_scale_id", 1),
    ("main_num_beats_id", 2),
    ("main_beat_unit_id", 3),
    ("main_bpm", 4),
    ("end_beat", 5),
])

MASKABLE_FIELDS = {
    "note": ["sd_id", "octave_id"],
    "chord": ["root_id", "type_id", "applied_id", "borrowed_kind_id"],
}

PRIMARY_MASK_FIELDS = {
    "note": ["sd_id"],
    "chord": ["root_id", "type_id", "applied_id", "borrowed_kind_id"],
}

NODE_LAYOUTS = {
    "song": SONG_LAYOUT,
    "bar": BAR_LAYOUT,
    "onset": ONSET_LAYOUT,
    "note": NOTE_LAYOUT,
    "chord": CHORD_LAYOUT,
}

NODE_DIMS = {
    "song": len(SONG_LAYOUT),
    "bar": len(BAR_LAYOUT),
    "onset": len(ONSET_LAYOUT),
    "note": len(NOTE_LAYOUT),
    "chord": _chord_cursor,
}
