# src/dataloader/utils_graph.py
import copy
import random
from typing import Dict, Iterable, List, Tuple

import torch
from torch_geometric.data import HeteroData

from .graph_layouts import (
    CHORD_COMPONENT_SIZES,
    CHORD_LAYOUT,
    DEFAULT_BEAT_UNIT,
    DEFAULT_BPM,
    DEFAULT_END_BEAT,
    DEFAULT_NUM_BEATS,
    MASKABLE_FIELDS,
    NODE_DIMS,
    NOTE_LAYOUT,
    PRIMARY_MASK_FIELDS,
    VALID_ID_SETS,
)


MANDATORY_NODE_TYPES = ("song", "bar", "onset", "note", "chord")
MANDATORY_EDGE_TYPES = (
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


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default=0):
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _ensure_node_storage(graph: HeteroData, node_type: str, dim: int, rows: List[List[float]]):
    if rows:
        graph[node_type].x = torch.tensor(rows, dtype=torch.float)
    else:
        graph[node_type].x = torch.empty((0, dim), dtype=torch.float)


def _ensure_edge_storage(graph: HeteroData, edge_type: Tuple[str, str, str], pairs: Iterable[Tuple[int, int]]):
    pairs = list(pairs)
    if pairs:
        graph[edge_type].edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    else:
        graph[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)


def _sorted_events(events: List[dict]) -> List[dict]:
    return sorted(events, key=lambda item: (_safe_float(item.get("beat"), 0.0), _safe_float(item.get("duration"), 0.0)))


def _sequence_pairs(num_nodes: int) -> List[Tuple[int, int]]:
    if num_nodes <= 1:
        return []
    return [(idx, idx + 1) for idx in range(num_nodes - 1)]


def _compute_bar_index(beat: float, num_beats: float) -> int:
    if num_beats <= 0:
        num_beats = DEFAULT_NUM_BEATS
    beat = max(1.0, beat)
    return int((beat - 1.0) // num_beats)


def _bar_start(bar_index: int, num_beats: float) -> float:
    return 1.0 + bar_index * num_beats


def _pos_in_bar(beat: float, num_beats: float) -> float:
    bar_index = _compute_bar_index(beat, num_beats)
    return beat - _bar_start(bar_index, num_beats)


def _song_meta(song_obj: dict) -> dict:
    return song_obj.get("meta", {})


def _infer_end_beat(song_obj: dict, notes: List[dict], chords: List[dict], meta: dict) -> float:
    candidate = _safe_float(meta.get("end_beat"), DEFAULT_END_BEAT)
    for event in notes + chords:
        beat = _safe_float(event.get("beat"), 1.0)
        duration = max(0.0, _safe_float(event.get("duration"), 0.0))
        candidate = max(candidate, beat, beat + duration)
    return max(1.0, candidate)


def _bar_count(end_beat: float, num_beats: float) -> int:
    if num_beats <= 0:
        num_beats = DEFAULT_NUM_BEATS
    return max(1, int((max(1.0, end_beat) - 1.0) // num_beats) + 1)


def _multi_hot_field(chord: dict, field_name: str, size: int) -> List[float]:
    values = chord.get(field_name) or []
    vector = list(values[:size])
    if len(vector) < size:
        vector.extend([0] * (size - len(vector)))
    return [float(v) for v in vector]


def build_graph_from_encoded(song_obj):
    """Build a hierarchical HeteroData graph from a teacher_encoded song object."""
    graph = HeteroData()
    meta = _song_meta(song_obj)
    notes = _sorted_events(song_obj.get("melody", []))
    chords = _sorted_events(song_obj.get("chords", []))

    num_beats = _safe_float(meta.get("main_num_beats"), DEFAULT_NUM_BEATS)
    beat_unit = _safe_float(meta.get("main_beat_unit"), DEFAULT_BEAT_UNIT)
    bpm = _safe_float(meta.get("main_bpm"), DEFAULT_BPM)
    end_beat = _infer_end_beat(song_obj, notes, chords, meta)
    n_bars = _bar_count(end_beat, num_beats)

    # Song node: use only encoded ids for categorical meter/key fields.
    song_row = [[
        float(_safe_int(meta.get("main_key_tonic_pc_id"), 0)),
        float(_safe_int(meta.get("main_key_scale_id"), 0)),
        float(_safe_int(meta.get("main_num_beats_id"), 0)),
        float(_safe_int(meta.get("main_beat_unit_id"), 0)),
        bpm,
        end_beat,
    ]]
    _ensure_node_storage(graph, "song", NODE_DIMS["song"], song_row)

    # Precompute event-to-bar mappings.
    note_bar_indices = []
    chord_bar_indices = []
    notes_per_bar = [0 for _ in range(n_bars)]
    chords_per_bar = [0 for _ in range(n_bars)]
    for note in notes:
        bar_index = min(_compute_bar_index(_safe_float(note.get("beat"), 1.0), num_beats), n_bars - 1)
        note_bar_indices.append(bar_index)
        notes_per_bar[bar_index] += 1
    for chord in chords:
        bar_index = min(_compute_bar_index(_safe_float(chord.get("beat"), 1.0), num_beats), n_bars - 1)
        chord_bar_indices.append(bar_index)
        chords_per_bar[bar_index] += 1

    # Onset nodes from unique note/chord start beats.
    onset_beats = sorted({
        _safe_float(item.get("beat"), 1.0)
        for item in notes + chords
        if item.get("beat") is not None
    })
    onset_index_by_beat = {beat: idx for idx, beat in enumerate(onset_beats)}
    onsets_per_bar = [0 for _ in range(n_bars)]
    notes_per_onset = {beat: 0 for beat in onset_beats}
    chords_per_onset = {beat: 0 for beat in onset_beats}
    for note in notes:
        beat = _safe_float(note.get("beat"), 1.0)
        if beat in notes_per_onset:
            notes_per_onset[beat] += 1
    for chord in chords:
        beat = _safe_float(chord.get("beat"), 1.0)
        if beat in chords_per_onset:
            chords_per_onset[beat] += 1

    onset_rows = []
    onset_bar_indices = []
    for beat in onset_beats:
        bar_index = min(_compute_bar_index(beat, num_beats), n_bars - 1)
        onset_bar_indices.append(bar_index)
        onsets_per_bar[bar_index] += 1
        onset_rows.append([
            beat,
            float(bar_index),
            _pos_in_bar(beat, num_beats),
            float(notes_per_onset[beat]),
            float(chords_per_onset[beat]),
        ])
    _ensure_node_storage(graph, "onset", NODE_DIMS["onset"], onset_rows)

    # Bar nodes after onset counts are known.
    bar_rows = []
    for bar_index in range(n_bars):
        start = _bar_start(bar_index, num_beats)
        bar_rows.append([
            float(bar_index),
            start,
            start + num_beats,
            float(notes_per_bar[bar_index]),
            float(chords_per_bar[bar_index]),
            float(onsets_per_bar[bar_index]),
        ])
    _ensure_node_storage(graph, "bar", NODE_DIMS["bar"], bar_rows)

    # Note nodes.
    note_rows = []
    for note, bar_index in zip(notes, note_bar_indices):
        beat = _safe_float(note.get("beat"), 1.0)
        note_rows.append([
            float(_safe_int(note.get("sd_id"), 0)),
            float(_safe_int(note.get("octave_id"), 0)),
            float(_safe_int(note.get("is_rest"), 0)),
            beat,
            _safe_float(note.get("duration"), 0.0),
            float(bar_index),
            _pos_in_bar(beat, num_beats),
        ])
    _ensure_node_storage(graph, "note", NODE_DIMS["note"], note_rows)

    # Chord nodes.
    chord_rows = []
    for chord, bar_index in zip(chords, chord_bar_indices):
        beat = _safe_float(chord.get("beat"), 1.0)
        row = [
            float(_safe_int(chord.get("root_id"), 0)),
            float(_safe_int(chord.get("type_id"), 0)),
            float(_safe_int(chord.get("inversion_id"), 0)),
            float(_safe_int(chord.get("applied_id"), 0)),
            float(_safe_int(chord.get("borrowed_kind_id"), 0)),
            float(_safe_int(chord.get("borrowed_mode_name_id"), 0)),
        ]
        for field_name, size in CHORD_COMPONENT_SIZES.items():
            row.extend(_multi_hot_field(chord, field_name, size))
        row.extend([
            float(_safe_int(chord.get("is_rest"), 0)),
            beat,
            _safe_float(chord.get("duration"), 0.0),
            float(bar_index),
            _pos_in_bar(beat, num_beats),
        ])
        chord_rows.append(row)
    _ensure_node_storage(graph, "chord", NODE_DIMS["chord"], chord_rows)

    # Sequence edges.
    _ensure_edge_storage(graph, ("song", "contains_bar", "bar"), [(0, idx) for idx in range(n_bars)])
    _ensure_edge_storage(graph, ("bar", "next_bar", "bar"), _sequence_pairs(n_bars))
    _ensure_edge_storage(
        graph,
        ("bar", "contains_onset", "onset"),
        [(bar_index, onset_idx) for onset_idx, bar_index in enumerate(onset_bar_indices)],
    )
    _ensure_edge_storage(graph, ("onset", "next_onset", "onset"), _sequence_pairs(len(onset_beats)))
    _ensure_edge_storage(graph, ("note", "next_note", "note"), _sequence_pairs(len(notes)))
    _ensure_edge_storage(graph, ("chord", "next_chord", "chord"), _sequence_pairs(len(chords)))

    # Incidence edges.
    note_onset_pairs = []
    for note_idx, note in enumerate(notes):
        onset_idx = onset_index_by_beat.get(_safe_float(note.get("beat"), 1.0))
        if onset_idx is not None:
            note_onset_pairs.append((onset_idx, note_idx))
    _ensure_edge_storage(graph, ("onset", "starts_note", "note"), note_onset_pairs)

    chord_onset_pairs = []
    for chord_idx, chord in enumerate(chords):
        onset_idx = onset_index_by_beat.get(_safe_float(chord.get("beat"), 1.0))
        if onset_idx is not None:
            chord_onset_pairs.append((onset_idx, chord_idx))
    _ensure_edge_storage(graph, ("onset", "starts_chord", "chord"), chord_onset_pairs)

    # Harmonic coverage edges.
    cover_pairs = []
    for chord_idx, chord in enumerate(chords):
        chord_start = _safe_float(chord.get("beat"), 1.0)
        chord_end = chord_start + max(0.0, _safe_float(chord.get("duration"), 0.0))
        for note_idx, note in enumerate(notes):
            note_beat = _safe_float(note.get("beat"), 1.0)
            if chord_start <= note_beat < chord_end:
                cover_pairs.append((chord_idx, note_idx))
    _ensure_edge_storage(graph, ("chord", "covers_note", "note"), cover_pairs)

    for node_type in MANDATORY_NODE_TYPES:
        graph[node_type].num_nodes = graph[node_type].x.size(0)
    for edge_type in MANDATORY_EDGE_TYPES:
        _ = graph[edge_type].edge_index

    # Debug convenience only; downstream training should use node tensors.
    graph.graph_metadata = {
        "song_id": song_obj.get("song_id"),
        "num_beats": num_beats,
        "beat_unit": beat_unit,
        "bpm": bpm,
        "end_beat": end_beat,
    }
    return graph


def mask_graph(
    graph: HeteroData,
    mask_prob: float = 0.15,
    min_nodes_to_mask: int = 1,
    optional_mask_field_prob: float = 0.5,
):
    """Field-aware masking for note/chord nodes used in SSL reconstruction."""
    masked_graph = copy.deepcopy(graph)
    masked_labels: Dict[str, dict] = {}

    for node_type, fields in MASKABLE_FIELDS.items():
        x = masked_graph[node_type].x
        if x.numel() == 0:
            masked_labels[node_type] = {
                "indices": torch.empty((0,), dtype=torch.long),
                "field_names": [],
                "target_values": {},
            }
            continue

        num_nodes = x.size(0)
        num_mask = max(int(min_nodes_to_mask), int(round(num_nodes * mask_prob)))
        num_mask = min(num_nodes, num_mask)
        indices = torch.tensor(sorted(random.sample(range(num_nodes), num_mask)), dtype=torch.long)

        selected_fields = list(PRIMARY_MASK_FIELDS.get(node_type, fields))
        optional_fields = [field for field in fields if field not in selected_fields]
        for field in optional_fields:
            if random.random() < optional_mask_field_prob:
                selected_fields.append(field)
        selected_fields = list(dict.fromkeys(selected_fields))

        target_values = {}
        for field in selected_fields:
            column = NOTE_LAYOUT[field] if node_type == "note" else CHORD_LAYOUT[field]
            target_values[field] = x[indices, column].clone()
            x[indices, column] = 0.0

        masked_graph[node_type].x = x
        masked_labels[node_type] = {
            "indices": indices,
            "field_names": selected_fields,
            "target_values": target_values,
        }

    return masked_graph, masked_labels


def _replace_with_valid_ids(x: torch.Tensor, column: int, valid_ids: Tuple[int, ...]) -> bool:
    if x.size(0) == 0:
        return False
    idx = random.randrange(x.size(0))
    current = int(x[idx, column].item())
    candidates = [value for value in valid_ids if value != current]
    if not candidates:
        return False
    x[idx, column] = float(random.choice(candidates))
    return True


def _swap_neighbor_rows(x: torch.Tensor) -> int | None:
    if x.size(0) < 2:
        return None
    idx = random.randrange(x.size(0) - 1)
    swapped = x.clone()
    swapped[idx], swapped[idx + 1] = x[idx + 1].clone(), x[idx].clone()
    x.copy_(swapped)
    return idx


def _onset_mismatch(corrupted: HeteroData) -> tuple[bool, int | None, int | None]:
    note_x = corrupted["note"].x
    onset_x = corrupted["onset"].x
    starts_note_edges = corrupted[("onset", "starts_note", "note")].edge_index
    if note_x.size(0) == 0 or onset_x.size(0) < 2 or starts_note_edges.size(1) == 0:
        return False, None, None

    edge_idx = random.randrange(starts_note_edges.size(1))
    src_onset = int(starts_note_edges[0, edge_idx].item())
    note_idx = int(starts_note_edges[1, edge_idx].item())
    if src_onset + 1 < onset_x.size(0):
        dst_onset = src_onset + 1
    elif src_onset - 1 >= 0:
        dst_onset = src_onset - 1
    else:
        return False, None, None
    if dst_onset == src_onset:
        return False, None, None

    starts_note_edges[0, edge_idx] = dst_onset
    corrupted[("onset", "starts_note", "note")].edge_index = starts_note_edges

    note_x[note_idx, NOTE_LAYOUT["beat"]] = onset_x[dst_onset, 0]
    note_x[note_idx, NOTE_LAYOUT["bar_index"]] = onset_x[dst_onset, 1]
    note_x[note_idx, NOTE_LAYOUT["pos_in_bar"]] = onset_x[dst_onset, 2]
    corrupted["note"].x = note_x
    return True, note_idx, dst_onset


def corrupt_graph(graph: HeteroData, corruption_modes: Tuple[str, ...] | None = None):
    """Create a semantically corrupted copy of the graph for discrimination tasks."""
    corrupted = copy.deepcopy(graph)
    available_corruption_modes = []

    if corrupted["note"].x.size(0) > 0:
        available_corruption_modes.append("note_sd_replacement")
    if corrupted["chord"].x.size(0) > 0:
        available_corruption_modes.extend(["chord_root_replacement", "chord_type_replacement"])
    if corrupted["chord"].x.size(0) > 1:
        available_corruption_modes.append("swap_neighboring_chords")
    if (
        corrupted["onset"].x.size(0) > 1
        and corrupted["note"].x.size(0) > 0
        and corrupted[("onset", "starts_note", "note")].edge_index.size(1) > 0
    ):
        available_corruption_modes.append("onset_mismatch")

    if corruption_modes is not None:
        allowed_modes = set(corruption_modes)
        available_corruption_modes = [mode for mode in available_corruption_modes if mode in allowed_modes]

    if not available_corruption_modes:
        corrupted.corruption_metadata = {
            "mode": "identity",
            "applied": False,
            "note_corrupted_indices": [],
            "chord_corrupted_indices": [],
            "onset_corrupted_indices": [],
        }
        return corrupted

    mode = random.choice(available_corruption_modes)
    applied = False
    note_corrupted_indices: list[int] = []
    chord_corrupted_indices: list[int] = []
    onset_corrupted_indices: list[int] = []
    if mode == "note_sd_replacement":
        before = corrupted["note"].x[:, NOTE_LAYOUT["sd_id"]].clone()
        applied = _replace_with_valid_ids(
            corrupted["note"].x,
            NOTE_LAYOUT["sd_id"],
            VALID_ID_SETS["note_sd_id"],
        )
        if applied:
            changed = torch.nonzero(corrupted["note"].x[:, NOTE_LAYOUT["sd_id"]] != before, as_tuple=False).view(-1)
            note_corrupted_indices.extend(changed.tolist())
    elif mode == "chord_root_replacement":
        before = corrupted["chord"].x[:, CHORD_LAYOUT["root_id"]].clone()
        applied = _replace_with_valid_ids(
            corrupted["chord"].x,
            CHORD_LAYOUT["root_id"],
            VALID_ID_SETS["chord_root_id"],
        )
        if applied:
            changed = torch.nonzero(corrupted["chord"].x[:, CHORD_LAYOUT["root_id"]] != before, as_tuple=False).view(-1)
            chord_corrupted_indices.extend(changed.tolist())
    elif mode == "chord_type_replacement":
        before = corrupted["chord"].x[:, CHORD_LAYOUT["type_id"]].clone()
        applied = _replace_with_valid_ids(
            corrupted["chord"].x,
            CHORD_LAYOUT["type_id"],
            VALID_ID_SETS["chord_type_id"],
        )
        if applied:
            changed = torch.nonzero(corrupted["chord"].x[:, CHORD_LAYOUT["type_id"]] != before, as_tuple=False).view(-1)
            chord_corrupted_indices.extend(changed.tolist())
    elif mode == "swap_neighboring_chords":
        swap_idx = _swap_neighbor_rows(corrupted["chord"].x)
        applied = swap_idx is not None
        if swap_idx is not None:
            chord_corrupted_indices.extend([swap_idx, swap_idx + 1])
    elif mode == "onset_mismatch":
        applied, note_idx, onset_idx = _onset_mismatch(corrupted)
        if note_idx is not None:
            note_corrupted_indices.append(note_idx)
        if onset_idx is not None:
            onset_corrupted_indices.append(onset_idx)

    corrupted.corruption_metadata = {
        "mode": mode,
        "applied": applied,
        "note_corrupted_indices": sorted(set(note_corrupted_indices)),
        "chord_corrupted_indices": sorted(set(chord_corrupted_indices)),
        "onset_corrupted_indices": sorted(set(onset_corrupted_indices)),
    }
    return corrupted


def extract_masked_labels(real_graph, masked_graph):
    """Backward-compatible extraction of masked labels from graph differences."""
    masked_labels = {}
    for node_type in ["note", "chord"]:
        if node_type not in masked_graph.node_types:
            continue
        if real_graph[node_type].x.numel() == 0:
            masked_labels[node_type] = {
                "indices": torch.empty((0,), dtype=torch.long),
                "field_names": [],
                "target_values": {},
            }
            continue
        diff = masked_graph[node_type].x != real_graph[node_type].x
        changed_rows = torch.nonzero(diff.any(dim=1), as_tuple=False).view(-1)
        changed_cols = torch.nonzero(diff.any(dim=0), as_tuple=False).view(-1).tolist()
        layout = NOTE_LAYOUT if node_type == "note" else CHORD_LAYOUT
        field_names = []
        target_values = {}
        for field, column in layout.items():
            if isinstance(column, slice):
                continue
            if column in changed_cols:
                field_names.append(field)
                target_values[field] = real_graph[node_type].x[changed_rows, column].clone()
        masked_labels[node_type] = {
            "indices": changed_rows,
            "field_names": field_names,
            "target_values": target_values,
        }
    return masked_labels
