"""Song-object-level theory-aware corruptions."""

from __future__ import annotations

import copy
import random
from typing import Callable

from .function_rules import STRICT_TRIPLET_PATTERNS_V1
from .theory_helpers import (
    chord_bass_and_top_pcs,
    chord_body_pcs_ordered,
    chord_implied_bass_pc,
    decode_chord_components,
    decode_inversion_raw,
    decode_root_raw,
    decode_sd_to_chromatic,
    chord_pitch_classes_tertian,
    classify_function_from_root_raw,
    find_covering_chord_index,
    is_strong_note_position,
    select_active_mode_name,
    safe_float,
    try_parse_float,
)

MIDI_MIN_PITCH = 0
MIDI_MAX_PITCH = 127
DEFAULT_EPSILON = 1e-4

STRICT_BENIGN_CORRUPTIONS = [
    "transpose_with_tonic_shift",
    "merge_repeated_melody_notes",
    "split_long_melody_note",
]

NEAR_BENIGN_CORRUPTIONS = [
    "melody_octave_shift",
    "drop_tonic_seventh_on_strong_beat",
]


def _identity_metadata(mode: str) -> dict:
    return {
        "mode": mode,
        "mode_family": "theory_aware",
        "applied": False,
        "corruption_name": mode,
        "corruption_params": {},
        "reason_skipped": None,
        "topology_changed": False,
        "note_corrupted_indices": [],
        "chord_corrupted_indices": [],
        "onset_corrupted_indices": [],
        "n_notes_modified": 0,
        "n_chords_modified": 0,
        "details": {},
    }


def _onset_grid(song_obj: dict) -> list[float]:
    beats = set()
    for event in song_obj.get("melody", []) + song_obj.get("chords", []):
        beat = try_parse_float(event.get("beat"))
        if beat is not None:
            beats.add(beat)
    return sorted(beats)


def _collect_post_onset_indices_for_metadata(post_grid: list[float], beats: set[float]) -> list[int]:
    """Collect onset indices only from corrupted/post onset grid.

    This keeps metadata indices aligned with build_graph_from_encoded(song_corrupted),
    which also builds onset nodes from the corrupted song only.
    """
    index_map = {beat: idx for idx, beat in enumerate(post_grid)}
    indices = [index_map[beat] for beat in beats if beat in index_map]
    return sorted(set(indices))


def _pick_new_sd_id(exclude_pcs: set[int], include_pcs: set[int] | None, theory_ctx: dict, rng: random.Random) -> int | None:
    candidates = []
    for sd_id, token in theory_ctx["sd_id_to_token"].items():
        if token.startswith("<"):
            continue
        pc = theory_ctx["sd_token_to_chromatic"].get(token)
        if pc is None or pc in exclude_pcs:
            continue
        if include_pcs is not None and pc not in include_pcs:
            continue
        candidates.append(sd_id)
    return int(rng.choice(candidates)) if candidates else None


def _pick_sd_id_for_pc(
    target_pc: int,
    theory_ctx: dict,
    rng: random.Random,
    exclude_sd_ids: set[int] | None = None,
) -> int | None:
    exclude_sd_ids = exclude_sd_ids or set()
    candidates = []
    for sd_id, token in theory_ctx["sd_id_to_token"].items():
        if token.startswith("<") or int(sd_id) in exclude_sd_ids:
            continue
        if theory_ctx["sd_token_to_chromatic"].get(token) == target_pc % 12:
            candidates.append(int(sd_id))
    return int(rng.choice(candidates)) if candidates else None


def _extract_tonic_pc(meta: dict) -> tuple[str | None, int | None]:
    for key in ("main_key_tonic_pc", "tonic_pc"):
        raw = meta.get(key)
        if raw is None:
            continue
        try:
            return key, int(raw) % 12
        except (TypeError, ValueError):
            continue
    return None, None


def _tonic_pc_to_id(tonic_pc: int) -> int:
    return (int(tonic_pc) % 12) + 1


def _transpose_tonic_fields(song_obj: dict, semitones: int) -> tuple[bool, dict]:
    meta = song_obj.get("meta", {})
    changed = False
    details: dict[str, int] = {}

    tonic_key, tonic_pc = _extract_tonic_pc(meta)
    if tonic_key is not None and tonic_pc is not None:
        new_pc = (int(tonic_pc) + int(semitones)) % 12
        meta[tonic_key] = new_pc
        details["original_tonic_pc"] = int(tonic_pc)
        details["new_tonic_pc"] = int(new_pc)
        changed = True

    if "main_key_tonic_pc_id" in meta and meta.get("main_key_tonic_pc_id") is not None:
        try:
            old_id = int(meta.get("main_key_tonic_pc_id"))
            old_pc = (old_id - 1) % 12
            new_pc = (old_pc + int(semitones)) % 12
            meta["main_key_tonic_pc_id"] = _tonic_pc_to_id(new_pc)
            if "original_tonic_pc" not in details:
                details["original_tonic_pc"] = old_pc
                details["new_tonic_pc"] = new_pc
            changed = True
        except (TypeError, ValueError):
            pass

    for region_key in ("key_regions", "keys", "key_changes"):
        regions = song_obj.get(region_key)
        if not isinstance(regions, list):
            continue
        for region in regions:
            if not isinstance(region, dict):
                continue
            if region.get("tonic_pc") is not None:
                try:
                    region["tonic_pc"] = (int(region["tonic_pc"]) + int(semitones)) % 12
                    changed = True
                except (TypeError, ValueError):
                    pass
            if region.get("tonic_pc_id") is not None:
                try:
                    old_id = int(region["tonic_pc_id"])
                    old_pc = (old_id - 1) % 12
                    region["tonic_pc_id"] = _tonic_pc_to_id(old_pc + int(semitones))
                    changed = True
                except (TypeError, ValueError):
                    pass
    return changed, details


def _melody_events(song_obj: dict) -> tuple[str | None, list[dict]]:
    melody = song_obj.get("melody")
    if isinstance(melody, list):
        return "melody", melody
    notes = song_obj.get("notes")
    if isinstance(notes, list):
        return "notes", notes
    return None, []


def _note_interval(note: dict) -> tuple[float | None, float | None]:
    start = try_parse_float(note.get("beat"))
    duration = try_parse_float(note.get("duration"))
    if duration is None:
        duration = try_parse_float(note.get("duration_beats"))
    if start is not None and duration is not None:
        return start, start + duration
    onset = try_parse_float(note.get("onset_time"))
    offset = try_parse_float(note.get("offset_time"))
    return onset, offset


def _set_note_interval(note: dict, start: float, end: float):
    if "beat" in note:
        note["beat"] = start
    duration = max(0.0, end - start)
    if "duration" in note:
        note["duration"] = duration
    if "duration_beats" in note:
        note["duration_beats"] = duration
    if "onset_time" in note:
        note["onset_time"] = start
    if "offset_time" in note:
        note["offset_time"] = end


def _intervals_overlap(start_a: float | None, end_a: float | None, start_b: float | None, end_b: float | None, eps: float = DEFAULT_EPSILON) -> bool:
    if None in (start_a, end_a, start_b, end_b):
        return False
    return float(start_a) < float(end_b) - eps and float(start_b) < float(end_a) - eps


def _collect_overlapping_melody_indices(song_obj: dict, chord: dict) -> list[int]:
    _, melody = _melody_events(song_obj)
    if not melody:
        return []
    chord_start, chord_end = _note_interval(chord)
    overlapping = []
    for note_idx, note in enumerate(melody):
        if int(note.get("is_rest", 0)) == 1:
            continue
        note_start, note_end = _note_interval(note)
        if _intervals_overlap(chord_start, chord_end, note_start, note_end):
            overlapping.append(note_idx)
    return overlapping


def _decode_total_chord_pcs(song_obj: dict, chord: dict, theory_ctx: dict) -> set[int]:
    decoded = decode_chord_components(song_obj, chord, theory_ctx)
    if decoded is None:
        return set()
    return {int(pc) % 12 for pc in decoded["body_pcs"] + decoded["add_pcs"]}


def _total_pcs_from_decoded(decoded: dict | None) -> set[int]:
    if decoded is None:
        return set()
    return {int(pc) % 12 for pc in decoded.get("body_pcs", []) + decoded.get("add_pcs", [])}


def _pad_bitvec(values: list[int] | None, size: int) -> list[int]:
    vec = list(values or [])
    if len(vec) < size:
        vec.extend([0] * (size - len(vec)))
    return [int(v) for v in vec[:size]]


def _iter_chord_melody_contexts(song_obj: dict, chord: dict, melody: list[dict], theory_ctx: dict) -> list[dict]:
    chord_start = try_parse_float(chord.get("beat"))
    overlapping_note_indices = _collect_overlapping_melody_indices(song_obj, chord)
    if not overlapping_note_indices:
        return []
    same_onset_indices = [
        note_idx
        for note_idx in overlapping_note_indices
        if abs(safe_float(melody[note_idx].get("beat"), -999.0) - safe_float(chord_start, -999.0)) <= DEFAULT_EPSILON
    ]
    note_indices = same_onset_indices or overlapping_note_indices
    contexts = []
    for note_idx in note_indices:
        note = melody[note_idx]
        if int(note.get("is_rest", 0)) == 1:
            continue
        melody_pc = decode_sd_to_chromatic(int(note.get("sd_id", 0)), theory_ctx)
        if melody_pc is None:
            continue
        contexts.append({
            "note_idx": note_idx,
            "melody_pc": int(melody_pc) % 12,
            "same_onset": abs(safe_float(note.get("beat"), -999.0) - safe_float(chord_start, -999.0)) <= DEFAULT_EPSILON,
            "is_strong": is_strong_note_position(note, song_obj),
        })
    return contexts


def _base_context_score(context: dict) -> float:
    score = 0.0
    if context.get("same_onset"):
        score += 4.0
    if context.get("is_strong"):
        score += 2.0
    return score


def _pc_distance(a: int | None, b: int | None) -> int:
    if a is None or b is None:
        return 0
    delta = abs(int(a) - int(b)) % 12
    return min(delta, 12 - delta)


def _chord_pos_in_bar(chord: dict, song_obj: dict) -> float | None:
    pos = try_parse_float(chord.get("pos_in_bar"))
    if pos is not None:
        return pos
    beat = try_parse_float(chord.get("beat"))
    if beat is None:
        return None
    num_beats = safe_float(song_obj.get("meta", {}).get("main_num_beats", 4.0), 4.0)
    bar_idx = int((beat - 1.0) // num_beats)
    bar_start = 1.0 + bar_idx * num_beats
    return beat - bar_start


def _is_strong_chord_position(chord: dict, song_obj: dict) -> bool:
    pos = _chord_pos_in_bar(chord, song_obj)
    if pos is None:
        return False
    num_beats = safe_float(song_obj.get("meta", {}).get("main_num_beats", 4.0), 4.0)
    if abs(pos) < 1e-6:
        return True
    if abs(num_beats - 4.0) < 1e-6 and abs(pos - 2.0) < 1e-6:
        return True
    return False


def _neighbor_implied_bass(song_obj: dict, chords: list[dict], start_idx: int, step: int, theory_ctx: dict) -> int | None:
    idx = start_idx + step
    while 0 <= idx < len(chords):
        chord = chords[idx]
        if int(chord.get("is_rest", 0)) == 0:
            bass_pc = chord_implied_bass_pc(song_obj, chord, theory_ctx)
            if bass_pc is not None:
                return bass_pc
        idx += step
    return None


def _inversion_instability_penalty(inversion_raw: int, body_len: int) -> float:
    max_inv = min(3, max(0, body_len - 1))
    inversion_raw = max(0, min(int(inversion_raw), max_inv))
    if max_inv <= 2:
        penalty_map = {0: 0.0, 1: 1.0, 2: 3.0}
    else:
        penalty_map = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}
    return float(penalty_map.get(inversion_raw, 0.0))


def _strongbeat_inversion_penalty(inversion_raw: int, strong_position: bool) -> float:
    if not strong_position:
        penalty_map = {0: 0.0, 1: 0.5, 2: 1.5, 3: 2.5}
    else:
        penalty_map = {0: 0.0, 1: 1.0, 2: 3.0, 3: 4.0}
    return float(penalty_map.get(int(inversion_raw), penalty_map[max(penalty_map)]))


def _bass_continuity_badness(
    candidate_bass_pc: int,
    inversion_raw: int,
    body_len: int,
    prev_bass_pc: int | None,
    next_bass_pc: int | None,
    strong_position: bool,
) -> float:
    continuity_penalty = float(_pc_distance(prev_bass_pc, candidate_bass_pc) + _pc_distance(candidate_bass_pc, next_bass_pc))
    return (
        2.0 * _inversion_instability_penalty(inversion_raw, body_len)
        + 2.5 * _strongbeat_inversion_penalty(inversion_raw, strong_position)
        + 1.5 * continuity_penalty
    )


def _has_pitch_in_midi_range(events: list[dict], shift: int) -> bool:
    for event in events:
        if "pitch" not in event:
            continue
        try:
            new_pitch = int(event["pitch"]) + int(shift)
        except (TypeError, ValueError):
            return False
        if new_pitch < MIDI_MIN_PITCH or new_pitch > MIDI_MAX_PITCH:
            return False
    return True


def _sync_pos_in_bar_if_present(event: dict, song_obj: dict) -> None:
    if "pos_in_bar" not in event:
        return
    beat = try_parse_float(event.get("beat"))
    if beat is None:
        return
    num_beats = safe_float(song_obj.get("meta", {}).get("main_num_beats", 4.0), 4.0)
    bar_idx = int((beat - 1.0) // num_beats)
    bar_start = 1.0 + bar_idx * num_beats
    event["pos_in_bar"] = beat - bar_start


def _restify_note(note: dict) -> None:
    note["is_rest"] = 1
    if "sd_id" in note:
        note["sd_id"] = 0
    if "octave_id" in note:
        note["octave_id"] = 0


def _restify_chord(chord: dict, *, zero_duration: bool = False) -> None:
    chord["is_rest"] = 1
    for field_name in ("root_id", "type_id", "inversion_id", "applied_id", "borrowed_kind_id", "borrowed_mode_name_id"):
        if field_name in chord:
            chord[field_name] = 0
    for field_name in ("adds_vec", "suspensions_vec", "omits_vec", "alterations_vec", "borrowed_pcset_vec"):
        if isinstance(chord.get(field_name), list):
            chord[field_name] = [0] * len(chord[field_name])
    if "root_degree_raw" in chord:
        chord["root_degree_raw"] = None
    if "type_raw" in chord:
        chord["type_raw"] = None
    if isinstance(chord.get("add_degrees"), list):
        chord["add_degrees"] = []
    if zero_duration:
        start, _ = _note_interval(chord)
        if start is not None:
            _set_note_interval(chord, float(start), float(start))
        elif "duration" in chord:
            chord["duration"] = 0.0
        if "duration_beats" in chord:
            chord["duration_beats"] = 0.0


def _duration_scale_factors(corruption_cfg: dict, specific_key: str) -> list[float]:
    raw_values = corruption_cfg.get(specific_key)
    if raw_values is None:
        raw_values = corruption_cfg.get("duration_scale_factors", [0.5, 1.5])
    factors: list[float] = []
    for raw_value in list(raw_values or []):
        factor = try_parse_float(raw_value)
        if factor is None or factor <= 0.0:
            continue
        factors.append(float(factor))
    return factors


def _section_span_bounds(span: dict) -> tuple[float | None, float | None]:
    start = try_parse_float(span.get("target_start_beat"))
    if start is None:
        start = try_parse_float(span.get("start_beat"))
    end = try_parse_float(span.get("target_end_beat"))
    if end is None:
        end = try_parse_float(span.get("end_beat"))
    if start is None or end is None or end <= start:
        return None, None
    return float(start), float(end)


def _normalized_section_spans(song_obj: dict) -> list[dict]:
    meta = song_obj.get("meta", {})
    spans = meta.get("section_spans") if isinstance(meta, dict) else None
    if not isinstance(spans, list):
        return []

    normalized = []
    for fallback_idx, span in enumerate(spans):
        if not isinstance(span, dict):
            continue
        start, end = _section_span_bounds(span)
        if start is None or end is None:
            continue
        item = copy.deepcopy(span)
        item["_original_section_list_index"] = fallback_idx
        item["_start_beat"] = start
        item["_end_beat"] = end
        item["_duration_beats"] = end - start
        normalized.append(item)
    normalized.sort(key=lambda x: (x["_start_beat"], x["_end_beat"], int(x.get("section_index", 0) or 0)))
    return normalized


def _section_label_for_metadata(span: dict) -> str:
    label = span.get("label")
    if isinstance(label, str) and label:
        return label
    labels = span.get("labels")
    if isinstance(labels, list) and labels:
        return "+".join(str(x) for x in labels)
    return "unknown"


def _beat_in_interval(beat: float | None, start: float, end: float, *, is_last: bool = False) -> bool:
    if beat is None:
        return False
    if beat < start - DEFAULT_EPSILON:
        return False
    if is_last:
        return beat <= end + DEFAULT_EPSILON
    return beat < end - DEFAULT_EPSILON


def _event_beat(event: dict) -> float | None:
    return try_parse_float(event.get("beat"))


def _shift_event_beat(event: dict, offset_beats: float, song_obj: dict) -> None:
    beat = _event_beat(event)
    if beat is None:
        return
    event["beat"] = float(beat + offset_beats)
    _sync_pos_in_bar_if_present(event, song_obj)


def _sort_events_in_place(song_obj: dict) -> None:
    for key in ("melody", "chords"):
        events = song_obj.get(key)
        if isinstance(events, list):
            events.sort(key=lambda event: (safe_float(event.get("beat"), 1.0), safe_float(event.get("duration"), 0.0)))
    meta = song_obj.get("meta", {})
    if isinstance(meta, dict):
        for key in ("key_regions", "tempo_regions", "meter_regions"):
            regions = meta.get(key)
            if isinstance(regions, list):
                regions.sort(key=lambda event: safe_float(event.get("beat"), 1.0))


def _section_blocks(song_obj: dict, spans: list[dict]) -> list[dict]:
    blocks = []
    meta = song_obj.get("meta", {})
    melody = song_obj.get("melody", []) if isinstance(song_obj.get("melody"), list) else []
    chords = song_obj.get("chords", []) if isinstance(song_obj.get("chords"), list) else []
    region_lists = {
        key: meta.get(key, []) if isinstance(meta, dict) and isinstance(meta.get(key), list) else []
        for key in ("key_regions", "tempo_regions", "meter_regions")
    }

    for idx, span in enumerate(spans):
        start = span["_start_beat"]
        end = span["_end_beat"]
        is_last = idx == len(spans) - 1
        blocks.append(
            {
                "span": span,
                "start": start,
                "end": end,
                "duration": span["_duration_beats"],
                "melody": [
                    (event_idx, event)
                    for event_idx, event in enumerate(melody)
                    if isinstance(event, dict) and _beat_in_interval(_event_beat(event), start, end, is_last=is_last)
                ],
                "chords": [
                    (event_idx, event)
                    for event_idx, event in enumerate(chords)
                    if isinstance(event, dict) and _beat_in_interval(_event_beat(event), start, end, is_last=is_last)
                ],
                "regions": {
                    key: [
                        (event_idx, event)
                        for event_idx, event in enumerate(regions)
                        if isinstance(event, dict) and _beat_in_interval(_event_beat(event), start, end, is_last=is_last)
                    ]
                    for key, regions in region_lists.items()
                },
            }
        )
    return blocks


def _copy_block_event(event: dict, old_start: float, new_start: float, song_obj: dict, new_section_index: int, source_section_index: int) -> dict:
    copied = copy.deepcopy(event)
    _shift_event_beat(copied, new_start - old_start, song_obj)
    copied["corrupted_section_index"] = int(new_section_index)
    copied["section_corruption_source_section_index"] = int(source_section_index)
    return copied


def _rebuild_song_from_section_order(song_obj: dict, order: list[int], *, duplicate_counts: dict[int, int] | None = None) -> dict:
    spans = _normalized_section_spans(song_obj)
    blocks = _section_blocks(song_obj, spans)
    duplicate_counts = duplicate_counts or {}
    new_melody: list[dict] = []
    new_chords: list[dict] = []
    new_regions = {"key_regions": [], "tempo_regions": [], "meter_regions": []}
    new_spans: list[dict] = []
    cursor = 1.0

    for new_section_index, source_section_index in enumerate(order):
        block = blocks[source_section_index]
        span = block["span"]
        duration = max(0.0, float(block["duration"]))
        start = cursor
        end = start + duration
        for _, event in block["melody"]:
            new_melody.append(_copy_block_event(event, block["start"], start, song_obj, new_section_index, source_section_index))
        for _, event in block["chords"]:
            new_chords.append(_copy_block_event(event, block["start"], start, song_obj, new_section_index, source_section_index))
        for key in new_regions:
            for _, region in block["regions"][key]:
                new_regions[key].append(_copy_block_event(region, block["start"], start, song_obj, new_section_index, source_section_index))

        copied_span = copy.deepcopy(span)
        for internal_key in ("_original_section_list_index", "_start_beat", "_end_beat", "_duration_beats"):
            copied_span.pop(internal_key, None)
        copied_span.setdefault("original_section_index", span.get("section_index", source_section_index))
        copied_span["section_index"] = int(new_section_index)
        copied_span["target_start_beat"] = float(start)
        copied_span["target_end_beat"] = float(end)
        copied_span["duration_beats"] = float(duration)
        copied_span["section_corruption_source_section_index"] = int(source_section_index)
        copied_span["section_corruption_duplicate_ordinal"] = int(duplicate_counts.get(source_section_index, 0))
        copied_span["inserted_gap_beats_before"] = 0.0
        copied_span["inserted_gap_seconds_before"] = 0.0
        copied_span["inserted_gap_bars_before"] = 0.0
        copied_span["extra_full_gap_bars_before"] = 0
        copied_span["gap_placement_reason"] = "section_corruption_compact_rebuild"
        new_spans.append(copied_span)
        duplicate_counts[source_section_index] = int(duplicate_counts.get(source_section_index, 0)) + 1
        cursor = end

    song_obj["melody"] = new_melody
    song_obj["chords"] = new_chords
    meta = song_obj.setdefault("meta", {})
    meta["section_spans"] = new_spans
    meta["end_beat"] = float(cursor)
    meta["section_corruption_rebuild_policy"] = "compact_sections"
    for key, values in new_regions.items():
        if values or key in meta:
            meta[key] = values
    _sort_events_in_place(song_obj)
    return song_obj


def _section_target_indices_from_cfg(corruption_cfg: dict, key: str, n_sections: int) -> list[int]:
    raw = corruption_cfg.get(key)
    if raw is None:
        raw = corruption_cfg.get("section_target_index")
    if raw is None:
        return []
    raw_values = raw if isinstance(raw, (list, tuple)) else [raw]
    indices = []
    for value in raw_values:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_sections:
            indices.append(idx)
    return list(dict.fromkeys(indices))


def _root_id_for_raw(root_raw: int, theory_ctx: dict) -> int | None:
    for root_id, raw_value in theory_ctx.get("root_id_to_raw", {}).items():
        if int(raw_value) == int(root_raw):
            return int(root_id)
    return None


_ROOT_LABELS_MAJOR = {
    0: "I",
    1: "ii",
    2: "iii",
    3: "IV",
    4: "V",
    5: "vi",
    6: "vii",
    7: "bVII",
}
_ROOT_LABELS_MINOR = {
    0: "i",
    1: "ii",
    2: "III",
    3: "iv",
    4: "V",
    5: "VI",
    6: "vii",
    7: "bVII",
}


def _mode_family_for_functions(song_obj: dict, theory_ctx: dict) -> str:
    scale_id = int(song_obj.get("meta", {}).get("main_key_scale_id", 2) or 2)
    mode_name = theory_ctx.get("scale_id_to_name", {}).get(scale_id, "major")
    return "minor" if mode_name in {"minor", "dorian", "phrygian", "locrian", "harmonic_minor"} else "major"


def classify_chord_function_root_only(song_obj: dict, chord: dict, theory_ctx: dict, *, rule_set: str = "boundary_strict") -> dict:
    """Classify a chord into broad T/PD/D slots using root degree only.

    `applied_id` is intentionally ignored here; section-boundary corruptions use
    this as a conservative root-level heuristic, not as full Roman-numeral analysis.
    """
    if int(chord.get("is_rest", 0) or 0) == 1:
        return {"slot": "UNKNOWN", "root_raw": None, "mode_family": None, "degree_label": "rest", "reason": "rest_chord"}
    root_raw = decode_root_raw(chord, theory_ctx)
    if root_raw is None:
        return {"slot": "UNKNOWN", "root_raw": None, "mode_family": None, "degree_label": "unknown", "reason": "missing_root"}

    mode_family = _mode_family_for_functions(song_obj, theory_ctx)
    if rule_set == "boundary_expanded":
        tonic_roots = {0, 2, 5}
    else:
        tonic_roots = {0}
    predominant_roots = {1, 3}
    dominant_roots = {4, 6}

    root_raw = int(root_raw)
    if root_raw in tonic_roots:
        slot = "T"
    elif root_raw in predominant_roots:
        slot = "PD"
    elif root_raw in dominant_roots:
        slot = "D"
    else:
        slot = "OTHER"

    labels = _ROOT_LABELS_MINOR if mode_family == "minor" else _ROOT_LABELS_MAJOR
    return {
        "slot": slot,
        "root_raw": root_raw,
        "mode_family": mode_family,
        "degree_label": labels.get(root_raw, str(root_raw)),
        "rule_set": rule_set,
        "reason": "root_only",
    }


def _chord_candidates_in_section(song_obj: dict, span: dict, *, is_last: bool) -> list[tuple[float, int]]:
    chords = song_obj.get("chords", []) if isinstance(song_obj.get("chords"), list) else []
    start, end = span["_start_beat"], span["_end_beat"]
    return [
        (safe_float(chord.get("beat"), 1.0), idx)
        for idx, chord in enumerate(chords)
        if isinstance(chord, dict)
        and int(chord.get("is_rest", 0) or 0) == 0
        and _beat_in_interval(_event_beat(chord), start, end, is_last=is_last)
    ]


def _first_chord_index_in_section(song_obj: dict, spans: list[dict], section_idx: int) -> int | None:
    candidates = _chord_candidates_in_section(song_obj, spans[section_idx], is_last=section_idx == len(spans) - 1)
    return min(candidates)[1] if candidates else None


def _last_chord_index_in_section(song_obj: dict, spans: list[dict], section_idx: int) -> int | None:
    candidates = _chord_candidates_in_section(song_obj, spans[section_idx], is_last=section_idx == len(spans) - 1)
    return max(candidates)[1] if candidates else None


def _set_chord_root_raw(chord: dict, new_root_raw: int, theory_ctx: dict) -> int | None:
    new_root_id = _root_id_for_raw(new_root_raw, theory_ctx)
    if new_root_id is None:
        return None
    chord["root_id"] = int(new_root_id)
    if "root_degree_raw" in chord:
        chord["root_degree_raw"] = int(new_root_raw)
    return int(new_root_id)


def _replacement_root_raw(candidates: list[int], current_root_raw: int, theory_ctx: dict, rng) -> int | None:
    valid = [root_raw for root_raw in candidates if root_raw != current_root_raw and _root_id_for_raw(root_raw, theory_ctx) is not None]
    if not valid:
        return None
    return int(rng.choice(valid))


def _section_order_metadata(mode: str, spans: list[dict], old_order: list[int], new_order: list[int], extra_details: dict | None = None) -> dict:
    metadata = _identity_metadata(mode)
    labels = [_section_label_for_metadata(spans[idx]) for idx in range(len(spans))]
    details = {
        "original_section_order": old_order,
        "new_section_order": new_order,
        "section_labels": labels,
        "boundary_beats": [
            [float(spans[idx]["_start_beat"]), float(spans[idx]["_end_beat"])]
            for idx in range(len(spans))
        ],
        "section_rebuild_policy": "compact_sections",
    }
    if extra_details:
        details.update(extra_details)
    metadata.update({
        "applied": True,
        "topology_changed": True,
        "details": details,
    })
    return metadata


def _corrupt_adjacent_section_swap(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("adjacent_section_swap")
    spans = _normalized_section_spans(song_obj)
    if len(spans) < 2:
        metadata["reason_skipped"] = "not_enough_sections"
        return song_obj, metadata, False

    configured = _section_target_indices_from_cfg(corruption_cfg, "section_swap_left_index", len(spans))
    left_candidates = [idx for idx in configured if idx + 1 < len(spans)]
    if not left_candidates:
        left_candidates = list(range(len(spans) - 1))
    left_idx = int(rng.choice(left_candidates))
    order = list(range(len(spans)))
    order[left_idx], order[left_idx + 1] = order[left_idx + 1], order[left_idx]
    _rebuild_song_from_section_order(song_obj, order)
    metadata = _section_order_metadata(
        "adjacent_section_swap",
        spans,
        old_order=list(range(len(spans))),
        new_order=order,
        extra_details={"affected_section_indices": [left_idx, left_idx + 1]},
    )
    return song_obj, metadata, True


def _corrupt_non_adjacent_section_swap(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("non_adjacent_section_swap")
    spans = _normalized_section_spans(song_obj)
    if len(spans) < 3:
        metadata["reason_skipped"] = "not_enough_sections"
        return song_obj, metadata, False

    configured = _section_target_indices_from_cfg(corruption_cfg, "section_swap_indices", len(spans))
    pair = None
    if len(configured) >= 2 and abs(configured[0] - configured[1]) > 1:
        pair = (configured[0], configured[1])
    if pair is None:
        pairs = [(left, right) for left in range(len(spans)) for right in range(left + 2, len(spans))]
        if not pairs:
            metadata["reason_skipped"] = "no_non_adjacent_section_pair"
            return song_obj, metadata, False
        pair = rng.choice(pairs)

    left_idx, right_idx = int(pair[0]), int(pair[1])
    order = list(range(len(spans)))
    order[left_idx], order[right_idx] = order[right_idx], order[left_idx]
    _rebuild_song_from_section_order(song_obj, order)
    metadata = _section_order_metadata(
        "non_adjacent_section_swap",
        spans,
        old_order=list(range(len(spans))),
        new_order=order,
        extra_details={"affected_section_indices": [left_idx, right_idx]},
    )
    return song_obj, metadata, True


def _corrupt_section_duplicate(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("section_duplicate")
    spans = _normalized_section_spans(song_obj)
    if len(spans) < 1:
        metadata["reason_skipped"] = "no_sections"
        return song_obj, metadata, False

    configured = _section_target_indices_from_cfg(corruption_cfg, "section_duplicate_index", len(spans))
    section_idx = int(configured[0]) if configured else int(rng.choice(list(range(len(spans)))))
    max_times = int(corruption_cfg.get("section_duplicate_max_times", corruption_cfg.get("section_duplicate_n_max", 2)) or 2)
    max_times = max(1, max_times)
    forced_times = corruption_cfg.get("section_duplicate_times")
    if forced_times is not None:
        try:
            duplicate_times = max(1, min(max_times, int(forced_times)))
        except (TypeError, ValueError):
            duplicate_times = 1
    else:
        duplicate_times = int(rng.randint(1, max_times))

    order = []
    for idx in range(len(spans)):
        order.append(idx)
        if idx == section_idx:
            order.extend([idx] * duplicate_times)

    _rebuild_song_from_section_order(song_obj, order)
    metadata = _section_order_metadata(
        "section_duplicate",
        spans,
        old_order=list(range(len(spans))),
        new_order=order,
        extra_details={"affected_section_indices": [section_idx], "duplicate_times": duplicate_times},
    )
    return song_obj, metadata, True


def _remove_section_events_keep_silence(song_obj: dict, section_idx: int, spans: list[dict]) -> tuple[list[int], list[int]]:
    span = spans[section_idx]
    start, end = span["_start_beat"], span["_end_beat"]
    is_last = section_idx == len(spans) - 1
    removed_note_indices = []
    removed_chord_indices = []

    melody = song_obj.get("melody", []) if isinstance(song_obj.get("melody"), list) else []
    new_melody = []
    for idx, note in enumerate(melody):
        if isinstance(note, dict) and _beat_in_interval(_event_beat(note), start, end, is_last=is_last):
            removed_note_indices.append(idx)
            continue
        new_melody.append(note)
    song_obj["melody"] = new_melody

    chords = song_obj.get("chords", []) if isinstance(song_obj.get("chords"), list) else []
    new_chords = []
    for idx, chord in enumerate(chords):
        if isinstance(chord, dict) and _beat_in_interval(_event_beat(chord), start, end, is_last=is_last):
            removed_chord_indices.append(idx)
            continue
        new_chords.append(chord)
    song_obj["chords"] = new_chords

    meta = song_obj.get("meta", {})
    if isinstance(meta, dict):
        for key in ("key_regions", "tempo_regions", "meter_regions"):
            regions = meta.get(key)
            if not isinstance(regions, list):
                continue
            meta[key] = [
                region
                for region in regions
                if not (isinstance(region, dict) and _beat_in_interval(_event_beat(region), start, end, is_last=is_last))
            ]
        meta["section_corruption_rebuild_policy"] = "keep_silence"
    _sort_events_in_place(song_obj)
    return removed_note_indices, removed_chord_indices


def _corrupt_section_drop_keep_silence(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("section_drop_keep_silence")
    spans = _normalized_section_spans(song_obj)
    if len(spans) < 2:
        metadata["reason_skipped"] = "not_enough_sections"
        return song_obj, metadata, False

    configured = _section_target_indices_from_cfg(corruption_cfg, "section_drop_index", len(spans))
    section_idx = int(configured[0]) if configured else int(rng.choice(list(range(len(spans)))))
    removed_note_indices, removed_chord_indices = _remove_section_events_keep_silence(song_obj, section_idx, spans)
    if not removed_note_indices and not removed_chord_indices:
        metadata["reason_skipped"] = "selected_section_has_no_events"
        return song_obj, metadata, False

    metadata.update({
        "applied": True,
        "topology_changed": True,
        "n_notes_modified": len(removed_note_indices),
        "n_chords_modified": len(removed_chord_indices),
        "details": {
            "affected_section_indices": [section_idx],
            "removed_note_indices_original": removed_note_indices,
            "removed_chord_indices_original": removed_chord_indices,
            "removed_section_label": _section_label_for_metadata(spans[section_idx]),
            "removed_section_boundary_beats": [spans[section_idx]["_start_beat"], spans[section_idx]["_end_beat"]],
            "section_rebuild_policy": "keep_silence",
        },
    })
    return song_obj, metadata, True


def _corrupt_section_drop_and_close_gap(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("section_drop_and_close_gap")
    spans = _normalized_section_spans(song_obj)
    if len(spans) < 2:
        metadata["reason_skipped"] = "not_enough_sections"
        return song_obj, metadata, False

    configured = _section_target_indices_from_cfg(corruption_cfg, "section_drop_index", len(spans))
    section_idx = int(configured[0]) if configured else int(rng.choice(list(range(len(spans)))))
    order = [idx for idx in range(len(spans)) if idx != section_idx]
    _rebuild_song_from_section_order(song_obj, order)
    metadata = _section_order_metadata(
        "section_drop_and_close_gap",
        spans,
        old_order=list(range(len(spans))),
        new_order=order,
        extra_details={
            "affected_section_indices": [section_idx],
            "removed_section_label": _section_label_for_metadata(spans[section_idx]),
            "removed_section_boundary_beats": [spans[section_idx]["_start_beat"], spans[section_idx]["_end_beat"]],
        },
    )
    return song_obj, metadata, True


def _corrupt_section_entry_non_tonic_substitution(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("section_entry_non_tonic_substitution")
    spans = _normalized_section_spans(song_obj)
    chords = song_obj.get("chords", []) if isinstance(song_obj.get("chords"), list) else []
    if not spans or not chords:
        metadata["reason_skipped"] = "missing_sections_or_chords"
        return song_obj, metadata, False

    rule_set = str(corruption_cfg.get("section_function_rule_set", "boundary_strict"))
    configured = _section_target_indices_from_cfg(corruption_cfg, "section_boundary_index", len(spans))
    section_indices = configured or list(range(len(spans)))
    rng.shuffle(section_indices)
    replacement_candidates = list(corruption_cfg.get("section_non_tonic_root_raws", [1, 3, 4, 6]))

    for section_idx in section_indices:
        chord_idx = _first_chord_index_in_section(song_obj, spans, section_idx)
        if chord_idx is None:
            continue
        chord = chords[chord_idx]
        before = classify_chord_function_root_only(song_obj, chord, theory_ctx, rule_set=rule_set)
        if before["slot"] != "T" or before["root_raw"] is None:
            continue
        new_root_raw = _replacement_root_raw(replacement_candidates, int(before["root_raw"]), theory_ctx, rng)
        if new_root_raw is None:
            continue
        old_root_id = int(chord.get("root_id", 0) or 0)
        new_root_id = _set_chord_root_raw(chord, new_root_raw, theory_ctx)
        after = classify_chord_function_root_only(song_obj, chord, theory_ctx, rule_set=rule_set)
        metadata.update({
            "applied": True,
            "n_chords_modified": 1,
            "chord_corrupted_indices": [chord_idx],
            "details": {
                "affected_section_indices": [section_idx],
                "boundary": "entry",
                "old_root_id": old_root_id,
                "new_root_id": new_root_id,
                "function_before": before,
                "function_after": after,
            },
        })
        return song_obj, metadata, True

    metadata["reason_skipped"] = "no_tonic_section_entry_chord"
    return song_obj, metadata, False


def _corrupt_section_exit_non_dominant_substitution(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("section_exit_non_dominant_substitution")
    spans = _normalized_section_spans(song_obj)
    chords = song_obj.get("chords", []) if isinstance(song_obj.get("chords"), list) else []
    if not spans or not chords:
        metadata["reason_skipped"] = "missing_sections_or_chords"
        return song_obj, metadata, False

    rule_set = str(corruption_cfg.get("section_function_rule_set", "boundary_strict"))
    configured = _section_target_indices_from_cfg(corruption_cfg, "section_boundary_index", len(spans))
    section_indices = configured or list(range(len(spans)))
    rng.shuffle(section_indices)
    replacement_candidates = list(corruption_cfg.get("section_non_dominant_root_raws", [0, 1, 2, 3, 5]))

    for section_idx in section_indices:
        chord_idx = _last_chord_index_in_section(song_obj, spans, section_idx)
        if chord_idx is None:
            continue
        chord = chords[chord_idx]
        before = classify_chord_function_root_only(song_obj, chord, theory_ctx, rule_set=rule_set)
        if before["slot"] != "D" or before["root_raw"] is None:
            continue
        new_root_raw = _replacement_root_raw(replacement_candidates, int(before["root_raw"]), theory_ctx, rng)
        if new_root_raw is None:
            continue
        old_root_id = int(chord.get("root_id", 0) or 0)
        new_root_id = _set_chord_root_raw(chord, new_root_raw, theory_ctx)
        after = classify_chord_function_root_only(song_obj, chord, theory_ctx, rule_set=rule_set)
        metadata.update({
            "applied": True,
            "n_chords_modified": 1,
            "chord_corrupted_indices": [chord_idx],
            "details": {
                "affected_section_indices": [section_idx],
                "boundary": "exit",
                "old_root_id": old_root_id,
                "new_root_id": new_root_id,
                "function_before": before,
                "function_after": after,
            },
        })
        return song_obj, metadata, True

    metadata["reason_skipped"] = "no_dominant_section_exit_chord"
    return song_obj, metadata, False


def _corrupt_transpose_with_tonic_shift(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("transpose_with_tonic_shift")
    semitones = list(corruption_cfg.get("transpose_semitones", [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]))
    if not semitones:
        metadata["reason_skipped"] = "empty_semitone_candidates"
        return song_obj, metadata, False
    k = int(rng.choice(semitones))

    tracks = []
    for key in ("melody", "notes", "chords", "bass"):
        events = song_obj.get(key)
        if isinstance(events, list):
            tracks.extend(events)
    tracks = [event for event in tracks if isinstance(event, dict)]
    pitch_events = [event for event in tracks if "pitch" in event]
    if pitch_events and not _has_pitch_in_midi_range(pitch_events, k):
        metadata["reason_skipped"] = "pitch_out_of_midi_range_after_shift"
        return song_obj, metadata, False

    changed_tonic, tonic_details = _transpose_tonic_fields(song_obj, semitones=k)
    if not changed_tonic and not pitch_events:
        metadata["reason_skipped"] = "missing_tonic_pc"
        return song_obj, metadata, False

    for event in pitch_events:
        event["pitch"] = int(event["pitch"]) + k
    metadata.update({
        "applied": True,
        "n_notes_modified": len(pitch_events),
        "corruption_params": {"k": k},
        "details": {
            **tonic_details,
            "tonic_shift_applied": bool(changed_tonic),
            "n_pitch_events_modified": len(pitch_events),
        },
    })
    return song_obj, metadata, True


def _corrupt_melody_octave_shift(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("melody_octave_shift")
    melody_key, melody = _melody_events(song_obj)
    if not melody_key:
        metadata["reason_skipped"] = "melody_track_not_found"
        return song_obj, metadata, False
    non_rest_notes = [note for note in melody if int(note.get("is_rest", 0)) == 0]
    if not non_rest_notes:
        metadata["reason_skipped"] = "no_non_rest_melody_notes"
        return song_obj, metadata, False

    options = list(corruption_cfg.get("melody_octave_shifts", [-12, 12]))
    rng.shuffle(options)
    for shift in options:
        if not _has_pitch_in_midi_range(non_rest_notes, int(shift)):
            continue
        for note in non_rest_notes:
            if "pitch" in note:
                note["pitch"] = int(note["pitch"]) + int(shift)
            if "octave_id" in note:
                note["octave_id"] = int(note["octave_id"]) + (int(shift) // 12)
        metadata.update({
            "applied": True,
            "n_notes_modified": len(non_rest_notes),
            "corruption_params": {"octave_shift": int(shift)},
            "details": {"melody_key": melody_key},
        })
        return song_obj, metadata, True

    metadata["reason_skipped"] = "pitch_out_of_midi_range_after_shift"
    return song_obj, metadata, False


def _corrupt_merge_repeated_melody_notes(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("merge_repeated_melody_notes")
    _, melody = _melody_events(song_obj)
    if not melody:
        metadata["reason_skipped"] = "melody_track_not_found"
        return song_obj, metadata, False
    eps = safe_float(corruption_cfg.get("merge_notes_eps", DEFAULT_EPSILON), DEFAULT_EPSILON)
    merged_indices = []
    i = 0
    while i < len(melody) - 1:
        curr, nxt = melody[i], melody[i + 1]
        if int(curr.get("is_rest", 0)) == 1 or int(nxt.get("is_rest", 0)) == 1:
            i += 1
            continue
        curr_pitch_sig = (curr.get("pitch"), curr.get("sd_id"), curr.get("octave_id"))
        next_pitch_sig = (nxt.get("pitch"), nxt.get("sd_id"), nxt.get("octave_id"))
        if curr_pitch_sig != next_pitch_sig:
            i += 1
            continue
        curr_start, curr_end = _note_interval(curr)
        next_start, next_end = _note_interval(nxt)
        if None in (curr_start, curr_end, next_start, next_end):
            i += 1
            continue
        if abs(next_start - curr_end) > eps:
            i += 1
            continue
        _set_note_interval(curr, float(curr_start), float(next_end))
        del melody[i + 1]
        merged_indices.append(i)
        continue
    if not merged_indices:
        metadata["reason_skipped"] = "no_mergeable_repeated_notes"
        return song_obj, metadata, False
    metadata.update({
        "applied": True,
        "n_notes_modified": len(merged_indices) + 1,
        "note_corrupted_indices": sorted(set(merged_indices)),
        "corruption_params": {"eps": eps},
        "details": {"merged_groups_count": len(merged_indices)},
    })
    return song_obj, metadata, True


def _corrupt_split_long_melody_note(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("split_long_melody_note")
    _, melody = _melody_events(song_obj)
    if not melody:
        metadata["reason_skipped"] = "melody_track_not_found"
        return song_obj, metadata, False
    min_duration = safe_float(corruption_cfg.get("split_min_duration_beats", 1.0), 1.0)
    eps = safe_float(corruption_cfg.get("split_notes_eps", DEFAULT_EPSILON), DEFAULT_EPSILON)
    candidate_indices = list(range(len(melody)))
    rng.shuffle(candidate_indices)
    for idx in candidate_indices:
        note = melody[idx]
        if int(note.get("is_rest", 0)) == 1:
            continue
        start, end = _note_interval(note)
        if start is None or end is None:
            continue
        duration = end - start
        if duration < min_duration:
            continue
        split_point = start + duration / 2.0
        if split_point - start <= eps or end - split_point <= eps:
            continue
        if idx + 1 < len(melody):
            next_start, _ = _note_interval(melody[idx + 1])
            if next_start is not None and split_point > next_start + eps:
                continue
        left = copy.deepcopy(note)
        right = copy.deepcopy(note)
        _set_note_interval(left, float(start), float(split_point))
        _set_note_interval(right, float(split_point), float(end))
        melody[idx] = left
        melody.insert(idx + 1, right)
        metadata.update({
            "applied": True,
            "n_notes_modified": 2,
            "note_corrupted_indices": [idx, idx + 1],
            "corruption_params": {"min_duration_beats": min_duration, "split_mode": "half"},
            "details": {"split_point": split_point},
        })
        return song_obj, metadata, True
    metadata["reason_skipped"] = "no_splittable_melody_note"
    return song_obj, metadata, False


def _corrupt_drop_tonic_seventh_on_strong_beat(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("drop_tonic_seventh_on_strong_beat")
    chords = song_obj.get("chords", [])
    if not chords:
        metadata["reason_skipped"] = "no_chords_found"
        return song_obj, metadata, False
    strong_positions = {float(x) for x in corruption_cfg.get("strong_positions", [0.0])}

    chord_indices = list(range(len(chords)))
    rng.shuffle(chord_indices)
    for idx in chord_indices:
        chord = chords[idx]
        pos_in_bar = try_parse_float(chord.get("pos_in_bar"))
        if pos_in_bar is None:
            beat = try_parse_float(chord.get("beat"))
            if beat is None:
                continue
            num_beats = safe_float(song_obj.get("meta", {}).get("num_beats", 4.0), 4.0)
            pos_in_bar = (beat % num_beats + num_beats) % num_beats
        if all(abs(pos_in_bar - strong) > DEFAULT_EPSILON for strong in strong_positions):
            continue
        root_degree_raw = chord.get("root_degree_raw")
        type_raw = chord.get("type_raw")
        if root_degree_raw is None and chord.get("root_id") is not None and isinstance(theory_ctx, dict):
            try:
                root_degree_raw = theory_ctx.get("root_id_to_raw", {}).get(int(chord.get("root_id")))
            except (TypeError, ValueError):
                root_degree_raw = None
        if type_raw is None and chord.get("type_id") is not None and isinstance(theory_ctx, dict):
            try:
                type_raw = theory_ctx.get("type_id_to_raw", {}).get(int(chord.get("type_id")))
            except (TypeError, ValueError):
                type_raw = None
        if root_degree_raw is None or type_raw is None:
            continue
        if int(root_degree_raw) != 0 or int(type_raw) not in {7, 9, 11, 13}:
            continue
        chord["type_raw"] = 5
        if chord.get("type_id") is not None and isinstance(theory_ctx, dict):
            raw_to_type_id = {int(raw): int(type_id) for type_id, raw in theory_ctx.get("type_id_to_raw", {}).items()}
            triad_id = raw_to_type_id.get(5)
            if triad_id is not None:
                chord["type_id"] = int(triad_id)
        if isinstance(chord.get("add_degrees"), list):
            chord["add_degrees"] = [int(x) for x in chord["add_degrees"] if int(x) != 7]
        metadata.update({
            "applied": True,
            "n_chords_modified": 1,
            "chord_corrupted_indices": [idx],
            "corruption_params": {"strong_positions": sorted(strong_positions)},
            "details": {"type_raw_before": int(type_raw), "type_raw_after": 5},
        })
        return song_obj, metadata, True
    metadata["reason_skipped"] = "no_matching_tonic_seventh_on_strong_beat"
    return song_obj, metadata, False

def _corrupt_strongbeat_nonchord_note(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("strongbeat_nonchord_note")
    min_duration = safe_float(corruption_cfg.get("strongbeat_min_duration", 1.0), 1.0)
    strongbeat_only = bool(corruption_cfg.get("strongbeat_only", True))

    indices = list(range(len(song_obj.get("melody", []))))
    rng.shuffle(indices)
    for note_idx in indices:
        note = song_obj["melody"][note_idx]
        if int(note.get("is_rest", 0)) == 1:
            continue
        note_duration = safe_float(note.get("duration", 0.0), 0.0)
        if note_duration < min_duration and not is_strong_note_position(note, song_obj):
            continue
        if strongbeat_only and not is_strong_note_position(note, song_obj):
            continue

        chord_idx = find_covering_chord_index(song_obj, note)
        if chord_idx is None:
            continue
        chord = song_obj["chords"][chord_idx]
        if int(chord.get("is_rest", 0)) == 1:
            continue

        chord_pcs = chord_pitch_classes_tertian(song_obj, chord, theory_ctx)
        if not chord_pcs:
            continue
        old_sd = int(note.get("sd_id", 0))
        new_sd = _pick_new_sd_id(exclude_pcs=chord_pcs, include_pcs=None, theory_ctx=theory_ctx, rng=rng)
        if new_sd is None or new_sd == old_sd:
            continue

        note["sd_id"] = new_sd
        metadata.update({
            "applied": True,
            "note_corrupted_indices": [note_idx],
            "details": {
                "original_sd_id": old_sd,
                "new_sd_id": new_sd,
                "covering_chord_index": chord_idx,
            },
        })
        return song_obj, metadata, True

    return song_obj, metadata, False


def _corrupt_borrowed_melody_conflict(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("borrowed_melody_conflict")
    note_indices = list(range(len(song_obj.get("melody", []))))
    rng.shuffle(note_indices)

    main_mode = theory_ctx["scale_id_to_name"].get(int(song_obj.get("meta", {}).get("main_key_scale_id", 2)), "major")
    main_pcset = set(theory_ctx["mode_to_pcset"].get(main_mode, theory_ctx["mode_to_pcset"]["major"]))

    for note_idx in note_indices:
        note = song_obj["melody"][note_idx]
        if int(note.get("is_rest", 0)) == 1:
            continue
        chord_idx = find_covering_chord_index(song_obj, note)
        if chord_idx is None:
            continue
        chord = song_obj["chords"][chord_idx]

        borrowed_kind = theory_ctx["borrowed_id_to_kind"].get(int(chord.get("borrowed_kind_id", 0)), "none")
        borrowed_mode = theory_ctx["borrowed_mode_id_to_name"].get(int(chord.get("borrowed_mode_name_id", 0)))
        if borrowed_kind != "mode_name" or borrowed_mode not in theory_ctx["mode_to_pcset"]:
            continue

        borrowed_pcset = set(theory_ctx["mode_to_pcset"][borrowed_mode])
        conflict_pcs = main_pcset - borrowed_pcset
        if not conflict_pcs:
            continue

        old_sd = int(note.get("sd_id", 0))
        new_sd = _pick_new_sd_id(exclude_pcs=set(), include_pcs=conflict_pcs, theory_ctx=theory_ctx, rng=rng)
        if new_sd is None or new_sd == old_sd:
            continue

        note["sd_id"] = new_sd
        metadata.update({
            "applied": True,
            "note_corrupted_indices": [note_idx],
            "details": {
                "original_sd_id": old_sd,
                "new_sd_id": new_sd,
                "borrowed_mode_name": borrowed_mode,
                "covering_chord_index": chord_idx,
            },
        })
        return song_obj, metadata, True

    return song_obj, metadata, False


def _mode_to_pcset_vec(mode_name: str | None, theory_ctx: dict) -> list[int]:
    vec = [0] * 12
    if mode_name and mode_name in theory_ctx["mode_to_pcset"]:
        for pc in theory_ctx["mode_to_pcset"][mode_name]:
            vec[pc] = 1
    return vec


def _corrupt_borrowed_kind_toggle(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("borrowed_kind_toggle_without_melody_change")
    melody_key, melody = _melody_events(song_obj)
    if not melody_key or not any(int(note.get("is_rest", 0)) == 0 for note in melody):
        metadata["reason_skipped"] = "no_non_rest_melody_notes"
        return song_obj, metadata, False
    chord_indices = list(range(len(song_obj.get("chords", []))))
    rng.shuffle(chord_indices)

    none_kind_id = next((idx for idx, name in theory_ctx["borrowed_id_to_kind"].items() if name == "none"), None)
    mode_name_kind_id = next((idx for idx, name in theory_ctx["borrowed_id_to_kind"].items() if name == "mode_name"), None)
    none_mode_id = next(
        (idx for idx, name in theory_ctx["borrowed_mode_id_to_name"].items() if isinstance(name, str) and "none" in name.lower()),
        None,
    )
    valid_mode_ids = [
        idx
        for idx, name in theory_ctx["borrowed_mode_id_to_name"].items()
        if isinstance(name, str) and name in theory_ctx["mode_to_pcset"]
    ]
    if none_kind_id is None or mode_name_kind_id is None or none_mode_id is None:
        return song_obj, metadata, False

    candidates: list[dict] = []
    for chord_idx in chord_indices:
        chord = song_obj["chords"][chord_idx]
        if int(chord.get("is_rest", 0)) == 1:
            continue
        overlapping_melody_indices = _collect_overlapping_melody_indices(song_obj, chord)
        if not overlapping_melody_indices:
            continue
        old_kind_id = int(chord.get("borrowed_kind_id", 2))
        old_mode_id = int(chord.get("borrowed_mode_name_id", 2))
        old_kind = theory_ctx["borrowed_id_to_kind"].get(old_kind_id, "none")
        old_mode = theory_ctx["borrowed_mode_id_to_name"].get(old_mode_id)
        before_pcs = _decode_total_chord_pcs(song_obj, chord, theory_ctx)
        if not before_pcs:
            continue

        options: list[tuple[int, int]] = [(none_kind_id, none_mode_id)]
        options.extend((mode_name_kind_id, mode_id) for mode_id in valid_mode_ids)
        rng.shuffle(options)

        for new_kind_id, new_mode_id in options:
            if new_kind_id == old_kind_id and new_mode_id == old_mode_id:
                continue
            new_kind = theory_ctx["borrowed_id_to_kind"].get(new_kind_id)
            new_mode = theory_ctx["borrowed_mode_id_to_name"].get(new_mode_id)
            chord_candidate = copy.deepcopy(chord)
            chord_candidate["borrowed_kind_id"] = new_kind_id
            chord_candidate["borrowed_mode_name_id"] = new_mode_id
            chord_candidate["borrowed_pcset_vec"] = _mode_to_pcset_vec(new_mode if new_kind == "mode_name" else None, theory_ctx)
            after_pcs = _decode_total_chord_pcs(song_obj, chord_candidate, theory_ctx)
            if not after_pcs or after_pcs == before_pcs:
                continue
            candidates.append({
                "chord_idx": chord_idx,
                "new_kind_id": new_kind_id,
                "new_mode_id": new_mode_id,
                "new_kind": new_kind,
                "new_mode": new_mode,
                "before_pcs": before_pcs,
                "after_pcs": after_pcs,
                "overlapping_melody_indices": overlapping_melody_indices,
                "score": len(before_pcs.symmetric_difference(after_pcs)),
                "old_kind": old_kind,
                "old_mode": old_mode,
            })

    if not candidates:
        metadata["reason_skipped"] = "no_chord_with_overlapping_melody_and_changed_pcset"
        return song_obj, metadata, False

    max_score = max(candidate["score"] for candidate in candidates)
    best_candidates = [candidate for candidate in candidates if candidate["score"] == max_score]
    chosen = rng.choice(best_candidates)
    chord = song_obj["chords"][chosen["chord_idx"]]
    chord["borrowed_kind_id"] = chosen["new_kind_id"]
    chord["borrowed_mode_name_id"] = chosen["new_mode_id"]
    chord["borrowed_pcset_vec"] = _mode_to_pcset_vec(chosen["new_mode"] if chosen["new_kind"] == "mode_name" else None, theory_ctx)

    metadata.update({
        "applied": True,
        "n_chords_modified": 1,
        "chord_corrupted_indices": [chosen["chord_idx"]],
        "details": {
            "borrowed_kind_before": chosen["old_kind"],
            "borrowed_kind_after": chosen["new_kind"],
            "borrowed_mode_before": chosen["old_mode"],
            "borrowed_mode_after": chosen["new_mode"],
            "overlapping_melody_indices": chosen["overlapping_melody_indices"],
            "chord_pcs_before": sorted(chosen["before_pcs"]),
            "chord_pcs_after": sorted(chosen["after_pcs"]),
        },
    })
    return song_obj, metadata, True


def _corrupt_melody_semitone_add_clash(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("melody_semitone_add_clash")
    melody_key, melody = _melody_events(song_obj)
    if not melody_key or not melody:
        metadata["reason_skipped"] = "melody_track_not_found"
        return song_obj, metadata, False
    chords = song_obj.get("chords", [])
    if not chords:
        metadata["reason_skipped"] = "no_chords_found"
        return song_obj, metadata, False

    add_allowed_values = [int(v) for v in theory_ctx.get("chord_add_allowed_values", [])]
    if not add_allowed_values:
        metadata["reason_skipped"] = "no_add_degrees_available"
        return song_obj, metadata, False

    chord_candidates = list(range(len(chords)))
    rng.shuffle(chord_candidates)
    all_candidates: list[dict] = []
    for chord_idx in chord_candidates:
        chord = chords[chord_idx]
        if int(chord.get("is_rest", 0)) == 1:
            continue
        chord_start = try_parse_float(chord.get("beat"))
        overlapping_note_indices = _collect_overlapping_melody_indices(song_obj, chord)
        if not overlapping_note_indices:
            continue
        same_onset_indices = [
            note_idx
            for note_idx in overlapping_note_indices
            if abs(safe_float(melody[note_idx].get("beat"), -999.0) - safe_float(chord_start, -999.0)) <= DEFAULT_EPSILON
        ]
        note_indices = same_onset_indices or overlapping_note_indices
        current_total_pcs = _decode_total_chord_pcs(song_obj, chord, theory_ctx)
        if not current_total_pcs:
            continue
        current_adds = list(chord.get("adds_vec") or [0] * len(add_allowed_values))
        if len(current_adds) < len(add_allowed_values):
            current_adds.extend([0] * (len(add_allowed_values) - len(current_adds)))

        for note_idx in note_indices:
            note = melody[note_idx]
            if int(note.get("is_rest", 0)) == 1:
                continue
            melody_pc = decode_sd_to_chromatic(int(note.get("sd_id", 0)), theory_ctx)
            if melody_pc is None:
                continue
            is_same_onset = abs(safe_float(note.get("beat"), -999.0) - safe_float(chord_start, -999.0)) <= DEFAULT_EPSILON
            is_strong = is_strong_note_position(note, song_obj)

            for add_idx, add_degree in enumerate(add_allowed_values):
                if add_idx < len(current_adds) and int(current_adds[add_idx]) == 1:
                    continue
                chord_candidate = copy.deepcopy(chord)
                next_adds = list(current_adds)
                next_adds[add_idx] = 1
                chord_candidate["adds_vec"] = next_adds
                candidate_total_pcs = _decode_total_chord_pcs(song_obj, chord_candidate, theory_ctx)
                if not candidate_total_pcs or candidate_total_pcs == current_total_pcs:
                    continue
                added_pcs = sorted(candidate_total_pcs - current_total_pcs)
                clash_pcs = [pc for pc in added_pcs if (pc - melody_pc) % 12 in {1, 11}]
                if not clash_pcs:
                    continue
                score = 0.01 * float(add_degree)
                if is_strong:
                    score += 2.0
                if is_same_onset:
                    score += 4.0
                all_candidates.append({
                    "score": score,
                    "chord_idx": chord_idx,
                    "note_idx": note_idx,
                    "add_idx": add_idx,
                    "add_degree": add_degree,
                    "new_adds_vec": next_adds,
                    "melody_pc": melody_pc,
                    "added_pcs": added_pcs,
                    "clash_pcs": clash_pcs,
                    "same_onset": is_same_onset,
                    "is_strong": is_strong,
                })

    if not all_candidates:
        metadata["reason_skipped"] = "no_add_candidate_with_melody_semitone_clash"
        return song_obj, metadata, False

    max_score = max(candidate["score"] for candidate in all_candidates)
    best_candidates = [candidate for candidate in all_candidates if abs(candidate["score"] - max_score) <= 1e-9]
    chosen = rng.choice(best_candidates)
    chord = chords[chosen["chord_idx"]]
    chord["adds_vec"] = chosen["new_adds_vec"]

    metadata.update({
        "applied": True,
        "n_chords_modified": 1,
        "chord_corrupted_indices": [chosen["chord_idx"]],
        "details": {
            "target_chord_index": chosen["chord_idx"],
            "target_note_index": chosen["note_idx"],
            "add_degree": chosen["add_degree"],
            "melody_pc": chosen["melody_pc"],
            "added_pcs": chosen["added_pcs"],
            "clash_pcs": chosen["clash_pcs"],
            "same_onset": chosen["same_onset"],
            "strong_note_position": chosen["is_strong"],
        },
    })
    return song_obj, metadata, True


def _corrupt_melody_suspension_clash(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("melody_suspension_clash")
    melody_key, melody = _melody_events(song_obj)
    if not melody_key or not melody:
        metadata["reason_skipped"] = "melody_track_not_found"
        return song_obj, metadata, False
    chords = song_obj.get("chords", [])
    if not chords:
        metadata["reason_skipped"] = "no_chords_found"
        return song_obj, metadata, False

    suspension_allowed_values = [int(v) for v in theory_ctx.get("chord_susp_allowed_values", [])]
    if not suspension_allowed_values:
        metadata["reason_skipped"] = "no_suspension_degrees_available"
        return song_obj, metadata, False

    all_candidates: list[dict] = []
    chord_indices = list(range(len(chords)))
    rng.shuffle(chord_indices)
    for chord_idx in chord_indices:
        chord = chords[chord_idx]
        if int(chord.get("is_rest", 0)) == 1:
            continue
        decoded = decode_chord_components(song_obj, chord, theory_ctx)
        if decoded is None:
            continue
        if 3 not in set(int(v) for v in decoded.get("body_degrees", [])):
            continue
        current_suspensions = set(int(v) for v in decoded.get("suspension_degrees", []))
        if current_suspensions:
            continue
        third_pc = decoded.get("degree_to_pc", {}).get(3)
        if third_pc is None:
            continue
        current_total_pcs = _total_pcs_from_decoded(decoded)
        contexts = _iter_chord_melody_contexts(song_obj, chord, melody, theory_ctx)
        if not contexts:
            continue
        current_sus_vec = _pad_bitvec(chord.get("suspensions_vec"), len(suspension_allowed_values))

        for context in contexts:
            for sus_idx, sus_degree in enumerate(suspension_allowed_values):
                if current_sus_vec[sus_idx] == 1:
                    continue
                sus_pc = decoded.get("degree_to_pc", {}).get(int(sus_degree))
                if sus_pc is None:
                    continue
                chord_candidate = copy.deepcopy(chord)
                next_sus_vec = list(current_sus_vec)
                next_sus_vec[sus_idx] = 1
                chord_candidate["suspensions_vec"] = next_sus_vec
                candidate_decoded = decode_chord_components(song_obj, chord_candidate, theory_ctx)
                candidate_total_pcs = _total_pcs_from_decoded(candidate_decoded)
                if not candidate_total_pcs or candidate_total_pcs == current_total_pcs:
                    continue
                added_pcs = sorted(candidate_total_pcs - current_total_pcs)
                removed_pcs = sorted(current_total_pcs - candidate_total_pcs)
                melody_reasserts_third = int(context["melody_pc"]) == int(third_pc)
                vertical_clash = any((int(pc) - int(context["melody_pc"])) % 12 in {1, 11} for pc in added_pcs)
                if not (melody_reasserts_third or vertical_clash):
                    continue
                score = _base_context_score(context)
                if melody_reasserts_third:
                    score += 3.0
                if vertical_clash:
                    score += 2.0
                if int(sus_degree) == 4:
                    score += 0.25
                all_candidates.append({
                    "score": score,
                    "chord_idx": chord_idx,
                    "note_idx": context["note_idx"],
                    "sus_degree": int(sus_degree),
                    "new_suspensions_vec": next_sus_vec,
                    "melody_pc": int(context["melody_pc"]),
                    "third_pc": int(third_pc),
                    "added_pcs": added_pcs,
                    "removed_pcs": removed_pcs,
                    "same_onset": bool(context["same_onset"]),
                    "is_strong": bool(context["is_strong"]),
                })

    if not all_candidates:
        metadata["reason_skipped"] = "no_suspension_candidate_with_melody_conflict"
        return song_obj, metadata, False

    max_score = max(candidate["score"] for candidate in all_candidates)
    best_candidates = [candidate for candidate in all_candidates if abs(candidate["score"] - max_score) <= 1e-9]
    chosen = rng.choice(best_candidates)
    chord = chords[chosen["chord_idx"]]
    chord["suspensions_vec"] = chosen["new_suspensions_vec"]

    metadata.update({
        "applied": True,
        "n_chords_modified": 1,
        "chord_corrupted_indices": [chosen["chord_idx"]],
        "details": {
            "target_chord_index": chosen["chord_idx"],
            "target_note_index": chosen["note_idx"],
            "suspension_degree": chosen["sus_degree"],
            "melody_pc": chosen["melody_pc"],
            "removed_third_pc": chosen["third_pc"],
            "added_pcs": chosen["added_pcs"],
            "removed_pcs": chosen["removed_pcs"],
            "same_onset": chosen["same_onset"],
            "strong_note_position": chosen["is_strong"],
        },
    })
    return song_obj, metadata, True


def _corrupt_melody_alteration_clash(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("melody_alteration_clash")
    melody_key, melody = _melody_events(song_obj)
    if not melody_key or not melody:
        metadata["reason_skipped"] = "melody_track_not_found"
        return song_obj, metadata, False
    chords = song_obj.get("chords", [])
    if not chords:
        metadata["reason_skipped"] = "no_chords_found"
        return song_obj, metadata, False

    alteration_allowed_values = [str(v) for v in theory_ctx.get("chord_alter_allowed_values", [])]
    if not alteration_allowed_values:
        metadata["reason_skipped"] = "no_alteration_tokens_available"
        return song_obj, metadata, False

    all_candidates: list[dict] = []
    chord_indices = list(range(len(chords)))
    rng.shuffle(chord_indices)
    for chord_idx in chord_indices:
        chord = chords[chord_idx]
        if int(chord.get("is_rest", 0)) == 1:
            continue
        decoded = decode_chord_components(song_obj, chord, theory_ctx)
        if decoded is None:
            continue
        current_total_pcs = _total_pcs_from_decoded(decoded)
        current_alterations = set(str(v) for v in decoded.get("alteration_tokens", []))
        contexts = _iter_chord_melody_contexts(song_obj, chord, melody, theory_ctx)
        if not contexts:
            continue
        current_alt_vec = _pad_bitvec(chord.get("alterations_vec"), len(alteration_allowed_values))

        for context in contexts:
            for alt_idx, alt_token in enumerate(alteration_allowed_values):
                if current_alt_vec[alt_idx] == 1 or alt_token in current_alterations:
                    continue
                chord_candidate = copy.deepcopy(chord)
                next_alt_vec = list(current_alt_vec)
                next_alt_vec[alt_idx] = 1
                chord_candidate["alterations_vec"] = next_alt_vec
                candidate_decoded = decode_chord_components(song_obj, chord_candidate, theory_ctx)
                candidate_total_pcs = _total_pcs_from_decoded(candidate_decoded)
                if not candidate_total_pcs or candidate_total_pcs == current_total_pcs:
                    continue
                added_pcs = sorted(candidate_total_pcs - current_total_pcs)
                removed_pcs = sorted(current_total_pcs - candidate_total_pcs)
                clash_pcs = [pc for pc in added_pcs if (int(pc) - int(context["melody_pc"])) % 12 in {1, 11}]
                if not clash_pcs:
                    continue
                score = _base_context_score(context) + 2.0
                if alt_token in {"b5", "#5"}:
                    score += 1.0
                all_candidates.append({
                    "score": score,
                    "chord_idx": chord_idx,
                    "note_idx": context["note_idx"],
                    "alteration_token": alt_token,
                    "new_alterations_vec": next_alt_vec,
                    "melody_pc": int(context["melody_pc"]),
                    "added_pcs": added_pcs,
                    "removed_pcs": removed_pcs,
                    "clash_pcs": clash_pcs,
                    "same_onset": bool(context["same_onset"]),
                    "is_strong": bool(context["is_strong"]),
                })

    if not all_candidates:
        metadata["reason_skipped"] = "no_alteration_candidate_with_melody_semitone_clash"
        return song_obj, metadata, False

    max_score = max(candidate["score"] for candidate in all_candidates)
    best_candidates = [candidate for candidate in all_candidates if abs(candidate["score"] - max_score) <= 1e-9]
    chosen = rng.choice(best_candidates)
    chord = chords[chosen["chord_idx"]]
    chord["alterations_vec"] = chosen["new_alterations_vec"]

    metadata.update({
        "applied": True,
        "n_chords_modified": 1,
        "chord_corrupted_indices": [chosen["chord_idx"]],
        "details": {
            "target_chord_index": chosen["chord_idx"],
            "target_note_index": chosen["note_idx"],
            "alteration_token": chosen["alteration_token"],
            "melody_pc": chosen["melody_pc"],
            "added_pcs": chosen["added_pcs"],
            "removed_pcs": chosen["removed_pcs"],
            "clash_pcs": chosen["clash_pcs"],
            "same_onset": chosen["same_onset"],
            "strong_note_position": chosen["is_strong"],
        },
    })
    return song_obj, metadata, True


def _corrupt_melody_omit_core_tone_conflict(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("melody_omit_core_tone_conflict")
    melody_key, melody = _melody_events(song_obj)
    if not melody_key or not melody:
        metadata["reason_skipped"] = "melody_track_not_found"
        return song_obj, metadata, False
    chords = song_obj.get("chords", [])
    if not chords:
        metadata["reason_skipped"] = "no_chords_found"
        return song_obj, metadata, False

    omit_allowed_values = [int(v) for v in theory_ctx.get("chord_omit_allowed_values", [])]
    if not omit_allowed_values:
        metadata["reason_skipped"] = "no_omit_degrees_available"
        return song_obj, metadata, False

    all_candidates: list[dict] = []
    chord_indices = list(range(len(chords)))
    rng.shuffle(chord_indices)
    for chord_idx in chord_indices:
        chord = chords[chord_idx]
        if int(chord.get("is_rest", 0)) == 1:
            continue
        decoded = decode_chord_components(song_obj, chord, theory_ctx)
        if decoded is None:
            continue
        current_total_pcs = _total_pcs_from_decoded(decoded)
        current_omits = set(int(v) for v in decoded.get("omit_degrees", []))
        current_suspensions = set(int(v) for v in decoded.get("suspension_degrees", []))
        current_alterations = set(str(v) for v in decoded.get("alteration_tokens", []))
        body_degrees = set(int(v) for v in decoded.get("body_degrees", []))
        degree_to_pc = {int(degree): int(pc) % 12 for degree, pc in decoded.get("degree_to_pc", {}).items()}
        contexts = _iter_chord_melody_contexts(song_obj, chord, melody, theory_ctx)
        if not contexts:
            continue
        current_omit_vec = _pad_bitvec(chord.get("omits_vec"), len(omit_allowed_values))

        for context in contexts:
            for omit_idx, omit_degree in enumerate(omit_allowed_values):
                if current_omit_vec[omit_idx] == 1 or omit_degree in current_omits:
                    continue
                if omit_degree not in body_degrees:
                    continue
                if omit_degree == 3 and current_suspensions:
                    continue
                if omit_degree == 5 and any(token.endswith("5") for token in current_alterations):
                    continue
                omitted_pc = degree_to_pc.get(int(omit_degree))
                if omitted_pc is None:
                    continue
                melody_supports_removed_tone = int(context["melody_pc"]) == int(omitted_pc)
                if not melody_supports_removed_tone:
                    continue
                chord_candidate = copy.deepcopy(chord)
                next_omit_vec = list(current_omit_vec)
                next_omit_vec[omit_idx] = 1
                chord_candidate["omits_vec"] = next_omit_vec
                candidate_decoded = decode_chord_components(song_obj, chord_candidate, theory_ctx)
                candidate_total_pcs = _total_pcs_from_decoded(candidate_decoded)
                if not candidate_total_pcs or candidate_total_pcs == current_total_pcs:
                    continue
                removed_pcs = sorted(current_total_pcs - candidate_total_pcs)
                if int(omitted_pc) not in {int(pc) for pc in removed_pcs}:
                    continue
                score = _base_context_score(context) + 3.0
                if int(omit_degree) == 3:
                    score += 1.0
                all_candidates.append({
                    "score": score,
                    "chord_idx": chord_idx,
                    "note_idx": context["note_idx"],
                    "omit_degree": int(omit_degree),
                    "new_omits_vec": next_omit_vec,
                    "melody_pc": int(context["melody_pc"]),
                    "removed_pcs": removed_pcs,
                    "same_onset": bool(context["same_onset"]),
                    "is_strong": bool(context["is_strong"]),
                })

    if not all_candidates:
        metadata["reason_skipped"] = "no_omit_candidate_supported_by_melody"
        return song_obj, metadata, False

    max_score = max(candidate["score"] for candidate in all_candidates)
    best_candidates = [candidate for candidate in all_candidates if abs(candidate["score"] - max_score) <= 1e-9]
    chosen = rng.choice(best_candidates)
    chord = chords[chosen["chord_idx"]]
    chord["omits_vec"] = chosen["new_omits_vec"]

    metadata.update({
        "applied": True,
        "n_chords_modified": 1,
        "chord_corrupted_indices": [chosen["chord_idx"]],
        "details": {
            "target_chord_index": chosen["chord_idx"],
            "target_note_index": chosen["note_idx"],
            "omit_degree": chosen["omit_degree"],
            "melody_pc": chosen["melody_pc"],
            "removed_pcs": chosen["removed_pcs"],
            "same_onset": chosen["same_onset"],
            "strong_note_position": chosen["is_strong"],
        },
    })
    return song_obj, metadata, True


def _corrupt_inversion_bass_continuity_conflict(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("inversion_bass_continuity_conflict")
    chords = song_obj.get("chords", [])
    if len(chords) < 2:
        metadata["reason_skipped"] = "not_enough_chords"
        return song_obj, metadata, False

    inversion_raw_to_id = {int(raw): int(inversion_id) for inversion_id, raw in theory_ctx.get("inversion_id_to_raw", {}).items()}
    min_badness_gain = float(corruption_cfg.get("inversion_min_badness_gain", 0.5))
    candidates: list[dict] = []
    chord_indices = list(range(len(chords)))
    rng.shuffle(chord_indices)

    for chord_idx in chord_indices:
        chord = chords[chord_idx]
        if int(chord.get("is_rest", 0)) == 1:
            continue
        body_pcs = chord_body_pcs_ordered(song_obj, chord, theory_ctx)
        if not body_pcs or len(body_pcs) < 2:
            continue
        current_inversion_raw = decode_inversion_raw(chord, theory_ctx)
        if current_inversion_raw is None:
            current_inversion_raw = 0
        max_inversion_raw = min(3, len(body_pcs) - 1)
        current_inversion_raw = max(0, min(int(current_inversion_raw), max_inversion_raw))
        prev_bass_pc = _neighbor_implied_bass(song_obj, chords, chord_idx, step=-1, theory_ctx=theory_ctx)
        next_bass_pc = _neighbor_implied_bass(song_obj, chords, chord_idx, step=1, theory_ctx=theory_ctx)
        if prev_bass_pc is None and next_bass_pc is None:
            continue
        strong_position = _is_strong_chord_position(chord, song_obj)
        current_bass_pc = int(body_pcs[current_inversion_raw]) % 12
        current_badness = _bass_continuity_badness(
            candidate_bass_pc=current_bass_pc,
            inversion_raw=current_inversion_raw,
            body_len=len(body_pcs),
            prev_bass_pc=prev_bass_pc,
            next_bass_pc=next_bass_pc,
            strong_position=strong_position,
        )

        for candidate_inversion_raw in range(max_inversion_raw + 1):
            if candidate_inversion_raw == current_inversion_raw:
                continue
            candidate_bass_pc = int(body_pcs[candidate_inversion_raw]) % 12
            candidate_badness = _bass_continuity_badness(
                candidate_bass_pc=candidate_bass_pc,
                inversion_raw=candidate_inversion_raw,
                body_len=len(body_pcs),
                prev_bass_pc=prev_bass_pc,
                next_bass_pc=next_bass_pc,
                strong_position=strong_position,
            )
            badness_gain = candidate_badness - current_badness
            if badness_gain <= min_badness_gain:
                continue
            candidate_inversion_id = inversion_raw_to_id.get(int(candidate_inversion_raw))
            if candidate_inversion_id is None:
                continue
            candidates.append({
                "chord_idx": chord_idx,
                "current_inversion_raw": current_inversion_raw,
                "candidate_inversion_raw": int(candidate_inversion_raw),
                "candidate_inversion_id": candidate_inversion_id,
                "current_bass_pc": current_bass_pc,
                "candidate_bass_pc": candidate_bass_pc,
                "prev_bass_pc": prev_bass_pc,
                "next_bass_pc": next_bass_pc,
                "current_badness": current_badness,
                "candidate_badness": candidate_badness,
                "badness_gain": badness_gain,
                "strong_position": strong_position,
            })

    if not candidates:
        metadata["reason_skipped"] = "no_inversion_with_higher_bass_badness"
        return song_obj, metadata, False

    max_gain = max(candidate["badness_gain"] for candidate in candidates)
    best_candidates = [candidate for candidate in candidates if abs(candidate["badness_gain"] - max_gain) <= 1e-9]
    chosen = rng.choice(best_candidates)
    chord = chords[chosen["chord_idx"]]
    chord["inversion_id"] = chosen["candidate_inversion_id"]

    metadata.update({
        "applied": True,
        "n_chords_modified": 1,
        "chord_corrupted_indices": [chosen["chord_idx"]],
        "details": {
            "target_chord_index": chosen["chord_idx"],
            "current_inversion_raw": chosen["current_inversion_raw"],
            "new_inversion_raw": chosen["candidate_inversion_raw"],
            "current_bass_pc": chosen["current_bass_pc"],
            "new_bass_pc": chosen["candidate_bass_pc"],
            "prev_bass_pc": chosen["prev_bass_pc"],
            "next_bass_pc": chosen["next_bass_pc"],
            "current_badness": chosen["current_badness"],
            "new_badness": chosen["candidate_badness"],
            "badness_gain": chosen["badness_gain"],
            "strong_position": chosen["strong_position"],
        },
    })
    return song_obj, metadata, True


def _corrupt_note_onset_shift(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("note_onset_shift")
    max_steps = int(corruption_cfg.get("rhythm_shift_max_steps", 1))
    onset_grid = _onset_grid(song_obj)
    if len(onset_grid) < 2:
        return song_obj, metadata, False

    note_indices = list(range(len(song_obj.get("melody", []))))
    rng.shuffle(note_indices)
    for note_idx in note_indices:
        note = song_obj["melody"][note_idx]
        if int(note.get("is_rest", 0)) == 1:
            continue
        old_beat = try_parse_float(note.get("beat"))
        if old_beat is None:
            continue
        if old_beat not in onset_grid:
            continue
        pos = onset_grid.index(old_beat)
        candidates = []
        for step in range(1, max(1, max_steps) + 1):
            if pos - step >= 0:
                candidates.append(onset_grid[pos - step])
            if pos + step < len(onset_grid):
                candidates.append(onset_grid[pos + step])
        if not candidates:
            continue
        new_beat = rng.choice(candidates)
        note["beat"] = new_beat
        post_grid = _onset_grid(song_obj)
        onset_indices = _collect_post_onset_indices_for_metadata(post_grid, {old_beat, new_beat})

        metadata.update({
            "applied": True,
            "topology_changed": True,
            "note_corrupted_indices": [note_idx],
            "onset_corrupted_indices": onset_indices,
            "details": {
                "source_onset_beat": old_beat,
                "target_onset_beat": new_beat,
            },
        })
        return song_obj, metadata, True

    return song_obj, metadata, False


def _corrupt_strong_weak_beat_flip(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("strong_weak_beat_flip")
    notes = song_obj.get("melody", [])
    num_beats = safe_float(song_obj.get("meta", {}).get("main_num_beats", 4.0), 4.0)

    strong_offsets = {0.0}
    if abs(num_beats - 4.0) < 1e-6:
        strong_offsets.add(2.0)

    weak_offsets = {float(x) for x in range(int(max(1.0, num_beats)))} - strong_offsets
    if not weak_offsets:
        weak_offsets = {1.0}

    indices = list(range(len(notes)))
    rng.shuffle(indices)
    for note_idx in indices:
        note = notes[note_idx]
        if int(note.get("is_rest", 0)) == 1:
            continue
        old_beat = try_parse_float(note.get("beat"))
        if old_beat is None:
            continue
        bar_idx = int((old_beat - 1.0) // num_beats)
        bar_start = 1.0 + bar_idx * num_beats
        old_pos = old_beat - bar_start
        on_strong = any(abs(old_pos - s) < 1e-6 for s in strong_offsets)

        target_offsets = sorted(weak_offsets if on_strong else strong_offsets)
        if not target_offsets:
            continue
        new_pos = float(rng.choice(target_offsets))
        new_beat = bar_start + new_pos
        if abs(new_beat - old_beat) < 1e-6:
            continue

        note["beat"] = new_beat
        post_grid = _onset_grid(song_obj)
        onset_indices = _collect_post_onset_indices_for_metadata(post_grid, {old_beat, new_beat})
        metadata.update({
            "applied": True,
            "topology_changed": True,
            "note_corrupted_indices": [note_idx],
            "onset_corrupted_indices": onset_indices,
            "details": {
                "source_onset_beat": old_beat,
                "target_onset_beat": new_beat,
                "flip_direction": "strong_to_weak" if on_strong else "weak_to_strong",
            },
        })
        return song_obj, metadata, True

    return song_obj, metadata, False


def _corrupt_drop_note_from_onset(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("drop_note_from_onset")
    melody_key, melody = _melody_events(song_obj)
    if not melody_key or not melody:
        metadata["reason_skipped"] = "melody_track_not_found"
        return song_obj, metadata, False

    candidates: list[dict] = []
    note_indices = list(range(len(melody)))
    rng.shuffle(note_indices)
    for note_idx in note_indices:
        note = melody[note_idx]
        if int(note.get("is_rest", 0)) == 1:
            continue
        start, end = _note_interval(note)
        if start is None or end is None:
            continue
        duration = max(0.0, float(end - start))
        if duration <= DEFAULT_EPSILON:
            continue
        covering_chord_index = find_covering_chord_index(song_obj, note)
        same_onset_with_chord = False
        if covering_chord_index is not None:
            chord_start = try_parse_float(song_obj["chords"][covering_chord_index].get("beat"))
            same_onset_with_chord = chord_start is not None and abs(float(chord_start) - float(start)) <= DEFAULT_EPSILON
        strong_note_position = is_strong_note_position(note, song_obj)
        score = 0.25 * duration
        if covering_chord_index is not None:
            score += 1.0
        if same_onset_with_chord:
            score += 3.0
        if strong_note_position:
            score += 2.0
        candidates.append({
            "note_idx": note_idx,
            "start": float(start),
            "duration": duration,
            "covering_chord_index": covering_chord_index,
            "same_onset_with_chord": same_onset_with_chord,
            "strong_note_position": strong_note_position,
            "score": score,
            "original_sd_id": note.get("sd_id"),
            "original_octave_id": note.get("octave_id"),
        })

    if not candidates:
        metadata["reason_skipped"] = "no_droppable_non_rest_note"
        return song_obj, metadata, False

    max_score = max(candidate["score"] for candidate in candidates)
    best_candidates = [candidate for candidate in candidates if abs(candidate["score"] - max_score) <= 1e-9]
    chosen = rng.choice(best_candidates)
    note = melody[chosen["note_idx"]]
    _restify_note(note)

    metadata.update({
        "applied": True,
        "n_notes_modified": 1,
        "note_corrupted_indices": [chosen["note_idx"]],
        "corruption_params": {"drop_mode": "restify"},
        "details": {
            "melody_key": melody_key,
            "source_onset_beat": chosen["start"],
            "duration_before": chosen["duration"],
            "covering_chord_index": chosen["covering_chord_index"],
            "same_onset_with_chord": chosen["same_onset_with_chord"],
            "strong_note_position": chosen["strong_note_position"],
            "original_sd_id": chosen["original_sd_id"],
            "original_octave_id": chosen["original_octave_id"],
        },
    })
    return song_obj, metadata, True


def _corrupt_drop_chord_from_onset(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("drop_chord_from_onset")
    chords = song_obj.get("chords", [])
    if not chords:
        metadata["reason_skipped"] = "no_chords_found"
        return song_obj, metadata, False

    _, melody = _melody_events(song_obj)
    candidates: list[dict] = []
    chord_indices = list(range(len(chords)))
    rng.shuffle(chord_indices)
    for chord_idx in chord_indices:
        chord = chords[chord_idx]
        if int(chord.get("is_rest", 0)) == 1:
            continue
        start, end = _note_interval(chord)
        if start is None or end is None:
            continue
        duration = max(0.0, float(end - start))
        if duration <= DEFAULT_EPSILON:
            continue
        overlapping_melody_indices = _collect_overlapping_melody_indices(song_obj, chord)
        same_onset_melody_indices = [
            note_idx
            for note_idx in overlapping_melody_indices
            if abs(safe_float(melody[note_idx].get("beat"), -999.0) - float(start)) <= DEFAULT_EPSILON
        ] if melody else []
        strong_chord_position = _is_strong_chord_position(chord, song_obj)
        score = 0.25 * duration + min(2.0, float(len(overlapping_melody_indices)))
        if same_onset_melody_indices:
            score += 4.0
        if strong_chord_position:
            score += 2.0
        candidates.append({
            "chord_idx": chord_idx,
            "start": float(start),
            "duration": duration,
            "overlapping_melody_indices": overlapping_melody_indices,
            "same_onset_melody_indices": same_onset_melody_indices,
            "strong_chord_position": strong_chord_position,
            "score": score,
        })

    if not candidates:
        metadata["reason_skipped"] = "no_droppable_non_rest_chord"
        return song_obj, metadata, False

    max_score = max(candidate["score"] for candidate in candidates)
    best_candidates = [candidate for candidate in candidates if abs(candidate["score"] - max_score) <= 1e-9]
    chosen = rng.choice(best_candidates)
    chord = chords[chosen["chord_idx"]]
    _restify_chord(chord, zero_duration=True)
    _sync_pos_in_bar_if_present(chord, song_obj)

    metadata.update({
        "applied": True,
        "topology_changed": True,
        "n_chords_modified": 1,
        "chord_corrupted_indices": [chosen["chord_idx"]],
        "corruption_params": {"drop_mode": "restify_zero_duration"},
        "details": {
            "source_onset_beat": chosen["start"],
            "duration_before": chosen["duration"],
            "overlapping_melody_indices": chosen["overlapping_melody_indices"],
            "same_onset_melody_indices": chosen["same_onset_melody_indices"],
            "strong_chord_position": chosen["strong_chord_position"],
        },
    })
    return song_obj, metadata, True


def _corrupt_chord_onset_shift(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("chord_onset_shift")
    chords = song_obj.get("chords", [])
    if not chords:
        metadata["reason_skipped"] = "no_chords_found"
        return song_obj, metadata, False

    max_steps = int(corruption_cfg.get("chord_shift_max_steps", corruption_cfg.get("rhythm_shift_max_steps", 1)))
    onset_grid = _onset_grid(song_obj)
    if len(onset_grid) < 2:
        metadata["reason_skipped"] = "insufficient_onset_grid"
        return song_obj, metadata, False

    _, melody = _melody_events(song_obj)
    candidates: list[dict] = []
    chord_indices = list(range(len(chords)))
    rng.shuffle(chord_indices)
    for chord_idx in chord_indices:
        chord = chords[chord_idx]
        if int(chord.get("is_rest", 0)) == 1:
            continue
        old_beat = try_parse_float(chord.get("beat"))
        if old_beat is None or old_beat not in onset_grid:
            continue
        start, end = _note_interval(chord)
        if start is None or end is None:
            continue
        duration = max(0.0, float(end - start))
        pos = onset_grid.index(old_beat)
        original_same_onset_melody_indices = [
            note_idx
            for note_idx, note in enumerate(melody or [])
            if int(note.get("is_rest", 0)) == 0 and abs(safe_float(note.get("beat"), -999.0) - float(old_beat)) <= DEFAULT_EPSILON
        ]
        for step in range(1, max(1, max_steps) + 1):
            for target_pos in (pos - step, pos + step):
                if target_pos < 0 or target_pos >= len(onset_grid):
                    continue
                new_beat = float(onset_grid[target_pos])
                if abs(new_beat - float(old_beat)) <= DEFAULT_EPSILON:
                    continue
                target_same_onset_melody_indices = [
                    note_idx
                    for note_idx, note in enumerate(melody or [])
                    if int(note.get("is_rest", 0)) == 0 and abs(safe_float(note.get("beat"), -999.0) - new_beat) <= DEFAULT_EPSILON
                ]
                strong_chord_position = _is_strong_chord_position(chord, song_obj)
                score = 0.0
                if strong_chord_position:
                    score += 2.0
                if original_same_onset_melody_indices:
                    score += 4.0
                if target_same_onset_melody_indices:
                    score += 1.0
                score += 0.1 / float(step)
                candidates.append({
                    "chord_idx": chord_idx,
                    "old_beat": float(old_beat),
                    "new_beat": new_beat,
                    "duration": duration,
                    "original_same_onset_melody_indices": original_same_onset_melody_indices,
                    "target_same_onset_melody_indices": target_same_onset_melody_indices,
                    "strong_chord_position": strong_chord_position,
                    "score": score,
                })

    if not candidates:
        metadata["reason_skipped"] = "no_shiftable_chord_onset"
        return song_obj, metadata, False

    max_score = max(candidate["score"] for candidate in candidates)
    best_candidates = [candidate for candidate in candidates if abs(candidate["score"] - max_score) <= 1e-9]
    chosen = rng.choice(best_candidates)
    chord = chords[chosen["chord_idx"]]
    _set_note_interval(chord, chosen["new_beat"], chosen["new_beat"] + chosen["duration"])
    _sync_pos_in_bar_if_present(chord, song_obj)
    post_grid = _onset_grid(song_obj)
    onset_indices = _collect_post_onset_indices_for_metadata(post_grid, {chosen["old_beat"], chosen["new_beat"]})

    metadata.update({
        "applied": True,
        "topology_changed": True,
        "n_chords_modified": 1,
        "chord_corrupted_indices": [chosen["chord_idx"]],
        "onset_corrupted_indices": onset_indices,
        "corruption_params": {"max_steps": max(1, max_steps)},
        "details": {
            "source_onset_beat": chosen["old_beat"],
            "target_onset_beat": chosen["new_beat"],
            "duration_before": chosen["duration"],
            "original_same_onset_melody_indices": chosen["original_same_onset_melody_indices"],
            "target_same_onset_melody_indices": chosen["target_same_onset_melody_indices"],
            "strong_chord_position": chosen["strong_chord_position"],
        },
    })
    return song_obj, metadata, True


def _corrupt_duration_stretch_shrink_note(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("duration_stretch_shrink_note")
    melody_key, melody = _melody_events(song_obj)
    if not melody_key or not melody:
        metadata["reason_skipped"] = "melody_track_not_found"
        return song_obj, metadata, False

    scale_factors = _duration_scale_factors(corruption_cfg, "note_duration_scale_factors")
    if not scale_factors:
        metadata["reason_skipped"] = "no_note_duration_scale_factors"
        return song_obj, metadata, False
    min_duration = safe_float(corruption_cfg.get("min_note_duration_beats", corruption_cfg.get("duration_min_beats", 0.25)), 0.25)

    candidates: list[dict] = []
    note_indices = list(range(len(melody)))
    rng.shuffle(note_indices)
    for note_idx in note_indices:
        note = melody[note_idx]
        if int(note.get("is_rest", 0)) == 1:
            continue
        start, end = _note_interval(note)
        if start is None or end is None:
            continue
        old_duration = max(0.0, float(end - start))
        if old_duration <= DEFAULT_EPSILON:
            continue
        next_start = None
        if note_idx + 1 < len(melody):
            next_start, _ = _note_interval(melody[note_idx + 1])
        covering_chord_index = find_covering_chord_index(song_obj, note)
        same_onset_with_chord = False
        if covering_chord_index is not None:
            chord_start = try_parse_float(song_obj["chords"][covering_chord_index].get("beat"))
            same_onset_with_chord = chord_start is not None and abs(float(chord_start) - float(start)) <= DEFAULT_EPSILON
        strong_note_position = is_strong_note_position(note, song_obj)
        for factor in scale_factors:
            new_duration = old_duration * float(factor)
            if new_duration < min_duration or abs(new_duration - old_duration) <= DEFAULT_EPSILON:
                continue
            new_end = float(start) + new_duration
            if factor > 1.0 and next_start is not None and new_end > float(next_start) - DEFAULT_EPSILON:
                continue
            score = 0.1 * old_duration
            if strong_note_position:
                score += 2.0
            if same_onset_with_chord:
                score += 1.0
            if factor < 1.0:
                score += 0.25
            candidates.append({
                "note_idx": note_idx,
                "start": float(start),
                "old_duration": old_duration,
                "new_duration": new_duration,
                "factor": float(factor),
                "covering_chord_index": covering_chord_index,
                "same_onset_with_chord": same_onset_with_chord,
                "strong_note_position": strong_note_position,
                "score": score,
            })

    if not candidates:
        metadata["reason_skipped"] = "no_note_duration_candidate"
        return song_obj, metadata, False

    max_score = max(candidate["score"] for candidate in candidates)
    best_candidates = [candidate for candidate in candidates if abs(candidate["score"] - max_score) <= 1e-9]
    chosen = rng.choice(best_candidates)
    note = melody[chosen["note_idx"]]
    _set_note_interval(note, chosen["start"], chosen["start"] + chosen["new_duration"])

    metadata.update({
        "applied": True,
        "n_notes_modified": 1,
        "note_corrupted_indices": [chosen["note_idx"]],
        "corruption_params": {"duration_scale_factor": chosen["factor"]},
        "details": {
            "melody_key": melody_key,
            "source_onset_beat": chosen["start"],
            "original_duration": chosen["old_duration"],
            "new_duration": chosen["new_duration"],
            "covering_chord_index": chosen["covering_chord_index"],
            "same_onset_with_chord": chosen["same_onset_with_chord"],
            "strong_note_position": chosen["strong_note_position"],
        },
    })
    return song_obj, metadata, True


def _corrupt_duration_stretch_shrink_chord(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("duration_stretch_shrink_chord")
    chords = song_obj.get("chords", [])
    if not chords:
        metadata["reason_skipped"] = "no_chords_found"
        return song_obj, metadata, False

    scale_factors = _duration_scale_factors(corruption_cfg, "chord_duration_scale_factors")
    if not scale_factors:
        metadata["reason_skipped"] = "no_chord_duration_scale_factors"
        return song_obj, metadata, False
    min_duration = safe_float(corruption_cfg.get("min_chord_duration_beats", corruption_cfg.get("duration_min_beats", 0.25)), 0.25)

    _, melody = _melody_events(song_obj)
    candidates: list[dict] = []
    chord_indices = list(range(len(chords)))
    rng.shuffle(chord_indices)
    for chord_idx in chord_indices:
        chord = chords[chord_idx]
        if int(chord.get("is_rest", 0)) == 1:
            continue
        start, end = _note_interval(chord)
        if start is None or end is None:
            continue
        old_duration = max(0.0, float(end - start))
        if old_duration <= DEFAULT_EPSILON:
            continue
        overlapping_melody_indices = _collect_overlapping_melody_indices(song_obj, chord)
        same_onset_melody_indices = [
            note_idx
            for note_idx in overlapping_melody_indices
            if melody and abs(safe_float(melody[note_idx].get("beat"), -999.0) - float(start)) <= DEFAULT_EPSILON
        ]
        strong_chord_position = _is_strong_chord_position(chord, song_obj)
        for factor in scale_factors:
            new_duration = old_duration * float(factor)
            if new_duration < min_duration or abs(new_duration - old_duration) <= DEFAULT_EPSILON:
                continue
            score = 0.1 * old_duration + min(2.0, 0.5 * float(len(overlapping_melody_indices)))
            if same_onset_melody_indices:
                score += 2.0
            if strong_chord_position:
                score += 2.0
            if factor > 1.0:
                score += 0.25
            candidates.append({
                "chord_idx": chord_idx,
                "start": float(start),
                "old_duration": old_duration,
                "new_duration": new_duration,
                "factor": float(factor),
                "overlapping_melody_indices": overlapping_melody_indices,
                "same_onset_melody_indices": same_onset_melody_indices,
                "strong_chord_position": strong_chord_position,
                "score": score,
            })

    if not candidates:
        metadata["reason_skipped"] = "no_chord_duration_candidate"
        return song_obj, metadata, False

    max_score = max(candidate["score"] for candidate in candidates)
    best_candidates = [candidate for candidate in candidates if abs(candidate["score"] - max_score) <= 1e-9]
    chosen = rng.choice(best_candidates)
    chord = chords[chosen["chord_idx"]]
    _set_note_interval(chord, chosen["start"], chosen["start"] + chosen["new_duration"])
    _sync_pos_in_bar_if_present(chord, song_obj)

    metadata.update({
        "applied": True,
        "topology_changed": True,
        "n_chords_modified": 1,
        "chord_corrupted_indices": [chosen["chord_idx"]],
        "corruption_params": {"duration_scale_factor": chosen["factor"]},
        "details": {
            "source_onset_beat": chosen["start"],
            "original_duration": chosen["old_duration"],
            "new_duration": chosen["new_duration"],
            "overlapping_melody_indices": chosen["overlapping_melody_indices"],
            "same_onset_melody_indices": chosen["same_onset_melody_indices"],
            "strong_chord_position": chosen["strong_chord_position"],
        },
    })
    return song_obj, metadata, True


def _strict_slot_to_replacement_roots(mode_name: str, current_slot: str, theory_ctx: dict) -> list[int]:
    mode_key = "minor" if mode_name == "minor" else "major"
    rules = theory_ctx["strict_functions_v1"][mode_key]
    if current_slot == "T":
        return rules["PD_roots_raw"] + rules["D_roots_raw"]
    if current_slot == "PD":
        return rules["T_roots_raw"] + rules["D_roots_raw"]
    if current_slot == "D":
        return rules["T_roots_raw"] + rules["PD_roots_raw"]
    return []


def _corrupt_functional_progression_violation(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("functional_progression_violation_strict")
    chords = song_obj.get("chords", [])
    if len(chords) < 3:
        return song_obj, metadata, False

    main_mode = theory_ctx["scale_id_to_name"].get(int(song_obj.get("meta", {}).get("main_key_scale_id", 2)), "major")
    none_kind_id = next((idx for idx, name in theory_ctx["borrowed_id_to_kind"].items() if name == "none"), None)
    if none_kind_id is None:
        return song_obj, metadata, False
    triplets = list(range(1, len(chords) - 1))
    rng.shuffle(triplets)

    for idx in triplets:
        prev_chord, curr_chord, next_chord = chords[idx - 1], chords[idx], chords[idx + 1]
        if any(int(ch.get("is_rest", 0)) == 1 for ch in (prev_chord, curr_chord, next_chord)):
            continue
        if any(int(ch.get("borrowed_kind_id", 0)) != none_kind_id for ch in (prev_chord, curr_chord, next_chord)):
            continue
        if any(theory_ctx["applied_id_to_raw"].get(int(ch.get("applied_id", 0)), 0) != 0 for ch in (prev_chord, curr_chord, next_chord)):
            continue

        p_raw = decode_root_raw(prev_chord, theory_ctx)
        c_raw = decode_root_raw(curr_chord, theory_ctx)
        n_raw = decode_root_raw(next_chord, theory_ctx)
        if None in (p_raw, c_raw, n_raw):
            continue
        if min(p_raw, c_raw, n_raw) < 0:
            continue
        if any(root not in {0, 1, 3, 4} for root in (p_raw, c_raw, n_raw)):
            continue

        f_prev = classify_function_from_root_raw(p_raw, main_mode, theory_ctx)
        f_curr = classify_function_from_root_raw(c_raw, main_mode, theory_ctx)
        f_next = classify_function_from_root_raw(n_raw, main_mode, theory_ctx)
        if None in (f_prev, f_curr, f_next):
            continue

        if (f_prev, f_curr, f_next) not in STRICT_TRIPLET_PATTERNS_V1:
            continue

        replacement_roots_raw = _strict_slot_to_replacement_roots(main_mode, f_curr, theory_ctx)
        replacement_root_ids = [raw + 1 for raw in replacement_roots_raw if raw in {0, 1, 3, 4}]
        current_root_id = int(curr_chord.get("root_id", 0))
        replacement_root_ids = [x for x in replacement_root_ids if x != current_root_id]
        if not replacement_root_ids:
            continue

        new_root_id = int(rng.choice(replacement_root_ids))
        curr_chord["root_id"] = new_root_id

        metadata.update({
            "applied": True,
            "chord_corrupted_indices": [idx],
            "details": {
                "original_root_id": current_root_id,
                "new_root_id": new_root_id,
                "triplet_functions_before": [f_prev, f_curr, f_next],
            },
        })
        return song_obj, metadata, True

    return song_obj, metadata, False


def _corrupt_out_of_key_note(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("out_of_key_note")
    note_indices = list(range(len(song_obj.get("melody", []))))
    rng.shuffle(note_indices)

    for note_idx in note_indices:
        note = song_obj["melody"][note_idx]
        if int(note.get("is_rest", 0)) == 1:
            continue
        old_sd = int(note.get("sd_id", 0))
        old_pc = decode_sd_to_chromatic(old_sd, theory_ctx)
        if old_pc is None:
            continue

        chord_idx = find_covering_chord_index(song_obj, note)
        chord = song_obj["chords"][chord_idx] if chord_idx is not None and chord_idx < len(song_obj.get("chords", [])) else None
        mode_name = select_active_mode_name(song_obj, chord, theory_ctx)
        allowed_pcs = set(theory_ctx["mode_to_pcset"].get(mode_name, theory_ctx["mode_to_pcset"]["major"]))
        out_of_key_pcs = {pc for pc in range(12) if pc not in allowed_pcs}
        if not out_of_key_pcs:
            continue
        out_of_key_pcs.discard(old_pc)
        if not out_of_key_pcs:
            continue
        new_sd = _pick_new_sd_id(exclude_pcs=set(), include_pcs=out_of_key_pcs, theory_ctx=theory_ctx, rng=rng)
        if new_sd is None or new_sd == old_sd:
            continue

        new_pc = decode_sd_to_chromatic(new_sd, theory_ctx)
        if new_pc is None or new_pc == old_pc:
            continue

        note["sd_id"] = new_sd
        metadata.update({
            "applied": True,
            "note_corrupted_indices": [note_idx],
            "details": {
                "reason": "out_of_key_note",
                "original_sd_id": old_sd,
                "new_sd_id": new_sd,
                "active_mode_name": mode_name,
                "covering_chord_index": chord_idx,
            },
        })
        return song_obj, metadata, True

    return song_obj, metadata, False


def _corrupt_local_semitone_fragment_shift(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("local_semitone_fragment_shift")
    notes = song_obj.get("melody", [])
    n_notes = len(notes)
    if n_notes < 2:
        return song_obj, metadata, False

    fragment_lengths = [2, 3, 4]
    rng.shuffle(fragment_lengths)
    shift_options = [-1, 1]
    rng.shuffle(shift_options)

    for frag_len in fragment_lengths:
        if frag_len > n_notes:
            continue
        start_indices = list(range(0, n_notes - frag_len + 1))
        rng.shuffle(start_indices)
        for start_idx in start_indices:
            frag_indices = list(range(start_idx, start_idx + frag_len))
            fragment_notes = [notes[i] for i in frag_indices]
            if any(int(n.get("is_rest", 0)) == 1 for n in fragment_notes):
                continue

            for shift in shift_options:
                original_sd_ids = [int(n.get("sd_id", 0)) for n in fragment_notes]
                original_pcs = [decode_sd_to_chromatic(sd_id, theory_ctx) for sd_id in original_sd_ids]
                if any(pc is None for pc in original_pcs):
                    continue

                new_sd_ids = []
                valid = True
                for sd_id, pc in zip(original_sd_ids, original_pcs):
                    target_pc = (int(pc) + shift) % 12
                    new_sd = _pick_sd_id_for_pc(target_pc, theory_ctx, rng, exclude_sd_ids={sd_id})
                    if new_sd is None:
                        valid = False
                        break
                    new_sd_ids.append(new_sd)
                if not valid:
                    continue

                for idx, new_sd in zip(frag_indices, new_sd_ids):
                    notes[idx]["sd_id"] = new_sd

                metadata.update({
                    "applied": True,
                    "note_corrupted_indices": frag_indices,
                    "details": {
                        "fragment_note_indices": frag_indices,
                        "shift_semitones": int(shift),
                        "original_sd_ids": original_sd_ids,
                        "new_sd_ids": new_sd_ids,
                    },
                })
                return song_obj, metadata, True

    return song_obj, metadata, False


def _corrupt_octave_leap_violation(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("octave_leap_violation")
    notes = song_obj.get("melody", [])
    if len(notes) < 2:
        return song_obj, metadata, False

    octave_min = int(corruption_cfg.get("octave_min_id", 1))
    octave_max = int(corruption_cfg.get("octave_max_id", 8))
    shifts = [2, -2, 1, -1]

    candidate_indices = list(range(len(notes)))
    rng.shuffle(candidate_indices)
    for note_idx in candidate_indices:
        note = notes[note_idx]
        if int(note.get("is_rest", 0)) == 1:
            continue
        if note_idx > 0 and int(notes[note_idx - 1].get("is_rest", 0)) == 0:
            neighbor_idx = note_idx - 1
        elif note_idx + 1 < len(notes) and int(notes[note_idx + 1].get("is_rest", 0)) == 0:
            neighbor_idx = note_idx + 1
        else:
            continue

        old_oct = int(note.get("octave_id", 0))
        for shift in shifts:
            new_oct = old_oct + shift
            if not (octave_min <= new_oct <= octave_max):
                continue
            if new_oct == old_oct:
                continue
            note["octave_id"] = new_oct
            metadata.update({
                "applied": True,
                "note_corrupted_indices": [note_idx],
                "details": {
                    "reason": "octave_leap_violation",
                    "target_note_index": note_idx,
                    "neighbor_note_index": neighbor_idx,
                    "original_octave_id": old_oct,
                    "new_octave_id": new_oct,
                    "octave_shift": shift,
                    "neighbor_octave_id": int(notes[neighbor_idx].get("octave_id", 0)),
                },
            })
            return song_obj, metadata, True

    return song_obj, metadata, False


def _corrupt_semitone_from_bass_or_chord_tone(song_obj, theory_ctx, rng, corruption_cfg):
    metadata = _identity_metadata("semitone_from_bass_or_chord_tone")
    note_indices = list(range(len(song_obj.get("melody", []))))
    rng.shuffle(note_indices)

    for note_idx in note_indices:
        note = song_obj["melody"][note_idx]
        if int(note.get("is_rest", 0)) == 1:
            continue
        chord_idx = find_covering_chord_index(song_obj, note)
        if chord_idx is None:
            continue
        chord = song_obj["chords"][chord_idx]
        if int(chord.get("is_rest", 0)) == 1:
            continue

        bass_top = chord_bass_and_top_pcs(song_obj, chord, theory_ctx)
        if bass_top is None:
            continue
        bass_pc, top_pc = bass_top
        chord_pcs = chord_pitch_classes_tertian(song_obj, chord, theory_ctx)
        if not chord_pcs:
            continue

        old_sd = int(note.get("sd_id", 0))
        candidate_refs = [("bass", bass_pc), ("top_voice", top_pc)]
        rng.shuffle(candidate_refs)

        for role, ref_pc in candidate_refs:
            conflict_pcs = {(ref_pc + 1) % 12, (ref_pc - 1) % 12}
            conflict_pcs = {pc for pc in conflict_pcs if pc not in chord_pcs and pc not in {bass_pc, top_pc}}
            if not conflict_pcs:
                continue
            target_pc = int(rng.choice(sorted(conflict_pcs)))
            new_sd = _pick_sd_id_for_pc(target_pc, theory_ctx, rng, exclude_sd_ids={old_sd})
            if new_sd is None:
                continue

            note["sd_id"] = new_sd
            metadata.update({
                "applied": True,
                "note_corrupted_indices": [note_idx],
                "chord_corrupted_indices": [chord_idx],
                "details": {
                    "target_note_index": note_idx,
                    "covering_chord_index": chord_idx,
                    "original_sd_id": old_sd,
                    "new_sd_id": new_sd,
                    "target_conflict_pc": target_pc,
                    "reference_pc": ref_pc,
                    "reference_role": role,
                },
            })
            return song_obj, metadata, True

    return song_obj, metadata, False


def _not_implemented_mode(song_obj, theory_ctx, rng, corruption_cfg, mode_name: str):
    metadata = _identity_metadata(mode_name)
    metadata["details"] = {"reason": "registered_but_not_implemented"}
    return song_obj, metadata, False


_CORRUPTION_REGISTRY: dict[str, Callable] = {
    "adjacent_section_swap": _corrupt_adjacent_section_swap,
    "non_adjacent_section_swap": _corrupt_non_adjacent_section_swap,
    "section_duplicate": _corrupt_section_duplicate,
    "section_drop_keep_silence": _corrupt_section_drop_keep_silence,
    "section_drop_and_close_gap": _corrupt_section_drop_and_close_gap,
    "section_exit_non_dominant_substitution": _corrupt_section_exit_non_dominant_substitution,
    "section_entry_non_tonic_substitution": _corrupt_section_entry_non_tonic_substitution,
    "transpose_with_tonic_shift": _corrupt_transpose_with_tonic_shift,
    "melody_octave_shift": _corrupt_melody_octave_shift,
    "merge_repeated_melody_notes": _corrupt_merge_repeated_melody_notes,
    "split_long_melody_note": _corrupt_split_long_melody_note,
    "drop_tonic_seventh_on_strong_beat": _corrupt_drop_tonic_seventh_on_strong_beat,
    "drop_note_from_onset": _corrupt_drop_note_from_onset,
    "drop_chord_from_onset": _corrupt_drop_chord_from_onset,
    "strongbeat_nonchord_note": _corrupt_strongbeat_nonchord_note,
    "borrowed_melody_conflict": _corrupt_borrowed_melody_conflict,
    "borrowed_kind_toggle_without_melody_change": _corrupt_borrowed_kind_toggle,
    "melody_semitone_add_clash": _corrupt_melody_semitone_add_clash,
    "melody_suspension_clash": _corrupt_melody_suspension_clash,
    "melody_alteration_clash": _corrupt_melody_alteration_clash,
    "melody_omit_core_tone_conflict": _corrupt_melody_omit_core_tone_conflict,
    "inversion_bass_continuity_conflict": _corrupt_inversion_bass_continuity_conflict,
    "chord_onset_shift": _corrupt_chord_onset_shift,
    "note_onset_shift": _corrupt_note_onset_shift,
    "strong_weak_beat_flip": _corrupt_strong_weak_beat_flip,
    "duration_stretch_shrink_note": _corrupt_duration_stretch_shrink_note,
    "duration_stretch_shrink_chord": _corrupt_duration_stretch_shrink_chord,
    "functional_progression_violation_strict": _corrupt_functional_progression_violation,
    "out_of_key_note": _corrupt_out_of_key_note,
    "local_semitone_fragment_shift": _corrupt_local_semitone_fragment_shift,
    "octave_leap_violation": _corrupt_octave_leap_violation,
    "semitone_from_bass_or_chord_tone": _corrupt_semitone_from_bass_or_chord_tone,
}

_PLACEHOLDER_MODES = {
    "applied_resolution_violation",
}


def corrupt_song_obj(song_obj, corruption_modes, corruption_cfg, theory_ctx, rng=None, shuffle_modes=True):
    """Apply a song-level corruption and return (song, metadata)."""
    rng = rng or random
    song_corrupted = copy.deepcopy(song_obj)
    requested_modes = list(corruption_modes or _CORRUPTION_REGISTRY.keys())

    available_modes = [m for m in requested_modes if m in _CORRUPTION_REGISTRY or m in _PLACEHOLDER_MODES]
    if not available_modes:
        return song_corrupted, _identity_metadata("identity")

    if shuffle_modes:
        rng.shuffle(available_modes)
    last_metadata = None
    attempted_modes = []
    skipped_attempts = []
    for mode in available_modes:
        attempted_modes.append(mode)
        if mode in _PLACEHOLDER_MODES:
            _, metadata, _ = _not_implemented_mode(song_corrupted, theory_ctx, rng, corruption_cfg, mode)
            metadata["corruption_name"] = mode
            metadata["applied"] = False
            if not metadata.get("reason_skipped"):
                metadata["reason_skipped"] = "registered_but_not_implemented"
            skipped_attempts.append({"mode": mode, "reason": str(metadata.get("reason_skipped") or "unknown")})
            last_metadata = metadata
            continue

        song_candidate = copy.deepcopy(song_corrupted)
        _, metadata, applied = _CORRUPTION_REGISTRY[mode](song_candidate, theory_ctx, rng, corruption_cfg)
        if applied:
            metadata["corruption_name"] = mode
            metadata["reason_skipped"] = None
            metadata["attempted_corruption_modes"] = list(attempted_modes)
            metadata["skipped_corruption_attempts"] = list(skipped_attempts)
            return song_candidate, metadata
        metadata["corruption_name"] = mode
        metadata["applied"] = False
        if not metadata.get("reason_skipped"):
            metadata["reason_skipped"] = "not_applicable"
        skipped_attempts.append({"mode": mode, "reason": str(metadata.get("reason_skipped") or "unknown")})
        last_metadata = metadata

    if last_metadata is not None:
        last_metadata["attempted_corruption_modes"] = list(attempted_modes)
        last_metadata["skipped_corruption_attempts"] = list(skipped_attempts)
        return song_corrupted, last_metadata
    identity = _identity_metadata("identity")
    identity["corruption_name"] = "identity"
    identity["reason_skipped"] = "no_applicable_corruption_found"
    identity["attempted_corruption_modes"] = list(attempted_modes)
    identity["skipped_corruption_attempts"] = list(skipped_attempts)
    return song_corrupted, identity
