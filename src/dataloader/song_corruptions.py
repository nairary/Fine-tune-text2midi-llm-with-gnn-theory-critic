"""Song-object-level theory-aware corruptions."""

from __future__ import annotations

import copy
import random
from typing import Callable

from .function_rules import STRICT_TRIPLET_PATTERNS_V1
from .theory_helpers import (
    chord_bass_and_top_pcs,
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

    for chord_idx in chord_indices:
        chord = song_obj["chords"][chord_idx]
        old_kind_id = int(chord.get("borrowed_kind_id", 2))
        old_mode_id = int(chord.get("borrowed_mode_name_id", 2))
        old_kind = theory_ctx["borrowed_id_to_kind"].get(old_kind_id, "none")
        old_mode = theory_ctx["borrowed_mode_id_to_name"].get(old_mode_id)

        options: list[tuple[int, int]] = [(none_kind_id, none_mode_id)]
        options.extend((mode_name_kind_id, mode_id) for mode_id in valid_mode_ids)
        rng.shuffle(options)

        for new_kind_id, new_mode_id in options:
            if new_kind_id == old_kind_id and new_mode_id == old_mode_id:
                continue
            new_kind = theory_ctx["borrowed_id_to_kind"].get(new_kind_id)
            new_mode = theory_ctx["borrowed_mode_id_to_name"].get(new_mode_id)
            chord["borrowed_kind_id"] = new_kind_id
            chord["borrowed_mode_name_id"] = new_mode_id
            chord["borrowed_pcset_vec"] = _mode_to_pcset_vec(new_mode if new_kind == "mode_name" else None, theory_ctx)

            metadata.update({
                "applied": True,
                "chord_corrupted_indices": [chord_idx],
                "details": {
                    "borrowed_kind_before": old_kind,
                    "borrowed_kind_after": new_kind,
                    "borrowed_mode_before": old_mode,
                    "borrowed_mode_after": new_mode,
                },
            })
            return song_obj, metadata, True

    return song_obj, metadata, False


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
    "transpose_with_tonic_shift": _corrupt_transpose_with_tonic_shift,
    "melody_octave_shift": _corrupt_melody_octave_shift,
    "merge_repeated_melody_notes": _corrupt_merge_repeated_melody_notes,
    "split_long_melody_note": _corrupt_split_long_melody_note,
    "drop_tonic_seventh_on_strong_beat": _corrupt_drop_tonic_seventh_on_strong_beat,
    "strongbeat_nonchord_note": _corrupt_strongbeat_nonchord_note,
    "borrowed_melody_conflict": _corrupt_borrowed_melody_conflict,
    "borrowed_kind_toggle_without_melody_change": _corrupt_borrowed_kind_toggle,
    "note_onset_shift": _corrupt_note_onset_shift,
    "strong_weak_beat_flip": _corrupt_strong_weak_beat_flip,
    "functional_progression_violation_strict": _corrupt_functional_progression_violation,
    "out_of_key_note": _corrupt_out_of_key_note,
    "local_semitone_fragment_shift": _corrupt_local_semitone_fragment_shift,
    "octave_leap_violation": _corrupt_octave_leap_violation,
    "semitone_from_bass_or_chord_tone": _corrupt_semitone_from_bass_or_chord_tone,
}

_PLACEHOLDER_MODES = {
    "drop_note_from_onset",
    "drop_chord_from_onset",
    "chord_onset_shift",
    "duration_stretch_shrink_note",
    "duration_stretch_shrink_chord",
    "applied_resolution_violation",
}


def corrupt_song_obj(song_obj, corruption_modes, corruption_cfg, theory_ctx, rng=None):
    """Apply a random song-level corruption and return (song, metadata)."""
    rng = rng or random
    song_corrupted = copy.deepcopy(song_obj)
    requested_modes = list(corruption_modes or _CORRUPTION_REGISTRY.keys())

    available_modes = [m for m in requested_modes if m in _CORRUPTION_REGISTRY or m in _PLACEHOLDER_MODES]
    if not available_modes:
        return song_corrupted, _identity_metadata("identity")

    rng.shuffle(available_modes)
    for mode in available_modes:
        if mode in _PLACEHOLDER_MODES:
            _, metadata, _ = _not_implemented_mode(song_corrupted, theory_ctx, rng, corruption_cfg, mode)
            continue

        song_candidate = copy.deepcopy(song_corrupted)
        _, metadata, applied = _CORRUPTION_REGISTRY[mode](song_candidate, theory_ctx, rng, corruption_cfg)
        if applied:
            metadata["corruption_name"] = mode
            metadata["reason_skipped"] = None
            return song_candidate, metadata
        metadata["corruption_name"] = mode
        metadata["applied"] = False
        if not metadata.get("reason_skipped"):
            metadata["reason_skipped"] = "not_applicable"

    identity = _identity_metadata("identity")
    identity["corruption_name"] = "identity"
    identity["reason_skipped"] = "no_applicable_corruption_found"
    return song_corrupted, identity
