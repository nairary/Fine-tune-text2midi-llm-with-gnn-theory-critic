from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import HeteroData

from src.data.encode_teacher_features import (
    build_allowed_value_id_map,
    build_range_value_id_map,
    encode_vocab,
    encode_with_value_map,
)
from src.dataloader.graph_layouts import CHORD_COMPONENT_SIZES
from src.dataloader.theory_helpers import build_theory_context
from src.observer.chord_parser import predict_observer_chords_for_midi, select_target_instrument
from src.observer.schema import OBSERVER_EDGE_TYPES

ONSET_EPSILON = 1e-4


_REQUIRED_SAMPLE_FIELDS = ("song_id", "midi_path", "tonic_pc", "mode_name")
_OPTIONAL_SAMPLE_FIELDS = ("bpm", "num_beats", "beat_unit", "sample_id", "pair_group_id", "is_corrupted", "corruption_name", "source_song_id", "encoded_song_path", "tonal_group", "corruption_group", "beat_origin")


class ObserverInputValidationError(ValueError):
    """Raised when observer input JSONL rows are invalid."""


def load_observer_input_jsonl(jsonl_path: str | Path) -> list[dict[str, Any]]:
    theory_ctx = build_theory_context()
    rows: list[dict[str, Any]] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ObserverInputValidationError(f"Invalid JSON at line {line_idx}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ObserverInputValidationError(f"Line {line_idx}: row must be a JSON object")
            rows.append(_validate_observer_input_row(payload, line_idx=line_idx, theory_ctx=theory_ctx))
    return rows


def _validate_observer_input_row(sample: dict[str, Any], line_idx: int, theory_ctx: dict[str, Any]) -> dict[str, Any]:
    for key in _REQUIRED_SAMPLE_FIELDS:
        if key not in sample:
            raise ObserverInputValidationError(f"Line {line_idx}: missing required field '{key}'")

    song_id = sample["song_id"]
    midi_path = sample["midi_path"]
    tonic_pc = sample["tonic_pc"]
    mode_name = sample["mode_name"]

    if not isinstance(song_id, str) or not song_id:
        raise ObserverInputValidationError(f"Line {line_idx}: song_id must be a non-empty string")
    if not isinstance(midi_path, str) or not midi_path:
        raise ObserverInputValidationError(f"Line {line_idx}: midi_path must be a non-empty string")
    if not isinstance(tonic_pc, int) or not 0 <= tonic_pc <= 11:
        raise ObserverInputValidationError(f"Line {line_idx}: tonic_pc must be int in [0..11]")
    if mode_name not in theory_ctx["mode_to_pcset"]:
        raise ObserverInputValidationError(f"Line {line_idx}: unknown mode_name '{mode_name}'")

    validated: dict[str, Any] = {
        "song_id": song_id,
        "midi_path": midi_path,
        "tonic_pc": tonic_pc,
        "mode_name": mode_name,
    }
    for key in _OPTIONAL_SAMPLE_FIELDS:
        if key in sample:
            validated[key] = sample[key]
    return validated


def _pick_midi_tempo_bpm(pm: Any) -> float | None:
    times, tempi = pm.get_tempo_changes()
    if tempi is None or len(tempi) == 0:
        return None
    return float(tempi[0])


def _pick_midi_time_signature(pm: Any) -> tuple[int | None, int | None]:
    changes = getattr(pm, "time_signature_changes", [])
    if not changes:
        return None, None
    first = changes[0]
    numerator = int(getattr(first, "numerator", 0) or 0)
    denominator = int(getattr(first, "denominator", 0) or 0)
    if denominator == 8 and numerator in {6, 9, 12}:
        beat_unit = 3
    else:
        beat_unit = 1
    return numerator, beat_unit


def extract_observer_meta(sample: dict[str, Any], pm: Any) -> dict[str, Any]:
    bpm = float(sample["bpm"]) if sample.get("bpm") is not None else _pick_midi_tempo_bpm(pm)

    if sample.get("num_beats") is not None:
        num_beats = int(sample["num_beats"])
    else:
        num_beats, _ = _pick_midi_time_signature(pm)

    if sample.get("beat_unit") is not None:
        beat_unit = int(sample["beat_unit"])
    else:
        _, beat_unit = _pick_midi_time_signature(pm)

    end_beat = None
    if bpm is not None:
        end_beat = float(pm.get_end_time()) * float(bpm) / 60.0

    return {
        "tonic_pc": int(sample["tonic_pc"]),
        "mode_name": str(sample["mode_name"]),
        "bpm": bpm,
        "num_beats": num_beats,
        "beat_unit": beat_unit,
        "end_beat": end_beat,
    }


@lru_cache(maxsize=1)
def _load_octave_bounds() -> tuple[int, int]:
    spec_path = Path(__file__).resolve().parents[2] / "metadata" / "specs" / "spec_global.json"
    with spec_path.open("r", encoding="utf-8") as handle:
        spec_global = json.load(handle)
    return int(spec_global["octave"]["min"]), int(spec_global["octave"]["max"])


@lru_cache(maxsize=1)
def _load_teacher_global_spec() -> dict[str, Any]:
    spec_path = Path(__file__).resolve().parents[2] / "metadata" / "specs" / "spec_global.json"
    with spec_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _load_teacher_chord_sets_spec() -> dict[str, Any]:
    spec_path = Path(__file__).resolve().parents[2] / "metadata" / "specs" / "spec_chord_sets.json"
    with spec_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _load_teacher_vocabs() -> dict[str, dict[str, int]]:
    vocabs_dir = Path(__file__).resolve().parents[2] / "metadata" / "vocabs"
    with (vocabs_dir / "vocab_melody_sd.json").open("r", encoding="utf-8") as handle:
        melody_sd = json.load(handle)
    with (vocabs_dir / "vocab_key_scale.json").open("r", encoding="utf-8") as handle:
        key_scale = json.load(handle)
    with (vocabs_dir / "vocab_borrowed_kind.json").open("r", encoding="utf-8") as handle:
        borrowed_kind = json.load(handle)
    with (vocabs_dir / "vocab_borrowed_mode_name.json").open("r", encoding="utf-8") as handle:
        borrowed_mode_name = json.load(handle)
    return {
        "melody_sd": melody_sd,
        "key_scale": key_scale,
        "borrowed_kind": borrowed_kind,
        "borrowed_mode_name": borrowed_mode_name,
    }


def _octave_value_to_teacher_octave_id(octave_value: int, octave_id_map: dict[int, int]) -> int:
    return int(encode_with_value_map(octave_id_map, octave_value, unknown_id=0))


def _build_relpc_to_sd_id(theory_ctx: dict[str, Any]) -> dict[int, int]:
    sd_token_to_id = theory_ctx["sd_token_to_id"]
    sd_token_to_chromatic = theory_ctx["sd_token_to_chromatic"]
    rel_to_token: dict[int, str] = {}
    for token, chromatic in sd_token_to_chromatic.items():
        if token.startswith("bb"):
            continue
        current = rel_to_token.get(chromatic)
        if current is None:
            rel_to_token[chromatic] = token
            continue
        if ("b" not in token and "#" not in token) and ("b" in current or "#" in current):
            rel_to_token[chromatic] = token
    return {chrom: sd_token_to_id[token] for chrom, token in rel_to_token.items() if token in sd_token_to_id}


@lru_cache(maxsize=1)
def _build_runtime_id_maps() -> dict[str, dict[int, int]]:
    spec_global = _load_teacher_global_spec()
    octave_id_map = build_range_value_id_map(
        int(spec_global["octave"]["min"]),
        int(spec_global["octave"]["max"]),
        reserve_zero_for_unknown=True,
    )
    return {
        "root_id_map": build_allowed_value_id_map(spec_global["root"]["allowed_values"], reserve_zero_for_unknown=True),
        "type_id_map": build_allowed_value_id_map(spec_global["type"]["allowed_values"], reserve_zero_for_unknown=True),
        "inversion_id_map": build_allowed_value_id_map(spec_global["inversion"]["allowed_values"], reserve_zero_for_unknown=True),
        "tonic_pc_id_map": build_allowed_value_id_map(spec_global["tonic_pc"]["allowed_values"], reserve_zero_for_unknown=True),
        "num_beats_id_map": build_allowed_value_id_map(spec_global["num_beats"]["allowed_values"], reserve_zero_for_unknown=True),
        "beat_unit_id_map": build_allowed_value_id_map(spec_global["beat_unit"]["allowed_values"], reserve_zero_for_unknown=True),
        "octave_id_map": octave_id_map,
    }


def _multi_hot(values: list[Any] | None, allowed_values: list[Any]) -> list[float]:
    index = {value: idx for idx, value in enumerate(allowed_values)}
    out = [0.0] * len(allowed_values)
    for value in values or []:
        if value in index:
            out[index[value]] = 1.0
    return out


def _fixed_range_multi_hot(values: list[int] | None, size: int) -> list[float]:
    out = [0.0] * int(size)
    for value in values or []:
        if isinstance(value, int) and 0 <= value < int(size):
            out[value] = 1.0
    return out


def extract_observer_note_events(pm: Any, tonic_pc: int, bpm: float | None) -> list[dict[str, Any]]:
    theory_ctx = build_theory_context()
    vocabs = _load_teacher_vocabs()
    runtime_maps = _build_runtime_id_maps()
    relpc_to_sd_id = _build_relpc_to_sd_id(theory_ctx)

    notes: list[dict[str, Any]] = []
    melody = select_target_instrument(pm, instrument_name="melody")
    for note in melody.notes:
        onset_time = float(note.start)
        offset_time = float(note.end)
        pitch = int(note.pitch)
        pitch_class = pitch % 12
        rel_pc = (pitch_class - int(tonic_pc)) % 12
        beat = None
        duration_beats = None
        if bpm is not None:
            beat = onset_time * float(bpm) / 60.0
            duration_beats = max(0.0, (offset_time - onset_time) * float(bpm) / 60.0)

        midi_octave = pitch // 12 - 1
        octave_value = midi_octave
        octave_id = _octave_value_to_teacher_octave_id(octave_value, runtime_maps["octave_id_map"])
        sd_id = encode_vocab(vocabs["melody_sd"], theory_ctx["sd_id_to_token"].get(relpc_to_sd_id.get(rel_pc, 0), "<UNK>"))

        notes.append(
            {
                "onset_time": onset_time,
                "offset_time": offset_time,
                "beat": beat,
                "duration_beats": duration_beats,
                "pitch": pitch,
                "pitch_class": pitch_class,
                "rel_pc": rel_pc,
                "sd_id": sd_id,
                "octave_id": octave_id,
            }
        )
    notes.sort(key=lambda x: (x["onset_time"], x["pitch"]))
    return notes


def build_observer_chord_events(
    midi_path: str,
    tonic_pc: int,
    mode_name: str,
    instrument_name: str = "chords",
    weights_yaml: str | None = None,
    bpm: float | None = None,
) -> list[dict[str, Any]]:
    chords = predict_observer_chords_for_midi(
        midi_path=midi_path,
        tonic_pc=tonic_pc,
        main_mode=mode_name,
        instrument_name=instrument_name,
        weights_yaml=weights_yaml,
    )
    for chord in chords:
        if bpm is None:
            chord["beat"] = None
            chord["duration_beats"] = None
            continue
        onset_time = chord.get("onset_time")
        offset_time = chord.get("offset_time")
        if onset_time is None or offset_time is None:
            chord["beat"] = None
            chord["duration_beats"] = None
            continue
        chord["beat"] = float(onset_time) * float(bpm) / 60.0
        chord["duration_beats"] = max(0.0, float(offset_time) - float(onset_time)) * float(bpm) / 60.0
    return chords


def build_bar_events(
    end_beat: float | None,
    num_beats: int | None,
    beat_unit: int | None,
    use_fallback_44: bool = True,
) -> list[dict[str, Any]]:
    _ = beat_unit
    if num_beats is None:
        if not use_fallback_44:
            return []
        num_beats = 4
    if end_beat is None:
        end_beat = float(num_beats)

    bar_count = max(1, int((float(end_beat) + float(num_beats) - 1e-9) // float(num_beats)))
    bars: list[dict[str, Any]] = []
    for bar_index in range(bar_count):
        start = float(bar_index * num_beats)
        bars.append({"bar_index": bar_index, "start_beat": start, "end_beat": start + float(num_beats)})
    return bars


def _dedup_sorted_times(times: list[float], eps: float = ONSET_EPSILON) -> list[float]:
    if not times:
        return []
    sorted_times = sorted(float(t) for t in times)
    deduped = [sorted_times[0]]
    for t in sorted_times[1:]:
        if abs(t - deduped[-1]) > eps:
            deduped.append(t)
    return deduped


def build_onset_events(
    notes: list[dict[str, Any]],
    chords: list[dict[str, Any]],
    bars: list[dict[str, Any]],
    bpm: float | None,
    num_beats: int | None,
    eps: float = ONSET_EPSILON,
) -> list[dict[str, Any]]:
    onset_times = _dedup_sorted_times(
        [n["onset_time"] for n in notes] + [c["onset_time"] for c in chords],
        eps=eps,
    )
    if not onset_times:
        return []

    out: list[dict[str, Any]] = []
    for t in onset_times:
        beat = None if bpm is None else (float(t) * float(bpm) / 60.0)
        bar_index = None
        pos_in_bar = None
        if beat is not None and num_beats:
            bar_index = int(beat // float(num_beats))
            pos_in_bar = float(beat - bar_index * float(num_beats))
            if bars and bar_index >= len(bars):
                bar_index = len(bars) - 1
        out.append({"onset_time": t, "beat": beat, "bar_index": bar_index, "pos_in_bar": pos_in_bar})
    return out


def build_observer_song_record(
    sample: dict[str, Any],
    chord_weights_yaml: str | None = None,
    chord_instrument_name: str = "chords",
    use_fallback_44: bool = True,
) -> dict[str, Any]:
    import pretty_midi

    theory_ctx = build_theory_context()
    validated = _validate_observer_input_row(sample, line_idx=1, theory_ctx=theory_ctx)
    pm = pretty_midi.PrettyMIDI(validated["midi_path"])

    meta = extract_observer_meta(validated, pm)
    notes = extract_observer_note_events(pm, tonic_pc=meta["tonic_pc"], bpm=meta["bpm"])
    chords = build_observer_chord_events(
        midi_path=validated["midi_path"],
        tonic_pc=meta["tonic_pc"],
        mode_name=meta["mode_name"],
        instrument_name=chord_instrument_name,
        weights_yaml=chord_weights_yaml,
        bpm=meta["bpm"],
    )
    bars = build_bar_events(
        end_beat=meta["end_beat"],
        num_beats=meta["num_beats"],
        beat_unit=meta["beat_unit"],
        use_fallback_44=use_fallback_44,
    )
    onsets = build_onset_events(
        notes=notes,
        chords=chords,
        bars=bars,
        bpm=meta["bpm"],
        num_beats=meta["num_beats"] or (4 if use_fallback_44 else None),
    )

    return {
        "song_id": validated["song_id"],
        "midi_path": validated["midi_path"],
        "meta": meta,
        "notes": notes,
        "chords": chords,
        "bars": bars,
        "onsets": onsets,
    }


def build_observer_graph(record: dict[str, Any]) -> HeteroData:
    graph = HeteroData()

    bars = record.get("bars", [])
    onsets = record.get("onsets", [])
    notes = record.get("notes", [])
    chords = record.get("chords", [])
    meta = record.get("meta", {})
    num_beats = meta.get("num_beats") or 4

    theory_ctx = build_theory_context()
    vocabs = _load_teacher_vocabs()
    runtime_maps = _build_runtime_id_maps()
    chord_set_spec = _load_teacher_chord_sets_spec()

    song_cat = torch.tensor(
        [[
            int(encode_with_value_map(runtime_maps["tonic_pc_id_map"], meta.get("tonic_pc"), unknown_id=0)),
            int(encode_vocab(vocabs["key_scale"], meta.get("mode_name"))),
            int(encode_with_value_map(runtime_maps["num_beats_id_map"], meta.get("num_beats"), unknown_id=0)),
            int(encode_with_value_map(runtime_maps["beat_unit_id_map"], meta.get("beat_unit"), unknown_id=0)),
        ]],
        dtype=torch.long,
    )
    song_num = torch.tensor([[float(meta.get("bpm") or 0.0), float(meta.get("end_beat") or 0.0)]], dtype=torch.float)
    graph["song"].x_cat = song_cat
    graph["song"].x_num = song_num
    graph["song"].x = torch.cat([song_cat.float(), song_num], dim=1)

    bar_num = torch.tensor(
        [
            [
                float(bar["bar_index"]),
                float(bar["start_beat"]),
                float(bar["end_beat"]),
                float(sum(1 for n in notes if n.get("beat") is not None and int(n["beat"] // num_beats) == bar["bar_index"])),
                float(sum(1 for c in chords if c.get("beat") is not None and int(c["beat"] // num_beats) == bar["bar_index"])),
                float(sum(1 for o in onsets if o.get("bar_index") == bar["bar_index"])),
            ]
            for bar in bars
        ],
        dtype=torch.float,
    ) if bars else torch.empty((0, 6), dtype=torch.float)
    graph["bar"].x_cat = torch.empty((bar_num.size(0), 0), dtype=torch.long)
    graph["bar"].x_num = bar_num
    graph["bar"].x = bar_num

    onset_times = [float(o["onset_time"]) for o in onsets]
    onset_num = torch.tensor(
        [
            [
                float(o.get("beat") or 0.0),
                float(-1 if o.get("bar_index") is None else o.get("bar_index")),
                float(o.get("pos_in_bar") or 0.0),
                float(sum(1 for n in notes if abs(n["onset_time"] - o["onset_time"]) <= ONSET_EPSILON)),
                float(sum(1 for c in chords if abs(c["onset_time"] - o["onset_time"]) <= ONSET_EPSILON)),
            ]
            for o in onsets
        ],
        dtype=torch.float,
    ) if onsets else torch.empty((0, 5), dtype=torch.float)
    graph["onset"].x_cat = torch.empty((onset_num.size(0), 0), dtype=torch.long)
    graph["onset"].x_num = onset_num
    graph["onset"].x = onset_num

    note_cat = torch.tensor(
        [
            [
                int(n.get("sd_id") or 0),
                int(n.get("octave_id") or 0),
            ]
            for n in notes
        ],
        dtype=torch.long,
    ) if notes else torch.empty((0, 2), dtype=torch.long)
    note_num = torch.tensor(
        [
            [
                float(n.get("beat") or 0.0),
                float(n.get("duration_beats") or 0.0),
                float(-1 if n.get("beat") is None else int((n["beat"] // num_beats))),
                float(0.0 if n.get("beat") is None else (n["beat"] % num_beats)),
            ]
            for n in notes
        ],
        dtype=torch.float,
    ) if notes else torch.empty((0, 4), dtype=torch.float)
    graph["note"].x_cat = note_cat
    graph["note"].x_num = note_num
    graph["note"].x = torch.cat([note_cat.float(), note_num], dim=1)

    chord_cat_rows: list[list[int]] = []
    chord_num_rows: list[list[float]] = []
    for chord in chords:
        borrowed = bool(chord.get("borrowed"))
        chord_cat_rows.append(
            [
                int(encode_with_value_map(runtime_maps["root_id_map"], chord.get("root_degree_raw"), unknown_id=0)),
                int(encode_with_value_map(runtime_maps["type_id_map"], chord.get("type_raw"), unknown_id=0)),
                int(encode_with_value_map(runtime_maps["inversion_id_map"], chord.get("inversion_raw"), unknown_id=0)),
                int(encode_vocab(vocabs["borrowed_kind"], "mode_name" if borrowed else "none")),
                int(encode_vocab(vocabs["borrowed_mode_name"], chord.get("mode_name") if borrowed else "<NONE>")),
            ]
        )
        adds_vec = _multi_hot(chord.get("add_degrees"), chord_set_spec["adds"]["allowed_values"])
        omits_vec = _multi_hot(chord.get("omit_degrees"), chord_set_spec["omits"]["allowed_values"])
        suspensions_vec = _multi_hot(chord.get("suspension_degrees"), chord_set_spec["suspensions"]["allowed_values"])
        alterations_vec = _multi_hot(chord.get("alteration_tokens"), chord_set_spec["alterations"]["allowed_values"])
        borrowed_pcset = theory_ctx["mode_to_pcset"].get(chord.get("mode_name"), []) if borrowed else []
        borrowed_pcset_vec = _fixed_range_multi_hot(borrowed_pcset, chord_set_spec["borrowed_pcset"]["size"])
        beat = float(chord.get("beat") or 0.0)
        chord_num_rows.append(
            [
                *adds_vec,
                *omits_vec,
                *suspensions_vec,
                *alterations_vec,
                *borrowed_pcset_vec,
                beat,
                float(chord.get("duration_beats") or 0.0),
                float(-1 if chord.get("beat") is None else int((beat // num_beats))),
                float(0.0 if chord.get("beat") is None else (beat % num_beats)),
            ]
        )
    chord_cat = torch.tensor(chord_cat_rows, dtype=torch.long) if chord_cat_rows else torch.empty((0, 5), dtype=torch.long)
    chord_num = torch.tensor(chord_num_rows, dtype=torch.float) if chord_num_rows else torch.empty(
        (0, sum(CHORD_COMPONENT_SIZES.values()) + 4), dtype=torch.float
    )
    graph["chord"].x_cat = chord_cat
    graph["chord"].x_num = chord_num
    graph["chord"].x = torch.cat([chord_cat.float(), chord_num], dim=1)

    onset_idx = {t: idx for idx, t in enumerate(onset_times)}

    def _edge(pairs: list[tuple[int, int]], edge_type: tuple[str, str, str]):
        graph[edge_type].edge_index = (
            torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
        )

    _edge([(0, i) for i in range(len(bars))], ("song", "contains_bar", "bar"))
    _edge([(i, i + 1) for i in range(max(0, len(bars) - 1))], ("bar", "next_bar", "bar"))
    _edge([(i, j) for i in range(len(bars)) for j, o in enumerate(onsets) if o.get("bar_index") == i], ("bar", "contains_onset", "onset"))
    _edge([(i, i + 1) for i in range(max(0, len(onsets) - 1))], ("onset", "next_onset", "onset"))
    _edge(
        [(onset_idx[o], i) for i, n in enumerate(notes) for o in onset_times if abs(n["onset_time"] - o) <= ONSET_EPSILON],
        ("onset", "starts_note", "note"),
    )
    _edge(
        [(onset_idx[o], i) for i, c in enumerate(chords) for o in onset_times if abs(c["onset_time"] - o) <= ONSET_EPSILON],
        ("onset", "starts_chord", "chord"),
    )
    _edge(
        [(i, i + 1) for i in range(max(0, len(notes) - 1))],
        ("note", "next_note", "note"),
    )
    _edge([(i, i + 1) for i in range(max(0, len(chords) - 1))], ("chord", "next_chord", "chord"))
    _edge(
        [
            (chord_idx, note_idx)
            for chord_idx, chord in enumerate(chords)
            for note_idx, note in enumerate(notes)
            if chord.get("beat") is not None
            and note.get("beat") is not None
            and chord["beat"] <= note["beat"] < (chord["beat"] + float(chord.get("duration_beats") or 0.0))
        ],
        ("chord", "covers_note", "note"),
    )
    for edge_type in OBSERVER_EDGE_TYPES:
        _ = graph[edge_type].edge_index
    for node_type in ("song", "bar", "onset", "note", "chord"):
        graph[node_type].num_nodes = int(graph[node_type].x.size(0))
    return graph
