from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from torch_geometric.data import HeteroData

from src.dataloader.theory_helpers import build_theory_context
from src.observer.chord_parser import predict_observer_chords_for_midi, select_target_instrument

ONSET_EPSILON = 1e-4


_REQUIRED_SAMPLE_FIELDS = ("song_id", "midi_path", "tonic_pc", "mode_name")
_OPTIONAL_SAMPLE_FIELDS = ("bpm", "num_beats", "beat_unit")


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


def _octave_value_to_teacher_octave_id(octave_value: int) -> int | None:
    octave_min, octave_max = _load_octave_bounds()
    if octave_min <= octave_value <= octave_max:
        return octave_value - octave_min + 1
    return None


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


def extract_observer_note_events(pm: Any, tonic_pc: int, bpm: float | None) -> list[dict[str, Any]]:
    theory_ctx = build_theory_context()
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
        octave_value = midi_octave - 5
        octave_id = _octave_value_to_teacher_octave_id(octave_value)

        notes.append(
            {
                "onset_time": onset_time,
                "offset_time": offset_time,
                "beat": beat,
                "duration_beats": duration_beats,
                "pitch": pitch,
                "pitch_class": pitch_class,
                "rel_pc": rel_pc,
                "sd_id": relpc_to_sd_id.get(rel_pc),
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
    mode_to_id = theory_ctx["scale_name_to_id"]
    mode_id = int(mode_to_id.get(meta.get("mode_name"), 0))

    graph["song"].x = __import__("torch").tensor(
        [[
            float(meta.get("tonic_pc", 0)),
            float(mode_id),
            float(meta.get("bpm") or 0.0),
            float(meta.get("num_beats") or 0.0),
            float(meta.get("beat_unit") or 0.0),
            float(meta.get("end_beat") or 0.0),
            float(len(bars)),
            float(len(onsets)),
            float(len(notes)),
            float(len(chords)),
        ]]
    )

    torch = __import__("torch")
    graph["bar"].x = torch.tensor(
        [
            [
                float(bar["bar_index"]),
                float(bar["bar_index"]) / max(1.0, float(len(bars))),
                float(bar["start_beat"]),
                float(bar["end_beat"]),
                float(sum(1 for o in onsets if o.get("bar_index") == bar["bar_index"])),
                float(sum(1 for n in notes if n.get("beat") is not None and int(n["beat"] // num_beats) == bar["bar_index"])),
                float(sum(1 for c in chords if c.get("beat") is not None and int(c["beat"] // num_beats) == bar["bar_index"])),
            ]
            for bar in bars
        ],
        dtype=torch.float,
    ) if bars else torch.empty((0, 7), dtype=torch.float)

    onset_times = [float(o["onset_time"]) for o in onsets]
    graph["onset"].x = torch.tensor(
        [
            [
                float(o.get("beat") or 0.0),
                float(o.get("pos_in_bar") or 0.0),
                float(1.0 if (o.get("pos_in_bar") or 0.0) < ONSET_EPSILON else 0.0),
                float(sum(1 for n in notes if abs(n["onset_time"] - o["onset_time"]) <= ONSET_EPSILON)),
                float(sum(1 for c in chords if abs(c["onset_time"] - o["onset_time"]) <= ONSET_EPSILON)),
                float(1.0 if any(abs(c["onset_time"] - o["onset_time"]) <= ONSET_EPSILON for c in chords) else 0.0),
            ]
            for o in onsets
        ],
        dtype=torch.float,
    ) if onsets else torch.empty((0, 6), dtype=torch.float)

    graph["note"].x = torch.tensor(
        [
            [
                float(n.get("beat") or 0.0),
                float(n.get("duration_beats") or 0.0),
                float(n.get("rel_pc") or 0),
                float(-1 if n.get("sd_id") is None else n.get("sd_id")),
                float(-1 if n.get("octave_id") is None else n.get("octave_id")),
            ]
            for n in notes
        ],
        dtype=torch.float,
    ) if notes else torch.empty((0, 5), dtype=torch.float)

    graph["chord"].x = torch.tensor(
        [
            [
                float(c.get("beat") or 0.0),
                float(c.get("duration_beats") or 0.0),
                float(c.get("root_degree_raw") or 0),
                float(c.get("type_raw") or 0),
                float(-1 if c.get("inversion_raw") is None else c.get("inversion_raw")),
                float(mode_to_id.get(c.get("mode_name"), 0)),
                float(1 if c.get("borrowed") else 0),
                float(len(c.get("add_degrees") or [])),
                float(len(c.get("suspension_degrees") or [])),
                float(len(c.get("omit_degrees") or [])),
                float(len(c.get("alteration_tokens") or [])),
            ]
            for c in chords
        ],
        dtype=torch.float,
    ) if chords else torch.empty((0, 11), dtype=torch.float)

    onset_idx = {t: idx for idx, t in enumerate(onset_times)}

    def _edge(pairs: list[tuple[int, int]], edge_type: tuple[str, str, str]):
        graph[edge_type].edge_index = (
            torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
        )

    _edge([(0, i) for i in range(len(bars))], ("song", "contains_bar", "bar"))
    _edge([(i, j) for i in range(len(bars)) for j, o in enumerate(onsets) if o.get("bar_index") == i], ("bar", "contains_onset", "onset"))
    _edge(
        [(onset_idx[o], i) for i, n in enumerate(notes) for o in onset_times if abs(n["onset_time"] - o) <= ONSET_EPSILON],
        ("onset", "starts_note", "note"),
    )
    _edge(
        [(onset_idx[o], i) for i, c in enumerate(chords) for o in onset_times if abs(c["onset_time"] - o) <= ONSET_EPSILON],
        ("onset", "starts_chord", "chord"),
    )
    _edge(
        [
            (i, j)
            for i, bar in enumerate(bars)
            for j, n in enumerate(notes)
            if n.get("beat") is not None and bar["start_beat"] <= n["beat"] < bar["end_beat"]
        ],
        ("bar", "contains_note", "note"),
    )
    _edge(
        [
            (i, j)
            for i, bar in enumerate(bars)
            for j, c in enumerate(chords)
            if c.get("beat") is not None and bar["start_beat"] <= c["beat"] < bar["end_beat"]
        ],
        ("bar", "contains_chord", "chord"),
    )
    return graph
