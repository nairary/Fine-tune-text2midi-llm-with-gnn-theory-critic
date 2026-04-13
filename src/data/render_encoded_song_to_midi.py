from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
import sys

import pretty_midi

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader.theory_helpers import (  # noqa: E402
    build_theory_context,
    decode_chord_components,
    decode_sd_to_chromatic,
)


@dataclass
class RunStats:
    total: int = 0
    rendered: int = 0
    saved: int = 0
    skipped: int = 0
    skipped_missing_song_id: int = 0
    skipped_bad_song_object: int = 0
    skipped_decode_error: int = 0
    skipped_empty_tracks: int = 0
    skipped_exists: int = 0


def load_encoded_dataset(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Encoded JSON must be a dict like {song_id: song_obj}")
    return data


def load_octave_id_map(repo_root: Path) -> dict[int, int]:
    spec_path = repo_root / "metadata" / "specs" / "spec_global.json"
    with spec_path.open("r", encoding="utf-8") as f:
        spec_global = json.load(f)
    octave_meta = spec_global["octave"]
    octave_values = list(range(int(octave_meta["min"]), int(octave_meta["max"]) + 1))
    return {idx + 1: value for idx, value in enumerate(octave_values)}


def pick_song_bpm(song_obj: dict) -> float:
    meta = song_obj.get("meta", {})
    tempo_regions = meta.get("tempo_regions") or []
    if tempo_regions:
        first_bpm = tempo_regions[0].get("bpm")
        if first_bpm is not None:
            try:
                return float(first_bpm)
            except (TypeError, ValueError):
                pass
    main_bpm = meta.get("main_bpm")
    if main_bpm is not None:
        try:
            return float(main_bpm)
        except (TypeError, ValueError):
            pass
    return 120.0


def beat_duration_to_seconds(beat: float, duration: float, bpm: float) -> tuple[float, float] | None:
    if duration <= 0:
        return None
    seconds_per_beat = 60.0 / bpm
    start_sec = max(0.0, (beat - 1.0) * seconds_per_beat)
    end_sec = max(start_sec + 1e-4, (beat - 1.0 + duration) * seconds_per_beat)
    return start_sec, end_sec


def decode_melody_note_to_midi_pitch(note: dict, song_obj: dict, theory_ctx: dict, octave_id_to_value: dict[int, int]) -> int | None:
    tonic_pc_raw = song_obj.get("meta", {}).get("main_key_tonic_pc")
    try:
        tonic_pc = int(tonic_pc_raw) % 12
    except (TypeError, ValueError):
        tonic_pc = 0

    sd_pc = decode_sd_to_chromatic(int(note.get("sd_id", 0)), theory_ctx)
    if sd_pc is None:
        return None

    octave_id = int(note.get("octave_id", 0) or 0)
    octave_offset = octave_id_to_value.get(octave_id)
    if octave_offset is None:
        return None

    # Center around C5 (MIDI 72), then apply key-relative pitch class and encoded octave offset.
    midi_pitch = 72 + (12 * octave_offset) + tonic_pc + int(sd_pc)
    if 0 <= midi_pitch <= 127:
        return midi_pitch
    return None


def _nearest_pitch_for_pc(pc: int, target: float) -> int:
    base = int(round((target - pc) / 12.0))
    candidates = [pc + 12 * (base + delta) for delta in (-1, 0, 1)]
    return min(candidates, key=lambda pitch: abs(pitch - target))


def voice_chord(body_pcs: list[int], add_pcs: list[int], inversion_raw: int, target_center: float) -> tuple[list[int], int]:
    body_midi: list[int] = []
    previous = None
    for pc in body_pcs:
        pitch = _nearest_pitch_for_pc(pc, target_center)
        if previous is not None:
            while pitch <= previous:
                pitch += 12
        body_midi.append(pitch)
        previous = pitch

    current_center = statistics.mean(body_midi)
    shift = int(round((target_center - current_center) / 12.0)) * 12
    body_midi = [pitch + shift for pitch in body_midi]

    inversion = max(0, min(int(inversion_raw), max(0, len(body_midi) - 1)))
    for idx in range(inversion):
        body_midi[idx] += 12
    body_midi = sorted(body_midi)

    max_body = max(body_midi)
    add_midi: list[int] = []
    for pc in add_pcs:
        pitch = _nearest_pitch_for_pc(pc, max_body + 12)
        while pitch <= max_body:
            pitch += 12
        add_midi.append(pitch)

    bass_pc = body_midi[0] % 12
    bass_pitch = _nearest_pitch_for_pc(bass_pc, min(body_midi) - 14)
    while bass_pitch >= min(body_midi) - 5:
        bass_pitch -= 12

    return sorted(body_midi + add_midi), bass_pitch


def render_song_to_pretty_midi(
    song_obj: dict,
    theory_ctx: dict,
    octave_id_to_value: dict[int, int],
    melody_velocity: int = 86,
    chord_velocity: int = 68,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, int]]:
    bpm = pick_song_bpm(song_obj)
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    tonic_pc_raw = song_obj.get("meta", {}).get("main_key_tonic_pc")
    try:
        tonic_pc = int(tonic_pc_raw) % 12
    except (TypeError, ValueError):
        tonic_pc = 0

    meta = song_obj.get("meta", {})
    meter_regions = meta.get("meter_regions") or []
    numerator = None
    if meter_regions:
        numerator = meter_regions[0].get("num_beats")
    if numerator is None:
        numerator = meta.get("main_num_beats")
    try:
        numerator_int = int(numerator)
        if numerator_int > 0:
            pm.time_signature_changes.append(pretty_midi.TimeSignature(numerator=numerator_int, denominator=4, time=0.0))
    except (TypeError, ValueError):
        pass

    melody_instr = pretty_midi.Instrument(program=0, is_drum=False, name="melody")
    chords_instr = pretty_midi.Instrument(program=0, is_drum=False, name="chords")

    skipped_melody_decode = 0
    melody_pitches: list[int] = []
    for note in song_obj.get("melody", []):
        if int(note.get("is_rest", 0) or 0) == 1:
            continue
        try:
            beat = float(note.get("beat", 0.0) or 0.0)
            duration = float(note.get("duration", 0.0) or 0.0)
        except (TypeError, ValueError):
            skipped_melody_decode += 1
            continue
        timing = beat_duration_to_seconds(beat, duration, bpm)
        if timing is None:
            continue

        pitch = decode_melody_note_to_midi_pitch(note, song_obj, theory_ctx, octave_id_to_value)
        if pitch is None:
            skipped_melody_decode += 1
            continue

        start_sec, end_sec = timing
        melody_instr.notes.append(pretty_midi.Note(velocity=melody_velocity, pitch=pitch, start=start_sec, end=end_sec))
        melody_pitches.append(pitch)

    target_center = (statistics.median(melody_pitches) - 8.0) if melody_pitches else 52.0

    skipped_chord_decode = 0
    for chord in song_obj.get("chords", []):
        if int(chord.get("is_rest", 0) or 0) == 1:
            continue
        try:
            beat = float(chord.get("beat", 0.0) or 0.0)
            duration = float(chord.get("duration", 0.0) or 0.0)
        except (TypeError, ValueError):
            skipped_chord_decode += 1
            continue
        timing = beat_duration_to_seconds(beat, duration, bpm)
        if timing is None:
            continue

        components = decode_chord_components(song_obj, chord, theory_ctx)
        if components is None:
            skipped_chord_decode += 1
            continue

        body_pcs_abs = [((int(pc) + tonic_pc) % 12) for pc in components["body_pcs"]]
        add_pcs_abs = [((int(pc) + tonic_pc) % 12) for pc in components["add_pcs"]]
        inversion_raw = theory_ctx["inversion_id_to_raw"].get(int(chord.get("inversion_id", 0)), 0)
        voiced, bass_pitch = voice_chord(
            body_pcs_abs,
            add_pcs_abs,
            inversion_raw=inversion_raw,
            target_center=target_center,
        )

        start_sec, end_sec = timing
        for pitch in [bass_pitch] + voiced:
            if 0 <= pitch <= 127:
                chords_instr.notes.append(pretty_midi.Note(velocity=chord_velocity, pitch=int(pitch), start=start_sec, end=end_sec))

    pm.instruments.extend([melody_instr, chords_instr])
    return pm, {
        "skipped_melody_decode": skipped_melody_decode,
        "skipped_chord_decode": skipped_chord_decode,
    }


def song_output_path(output_root: Path, song_id: str, split: str | None) -> Path:
    split_name = split if split in {"train", "val", "test"} else "unknown"
    return output_root / split_name / f"{song_id}.mid"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render new MIDI from teacher_encoded JSON without using source MIDI")
    parser.add_argument("--encoded-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--song-id", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    dataset = load_encoded_dataset(args.encoded_json)
    theory_ctx = build_theory_context()
    octave_id_to_value = load_octave_id_map(REPO_ROOT)
    stats = RunStats(total=len(dataset))

    items = dataset.items()
    if args.song_id is not None:
        if args.song_id not in dataset:
            logging.error("song_id=%s not found in encoded dataset", args.song_id)
            return
        items = [(args.song_id, dataset[args.song_id])]

    processed = 0
    for song_id, song_obj in items:
        if args.limit is not None and processed >= args.limit:
            break

        if not isinstance(song_obj, dict):
            stats.skipped += 1
            stats.skipped_bad_song_object += 1
            logging.warning("Skip %s: bad song object type=%s", song_id, type(song_obj).__name__)
            continue

        split = song_obj.get("meta", {}).get("split")
        if args.split is not None and split != args.split:
            continue

        processed += 1
        output_path = song_output_path(args.output_root, song_id, split)
        if output_path.exists() and not args.overwrite:
            stats.skipped += 1
            stats.skipped_exists += 1
            logging.info("Skip %s: output exists (%s)", song_id, output_path)
            continue

        try:
            pm, decode_stats = render_song_to_pretty_midi(song_obj, theory_ctx, octave_id_to_value)
            stats.rendered += 1

            melody_notes = len(pm.instruments[0].notes) if len(pm.instruments) > 0 else 0
            chord_notes = len(pm.instruments[1].notes) if len(pm.instruments) > 1 else 0
            if melody_notes == 0 or chord_notes == 0:
                stats.skipped += 1
                stats.skipped_empty_tracks += 1
                logging.warning(
                    "Skip %s: empty rendered content melody_notes=%d chord_notes=%d",
                    song_id,
                    melody_notes,
                    chord_notes,
                )
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            pm.write(str(output_path))
            stats.saved += 1
            logging.info(
                "Saved %s (melody_notes=%d chord_notes=%d melody_decode_skips=%d chord_decode_skips=%d)",
                output_path,
                melody_notes,
                chord_notes,
                decode_stats["skipped_melody_decode"],
                decode_stats["skipped_chord_decode"],
            )
        except Exception as exc:  # noqa: BLE001
            stats.skipped += 1
            stats.skipped_decode_error += 1
            logging.exception("Skip %s due to render/decode error: %s", song_id, exc)

    logging.info(
        "Summary | total=%d processed=%d rendered=%d saved=%d skipped=%d | bad_song=%d "
        "decode_error=%d empty_tracks=%d exists=%d",
        stats.total,
        processed,
        stats.rendered,
        stats.saved,
        stats.skipped,
        stats.skipped_bad_song_object,
        stats.skipped_decode_error,
        stats.skipped_empty_tracks,
        stats.skipped_exists,
    )


if __name__ == "__main__":
    main()
