from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, merge_tracks

from src.dataloader.theory_helpers import (
    build_theory_context,
    decode_chord_components,
)

SPLIT_TO_MIDI_DIR = {
    "train": "HookTheory_Train_MIDI",
    "val": "HookTheory_Val_MIDI",
    "test": "HookTheory_Test_MIDI",
}

@dataclass
class RunStats:
    total: int = 0
    midi_found: int = 0
    saved: int = 0
    skipped: int = 0
    skipped_no_split: int = 0
    skipped_bad_split: int = 0
    skipped_no_midi: int = 0
    skipped_ambiguous_midi: int = 0
    skipped_decode_error: int = 0
    skipped_exists: int = 0


def load_encoded_dataset(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Encoded JSON должен быть объектом вида {song_id: song_obj}")
    return data


def resolve_split_midi_dir(split: str, midi_root: Path) -> Path | None:
    split_name = SPLIT_TO_MIDI_DIR.get(split.lower())
    if split_name is None:
        return None
    return midi_root / split_name


def find_midi_for_song_id(song_id: str, midi_dir: Path) -> Path:
    candidates = []
    for suffix in (".mid", ".midi"):
        candidate = midi_dir / f"{song_id}{suffix}"
        if candidate.exists():
            candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f"Ambiguous MIDI match for {song_id}: {candidates}")
    raise FileNotFoundError(f"MIDI for song_id={song_id} not found in {midi_dir}")


def _tempo_events(mid: MidiFile) -> list[tuple[int, int]]:
    merged = merge_tracks(mid.tracks)
    abs_tick = 0
    events = [(0, bpm2tempo(120))]
    for msg in merged:
        abs_tick += msg.time
        if msg.type == "set_tempo":
            events.append((abs_tick, msg.tempo))
    events.sort(key=lambda x: x[0])
    compact = []
    for tick, tempo in events:
        if compact and compact[-1][0] == tick:
            compact[-1] = (tick, tempo)
        else:
            compact.append((tick, tempo))
    return compact


def beats_to_seconds(beat: float, ppq: int, tempo_events: list[tuple[int, int]]) -> float:
    tick_target = int(round(max(0.0, beat - 1.0) * ppq))
    total = 0.0
    for i, (start_tick, tempo) in enumerate(tempo_events):
        end_tick = tempo_events[i + 1][0] if i + 1 < len(tempo_events) else tick_target
        if tick_target <= start_tick:
            break
        seg_end = min(end_tick, tick_target)
        if seg_end > start_tick:
            total += ((seg_end - start_tick) * tempo) / (1_000_000.0 * ppq)
        if seg_end == tick_target:
            break
    return total


def estimate_melody_register(mid: MidiFile) -> float:
    pitches: list[int] = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0 and getattr(msg, "channel", 0) != 9:
                pitches.append(int(msg.note))
    return float(statistics.median(pitches)) if pitches else 60.0


def _nearest_pitch_for_pc(pc: int, target: float) -> int:
    base = int(round((target - pc) / 12.0))
    candidates = [pc + 12 * (base + delta) for delta in (-1, 0, 1)]
    return min(candidates, key=lambda p: abs(p - target))


def voice_chord(body_pcs: list[int], add_pcs: list[int], inversion_raw: int, target_center: float) -> tuple[list[int], int]:
    body_midi = []
    prev = None
    for pc in body_pcs:
        pitch = _nearest_pitch_for_pc(pc, target_center)
        if prev is not None:
            while pitch <= prev:
                pitch += 12
        body_midi.append(pitch)
        prev = pitch

    current_center = statistics.mean(body_midi)
    shift = int(round((target_center - current_center) / 12.0)) * 12
    body_midi = [p + shift for p in body_midi]

    inversion = max(0, int(inversion_raw))
    inversion = min(inversion, max(0, len(body_midi) - 1))
    for i in range(inversion):
        body_midi[i] += 12
    body_midi = sorted(body_midi)

    max_body = max(body_midi)
    add_midi = []
    for pc in add_pcs:
        pitch = _nearest_pitch_for_pc(pc, max_body + 12)
        while pitch <= max_body:
            pitch += 12
        add_midi.append(pitch)

    bass_pc = body_midi[0] % 12
    bass_pitch = _nearest_pitch_for_pc(bass_pc, min(body_midi) - 14)
    while bass_pitch >= min(body_midi) - 5:
        bass_pitch -= 12

    full = sorted(body_midi + add_midi)
    return full, bass_pitch


def render_chord_track(mid: MidiFile, song_obj: dict, theory_ctx: dict, velocity: int = 68) -> MidiTrack:
    target_center = estimate_melody_register(mid) - 8.0
    _ = _tempo_events(mid)  # v1: preserve tempo awareness, events are placed on beat grid in ticks.

    events: list[tuple[int, Message]] = []
    track = MidiTrack()
    track.append(MetaMessage("track_name", name="chords", time=0))
    track.append(Message("program_change", program=0, channel=1, time=0))

    for chord in song_obj.get("chords", []):
        if int(chord.get("is_rest", 0)) == 1:
            continue
        beat = float(chord.get("beat", 0.0) or 0.0)
        duration = float(chord.get("duration", 0.0) or 0.0)
        if duration <= 0:
            continue

        chord_components = decode_chord_components(song_obj, chord, theory_ctx)
        if chord_components is None:
            continue
        body_pcs = chord_components["body_pcs"]
        add_pcs = chord_components["add_pcs"]
        inversion_raw = theory_ctx["inversion_id_to_raw"].get(int(chord.get("inversion_id", 0)), 0)
        voiced, bass_pitch = voice_chord(body_pcs, add_pcs, inversion_raw=inversion_raw, target_center=target_center)

        start_tick = int(round(max(0.0, beat - 1.0) * mid.ticks_per_beat))
        end_tick = max(start_tick + 1, int(round(max(0.0, beat + duration - 1.0) * mid.ticks_per_beat)))

        all_notes = [bass_pitch] + voiced
        for note in all_notes:
            if note < 0 or note > 127:
                continue
            events.append((start_tick, Message("note_on", note=int(note), velocity=velocity, channel=1, time=0)))
            events.append((end_tick, Message("note_off", note=int(note), velocity=0, channel=1, time=0)))

    events.sort(key=lambda x: (x[0], 0 if x[1].type == "note_off" else 1))
    last_tick = 0
    for tick, msg in events:
        msg.time = tick - last_tick
        track.append(msg)
        last_tick = tick
    return track


def enrich_midi_file(song_id: str, song_obj: dict, midi_path: Path, output_path: Path, theory_ctx: dict) -> None:
    mid = MidiFile(midi_path)
    chord_track = render_chord_track(mid, song_obj, theory_ctx)
    mid.tracks.append(chord_track)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich MIDI files with chord track from encoded_full JSON")
    parser.add_argument("--encoded-json", type=Path, required=True)
    parser.add_argument("--midi-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    dataset = load_encoded_dataset(args.encoded_json)
    theory_ctx = build_theory_context()
    stats = RunStats(total=len(dataset))
    logging.info("Loaded %d song objects from %s", len(dataset), args.encoded_json)

    processed = 0
    for song_id, song_obj in dataset.items():
        if args.limit is not None and processed >= args.limit:
            break
        processed += 1

        split = song_obj.get("meta", {}).get("split")
        if args.split and split != args.split:
            continue
        if split is None:
            stats.skipped += 1
            stats.skipped_no_split += 1
            logging.warning("Skip %s: missing meta.split", song_id)
            continue

        midi_dir = resolve_split_midi_dir(split, args.midi_root)
        if midi_dir is None:
            stats.skipped += 1
            stats.skipped_bad_split += 1
            logging.warning("Skip %s: unknown split '%s'", song_id, split)
            continue

        try:
            midi_path = find_midi_for_song_id(song_id, midi_dir)
            stats.midi_found += 1
        except FileNotFoundError as exc:
            stats.skipped += 1
            stats.skipped_no_midi += 1
            logging.warning("Skip %s: %s", song_id, exc)
            continue
        except RuntimeError as exc:
            stats.skipped += 1
            stats.skipped_ambiguous_midi += 1
            logging.warning("Skip %s: %s", song_id, exc)
            continue

        output_path = args.output_root / midi_dir.name / midi_path.name
        if output_path.exists() and not args.overwrite:
            stats.skipped += 1
            stats.skipped_exists += 1
            logging.info("Skip %s: output exists (%s)", song_id, output_path)
            continue

        try:
            enrich_midi_file(song_id, song_obj, midi_path, output_path, theory_ctx)
            stats.saved += 1
            logging.info("Saved enriched MIDI: %s", output_path)
        except Exception as exc:  # noqa: BLE001
            stats.skipped += 1
            stats.skipped_decode_error += 1
            logging.exception("Skip %s due to chord decoding/render error: %s", song_id, exc)

    logging.info(
        "Summary | total=%d processed=%d midi_found=%d saved=%d skipped=%d | "
        "no_split=%d bad_split=%d no_midi=%d ambiguous=%d decode_error=%d exists=%d",
        stats.total,
        processed,
        stats.midi_found,
        stats.saved,
        stats.skipped,
        stats.skipped_no_split,
        stats.skipped_bad_split,
        stats.skipped_no_midi,
        stats.skipped_ambiguous_midi,
        stats.skipped_decode_error,
        stats.skipped_exists,
    )


if __name__ == "__main__":
    main()
