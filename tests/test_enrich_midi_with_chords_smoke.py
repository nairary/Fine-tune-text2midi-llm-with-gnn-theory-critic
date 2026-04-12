from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mido import Message, MidiFile, MidiTrack

from src.data.enrich_midi_with_chords import build_theory_context, enrich_midi_file


class EnrichMidiWithChordsSmokeTest(unittest.TestCase):
    def test_enrich_adds_chord_track_with_notes(self):
        theory_ctx = build_theory_context()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src_midi = tmp / "input.mid"
            out_midi = tmp / "out.mid"

            mid = MidiFile(ticks_per_beat=480)
            melody = MidiTrack()
            melody.append(Message("program_change", program=0, channel=0, time=0))
            melody.append(Message("note_on", note=72, velocity=80, channel=0, time=0))
            melody.append(Message("note_off", note=72, velocity=0, channel=0, time=480))
            melody.append(Message("note_on", note=76, velocity=80, channel=0, time=0))
            melody.append(Message("note_off", note=76, velocity=0, channel=0, time=480))
            mid.tracks.append(melody)
            mid.save(src_midi)

            song_obj = {
                "meta": {
                    "split": "train",
                    "main_key_scale_id": theory_ctx["scale_name_to_id"]["major"],
                },
                "chords": [
                    {
                        "beat": 1.0,
                        "duration": 2.0,
                        "root_id": 1,
                        "type_id": 1,
                        "inversion_id": 1,
                        "applied_id": 1,
                        "borrowed_kind_id": theory_ctx["borrowed_kind_to_id"]["none"],
                        "borrowed_mode_name_id": next(
                            k for k, v in theory_ctx["borrowed_mode_id_to_name"].items() if "none" in str(v).lower()
                        ),
                        "adds_vec": [0, 0, 0, 1, 0, 0],
                        "omits_vec": [0, 0],
                        "suspensions_vec": [1, 0],
                        "alterations_vec": [0, 1, 0, 0, 0, 0],
                        "borrowed_pcset_vec": [0] * 12,
                        "is_rest": 0,
                    }
                ],
            }

            enrich_midi_file("song_a", song_obj, src_midi, out_midi, theory_ctx)

            self.assertTrue(out_midi.exists())
            enriched = MidiFile(out_midi)
            self.assertGreaterEqual(len(enriched.tracks), 2)
            chord_track = enriched.tracks[-1]
            note_on_count = sum(1 for msg in chord_track if msg.type == "note_on" and msg.velocity > 0)
            self.assertGreater(note_on_count, 0)

    def test_enrich_aligns_chords_to_existing_melody_timeline(self):
        theory_ctx = build_theory_context()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src_midi = tmp / "input_shifted.mid"
            out_midi = tmp / "out_shifted.mid"

            mid = MidiFile(ticks_per_beat=480)
            melody = MidiTrack()
            melody.append(Message("program_change", program=0, channel=0, time=0))
            melody.append(Message("note_on", note=72, velocity=80, channel=0, time=480))
            melody.append(Message("note_off", note=72, velocity=0, channel=0, time=480))
            mid.tracks.append(melody)
            mid.save(src_midi)

            song_obj = {
                "meta": {
                    "split": "train",
                    "main_key_scale_id": theory_ctx["scale_name_to_id"]["major"],
                },
                "melody": [
                    {"beat": 1.0, "duration": 1.0, "sd_id": 1, "octave_id": 5, "is_rest": 0},
                ],
                "chords": [
                    {
                        "beat": 1.0,
                        "duration": 1.0,
                        "root_id": 1,
                        "type_id": 1,
                        "inversion_id": 1,
                        "applied_id": 1,
                        "borrowed_kind_id": theory_ctx["borrowed_kind_to_id"]["none"],
                        "borrowed_mode_name_id": next(
                            k for k, v in theory_ctx["borrowed_mode_id_to_name"].items() if "none" in str(v).lower()
                        ),
                        "adds_vec": [0, 0, 0, 0, 0, 0],
                        "omits_vec": [0, 0],
                        "suspensions_vec": [0, 0],
                        "alterations_vec": [0, 0, 0, 0, 0, 0],
                        "borrowed_pcset_vec": [0] * 12,
                        "is_rest": 0,
                    }
                ],
            }

            enrich_midi_file("song_b", song_obj, src_midi, out_midi, theory_ctx)

            enriched = MidiFile(out_midi)
            chord_track = enriched.tracks[-1]
            abs_tick = 0
            first_chord_note_tick = None
            for msg in chord_track:
                abs_tick += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    first_chord_note_tick = abs_tick
                    break
            self.assertEqual(first_chord_note_tick, 480)


if __name__ == "__main__":
    unittest.main()
