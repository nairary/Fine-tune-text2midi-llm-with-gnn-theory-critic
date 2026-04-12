from __future__ import annotations

import statistics
import tempfile
import unittest
from pathlib import Path

import pretty_midi

from src.data.render_encoded_song_to_midi import load_encoded_dataset, main as render_main
from src.dataloader.theory_helpers import build_theory_context


class RenderEncodedSongToMidiSmokeTest(unittest.TestCase):
    def test_rendered_midi_has_melody_and_chords_with_expected_timing(self):
        ctx = build_theory_context()
        song_id = "song_render_1"
        bpm = 120.0

        # melody first note -> pitch 72 (C5), second note is rest and must be skipped.
        dataset = {
            song_id: {
                "meta": {
                    "split": "train",
                    "main_key_tonic_pc": 0,
                    "main_key_scale_id": ctx["scale_name_to_id"]["major"],
                    "main_bpm": bpm,
                    "main_num_beats": 4,
                },
                "melody": [
                    {"beat": 1.0, "duration": 1.0, "sd_id": ctx["sd_token_to_id"]["1"], "octave_id": 5, "is_rest": 0},
                    {"beat": 2.0, "duration": 1.0, "sd_id": ctx["sd_token_to_id"]["2"], "octave_id": 5, "is_rest": 1},
                    {"beat": 3.0, "duration": 1.0, "sd_id": ctx["sd_token_to_id"]["3"], "octave_id": 5, "is_rest": 0},
                ],
                "chords": [
                    {
                        "beat": 1.0,
                        "duration": 2.0,
                        "root_id": 1,
                        "type_id": 1,
                        "inversion_id": 1,
                        "applied_id": 1,
                        "borrowed_kind_id": ctx["borrowed_kind_to_id"]["none"],
                        "borrowed_mode_name_id": next(
                            idx for idx, name in ctx["borrowed_mode_id_to_name"].items() if "none" in str(name).lower()
                        ),
                        "adds_vec": [0, 0, 0, 1, 0, 0],
                        "omits_vec": [0, 0],
                        "suspensions_vec": [0, 0],
                        "alterations_vec": [0, 0, 0, 0, 0, 0],
                        "borrowed_pcset_vec": [0] * 12,
                        "is_rest": 0,
                    }
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            encoded_path = tmp / "teacher_encoded.json"
            output_root = tmp / "rendered"
            encoded_path.write_text(__import__("json").dumps(dataset), encoding="utf-8")

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "render_encoded_song_to_midi.py",
                    "--encoded-json",
                    str(encoded_path),
                    "--output-root",
                    str(output_root),
                    "--song-id",
                    song_id,
                ]
                render_main()
            finally:
                sys.argv = old_argv

            rendered_path = output_root / "train" / f"{song_id}.mid"
            self.assertTrue(rendered_path.exists())
            _ = load_encoded_dataset(encoded_path)

            pm = pretty_midi.PrettyMIDI(str(rendered_path))
            self.assertGreaterEqual(len(pm.instruments), 2)

            melody = next(instr for instr in pm.instruments if instr.name == "melody")
            chords = next(instr for instr in pm.instruments if instr.name == "chords")

            self.assertGreater(len(melody.notes), 0)
            self.assertGreater(len(chords.notes), 0)

            for instr in (melody, chords):
                for note in instr.notes:
                    self.assertLess(note.start, note.end)

            melody_starts = sorted(note.start for note in melody.notes)
            melody_ends = sorted(note.end for note in melody.notes)
            self.assertAlmostEqual(melody_starts[0], 0.0, places=4)
            self.assertAlmostEqual(melody_ends[0], 0.5, places=4)

            # rest note should not be rendered, so only two melody notes stay.
            self.assertEqual(len(melody.notes), 2)

            chord_starts = sorted(note.start for note in chords.notes)
            self.assertAlmostEqual(chord_starts[0], 0.0, places=4)

            # inversion + add tone should make upper notes include at least one tone above body median.
            chord_pitches = sorted(note.pitch for note in chords.notes)
            self.assertGreater(max(chord_pitches), statistics.median(chord_pitches))


if __name__ == "__main__":
    unittest.main()
