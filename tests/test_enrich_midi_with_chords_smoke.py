from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from src.data.enrich_midi_with_chords import build_theory_context, enrich_midi_file


class EnrichMidiWithChordsSmokeTest(unittest.TestCase):
    def test_enrich_adds_chord_track_with_notes(self):
        theory_ctx = build_theory_context()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src_midi = tmp / "input.mid"
            out_midi = tmp / "out.mid"

            pm = pretty_midi.PrettyMIDI(initial_tempo=120)
            melody = pretty_midi.Instrument(program=0, name="melody", is_drum=False)
            melody.notes.append(pretty_midi.Note(velocity=80, pitch=72, start=0.0, end=0.5))
            melody.notes.append(pretty_midi.Note(velocity=80, pitch=76, start=0.5, end=1.0))
            pm.instruments.append(melody)
            pm.write(str(src_midi))

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
            enriched = pretty_midi.PrettyMIDI(str(out_midi))
            self.assertEqual(len(enriched.instruments), len(pm.instruments) + 1)
            chord_instr = enriched.instruments[-1]
            self.assertGreater(len(chord_instr.notes), 0)
            self.assertTrue(all(note.start < note.end for note in chord_instr.notes))


if __name__ == "__main__":
    unittest.main()
