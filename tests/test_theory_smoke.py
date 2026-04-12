from __future__ import annotations

import copy
import random
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader.song_corruptions import corrupt_song_obj
from src.dataloader.theory_helpers import (
    build_theory_context,
    chord_bass_and_top_pcs,
    decode_chord_components,
    chord_pitch_classes_tertian,
    decode_sd_to_chromatic,
    find_covering_chord_index,
    is_strong_note_position,
)


class TheorySmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = build_theory_context()

    def _song(self):
        return {
            "song_id": "smoke",
            "meta": {"main_key_scale_id": self.ctx["scale_name_to_id"]["major"], "main_num_beats": 4},
            "melody": [
                {"beat": 1.0, "duration": 1.0, "sd_id": self.ctx["sd_token_to_id"]["1"], "octave_id": 5, "is_rest": 0},
                {"beat": 2.0, "duration": 1.0, "sd_id": self.ctx["sd_token_to_id"]["2"], "octave_id": 5, "is_rest": 0},
            ],
            "chords": [
                {
                    "beat": 1.0,
                    "duration": 2.0,
                    "root_id": 1,
                    "type_id": 1,
                    "inversion_id": 1,
                    "applied_id": 1,
                    "borrowed_kind_id": self.ctx["borrowed_kind_to_id"]["none"],
                    "borrowed_mode_name_id": next(k for k, v in self.ctx["borrowed_mode_id_to_name"].items() if "none" in str(v).lower()),
                    "adds_vec": [0, 0, 0, 0, 0, 0],
                    "borrowed_pcset_vec": [0] * 12,
                    "is_rest": 0,
                },
                {
                    "beat": 3.0,
                    "duration": 1.0,
                    "root_id": 4,
                    "type_id": 1,
                    "inversion_id": 1,
                    "applied_id": 1,
                    "borrowed_kind_id": self.ctx["borrowed_kind_to_id"]["none"],
                    "borrowed_mode_name_id": next(k for k, v in self.ctx["borrowed_mode_id_to_name"].items() if "none" in str(v).lower()),
                    "adds_vec": [0, 0, 0, 0, 0, 0],
                    "borrowed_pcset_vec": [0] * 12,
                    "is_rest": 0,
                },
            ],
        }

    @staticmethod
    def _expected_post_onset_indices(corrupted_song: dict, source_beat: float, target_beat: float) -> list[int]:
        post_grid = sorted({float(x["beat"]) for x in corrupted_song["melody"] + corrupted_song["chords"]})
        indices = []
        if source_beat in post_grid:
            indices.append(post_grid.index(source_beat))
        if target_beat in post_grid:
            indices.append(post_grid.index(target_beat))
        return sorted(set(indices))

    def test_chord_pitch_classes_tertian(self):
        song = self._song()
        c_major = copy.deepcopy(song["chords"][0])
        c_major["root_id"] = 1
        c_major["type_id"] = 1  # raw type 5
        self.assertEqual(chord_pitch_classes_tertian(song, c_major, self.ctx), {0, 4, 7})

        c7 = copy.deepcopy(song["chords"][0])
        c7["type_id"] = 2  # raw type 7
        self.assertEqual(chord_pitch_classes_tertian(song, c7, self.ctx), {0, 4, 7, 11})

        borrowed = copy.deepcopy(song["chords"][0])
        borrowed["borrowed_kind_id"] = self.ctx["borrowed_kind_to_id"]["mode_name"]
        borrowed["borrowed_mode_name_id"] = self.ctx["borrowed_mode_to_id"]["dorian"]
        borrowed["root_id"] = 2  # II in current encoding -> degree 2 in active mode
        self.assertEqual(chord_pitch_classes_tertian(song, borrowed, self.ctx), {2, 5, 9})

    def test_chord_pitch_classes_tertian_bvii_supported(self):
        song = self._song()
        bvii = copy.deepcopy(song["chords"][0])
        bvii["root_id"] = 8  # raw_root == 7 (bVII)
        bvii["type_id"] = 1  # raw type 5
        # In major template, VII=11 => bVII root is 10; tertian stack anchored on VII gives {10, 2, 5}
        self.assertEqual(chord_pitch_classes_tertian(song, bvii, self.ctx), {10, 2, 5})

        song_minor = copy.deepcopy(song)
        song_minor["meta"]["main_key_scale_id"] = self.ctx["scale_name_to_id"]["minor"]
        # bVII in minor should keep b7 root class 10 (not collapse to VI).
        self.assertEqual(chord_pitch_classes_tertian(song_minor, bvii, self.ctx), {10, 2, 5})

    def test_decode_chord_components_base_body(self):
        song = self._song()
        chord = copy.deepcopy(song["chords"][0])
        chord.update({
            "root_id": 1,  # I
            "type_id": 1,  # triad
            "adds_vec": [0, 0, 0, 0, 0, 0],
            "suspensions_vec": [0, 0],
            "omits_vec": [0, 0],
            "alterations_vec": [0, 0, 0, 0, 0, 0],
        })
        decoded = decode_chord_components(song, chord, self.ctx)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["body_pcs"], [0, 4, 7])

    def test_decode_chord_components_adds_vec(self):
        song = self._song()
        chord = copy.deepcopy(song["chords"][0])
        chord.update({
            "adds_vec": [0, 0, 0, 0, 1, 0],  # add11
            "suspensions_vec": [0, 0],
            "omits_vec": [0, 0],
            "alterations_vec": [0, 0, 0, 0, 0, 0],
        })
        decoded = decode_chord_components(song, chord, self.ctx)
        self.assertIsNotNone(decoded)
        self.assertIn(5, decoded["add_pcs"])

    def test_decode_chord_components_suspensions_vec(self):
        song = self._song()
        chord = copy.deepcopy(song["chords"][0])
        chord.update({
            "adds_vec": [0, 0, 0, 0, 0, 0],
            "suspensions_vec": [1, 0],  # sus2
            "omits_vec": [0, 0],
            "alterations_vec": [0, 0, 0, 0, 0, 0],
        })
        decoded = decode_chord_components(song, chord, self.ctx)
        self.assertIsNotNone(decoded)
        self.assertIn(2, decoded["body_pcs"])
        self.assertNotIn(4, decoded["body_pcs"])

    def test_decode_chord_components_omits_vec(self):
        song = self._song()
        chord = copy.deepcopy(song["chords"][0])
        chord.update({
            "adds_vec": [0, 0, 0, 0, 0, 0],
            "suspensions_vec": [0, 0],
            "omits_vec": [0, 1],  # omit5
            "alterations_vec": [0, 0, 0, 0, 0, 0],
        })
        decoded = decode_chord_components(song, chord, self.ctx)
        self.assertIsNotNone(decoded)
        self.assertNotIn(7, decoded["body_pcs"])

    def test_decode_chord_components_alterations_vec(self):
        song = self._song()
        chord = copy.deepcopy(song["chords"][0])
        chord.update({
            "adds_vec": [0, 0, 0, 0, 0, 0],
            "suspensions_vec": [0, 0],
            "omits_vec": [0, 0],
            "alterations_vec": [0, 1, 0, 0, 0, 0],  # #5
        })
        decoded = decode_chord_components(song, chord, self.ctx)
        self.assertIsNotNone(decoded)
        self.assertIn(8, decoded["body_pcs"])

    def test_decode_chord_components_borrowed_mode_changes_body(self):
        song = self._song()
        chord = copy.deepcopy(song["chords"][0])
        chord.update({
            "root_id": 2,  # II degree in active mode
            "type_id": 1,  # triad
            "borrowed_kind_id": self.ctx["borrowed_kind_to_id"]["mode_name"],
            "borrowed_mode_name_id": self.ctx["borrowed_mode_to_id"]["dorian"],
            "adds_vec": [0, 0, 0, 0, 0, 0],
            "suspensions_vec": [0, 0],
            "omits_vec": [0, 0],
            "alterations_vec": [0, 0, 0, 0, 0, 0],
        })
        decoded = decode_chord_components(song, chord, self.ctx)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["active_mode_name"], "dorian")
        self.assertEqual(decoded["body_pcs"], [2, 5, 9])

    def test_reproducibility_with_seed(self):
        song = self._song()
        cfg = {"rhythm_shift_max_steps": 1}
        modes = ["note_onset_shift"]

        s1, m1 = corrupt_song_obj(copy.deepcopy(song), modes, cfg, self.ctx, rng=random.Random(42))
        s2, m2 = corrupt_song_obj(copy.deepcopy(song), modes, cfg, self.ctx, rng=random.Random(42))

        self.assertEqual(m1, m2)
        self.assertEqual(s1["melody"], s2["melody"])

    def test_is_strong_note_position_across_bars(self):
        song = self._song()
        self.assertTrue(is_strong_note_position({"beat": 1.0}, song))
        self.assertTrue(is_strong_note_position({"beat": 5.0}, song))
        self.assertTrue(is_strong_note_position({"beat": 7.0}, song))
        self.assertFalse(is_strong_note_position({"beat": 6.0}, song))

    def test_helpers_handle_missing_or_invalid_timing_fields(self):
        song = self._song()
        self.assertIsNone(find_covering_chord_index(song, {"beat": None}))
        self.assertFalse(is_strong_note_position({"beat": None}, song))
        self.assertFalse(is_strong_note_position({"beat": 2.0, "pos_in_bar": None}, song))
        self.assertFalse(is_strong_note_position({"beat": 2.0, "pos_in_bar": "bad"}, song))

        song_bad_meta = copy.deepcopy(song)
        song_bad_meta["meta"]["main_num_beats"] = None
        self.assertTrue(is_strong_note_position({"beat": 1.0}, song_bad_meta))

    def test_corrupt_song_obj_skips_none_beat_notes_without_crashing(self):
        song = self._song()
        song["melody"][0]["beat"] = None
        corrupted_song, metadata = corrupt_song_obj(
            copy.deepcopy(song),
            ["borrowed_melody_conflict", "note_onset_shift", "strong_weak_beat_flip"],
            {"rhythm_shift_max_steps": 1},
            self.ctx,
            rng=random.Random(4),
        )
        self.assertIsInstance(corrupted_song, dict)
        self.assertIsInstance(metadata, dict)
        self.assertIn("applied", metadata)

    def test_rhythm_corruptions_keep_onset_metadata(self):
        song = self._song()
        cfg = {"rhythm_shift_max_steps": 1}

        for mode in ["note_onset_shift", "strong_weak_beat_flip"]:
            corrupted_song, meta = corrupt_song_obj(copy.deepcopy(song), [mode], cfg, self.ctx, rng=random.Random(7))
            if not meta["applied"]:
                continue
            self.assertIn("source_onset_beat", meta["details"])
            self.assertIn("target_onset_beat", meta["details"])
            source = float(meta["details"]["source_onset_beat"])
            target = float(meta["details"]["target_onset_beat"])
            expected = self._expected_post_onset_indices(corrupted_song, source, target)
            self.assertEqual(sorted(meta["onset_corrupted_indices"]), expected)

    def test_out_of_key_note_smoke(self):
        song = self._song()
        corrupted_song, meta = corrupt_song_obj(copy.deepcopy(song), ["out_of_key_note"], {}, self.ctx, rng=random.Random(1))
        self.assertTrue(meta["applied"])
        self.assertEqual(meta["mode"], "out_of_key_note")
        idx = meta["note_corrupted_indices"][0]
        new_sd = int(corrupted_song["melody"][idx]["sd_id"])
        new_pc = decode_sd_to_chromatic(new_sd, self.ctx)
        main_mode = self.ctx["scale_id_to_name"][song["meta"]["main_key_scale_id"]]
        self.assertNotIn(new_pc, set(self.ctx["mode_to_pcset"][main_mode]))

    def test_local_semitone_fragment_shift_smoke(self):
        song = self._song()
        song["melody"].extend([
            {"beat": 3.0, "duration": 0.5, "sd_id": self.ctx["sd_token_to_id"]["3"], "octave_id": 5, "is_rest": 0},
            {"beat": 3.5, "duration": 0.5, "sd_id": self.ctx["sd_token_to_id"]["4"], "octave_id": 5, "is_rest": 0},
        ])
        before = [n["sd_id"] for n in song["melody"]]
        corrupted_song, meta = corrupt_song_obj(
            copy.deepcopy(song),
            ["local_semitone_fragment_shift"],
            {},
            self.ctx,
            rng=random.Random(2),
        )
        self.assertTrue(meta["applied"])
        changed = meta["note_corrupted_indices"]
        self.assertGreaterEqual(len(changed), 2)
        after = [n["sd_id"] for n in corrupted_song["melody"]]
        self.assertNotEqual(before, after)
        for idx in changed:
            old_pc = decode_sd_to_chromatic(int(before[idx]), self.ctx)
            new_pc = decode_sd_to_chromatic(int(after[idx]), self.ctx)
            self.assertEqual((new_pc - old_pc) % 12, meta["details"]["shift_semitones"] % 12)

    def test_octave_leap_violation_smoke(self):
        song = self._song()
        corrupted_song, meta = corrupt_song_obj(copy.deepcopy(song), ["octave_leap_violation"], {}, self.ctx, rng=random.Random(3))
        self.assertTrue(meta["applied"])
        idx = meta["details"]["target_note_index"]
        self.assertNotEqual(
            song["melody"][idx]["octave_id"],
            corrupted_song["melody"][idx]["octave_id"],
        )
        self.assertIn("octave_shift", meta["details"])
        self.assertIn("neighbor_octave_id", meta["details"])

    def test_semitone_from_bass_or_chord_tone_smoke(self):
        song = self._song()
        corrupted_song, meta = corrupt_song_obj(
            copy.deepcopy(song),
            ["semitone_from_bass_or_chord_tone"],
            {},
            self.ctx,
            rng=random.Random(4),
        )
        self.assertTrue(meta["applied"])
        self.assertIn(meta["details"]["reference_role"], {"bass", "top_voice"})

        note_idx = meta["details"]["target_note_index"]
        chord_idx = meta["details"]["covering_chord_index"]
        chord = corrupted_song["chords"][chord_idx]
        bass_top = chord_bass_and_top_pcs(corrupted_song, chord, self.ctx)
        self.assertIsNotNone(bass_top)
        bass_pc, top_pc = bass_top

        new_sd = int(corrupted_song["melody"][note_idx]["sd_id"])
        new_pc = decode_sd_to_chromatic(new_sd, self.ctx)
        if meta["details"]["reference_role"] == "bass":
            self.assertIn((new_pc - bass_pc) % 12, {1, 11})
        else:
            self.assertIn((new_pc - top_pc) % 12, {1, 11})

        chord_pcs = chord_pitch_classes_tertian(corrupted_song, chord, self.ctx)
        self.assertIn(meta["details"]["reference_pc"], chord_pcs)


if __name__ == "__main__":
    unittest.main()
