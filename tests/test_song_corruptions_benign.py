from __future__ import annotations

import copy
import random
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader.song_corruptions import (
    NEAR_BENIGN_CORRUPTIONS,
    STRICT_BENIGN_CORRUPTIONS,
    corrupt_song_obj,
)
from src.dataloader.theory_helpers import build_theory_context
from src.dataloader.utils_graph import build_graph_from_encoded


class BenignCorruptionsTests(unittest.TestCase):
    def _base_song(self) -> dict:
        return {
            "song_id": "s1",
            "meta": {
                "tonic_pc": 0,
                "main_key_tonic_pc": 0,
                "main_key_tonic_pc_id": 1,
                "main_key_scale_id": 2,
                "main_num_beats_id": 3,
                "main_beat_unit_id": 3,
                "main_num_beats": 4,
                "main_beat_unit": 1,
                "main_bpm": 120.0,
            },
            "key_regions": [{"beat": 1.0, "tonic_pc": 0, "tonic_pc_id": 1}],
            "melody": [
                {"beat": 1.0, "duration": 1.0, "pitch": 60, "sd_id": 4, "octave_id": 5, "is_rest": 0},
                {"beat": 2.0, "duration": 1.0, "pitch": 60, "sd_id": 4, "octave_id": 5, "is_rest": 0},
                {"beat": 3.0, "duration": 2.0, "pitch": 62, "sd_id": 7, "octave_id": 5, "is_rest": 0},
            ],
            "chords": [
                {"beat": 1.0, "duration": 2.0, "pos_in_bar": 0.0, "root_degree_raw": 0, "type_raw": 7, "add_degrees": [7], "root_id": 1, "type_id": 2, "inversion_id": 1, "applied_id": 1, "borrowed_kind_id": 1, "borrowed_mode_name_id": 1},
                {"beat": 3.0, "duration": 2.0, "pos_in_bar": 2.0, "root_degree_raw": 4, "type_raw": 5, "add_degrees": [], "root_id": 5, "type_id": 1, "inversion_id": 1, "applied_id": 1, "borrowed_kind_id": 1, "borrowed_mode_name_id": 1},
            ],
        }

    def _apply(self, song: dict, mode: str, cfg: dict | None = None) -> tuple[dict, dict]:
        theory_ctx = build_theory_context()
        corrupted, metadata = corrupt_song_obj(
            song_obj=copy.deepcopy(song),
            corruption_modes=[mode],
            corruption_cfg=cfg or {},
            theory_ctx=theory_ctx,
            rng=random.Random(7),
        )
        return corrupted, metadata

    def _assert_note_invariants(self, song: dict):
        prev_end = None
        for note in song.get("melody", []):
            if int(note.get("is_rest", 0)) == 1:
                continue
            pitch = int(note["pitch"])
            self.assertGreaterEqual(pitch, 0)
            self.assertLessEqual(pitch, 127)
            start = float(note["beat"])
            end = start + float(note["duration"])
            self.assertGreaterEqual(end, start)
            if prev_end is not None:
                self.assertGreaterEqual(start, prev_end)
            prev_end = end

    def test_transpose_with_tonic_shift_updates_teacher_encoded_key_fields(self):
        song = self._base_song()
        corrupted, metadata = self._apply(song, "transpose_with_tonic_shift", {"transpose_semitones": [2]})

        self.assertTrue(metadata["applied"])
        self.assertEqual(corrupted["meta"]["main_key_tonic_pc"], 2)
        self.assertEqual(corrupted["meta"]["main_key_tonic_pc_id"], 3)
        self.assertEqual(corrupted["key_regions"][0]["tonic_pc"], 2)
        self.assertEqual(corrupted["key_regions"][0]["tonic_pc_id"], 3)
        self.assertEqual([n["pitch"] for n in corrupted["melody"]], [62, 62, 64])
        self._assert_note_invariants(corrupted)

    def test_melody_octave_shift_changes_only_melody(self):
        song = self._base_song()
        before_chords = copy.deepcopy(song["chords"])
        corrupted, metadata = self._apply(song, "melody_octave_shift", {"melody_octave_shifts": [12]})
        self.assertTrue(metadata["applied"])
        self.assertEqual([n["pitch"] for n in corrupted["melody"]], [72, 72, 74])
        self.assertEqual(corrupted["chords"], before_chords)

    def test_merge_repeated_melody_notes(self):
        song = self._base_song()
        corrupted, metadata = self._apply(song, "merge_repeated_melody_notes", {"merge_notes_eps": 1e-4})
        self.assertTrue(metadata["applied"])
        self.assertEqual(len(corrupted["melody"]), 2)
        self.assertAlmostEqual(corrupted["melody"][0]["duration"], 2.0)
        self._assert_note_invariants(corrupted)

    def test_split_long_melody_note(self):
        song = self._base_song()
        corrupted, metadata = self._apply(song, "split_long_melody_note", {"split_min_duration_beats": 1.5})
        self.assertTrue(metadata["applied"])
        self.assertEqual(len(corrupted["melody"]), 4)
        self.assertAlmostEqual(corrupted["melody"][2]["duration"], 1.0, places=6)
        self.assertAlmostEqual(corrupted["melody"][3]["duration"], 1.0, places=6)

    def test_drop_tonic_seventh_updates_type_id_and_type_raw(self):
        song = self._base_song()
        corrupted, metadata = self._apply(song, "drop_tonic_seventh_on_strong_beat")
        self.assertTrue(metadata["applied"])
        self.assertEqual(corrupted["chords"][0]["type_raw"], 5)
        self.assertEqual(corrupted["chords"][0]["type_id"], 1)
        self.assertNotIn(7, corrupted["chords"][0]["add_degrees"])

    def test_not_applicable_contract(self):
        song = self._base_song()
        song["meta"]["main_key_tonic_pc"] = None
        song["meta"]["main_key_tonic_pc_id"] = None
        song["meta"]["tonic_pc"] = None
        song["key_regions"] = []
        for note in song["melody"]:
            note.pop("pitch", None)
        corrupted, metadata = self._apply(song, "transpose_with_tonic_shift", {"transpose_semitones": [2]})
        self.assertEqual(corrupted, song)
        self.assertFalse(metadata["applied"])
        self.assertIsNotNone(metadata["reason_skipped"])

    def test_mode_groups(self):
        self.assertEqual(
            set(STRICT_BENIGN_CORRUPTIONS),
            {"transpose_with_tonic_shift", "merge_repeated_melody_notes", "split_long_melody_note"},
        )
        self.assertEqual(
            set(NEAR_BENIGN_CORRUPTIONS),
            {"melody_octave_shift", "drop_tonic_seventh_on_strong_beat"},
        )

    def test_teacher_graph_changes_for_all_new_benign_corruptions(self):
        song = self._base_song()
        mode_to_cfg = {
            "transpose_with_tonic_shift": {"transpose_semitones": [2]},
            "merge_repeated_melody_notes": {"merge_notes_eps": 1e-4},
            "split_long_melody_note": {"split_min_duration_beats": 1.5},
            "melody_octave_shift": {"melody_octave_shifts": [12]},
            "drop_tonic_seventh_on_strong_beat": {},
        }
        original = build_graph_from_encoded(song)
        for mode, cfg in mode_to_cfg.items():
            corrupted, metadata = self._apply(song, mode, cfg)
            self.assertTrue(metadata["applied"], msg=f"{mode} must be applicable in smoke test")
            mutated = build_graph_from_encoded(corrupted)
            changed = False
            for node_type in ("song", "note", "chord"):
                if node_type in original.node_types and node_type in mutated.node_types:
                    if original[node_type].x.shape == mutated[node_type].x.shape:
                        if not torch.equal(original[node_type].x, mutated[node_type].x):
                            changed = True
                    else:
                        changed = True
            self.assertTrue(changed, msg=f"{mode} should change teacher graph tensors")


if __name__ == "__main__":
    unittest.main()

