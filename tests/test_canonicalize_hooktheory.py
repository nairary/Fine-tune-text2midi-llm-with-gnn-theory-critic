from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.canonicalize_hooktheory import Reporter, canonical_root, normalize_song
from src.data.encode_teacher_features import build_allowed_value_id_map, encode_with_value_map


class CanonicalizeHookTheoryRootTests(unittest.TestCase):
    def test_raw_root_mapping_non_rest(self):
        cases = [
            (1, 0),
            (2, 1),
            (7, 6),
            (8, 7),
        ]
        for raw_root, expected in cases:
            with self.subTest(raw_root=raw_root):
                reporter = Reporter()
                got = canonical_root(raw_root, is_rest=False, reporter=reporter, song_id="song")
                self.assertEqual(got, expected)
                self.assertEqual(reporter.counts.get("unexpected_non_rest_root_zero", 0), 0)

    def test_raw_zero_root_rest_returns_none(self):
        reporter = Reporter()
        got = canonical_root(0, is_rest=True, reporter=reporter, song_id="song")
        self.assertIsNone(got)
        self.assertEqual(reporter.counts.get("unexpected_non_rest_root_zero", 0), 0)

    def test_raw_zero_root_non_rest_warns_and_returns_none(self):
        reporter = Reporter()
        got = canonical_root(0, is_rest=False, reporter=reporter, song_id="song")
        self.assertIsNone(got)
        self.assertEqual(reporter.counts.get("unexpected_non_rest_root_zero", 0), 1)

    def test_normalize_song_respects_is_rest_before_root_mapping(self):
        reporter = Reporter()
        song = {
            "meta": {},
            "melody": [],
            "chords": [
                {"beat": 1.0, "duration": 1.0, "root": 1, "is_rest": True},
            ],
            "sections": [],
        }

        out = normalize_song("song_rest", song, reporter=reporter)
        self.assertEqual(len(out["chords"]), 1)
        self.assertIsNone(out["chords"][0]["root"])

    def test_regression_hooktheory_first_chord_root_one_encodes_to_root_id_one(self):
        reporter = Reporter()
        song = {
            "meta": {},
            "melody": [],
            "chords": [
                {"beat": 1.0, "duration": 1.0, "root": 1, "type": 5, "inversion": 0, "applied": 0, "is_rest": False},
            ],
            "sections": [],
        }

        canonical = normalize_song("_QLgnBnpg-V", song, reporter=reporter)
        canonical_root_value = canonical["chords"][0]["root"]
        self.assertEqual(canonical_root_value, 0)

        root_id_map = build_allowed_value_id_map([0, 1, 2, 3, 4, 5, 6, 7])
        root_id = encode_with_value_map(root_id_map, canonical_root_value, unknown_id=0)

        self.assertEqual(root_id, 1)
        self.assertNotEqual(root_id, 2)


if __name__ == "__main__":
    unittest.main()
