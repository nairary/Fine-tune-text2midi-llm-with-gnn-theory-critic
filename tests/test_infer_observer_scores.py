from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.inference.infer_observer_scores import (
    ObserverInferenceError,
    assign_descending_ranks,
    load_input_rows,
    meter_denominator_to_beat_unit,
    normalize_grpo_input_row,
    parse_mode_name,
    parse_tonic_pc,
)


class InferObserverScoresTests(unittest.TestCase):
    def test_parse_tonic_pc_accepts_common_spellings(self):
        self.assertEqual(parse_tonic_pc("C"), 0)
        self.assertEqual(parse_tonic_pc("C#"), 1)
        self.assertEqual(parse_tonic_pc("Db"), 1)
        self.assertEqual(parse_tonic_pc("B♭"), 10)
        self.assertEqual(parse_tonic_pc(11), 11)

    def test_parse_mode_name_normalizes_aliases(self):
        self.assertEqual(parse_mode_name("maj"), "major")
        self.assertEqual(parse_mode_name("Aeolian"), "minor")
        self.assertEqual(parse_mode_name("harmonic minor"), "harmonic_minor")
        self.assertEqual(parse_mode_name("phrygian-dominant"), "phrygian_dominant")

    def test_meter_denominator_to_beat_unit_matches_observer_schema(self):
        self.assertEqual(meter_denominator_to_beat_unit(4, 4), 1)
        self.assertEqual(meter_denominator_to_beat_unit(3, 4), 1)
        self.assertEqual(meter_denominator_to_beat_unit(6, 8), 3)
        self.assertEqual(meter_denominator_to_beat_unit(9, 8), 3)
        self.assertEqual(meter_denominator_to_beat_unit(12, 8), 3)

    def test_normalize_grpo_input_row_accepts_requested_typo_fields(self):
        row = {
            "midi_path": "candidate.mid",
            "key": "Bb",
            "mode": "minor",
            "bpm": 96,
            "meter_numenator": 6,
            "meter_denumenator": 8,
        }

        sample = normalize_grpo_input_row(row, index=3)

        self.assertEqual(sample["song_id"], "candidate_3")
        self.assertEqual(sample["midi_path"], "candidate.mid")
        self.assertEqual(sample["tonic_pc"], 10)
        self.assertEqual(sample["mode_name"], "minor")
        self.assertEqual(sample["bpm"], 96.0)
        self.assertEqual(sample["num_beats"], 6)
        self.assertEqual(sample["beat_unit"], 3)
        self.assertEqual(sample["meter_numerator"], 6)
        self.assertEqual(sample["meter_denominator"], 8)

    def test_normalize_grpo_input_row_accepts_correct_meter_spellings(self):
        row = {
            "id": "cand-a",
            "midi_path": "candidate.mid",
            "key": "F#",
            "mode": "major",
            "bpm": "120",
            "meter_numerator": "4",
            "meter_denominator": "4",
        }

        sample = normalize_grpo_input_row(row, index=0)

        self.assertEqual(sample["song_id"], "cand-a")
        self.assertEqual(sample["sample_id"], "cand-a")
        self.assertEqual(sample["tonic_pc"], 6)
        self.assertEqual(sample["num_beats"], 4)
        self.assertEqual(sample["beat_unit"], 1)

    def test_normalize_grpo_input_row_requires_midi_path(self):
        with self.assertRaisesRegex(ObserverInferenceError, "midi_path"):
            normalize_grpo_input_row(
                {
                    "key": "C",
                    "mode": "major",
                    "bpm": 120,
                    "meter_numenator": 4,
                    "meter_denumenator": 4,
                },
                index=0,
            )

    def test_assign_descending_ranks_preserves_input_rows(self):
        results = [
            {"index": 0, "score": 0.2},
            {"index": 1, "score": 0.8},
            {"index": 2, "score": None},
            {"index": 3, "score": 0.5},
        ]

        assign_descending_ranks(results)

        self.assertEqual([row["index"] for row in results], [0, 1, 2, 3])
        self.assertEqual([row["rank"] for row in results], [3, 1, None, 2])

    def test_load_input_rows_accepts_array_and_items_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            array_path = tmp / "array.json"
            object_path = tmp / "object.json"
            rows = [{"midi_path": "a.mid"}]
            array_path.write_text(json.dumps(rows), encoding="utf-8")
            object_path.write_text(json.dumps({"items": rows}), encoding="utf-8")

            self.assertEqual(load_input_rows(array_path), rows)
            self.assertEqual(load_input_rows(object_path), rows)


if __name__ == "__main__":
    unittest.main()
