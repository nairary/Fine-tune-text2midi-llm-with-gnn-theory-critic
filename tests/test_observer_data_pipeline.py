from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.observer.data_pipeline import (
    ONSET_EPSILON,
    build_bar_events,
    build_observer_chord_events,
    build_observer_graph,
    build_observer_song_record,
    build_onset_events,
    extract_observer_meta,
    extract_observer_note_events,
    load_observer_input_jsonl,
)
from src.observer.schema import OBSERVER_CAT_FIELDS, OBSERVER_EDGE_TYPES, OBSERVER_NUM_FIELDS


class ObserverDataPipelineTests(unittest.TestCase):
    def setUp(self):
        specs_dir = REPO_ROOT / "metadata" / "specs"
        vocabs_dir = REPO_ROOT / "metadata" / "vocabs"
        self.spec_global = json.loads((specs_dir / "spec_global.json").read_text(encoding="utf-8"))
        self.vocab_scale = json.loads((vocabs_dir / "vocab_key_scale.json").read_text(encoding="utf-8"))
        self.vocab_sd = json.loads((vocabs_dir / "vocab_melody_sd.json").read_text(encoding="utf-8"))
        self.vocab_borrowed_kind = json.loads((vocabs_dir / "vocab_borrowed_kind.json").read_text(encoding="utf-8"))
        self.vocab_borrowed_mode = json.loads((vocabs_dir / "vocab_borrowed_mode_name.json").read_text(encoding="utf-8"))

    def _fake_pm(self, with_tempo: bool = True, with_meter: bool = True):
        def note(pitch: int, start: float, end: float):
            return SimpleNamespace(pitch=pitch, start=start, end=end)

        melody = SimpleNamespace(name="melody", is_drum=False, notes=[note(60, 0.0, 0.5), note(62, 0.5, 1.0)])
        chords = SimpleNamespace(name="chords", is_drum=False, notes=[note(60, 0.0, 1.0), note(64, 0.0, 1.0), note(67, 0.0, 1.0)])

        class FakePM:
            instruments = [melody, chords]
            time_signature_changes = [SimpleNamespace(numerator=4, denominator=4)] if with_meter else []

            def get_tempo_changes(self):
                if with_tempo:
                    return [0.0], [120.0]
                return [], []

            def get_end_time(self):
                return 2.0

        return FakePM()

    def test_input_loader_valid_row(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps({"song_id": "s1", "midi_path": "x.mid", "tonic_pc": 0, "mode_name": "minor", "bpm": 90}) + "\n")
            path = fh.name
        rows = load_observer_input_jsonl(path)
        self.assertEqual(rows[0]["song_id"], "s1")
        self.assertEqual(rows[0]["bpm"], 90)

    def test_input_loader_missing_field(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps({"song_id": "s1", "midi_path": "x.mid", "mode_name": "minor"}) + "\n")
            path = fh.name
        with self.assertRaises(ValueError):
            load_observer_input_jsonl(path)

    def test_input_loader_bad_tonic(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps({"song_id": "s1", "midi_path": "x.mid", "tonic_pc": 99, "mode_name": "minor"}) + "\n")
            path = fh.name
        with self.assertRaises(ValueError):
            load_observer_input_jsonl(path)

    def test_input_loader_unknown_mode(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps({"song_id": "s1", "midi_path": "x.mid", "tonic_pc": 0, "mode_name": "weird"}) + "\n")
            path = fh.name
        with self.assertRaises(ValueError):
            load_observer_input_jsonl(path)

    def test_meta_extraction_priority(self):
        pm = self._fake_pm(with_tempo=True, with_meter=True)
        sample = {"song_id": "s", "midi_path": "x.mid", "tonic_pc": 0, "mode_name": "major", "bpm": 90, "num_beats": 3, "beat_unit": 1}
        meta = extract_observer_meta(sample, pm)
        self.assertEqual(meta["bpm"], 90.0)
        self.assertEqual(meta["num_beats"], 3)
        self.assertEqual(meta["beat_unit"], 1)

    def test_meta_extraction_from_midi(self):
        pm = self._fake_pm(with_tempo=True, with_meter=True)
        sample = {"song_id": "s", "midi_path": "x.mid", "tonic_pc": 0, "mode_name": "major"}
        meta = extract_observer_meta(sample, pm)
        self.assertEqual(meta["bpm"], 120.0)
        self.assertEqual(meta["num_beats"], 4)

    def test_note_extraction_relpc_and_beats(self):
        pm = self._fake_pm(with_tempo=True)
        notes = extract_observer_note_events(pm, tonic_pc=0, bpm=120.0)
        self.assertEqual(len(notes), 2)
        self.assertEqual(notes[0]["rel_pc"], 0)
        self.assertTrue(all(n["pitch"] in {60, 62} for n in notes))
        melody_note = next(n for n in notes if n["pitch"] == 62)
        self.assertAlmostEqual(melody_note["beat"], 1.0)
        self.assertAlmostEqual(melody_note["duration_beats"], 1.0)

    def test_note_extraction_raises_without_melody(self):
        pm = self._fake_pm()
        pm.instruments = [SimpleNamespace(name="chords", is_drum=False, notes=[])]
        with self.assertRaises(ValueError):
            extract_observer_note_events(pm, tonic_pc=0, bpm=120.0)

    def test_note_extraction_no_tempo_sets_none(self):
        pm = self._fake_pm(with_tempo=False)
        notes = extract_observer_note_events(pm, tonic_pc=0, bpm=None)
        self.assertIsNone(notes[0]["beat"])
        self.assertIsNone(notes[0]["duration_beats"])

    def test_chord_adapter_uses_external_api(self):
        with patch(
            "src.observer.data_pipeline.predict_observer_chords_for_midi",
            return_value=[{"onset_time": 0.5, "offset_time": 1.0}],
        ) as mock_predict:
            chords = build_observer_chord_events("x.mid", tonic_pc=0, mode_name="major", bpm=120.0)
        self.assertEqual(len(chords), 1)
        self.assertAlmostEqual(chords[0]["beat"], 1.0)
        self.assertAlmostEqual(chords[0]["duration_beats"], 1.0)
        mock_predict.assert_called_once()

    def test_bar_builder(self):
        bars = build_bar_events(end_beat=8.0, num_beats=4, beat_unit=1)
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[1]["start_beat"], 4.0)

    def test_meter_fallback_from_midi_variants(self):
        sample = {"song_id": "s", "midi_path": "x.mid", "tonic_pc": 0, "mode_name": "major"}

        class PM34:
            time_signature_changes = [SimpleNamespace(numerator=3, denominator=4)]

            def get_tempo_changes(self):
                return [0.0], [100.0]

            def get_end_time(self):
                return 1.0

        class PM68:
            time_signature_changes = [SimpleNamespace(numerator=6, denominator=8)]

            def get_tempo_changes(self):
                return [0.0], [100.0]

            def get_end_time(self):
                return 1.0

        class PMNoTS:
            time_signature_changes = []

            def get_tempo_changes(self):
                return [0.0], [100.0]

            def get_end_time(self):
                return 1.0

        meta_34 = extract_observer_meta(sample, PM34())
        self.assertEqual(meta_34["num_beats"], 3)
        self.assertEqual(meta_34["beat_unit"], 1)

        meta_68 = extract_observer_meta(sample, PM68())
        self.assertEqual(meta_68["num_beats"], 6)
        self.assertEqual(meta_68["beat_unit"], 3)

        meta_none = extract_observer_meta(sample, PMNoTS())
        self.assertIsNone(meta_none["num_beats"])
        self.assertIsNone(meta_none["beat_unit"])

    def test_onsets_dedup_and_sorted(self):
        notes = [{"onset_time": 1.0}, {"onset_time": 1.0 + ONSET_EPSILON / 2}, {"onset_time": 0.5}]
        chords = [{"onset_time": 2.0}]
        onsets = build_onset_events(notes, chords, bars=[], bpm=None, num_beats=None)
        self.assertEqual([x["onset_time"] for x in onsets], [0.5, 1.0, 2.0])

    def test_record_and_graph_integration(self):
        pm = self._fake_pm()
        sample = {"song_id": "example_001", "midi_path": "data/example.mid", "tonic_pc": 0, "mode_name": "minor"}
        fake_pretty_midi = SimpleNamespace(PrettyMIDI=lambda _: pm)
        fake_chords = [{
            "onset_time": 0.0,
            "offset_time": 1.0,
            "beat": 0.0,
            "duration": 2.0,
            "duration_beats": 2.0,
            "root_degree_raw": 0,
            "type_raw": 5,
            "inversion_raw": 0,
            "mode_name": "minor",
            "borrowed": False,
            "add_degrees": [],
            "suspension_degrees": [],
            "omit_degrees": [],
            "alteration_tokens": [],
            "score": 1.0,
            "score_source": "manual",
        }]
        with patch.dict(sys.modules, {"pretty_midi": fake_pretty_midi}), patch(
            "src.observer.data_pipeline.predict_observer_chords_for_midi", return_value=fake_chords
        ):
            record = build_observer_song_record(sample)
        self.assertTrue(record["notes"])
        self.assertTrue(record["chords"])

        graph = build_observer_graph(record)
        self.assertEqual(graph["song"].x.shape[0], 1)
        self.assertEqual(graph["note"].x_cat.shape[1], 2)
        self.assertEqual(graph["note"].x_num.shape[1], 4)
        self.assertEqual(graph["chord"].x_cat.shape[1], 5)
        self.assertEqual(graph["song"].x_cat[0, 1].item(), 3)  # "minor" scale_id in teacher vocab
        self.assertIn(("onset", "starts_note", "note"), graph.edge_types)
        self.assertIn(("chord", "covers_note", "note"), graph.edge_types)

    def test_teacher_allowed_value_mapping_ids(self):
        record = {
            "meta": {"tonic_pc": 11, "mode_name": "major", "num_beats": 4, "beat_unit": 1, "bpm": 120.0, "end_beat": 4.0},
            "bars": [{"bar_index": 0, "start_beat": 0.0, "end_beat": 4.0}],
            "onsets": [{"onset_time": 0.0, "beat": 0.0, "bar_index": 0, "pos_in_bar": 0.0}],
            "notes": [{"onset_time": 0.0, "beat": 0.0, "duration_beats": 1.0, "sd_id": 4, "octave_id": 6, "is_rest": False}],
            "chords": [{
                "onset_time": 0.0,
                "beat": 0.0,
                "duration_beats": 1.0,
                "root_degree_raw": 7,
                "type_raw": 13,
                "inversion_raw": 3,
                "mode_name": "major",
                "borrowed": False,
                "add_degrees": [],
                "suspension_degrees": [],
                "omit_degrees": [],
                "alteration_tokens": [],
            }],
        }
        graph = build_observer_graph(record)
        self.assertEqual(graph["song"].x_cat[0, 0].item(), 12)  # tonic 11 -> idx+1
        self.assertEqual(graph["song"].x_cat[0, 2].item(), 3)   # num_beats=4 -> idx+1 in [2,3,4,6]
        self.assertEqual(graph["song"].x_cat[0, 3].item(), 1)   # beat_unit=1 -> idx+1
        self.assertEqual(graph["chord"].x_cat[0, 0].item(), 8)  # root_raw 7 -> idx+1
        self.assertEqual(graph["chord"].x_cat[0, 1].item(), 5)  # type 13 -> idx+1 among [5,7,9,11,13]
        self.assertEqual(graph["chord"].x_cat[0, 2].item(), 4)  # inversion 3 -> idx+1

    def test_unknown_allowed_value_maps_to_zero(self):
        record = {
            "meta": {"tonic_pc": 99, "mode_name": "major", "num_beats": 99, "beat_unit": 99, "bpm": 120.0, "end_beat": 4.0},
            "bars": [],
            "onsets": [],
            "notes": [],
            "chords": [{
                "onset_time": 0.0,
                "beat": 0.0,
                "duration_beats": 1.0,
                "root_degree_raw": 99,
                "type_raw": 99,
                "inversion_raw": 99,
                "mode_name": "major",
                "borrowed": False,
                "add_degrees": [],
                "suspension_degrees": [],
                "omit_degrees": [],
                "alteration_tokens": [],
            }],
        }
        graph = build_observer_graph(record)
        self.assertEqual(graph["song"].x_cat[0, 0].item(), 0)
        self.assertEqual(graph["song"].x_cat[0, 2].item(), 0)
        self.assertEqual(graph["song"].x_cat[0, 3].item(), 0)
        self.assertTrue((graph["chord"].x_cat[0, :3] == 0).all().item())

    def test_vocab_coded_fields_use_teacher_vocab_space(self):
        record = {
            "meta": {"tonic_pc": 0, "mode_name": "minor", "num_beats": 4, "beat_unit": 1, "bpm": 120.0, "end_beat": 4.0},
            "bars": [{"bar_index": 0, "start_beat": 0.0, "end_beat": 4.0}],
            "onsets": [{"onset_time": 0.0, "beat": 0.0, "bar_index": 0, "pos_in_bar": 0.0}],
            "notes": [{"onset_time": 0.0, "beat": 0.0, "duration_beats": 1.0, "sd_id": self.vocab_sd["1"], "octave_id": 6, "is_rest": False}],
            "chords": [{
                "onset_time": 0.0,
                "beat": 0.0,
                "duration_beats": 1.0,
                "root_degree_raw": 0,
                "type_raw": 5,
                "inversion_raw": 0,
                "mode_name": "dorian",
                "borrowed": True,
                "add_degrees": [],
                "suspension_degrees": [],
                "omit_degrees": [],
                "alteration_tokens": [],
            }],
        }
        graph = build_observer_graph(record)
        self.assertEqual(graph["song"].x_cat[0, 1].item(), self.vocab_scale["minor"])
        self.assertEqual(graph["note"].x_cat[0, 0].item(), self.vocab_sd["1"])
        self.assertEqual(graph["chord"].x_cat[0, 3].item(), self.vocab_borrowed_kind["mode_name"])
        self.assertEqual(graph["chord"].x_cat[0, 4].item(), self.vocab_borrowed_mode["dorian"])

    def test_borrowed_pcset_vec_behavior(self):
        record = {
            "meta": {"tonic_pc": 0, "mode_name": "major", "num_beats": 4, "beat_unit": 1, "bpm": 120.0, "end_beat": 4.0},
            "bars": [{"bar_index": 0, "start_beat": 0.0, "end_beat": 4.0}],
            "onsets": [{"onset_time": 0.0, "beat": 0.0, "bar_index": 0, "pos_in_bar": 0.0}],
            "notes": [],
            "chords": [
                {"onset_time": 0.0, "beat": 0.0, "duration_beats": 1.0, "root_degree_raw": 0, "type_raw": 5, "inversion_raw": 0, "mode_name": "major", "borrowed": False, "add_degrees": [], "suspension_degrees": [], "omit_degrees": [], "alteration_tokens": []},
                {"onset_time": 1.0, "beat": 1.0, "duration_beats": 1.0, "root_degree_raw": 1, "type_raw": 7, "inversion_raw": 1, "mode_name": "dorian", "borrowed": True, "add_degrees": [], "suspension_degrees": [], "omit_degrees": [], "alteration_tokens": []},
            ],
        }
        graph = build_observer_graph(record)
        borrowed_slice_start = 6 + 2 + 2 + 6
        borrowed_slice_end = borrowed_slice_start + 12
        non_borrowed_vec = graph["chord"].x_num[0, borrowed_slice_start:borrowed_slice_end]
        borrowed_vec = graph["chord"].x_num[1, borrowed_slice_start:borrowed_slice_end]
        self.assertTrue(torch.equal(non_borrowed_vec, torch.zeros_like(non_borrowed_vec)))
        self.assertGreater(float(borrowed_vec.sum().item()), 0.0)

    def test_graph_contract_and_reverse_edges_policy(self):
        record = {
            "meta": {"tonic_pc": 0, "mode_name": "minor", "num_beats": 4, "beat_unit": 1, "bpm": 120.0, "end_beat": 4.0},
            "bars": [{"bar_index": 0, "start_beat": 0.0, "end_beat": 4.0}],
            "onsets": [{"onset_time": 0.0, "beat": 0.0, "bar_index": 0, "pos_in_bar": 0.0}],
            "notes": [{"onset_time": 0.0, "beat": 0.0, "duration_beats": 1.0, "sd_id": 4, "octave_id": 6}],
            "chords": [{"onset_time": 0.0, "beat": 0.0, "duration_beats": 1.0, "root_degree_raw": 0, "type_raw": 5, "inversion_raw": 0, "mode_name": "minor", "borrowed": False, "add_degrees": [], "suspension_degrees": [], "omit_degrees": [], "alteration_tokens": []}],
        }
        graph = build_observer_graph(record)
        for node_type in ("song", "bar", "onset", "note", "chord"):
            self.assertEqual(graph[node_type].x_cat.dtype, torch.long)
            self.assertEqual(graph[node_type].x_num.dtype, torch.float)
            self.assertEqual(graph[node_type].x_cat.shape[1], len(OBSERVER_CAT_FIELDS[node_type]))
            self.assertEqual(graph[node_type].x_num.shape[1], len(OBSERVER_NUM_FIELDS[node_type]))
            self.assertTrue(torch.equal(graph[node_type].x, torch.cat([graph[node_type].x_cat.float(), graph[node_type].x_num], dim=1)))
        self.assertTrue((graph["note"].x_cat >= 0).all().item())
        self.assertTrue((graph["chord"].x_cat >= 0).all().item())
        self.assertEqual(set(graph.edge_types), set(OBSERVER_EDGE_TYPES))
        self.assertNotIn(("bar", "rev_contains_bar", "song"), graph.edge_types)
        self.assertNotIn("applied_id", OBSERVER_CAT_FIELDS["chord"])
        self.assertNotIn("is_rest", OBSERVER_CAT_FIELDS["note"])
        self.assertNotIn("is_rest", OBSERVER_CAT_FIELDS["chord"])


if __name__ == "__main__":
    unittest.main()
