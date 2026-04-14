from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.observer.dataset import ObserverDataset, ObserverDatasetValidationError
from src.observer.train_observer import create_loss, run_epoch


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_simple_graph() -> HeteroData:
    graph = HeteroData()
    graph["song"].x_cat = torch.zeros((1, 4), dtype=torch.long)
    graph["song"].x_num = torch.zeros((1, 2), dtype=torch.float)
    graph["song"].x = torch.zeros((1, 6), dtype=torch.float)

    graph["bar"].x_cat = torch.empty((0, 0), dtype=torch.long)
    graph["bar"].x_num = torch.empty((0, 6), dtype=torch.float)
    graph["bar"].x = torch.empty((0, 6), dtype=torch.float)

    graph["onset"].x_cat = torch.empty((0, 0), dtype=torch.long)
    graph["onset"].x_num = torch.empty((0, 5), dtype=torch.float)
    graph["onset"].x = torch.empty((0, 5), dtype=torch.float)

    graph["note"].x_cat = torch.empty((0, 2), dtype=torch.long)
    graph["note"].x_num = torch.empty((0, 4), dtype=torch.float)
    graph["note"].x = torch.empty((0, 6), dtype=torch.float)

    graph["chord"].x_cat = torch.empty((0, 5), dtype=torch.long)
    graph["chord"].x_num = torch.empty((0, 32), dtype=torch.float)
    graph["chord"].x = torch.empty((0, 37), dtype=torch.float)

    for edge_type in (
        ("song", "contains_bar", "bar"),
        ("bar", "next_bar", "bar"),
        ("bar", "contains_onset", "onset"),
        ("onset", "next_onset", "onset"),
        ("onset", "starts_note", "note"),
        ("onset", "starts_chord", "chord"),
        ("note", "next_note", "note"),
        ("chord", "next_chord", "chord"),
        ("chord", "covers_note", "note"),
    ):
        graph[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
    return graph


class FakeObserverModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        return batch.y.view(-1) + self.bias


class ObserverDatasetTests(unittest.TestCase):
    def test_dataset_matches_targets_and_attaches_y(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_path = tmp / "input.jsonl"
            target_path = tmp / "targets.jsonl"
            _write_jsonl(input_path, [{"song_id": "s1", "midi_path": "a.mid", "tonic_pc": 0, "mode_name": "minor"}])
            _write_jsonl(target_path, [{"song_id": "s1", "teacher_score": 0.75}])

            with patch("src.observer.dataset.build_observer_song_record", return_value={"song_id": "s1"}), patch(
                "src.observer.dataset.build_observer_graph", return_value=_make_simple_graph()
            ):
                dataset = ObserverDataset(input_path, target_path)
                graph = dataset[0]

        self.assertTrue(torch.isfinite(graph.y).all())
        self.assertEqual(graph.y.shape, (1,))
        self.assertAlmostEqual(float(graph.y.item()), 0.75)

    def test_missing_target_raises_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_path = tmp / "input.jsonl"
            target_path = tmp / "targets.jsonl"
            _write_jsonl(input_path, [{"song_id": "s1", "midi_path": "a.mid", "tonic_pc": 0, "mode_name": "minor"}])
            _write_jsonl(target_path, [{"song_id": "s2", "teacher_score": 0.1}])

            with self.assertRaisesRegex(ObserverDatasetValidationError, "Missing teacher_score"):
                ObserverDataset(input_path, target_path)

    def test_duplicate_song_id_in_targets_raises_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_path = tmp / "input.jsonl"
            target_path = tmp / "targets.jsonl"
            _write_jsonl(input_path, [{"song_id": "s1", "midi_path": "a.mid", "tonic_pc": 0, "mode_name": "minor"}])
            _write_jsonl(target_path, [{"song_id": "s1", "teacher_score": 0.2}, {"song_id": "s1", "teacher_score": 0.3}])

            with self.assertRaisesRegex(ObserverDatasetValidationError, "Duplicate song_id"):
                ObserverDataset(input_path, target_path)

    def test_extra_target_row_raises_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_path = tmp / "input.jsonl"
            target_path = tmp / "targets.jsonl"
            _write_jsonl(input_path, [{"song_id": "s1", "midi_path": "a.mid", "tonic_pc": 0, "mode_name": "minor"}])
            _write_jsonl(target_path, [{"song_id": "s1", "teacher_score": 0.2}, {"song_id": "s2", "teacher_score": 0.3}])

            with self.assertRaisesRegex(ObserverDatasetValidationError, "absent from input manifest"):
                ObserverDataset(input_path, target_path)

    def test_in_memory_prebuild_works(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_path = tmp / "input.jsonl"
            target_path = tmp / "targets.jsonl"
            _write_jsonl(input_path, [{"song_id": "s1", "midi_path": "a.mid", "tonic_pc": 0, "mode_name": "minor"}])
            _write_jsonl(target_path, [{"song_id": "s1", "teacher_score": 0.75}])

            with patch("src.observer.dataset.build_observer_song_record", return_value={"song_id": "s1"}), patch(
                "src.observer.dataset.build_observer_graph", return_value=_make_simple_graph()
            ) as mock_graph:
                dataset = ObserverDataset(input_path, target_path, in_memory=True)
                _ = dataset[0]
                _ = dataset[0]

        self.assertEqual(mock_graph.call_count, 1)

    def test_smoke_batch_forward_backward(self):
        class TinyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                graph = _make_simple_graph()
                graph.y = torch.tensor([float(idx)], dtype=torch.float)
                return graph

        loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)
        model = FakeObserverModel()
        criterion = create_loss("mse")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        metrics = run_epoch(model, loader, criterion, device=torch.device("cpu"), optimizer=optimizer)
        self.assertTrue(math.isfinite(metrics.loss))


if __name__ == "__main__":
    unittest.main()
