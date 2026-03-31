from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.teacher_gnn import TeacherGNN


def build_dummy_hetero() -> HeteroData:
    data = HeteroData()
    data["song"].x = torch.randn(2, 4)
    data["onset"].x = torch.randn(4, 5)
    data["note"].x = torch.randn(4, 6)
    data["chord"].x = torch.randn(4, 7)

    data["song"].batch = torch.tensor([0, 1], dtype=torch.long)
    data["onset"].batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    data["note"].batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    data["chord"].batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    data[("onset", "starts_note", "note")].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    data[("onset", "starts_chord", "chord")].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    data[("chord", "covers_note", "note")].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)

    empty = torch.empty((2, 0), dtype=torch.long)
    data[("note", "next_note", "note")].edge_index = empty
    data[("chord", "next_chord", "chord")].edge_index = empty
    data[("onset", "next_onset", "onset")].edge_index = empty

    return data


class HybridGraphScorerSmokeTest(unittest.TestCase):
    def _build_model(self, sample: HeteroData, pooling_mode: str, use_hybrid_graph_scorer: bool) -> TeacherGNN:
        return TeacherGNN.from_hetero_data(
            sample,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            pooling_mode=pooling_mode,
            pooling_output_dim=16,
            score_head_hidden_dim=8,
            reconstruction_head_hidden_dim=8,
            local_score_head_hidden_dim=8,
            use_hybrid_graph_scorer=use_hybrid_graph_scorer,
            local_summary_use_mean=True,
            local_summary_use_max=True,
            local_summary_use_topk_mean=True,
            local_summary_topk=3,
        )

    def test_forward_shapes_in_mean_and_mean_max(self):
        sample = build_dummy_hetero()

        for pooling_mode in ("mean", "mean_max"):
            model = self._build_model(sample, pooling_mode=pooling_mode, use_hybrid_graph_scorer=True)
            outputs = model(sample)

            self.assertEqual(tuple(outputs["graph_embedding"].shape), (2, 16))
            self.assertEqual(tuple(outputs["local_score_summaries"].shape), (2, 9))
            self.assertEqual(tuple(outputs["hybrid_graph_features"].shape), (2, 25))
            self.assertEqual(tuple(outputs["graph_score"].shape), (2,))

            expected_pooled_dim = 16 if pooling_mode == "mean" else 32
            self.assertEqual(outputs["pooled_by_type"]["note"].size(-1), expected_pooled_dim)

    def test_non_hybrid_matches_legacy_graph_score_input_shape(self):
        sample = build_dummy_hetero()
        model = self._build_model(sample, pooling_mode="mean", use_hybrid_graph_scorer=False)
        outputs = model(sample)

        self.assertEqual(tuple(outputs["graph_embedding"].shape), (2, 16))
        self.assertEqual(tuple(outputs["hybrid_graph_features"].shape), (2, 16))
        self.assertEqual(tuple(outputs["local_score_summaries"].shape), (2, 9))


if __name__ == "__main__":
    unittest.main()
