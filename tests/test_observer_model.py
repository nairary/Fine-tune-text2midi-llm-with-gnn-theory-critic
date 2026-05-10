from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader.theory_helpers import build_theory_context
from src.observer.data_pipeline import build_observer_graph
from src.observer.model import ObserverGNN
from src.observer.schema import OBSERVER_EDGE_TYPES, OBSERVER_NUM_FIELDS, build_observer_vocab_sizes


class ObserverModelTests(unittest.TestCase):
    def test_forward_scalar_score(self):
        record = {
            "song_id": "synthetic",
            "midi_path": "synthetic.mid",
            "meta": {
                "tonic_pc": 0,
                "mode_name": "minor",
                "bpm": 120.0,
                "num_beats": 4,
                "beat_unit": 1,
                "end_beat": 8.0,
            },
            "bars": [
                {"bar_index": 0, "start_beat": 0.0, "end_beat": 4.0},
                {"bar_index": 1, "start_beat": 4.0, "end_beat": 8.0},
            ],
            "onsets": [
                {"onset_time": 0.0, "beat": 0.0, "bar_index": 0, "pos_in_bar": 0.0},
                {"onset_time": 0.5, "beat": 1.0, "bar_index": 0, "pos_in_bar": 1.0},
                {"onset_time": 1.0, "beat": 2.0, "bar_index": 0, "pos_in_bar": 2.0},
            ],
            "notes": [
                {"onset_time": 0.0, "offset_time": 0.5, "beat": 0.0, "duration_beats": 1.0, "pitch": 60, "rel_pc": 0, "sd_id": 4, "octave_id": 6},
                {"onset_time": 0.5, "offset_time": 1.0, "beat": 1.0, "duration_beats": 1.0, "pitch": 62, "rel_pc": 2, "sd_id": 7, "octave_id": 6},
            ],
            "chords": [
                {
                    "onset_time": 0.0,
                    "offset_time": 1.0,
                    "beat": 0.0,
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
                }
            ],
        }

        graph = build_observer_graph(record)
        batch = Batch.from_data_list([graph])

        spec_global = json.loads((REPO_ROOT / "metadata" / "specs" / "spec_global.json").read_text(encoding="utf-8"))
        theory_ctx = build_theory_context()
        model = ObserverGNN(
            cat_vocab_sizes=build_observer_vocab_sizes(theory_ctx, spec_global),
            num_feature_dims={
                "song": len(OBSERVER_NUM_FIELDS["song"]),
                "bar": len(OBSERVER_NUM_FIELDS["bar"]),
                "onset": len(OBSERVER_NUM_FIELDS["onset"]),
                "note": len(OBSERVER_NUM_FIELDS["note"]),
                "chord": len(OBSERVER_NUM_FIELDS["chord"]),
            },
            edge_types=OBSERVER_EDGE_TYPES,
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
            cat_embedding_dim=8,
        )

        out = model(batch)
        self.assertEqual(tuple(out.shape), (1,))
        self.assertTrue(torch.isfinite(out).all())

        outputs = model(batch, return_outputs=True)
        self.assertEqual(tuple(outputs["score"].shape), (1,))
        self.assertEqual(tuple(outputs["graph_embedding"].shape), (1, model.pooling_output_dim))
        self.assertIn("song", outputs["pooled_by_type"])
        self.assertTrue(torch.isfinite(outputs["graph_embedding"]).all())

        sequence_model = ObserverGNN(
            cat_vocab_sizes=build_observer_vocab_sizes(theory_ctx, spec_global),
            num_feature_dims={
                "song": len(OBSERVER_NUM_FIELDS["song"]),
                "bar": len(OBSERVER_NUM_FIELDS["bar"]),
                "onset": len(OBSERVER_NUM_FIELDS["onset"]),
                "note": len(OBSERVER_NUM_FIELDS["note"]),
                "chord": len(OBSERVER_NUM_FIELDS["chord"]),
            },
            edge_types=OBSERVER_EDGE_TYPES,
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
            cat_embedding_dim=8,
            pooling_mode="mean_max",
            use_bar_sequence_transformer=True,
            score_head_activation="leaky_relu",
            score_head_layer_norm=True,
        )
        sequence_out = sequence_model(batch)
        self.assertEqual(tuple(sequence_out.shape), (1,))
        self.assertTrue(torch.isfinite(sequence_out).all())

    def test_configurable_score_head_uses_norm_and_non_dead_activation(self):
        spec_global = json.loads((REPO_ROOT / "metadata" / "specs" / "spec_global.json").read_text(encoding="utf-8"))
        theory_ctx = build_theory_context()
        model = ObserverGNN(
            cat_vocab_sizes=build_observer_vocab_sizes(theory_ctx, spec_global),
            num_feature_dims={node_type: len(OBSERVER_NUM_FIELDS[node_type]) for node_type in OBSERVER_NUM_FIELDS},
            edge_types=OBSERVER_EDGE_TYPES,
            hidden_dim=32,
            num_layers=1,
            dropout=0.0,
            cat_embedding_dim=8,
            score_head_activation="leaky_relu",
            score_head_layer_norm=True,
        )

        self.assertIsInstance(model.graph_head[0], torch.nn.LayerNorm)
        self.assertIsInstance(model.graph_head[2], torch.nn.LeakyReLU)


if __name__ == "__main__":
    unittest.main()
