from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from src.dataloader.theory_helpers import build_theory_context
from src.observer.chord_parser import ChordCandidate, explain_score_candidate
import src.observer.chord_score_fitting as chord_score_fitting
from src.observer.chord_score_fitting import (
    LearnableChordScore,
    compute_weighted_candidate_score,
    extract_candidate_feature_dict,
    match_candidate_to_ground_truth,
    multi_positive_softmax_loss,
)


class ChordScoreFittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = build_theory_context()

    def test_extract_candidate_feature_dict_matches_explain_terms(self):
        candidate = ChordCandidate(
            mode_name="major",
            borrowed=False,
            root_degree_raw=0,
            type_raw=5,
            inversion_raw=0,
            body_pcs=[0, 4, 7],
            add_degrees=[9],
            suspension_degrees=[],
            omit_degrees=[],
            alteration_tokens=[],
            explained_pcs=[0, 2, 4, 7],
            unexplained_pcs=[],
            missing_core_pcs=[],
            score=0,
        )
        observed_pcs = [0, 2, 4, 7]
        terms = explain_score_candidate(candidate, observed_pcs, 0, "major", self.ctx)
        feats = extract_candidate_feature_dict(candidate, observed_pcs, 0, "major", self.ctx)

        for key, value in terms["positive_terms"].items():
            self.assertAlmostEqual(feats[key], float(value))
        for key, value in terms["negative_terms"].items():
            self.assertAlmostEqual(feats[key], float(value))

    def test_match_candidate_to_ground_truth_exact(self):
        candidate = ChordCandidate(
            mode_name="major",
            borrowed=False,
            root_degree_raw=0,
            type_raw=7,
            inversion_raw=1,
            body_pcs=[0, 4, 7, 11],
            add_degrees=[9],
            suspension_degrees=[4],
            omit_degrees=[5],
            alteration_tokens=["b9"],
            explained_pcs=[0, 1, 4, 5, 7, 11],
            unexplained_pcs=[],
            missing_core_pcs=[],
            score=0,
        )
        gt = {
            "mode_name": "major",
            "root_degree_raw": 0,
            "type_raw": 7,
            "inversion_raw": 1,
            "add_degrees": [9],
            "suspension_degrees": [4],
            "omit_degrees": [5],
            "alteration_tokens": ["b9"],
        }
        self.assertTrue(match_candidate_to_ground_truth(candidate, gt, self.ctx))

    def test_compute_weighted_candidate_score(self):
        features = {
            "body_match_count": 3,
            "extras_explained_count": 1,
            "bass_matches_body": 1,
            "mode_equals_main": 1,
            "unexplained_pcs_count": 0,
            "missing_core_pcs_count": 1,
            "borrowed_mode_penalty": 0,
            "mode_distance_penalty": 0,
            "add_penalty": 1,
            "suspension_penalty": 0,
            "alteration_penalty": 0,
            "omit_penalty": 0,
            "body_size_penalty": 0,
        }
        weights = {
            "bias": 0.5,
            "positive": {
                "body_match_count": 1.0,
                "extras_explained_count": 0.5,
                "bass_matches_body": 0.2,
                "mode_equals_main": 0.3,
            },
            "negative": {
                "unexplained_pcs_count": 2.0,
                "missing_core_pcs_count": 1.5,
                "borrowed_mode_penalty": 1.0,
                "mode_distance_penalty": 0.1,
                "add_penalty": 0.4,
                "suspension_penalty": 0.7,
                "alteration_penalty": 0.8,
                "omit_penalty": 0.9,
                "body_size_penalty": 0.2,
            },
        }
        score = compute_weighted_candidate_score(features, weights)
        self.assertAlmostEqual(score, 2.6)

    def test_multi_positive_softmax_loss_decreases(self):
        param = torch.nn.Parameter(torch.tensor([0.0, -1.0, 0.5], dtype=torch.float32))
        optimizer = torch.optim.SGD([param], lr=0.3)
        positive_mask = torch.tensor([True, False, True])

        before = float(multi_positive_softmax_loss(param, positive_mask).item())
        for _ in range(40):
            optimizer.zero_grad()
            loss = multi_positive_softmax_loss(param, positive_mask)
            loss.backward()
            optimizer.step()
        after = float(multi_positive_softmax_loss(param, positive_mask).item())

        self.assertLess(after, before)

    def test_synthetic_train_loop_outranks_negative(self):
        group = {
            "features": [
                {
                    "body_match_count": 3.0,
                    "extras_explained_count": 1.0,
                    "bass_matches_body": 1.0,
                    "mode_equals_main": 1.0,
                    "unexplained_pcs_count": 0.0,
                    "missing_core_pcs_count": 0.0,
                    "borrowed_mode_penalty": 0.0,
                    "mode_distance_penalty": 0.0,
                    "add_penalty": 0.0,
                    "suspension_penalty": 0.0,
                    "alteration_penalty": 0.0,
                    "omit_penalty": 0.0,
                    "body_size_penalty": 0.0,
                },
                {
                    "body_match_count": 1.0,
                    "extras_explained_count": 0.0,
                    "bass_matches_body": 0.0,
                    "mode_equals_main": 0.0,
                    "unexplained_pcs_count": 2.0,
                    "missing_core_pcs_count": 1.0,
                    "borrowed_mode_penalty": 1.0,
                    "mode_distance_penalty": 2.0,
                    "add_penalty": 1.0,
                    "suspension_penalty": 0.0,
                    "alteration_penalty": 1.0,
                    "omit_penalty": 1.0,
                    "body_size_penalty": 2.0,
                },
            ],
            "positive_mask": [True, False],
            "candidates": [
                {"root_degree_raw": 0, "type_raw": 5},
                {"root_degree_raw": 4, "type_raw": 7},
            ],
        }

        model = LearnableChordScore()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        with torch.no_grad():
            initial_scores = model.score_group(group["features"], device="cpu")
        mask = torch.tensor(group["positive_mask"], dtype=torch.bool)

        for _ in range(60):
            optimizer.zero_grad()
            scores = model.score_group(group["features"], device="cpu")
            loss = multi_positive_softmax_loss(scores, mask)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_scores = model.score_group(group["features"], device="cpu")

        self.assertGreater(float(final_scores[0].item() - final_scores[1].item()), float(initial_scores[0].item() - initial_scores[1].item()))

    def test_train_summary_uses_best_epoch_not_last(self):
        train_metrics = {
            "loss": 0.0,
            "top1_exact_acc": 0.0,
            "topk_contains_gt_acc": 0.0,
            "root_acc": 0.0,
            "type_acc": 0.0,
            "group_count": 1.0,
            "positive_coverage": 1.0,
            "valid_group_count": 1.0,
        }
        val_metrics_seq = [
            {**train_metrics, "top1_exact_acc": 0.1, "loss": 0.8},
            {**train_metrics, "top1_exact_acc": 0.9, "loss": 0.2},
            {**train_metrics, "top1_exact_acc": 0.2, "loss": 0.7},
        ]

        side_effect = []
        for val_row in val_metrics_seq:
            side_effect.extend([dict(train_metrics), dict(val_row)])

        with patch.object(chord_score_fitting, "evaluate_groups", side_effect=side_effect):
            model, summary, metrics_log = chord_score_fitting.train_learnable_chord_score(
                train_groups=[],
                val_groups=[],
                epochs=3,
                lr=0.01,
                weight_decay=0.0,
                seed=123,
                device="cpu",
            )

        self.assertEqual(metrics_log[-1]["epoch"], 3)
        self.assertEqual(summary["epoch"], 2)
        self.assertEqual(summary["val_top1_exact_acc"], 0.9)
        self.assertEqual(summary["val_top1_exact_acc"], max(row["val_top1_exact_acc"] for row in metrics_log))
        self.assertNotEqual(summary["epoch"], metrics_log[-1]["epoch"])
        self.assertEqual(summary["learned_weights"], model.export_weights())

    def test_train_logging_interval_and_new_best(self):
        base_metrics = {
            "loss": 0.5,
            "top1_exact_acc": 0.0,
            "topk_contains_gt_acc": 0.0,
            "root_acc": 0.0,
            "type_acc": 0.0,
            "group_count": 1.0,
            "positive_coverage": 1.0,
            "valid_group_count": 1.0,
        }
        val_top1_seq = [0.1, 0.1, 0.2, 0.15, 0.25]

        side_effect = []
        for epoch, val_top1 in enumerate(val_top1_seq, start=1):
            side_effect.extend(
                [
                    {**base_metrics, "loss": 0.6 - 0.01 * epoch},
                    {**base_metrics, "top1_exact_acc": val_top1, "loss": 0.7 - 0.02 * epoch},
                ]
            )

        with (
            patch.object(chord_score_fitting, "evaluate_groups", side_effect=side_effect),
            patch.object(chord_score_fitting.LOGGER, "info") as mock_info,
        ):
            model, summary, metrics_log = chord_score_fitting.train_learnable_chord_score(
                train_groups=[],
                val_groups=[],
                epochs=5,
                lr=0.01,
                weight_decay=0.0,
                seed=123,
                device="cpu",
                log_every=2,
            )

        info_msgs = [call.args[0] for call in mock_info.call_args_list]
        epoch_logs = [msg for msg in info_msgs if isinstance(msg, str) and msg.startswith("epoch=")]
        best_logs = [msg for msg in info_msgs if isinstance(msg, str) and msg.startswith("new_best")]

        self.assertEqual(len(metrics_log), 5)
        self.assertIsInstance(model, LearnableChordScore)
        self.assertIn("learned_weights", summary)

        self.assertEqual(len(epoch_logs), 4)
        self.assertEqual(len(best_logs), 4)

        logged_epochs = [call.args[1] for call in mock_info.call_args_list if call.args and call.args[0].startswith("epoch=")]
        self.assertEqual(logged_epochs, [1, 2, 4, 5])

        new_best_epochs = [call.args[1] for call in mock_info.call_args_list if call.args and call.args[0].startswith("new_best")]
        self.assertEqual(new_best_epochs, [1, 2, 3, 5])


if __name__ == "__main__":
    unittest.main()
