from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.dataloader.theory_helpers import build_theory_context
from src.observer.chord_parser import ChordCandidate, explain_score_candidate
import src.observer.chord_score_fitting as chord_score_fitting
import scripts.fit_chord_score_weights as fit_script
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
        epoch_logs = [msg for msg in info_msgs if isinstance(msg, str) and msg.startswith("epoch=") and "train_loss" in msg]
        best_logs = [msg for msg in info_msgs if isinstance(msg, str) and msg.startswith("new_best")]

        self.assertEqual(len(metrics_log), 5)
        self.assertIsInstance(model, LearnableChordScore)
        self.assertIn("learned_weights", summary)

        self.assertEqual(len(epoch_logs), 4)
        self.assertEqual(len(best_logs), 4)

        logged_epochs = [
            call.args[1]
            for call in mock_info.call_args_list
            if call.args and call.args[0].startswith("epoch=") and "train_loss" in call.args[0]
        ]
        self.assertEqual(logged_epochs, [1, 2, 4, 5])

        new_best_epochs = [call.args[1] for call in mock_info.call_args_list if call.args and call.args[0].startswith("new_best")]
        self.assertEqual(new_best_epochs, [1, 2, 3, 5])

    def test_iter_chunked_song_items_splits_expected_sizes(self):
        song_items = [(f"s{i}", {"meta": {"split": "train"}}) for i in range(5)]
        chunks = list(chord_score_fitting.iter_chunked_song_items(song_items, chunk_size=2))
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])

    def test_train_chunked_uses_chunk_provider_without_train_groups_list(self):
        base_group = {
            "features": [{"body_match_count": 1.0}],
            "positive_mask": [True],
            "candidates": [{"root_degree_raw": 0, "type_raw": 5}],
        }
        train_call_counter = {"count": 0}
        val_call_counter = {"count": 0}

        def train_provider():
            train_call_counter["count"] += 1
            yield [dict(base_group)], {"songs_total": 1}

        def val_provider():
            val_call_counter["count"] += 1
            yield [dict(base_group)], {"songs_total": 1}

        with patch.object(chord_score_fitting, "evaluate_group_chunks") as mock_eval_chunks:
            mock_eval_chunks.return_value = {
                "loss": 0.1,
                "top1_exact_acc": 1.0,
                "topk_contains_gt_acc": 1.0,
                "root_acc": 1.0,
                "type_acc": 1.0,
                "group_count": 1.0,
                "positive_coverage": 1.0,
                "valid_group_count": 1.0,
            }
            _, _, metrics_log = chord_score_fitting.train_learnable_chord_score(
                train_group_chunks=train_provider,
                val_group_chunks=val_provider,
                epochs=3,
                lr=0.01,
                weight_decay=0.0,
                device="cpu",
            )

        self.assertEqual(len(metrics_log), 3)
        self.assertGreater(train_call_counter["count"], 0)
        self.assertTrue(mock_eval_chunks.called)

    def test_collect_chunked_train_stats_merges_chunk_stats(self):
        chunks = [
            ([], {"songs_total": 2, "events_total": 4, "groups_kept": 1}),
            ([], {"songs_total": 1, "events_total": 3, "events_positive_missing": 2, "groups_kept": 0}),
        ]
        merged = fit_script.collect_chunked_train_stats(chunks)
        self.assertEqual(merged["songs_total"], 3)
        self.assertEqual(merged["events_total"], 7)
        self.assertEqual(merged["events_positive_missing"], 2)
        self.assertEqual(merged["groups_kept"], 1)

    def test_main_save_train_groups_json_materializes_train_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            encoded_path = tmp / "encoded.json"
            encoded_path.write_text(json.dumps({"song": {"meta": {"split": "train"}, "chords": []}}), encoding="utf-8")
            outdir = tmp / "out"
            args = SimpleNamespace(
                encoded_json=encoded_path,
                midi_root=tmp / "midi",
                train_split="train",
                val_split="val",
                instrument_name="chords",
                epochs=1,
                lr=0.01,
                weight_decay=0.0,
                seed=123,
                log_every=1,
                chunk_size=2,
                eval_every=1,
                limit_train=None,
                limit_val=None,
                outdir=outdir,
                device="cpu",
                save_train_groups_json=True,
                save_val_groups_json=False,
                verbose=False,
            )

            class _DummyModel:
                def export_weights(self):
                    return {"bias": 0.0, "positive": {}, "negative": {}}

                def state_dict(self):
                    return {}

            train_groups_payload = [{"features": [], "positive_mask": []}]
            train_stats_payload = fit_script.empty_build_stats() | {"songs_total": 1, "groups_kept": 1}
            val_stats_payload = fit_script.empty_build_stats()
            with (
                patch.object(fit_script, "build_arg_parser") as mock_parser,
                patch.object(fit_script, "build_theory_context", return_value={}),
                patch.object(fit_script, "build_training_groups", side_effect=[([], val_stats_payload), (train_groups_payload, train_stats_payload)]),
                patch.object(fit_script, "train_learnable_chord_score", return_value=(_DummyModel(), {"epoch": 1}, [{"epoch": 1}])),
                patch.object(fit_script, "save_json") as mock_save_json,
                patch.object(fit_script.torch, "save"),
            ):
                mock_parser.return_value.parse_args.return_value = args
                fit_script.main()

            saved_train_payloads = [
                call.args[1] for call in mock_save_json.call_args_list if str(call.args[0]).endswith("train_groups.json")
            ]
            self.assertEqual(saved_train_payloads, [train_groups_payload])

    def test_train_groups_without_positives_do_not_update_weights(self):
        no_positive_group = {
            "features": [{"body_match_count": 2.0}],
            "positive_mask": [False],
        }

        def provider():
            yield [dict(no_positive_group)], {"songs_total": 1}

        model_before = LearnableChordScore()
        before_state = {k: v.detach().clone() for k, v in model_before.state_dict().items()}
        with patch.object(chord_score_fitting, "LearnableChordScore", return_value=model_before):
            model_after, _, _ = chord_score_fitting.train_learnable_chord_score(
                train_group_chunks=provider,
                val_groups=[],
                epochs=2,
                lr=0.1,
                weight_decay=0.0,
                device="cpu",
            )

        for key, tensor in model_after.state_dict().items():
            self.assertTrue(torch.equal(tensor.detach(), before_state[key]))


if __name__ == "__main__":
    unittest.main()
