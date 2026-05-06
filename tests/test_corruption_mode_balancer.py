from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader.corruption_balancer import CorruptionModeBalancer
from src.dataloader.hooktheory_dataset import HookTheoryDataset


class CorruptionModeBalancerTests(unittest.TestCase):
    def test_balancer_pushes_frequent_mode_down(self):
        balancer = CorruptionModeBalancer(["mode_a", "mode_b", "mode_c"])

        first_mode = balancer.ordered_modes(random.Random(0))[0]
        balancer.record_applied(first_mode)
        second_mode = balancer.ordered_modes(random.Random(1))[0]

        self.assertNotEqual(first_mode, second_mode)
        self.assertEqual(balancer.usage_counts()[first_mode], 1)

    def test_balancer_respects_mode_weights(self):
        balancer = CorruptionModeBalancer(["section_mode", "local_mode"], mode_weights={"section_mode": 0.2, "local_mode": 0.8})
        rng = random.Random(0)

        for _ in range(50):
            mode = balancer.ordered_modes(rng)[0]
            balancer.record_applied(mode)

        counts = balancer.usage_counts()
        self.assertGreater(counts["local_mode"], counts["section_mode"] * 2)

    def test_dataset_balances_applied_modes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset.json"
            dataset_path.write_text(
                json.dumps(
                    [
                        {"song_id": "s1", "meta": {}, "melody": [], "chords": []},
                        {"song_id": "s2", "meta": {}, "melody": [], "chords": []},
                        {"song_id": "s3", "meta": {}, "melody": [], "chords": []},
                    ]
                ),
                encoding="utf-8",
            )

            seen_first_modes: list[str] = []
            seen_shuffle_flags: list[bool] = []

            def fake_corrupt_song_obj(song_obj, corruption_modes, corruption_cfg, theory_ctx, rng=None, shuffle_modes=True):
                ordered_modes = list(corruption_modes)
                seen_first_modes.append(str(ordered_modes[0]))
                seen_shuffle_flags.append(bool(shuffle_modes))
                return song_obj, {"applied": True, "corruption_name": ordered_modes[0]}

            random.seed(0)
            with patch("src.dataloader.hooktheory_dataset.build_theory_context", return_value={}), patch(
                "src.dataloader.hooktheory_dataset.build_graph_from_encoded", side_effect=lambda song: {"song_id": song.get("song_id")}
            ), patch(
                "src.dataloader.hooktheory_dataset.mask_graph", side_effect=lambda graph, **kwargs: (graph, {"note": {}})
            ), patch(
                "src.dataloader.hooktheory_dataset.corrupt_song_obj", side_effect=fake_corrupt_song_obj
            ):
                dataset = HookTheoryDataset(
                    json_path=str(dataset_path),
                    corruption_backend="song_theory",
                    corruption_modes=["mode_a", "mode_b", "mode_c"],
                    theory_aware_cfg={"balance_mode_usage": True, "deterministic_per_sample": False},
                )
                _ = dataset[0]
                _ = dataset[1]
                _ = dataset[2]

            self.assertEqual(set(seen_first_modes), {"mode_a", "mode_b", "mode_c"})
            self.assertEqual(seen_shuffle_flags, [False, False, False])
            self.assertEqual(dataset.get_corruption_usage_counts(), {"mode_a": 1, "mode_b": 1, "mode_c": 1})

    def test_dataset_uses_corruption_family_weights(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset.json"
            dataset_path.write_text(
                json.dumps([{"song_id": f"s{i}", "meta": {}, "melody": [], "chords": []} for i in range(50)]),
                encoding="utf-8",
            )

            def fake_corrupt_song_obj(song_obj, corruption_modes, corruption_cfg, theory_ctx, rng=None, shuffle_modes=True):
                ordered_modes = list(corruption_modes)
                return song_obj, {"applied": True, "corruption_name": ordered_modes[0]}

            random.seed(0)
            with patch("src.dataloader.hooktheory_dataset.build_theory_context", return_value={}), patch(
                "src.dataloader.hooktheory_dataset.build_graph_from_encoded", side_effect=lambda song: {"song_id": song.get("song_id")}
            ), patch(
                "src.dataloader.hooktheory_dataset.mask_graph", side_effect=lambda graph, **kwargs: (graph, {"note": {}})
            ), patch(
                "src.dataloader.hooktheory_dataset.corrupt_song_obj", side_effect=fake_corrupt_song_obj
            ):
                dataset = HookTheoryDataset(
                    json_path=str(dataset_path),
                    corruption_backend="song_theory",
                    corruption_modes=["adjacent_section_swap", "strongbeat_nonchord_note"],
                    theory_aware_cfg={
                        "balance_mode_usage": True,
                        "deterministic_per_sample": False,
                        "corruption_family_weights": {"section": 0.2, "local": 0.8},
                    },
                )
                for idx in range(len(dataset)):
                    _ = dataset[idx]

            counts = dataset.get_corruption_usage_counts()
            self.assertGreater(counts["strongbeat_nonchord_note"], counts["adjacent_section_swap"] * 2)

    def test_dataset_disables_balancing_for_deterministic_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset.json"
            dataset_path.write_text(json.dumps([{"song_id": "s1", "meta": {}, "melody": [], "chords": []}]), encoding="utf-8")

            call_args: list[tuple[list[str], bool]] = []

            def fake_corrupt_song_obj(song_obj, corruption_modes, corruption_cfg, theory_ctx, rng=None, shuffle_modes=True):
                call_args.append((list(corruption_modes), bool(shuffle_modes)))
                return song_obj, {"applied": True, "corruption_name": "mode_a"}

            with patch("src.dataloader.hooktheory_dataset.build_theory_context", return_value={}), patch(
                "src.dataloader.hooktheory_dataset.build_graph_from_encoded", side_effect=lambda song: {"song_id": song.get("song_id")}
            ), patch(
                "src.dataloader.hooktheory_dataset.mask_graph", side_effect=lambda graph, **kwargs: (graph, {"note": {}})
            ), patch(
                "src.dataloader.hooktheory_dataset.corrupt_song_obj", side_effect=fake_corrupt_song_obj
            ):
                dataset = HookTheoryDataset(
                    json_path=str(dataset_path),
                    corruption_backend="song_theory",
                    corruption_modes=["mode_a", "mode_b"],
                    theory_aware_cfg={"balance_mode_usage": True, "deterministic_per_sample": True, "deterministic_seed": 5},
                )
                _ = dataset[0]

            self.assertEqual(call_args, [(["mode_a", "mode_b"], True)])
            self.assertEqual(dataset.get_corruption_usage_counts(), {})


if __name__ == "__main__":
    unittest.main()
