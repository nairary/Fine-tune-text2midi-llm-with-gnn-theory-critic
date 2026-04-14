from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.observer.train_observer import main


class EmptyDataset:
    def __len__(self) -> int:
        return 0


class NonEmptyDataset:
    def __len__(self) -> int:
        return 1


class TrainObserverEntrypointTests(unittest.TestCase):
    def _args(self, out_dir: Path) -> Namespace:
        return Namespace(
            train_input_jsonl=Path("train_input.jsonl"),
            train_target_jsonl=Path("train_targets.jsonl"),
            val_input_jsonl=Path("val_input.jsonl"),
            val_target_jsonl=Path("val_targets.jsonl"),
            output_dir=out_dir,
            batch_size=2,
            epochs=1,
            lr=1e-3,
            weight_decay=0.0,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            device="cpu",
            num_workers=0,
            seed=1,
            loss="mse",
            in_memory=False,
            chord_weights_yaml=None,
            chord_instrument_name="chords",
            use_fallback_44=True,
        )

    def test_empty_train_dataset_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._args(Path(tmp_dir) / "out")
            with patch("src.observer.train_observer.parse_args", return_value=args), patch(
                "src.observer.train_observer.ObserverDataset", side_effect=[EmptyDataset(), NonEmptyDataset()]
            ):
                with self.assertRaisesRegex(ValueError, "Train dataset is empty"):
                    main()

    def test_empty_val_dataset_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._args(Path(tmp_dir) / "out")
            with patch("src.observer.train_observer.parse_args", return_value=args), patch(
                "src.observer.train_observer.ObserverDataset", side_effect=[NonEmptyDataset(), EmptyDataset()]
            ):
                with self.assertRaisesRegex(ValueError, "Validation dataset is empty"):
                    main()


if __name__ == "__main__":
    unittest.main()
