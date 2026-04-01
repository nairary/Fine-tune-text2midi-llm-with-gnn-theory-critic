from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train_teacher import (
    effective_artifact_examples_limit,
    effective_diagnostics_max_batches,
    effective_epochs,
    effective_max_batches,
)


def _cfg(limit_train_batches):
    training = OmegaConf.create({"limit_train_batches": limit_train_batches, "limit_val_batches": None})
    experiment = OmegaConf.create({"limit_train_batches": None, "limit_val_batches": None})
    return training, experiment


def test_effective_max_batches_both_null_returns_none():
    training, experiment = _cfg(None)
    assert effective_max_batches(training, experiment, "train") is None


def test_effective_max_batches_training_when_experiment_null():
    training, experiment = _cfg(5)
    assert effective_max_batches(training, experiment, "train") == 5


def test_effective_max_batches_experiment_overrides_training():
    training, experiment = _cfg(5)
    experiment.limit_train_batches = 1
    assert effective_max_batches(training, experiment, "train") == 1


def test_effective_max_batches_zero_is_error():
    training, experiment = _cfg(0)
    with pytest.raises(ValueError, match="must be > 0"):
        effective_max_batches(training, experiment, "train")


def test_effective_max_batches_string_null_returns_none():
    training, experiment = _cfg("null")
    assert effective_max_batches(training, experiment, "train") is None


def test_effective_max_batches_supports_experiment_cfg_none():
    training = OmegaConf.create({"limit_train_batches": 7, "limit_val_batches": None})
    assert effective_max_batches(training, None, "train") == 7


def test_effective_epochs_uses_training_when_experiment_null():
    training = OmegaConf.create({"epochs": 10})
    experiment = OmegaConf.create({"epochs": None})
    assert effective_epochs(training, experiment) == 10


def test_effective_epochs_experiment_overrides_training():
    training = OmegaConf.create({"epochs": 10})
    experiment = OmegaConf.create({"epochs": 5})
    assert effective_epochs(training, experiment) == 5


def test_effective_epochs_string_null_falls_back_to_training():
    training = OmegaConf.create({"epochs": 10})
    experiment = OmegaConf.create({"epochs": "null"})
    assert effective_epochs(training, experiment) == 10


def test_effective_epochs_zero_raises_value_error():
    training = OmegaConf.create({"epochs": 0})
    experiment = OmegaConf.create({"epochs": None})
    with pytest.raises(ValueError, match="must be > 0"):
        effective_epochs(training, experiment)


def test_effective_diagnostics_max_batches_fallbacks():
    wandb_cfg = OmegaConf.create({"diagnostics_max_scan_batches": 3})
    assert effective_diagnostics_max_batches(wandb_cfg, "val", train_batch_limit=11, val_batch_limit=7) == 3

    wandb_cfg.diagnostics_max_scan_batches = None
    assert effective_diagnostics_max_batches(wandb_cfg, "val", train_batch_limit=11, val_batch_limit=7) == 7
    assert effective_diagnostics_max_batches(wandb_cfg, "train", train_batch_limit=11, val_batch_limit=7) == 11

    wandb_cfg.diagnostics_max_scan_batches = "null"
    assert effective_diagnostics_max_batches(wandb_cfg, "train", train_batch_limit=None, val_batch_limit=7) is None


def test_effective_artifact_examples_limit_resolution():
    assert effective_artifact_examples_limit(OmegaConf.create({"artifact_examples_limit": 2})) == 2
    assert effective_artifact_examples_limit(OmegaConf.create({"artifact_examples_limit": 0})) == 0
    assert effective_artifact_examples_limit(OmegaConf.create({"artifact_examples_limit": None})) == 0
    assert effective_artifact_examples_limit(OmegaConf.create({"artifact_examples_limit": "null"})) == 0

    with pytest.raises(ValueError, match="must be >= 0"):
        effective_artifact_examples_limit(OmegaConf.create({"artifact_examples_limit": -1}))
    with pytest.raises(ValueError, match="must be an integer"):
        effective_artifact_examples_limit(OmegaConf.create({"artifact_examples_limit": "abc"}))
