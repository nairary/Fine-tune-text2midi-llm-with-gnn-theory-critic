from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.teacher_heads import RECONSTRUCTION_SPECS
from src.training.teacher_losses import compute_teacher_ssl_losses
from src.training.train_teacher import build_training_stages


def _loss_cfg(
    *,
    enable_graph_rank: bool = True,
    enable_note_local: bool = True,
    enable_chord_local: bool = True,
    enable_onset_local: bool = True,
):
    return OmegaConf.create(
        {
            "enable_graph_rank": enable_graph_rank,
            "enable_note_local": enable_note_local,
            "enable_chord_local": enable_chord_local,
            "enable_onset_local": enable_onset_local,
        }
    )


def _training_cfg(total_epochs: int, mlm_ssl_epochs=None, corruption_epochs=None):
    return OmegaConf.create(
        {
            "epochs": total_epochs,
            "mlm_ssl_epochs": mlm_ssl_epochs,
            "corruption_epochs": corruption_epochs,
        }
    )


def test_build_training_stages_auto_splits_epochs_and_orders_stages():
    training_cfg = _training_cfg(total_epochs=5)
    losses_cfg = _loss_cfg()

    stages = build_training_stages(training_cfg, losses_cfg, total_epochs=5)

    assert [stage["name"] for stage in stages] == ["mlm_ssl", "corruption"]
    assert stages[0]["epochs"] == 2
    assert stages[1]["epochs"] == 3
    assert stages[0]["enable_recon"] is True
    assert stages[0]["enable_graph_rank"] is False
    assert stages[1]["enable_recon"] is False
    assert stages[1]["enable_graph_rank"] is True


def test_build_training_stages_rejects_inconsistent_epoch_sum():
    training_cfg = _training_cfg(total_epochs=5, mlm_ssl_epochs=4, corruption_epochs=4)
    losses_cfg = _loss_cfg()

    with pytest.raises(ValueError, match="must equal total epochs"):
        build_training_stages(training_cfg, losses_cfg, total_epochs=5)


def test_compute_teacher_ssl_losses_supports_mlm_stage_without_corruption_outputs():
    note_spec = RECONSTRUCTION_SPECS["note_sd"]
    valid_ids = list(note_spec["valid_ids"])
    logits = torch.zeros((1, len(valid_ids)), dtype=torch.float)
    logits[0, 0] = 2.0

    masked_outputs = {"recon_logits": {"note_sd": logits}}
    masked_batch = {
        "note": SimpleNamespace(
            ptr=torch.tensor([0, 1], dtype=torch.long),
            x=torch.zeros((1, 1), dtype=torch.float),
        )
    }
    masked_labels = [
        {
            "note": {
                "field_names": ["sd_id"],
                "indices": torch.tensor([0], dtype=torch.long),
                "target_values": {"sd_id": torch.tensor([valid_ids[0]], dtype=torch.long)},
            }
        }
    ]

    loss_dict, metric_dict = compute_teacher_ssl_losses(
        masked_outputs=masked_outputs,
        masked_batch=masked_batch,
        masked_labels=masked_labels,
        enable_recon=True,
        enable_graph_rank=False,
        enable_note_local=False,
        enable_chord_local=False,
        enable_onset_local=False,
        enabled_heads={"note_sd": True},
        recon_weights={"note_sd": 1.0},
    )

    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["rank_loss"].item() == pytest.approx(0.0)
    assert metric_dict["rank_acc"].item() == pytest.approx(0.0)
    assert "note_sd_acc" in metric_dict


def test_compute_teacher_ssl_losses_supports_corruption_stage_without_reconstruction_outputs():
    real_outputs = {"graph_score": torch.tensor([1.5], dtype=torch.float)}
    corrupted_outputs = {
        "graph_score": torch.tensor([0.25], dtype=torch.float),
        "local_scores": {
            "note": torch.tensor([0.0], dtype=torch.float),
            "chord": torch.tensor([0.0], dtype=torch.float),
            "onset": torch.tensor([0.0], dtype=torch.float),
        },
    }

    loss_dict, metric_dict = compute_teacher_ssl_losses(
        real_outputs=real_outputs,
        corrupted_outputs=corrupted_outputs,
        enable_recon=False,
        enable_graph_rank=True,
        enable_note_local=False,
        enable_chord_local=False,
        enable_onset_local=False,
    )

    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["recon_loss"].item() == pytest.approx(0.0)
    assert loss_dict["rank_loss"].item() > 0.0
    assert metric_dict["rank_acc"].item() == pytest.approx(1.0)
