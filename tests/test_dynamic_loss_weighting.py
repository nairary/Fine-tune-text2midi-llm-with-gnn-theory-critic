from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.dynamic_loss_weighting import DynamicLossWeighter, build_teacher_dynamic_loss_weighter
from src.training.train_teacher import (
    build_optimizer,
    collect_dynamic_teacher_objectives,
    load_dynamic_loss_weighter_from_checkpoint,
    save_checkpoint,
)


def test_dynamic_loss_weighter_matches_fixed_sum_at_zero_log_var_and_backprops():
    weighter = DynamicLossWeighter(["recon", "graph_rank"], init_log_var=0.0)
    losses = {
        "recon": torch.tensor(2.0, requires_grad=True),
        "graph_rank": torch.tensor(4.0, requires_grad=True),
    }
    base_weights = {"recon": 1.0, "graph_rank": 0.5}

    total, metrics = weighter(losses, base_weights)
    total.backward()

    assert total.item() == pytest.approx(4.0)
    assert metrics["dynamic_weight_recon"].item() == pytest.approx(1.0)
    assert metrics["dynamic_weight_graph_rank"].item() == pytest.approx(0.5)
    assert metrics["dynamic_active_objectives"].item() == pytest.approx(2.0)
    assert weighter.log_vars["recon"].grad is not None
    assert weighter.log_vars["graph_rank"].grad is not None


def test_collect_dynamic_teacher_objectives_honors_stage_and_allowed_objectives():
    losses_cfg = OmegaConf.create(
        {
            "lambda_recon": 1.0,
            "lambda_graph_rank": 0.5,
            "lambda_note_local": 0.25,
            "lambda_chord_local": 0.125,
            "lambda_onset_local": 0.75,
        }
    )
    stage_cfg = {
        "enable_recon": False,
        "enable_graph_rank": True,
        "enable_note_local": True,
        "enable_chord_local": False,
        "enable_onset_local": True,
    }
    loss_dict = {
        "recon_loss": torch.tensor(10.0),
        "rank_loss": torch.tensor(1.0),
        "note_local_loss": torch.tensor(2.0),
        "chord_local_loss": torch.tensor(3.0),
        "onset_local_loss": torch.tensor(4.0),
    }

    objective_losses, base_weights = collect_dynamic_teacher_objectives(
        loss_dict,
        losses_cfg,
        stage_cfg,
        allowed_objectives={"graph_rank", "note_local"},
    )

    assert set(objective_losses) == {"graph_rank", "note_local"}
    assert base_weights == {"graph_rank": 0.5, "note_local": 0.25}


def test_build_teacher_dynamic_loss_weighter_respects_config_switch():
    disabled_cfg = OmegaConf.create({"dynamic_weighting": {"enabled": False}})
    assert build_teacher_dynamic_loss_weighter(disabled_cfg) is None

    enabled_cfg = OmegaConf.create(
        {
            "dynamic_weighting": {
                "enabled": True,
                "method": "uncertainty",
                "objectives": {
                    "recon": True,
                    "graph_rank": False,
                    "note_local": True,
                    "chord_local": False,
                    "onset_local": False,
                },
            }
        }
    )

    weighter = build_teacher_dynamic_loss_weighter(enabled_cfg)

    assert isinstance(weighter, DynamicLossWeighter)
    assert weighter.objective_names == ("recon", "note_local")


def test_optimizer_adds_dynamic_parameters_without_weight_decay():
    model = nn.Linear(2, 1)
    weighter = DynamicLossWeighter(["recon"])
    optimizer_cfg = OmegaConf.create({"name": "adamw", "lr": 1e-3, "weight_decay": 1e-2, "betas": [0.9, 0.999]})

    optimizer = build_optimizer(model, optimizer_cfg, extra_parameters=weighter.parameters())

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-2)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(0.0)


def test_checkpoint_roundtrip_restores_dynamic_loss_weighter(tmp_path):
    model = nn.Linear(2, 1)
    weighter = DynamicLossWeighter(["recon"])
    with torch.no_grad():
        weighter.log_vars["recon"].fill_(1.25)
    optimizer = AdamW(list(model.parameters()) + list(weighter.parameters()), lr=1e-3)
    checkpoint_path = tmp_path / "teacher.pt"

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=3,
        metrics={"loss": 1.0},
        stage_name="mlm_ssl",
        stage_epoch=2,
        dynamic_loss_weighter=weighter,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = DynamicLossWeighter(["recon"])
    loaded = load_dynamic_loss_weighter_from_checkpoint(checkpoint, restored)

    assert loaded is True
    assert restored.log_vars["recon"].item() == pytest.approx(1.25)
