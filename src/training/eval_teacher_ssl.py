from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from hydra import compose, initialize_config_dir

from src.training.train_teacher import (
    build_loaders,
    build_model,
    build_training_stages,
    effective_epochs,
    evaluate,
    load_dynamic_loss_weighter_from_checkpoint,
    print_metrics,
)
from src.training.dynamic_loss_weighting import build_teacher_dynamic_loss_weighter


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a TeacherGNN SSL checkpoint.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("overrides", nargs="*", help="Optional Hydra overrides, e.g. model=teacher_gnn_small")
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = str((ROOT / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=args.overrides)

    device = torch.device(cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    _, _, val_loader = build_loaders(cfg)
    sample = val_loader.dataset[0]

    model = build_model(sample["graph_real"], cfg.model, cfg.losses).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    dynamic_loss_weighter = build_teacher_dynamic_loss_weighter(cfg.losses)
    if dynamic_loss_weighter is not None:
        dynamic_loss_weighter = dynamic_loss_weighter.to(device)
        load_dynamic_loss_weighter_from_checkpoint(checkpoint, dynamic_loss_weighter, strict=False)
    checkpoint_stage = checkpoint.get("stage")
    if checkpoint_stage is None:
        stage_cfg = {
            "name": "legacy_mixed",
            "enable_recon": True,
            "enable_graph_rank": bool(cfg.losses.enable_graph_rank),
            "enable_note_local": bool(cfg.losses.enable_note_local),
            "enable_chord_local": bool(cfg.losses.enable_chord_local),
            "enable_onset_local": bool(cfg.losses.enable_onset_local),
        }
    else:
        stage_plan = build_training_stages(cfg.training, cfg.losses, effective_epochs(cfg.training, cfg.experiment))
        stage_cfg = next((stage for stage in stage_plan if stage["name"] == checkpoint_stage), stage_plan[-1])

    metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        losses_cfg=cfg.losses,
        training_cfg=cfg.training,
        stage_cfg=stage_cfg,
        dynamic_loss_weighter=dynamic_loss_weighter,
        max_batches=cfg.training.limit_val_batches,
    )
    print_metrics("Validation", metrics)


if __name__ == "__main__":
    main()
