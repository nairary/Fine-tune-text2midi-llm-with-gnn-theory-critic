from __future__ import annotations

import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.hooktheory_dataset import HookTheoryDataset, collate_fn
from src.models.teacher_gnn import TeacherGNN
from src.training.teacher_losses import compute_teacher_ssl_losses

LOGGER = logging.getLogger(__name__)


class SplitFilteredDataset(Dataset):
    def __init__(self, base_dataset: HookTheoryDataset, split: str | None = None):
        self.base_dataset = base_dataset
        if split is None:
            self.indices = list(range(len(base_dataset)))
        else:
            self.indices = [
                index
                for index, song_obj in enumerate(base_dataset.data)
                if song_obj.get("meta", {}).get("split") == split
            ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base_dataset[self.indices[index]]


class MetricTracker:
    def __init__(self):
        self.sums = defaultdict(float)
        self.weights = defaultdict(float)

    def update(self, values: Mapping[str, float | torch.Tensor], weight: float):
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = float(value.detach().cpu().item())
            self.sums[key] += float(value) * weight
            self.weights[key] += weight

    def average(self) -> Dict[str, float]:
        averaged = {}
        for key, total in self.sums.items():
            weight = self.weights[key]
            if weight > 0:
                averaged[key] = total / weight
        return averaged


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    try:
        base_dir = Path(get_original_cwd())
    except Exception:
        base_dir = ROOT
    return base_dir / path


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        "graph_real": batch["graph_real"].to(device),
        "graph_masked": batch["graph_masked"].to(device),
        "graph_corrupted": batch["graph_corrupted"].to(device),
        "masked_labels": batch["masked_labels"],
        "corruption_metadata": batch["corruption_metadata"],
        "graph_score_label": batch["graph_score_label"].to(device),
    }


def build_model(sample_graph, model_cfg: DictConfig, losses_cfg: DictConfig) -> TeacherGNN:
    return TeacherGNN.from_hetero_data(
        sample_graph,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        residual=model_cfg.use_residual,
        encoder_hidden_dims=list(model_cfg.encoder_hidden_dims),
        pooling_mode=model_cfg.pooling_mode,
        pooling_output_dim=model_cfg.pooling_output_dim,
        score_head_hidden_dim=model_cfg.score_head_hidden_dim,
        reconstruction_head_hidden_dim=model_cfg.reconstruction_head_hidden_dim,
        enabled_heads=OmegaConf.to_container(losses_cfg.enabled_heads, resolve=True),
        use_note_score_head=bool(model_cfg.use_note_score_head),
        use_chord_score_head=bool(model_cfg.use_chord_score_head),
        use_onset_score_head=bool(model_cfg.use_onset_score_head),
        local_score_head_hidden_dim=model_cfg.local_score_head_hidden_dim,
    )


def build_optimizer(model: TeacherGNN, optimizer_cfg: DictConfig):
    if optimizer_cfg.name != "adamw":
        raise ValueError(f"Unsupported optimizer '{optimizer_cfg.name}'.")
    betas = tuple(float(beta) for beta in optimizer_cfg.betas)
    return AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.lr),
        weight_decay=float(optimizer_cfg.weight_decay),
        betas=betas,
    )


def build_scheduler(optimizer: AdamW, scheduler_cfg: DictConfig):
    if scheduler_cfg.name == "none":
        return None
    if scheduler_cfg.name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.t_max),
            eta_min=float(scheduler_cfg.eta_min),
        )
    raise ValueError(f"Unsupported scheduler '{scheduler_cfg.name}'.")


def build_loaders(cfg: DictConfig):
    json_path = resolve_path(cfg.data.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")

    dataset = HookTheoryDataset(
        str(json_path),
        mask_prob=float(cfg.dataloader.mask_prob),
        mask_min_nodes=int(cfg.dataloader.mask_min_nodes),
        optional_mask_field_prob=float(cfg.dataloader.optional_mask_field_prob),
        corruption_modes=list(cfg.dataloader.corruption_modes),
    )
    train_dataset = SplitFilteredDataset(dataset, split=cfg.data.split.train)
    val_dataset = SplitFilteredDataset(dataset, split=cfg.data.split.val)

    if cfg.training.limit_train_samples is not None:
        train_dataset.indices = train_dataset.indices[: int(cfg.training.limit_train_samples)]
    if cfg.training.limit_val_samples is not None:
        val_dataset.indices = val_dataset.indices[: int(cfg.training.limit_val_samples)]

    if len(train_dataset) == 0:
        raise ValueError(f"No samples found for train split '{cfg.data.split.train}'.")
    if len(val_dataset) == 0:
        raise ValueError(f"No samples found for val split '{cfg.data.split.val}'.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=bool(cfg.dataloader.shuffle),
        num_workers=int(cfg.dataloader.num_workers),
        pin_memory=bool(cfg.dataloader.pin_memory),
        drop_last=bool(cfg.dataloader.drop_last),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=False,
        num_workers=int(cfg.dataloader.num_workers),
        pin_memory=bool(cfg.dataloader.pin_memory),
        drop_last=False,
        collate_fn=collate_fn,
    )
    return dataset, train_loader, val_loader


def effective_max_batches(training_cfg: DictConfig, experiment_cfg: DictConfig, split: str) -> int | None:
    experiment_value = experiment_cfg.get(f"limit_{split}_batches")
    training_value = training_cfg.get(f"limit_{split}_batches")
    return experiment_value if experiment_value is not None else training_value


def effective_epochs(training_cfg: DictConfig, experiment_cfg: DictConfig) -> int:
    return int(experiment_cfg.epochs) if experiment_cfg.get("epochs") is not None else int(training_cfg.epochs)


def loss_cfg_to_runtime(losses_cfg: DictConfig) -> tuple[dict, dict]:
    recon_weights = OmegaConf.to_container(losses_cfg.recon_weights, resolve=True)
    enabled_heads = OmegaConf.to_container(losses_cfg.enabled_heads, resolve=True)
    return recon_weights, enabled_heads


def run_epoch(
    model: TeacherGNN,
    loader: DataLoader,
    device: torch.device,
    losses_cfg: DictConfig,
    training_cfg: DictConfig,
    optimizer: AdamW | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    max_batches: int | None = None,
):
    is_train = optimizer is not None
    model.train(is_train)
    tracker = MetricTracker()
    recon_weights, enabled_heads = loss_cfg_to_runtime(losses_cfg)
    grad_clip = float(training_cfg.grad_clip) if training_cfg.grad_clip is not None else None
    autocast_enabled = bool(training_cfg.use_amp and device.type == "cuda")

    for step_index, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            masked_outputs = model(batch["graph_masked"])
            real_outputs = model(batch["graph_real"])
            corrupted_outputs = model(batch["graph_corrupted"])
            loss_dict, metric_dict = compute_teacher_ssl_losses(
                masked_outputs=masked_outputs,
                real_outputs=real_outputs,
                corrupted_outputs=corrupted_outputs,
                masked_batch=batch["graph_masked"],
                masked_labels=batch["masked_labels"],
                lambda_recon=float(losses_cfg.lambda_recon),
                lambda_graph_rank=float(losses_cfg.lambda_graph_rank),
                lambda_note_local=float(losses_cfg.lambda_note_local),
                lambda_chord_local=float(losses_cfg.lambda_chord_local),
                lambda_onset_local=float(losses_cfg.lambda_onset_local),
                enable_graph_rank=bool(losses_cfg.enable_graph_rank),
                enable_note_local=bool(losses_cfg.enable_note_local),
                enable_chord_local=bool(losses_cfg.enable_chord_local),
                enable_onset_local=bool(losses_cfg.enable_onset_local),
                recon_weights=recon_weights,
                enabled_heads=enabled_heads,
                corruption_metadata=batch["corruption_metadata"],
                corrupted_batch=batch["graph_corrupted"],
                local_negatives_per_positive=int(losses_cfg.local_negatives_per_positive),
            )

        if is_train:
            loss = loss_dict["loss"]
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        scalar_losses = {key: value.detach() for key, value in loss_dict.items() if isinstance(value, torch.Tensor)}
        tracker.update(scalar_losses, weight=1.0)
        tracker.update(metric_dict, weight=1.0)

        if step_index % int(training_cfg.log_every) == 0 or (max_batches is not None and step_index == max_batches):
            LOGGER.info("step=%s metrics=%s", step_index, json.dumps(tracker.average(), sort_keys=True))

        if max_batches is not None and step_index >= max_batches:
            break

    return tracker.average()


@torch.no_grad()
def evaluate(
    model: TeacherGNN,
    loader: DataLoader,
    device: torch.device,
    losses_cfg: DictConfig,
    training_cfg: DictConfig,
    max_batches: int | None = None,
):
    return run_epoch(
        model=model,
        loader=loader,
        device=device,
        losses_cfg=losses_cfg,
        training_cfg=training_cfg,
        optimizer=None,
        scaler=None,
        max_batches=max_batches,
    )


def save_checkpoint(path: Path, model: TeacherGNN, optimizer: AdamW, epoch: int, metrics: Mapping[str, float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": dict(metrics),
        },
        path,
    )


def print_metrics(prefix: str, metrics: Mapping[str, float]):
    ordered_keys = [
        "loss",
        "recon_loss",
        "note_sd_loss",
        "chord_root_loss",
        "chord_type_loss",
        "chord_applied_loss",
        "chord_borrowed_kind_loss",
        "rank_loss",
        "note_local_loss",
        "chord_local_loss",
        "onset_local_loss",
        "note_sd_acc",
        "chord_root_acc",
        "chord_type_acc",
        "chord_applied_acc",
        "chord_borrowed_kind_acc",
        "rank_acc",
        "note_local_acc",
        "chord_local_acc",
        "onset_local_acc",
        "mean_margin",
        "score_real_mean",
        "score_corrupted_mean",
    ]
    rendered = [f"{key}={metrics[key]:.4f}" for key in ordered_keys if key in metrics]
    LOGGER.info("%s: %s", prefix, ", ".join(rendered))


def persist_metrics(output_dir: Path, epoch: int, train_metrics: Mapping[str, float], val_metrics: Mapping[str, float]):
    metrics_path = output_dir / "metrics.jsonl"
    payload = {
        "epoch": epoch,
        "train": dict(train_metrics),
        "val": dict(val_metrics),
    }
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, output_dir / "composed_config.yaml", resolve=True)
    LOGGER.info("Hydra output directory: %s", output_dir)
    LOGGER.info("Composed config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    set_seed(int(cfg.seed), deterministic=bool(cfg.training.deterministic))
    device = torch.device(cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    _, train_loader, val_loader = build_loaders(cfg)
    sample = train_loader.dataset[0]
    model = build_model(sample["graph_real"], cfg.model, cfg.losses).to(device)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.training.use_amp and device.type == "cuda"))

    epochs = effective_epochs(cfg.training, cfg.experiment)
    train_batch_limit = effective_max_batches(cfg.training, cfg.experiment, "train")
    val_batch_limit = effective_max_batches(cfg.training, cfg.experiment, "val")
    checkpoint_dir = output_dir / "checkpoints"
    best_rank_acc = float("-inf")

    metadata = {
        "project": OmegaConf.to_container(cfg.project, resolve=True),
        "run_name": cfg.run_name,
        "device": str(device),
        "dataset_json": str(resolve_path(cfg.data.json_path)),
        "metadata_dir": str(resolve_path(cfg.data.metadata_dir)),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info(
        "Training TeacherGNN on %s with %s train and %s val samples.",
        device.type,
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            losses_cfg=cfg.losses,
            training_cfg=cfg.training,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=train_batch_limit,
        )

        if scheduler is not None:
            scheduler.step()

        if epoch % int(cfg.training.eval_every) == 0:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                losses_cfg=cfg.losses,
                training_cfg=cfg.training,
                max_batches=val_batch_limit,
            )
        else:
            val_metrics = {}

        print_metrics(f"Epoch {epoch:03d} train", train_metrics)
        if val_metrics:
            print_metrics(f"Epoch {epoch:03d} val", val_metrics)
        persist_metrics(output_dir, epoch, train_metrics, val_metrics)

        if epoch % int(cfg.training.save_every) == 0:
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, val_metrics or train_metrics)

        current_rank_acc = val_metrics.get("rank_acc", float("-inf"))
        if current_rank_acc > best_rank_acc:
            best_rank_acc = current_rank_acc
            save_checkpoint(checkpoint_dir / "best_rank_acc.pt", model, optimizer, epoch, val_metrics)


if __name__ == "__main__":
    main()
