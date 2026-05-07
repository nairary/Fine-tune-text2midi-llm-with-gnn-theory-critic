from __future__ import annotations

import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parameter import is_lazy
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.hooktheory_dataset import HookTheoryDataset, collate_fn
from src.dataloader.precomputed_teacher_pairs import PrecomputedTeacherPairDataset
from src.evaluation.teacher_local_metrics import (
    evaluate_teacher_local_corruption,
    save_local_diagnostic_reports,
)
from src.models.teacher_gnn import TeacherGNN
from src.training.dynamic_loss_weighting import DynamicLossWeighter, build_teacher_dynamic_loss_weighter
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


def update_corruption_usage_counts(
    metadata_items: list[Mapping[str, Any] | None],
    *,
    attempted_counts: Counter[str],
    applied_counts: Counter[str],
    skipped_counts: Counter[str],
    skipped_attempt_counts: Counter[str],
    skipped_reason_counts: Counter[str],
    skipped_attempt_reason_counts: Counter[str],
) -> None:
    for metadata in metadata_items:
        if not isinstance(metadata, Mapping):
            skipped_counts["unknown"] += 1
            skipped_reason_counts["missing_metadata"] += 1
            continue
        mode = str(metadata.get("corruption_name") or metadata.get("mode") or "unknown")
        attempted_modes = metadata.get("attempted_corruption_modes")
        if isinstance(attempted_modes, list):
            for attempted_mode in attempted_modes:
                attempted_counts[str(attempted_mode)] += 1
        else:
            attempted_counts[mode] += 1

        skipped_attempts = metadata.get("skipped_corruption_attempts")
        if isinstance(skipped_attempts, list):
            for attempt in skipped_attempts:
                if not isinstance(attempt, Mapping):
                    skipped_attempt_counts["unknown"] += 1
                    skipped_attempt_reason_counts["unknown"] += 1
                    continue
                skipped_mode = str(attempt.get("mode") or "unknown")
                skipped_reason = str(attempt.get("reason") or "unknown")
                skipped_attempt_counts[skipped_mode] += 1
                skipped_attempt_reason_counts[skipped_reason] += 1

        if bool(metadata.get("applied", False)):
            applied_counts[mode] += 1
        else:
            skipped_counts[mode] += 1
            reason = str(metadata.get("reason_skipped") or "unknown")
            skipped_reason_counts[reason] += 1


def add_corruption_usage_metrics(
    metrics: Dict[str, float],
    *,
    attempted_counts: Counter[str],
    applied_counts: Counter[str],
    skipped_counts: Counter[str],
    skipped_attempt_counts: Counter[str],
    skipped_reason_counts: Counter[str],
    skipped_attempt_reason_counts: Counter[str],
) -> Dict[str, float]:
    enriched = dict(metrics)
    enriched["corruption_attempted_total"] = float(sum(attempted_counts.values()))
    enriched["corruption_applied_total"] = float(sum(applied_counts.values()))
    enriched["corruption_skipped_total"] = float(sum(skipped_counts.values()))
    enriched["corruption_skipped_attempt_total"] = float(sum(skipped_attempt_counts.values()))
    for mode, count in sorted(attempted_counts.items()):
        enriched[f"corruption_attempted_{mode}"] = float(count)
    for mode, count in sorted(applied_counts.items()):
        enriched[f"corruption_applied_{mode}"] = float(count)
    for mode, count in sorted(skipped_counts.items()):
        enriched[f"corruption_skipped_{mode}"] = float(count)
    for mode, count in sorted(skipped_attempt_counts.items()):
        enriched[f"corruption_skipped_attempt_{mode}"] = float(count)
    for reason, count in sorted(skipped_reason_counts.items()):
        enriched[f"corruption_skipped_reason_{reason}"] = float(count)
    for reason, count in sorted(skipped_attempt_reason_counts.items()):
        enriched[f"corruption_skipped_attempt_reason_{reason}"] = float(count)
    return enriched


def print_corruption_usage(prefix: str, metrics: Mapping[str, float]) -> None:
    attempted = {
        key.removeprefix("corruption_attempted_"): int(value)
        for key, value in metrics.items()
        if key.startswith("corruption_attempted_") and key != "corruption_attempted_total"
    }
    applied = {
        key.removeprefix("corruption_applied_"): int(value)
        for key, value in metrics.items()
        if key.startswith("corruption_applied_") and key != "corruption_applied_total"
    }
    skipped = {
        key.removeprefix("corruption_skipped_"): int(value)
        for key, value in metrics.items()
        if key.startswith("corruption_skipped_")
        and key != "corruption_skipped_total"
        and not key.startswith("corruption_skipped_reason_")
        and not key.startswith("corruption_skipped_attempt_")
    }
    skipped_attempts = {
        key.removeprefix("corruption_skipped_attempt_"): int(value)
        for key, value in metrics.items()
        if key.startswith("corruption_skipped_attempt_")
        and key != "corruption_skipped_attempt_total"
        and not key.startswith("corruption_skipped_attempt_reason_")
    }
    skipped_reasons = {
        key.removeprefix("corruption_skipped_reason_"): int(value)
        for key, value in metrics.items()
        if key.startswith("corruption_skipped_reason_")
        and not key.startswith("corruption_skipped_attempt_reason_")
    }
    skipped_attempt_reasons = {
        key.removeprefix("corruption_skipped_attempt_reason_"): int(value)
        for key, value in metrics.items()
        if key.startswith("corruption_skipped_attempt_reason_")
    }
    if not attempted and not applied and not skipped and not skipped_attempts:
        return
    LOGGER.info(
        "%s corruption_usage: attempted_total=%s, applied_total=%s, skipped_total=%s, skipped_attempt_total=%s, attempted_by_mode=%s, applied_by_mode=%s, skipped_by_mode=%s, skipped_attempts_by_mode=%s, skipped_reasons=%s, skipped_attempt_reasons=%s",
        prefix,
        int(metrics.get("corruption_attempted_total", 0)),
        int(metrics.get("corruption_applied_total", 0)),
        int(metrics.get("corruption_skipped_total", 0)),
        int(metrics.get("corruption_skipped_attempt_total", 0)),
        json.dumps(attempted, sort_keys=True),
        json.dumps(applied, sort_keys=True),
        json.dumps(skipped, sort_keys=True),
        json.dumps(skipped_attempts, sort_keys=True),
        json.dumps(skipped_reasons, sort_keys=True),
        json.dumps(skipped_attempt_reasons, sort_keys=True),
    )


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


def _move_optional_graph(graph, device: torch.device, *, required: bool):
    if graph is None:
        if required:
            raise ValueError("Required graph branch is missing from the batch.")
        return None
    return graph.to(device) if required else graph


def move_batch_to_device(
    batch: dict,
    device: torch.device,
    *,
    need_real: bool = True,
    need_masked: bool = True,
    need_corrupted: bool = True,
) -> dict:
    return {
        "graph_real": _move_optional_graph(batch["graph_real"], device, required=need_real),
        "graph_masked": _move_optional_graph(batch["graph_masked"], device, required=need_masked),
        "graph_corrupted": _move_optional_graph(batch["graph_corrupted"], device, required=need_corrupted),
        "masked_labels": batch["masked_labels"],
        "corruption_metadata": batch["corruption_metadata"],
        "graph_score_label": batch["graph_score_label"].to(device),
    }


def set_dataset_stage_outputs(dataset, *, masked: bool, corrupted: bool) -> None:
    if hasattr(dataset, "set_stage_outputs"):
        dataset.set_stage_outputs(masked=masked, corrupted=corrupted)
    if hasattr(dataset, "base_dataset"):
        set_dataset_stage_outputs(dataset.base_dataset, masked=masked, corrupted=corrupted)
    if hasattr(dataset, "dataset"):
        set_dataset_stage_outputs(dataset.dataset, masked=masked, corrupted=corrupted)


def set_loader_stage_outputs(loader: DataLoader, *, masked: bool, corrupted: bool) -> None:
    set_dataset_stage_outputs(loader.dataset, masked=masked, corrupted=corrupted)


def build_model(sample_graph, model_cfg: DictConfig, losses_cfg: DictConfig) -> TeacherGNN:
    return TeacherGNN.from_hetero_data(
        sample_graph,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        residual=model_cfg.use_residual,
        backbone=str(model_cfg.get("backbone", "sage")),
        hgt_num_heads=int(model_cfg.get("hgt_num_heads", 4)),
        encoder_hidden_dims=list(model_cfg.encoder_hidden_dims),
        pooling_mode=model_cfg.pooling_mode,
        pooling_attention_hidden_dim=model_cfg.get("pooling_attention_hidden_dim"),
        pooling_type_attention=bool(model_cfg.get("pooling_type_attention", False)),
        pooling_output_dim=model_cfg.pooling_output_dim,
        score_head_hidden_dim=model_cfg.score_head_hidden_dim,
        reconstruction_head_hidden_dim=model_cfg.reconstruction_head_hidden_dim,
        enabled_heads=OmegaConf.to_container(losses_cfg.enabled_heads, resolve=True),
        use_note_score_head=bool(model_cfg.use_note_score_head),
        use_chord_score_head=bool(model_cfg.use_chord_score_head),
        use_onset_score_head=bool(model_cfg.use_onset_score_head),
        local_score_head_hidden_dim=model_cfg.local_score_head_hidden_dim,
        local_context_mode=str(model_cfg.get("local_context_mode", "mean")),
        local_context_num_heads=int(model_cfg.get("local_context_num_heads", 4)),
        use_hybrid_graph_scorer=bool(model_cfg.use_hybrid_graph_scorer),
        score_fusion_mode=str(model_cfg.get("score_fusion_mode", "none")),
        score_fusion_hidden_dim=model_cfg.get("score_fusion_hidden_dim"),
        local_summary_use_mean=bool(model_cfg.local_summary_use_mean),
        local_summary_use_max=bool(model_cfg.local_summary_use_max),
        local_summary_use_topk_mean=bool(model_cfg.local_summary_use_topk_mean),
        local_summary_topk=int(model_cfg.local_summary_topk),
    )


def build_optimizer(model: TeacherGNN, optimizer_cfg: DictConfig, extra_parameters=None):
    if optimizer_cfg.name != "adamw":
        raise ValueError(f"Unsupported optimizer '{optimizer_cfg.name}'.")
    betas = tuple(float(beta) for beta in optimizer_cfg.betas)
    param_groups = [{"params": list(model.parameters())}]
    extra_parameters = list(extra_parameters) if extra_parameters is not None else []
    if extra_parameters:
        param_groups.append({"params": extra_parameters, "weight_decay": 0.0})
    return AdamW(
        param_groups,
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


def build_stage_scheduler(optimizer: AdamW, scheduler_cfg: DictConfig, stage_epochs: int):
    if scheduler_cfg.name == "none":
        return None
    if scheduler_cfg.name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(stage_epochs)),
            eta_min=float(scheduler_cfg.eta_min),
        )
    raise ValueError(f"Unsupported scheduler '{scheduler_cfg.name}'.")


def build_loaders(cfg: DictConfig):
    if str(cfg.dataloader.get("source", "")) == "precomputed_pairs" or str(
        cfg.dataloader.get("corruption_backend", "")
    ) == "precomputed_pairs":
        pair_corpus_root = resolve_path(str(cfg.dataloader.pair_corpus_root))
        manifest_dir = str(cfg.dataloader.get("manifest_output_dir", "pairs/manifests"))
        pair_index_dir = str(cfg.dataloader.get("pair_index_output_dir", "pairs/index"))
        teacher_graph_index_dir = cfg.dataloader.get("teacher_graph_index_dir")
        train_split_name = str(cfg.dataloader.get("precomputed_train_split", "train"))
        val_split_name = str(cfg.dataloader.get("precomputed_val_split", "val"))
        train_graph_index = (
            pair_corpus_root / str(teacher_graph_index_dir) / f"{train_split_name}.jsonl"
            if teacher_graph_index_dir
            else None
        )
        val_graph_index = (
            pair_corpus_root / str(teacher_graph_index_dir) / f"{val_split_name}.jsonl"
            if teacher_graph_index_dir
            else None
        )

        train_dataset = PrecomputedTeacherPairDataset(
            pair_index_jsonl=pair_corpus_root / pair_index_dir / f"{train_split_name}_pairs.jsonl",
            manifest_jsonl=pair_corpus_root / manifest_dir / f"{train_split_name}.jsonl",
            mask_prob=float(cfg.dataloader.mask_prob),
            mask_min_nodes=int(cfg.dataloader.mask_min_nodes),
            optional_mask_field_prob=float(cfg.dataloader.optional_mask_field_prob),
            base_dir=resolve_path("."),
            graph_index_jsonl=train_graph_index,
        )
        val_dataset = PrecomputedTeacherPairDataset(
            pair_index_jsonl=pair_corpus_root / pair_index_dir / f"{val_split_name}_pairs.jsonl",
            manifest_jsonl=pair_corpus_root / manifest_dir / f"{val_split_name}.jsonl",
            mask_prob=float(cfg.dataloader.mask_prob),
            mask_min_nodes=int(cfg.dataloader.mask_min_nodes),
            optional_mask_field_prob=float(cfg.dataloader.optional_mask_field_prob),
            base_dir=resolve_path("."),
            graph_index_jsonl=val_graph_index,
        )

        if cfg.training.limit_train_samples is not None:
            train_dataset = Subset(
                train_dataset,
                list(range(min(len(train_dataset), int(cfg.training.limit_train_samples)))),
            )
        if cfg.training.limit_val_samples is not None:
            val_dataset = Subset(
                val_dataset,
                list(range(min(len(val_dataset), int(cfg.training.limit_val_samples)))),
            )

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
        return None, train_loader, val_loader

    json_path = resolve_path(cfg.data.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")

    dataset = HookTheoryDataset(
        str(json_path),
        mask_prob=float(cfg.dataloader.mask_prob),
        mask_min_nodes=int(cfg.dataloader.mask_min_nodes),
        optional_mask_field_prob=float(cfg.dataloader.optional_mask_field_prob),
        corruption_modes=list(cfg.dataloader.corruption_modes),
        corruption_backend=str(cfg.dataloader.get("corruption_backend", "graph")),
        theory_aware_cfg=OmegaConf.to_container(cfg.dataloader.get("theory_aware", {}), resolve=True),
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


def collect_dynamic_teacher_objectives(
    loss_dict: Mapping[str, torch.Tensor],
    losses_cfg: DictConfig,
    stage_cfg: Mapping[str, Any],
    *,
    allowed_objectives: set[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    objective_specs = {
        "recon": ("recon_loss", "lambda_recon", "enable_recon"),
        "graph_rank": ("rank_loss", "lambda_graph_rank", "enable_graph_rank"),
        "note_local": ("note_local_loss", "lambda_note_local", "enable_note_local"),
        "chord_local": ("chord_local_loss", "lambda_chord_local", "enable_chord_local"),
        "onset_local": ("onset_local_loss", "lambda_onset_local", "enable_onset_local"),
    }
    objective_losses: dict[str, torch.Tensor] = {}
    base_weights: dict[str, float] = {}

    for objective_name, (loss_key, weight_key, enable_key) in objective_specs.items():
        if allowed_objectives is not None and objective_name not in allowed_objectives:
            continue
        if not bool(stage_cfg.get(enable_key, False)):
            continue
        if loss_key not in loss_dict:
            continue
        objective_losses[objective_name] = loss_dict[loss_key]
        base_weights[objective_name] = float(losses_cfg.get(weight_key))

    return objective_losses, base_weights


def build_training_stages(training_cfg: DictConfig, losses_cfg: DictConfig, total_epochs: int) -> list[dict[str, Any]]:
    mlm_ssl_epochs = training_cfg.get("mlm_ssl_epochs")
    corruption_epochs = training_cfg.get("corruption_epochs")
    mlm_ssl_epochs = None if mlm_ssl_epochs is None else int(mlm_ssl_epochs)
    corruption_epochs = None if corruption_epochs is None else int(corruption_epochs)

    if mlm_ssl_epochs is None and corruption_epochs is None:
        mlm_ssl_epochs = max(1, total_epochs // 2) if total_epochs > 0 else 0
        corruption_epochs = total_epochs - mlm_ssl_epochs
    elif mlm_ssl_epochs is None:
        corruption_epochs = max(0, corruption_epochs)
        mlm_ssl_epochs = total_epochs - corruption_epochs
    elif corruption_epochs is None:
        mlm_ssl_epochs = max(0, mlm_ssl_epochs)
        corruption_epochs = total_epochs - mlm_ssl_epochs

    if mlm_ssl_epochs < 0 or corruption_epochs < 0:
        raise ValueError("training.mlm_ssl_epochs and training.corruption_epochs must be non-negative.")
    if mlm_ssl_epochs + corruption_epochs != total_epochs:
        raise ValueError(
            "The staged training plan is inconsistent: "
            f"mlm_ssl_epochs ({mlm_ssl_epochs}) + corruption_epochs ({corruption_epochs}) "
            f"must equal total epochs ({total_epochs})."
        )

    stages: list[dict[str, Any]] = []
    if mlm_ssl_epochs > 0:
        stages.append(
            {
                "name": "mlm_ssl",
                "epochs": mlm_ssl_epochs,
                "enable_recon": True,
                "enable_graph_rank": False,
                "enable_note_local": False,
                "enable_chord_local": False,
                "enable_onset_local": False,
                "selection_metric": "recon_loss",
                "selection_mode": "min",
                "best_checkpoint_name": "best_recon_loss.pt",
                "run_local_eval": False,
            }
        )

    corruption_objectives_enabled = any(
        (
            bool(losses_cfg.enable_graph_rank),
            bool(losses_cfg.enable_note_local),
            bool(losses_cfg.enable_chord_local),
            bool(losses_cfg.enable_onset_local),
        )
    )
    if corruption_epochs > 0:
        if not corruption_objectives_enabled:
            raise ValueError(
                "training.corruption_epochs is positive, but all corruption objectives are disabled in losses config."
            )
        selection_metric = "rank_acc" if bool(losses_cfg.enable_graph_rank) else "loss"
        selection_mode = "max" if selection_metric == "rank_acc" else "min"
        stages.append(
            {
                "name": "corruption",
                "epochs": corruption_epochs,
                "enable_recon": False,
                "enable_graph_rank": bool(losses_cfg.enable_graph_rank),
                "enable_note_local": bool(losses_cfg.enable_note_local),
                "enable_chord_local": bool(losses_cfg.enable_chord_local),
                "enable_onset_local": bool(losses_cfg.enable_onset_local),
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "best_checkpoint_name": "best_rank_acc.pt" if selection_metric == "rank_acc" else "best_corruption_loss.pt",
                "run_local_eval": any(
                    (
                        bool(losses_cfg.enable_note_local),
                        bool(losses_cfg.enable_chord_local),
                        bool(losses_cfg.enable_onset_local),
                    )
                ),
            }
        )

    if not stages:
        raise ValueError("No training stages were scheduled. Increase training.epochs or stage-specific epoch counts.")
    return stages


def metric_improved(current_value: float, best_value: float, mode: str) -> bool:
    if mode == "max":
        return current_value > best_value
    if mode == "min":
        return current_value < best_value
    raise ValueError(f"Unsupported selection mode '{mode}'.")


def run_epoch(
    model: TeacherGNN,
    loader: DataLoader,
    device: torch.device,
    losses_cfg: DictConfig,
    training_cfg: DictConfig,
    stage_cfg: Mapping[str, Any],
    optimizer: AdamW | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    dynamic_loss_weighter: DynamicLossWeighter | None = None,
    max_batches: int | None = None,
):
    is_train = optimizer is not None
    model.train(is_train)
    if dynamic_loss_weighter is not None:
        dynamic_loss_weighter.train(is_train)
    tracker = MetricTracker()
    corruption_attempted_counts: Counter[str] = Counter()
    corruption_applied_counts: Counter[str] = Counter()
    corruption_skipped_counts: Counter[str] = Counter()
    corruption_skipped_attempt_counts: Counter[str] = Counter()
    corruption_skipped_reason_counts: Counter[str] = Counter()
    corruption_skipped_attempt_reason_counts: Counter[str] = Counter()
    recon_weights, enabled_heads = loss_cfg_to_runtime(losses_cfg)
    grad_clip = float(training_cfg.grad_clip) if training_cfg.grad_clip is not None else None
    autocast_enabled = bool(training_cfg.use_amp and device.type == "cuda")
    enable_recon = bool(stage_cfg.get("enable_recon", True))
    enable_graph_rank = bool(stage_cfg.get("enable_graph_rank", bool(losses_cfg.enable_graph_rank)))
    enable_note_local = bool(stage_cfg.get("enable_note_local", bool(losses_cfg.enable_note_local)))
    enable_chord_local = bool(stage_cfg.get("enable_chord_local", bool(losses_cfg.enable_chord_local)))
    enable_onset_local = bool(stage_cfg.get("enable_onset_local", bool(losses_cfg.enable_onset_local)))
    require_corrupted_outputs = any((enable_graph_rank, enable_note_local, enable_chord_local, enable_onset_local))
    set_loader_stage_outputs(loader, masked=enable_recon, corrupted=require_corrupted_outputs)
    grad_clip_parameters = list(model.parameters())
    if dynamic_loss_weighter is not None:
        grad_clip_parameters.extend(dynamic_loss_weighter.parameters())

    for step_index, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(
            batch,
            device,
            need_real=enable_graph_rank,
            need_masked=enable_recon,
            need_corrupted=require_corrupted_outputs,
        )
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            masked_outputs = (
                model(
                    batch["graph_masked"],
                    compute_recon=True,
                    compute_graph_score=False,
                    compute_local_scores=False,
                )
                if enable_recon
                else None
            )
            real_outputs = (
                model(
                    batch["graph_real"],
                    compute_recon=False,
                    compute_graph_score=True,
                    compute_local_scores=False,
                )
                if enable_graph_rank
                else None
            )
            corrupted_outputs = (
                model(
                    batch["graph_corrupted"],
                    compute_recon=False,
                    compute_graph_score=enable_graph_rank,
                    compute_local_scores=any((enable_note_local, enable_chord_local, enable_onset_local)),
                )
                if require_corrupted_outputs
                else None
            )
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
                graph_rank_intra_weight=float(losses_cfg.graph_rank_intra_weight),
                graph_rank_inter_weight=float(losses_cfg.graph_rank_inter_weight),
                graph_binary_weight=float(losses_cfg.graph_binary_weight),
                graph_rank_margin=float(losses_cfg.graph_rank_margin),
                enable_recon=enable_recon,
                enable_graph_rank=enable_graph_rank,
                enable_note_local=enable_note_local,
                enable_chord_local=enable_chord_local,
                enable_onset_local=enable_onset_local,
                recon_weights=recon_weights,
                enabled_heads=enabled_heads,
                corruption_metadata=batch["corruption_metadata"],
                corrupted_batch=batch["graph_corrupted"],
                local_negatives_per_positive=int(losses_cfg.local_negatives_per_positive),
            )
            if dynamic_loss_weighter is not None:
                objective_losses, base_weights = collect_dynamic_teacher_objectives(
                    loss_dict,
                    losses_cfg,
                    stage_cfg,
                    allowed_objectives=set(dynamic_loss_weighter.objective_names),
                )
                dynamic_loss, dynamic_metrics = dynamic_loss_weighter(objective_losses, base_weights)
                loss_dict["loss"] = dynamic_loss
                metric_dict.update(dynamic_metrics)

        if is_train:
            loss = loss_dict["loss"]
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(grad_clip_parameters, grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(grad_clip_parameters, grad_clip)
                optimizer.step()
        scalar_losses = {key: value.detach() for key, value in loss_dict.items() if isinstance(value, torch.Tensor)}
        tracker.update(scalar_losses, weight=1.0)
        tracker.update(metric_dict, weight=1.0)
        if require_corrupted_outputs:
            update_corruption_usage_counts(
                batch["corruption_metadata"],
                attempted_counts=corruption_attempted_counts,
                applied_counts=corruption_applied_counts,
                skipped_counts=corruption_skipped_counts,
                skipped_attempt_counts=corruption_skipped_attempt_counts,
                skipped_reason_counts=corruption_skipped_reason_counts,
                skipped_attempt_reason_counts=corruption_skipped_attempt_reason_counts,
            )

        if step_index % int(training_cfg.log_every) == 0 or (max_batches is not None and step_index == max_batches):
            step_metrics = add_corruption_usage_metrics(
                tracker.average(),
                attempted_counts=corruption_attempted_counts,
                applied_counts=corruption_applied_counts,
                skipped_counts=corruption_skipped_counts,
                skipped_attempt_counts=corruption_skipped_attempt_counts,
                skipped_reason_counts=corruption_skipped_reason_counts,
                skipped_attempt_reason_counts=corruption_skipped_attempt_reason_counts,
            )
            LOGGER.info("step=%s metrics=%s", step_index, json.dumps(step_metrics, sort_keys=True))

        if max_batches is not None and step_index >= max_batches:
            break

    return add_corruption_usage_metrics(
        tracker.average(),
        attempted_counts=corruption_attempted_counts,
        applied_counts=corruption_applied_counts,
        skipped_counts=corruption_skipped_counts,
        skipped_attempt_counts=corruption_skipped_attempt_counts,
        skipped_reason_counts=corruption_skipped_reason_counts,
        skipped_attempt_reason_counts=corruption_skipped_attempt_reason_counts,
    )


@torch.no_grad()
def evaluate(
    model: TeacherGNN,
    loader: DataLoader,
    device: torch.device,
    losses_cfg: DictConfig,
    training_cfg: DictConfig,
    stage_cfg: Mapping[str, Any],
    dynamic_loss_weighter: DynamicLossWeighter | None = None,
    max_batches: int | None = None,
):
    return run_epoch(
        model=model,
        loader=loader,
        device=device,
        losses_cfg=losses_cfg,
        training_cfg=training_cfg,
        stage_cfg=stage_cfg,
        optimizer=None,
        scaler=None,
        dynamic_loss_weighter=dynamic_loss_weighter,
        max_batches=max_batches,
    )


def save_checkpoint(
    path: Path,
    model: TeacherGNN,
    optimizer: AdamW,
    epoch: int,
    metrics: Mapping[str, float],
    *,
    stage_name: str,
    stage_epoch: int,
    dynamic_loss_weighter: DynamicLossWeighter | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "stage": stage_name,
        "stage_epoch": stage_epoch,
        "metrics": dict(metrics),
    }
    if dynamic_loss_weighter is not None:
        payload["dynamic_loss_weighter_state_dict"] = dynamic_loss_weighter.state_dict()
    torch.save(payload, path)


def load_model_weights_from_checkpoint(
    checkpoint_path: Path,
    model: TeacherGNN,
    device: torch.device,
    *,
    strict: bool = True,
) -> Mapping[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint must be a mapping: {checkpoint_path}")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no 'model_state_dict': {checkpoint_path}")
    state_dict = checkpoint["model_state_dict"]
    if not strict:
        current_state = model.state_dict()
        skipped_mismatched = []
        filtered_state = {}
        for key, value in state_dict.items():
            current_value = current_state.get(key)
            if current_value is None:
                filtered_state[key] = value
                continue
            if is_lazy(current_value):
                filtered_state[key] = value
                continue
            if tuple(current_value.shape) != tuple(value.shape):
                skipped_mismatched.append((key, tuple(value.shape), tuple(current_value.shape)))
                continue
            filtered_state[key] = value
        if skipped_mismatched:
            LOGGER.warning(
                "Skipped checkpoint keys with shape mismatch=%s",
                [
                    {"key": key, "checkpoint_shape": old_shape, "model_shape": new_shape}
                    for key, old_shape, new_shape in skipped_mismatched
                ],
            )
        state_dict = filtered_state
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        LOGGER.warning(
            "Loaded checkpoint with missing keys=%s unexpected keys=%s",
            list(missing),
            list(unexpected),
        )
    return checkpoint


def load_dynamic_loss_weighter_from_checkpoint(
    checkpoint: Mapping[str, Any],
    dynamic_loss_weighter: DynamicLossWeighter,
    *,
    strict: bool = True,
) -> bool:
    state_dict = checkpoint.get("dynamic_loss_weighter_state_dict")
    if state_dict is None:
        LOGGER.info("Checkpoint has no dynamic loss weighter state; initializing dynamic weights from config.")
        return False
    missing, unexpected = dynamic_loss_weighter.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        LOGGER.warning(
            "Loaded dynamic loss weighter with missing keys=%s unexpected keys=%s",
            list(missing),
            list(unexpected),
        )
    return True


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
        "intra_rank_loss",
        "inter_rank_loss",
        "graph_binary_loss",
        "note_local_loss",
        "chord_local_loss",
        "onset_local_loss",
        "dynamic_weight_recon",
        "dynamic_weight_graph_rank",
        "dynamic_weight_note_local",
        "dynamic_weight_chord_local",
        "dynamic_weight_onset_local",
        "dynamic_log_var_recon",
        "dynamic_log_var_graph_rank",
        "dynamic_log_var_note_local",
        "dynamic_log_var_chord_local",
        "dynamic_log_var_onset_local",
        "dynamic_active_objectives",
        "note_sd_acc",
        "chord_root_acc",
        "chord_type_acc",
        "chord_applied_acc",
        "chord_borrowed_kind_acc",
        "rank_acc",
        "intra_rank_acc",
        "inter_rank_acc",
        "graph_binary_acc",
        "graph_binary_clean_acc",
        "graph_binary_corrupted_acc",
        "note_local_acc",
        "chord_local_acc",
        "onset_local_acc",
        "mean_margin",
        "intra_mean_margin",
        "inter_mean_margin",
        "score_real_mean",
        "score_corrupted_mean",
        "score_real_min",
        "score_corrupted_max",
        "score_gap_minmax",
        "score_base_real_mean",
        "score_base_corrupted_mean",
        "score_base_mean_margin",
        "score_base_rank_acc",
        "score_local_summary_real_mean",
        "score_local_summary_corrupted_mean",
        "score_local_summary_mean_margin",
        "score_local_summary_rank_acc",
    ]
    rendered = [f"{key}={metrics[key]:.4f}" for key in ordered_keys if key in metrics]
    LOGGER.info("%s: %s", prefix, ", ".join(rendered))


def persist_metrics(
    output_dir: Path,
    epoch: int,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    *,
    stage_name: str,
    stage_epoch: int,
):
    metrics_path = output_dir / "metrics.jsonl"
    payload = {
        "epoch": epoch,
        "stage": stage_name,
        "stage_epoch": stage_epoch,
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
    dynamic_loss_weighter = build_teacher_dynamic_loss_weighter(cfg.losses)
    if dynamic_loss_weighter is not None:
        dynamic_loss_weighter = dynamic_loss_weighter.to(device)

    init_checkpoint = cfg.training.get("init_checkpoint")
    init_checkpoint_metadata = None
    if init_checkpoint:
        init_checkpoint_path = resolve_path(str(init_checkpoint))
        init_checkpoint_metadata = load_model_weights_from_checkpoint(
            init_checkpoint_path,
            model,
            device,
            strict=bool(cfg.training.get("init_checkpoint_strict", True)),
        )
        LOGGER.info(
            "Initialized model weights from checkpoint=%s stage=%s stage_epoch=%s epoch=%s",
            init_checkpoint_path,
            init_checkpoint_metadata.get("stage"),
            init_checkpoint_metadata.get("stage_epoch"),
            init_checkpoint_metadata.get("epoch"),
        )
        if dynamic_loss_weighter is not None:
            loaded_dynamic = load_dynamic_loss_weighter_from_checkpoint(
                init_checkpoint_metadata,
                dynamic_loss_weighter,
                strict=bool(cfg.training.get("init_checkpoint_strict", True)),
            )
            if loaded_dynamic:
                LOGGER.info("Initialized dynamic loss weights from checkpoint=%s", init_checkpoint_path)

    epochs = effective_epochs(cfg.training, cfg.experiment)
    stage_plan = build_training_stages(cfg.training, cfg.losses, epochs)
    train_batch_limit = effective_max_batches(cfg.training, cfg.experiment, "train")
    val_batch_limit = effective_max_batches(cfg.training, cfg.experiment, "val")
    checkpoint_dir = output_dir / "checkpoints"

    metadata = {
        "project": OmegaConf.to_container(cfg.project, resolve=True),
        "run_name": cfg.run_name,
        "device": str(device),
        "dataset_json": str(resolve_path(cfg.data.json_path)),
        "dataloader_name": str(cfg.dataloader.get("name", "")),
        "pair_corpus_root": (
            str(resolve_path(str(cfg.dataloader.pair_corpus_root)))
            if str(cfg.dataloader.get("source", "")) == "precomputed_pairs"
            or str(cfg.dataloader.get("corruption_backend", "")) == "precomputed_pairs"
            else None
        ),
        "metadata_dir": str(resolve_path(cfg.data.metadata_dir)),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "training_stages": [
            {
                "name": stage["name"],
                "epochs": stage["epochs"],
                "selection_metric": stage["selection_metric"],
                "selection_mode": stage["selection_mode"],
            }
            for stage in stage_plan
        ],
        "init_checkpoint": str(resolve_path(str(init_checkpoint))) if init_checkpoint else None,
        "init_checkpoint_stage": init_checkpoint_metadata.get("stage") if init_checkpoint_metadata else None,
        "init_checkpoint_stage_epoch": init_checkpoint_metadata.get("stage_epoch") if init_checkpoint_metadata else None,
        "init_checkpoint_epoch": init_checkpoint_metadata.get("epoch") if init_checkpoint_metadata else None,
        "dynamic_loss_weighting": OmegaConf.to_container(cfg.losses.get("dynamic_weighting", {}), resolve=True),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info(
        "Training TeacherGNN on %s with %s train and %s val samples.",
        device.type,
        len(train_loader.dataset),
        len(val_loader.dataset),
    )
    LOGGER.info("Staged training plan: %s", json.dumps(metadata["training_stages"], sort_keys=True))

    global_epoch = 0
    for stage in stage_plan:
        LOGGER.info(
            "Starting stage '%s' for %s epochs (recon=%s, graph_rank=%s, note_local=%s, chord_local=%s, onset_local=%s).",
            stage["name"],
            stage["epochs"],
            stage["enable_recon"],
            stage["enable_graph_rank"],
            stage["enable_note_local"],
            stage["enable_chord_local"],
            stage["enable_onset_local"],
        )
        optimizer = build_optimizer(
            model,
            cfg.optimizer,
            extra_parameters=dynamic_loss_weighter.parameters() if dynamic_loss_weighter is not None else None,
        )
        scheduler = build_stage_scheduler(optimizer, cfg.scheduler, stage["epochs"])
        scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.training.use_amp and device.type == "cuda"))
        stage_best_metric = float("-inf") if stage["selection_mode"] == "max" else float("inf")
        stage_checkpoint_dir = checkpoint_dir / stage["name"]

        for stage_epoch in range(1, stage["epochs"] + 1):
            global_epoch += 1
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                losses_cfg=cfg.losses,
                training_cfg=cfg.training,
                stage_cfg=stage,
                optimizer=optimizer,
                scaler=scaler,
                dynamic_loss_weighter=dynamic_loss_weighter,
                max_batches=train_batch_limit,
            )

            if scheduler is not None:
                scheduler.step()

            if stage_epoch % int(cfg.training.eval_every) == 0:
                val_metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    losses_cfg=cfg.losses,
                    training_cfg=cfg.training,
                    stage_cfg=stage,
                    dynamic_loss_weighter=dynamic_loss_weighter,
                    max_batches=val_batch_limit,
                )
                if stage["run_local_eval"]:
                    local_report, local_examples = evaluate_teacher_local_corruption(
                        model=model,
                        loader=val_loader,
                        device=device,
                        max_batches=val_batch_limit,
                        threshold=0.5,
                    )
                    save_local_diagnostic_reports(
                        output_dir=output_dir,
                        report=local_report,
                        examples=local_examples,
                    )
            else:
                val_metrics = {}

            metric_prefix = f"Epoch {global_epoch:03d} [{stage['name']}:{stage_epoch:03d}]"
            print_metrics(f"{metric_prefix} train", train_metrics)
            print_corruption_usage(f"{metric_prefix} train", train_metrics)
            if val_metrics:
                print_metrics(f"{metric_prefix} val", val_metrics)
                print_corruption_usage(f"{metric_prefix} val", val_metrics)
            persist_metrics(
                output_dir,
                global_epoch,
                train_metrics,
                val_metrics,
                stage_name=stage["name"],
                stage_epoch=stage_epoch,
            )

            if stage_epoch % int(cfg.training.save_every) == 0:
                save_checkpoint(
                    stage_checkpoint_dir / "last.pt",
                    model,
                    optimizer,
                    global_epoch,
                    val_metrics or train_metrics,
                    stage_name=stage["name"],
                    stage_epoch=stage_epoch,
                    dynamic_loss_weighter=dynamic_loss_weighter,
                )
                save_checkpoint(
                    checkpoint_dir / "last.pt",
                    model,
                    optimizer,
                    global_epoch,
                    val_metrics or train_metrics,
                    stage_name=stage["name"],
                    stage_epoch=stage_epoch,
                    dynamic_loss_weighter=dynamic_loss_weighter,
                )

            current_metric = val_metrics.get(stage["selection_metric"])
            if current_metric is not None and metric_improved(float(current_metric), stage_best_metric, stage["selection_mode"]):
                stage_best_metric = float(current_metric)
                save_checkpoint(
                    stage_checkpoint_dir / stage["best_checkpoint_name"],
                    model,
                    optimizer,
                    global_epoch,
                    val_metrics,
                    stage_name=stage["name"],
                    stage_epoch=stage_epoch,
                    dynamic_loss_weighter=dynamic_loss_weighter,
                )
                save_checkpoint(
                    checkpoint_dir / stage["best_checkpoint_name"],
                    model,
                    optimizer,
                    global_epoch,
                    val_metrics,
                    stage_name=stage["name"],
                    stage_epoch=stage_epoch,
                    dynamic_loss_weighter=dynamic_loss_weighter,
                )


if __name__ == "__main__":
    main()
