from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from src.dataloader.theory_helpers import build_theory_context
from src.observer.cached_dataset import ObserverPairCachedDataset
from src.observer.model import ObserverGNN
from src.observer.pipeline_paths import resolve_observer_pipeline_paths
from src.observer.schema import OBSERVER_EDGE_TYPES, OBSERVER_NUM_FIELDS, build_observer_vocab_sizes

LOGGER = logging.getLogger(__name__)


def _base_cwd() -> Path:
    try:
        return Path(hydra.utils.get_original_cwd())
    except Exception:
        return Path(os.getcwd())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _as_feature_tensor(values: list[Any]) -> torch.Tensor | None:
    if not values or any(value is None for value in values):
        return None
    try:
        tensor = torch.tensor(values, dtype=torch.float)
    except (TypeError, ValueError):
        return None
    if tensor.ndim != 2 or not torch.isfinite(tensor).all():
        return None
    return tensor


def _collate_distillation_side(batch_items, side: str) -> dict[str, Any]:
    key = f"teacher_distill_{side}"
    rows = [item.get(key) or {} for item in batch_items]
    out: dict[str, Any] = {}

    graph_embedding = _as_feature_tensor([row.get("teacher_graph_embedding") for row in rows])
    if graph_embedding is not None:
        out["teacher_graph_embedding"] = graph_embedding

    local_summaries = _as_feature_tensor([row.get("teacher_local_score_summaries") for row in rows])
    if local_summaries is not None:
        out["teacher_local_score_summaries"] = local_summaries

    pooled_rows = [row.get("teacher_pooled_by_type") for row in rows]
    if pooled_rows and all(isinstance(row, dict) for row in pooled_rows):
        common_types = set(str(node_type) for node_type in pooled_rows[0].keys())
        for row in pooled_rows[1:]:
            common_types &= set(str(node_type) for node_type in row.keys())
        pooled_by_type = {}
        for node_type in sorted(common_types):
            tensor = _as_feature_tensor([row.get(node_type) for row in pooled_rows])
            if tensor is not None:
                pooled_by_type[node_type] = tensor
        if pooled_by_type:
            out["teacher_pooled_by_type"] = pooled_by_type

    return out


def _collate_pairs(batch_items):
    batch = {
        "graph_clean": Batch.from_data_list([x["graph_clean"] for x in batch_items]),
        "graph_corrupted": Batch.from_data_list([x["graph_corrupted"] for x in batch_items]),
        "teacher_clean": torch.tensor([x["teacher_score_clean"] for x in batch_items], dtype=torch.float),
        "teacher_corrupted": torch.tensor([x["teacher_score_corrupted"] for x in batch_items], dtype=torch.float),
        "pair_metadata": [x["pair_metadata"] for x in batch_items],
    }
    clean_distill = _collate_distillation_side(batch_items, "clean")
    corrupted_distill = _collate_distillation_side(batch_items, "corrupted")
    if clean_distill:
        batch["teacher_distill_clean"] = clean_distill
    if corrupted_distill:
        batch["teacher_distill_corrupted"] = corrupted_distill
    return batch


def _rank_term(
    pred_margin: torch.Tensor,
    teacher_margin: torch.Tensor,
    min_gap: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    gap = teacher_margin.abs()
    sign = torch.sign(teacher_margin)
    mask = (gap >= float(min_gap)) & (sign != 0)
    if not torch.any(mask):
        return pred_margin.new_tensor(0.0), mask, 0, 0
    logits = pred_margin * sign
    correct = int((logits[mask] > 0).sum().item())
    valid = int(mask.sum().item())
    return -F.logsigmoid(logits[mask]).mean(), mask, correct, valid


def _rank_loss(pred_clean, pred_corr, y_clean, y_corr, min_gap: float) -> tuple[torch.Tensor, torch.Tensor]:
    rank, mask, _, _ = _rank_term(pred_clean - pred_corr, y_clean - y_corr, min_gap)
    return rank, mask


def _loss_weight(cfg_losses, new_key: str, old_key: str | None = None, default: float = 0.0) -> float:
    if new_key in cfg_losses:
        return float(cfg_losses.get(new_key))
    if old_key is not None and old_key in cfg_losses:
        return float(cfg_losses.get(old_key))
    return float(default)


def _intermediate_distillation_enabled(cfg_losses) -> bool:
    return any(
        _loss_weight(cfg_losses, key, default=0.0) > 0.0
        for key in (
            "lambda_graph_embedding_distill",
            "lambda_node_type_embedding_distill",
            "lambda_local_summary_distill",
        )
    )


def _embedding_loss(pred: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    mode = str(mode or "mse").lower()
    if pred.shape != target.shape:
        raise ValueError(f"Distillation shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    mse = F.mse_loss(pred, target)
    if mode == "mse":
        return mse
    cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1e-8).mean()
    if mode == "cosine":
        return cosine
    if mode in {"mse_cosine", "mse+cosine"}:
        return mse + cosine
    raise ValueError(f"Unsupported embedding_distill_loss='{mode}'. Supported: mse, cosine, mse_cosine")


def _to_device_distill_targets(targets: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in targets.items():
        if key == "teacher_pooled_by_type":
            out[key] = {node_type: tensor.to(device) for node_type, tensor in value.items()}
        else:
            out[key] = value.to(device)
    return out


class ObserverDistillationAdapters(nn.Module):
    def __init__(
        self,
        *,
        observer_graph_dim: int,
        observer_node_dim: int,
        target_dims: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        target_dims = target_dims or {}
        graph_dim = target_dims.get("teacher_graph_embedding")
        local_dim = target_dims.get("teacher_local_score_summaries")
        pooled_dims = target_dims.get("teacher_pooled_by_type") or {}

        self.graph_projection = self._make_projection(observer_graph_dim, graph_dim)
        self.local_summary_projection = self._make_projection(observer_graph_dim, local_dim)
        self.node_type_projections = nn.ModuleDict(
            {
                str(node_type): self._make_projection(observer_node_dim, target_dim)
                for node_type, target_dim in sorted(pooled_dims.items())
                if target_dim is not None
            }
        )

    @staticmethod
    def _make_projection(in_dim: int, out_dim: int | None) -> nn.Module | None:
        if out_dim is None:
            return None
        out_dim = int(out_dim)
        in_dim = int(in_dim)
        if out_dim == in_dim:
            return nn.Identity()
        return nn.Linear(in_dim, out_dim)

    def project_graph(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        if self.graph_projection is None:
            raise ValueError("Graph embedding distillation requires cached teacher_graph_embedding targets")
        return self.graph_projection(graph_embedding)

    def project_local_summary(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        if self.local_summary_projection is None:
            raise ValueError("Local-summary distillation requires cached teacher_local_score_summaries targets")
        return self.local_summary_projection(graph_embedding)

    def project_node_type(self, node_type: str, pooled_embedding: torch.Tensor) -> torch.Tensor:
        if node_type not in self.node_type_projections:
            raise ValueError(f"Node-type distillation requires cached teacher_pooled_by_type['{node_type}'] targets")
        return self.node_type_projections[node_type](pooled_embedding)


def _distill_side_loss(
    observer_outputs: dict[str, torch.Tensor],
    teacher_targets: dict[str, Any],
    adapters: ObserverDistillationAdapters,
    cfg_losses,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reference = observer_outputs["score"]
    zero = reference.new_tensor(0.0)
    losses = {
        "graph_embedding_distill_loss": zero,
        "node_type_embedding_distill_loss": zero,
        "local_summary_distill_loss": zero,
    }
    mode = str(cfg_losses.get("embedding_distill_loss", "mse"))

    if _loss_weight(cfg_losses, "lambda_graph_embedding_distill", default=0.0) > 0.0:
        target = teacher_targets.get("teacher_graph_embedding")
        if target is None:
            raise ValueError("lambda_graph_embedding_distill > 0 requires cached teacher_graph_embedding targets")
        pred = adapters.project_graph(observer_outputs["graph_embedding"])
        losses["graph_embedding_distill_loss"] = _embedding_loss(pred, target, mode)

    if _loss_weight(cfg_losses, "lambda_node_type_embedding_distill", default=0.0) > 0.0:
        target_by_type = teacher_targets.get("teacher_pooled_by_type")
        if not target_by_type:
            raise ValueError("lambda_node_type_embedding_distill > 0 requires cached teacher_pooled_by_type targets")
        type_losses = []
        pooled_by_type = observer_outputs.get("pooled_by_type") or {}
        for node_type, target in target_by_type.items():
            if node_type not in pooled_by_type:
                continue
            pred = adapters.project_node_type(str(node_type), pooled_by_type[node_type])
            type_losses.append(_embedding_loss(pred, target, mode))
        if not type_losses:
            raise ValueError("No overlapping node types for node-type embedding distillation")
        losses["node_type_embedding_distill_loss"] = torch.stack(type_losses).mean()

    if _loss_weight(cfg_losses, "lambda_local_summary_distill", default=0.0) > 0.0:
        target = teacher_targets.get("teacher_local_score_summaries")
        if target is None:
            raise ValueError("lambda_local_summary_distill > 0 requires cached teacher_local_score_summaries targets")
        pred = adapters.project_local_summary(observer_outputs["graph_embedding"])
        losses["local_summary_distill_loss"] = _embedding_loss(pred, target, mode)

    return sum(losses.values(), zero), losses


def _distillation_losses(
    clean_outputs: dict[str, torch.Tensor],
    corrupted_outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    adapters: ObserverDistillationAdapters | None,
    cfg_losses,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    reference = clean_outputs["score"]
    zero = reference.new_tensor(0.0)
    losses = {
        "graph_embedding_distill_loss": zero,
        "node_type_embedding_distill_loss": zero,
        "local_summary_distill_loss": zero,
    }
    if adapters is None or not _intermediate_distillation_enabled(cfg_losses):
        return losses

    clean_targets = _to_device_distill_targets(batch.get("teacher_distill_clean") or {}, device)
    corr_targets = _to_device_distill_targets(batch.get("teacher_distill_corrupted") or {}, device)
    _, clean_losses = _distill_side_loss(clean_outputs, clean_targets, adapters, cfg_losses)
    _, corr_losses = _distill_side_loss(corrupted_outputs, corr_targets, adapters, cfg_losses)
    return {
        key: 0.5 * (clean_losses[key] + corr_losses[key])
        for key in losses.keys()
    }


def _batch_rank_loss(
    pred_clean: torch.Tensor,
    pred_corr: torch.Tensor,
    y_clean: torch.Tensor,
    y_corr: torch.Tensor,
    min_gap: float,
    intra_weight: float,
    inter_weight: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    intra_loss, _, intra_correct, intra_valid = _rank_term(pred_clean - pred_corr, y_clean - y_corr, min_gap)

    global_pred_margin = pred_clean[:, None] - pred_corr[None, :]
    global_teacher_margin = y_clean[:, None] - y_corr[None, :]
    if pred_clean.numel() > 1:
        off_diagonal = ~torch.eye(pred_clean.numel(), dtype=torch.bool, device=pred_clean.device)
        inter_pred_margin = global_pred_margin[off_diagonal]
        inter_teacher_margin = global_teacher_margin[off_diagonal]
    else:
        inter_pred_margin = global_pred_margin.new_empty((0,))
        inter_teacher_margin = global_teacher_margin.new_empty((0,))
    inter_loss, _, inter_correct, inter_valid = _rank_term(inter_pred_margin, inter_teacher_margin, min_gap)

    weighted_terms = []
    active_weights = []
    if float(intra_weight) > 0.0 and intra_valid > 0:
        weighted_terms.append(float(intra_weight) * intra_loss)
        active_weights.append(float(intra_weight))
    if float(inter_weight) > 0.0 and inter_valid > 0:
        weighted_terms.append(float(inter_weight) * inter_loss)
        active_weights.append(float(inter_weight))
    if weighted_terms:
        rank_loss = torch.stack(weighted_terms).sum() / float(sum(active_weights))
    else:
        rank_loss = pred_clean.new_tensor(0.0)

    stats = {
        "intra_correct": intra_correct,
        "intra_valid": intra_valid,
        "inter_correct": inter_correct,
        "inter_valid": inter_valid,
        "total_correct": intra_correct + inter_correct,
        "total_valid": intra_valid + inter_valid,
    }
    return rank_loss, stats


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    cfg_losses,
    adapters: ObserverDistillationAdapters | None = None,
    *,
    phase: str = "train",
    epoch: int | None = None,
    total_epochs: int | None = None,
    log_every_batches: int = 100,
):
    is_train = optimizer is not None
    model.train(is_train)
    if adapters is not None:
        adapters.train(is_train)

    total_examples = 0
    total_reg_loss = 0.0
    total_rank_loss = 0.0
    total_graph_embedding_distill_loss = 0.0
    total_node_type_embedding_distill_loss = 0.0
    total_local_summary_distill_loss = 0.0
    total_valid_rank = 0
    total_rank_correct = 0
    total_intra_valid = 0
    total_intra_correct = 0
    total_inter_valid = 0
    total_inter_correct = 0
    preds_all: list[float] = []
    targets_all: list[float] = []
    pred_margins: list[float] = []
    teacher_margins: list[float] = []

    score_weight = _loss_weight(cfg_losses, "lambda_score_distill", "lambda_reg", 1.0)
    rank_weight = _loss_weight(cfg_losses, "lambda_margin_distill", "lambda_rank", 0.0)
    graph_embedding_weight = _loss_weight(cfg_losses, "lambda_graph_embedding_distill", default=0.0)
    node_type_embedding_weight = _loss_weight(cfg_losses, "lambda_node_type_embedding_distill", default=0.0)
    local_summary_weight = _loss_weight(cfg_losses, "lambda_local_summary_distill", default=0.0)
    need_intermediates = _intermediate_distillation_enabled(cfg_losses)

    use_batch_rank = bool(cfg_losses.get("use_batch_rank", False)) and rank_weight > 0.0
    use_pair_rank = bool(cfg_losses.get("use_pair_rank", True)) and rank_weight > 0.0

    num_batches = len(loader)
    epoch_label = f" epoch={epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else ""
    LOGGER.info("%s%s start batches=%d", phase, epoch_label, num_batches)
    epoch_started = time.perf_counter()
    log_every_batches = max(0, int(log_every_batches))

    for batch_idx, batch in enumerate(loader, start=1):
        g_clean = batch["graph_clean"].to(device)
        g_corr = batch["graph_corrupted"].to(device)
        y_clean = batch["teacher_clean"].to(device)
        y_corr = batch["teacher_corrupted"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if need_intermediates:
                clean_outputs_raw = model(g_clean, return_outputs=True)
                corr_outputs_raw = model(g_corr, return_outputs=True)
            else:
                clean_outputs_raw = model(g_clean)
                corr_outputs_raw = model(g_corr)
            if isinstance(clean_outputs_raw, dict):
                clean_outputs = clean_outputs_raw
                corr_outputs = corr_outputs_raw
                s_clean = clean_outputs["score"].view(-1)
                s_corr = corr_outputs["score"].view(-1)
            else:
                s_clean = clean_outputs_raw.view(-1)
                s_corr = corr_outputs_raw.view(-1)
                clean_outputs = {"score": s_clean}
                corr_outputs = {"score": s_corr}
            reg = F.smooth_l1_loss(s_clean, y_clean) + F.smooth_l1_loss(s_corr, y_corr)

            if use_batch_rank:
                rank, rank_stats = _batch_rank_loss(
                    s_clean,
                    s_corr,
                    y_clean,
                    y_corr,
                    min_gap=float(cfg_losses.min_teacher_gap_for_rank),
                    intra_weight=float(cfg_losses.get("rank_intra_weight", 1.0)),
                    inter_weight=float(cfg_losses.get("rank_inter_weight", 1.0)),
                )
                rank_mask = torch.ones_like(y_clean, dtype=torch.bool) if rank_stats["total_valid"] > 0 else torch.zeros_like(y_clean, dtype=torch.bool)
            elif use_pair_rank:
                rank, rank_mask = _rank_loss(s_clean, s_corr, y_clean, y_corr, float(cfg_losses.min_teacher_gap_for_rank))
                rank_stats = {}
            else:
                rank = reg.new_tensor(0.0)
                rank_mask = torch.zeros_like(y_clean, dtype=torch.bool)
                rank_stats = {}

            distill_losses = _distillation_losses(
                clean_outputs,
                corr_outputs,
                batch,
                adapters,
                cfg_losses,
                device,
            )
            loss = (
                score_weight * reg
                + rank_weight * rank
                + graph_embedding_weight * distill_losses["graph_embedding_distill_loss"]
                + node_type_embedding_weight * distill_losses["node_type_embedding_distill_loss"]
                + local_summary_weight * distill_losses["local_summary_distill_loss"]
            )
            if is_train:
                loss.backward()
                optimizer.step()

        batch_examples = int(y_clean.numel())
        total_examples += batch_examples
        total_reg_loss += float(reg.detach().cpu()) * batch_examples
        total_graph_embedding_distill_loss += float(distill_losses["graph_embedding_distill_loss"].detach().cpu()) * batch_examples
        total_node_type_embedding_distill_loss += float(distill_losses["node_type_embedding_distill_loss"].detach().cpu()) * batch_examples
        total_local_summary_distill_loss += float(distill_losses["local_summary_distill_loss"].detach().cpu()) * batch_examples

        pred = torch.cat([s_clean.detach().cpu(), s_corr.detach().cpu()]).numpy()
        tgt = torch.cat([y_clean.detach().cpu(), y_corr.detach().cpu()]).numpy()
        preds_all.extend(pred.tolist())
        targets_all.extend(tgt.tolist())

        if use_batch_rank:
            valid = int(rank_stats.get("total_valid", 0))
            if valid > 0:
                total_rank_correct += int(rank_stats["total_correct"])
                total_valid_rank += valid
                total_rank_loss += float(rank.detach().cpu()) * valid
                total_intra_correct += int(rank_stats["intra_correct"])
                total_intra_valid += int(rank_stats["intra_valid"])
                total_inter_correct += int(rank_stats["inter_correct"])
                total_inter_valid += int(rank_stats["inter_valid"])
        elif torch.any(rank_mask):
            sign = torch.sign((y_clean - y_corr)[rank_mask])
            correct = int((((s_clean - s_corr)[rank_mask] * sign) > 0).sum().item())
            valid = int(rank_mask.sum().item())
            total_rank_correct += correct
            total_valid_rank += valid
            total_rank_loss += float(rank.detach().cpu()) * valid
            total_intra_correct += correct
            total_intra_valid += valid
        pred_margins.extend((s_clean - s_corr).detach().cpu().tolist())
        teacher_margins.extend((y_clean - y_corr).detach().cpu().tolist())

        if log_every_batches > 0 and (
            batch_idx == 1 or batch_idx == num_batches or batch_idx % log_every_batches == 0
        ):
            elapsed = time.perf_counter() - epoch_started
            batches_per_sec = float(batch_idx) / elapsed if elapsed > 0.0 else 0.0
            LOGGER.info(
                "%s%s batch=%d/%d examples=%d reg_loss=%.4f rank_valid=%d speed=%.2f batch/s",
                phase,
                epoch_label,
                batch_idx,
                num_batches,
                total_examples,
                (total_reg_loss / total_examples) if total_examples > 0 else float("nan"),
                total_valid_rank,
                batches_per_sec,
            )

    p = np.asarray(preds_all)
    t = np.asarray(targets_all)
    err = p - t
    reg_loss = (total_reg_loss / total_examples) if total_examples > 0 else float("nan")
    graph_embedding_distill_loss = (total_graph_embedding_distill_loss / total_examples) if total_examples > 0 else 0.0
    node_type_embedding_distill_loss = (total_node_type_embedding_distill_loss / total_examples) if total_examples > 0 else 0.0
    local_summary_distill_loss = (total_local_summary_distill_loss / total_examples) if total_examples > 0 else 0.0
    rank_loss = (total_rank_loss / total_valid_rank) if total_valid_rank > 0 else 0.0
    total_loss = (
        score_weight * reg_loss
        + rank_weight * rank_loss
        + graph_embedding_weight * graph_embedding_distill_loss
        + node_type_embedding_weight * node_type_embedding_distill_loss
        + local_summary_weight * local_summary_distill_loss
    )
    metrics = {
        "loss": total_loss,
        "reg_loss": reg_loss,
        "score_distill_loss": reg_loss,
        "rank_loss": rank_loss,
        "margin_distill_loss": rank_loss,
        "graph_embedding_distill_loss": graph_embedding_distill_loss,
        "node_type_embedding_distill_loss": node_type_embedding_distill_loss,
        "local_summary_distill_loss": local_summary_distill_loss,
        "mae": float(np.mean(np.abs(err))) if err.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(err**2))) if err.size else float("nan"),
        "pearson": _safe_corr(p, t),
        "spearman": _safe_corr(_rankdata(p), _rankdata(t)) if p.size else float("nan"),
        "pair_rank_acc": (float(total_intra_correct) / float(total_intra_valid)) if total_intra_valid > 0 else float("nan"),
        "intra_rank_acc": (float(total_intra_correct) / float(total_intra_valid)) if total_intra_valid > 0 else float("nan"),
        "inter_rank_acc": (float(total_inter_correct) / float(total_inter_valid)) if total_inter_valid > 0 else float("nan"),
        "batch_rank_acc": (float(total_rank_correct) / float(total_valid_rank)) if total_valid_rank > 0 else float("nan"),
        "mean_pred_margin": float(np.mean(pred_margins)) if pred_margins else float("nan"),
        "mean_teacher_margin": float(np.mean(teacher_margins)) if teacher_margins else float("nan"),
    }
    LOGGER.info(
        "%s%s done loss=%.4f mae=%.4f pair_rank_acc=%.4f batch_rank_acc=%.4f elapsed=%.1fs",
        phase,
        epoch_label,
        float(metrics["loss"]),
        float(metrics["mae"]),
        float(metrics["pair_rank_acc"]),
        float(metrics["batch_rank_acc"]),
        time.perf_counter() - epoch_started,
    )
    return metrics


def _merge_distillation_target_dims(*datasets) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "teacher_graph_embedding": None,
        "teacher_local_score_summaries": None,
        "teacher_pooled_by_type": {},
    }
    pooled: dict[str, int] = {}
    for dataset in datasets:
        dims = getattr(dataset, "distillation_target_dims", {}) or {}
        if merged["teacher_graph_embedding"] is None:
            merged["teacher_graph_embedding"] = dims.get("teacher_graph_embedding")
        if merged["teacher_local_score_summaries"] is None:
            merged["teacher_local_score_summaries"] = dims.get("teacher_local_score_summaries")
        for node_type, dim in (dims.get("teacher_pooled_by_type") or {}).items():
            pooled.setdefault(str(node_type), int(dim))
    merged["teacher_pooled_by_type"] = pooled
    return merged


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    best_val_loss: float,
    cfg: DictConfig,
    adapters: ObserverDistillationAdapters | None = None,
    best_val_rank: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if best_val_rank is not None:
        payload["best_val_rank"] = float(best_val_rank)
    if adapters is not None:
        payload["distillation_adapters_state_dict"] = adapters.state_dict()
    torch.save(payload, path)


def train(cfg: DictConfig) -> None:
    _set_seed(int(cfg.observer_training.seed))
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError as exc:
        LOGGER.warning("Could not set torch multiprocessing sharing strategy: %s", exc)

    paths = resolve_observer_pipeline_paths(cfg)
    targets_root = paths["targets_root"]
    index_root = paths["cache_index_root"]
    out_dir = paths["training_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    config_path = out_dir / "config.json"
    best_path = out_dir / "best.pt"
    best_rank_path = out_dir / "best_rank.pt"
    last_path = out_dir / "last.pt"
    resume = bool(cfg.observer_training.get("resume", False))
    start_epoch = 1
    best_val_loss = math.inf
    best_val_rank = -math.inf

    if not resume:
        if metrics_path.exists():
            metrics_path.unlink()
        best_path.unlink(missing_ok=True)
        best_rank_path.unlink(missing_ok=True)
        last_path.unlink(missing_ok=True)
    elif not last_path.exists():
        raise ValueError("observer_training.resume=true but last.pt does not exist")

    train_ds = ObserverPairCachedDataset(index_root / "train.jsonl", targets_root / "train_pairs.jsonl", mode="pair")
    val_ds = ObserverPairCachedDataset(index_root / "val.jsonl", targets_root / "val_pairs.jsonl", mode="pair")
    if len(train_ds) == 0:
        raise ValueError("Train cached dataset is empty")
    if len(val_ds) == 0:
        raise ValueError("Validation cached dataset is empty")
    LOGGER.info(
        "Observer cached datasets train_pairs=%d val_pairs=%d train_graphs=%d val_graphs=%d",
        len(train_ds),
        len(val_ds),
        len(train_ds.graph_rows),
        len(val_ds.graph_rows),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=bool(cfg.dataloader.get("shuffle", True)),
        num_workers=int(cfg.dataloader.num_workers),
        pin_memory=bool(cfg.dataloader.get("pin_memory", False)),
        drop_last=bool(cfg.dataloader.get("drop_last", False)),
        collate_fn=_collate_pairs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=False,
        num_workers=int(cfg.dataloader.num_workers),
        pin_memory=bool(cfg.dataloader.get("pin_memory", False)),
        drop_last=False,
        collate_fn=_collate_pairs,
    )
    LOGGER.info(
        "Observer dataloaders train_batches=%d val_batches=%d batch_size=%d num_workers=%d pin_memory=%s",
        len(train_loader),
        len(val_loader),
        int(cfg.dataloader.batch_size),
        int(cfg.dataloader.num_workers),
        bool(cfg.dataloader.get("pin_memory", False)),
    )

    spec_global = json.loads((_base_cwd() / "metadata" / "specs" / "spec_global.json").read_text(encoding="utf-8"))
    model = ObserverGNN(
        cat_vocab_sizes=build_observer_vocab_sizes(build_theory_context(), spec_global),
        num_feature_dims={node_type: len(OBSERVER_NUM_FIELDS[node_type]) for node_type in OBSERVER_NUM_FIELDS},
        edge_types=OBSERVER_EDGE_TYPES,
        hidden_dim=int(cfg.observer_model.hidden_dim),
        num_layers=int(cfg.observer_model.num_layers),
        dropout=float(cfg.observer_model.dropout),
        pooling_mode=str(cfg.observer_model.get("pooling_mode", "mean")),
        pooling_output_dim=cfg.observer_model.get("pooling_output_dim", None),
        score_head_hidden_dim=cfg.observer_model.get("score_head_hidden_dim", None),
        use_bar_sequence_transformer=bool(cfg.observer_model.get("use_bar_sequence_transformer", False)),
        bar_transformer_num_layers=int(cfg.observer_model.get("bar_transformer_num_layers", 2)),
        bar_transformer_num_heads=int(cfg.observer_model.get("bar_transformer_num_heads", 4)),
        bar_transformer_ff_dim=cfg.observer_model.get("bar_transformer_ff_dim", None),
        bar_transformer_dropout=cfg.observer_model.get("bar_transformer_dropout", None),
        bar_transformer_pooling=str(cfg.observer_model.get("bar_transformer_pooling", "cls")),
        bar_transformer_combine=str(cfg.observer_model.get("bar_transformer_combine", "concat")),
        score_head_activation=str(cfg.observer_model.get("score_head_activation", "relu")),
        score_head_layer_norm=bool(cfg.observer_model.get("score_head_layer_norm", False)),
    )
    target_dims = _merge_distillation_target_dims(train_ds, val_ds)
    adapters = ObserverDistillationAdapters(
        observer_graph_dim=int(getattr(model, "pooling_output_dim", cfg.observer_model.hidden_dim)),
        observer_node_dim=int(getattr(model, "pool_per_type_dim", getattr(model, "hidden_dim", cfg.observer_model.hidden_dim))),
        target_dims=target_dims,
    )
    device = torch.device(str(cfg.observer_training.device))
    model = model.to(device)
    adapters = adapters.to(device)
    optimizer_params = list(model.parameters()) + list(adapters.parameters())
    optimizer = torch.optim.AdamW(optimizer_params, lr=float(cfg.optimizer.lr), weight_decay=float(cfg.optimizer.weight_decay))

    if resume:
        if not last_path.exists():
            raise ValueError("observer_training.resume=true but last.pt does not exist")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "distillation_adapters_state_dict" in checkpoint:
            adapters.load_state_dict(checkpoint["distillation_adapters_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_val_rank = float(checkpoint.get("best_val_rank", best_val_rank))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if start_epoch > int(cfg.observer_training.epochs):
            raise ValueError("resume checkpoint epoch already exceeds configured epochs")
    config_path.write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), ensure_ascii=False, indent=2), encoding="utf-8")

    total_epochs = int(cfg.observer_training.epochs)
    log_every_batches = int(cfg.observer_training.get("log_every_batches", 100))
    LOGGER.info(
        "Observer training start epochs=%d start_epoch=%d device=%s log_every_batches=%d",
        total_epochs,
        start_epoch,
        device,
        log_every_batches,
    )

    for epoch in range(start_epoch, total_epochs + 1):
        train_m = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg.losses,
            adapters=adapters,
            phase="train",
            epoch=epoch,
            total_epochs=total_epochs,
            log_every_batches=log_every_batches,
        )
        val_m = _run_epoch(
            model,
            val_loader,
            None,
            device,
            cfg.losses,
            adapters=adapters,
            phase="val",
            epoch=epoch,
            total_epochs=total_epochs,
            log_every_batches=log_every_batches,
        )

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train": train_m, "val": val_m}, ensure_ascii=False) + "\n")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            _save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, cfg, adapters=adapters, best_val_rank=best_val_rank)
        val_rank = float(val_m.get("batch_rank_acc", float("nan")))
        if math.isfinite(val_rank) and val_rank > best_val_rank:
            best_val_rank = val_rank
            _save_checkpoint(best_rank_path, model, optimizer, epoch, best_val_loss, cfg, adapters=adapters, best_val_rank=best_val_rank)
        _save_checkpoint(last_path, model, optimizer, epoch, best_val_loss, cfg, adapters=adapters, best_val_rank=best_val_rank)

    if not best_path.exists():
        _save_checkpoint(best_path, model, optimizer, int(cfg.observer_training.epochs), best_val_loss, cfg, adapters=adapters, best_val_rank=best_val_rank)


@hydra.main(version_base=None, config_path="../../configs", config_name="observer_distill")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    train(cfg)


if __name__ == "__main__":
    main()
