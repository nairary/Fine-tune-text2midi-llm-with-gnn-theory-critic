from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
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


def _collate_pairs(batch_items):
    return {
        "graph_clean": Batch.from_data_list([x["graph_clean"] for x in batch_items]),
        "graph_corrupted": Batch.from_data_list([x["graph_corrupted"] for x in batch_items]),
        "teacher_clean": torch.tensor([x["teacher_score_clean"] for x in batch_items], dtype=torch.float),
        "teacher_corrupted": torch.tensor([x["teacher_score_corrupted"] for x in batch_items], dtype=torch.float),
        "pair_metadata": [x["pair_metadata"] for x in batch_items],
    }


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


def _run_epoch(model, loader, optimizer, device, cfg_losses):
    is_train = optimizer is not None
    model.train(is_train)

    total_examples = 0
    total_reg_loss = 0.0
    total_rank_loss = 0.0
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

    use_batch_rank = bool(cfg_losses.get("use_batch_rank", False)) and float(cfg_losses.lambda_rank) > 0.0
    use_pair_rank = bool(cfg_losses.get("use_pair_rank", True)) and float(cfg_losses.lambda_rank) > 0.0

    for batch in loader:
        g_clean = batch["graph_clean"].to(device)
        g_corr = batch["graph_corrupted"].to(device)
        y_clean = batch["teacher_clean"].to(device)
        y_corr = batch["teacher_corrupted"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            s_clean = model(g_clean).view(-1)
            s_corr = model(g_corr).view(-1)
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

            loss = float(cfg_losses.lambda_reg) * reg + float(cfg_losses.lambda_rank) * rank
            if is_train:
                loss.backward()
                optimizer.step()

        batch_examples = int(y_clean.numel())
        total_examples += batch_examples
        total_reg_loss += float(reg.detach().cpu()) * batch_examples

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

    p = np.asarray(preds_all)
    t = np.asarray(targets_all)
    err = p - t
    reg_loss = (total_reg_loss / total_examples) if total_examples > 0 else float("nan")
    rank_loss = (total_rank_loss / total_valid_rank) if total_valid_rank > 0 else 0.0
    total_loss = float(cfg_losses.lambda_reg) * reg_loss + float(cfg_losses.lambda_rank) * rank_loss
    return {
        "loss": total_loss,
        "reg_loss": reg_loss,
        "rank_loss": rank_loss,
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


def _save_checkpoint(path: Path, model, optimizer, epoch: int, best_val_loss: float, cfg: DictConfig) -> None:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(payload, path)


def train(cfg: DictConfig) -> None:
    _set_seed(int(cfg.observer_training.seed))
    paths = resolve_observer_pipeline_paths(cfg)
    targets_root = paths["targets_root"]
    index_root = paths["cache_index_root"]
    out_dir = paths["training_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    config_path = out_dir / "config.json"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    resume = bool(cfg.observer_training.get("resume", False))
    start_epoch = 1
    best_val_loss = math.inf

    if not resume:
        if metrics_path.exists():
            metrics_path.unlink()
        best_path.unlink(missing_ok=True)
        last_path.unlink(missing_ok=True)
    elif not last_path.exists():
        raise ValueError("observer_training.resume=true but last.pt does not exist")

    train_ds = ObserverPairCachedDataset(index_root / "train.jsonl", targets_root / "train_pairs.jsonl", mode="pair")
    val_ds = ObserverPairCachedDataset(index_root / "val.jsonl", targets_root / "val_pairs.jsonl", mode="pair")
    if len(train_ds) == 0:
        raise ValueError("Train cached dataset is empty")
    if len(val_ds) == 0:
        raise ValueError("Validation cached dataset is empty")

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

    spec_global = json.loads((_base_cwd() / "metadata" / "specs" / "spec_global.json").read_text(encoding="utf-8"))
    model = ObserverGNN(
        cat_vocab_sizes=build_observer_vocab_sizes(build_theory_context(), spec_global),
        num_feature_dims={node_type: len(OBSERVER_NUM_FIELDS[node_type]) for node_type in OBSERVER_NUM_FIELDS},
        edge_types=OBSERVER_EDGE_TYPES,
        hidden_dim=int(cfg.observer_model.hidden_dim),
        num_layers=int(cfg.observer_model.num_layers),
        dropout=float(cfg.observer_model.dropout),
    )
    device = torch.device(str(cfg.observer_training.device))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.optimizer.lr), weight_decay=float(cfg.optimizer.weight_decay))

    if resume:
        if not last_path.exists():
            raise ValueError("observer_training.resume=true but last.pt does not exist")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if start_epoch > int(cfg.observer_training.epochs):
            raise ValueError("resume checkpoint epoch already exceeds configured epochs")
    config_path.write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), ensure_ascii=False, indent=2), encoding="utf-8")

    for epoch in range(start_epoch, int(cfg.observer_training.epochs) + 1):
        train_m = _run_epoch(model, train_loader, optimizer, device, cfg.losses)
        val_m = _run_epoch(model, val_loader, None, device, cfg.losses)

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train": train_m, "val": val_m}, ensure_ascii=False) + "\n")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            _save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, cfg)
        _save_checkpoint(last_path, model, optimizer, epoch, best_val_loss, cfg)

    if not best_path.exists():
        _save_checkpoint(best_path, model, optimizer, int(cfg.observer_training.epochs), best_val_loss, cfg)


@hydra.main(version_base=None, config_path="../../configs", config_name="observer_distill")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    train(cfg)


if __name__ == "__main__":
    main()
