from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from src.dataloader.theory_helpers import build_theory_context
from src.observer.dataset import ObserverDataset
from src.observer.model import ObserverGNN
from src.observer.schema import OBSERVER_EDGE_TYPES, OBSERVER_NUM_FIELDS, build_observer_vocab_sizes


@dataclass
class EpochMetrics:
    loss: float
    mae: float
    rmse: float
    pearson: float
    spearman: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ObserverGNN on offline teacher scalar targets.")
    parser.add_argument("--train-input-jsonl", type=Path, required=True)
    parser.add_argument("--train-target-jsonl", type=Path, required=True)
    parser.add_argument("--val-input-jsonl", type=Path, required=True)
    parser.add_argument("--val-target-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling-mode", choices=["mean", "mean_max", "attention"], default="mean")
    parser.add_argument("--pooling-output-dim", type=int, default=None)
    parser.add_argument("--score-head-hidden-dim", type=int, default=None)
    parser.add_argument("--score-head-activation", choices=["relu", "leaky_relu", "gelu", "silu"], default="relu")
    parser.add_argument("--score-head-layer-norm", action="store_true")
    parser.add_argument("--use-bar-sequence-transformer", action="store_true")
    parser.add_argument("--bar-transformer-num-layers", type=int, default=2)
    parser.add_argument("--bar-transformer-num-heads", type=int, default=4)
    parser.add_argument("--bar-transformer-ff-dim", type=int, default=None)
    parser.add_argument("--bar-transformer-dropout", type=float, default=None)
    parser.add_argument("--bar-transformer-pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--bar-transformer-combine", choices=["concat", "replace"], default="concat")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=["mse", "smooth_l1"], default="smooth_l1")
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--chord-weights-yaml", type=str, default=None)
    parser.add_argument("--chord-instrument-name", type=str, default="chords")
    parser.add_argument("--use-fallback-44", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_loss(loss_name: str) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0.0 or y_std <= 0.0:
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
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def compute_metrics(preds: list[float], targets: list[float], mean_loss: float) -> EpochMetrics:
    pred_np = np.asarray(preds, dtype=float)
    target_np = np.asarray(targets, dtype=float)
    errors = pred_np - target_np
    mae = float(np.mean(np.abs(errors))) if errors.size else float("nan")
    rmse = float(np.sqrt(np.mean(errors**2))) if errors.size else float("nan")
    pearson = _safe_corr(pred_np, target_np)
    spearman = _safe_corr(_rankdata(pred_np), _rankdata(target_np)) if errors.size else float("nan")
    return EpochMetrics(loss=float(mean_loss), mae=mae, rmse=rmse, pearson=pearson, spearman=spearman)


def run_epoch(
    model: ObserverGNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> EpochMetrics:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    losses: list[float] = []
    all_preds: list[float] = []
    all_targets: list[float] = []

    for batch in loader:
        batch = batch.to(device)
        targets = batch.y.view(-1).float()

        with (torch.enable_grad() if optimizer is not None else torch.no_grad()):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            preds = model(batch).view(-1)
            loss = criterion(preds, targets)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return compute_metrics(all_preds, all_targets, mean_loss)


def save_checkpoint(
    path: Path,
    model: ObserverGNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataset_kwargs = {
        "chord_weights_yaml": args.chord_weights_yaml,
        "chord_instrument_name": args.chord_instrument_name,
        "use_fallback_44": args.use_fallback_44,
        "in_memory": args.in_memory,
    }
    train_dataset = ObserverDataset(args.train_input_jsonl, args.train_target_jsonl, **dataset_kwargs)
    val_dataset = ObserverDataset(args.val_input_jsonl, args.val_target_jsonl, **dataset_kwargs)
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    spec_global = json.loads((Path(__file__).resolve().parents[2] / "metadata" / "specs" / "spec_global.json").read_text(encoding="utf-8"))
    model = ObserverGNN(
        cat_vocab_sizes=build_observer_vocab_sizes(build_theory_context(), spec_global),
        num_feature_dims={node_type: len(OBSERVER_NUM_FIELDS[node_type]) for node_type in OBSERVER_NUM_FIELDS},
        edge_types=OBSERVER_EDGE_TYPES,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling_mode=args.pooling_mode,
        pooling_output_dim=args.pooling_output_dim,
        score_head_hidden_dim=args.score_head_hidden_dim,
        use_bar_sequence_transformer=args.use_bar_sequence_transformer,
        bar_transformer_num_layers=args.bar_transformer_num_layers,
        bar_transformer_num_heads=args.bar_transformer_num_heads,
        bar_transformer_ff_dim=args.bar_transformer_ff_dim,
        bar_transformer_dropout=args.bar_transformer_dropout,
        bar_transformer_pooling=args.bar_transformer_pooling,
        bar_transformer_combine=args.bar_transformer_combine,
        score_head_activation=args.score_head_activation,
        score_head_layer_norm=args.score_head_layer_norm,
    ).to(device)

    criterion = create_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["device"] = str(device)
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_val_loss = math.inf
    metrics_path = output_dir / "metrics.jsonl"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device=device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device=device, optimizer=None)

        metrics_row = {
            "epoch": epoch,
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics_row, ensure_ascii=False) + "\n")

        is_best = val_metrics.loss < best_val_loss
        if is_best:
            best_val_loss = val_metrics.loss
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_val_loss, config)
        if is_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_val_loss, config)


if __name__ == "__main__":
    main()
