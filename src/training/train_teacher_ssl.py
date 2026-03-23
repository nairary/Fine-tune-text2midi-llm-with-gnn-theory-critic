from __future__ import annotations

import argparse
import sys
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Dict, Iterable, List, Mapping, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.dataloader.hooktheory_dataset import HookTheoryDataset, collate_fn
from src.models.teacher_gnn import TeacherGNN
from src.training.teacher_losses import compute_teacher_ssl_losses


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


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        "graph_real": batch["graph_real"].to(device),
        "graph_masked": batch["graph_masked"].to(device),
        "graph_corrupted": batch["graph_corrupted"].to(device),
        "masked_labels": batch["masked_labels"],
        "graph_score_label": batch["graph_score_label"].to(device),
    }


def build_model(sample_graph, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1) -> TeacherGNN:
    return TeacherGNN.from_hetero_data(
        sample_graph,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def run_epoch(
    model: TeacherGNN,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None = None,
    lambda_recon: float = 1.0,
    lambda_rank: float = 0.5,
    max_batches: int | None = None,
):
    is_train = optimizer is not None
    model.train(is_train)
    tracker = MetricTracker()

    for step_index, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad()

        masked_outputs = model(batch["graph_masked"])
        real_outputs = model(batch["graph_real"])
        corrupted_outputs = model(batch["graph_corrupted"])
        loss_dict, metric_dict = compute_teacher_ssl_losses(
            masked_outputs=masked_outputs,
            real_outputs=real_outputs,
            corrupted_outputs=corrupted_outputs,
            masked_batch=batch["graph_masked"],
            masked_labels=batch["masked_labels"],
            lambda_recon=lambda_recon,
            lambda_rank=lambda_rank,
        )

        if is_train:
            loss_dict["loss"].backward()
            optimizer.step()

        scalar_losses = {key: value.detach() for key, value in loss_dict.items() if isinstance(value, torch.Tensor)}
        tracker.update(scalar_losses, weight=1.0)
        tracker.update(metric_dict, weight=1.0)

        if max_batches is not None and step_index >= max_batches:
            break

    return tracker.average()


@torch.no_grad()
def evaluate(
    model: TeacherGNN,
    loader: DataLoader,
    device: torch.device,
    lambda_recon: float = 1.0,
    lambda_rank: float = 0.5,
    max_batches: int | None = None,
):
    return run_epoch(
        model=model,
        loader=loader,
        device=device,
        optimizer=None,
        lambda_recon=lambda_recon,
        lambda_rank=lambda_rank,
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
        "note_sd_acc",
        "chord_root_acc",
        "chord_type_acc",
        "chord_applied_acc",
        "chord_borrowed_kind_acc",
        "rank_acc",
        "mean_margin",
        "score_real_mean",
        "score_corrupted_mean",
    ]
    rendered = [f"{key}={metrics[key]:.4f}" for key in ordered_keys if key in metrics]
    print(f"{prefix}: " + ", ".join(rendered))


def build_loaders(args):
    dataset = HookTheoryDataset(args.json_path, mask_prob=args.mask_prob)
    train_dataset = SplitFilteredDataset(dataset, split=args.train_split)
    val_dataset = SplitFilteredDataset(dataset, split=args.val_split)

    if getattr(args, "limit_train_samples", None):
        train_dataset.indices = train_dataset.indices[: args.limit_train_samples]
    if getattr(args, "limit_val_samples", None):
        val_dataset.indices = val_dataset.indices[: args.limit_val_samples]

    if len(train_dataset) == 0:
        raise ValueError(f"No samples found for train split '{args.train_split}'.")
    if len(val_dataset) == 0:
        raise ValueError(f"No samples found for val split '{args.val_split}'.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    return dataset, train_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Train the TeacherGNN SSL baseline.")
    parser.add_argument("--json-path", default="data/HTCanon/encoded_full/teacher_encoded.json")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints/teacher_ssl")
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, train_loader, val_loader = build_loaders(args)
    sample = train_loader.dataset[0]
    model = build_model(
        sample["graph_real"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_dir = Path(args.checkpoint_dir)
    best_rank_acc = float("-inf")

    print(f"Training TeacherGNN on {device.type} with {len(train_loader.dataset)} train and {len(val_loader.dataset)} val samples.")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            lambda_recon=args.lambda_recon,
            lambda_rank=args.lambda_rank,
            max_batches=args.max_train_batches,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            lambda_recon=args.lambda_recon,
            lambda_rank=args.lambda_rank,
            max_batches=args.max_val_batches,
        )

        print_metrics(f"Epoch {epoch:03d} train", train_metrics)
        print_metrics(f"Epoch {epoch:03d} val", val_metrics)

        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, val_metrics)
        current_rank_acc = val_metrics.get("rank_acc", float("-inf"))
        if current_rank_acc > best_rank_acc:
            best_rank_acc = current_rank_acc
            save_checkpoint(checkpoint_dir / "best_rank_acc.pt", model, optimizer, epoch, val_metrics)


if __name__ == "__main__":
    main()
