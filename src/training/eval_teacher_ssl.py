from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.training.train_teacher_ssl import build_loaders, build_model, evaluate, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a TeacherGNN SSL checkpoint.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--json-path", default="data/HTCanon/encoded_full/teacher_encoded.json")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=0.5)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, val_loader = build_loaders(args)
    sample = val_loader.dataset[0]

    model = build_model(
        sample["graph_real"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        lambda_recon=args.lambda_recon,
        lambda_rank=args.lambda_rank,
        max_batches=args.max_val_batches,
    )
    print_metrics("Validation", metrics)


if __name__ == "__main__":
    main()
