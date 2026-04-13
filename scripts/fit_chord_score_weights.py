from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader.theory_helpers import build_theory_context
from src.observer.chord_score_fitting import build_training_groups, save_json, train_learnable_chord_score

LOGGER = logging.getLogger("fit_chord_score_weights")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone fitter for chord candidate score weights")
    parser.add_argument("--encoded-json", type=Path, required=True)
    parser.add_argument("--midi-root", type=Path, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--instrument-name", type=str, default="chords")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-train-groups-json", action="store_true")
    parser.add_argument("--save-val-groups-json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    encoded_data = json.loads(args.encoded_json.read_text(encoding="utf-8"))
    theory_ctx = build_theory_context()

    train_groups, train_stats = build_training_groups(
        encoded_data=encoded_data,
        midi_root=args.midi_root,
        split=args.train_split,
        instrument_name=args.instrument_name,
        limit=args.limit_train,
        theory_ctx=theory_ctx,
    )
    val_groups, val_stats = build_training_groups(
        encoded_data=encoded_data,
        midi_root=args.midi_root,
        split=args.val_split,
        instrument_name=args.instrument_name,
        limit=args.limit_val,
        theory_ctx=theory_ctx,
    )

    LOGGER.info("Train groups=%d stats=%s", len(train_groups), train_stats)
    LOGGER.info("Val groups=%d stats=%s", len(val_groups), val_stats)

    if args.save_train_groups_json:
        save_json(args.outdir / "train_groups.json", train_groups)
    if args.save_val_groups_json:
        save_json(args.outdir / "val_groups.json", val_groups)

    model, summary, metrics_log = train_learnable_chord_score(
        train_groups=train_groups,
        val_groups=val_groups,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
    )

    learned_weights = model.export_weights()
    (args.outdir / "learned_weights.yaml").write_text(yaml.safe_dump(learned_weights, sort_keys=False), encoding="utf-8")

    metrics = {
        "train_loss": summary.get("train_loss"),
        "val_loss": summary.get("val_loss"),
        "train_top1_exact_acc": summary.get("train_top1_exact_acc"),
        "val_top1_exact_acc": summary.get("val_top1_exact_acc"),
        "train_topk_contains_gt_acc": summary.get("train_topk_contains_gt_acc"),
        "val_topk_contains_gt_acc": summary.get("val_topk_contains_gt_acc"),
        "train_root_acc": summary.get("train_root_acc"),
        "val_root_acc": summary.get("val_root_acc"),
        "train_type_acc": summary.get("train_type_acc"),
        "val_type_acc": summary.get("val_type_acc"),
        "train_group_count": summary.get("train_group_count", len(train_groups)),
        "val_group_count": summary.get("val_group_count", len(val_groups)),
        "train_positive_coverage": summary.get("train_positive_coverage"),
        "val_positive_coverage": summary.get("val_positive_coverage"),
        "train_build_stats": train_stats,
        "val_build_stats": val_stats,
    }
    save_json(args.outdir / "metrics.json", metrics)

    with (args.outdir / "metrics.jsonl").open("w", encoding="utf-8") as fout:
        for row in metrics_log:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    torch.save(
        {
            "epoch": int(summary.get("epoch", len(metrics_log))),
            "model_state": model.state_dict(),
            # Inference-only artifact: we keep best model weights/metadata but do
            # not store optimizer internals for training resume.
            "optimizer_state": None,
            "best_metric": summary.get("val_top1_exact_acc"),
        },
        args.outdir / "last.pt",
    )


if __name__ == "__main__":
    main()
