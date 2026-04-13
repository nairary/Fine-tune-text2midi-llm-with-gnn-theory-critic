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
from src.observer.chord_score_fitting import (
    build_training_groups,
    iter_training_group_chunks,
    save_json,
    train_learnable_chord_score,
)

LOGGER = logging.getLogger("fit_chord_score_weights")
BUILD_STATS_KEYS = [
    "songs_total",
    "songs_missing_midi",
    "songs_bad_midi",
    "songs_no_instrument",
    "events_total",
    "events_skipped_rest",
    "events_skipped_bad_gt",
    "events_skipped_sparse_sonority",
    "events_skipped_no_candidates",
    "events_positive_missing",
    "groups_kept",
]


def empty_build_stats() -> dict[str, int]:
    return {key: 0 for key in BUILD_STATS_KEYS}


def merge_build_stats(total: dict[str, int], chunk_stats: dict[str, int]) -> dict[str, int]:
    for key in BUILD_STATS_KEYS:
        total[key] = int(total.get(key, 0)) + int(chunk_stats.get(key, 0))
    return total


def collect_chunked_build_stats(chunk_iterable) -> dict[str, int]:
    total = empty_build_stats()
    for _, chunk_stats in chunk_iterable:
        merge_build_stats(total, chunk_stats)
    return total


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
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-train-groups-json", action="store_true")
    parser.add_argument("--save-val-groups-json", action="store_true")
    parser.add_argument("--materialize-val", action="store_true")
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

    train_groups: list[dict] = []
    train_stats: dict[str, int] = empty_build_stats()
    train_group_chunks = None
    val_groups: list[dict] = []
    val_stats: dict[str, int] = empty_build_stats()
    val_group_chunks = None

    if args.save_train_groups_json:
        train_groups, train_stats = build_training_groups(
            encoded_data=encoded_data,
            midi_root=args.midi_root,
            split=args.train_split,
            instrument_name=args.instrument_name,
            limit=args.limit_train,
            theory_ctx=theory_ctx,
        )
    else:
        train_group_chunks = lambda: iter_training_group_chunks(
            encoded_data=encoded_data,
            midi_root=args.midi_root,
            split=args.train_split,
            instrument_name=args.instrument_name,
            limit=args.limit_train,
            chunk_size=args.chunk_size,
            theory_ctx=theory_ctx,
            include_candidate_metadata=False,
            drop_groups_without_positives=True,
        )
        train_stats = collect_chunked_build_stats(train_group_chunks())

    materialize_val = bool(args.materialize_val or args.save_val_groups_json)
    if materialize_val:
        val_groups, val_stats = build_training_groups(
            encoded_data=encoded_data,
            midi_root=args.midi_root,
            split=args.val_split,
            instrument_name=args.instrument_name,
            limit=args.limit_val,
            theory_ctx=theory_ctx,
        )
    else:
        val_group_chunks = lambda: iter_training_group_chunks(
            encoded_data=encoded_data,
            midi_root=args.midi_root,
            split=args.val_split,
            instrument_name=args.instrument_name,
            limit=args.limit_val,
            chunk_size=args.chunk_size,
            theory_ctx=theory_ctx,
            include_candidate_metadata=True,
            drop_groups_without_positives=False,
        )
        val_stats = collect_chunked_build_stats(val_group_chunks())

    train_mode = "materialized" if args.save_train_groups_json else "chunked"
    val_mode = "materialized" if materialize_val else "chunked"
    LOGGER.info(
        "train_mode=%s val_mode=%s chunk_size=%d eval_every=%d",
        train_mode,
        val_mode,
        int(args.chunk_size),
        int(args.eval_every),
    )
    LOGGER.info("Train groups=%d stats=%s", len(train_groups), train_stats)
    if materialize_val:
        LOGGER.info("Val groups=%d stats=%s", len(val_groups), val_stats)
    else:
        LOGGER.info("Val groups materialized=0 (chunked mode) stats=%s", val_stats)

    if args.save_train_groups_json:
        save_json(args.outdir / "train_groups.json", train_groups)
    if args.save_val_groups_json:
        save_json(args.outdir / "val_groups.json", val_groups)

    model, summary, metrics_log = train_learnable_chord_score(
        train_groups=train_groups if args.save_train_groups_json else None,
        val_groups=val_groups if materialize_val else None,
        train_group_chunks=train_group_chunks,
        val_group_chunks=val_group_chunks,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        eval_every=args.eval_every,
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
        "chunk_size": args.chunk_size,
        "eval_every": args.eval_every,
        "materialize_val": bool(args.materialize_val),
        "val_mode": val_mode,
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
