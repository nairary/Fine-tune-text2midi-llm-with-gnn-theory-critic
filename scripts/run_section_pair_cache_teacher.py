#!/usr/bin/env python3
"""Build section-aware pair corpus, cache TeacherGNN graphs, and optionally train from cache."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/section_pair_cache_v1"))
    parser.add_argument("--dataloader", default="section_cache_balanced")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--skip-pair-build", action="store_true")
    parser.add_argument("--skip-teacher-graph-cache", action="store_true")
    parser.add_argument("--train", action="store_true")

    parser.add_argument("--teacher-graph-cache-dir", default="teacher_graphs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mlm-epochs", type=int, default=0)
    parser.add_argument("--corruption-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="teacher_from_section_pair_cache")
    parser.add_argument("--init-checkpoint", type=Path, default=None)

    parser.add_argument("--section-pairs-per-mode", type=int, default=None)
    parser.add_argument("--local-pairs-per-song", type=int, default=None)
    parser.add_argument("--max-pairs-per-split-per-mode", type=int, default=None)
    return parser.parse_args()


def command_to_text(cmd: list[str | Path]) -> str:
    return shlex.join(str(part) for part in cmd)


def run_command(cmd: list[str | Path], *, dry_run: bool) -> None:
    print(f"\n$ {command_to_text(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], check=True)


def main() -> None:
    args = parse_args()

    if not args.data_json.exists():
        raise FileNotFoundError(f"data JSON not found: {args.data_json}")

    if not args.skip_pair_build:
        pair_cmd: list[str | Path] = [
            args.python,
            "-m",
            "src.observer.run_observer_pipeline",
            "+run_name=section_pair_cache_build",
            f"dataloader={args.dataloader}",
            f"data.json_path={args.data_json}",
            f"observer_pipeline.output_root={args.output_root}",
            f"observer_pipeline.overwrite={str(bool(args.overwrite)).lower()}",
            "observer_pipeline.build_pairs=true",
            "observer_pipeline.build_targets=false",
            "observer_pipeline.build_graph_cache=false",
            "observer_pipeline.train=false",
        ]
        if args.section_pairs_per_mode is not None:
            pair_cmd.append(f"dataloader.section_pairs_per_mode={int(args.section_pairs_per_mode)}")
        if args.local_pairs_per_song is not None:
            pair_cmd.append(f"dataloader.local_pairs_per_song={int(args.local_pairs_per_song)}")
        if args.max_pairs_per_split_per_mode is not None:
            pair_cmd.append(f"dataloader.max_pairs_per_split_per_mode={int(args.max_pairs_per_split_per_mode)}")
        run_command(pair_cmd, dry_run=args.dry_run)

    if not args.skip_teacher_graph_cache:
        cache_cmd: list[str | Path] = [
            args.python,
            "-m",
            "src.dataloader.build_teacher_pair_graph_cache",
            "--pair-corpus-root",
            args.output_root,
            "--graph-cache-dir",
            args.teacher_graph_cache_dir,
        ]
        if args.overwrite:
            cache_cmd.append("--overwrite")
        run_command(cache_cmd, dry_run=args.dry_run)

    if args.train:
        corruption_epochs = args.epochs if args.corruption_epochs is None else args.corruption_epochs
        run_dir = args.run_dir
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = args.output_root / "teacher_training" / f"{timestamp}_{args.run_name}"
        train_cmd: list[str | Path] = [
            args.python,
            "-m",
            "src.training.train_teacher",
            "--config-name",
            "full_data_precomputed_pairs",
            f"hydra.run.dir={run_dir}",
            f"run_name={args.run_name}",
            f"dataloader.pair_corpus_root={args.output_root}",
            f"dataloader.teacher_graph_index_dir={args.teacher_graph_cache_dir}/index",
            f"dataloader.batch_size={int(args.batch_size)}",
            f"training.epochs={int(args.epochs)}",
            f"experiment.epochs={int(args.epochs)}",
            f"scheduler.t_max={max(1, int(args.epochs))}",
            f"training.mlm_ssl_epochs={int(args.mlm_epochs)}",
            f"training.corruption_epochs={int(corruption_epochs)}",
            f"optimizer.lr={float(args.lr)}",
            f"device={args.device}",
            f"training.device={args.device}",
        ]
        if args.init_checkpoint is not None:
            train_cmd.extend(
                [
                    f"training.init_checkpoint={args.init_checkpoint}",
                    "training.init_checkpoint_strict=true",
                ]
            )
        run_command(train_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
