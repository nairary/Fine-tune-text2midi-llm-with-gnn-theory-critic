#!/usr/bin/env python3
"""Run a small TeacherGNN feature ablation sweep.

The sweep trains independent runs on the same section-aware dataset:

1. baseline: current section-aware SAGE setup with attention pooling;
2. hgt: baseline + HGT backbone;
3. dynamic_weights: baseline + uncertainty-based dynamic loss weights;
4. logit_fusion: baseline + learned graph/local logit fusion.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASELINE_OVERRIDES = [
    "model.pooling_mode=attention",
]

VARIANT_OVERRIDES: dict[str, list[str]] = {
    "baseline": [],
    "hgt": [
        "model.backbone=hgt",
        "model.hgt_num_heads=4",
    ],
    "dynamic_weights": [
        "losses.dynamic_weighting.enabled=true",
    ],
    "logit_fusion": [
        "model.score_fusion_mode=learned_logit_fusion",
        "model.score_fusion_hidden_dim=64",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python executable for child training commands.")
    parser.add_argument("--data-json", type=Path, required=True, help="Section-aware encoded JSON used for all runs.")
    parser.add_argument("--run-root", type=Path, default=None, help="Output root for all ablation runs.")
    parser.add_argument("--run-prefix", default="teacher_struct_ablation", help="Prefix for Hydra run names.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Training device.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mlm-epochs", type=int, default=5)
    parser.add_argument("--corruption-epochs", type=int, default=20)
    parser.add_argument(
        "--baseline-mlm-checkpoint",
        type=Path,
        default=None,
        help=(
            "Reuse an existing baseline MLM checkpoint and run only the corruption stage for baseline. "
            "Can point either to a .pt file or to a checkpoints/mlm_ssl directory."
        ),
    )
    parser.add_argument(
        "--baseline-mlm-checkpoint-strict",
        action="store_true",
        help=(
            "Load the baseline MLM checkpoint strictly. Default is non-strict so old mean/mean_max MLM checkpoints "
            "can initialize the attention-pooling baseline backbone/reconstruction weights."
        ),
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running later variants if one fails.")
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANT_OVERRIDES),
        help="Variant to run. Can be repeated. Defaults to all variants.",
    )
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Extra Hydra override appended to every run. Can be repeated.",
    )
    return parser.parse_args()


def command_to_text(cmd: list[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(f"\n$ {command_to_text(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], check=True)


def resolve_checkpoint_path(path: Path, *, dry_run: bool) -> Path:
    if dry_run and not path.exists():
        return path / "best_recon_loss.pt" if path.suffix != ".pt" else path
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("best_recon_loss.pt", "last.pt"):
            candidate = path / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"checkpoint not found: {path}")


def variant_epoch_plan(args: argparse.Namespace, variant: str) -> tuple[int, int, int, Path | None, bool]:
    if variant == "baseline" and args.baseline_mlm_checkpoint is not None:
        checkpoint = resolve_checkpoint_path(args.baseline_mlm_checkpoint, dry_run=args.dry_run)
        return (
            int(args.corruption_epochs),
            0,
            int(args.corruption_epochs),
            checkpoint,
            bool(args.baseline_mlm_checkpoint_strict),
        )
    total_epochs = int(args.mlm_epochs) + int(args.corruption_epochs)
    return total_epochs, int(args.mlm_epochs), int(args.corruption_epochs), None, True


def build_command(
    args: argparse.Namespace,
    *,
    variant: str,
    run_dir: Path,
    total_epochs: int,
    mlm_epochs: int,
    corruption_epochs: int,
    init_checkpoint: Path | None,
    init_checkpoint_strict: bool,
) -> list[str]:
    run_name = f"{args.run_prefix}_{variant}_attnpool_mlm{mlm_epochs}_corr{corruption_epochs}"
    cmd = [
        args.python,
        "-m",
        "src.training.train_teacher",
        f"hydra.run.dir={run_dir}",
        f"run_name={run_name}",
        f"data.json_path={args.data_json}",
        f"dataloader.batch_size={args.batch_size}",
        f"dataloader.num_workers={args.num_workers}",
        f"dataloader.pin_memory={str(bool(args.pin_memory)).lower()}",
        f"training.epochs={total_epochs}",
        f"experiment.epochs={total_epochs}",
        f"scheduler.t_max={max(1, total_epochs)}",
        f"training.mlm_ssl_epochs={mlm_epochs}",
        f"training.corruption_epochs={corruption_epochs}",
        f"training.log_every={args.log_every}",
        f"training.eval_every={args.eval_every}",
        f"training.save_every={args.save_every}",
        f"training.use_amp={str(bool(args.use_amp)).lower()}",
        f"training.seed={args.seed}",
        f"seed={args.seed}",
        f"optimizer.lr={args.lr}",
        f"device={args.device}",
        f"training.device={args.device}",
    ]
    if init_checkpoint is not None:
        cmd.extend(
            [
                f"training.init_checkpoint={init_checkpoint}",
                f"training.init_checkpoint_strict={str(bool(init_checkpoint_strict)).lower()}",
            ]
        )
    if args.limit_train_samples is not None:
        cmd.append(f"training.limit_train_samples={int(args.limit_train_samples)}")
    if args.limit_val_samples is not None:
        cmd.append(f"training.limit_val_samples={int(args.limit_val_samples)}")
    if args.limit_train_batches is not None:
        cmd.append(f"training.limit_train_batches={int(args.limit_train_batches)}")
    if args.limit_val_batches is not None:
        cmd.append(f"training.limit_val_batches={int(args.limit_val_batches)}")

    cmd.extend(BASELINE_OVERRIDES)
    cmd.extend(VARIANT_OVERRIDES[variant])
    cmd.extend(args.extra_override)
    return cmd


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.data_json.exists():
        raise FileNotFoundError(f"data JSON not found: {args.data_json}")
    if args.mlm_epochs < 0 or args.corruption_epochs < 0:
        raise ValueError("--mlm-epochs and --corruption-epochs must be non-negative.")

    variants = args.variant or list(VARIANT_OVERRIDES)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.run_root or Path("outputs/teacher_feature_ablation") / (
        f"{timestamp}_{args.run_prefix}_attnpool_mlm{args.mlm_epochs}_corr{args.corruption_epochs}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Run root: {run_root}", flush=True)
    print(f"Variants: {', '.join(variants)}", flush=True)
    print(f"Default epochs: mlm={args.mlm_epochs}, corruption={args.corruption_epochs}", flush=True)
    if args.baseline_mlm_checkpoint is not None:
        print(f"Baseline MLM checkpoint: {resolve_checkpoint_path(args.baseline_mlm_checkpoint, dry_run=args.dry_run)}", flush=True)

    failures: list[tuple[str, Exception]] = []
    for variant in variants:
        run_dir = run_root / variant
        total_epochs, mlm_epochs, corruption_epochs, init_checkpoint, init_checkpoint_strict = variant_epoch_plan(args, variant)
        cmd = build_command(
            args,
            variant=variant,
            run_dir=run_dir,
            total_epochs=total_epochs,
            mlm_epochs=mlm_epochs,
            corruption_epochs=corruption_epochs,
            init_checkpoint=init_checkpoint,
            init_checkpoint_strict=init_checkpoint_strict,
        )
        try:
            run_command(cmd, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            failures.append((variant, exc))
            print(f"Variant failed: {variant} (exit code {exc.returncode})", flush=True)
            if not args.continue_on_error:
                raise

    if failures:
        failed = ", ".join(name for name, _ in failures)
        raise SystemExit(f"Failed variants: {failed}")

    print(f"\nDone. Ablation outputs: {run_root}", flush=True)


if __name__ == "__main__":
    main()
