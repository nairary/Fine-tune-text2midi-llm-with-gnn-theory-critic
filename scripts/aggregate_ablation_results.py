#!/usr/bin/env python3
"""Aggregate one-batch ablation runs into summary/history CSV and Excel tables.

This script scans run directories under an ablation multirun root (default:
``multirun/ablation_one_batch``), reads ``composed_config.yaml`` and
``metrics.jsonl`` from each run, computes critic-oriented derived metrics
(``*_score_gap``), and exports:

- ablation_summary.csv
- ablation_history.csv
- ablation_results.xlsx (sheets: ``summary``, ``history``)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


CONFIG_FIELDS = {
    "run_name": "run_name",
    "dataloader_name": "dataloader.name",
    "corruption_backend": "dataloader.corruption_backend",
    "batch_size": "dataloader.batch_size",
    "pooling_mode": "model.pooling_mode",
    "use_hybrid_graph_scorer": "model.use_hybrid_graph_scorer",
    "local_summary_use_mean": "model.local_summary_use_mean",
    "local_summary_use_max": "model.local_summary_use_max",
    "local_summary_use_topk_mean": "model.local_summary_use_topk_mean",
    "local_summary_topk": "model.local_summary_topk",
    "epochs_planned": "experiment.epochs",
    "train_split": "data.split.train",
    "val_split": "data.split.val",
    "optimizer_lr": "optimizer.lr",
    "scheduler_name": "scheduler.name",
}

HISTORY_CONFIG_COLUMNS = [
    "dataloader_name",
    "corruption_backend",
    "batch_size",
    "pooling_mode",
    "use_hybrid_graph_scorer",
    "local_summary_use_topk_mean",
    "local_summary_topk",
    "epochs_planned",
]

TRAIN_METRIC_NAMES = [
    "rank_acc",
    "mean_margin",
    "score_real_mean",
    "score_corrupted_mean",
    "rank_loss",
    "recon_loss",
    "loss",
]

VAL_METRIC_NAMES = [
    "rank_acc",
    "mean_margin",
    "score_real_mean",
    "score_corrupted_mean",
    "rank_loss",
    "recon_loss",
    "loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("multirun/ablation_one_batch"),
        help="Root directory containing run subdirectories.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to --root.",
    )
    return parser.parse_args()


def get_nested(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    extracted = {column: get_nested(cfg, path) for column, path in CONFIG_FIELDS.items()}
    if extracted.get("run_name") is None:
        extracted["run_name"] = config_path.parent.name
    return extracted


def load_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            train = payload.get("train") or {}
            val = payload.get("val") or {}
            row: dict[str, Any] = {"epoch": payload.get("epoch")}
            for metric_name in TRAIN_METRIC_NAMES:
                row[f"train_{metric_name}"] = train.get(metric_name)
            for metric_name in VAL_METRIC_NAMES:
                row[f"val_{metric_name}"] = val.get(metric_name)
            row["train_score_gap"] = _score_gap(train)
            row["val_score_gap"] = _score_gap(val)
            rows.append(row)
    return rows


def _score_gap(split_metrics: dict[str, Any]) -> float | None:
    real = split_metrics.get("score_real_mean")
    corrupted = split_metrics.get("score_corrupted_mean")
    if real is None or corrupted is None:
        return None
    return float(real) - float(corrupted)


def _to_float(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _safe_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _row_for_best(history_df: pd.DataFrame, column: str) -> pd.Series:
    valid = history_df.dropna(subset=[column])
    if valid.empty:
        return history_df.iloc[-1]
    return valid.loc[valid[column].idxmax()]


def build_summary_row(config_row: dict[str, Any], history_df: pd.DataFrame) -> dict[str, Any]:
    final_row = history_df.iloc[-1]
    best_val_rank = _row_for_best(history_df, "val_rank_acc")
    best_val_margin = _row_for_best(history_df, "val_mean_margin")
    best_val_gap = _row_for_best(history_df, "val_score_gap")

    summary: dict[str, Any] = dict(config_row)
    summary.update(
        {
            "final_epoch": _safe_int(final_row["epoch"]),
            "final_train_rank_acc": _to_float(final_row["train_rank_acc"]),
            "final_val_rank_acc": _to_float(final_row["val_rank_acc"]),
            "final_train_mean_margin": _to_float(final_row["train_mean_margin"]),
            "final_val_mean_margin": _to_float(final_row["val_mean_margin"]),
            "final_train_score_real_mean": _to_float(final_row["train_score_real_mean"]),
            "final_train_score_corrupted_mean": _to_float(final_row["train_score_corrupted_mean"]),
            "final_train_score_gap": _to_float(final_row["train_score_gap"]),
            "final_val_score_real_mean": _to_float(final_row["val_score_real_mean"]),
            "final_val_score_corrupted_mean": _to_float(final_row["val_score_corrupted_mean"]),
            "final_val_score_gap": _to_float(final_row["val_score_gap"]),
            "final_train_rank_loss": _to_float(final_row["train_rank_loss"]),
            "final_val_rank_loss": _to_float(final_row["val_rank_loss"]),
            "final_train_recon_loss": _to_float(final_row["train_recon_loss"]),
            "final_val_recon_loss": _to_float(final_row["val_recon_loss"]),
            "final_train_loss": _to_float(final_row["train_loss"]),
            "final_val_loss": _to_float(final_row["val_loss"]),
            "best_val_rank_acc": _to_float(best_val_rank["val_rank_acc"]),
            "best_epoch_by_val_rank_acc": _safe_int(best_val_rank["epoch"]),
            "train_rank_acc_at_best_val_rank_acc": _to_float(best_val_rank["train_rank_acc"]),
            "val_mean_margin_at_best_val_rank_acc": _to_float(best_val_rank["val_mean_margin"]),
            "val_score_gap_at_best_val_rank_acc": _to_float(best_val_rank["val_score_gap"]),
            "best_val_mean_margin": _to_float(best_val_margin["val_mean_margin"]),
            "best_epoch_by_val_mean_margin": _safe_int(best_val_margin["epoch"]),
            "val_rank_acc_at_best_val_mean_margin": _to_float(best_val_margin["val_rank_acc"]),
            "val_score_gap_at_best_val_mean_margin": _to_float(best_val_margin["val_score_gap"]),
            "best_val_score_gap": _to_float(best_val_gap["val_score_gap"]),
            "best_epoch_by_val_score_gap": _safe_int(best_val_gap["epoch"]),
            "val_rank_acc_at_best_val_score_gap": _to_float(best_val_gap["val_rank_acc"]),
            "val_mean_margin_at_best_val_score_gap": _to_float(best_val_gap["val_mean_margin"]),
            "best_train_rank_acc": _to_float(history_df["train_rank_acc"].max(skipna=True)),
            "best_train_mean_margin": _to_float(history_df["train_mean_margin"].max(skipna=True)),
            "best_train_score_gap": _to_float(history_df["train_score_gap"].max(skipna=True)),
        }
    )
    summary["generalization_gap_rank"] = summary["best_train_rank_acc"] - summary["best_val_rank_acc"]
    return summary


def aggregate(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        config_row = load_config(run_dir / "composed_config.yaml")
        metrics_rows = load_metrics(run_dir / "metrics.jsonl")
        history_df = pd.DataFrame(metrics_rows)
        history_df = history_df.sort_values("epoch", kind="stable").reset_index(drop=True)

        for column in HISTORY_CONFIG_COLUMNS:
            history_df[column] = config_row.get(column)
        history_df.insert(0, "run_name", config_row["run_name"])

        history_rows.extend(history_df.to_dict(orient="records"))
        summary_rows.append(build_summary_row(config_row, history_df))

    summary_df = pd.DataFrame(summary_rows)
    history_df = pd.DataFrame(history_rows)

    summary_df = summary_df.sort_values(
        by=["best_val_rank_acc", "best_val_mean_margin", "best_val_score_gap"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    history_columns = [
        "run_name",
        "epoch",
        *HISTORY_CONFIG_COLUMNS,
        "train_rank_acc",
        "train_mean_margin",
        "train_score_real_mean",
        "train_score_corrupted_mean",
        "train_score_gap",
        "train_rank_loss",
        "train_recon_loss",
        "train_loss",
        "val_rank_acc",
        "val_mean_margin",
        "val_score_real_mean",
        "val_score_corrupted_mean",
        "val_score_gap",
        "val_rank_loss",
        "val_recon_loss",
        "val_loss",
    ]
    history_df = history_df[history_columns]

    return summary_df, history_df


def write_outputs(summary_df: pd.DataFrame, history_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "ablation_summary.csv"
    history_path = outdir / "ablation_history.csv"
    excel_path = outdir / "ablation_results.xlsx"

    summary_df.to_csv(summary_path, index=False)
    history_df.to_csv(history_path, index=False)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        history_df.to_excel(writer, index=False, sheet_name="history")

    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved history CSV: {history_path}")
    print(f"Saved Excel workbook: {excel_path}")


def main() -> None:
    args = parse_args()
    root = args.root
    outdir = args.outdir or root

    summary_df, history_df = aggregate(root)
    write_outputs(summary_df, history_df, outdir)


if __name__ == "__main__":
    main()
