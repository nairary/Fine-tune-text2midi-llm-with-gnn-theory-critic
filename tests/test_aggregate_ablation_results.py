from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.aggregate_ablation_results import aggregate, write_outputs


def _write_run(run_dir: Path, run_name: str, val_accs: list[float], val_margins: list[float]):
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_name": run_name,
        "dataloader": {
            "name": "graph_ablation",
            "corruption_backend": "graph",
            "batch_size": 16,
        },
        "model": {
            "pooling_mode": "mean",
            "use_hybrid_graph_scorer": True,
            "local_summary_use_mean": True,
            "local_summary_use_max": False,
            "local_summary_use_topk_mean": True,
            "local_summary_topk": 5,
        },
        "experiment": {"epochs": 3},
        "data": {"split": {"train": "train", "val": "train"}},
        "optimizer": {"lr": 1e-3},
        "scheduler": {"name": "none"},
    }
    (run_dir / "composed_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    rows = []
    for epoch in range(1, 4):
        rows.append(
            {
                "epoch": epoch,
                "train": {
                    "rank_acc": 0.55 + 0.1 * epoch,
                    "mean_margin": 0.1 * epoch,
                    "score_real_mean": 1.0 + 0.1 * epoch,
                    "score_corrupted_mean": 0.6,
                    "rank_loss": 0.5,
                    "recon_loss": 0.4,
                    "loss": 0.9,
                },
                "val": {
                    "rank_acc": val_accs[epoch - 1],
                    "mean_margin": val_margins[epoch - 1],
                    "score_real_mean": 0.9 + 0.1 * epoch,
                    "score_corrupted_mean": 0.5,
                    "rank_loss": 0.45,
                    "recon_loss": 0.35,
                    "loss": 0.8,
                },
            }
        )

    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_aggregate_builds_summary_history_and_writes_exports(tmp_path: Path):
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    import pandas as pd

    root = tmp_path / "multirun" / "ablation_one_batch"
    _write_run(root / "run_a", run_name="run_a", val_accs=[0.6, 0.7, 0.68], val_margins=[0.1, 0.2, 0.15])
    _write_run(root / "run_b", run_name="run_b", val_accs=[0.5, 0.62, 0.64], val_margins=[0.05, 0.12, 0.22])

    summary_df, history_df = aggregate(root)

    assert len(summary_df) == 2
    assert len(history_df) == 6
    assert {"train_score_gap", "val_score_gap"}.issubset(history_df.columns)

    run_a_epoch1 = history_df[(history_df["run_name"] == "run_a") & (history_df["epoch"] == 1)].iloc[0]
    assert run_a_epoch1["train_score_gap"] == pytest.approx((1.0 + 0.1 * 1) - 0.6)
    assert run_a_epoch1["val_score_gap"] == pytest.approx((0.9 + 0.1 * 1) - 0.5)

    run_a_summary = summary_df[summary_df["run_name"] == "run_a"].iloc[0]
    assert run_a_summary["best_val_rank_acc"] == pytest.approx(0.7)
    assert run_a_summary["best_epoch_by_val_rank_acc"] == 2

    outdir = tmp_path / "exports"
    write_outputs(summary_df, history_df, outdir)

    assert (outdir / "ablation_summary.csv").exists()
    assert (outdir / "ablation_history.csv").exists()
    assert (outdir / "ablation_results.xlsx").exists()

    excel = pd.ExcelFile(outdir / "ablation_results.xlsx")
    assert {"summary", "history"}.issubset(set(excel.sheet_names))
