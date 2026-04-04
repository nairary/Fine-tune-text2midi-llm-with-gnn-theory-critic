from __future__ import annotations

import itertools
import math
import os
import re
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train_teacher import build_loaders

CONFIG_DIR = REPO_ROOT / "configs"
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_ablation_one_batch.sh"

EXPECTED_AXES = {
    "experiment.epochs": "50,500",
    "model.use_hybrid_graph_scorer": "false,true",
    "model.pooling_mode": "mean,mean_max",
    "dataloader": "graph_ablation,theory_aware_ablation",
    "dataloader.batch_size": "1,16",
}


def _compose(overrides: list[str] | None = None, *, return_hydra_config: bool = False):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="ablation_one_batch",
            overrides=overrides or [],
            return_hydra_config=return_hydra_config,
        )


def test_ablation_one_batch_compose_limits_batches_and_hydra_subdir():
    cfg = _compose(return_hydra_config=True)
    assert cfg.experiment.limit_train_batches == 1
    assert cfg.experiment.limit_val_batches == 1
    assert cfg.hydra.sweep.dir == "multirun/ablation_one_batch"
    assert cfg.hydra.sweep.subdir == cfg.run_name


def test_ablation_dataloader_backends():
    graph_cfg = _compose(["dataloader=graph_ablation"])
    theory_cfg = _compose(["dataloader=theory_aware_ablation"])

    assert graph_cfg.dataloader.corruption_backend == "graph"
    assert theory_cfg.dataloader.corruption_backend == "song_theory"


def test_run_name_contains_all_sweep_dimensions():
    cfg = _compose(
        [
            "experiment.epochs=50",
            "model.use_hybrid_graph_scorer=false",
            "model.pooling_mode=mean",
            "dataloader=graph_ablation",
            "dataloader.batch_size=1",
        ]
    )
    run_name_1 = OmegaConf.to_container(cfg, resolve=True)["run_name"]
    assert "graph_ablation" in run_name_1
    assert "pool-mean" in run_name_1
    assert "hyb-False" in run_name_1 or "hyb-false" in run_name_1
    assert "bs-1" in run_name_1
    assert "ep-50" in run_name_1

    cfg = _compose(
        [
            "experiment.epochs=500",
            "model.use_hybrid_graph_scorer=true",
            "model.pooling_mode=mean_max",
            "dataloader=theory_aware_ablation",
            "dataloader.batch_size=16",
        ]
    )
    run_name_2 = OmegaConf.to_container(cfg, resolve=True)["run_name"]
    assert "theory_aware_ablation" in run_name_2
    assert "pool-mean_max" in run_name_2
    assert "hyb-True" in run_name_2 or "hyb-true" in run_name_2
    assert "bs-16" in run_name_2
    assert "ep-500" in run_name_2


def test_run_names_are_unique_for_all_32_sweep_combinations():
    names: set[str] = set()
    for epochs, hybrid, pool, dataloader_name, batch_size in itertools.product(
        [50, 500],
        [False, True],
        ["mean", "mean_max"],
        ["graph_ablation", "theory_aware_ablation"],
        [1, 16],
    ):
        cfg = _compose(
            [
                f"experiment.epochs={epochs}",
                f"model.use_hybrid_graph_scorer={str(hybrid).lower()}",
                f"model.pooling_mode={pool}",
                f"dataloader={dataloader_name}",
                f"dataloader.batch_size={batch_size}",
            ]
        )
        run_name = OmegaConf.to_container(cfg, resolve=True)["run_name"]
        names.add(run_name)

        assert dataloader_name in run_name
        assert f"pool-{pool}" in run_name
        assert ("hyb-True" in run_name or "hyb-true" in run_name) if hybrid else ("hyb-False" in run_name or "hyb-false" in run_name)
        assert f"bs-{batch_size}" in run_name
        assert f"ep-{epochs}" in run_name

    assert len(names) == 32


def test_integration_ablation_dataloaders_runtime_smoke_via_build_loaders():
    if os.getenv("RUN_ABLATION_INTEGRATION") != "1":
        pytest.skip("Set RUN_ABLATION_INTEGRATION=1 to run runtime dataloader smoke checks.")

    for dataloader_name in ["graph_ablation", "theory_aware_ablation"]:
        overrides = [
            f"dataloader={dataloader_name}",
            "dataloader.batch_size=1",
            "training.limit_train_samples=16",
            "training.limit_val_samples=16",
        ]
        if dataloader_name == "theory_aware_ablation":
            overrides.append("dataloader.corruption_modes=[borrowed_kind_toggle_without_melody_change]")

        cfg = _compose(overrides)
        dataset_path = REPO_ROOT / str(cfg.data.json_path)
        if not dataset_path.exists():
            pytest.skip(f"Dataset file not found for integration smoke: {dataset_path}")

        _, train_loader, _ = build_loaders(cfg)
        batch = next(iter(train_loader))

        assert "graph_real" in batch
        assert "graph_masked" in batch
        assert "graph_corrupted" in batch
        assert "graph_score_label" in batch


def test_shell_script_has_expected_axes_without_max_pooling_and_32_combinations():
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    for key, values in EXPECTED_AXES.items():
        assert f"{key}={values}" in content

    assert "pooling_mode=max" not in content

    axis_pattern = re.compile(r"([a-zA-Z0-9_.]+)=([^\\\n]+)")
    axes_in_script = dict(axis_pattern.findall(content))
    for key, values in EXPECTED_AXES.items():
        assert axes_in_script[key].strip() == values

    total = math.prod(len(values.split(",")) for values in EXPECTED_AXES.values())
    assert total == 32
