from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.observer.build_observer_graph_cache import build_graph_cache
from src.observer.build_observer_pair_dataset import build_pairs
from src.observer.build_observer_pair_targets import build_pair_targets
from src.observer.pipeline_paths import resolve_observer_pipeline_paths
from src.observer.train_observer_distill import train

LOGGER = logging.getLogger(__name__)


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for x in handle if x.strip())


def _validate_after_pairs(cfg: DictConfig, output_root: Path) -> None:
    _ = output_root
    paths = resolve_observer_pipeline_paths(cfg)
    manifest_root = paths["manifests_root"]
    pair_root = paths["pair_index_root"]
    train_rows = _count_jsonl(manifest_root / "train.jsonl")
    val_rows = _count_jsonl(manifest_root / "val.jsonl")
    train_pairs = _count_jsonl(pair_root / "train_pairs.jsonl")
    val_pairs = _count_jsonl(pair_root / "val_pairs.jsonl")
    LOGGER.info("pairs summary train_rows=%d val_rows=%d train_pairs=%d val_pairs=%d", train_rows, val_rows, train_pairs, val_pairs)
    if train_pairs == 0:
        raise ValueError("No train pairs were built for observer pipeline")
    if val_pairs == 0:
        raise ValueError("No validation pairs were built for observer pipeline")


def _validate_after_targets(cfg: DictConfig, output_root: Path) -> None:
    _ = output_root
    paths = resolve_observer_pipeline_paths(cfg)
    targets_root = paths["targets_root"]
    train_rows = _count_jsonl(targets_root / "train.jsonl")
    val_rows = _count_jsonl(targets_root / "val.jsonl")
    train_pairs = _count_jsonl(targets_root / "train_pairs.jsonl")
    val_pairs = _count_jsonl(targets_root / "val_pairs.jsonl")
    LOGGER.info("targets summary train=%d val=%d train_pairs=%d val_pairs=%d", train_rows, val_rows, train_pairs, val_pairs)
    if train_pairs == 0:
        raise ValueError("No train pair targets were built")
    if val_pairs == 0:
        raise ValueError("No validation pair targets were built")


def _validate_after_cache(cfg: DictConfig, output_root: Path) -> None:
    _ = output_root
    paths = resolve_observer_pipeline_paths(cfg)
    index_root = paths["cache_index_root"]
    train_rows = _count_jsonl(index_root / "train.jsonl")
    val_rows = _count_jsonl(index_root / "val.jsonl")
    LOGGER.info("cache summary train=%d val=%d", train_rows, val_rows)
    if train_rows == 0:
        raise ValueError("No cached train graphs were built")
    if val_rows == 0:
        raise ValueError("No cached validation graphs were built")


@hydra.main(version_base=None, config_path="../../configs", config_name="observer_distill")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    output_root = resolve_observer_pipeline_paths(cfg)["output_root"]

    if bool(cfg.observer_pipeline.build_pairs):
        LOGGER.info("[1/4] build pairs")
        build_pairs(cfg)
        _validate_after_pairs(cfg, output_root)
    if bool(cfg.observer_pipeline.build_targets):
        LOGGER.info("[2/4] build targets")
        build_pair_targets(cfg)
        _validate_after_targets(cfg, output_root)
    if bool(cfg.observer_pipeline.build_graph_cache):
        LOGGER.info("[3/4] build graph cache")
        build_graph_cache(cfg)
        _validate_after_cache(cfg, output_root)
    if bool(cfg.observer_pipeline.get("train", True)):
        LOGGER.info("[4/4] train observer")
        train(cfg)


if __name__ == "__main__":
    main()
