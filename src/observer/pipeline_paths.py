from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig


def _base_cwd() -> Path:
    try:
        return Path(hydra.utils.get_original_cwd())
    except Exception:
        return Path(os.getcwd())


def resolve_observer_pipeline_paths(cfg: DictConfig, base_cwd: Path | None = None) -> dict[str, Path]:
    root = Path(cfg.observer_pipeline.output_root)
    cwd = base_cwd or _base_cwd()
    if not root.is_absolute():
        root = cwd / root

    encoded_root = root / str(cfg.observer_pipeline.get("encoded_output_dir", "pairs/encoded"))
    midi_root = root / str(cfg.observer_pipeline.get("midi_output_dir", "pairs/midi"))
    manifests_root = root / str(cfg.observer_pipeline.get("manifest_output_dir", "pairs/manifests"))
    pair_index_root = manifests_root.parent / "index"
    skipped_log_path = manifests_root.parent / "skipped_manifest_rows.jsonl"

    targets_root = root / str(cfg.observer_pipeline.get("targets_output_dir", "targets"))
    cache_root = root / str(cfg.observer_pipeline.get("cache_output_dir", "cache"))
    graph_cache_root = cache_root / "graphs"
    cache_index_root = cache_root / "index"
    training_root = root / str(cfg.observer_pipeline.get("training_output_dir", "training"))

    return {
        "output_root": root,
        "encoded_root": encoded_root,
        "midi_root": midi_root,
        "manifests_root": manifests_root,
        "pair_index_root": pair_index_root,
        "targets_root": targets_root,
        "cache_root": cache_root,
        "graph_cache_root": graph_cache_root,
        "cache_index_root": cache_index_root,
        "skipped_log_path": skipped_log_path,
        "training_root": training_root,
    }
