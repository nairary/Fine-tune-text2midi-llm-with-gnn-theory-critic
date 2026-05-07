from __future__ import annotations

import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from src.observer.data_pipeline import build_observer_graph, build_observer_song_record, load_observer_input_jsonl
from src.observer.pipeline_paths import resolve_observer_pipeline_paths

LOGGER = logging.getLogger(__name__)


def _base_cwd() -> Path:
    try:
        return Path(hydra.utils.get_original_cwd())
    except Exception:
        return Path(os.getcwd())


def _load_targets(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row.get("sample_id", row.get("song_id")))] = row
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_graph_cache(cfg: DictConfig) -> None:
    paths = resolve_observer_pipeline_paths(cfg)
    manifests_root = paths["manifests_root"]
    targets_root = paths["targets_root"]
    cache_root = paths["graph_cache_root"]
    index_root = paths["cache_index_root"]

    for split in cfg.data.split.keys():
        manifest_path = manifests_root / f"{split}.jsonl"
        targets_path = targets_root / f"{split}.jsonl"
        split_graph_dir = cache_root / split
        split_index_path = index_root / f"{split}.jsonl"
        if not manifest_path.exists() or not targets_path.exists():
            split_index_path.unlink(missing_ok=True)
            if split_graph_dir.exists():
                shutil.rmtree(split_graph_dir)
            continue

        if split_graph_dir.exists():
            shutil.rmtree(split_graph_dir)

        samples = load_observer_input_jsonl(manifest_path)
        targets = _load_targets(targets_path)

        rows: list[dict[str, Any]] = []
        skipped = 0
        for sample in samples:
            sample_id = str(sample.get("sample_id", sample["song_id"]))
            target_row = targets.get(sample_id)
            if target_row is None:
                skipped += 1
                LOGGER.warning("Split=%s sample_id=%s skipped: missing target row", split, sample_id)
                continue

            teacher_score = float(target_row["teacher_score"])
            if not math.isfinite(teacher_score):
                raise ValueError(f"sample_id='{sample_id}' has non-finite teacher_score={teacher_score}")

            try:
                record = build_observer_song_record(
                    sample,
                    chord_weights_yaml=cfg.observer_training.get("chord_weights_yaml"),
                    chord_instrument_name=str(cfg.observer_training.get("chord_instrument_name", "chords")),
                    use_fallback_44=bool(cfg.observer_training.get("use_fallback_44", True)),
                )
                graph = build_observer_graph(record)
            except Exception as exc:  # noqa: BLE001
                if bool(cfg.observer_pipeline.get("skip_graph_build_failures", True)):
                    skipped += 1
                    LOGGER.warning("Split=%s sample_id=%s skipped graph build: %s", split, sample_id, exc)
                    continue
                raise

            graph.y = torch.tensor([teacher_score], dtype=torch.float)
            graph.sample_id = sample_id
            graph_path = split_graph_dir / f"{sample_id}.pt"
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(graph, graph_path)
            index_row = {
                "sample_id": sample_id,
                "pair_group_id": target_row.get("pair_group_id"),
                "source_song_id": target_row.get("source_song_id"),
                "split": split,
                "is_corrupted": bool(target_row.get("is_corrupted", False)),
                "corruption_name": target_row.get("corruption_name", "identity"),
                "teacher_score": teacher_score,
                "graph_path": str(graph_path),
            }
            for key in ("teacher_graph_embedding", "teacher_pooled_by_type", "teacher_local_score_summaries"):
                if key in target_row:
                    index_row[key] = target_row[key]
            rows.append(index_row)
        _write_jsonl(split_index_path, sorted(rows, key=lambda x: str(x["sample_id"])))
        LOGGER.info("Graph cache split=%s built=%d skipped=%d", split, len(rows), skipped)


@hydra.main(version_base=None, config_path="../../configs", config_name="observer_distill")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    build_graph_cache(cfg)


if __name__ == "__main__":
    main()
