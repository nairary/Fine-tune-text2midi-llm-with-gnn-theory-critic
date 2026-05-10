from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from src.observer.build_teacher_targets import build_teacher_targets_jsonl, load_jsonl_rows, write_jsonl
from src.observer.pipeline_paths import resolve_observer_pipeline_paths

LOGGER = logging.getLogger(__name__)


class PairTargetJoinError(ValueError):
    pass


def _base_cwd() -> Path:
    try:
        return Path(hydra.utils.get_original_cwd())
    except Exception:
        return Path(os.getcwd())


def _join_pair_targets(target_rows: list[dict[str, Any]], pair_index_path: Path, output_path: Path) -> tuple[int, int]:
    by_sample_id = {str(row.get("sample_id", row.get("song_id"))): row for row in target_rows}
    joined: list[dict[str, Any]] = []
    skipped = 0

    with pair_index_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            pair = json.loads(line)
            pair_group_id = str(pair.get("pair_group_id", f"line_{line_no}"))
            clean_id = str(pair.get("clean_sample_id"))
            corrupted_id = str(pair.get("corrupted_sample_id"))

            if clean_id not in by_sample_id:
                raise PairTargetJoinError(f"pair_group_id='{pair_group_id}' references missing clean_sample_id='{clean_id}'")
            if corrupted_id not in by_sample_id:
                raise PairTargetJoinError(f"pair_group_id='{pair_group_id}' references missing corrupted_sample_id='{corrupted_id}'")

            clean = by_sample_id[clean_id]
            corrupted = by_sample_id[corrupted_id]
            clean_score = float(clean["teacher_score"])
            corr_score = float(corrupted["teacher_score"])
            gap = clean_score - corr_score
            if not all(math.isfinite(v) for v in (clean_score, corr_score, gap)):
                raise PairTargetJoinError(
                    f"pair_group_id='{pair_group_id}' has non-finite scores: clean={clean_score} corrupted={corr_score} gap={gap}"
                )

            is_valid_pair_for_rank = bool(pair.get("is_valid_pair_for_rank", True))
            if not is_valid_pair_for_rank:
                skipped += 1
                continue

            joined.append(
                {
                    "pair_group_id": pair_group_id,
                    "clean_sample_id": clean_id,
                    "corrupted_sample_id": corrupted_id,
                    "teacher_score_clean": clean_score,
                    "teacher_score_corrupted": corr_score,
                    "teacher_score_gap": gap,
                    "is_valid_pair_for_rank": True,
                }
            )
    write_jsonl(output_path, joined)
    return len(joined), skipped


def build_pair_targets(cfg: DictConfig) -> None:
    paths = resolve_observer_pipeline_paths(cfg)
    manifests_root = paths["manifests_root"]
    index_root = paths["pair_index_root"]
    targets_root = paths["targets_root"]
    overwrite = bool(cfg.observer_pipeline.get("overwrite", False))
    resume = not overwrite
    target_log_every = int(cfg.observer_training.get("target_log_every", 100))
    teacher_checkpoint = Path(cfg.observer_training.teacher_checkpoint)
    teacher_config = Path(cfg.observer_training.teacher_config)
    if not teacher_checkpoint.is_absolute():
        teacher_checkpoint = _base_cwd() / teacher_checkpoint
    if not teacher_config.is_absolute():
        teacher_config = _base_cwd() / teacher_config

    for split in cfg.data.split.keys():
        manifest_path = manifests_root / f"{split}.jsonl"
        pair_path = index_root / f"{split}_pairs.jsonl"
        split_targets = targets_root / f"{split}.jsonl"
        split_pair_targets = targets_root / f"{split}_pairs.jsonl"
        if not manifest_path.exists() or not pair_path.exists():
            split_targets.unlink(missing_ok=True)
            split_pair_targets.unlink(missing_ok=True)
            continue

        if overwrite:
            split_targets.unlink(missing_ok=True)
            split_pair_targets.unlink(missing_ok=True)

        rows = load_jsonl_rows(manifest_path)
        LOGGER.info(
            "Building teacher targets split=%s samples=%d output=%s resume=%s",
            split,
            len(rows),
            split_targets,
            resume,
        )
        built_targets = build_teacher_targets_jsonl(
            rows=rows,
            output_jsonl=split_targets,
            teacher_checkpoint=teacher_checkpoint,
            teacher_config=teacher_config,
            encoded_song_field="encoded_song_path",
            encoded_song_root=None,
            split=split,
            device=str(cfg.observer_training.device),
            include_intermediates=bool(cfg.observer_training.get("cache_teacher_intermediates", False)),
            resume=resume,
            log_every=target_log_every,
        )
        target_rows = load_jsonl_rows(split_targets)
        built_pairs, skipped_pairs = _join_pair_targets(
            target_rows=target_rows,
            pair_index_path=pair_path,
            output_path=split_pair_targets,
        )
        LOGGER.info(
            "Targets split=%s samples=%d built_targets=%d pairs=%d skipped_pairs=%d output=%s",
            split,
            len(target_rows),
            built_targets,
            built_pairs,
            skipped_pairs,
            split_pair_targets,
        )


@hydra.main(version_base=None, config_path="../../configs", config_name="observer_distill")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    build_pair_targets(cfg)


if __name__ == "__main__":
    main()
