from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Dataset, HeteroData

from src.observer.data_pipeline import build_observer_graph, build_observer_song_record, load_observer_input_jsonl


class ObserverDatasetValidationError(ValueError):
    """Raised when observer training dataset inputs are inconsistent."""


_REQUIRED_TARGET_FIELDS = ("song_id", "teacher_score")


class ObserverDataset(Dataset):
    def __init__(
        self,
        input_jsonl: str | Path,
        target_jsonl: str | Path,
        chord_weights_yaml: str | None = None,
        chord_instrument_name: str = "chords",
        use_fallback_44: bool = True,
        in_memory: bool = False,
    ) -> None:
        super().__init__()
        self.input_jsonl = Path(input_jsonl)
        self.target_jsonl = Path(target_jsonl)
        self.chord_weights_yaml = chord_weights_yaml
        self.chord_instrument_name = chord_instrument_name
        self.use_fallback_44 = bool(use_fallback_44)
        self.in_memory = bool(in_memory)

        self._samples = load_observer_input_jsonl(self.input_jsonl)
        self._targets_by_song_id = _load_teacher_targets_jsonl(self.target_jsonl)
        self._joined_rows = _join_samples_and_targets(self._samples, self._targets_by_song_id)
        self._cache: list[HeteroData] | None = None

        if self.in_memory:
            self._cache = [self._build_graph(row) for row in self._joined_rows]

    def len(self) -> int:  # type: ignore[override]
        return len(self._joined_rows)

    def get(self, idx: int) -> HeteroData:  # type: ignore[override]
        if idx < 0 or idx >= len(self._joined_rows):
            raise IndexError(f"ObserverDataset index out of range: {idx}")
        if self._cache is not None:
            return self._cache[idx]
        return self._build_graph(self._joined_rows[idx])

    def _build_graph(self, row: dict[str, Any]) -> HeteroData:
        sample = row["sample"]
        target = row["target"]
        try:
            record = build_observer_song_record(
                sample,
                chord_weights_yaml=self.chord_weights_yaml,
                chord_instrument_name=self.chord_instrument_name,
                use_fallback_44=self.use_fallback_44,
            )
            graph = build_observer_graph(record)
        except Exception as exc:  # noqa: BLE001
            raise ObserverDatasetValidationError(
                f"Failed to build observer graph for song_id='{sample['song_id']}'"
            ) from exc

        teacher_score = float(target["teacher_score"])
        if not math.isfinite(teacher_score):
            raise ObserverDatasetValidationError(
                f"teacher_score for song_id='{sample['song_id']}' must be finite, got {teacher_score}"
            )

        graph.y = torch.tensor([teacher_score], dtype=torch.float)
        graph.song_id = sample["song_id"]
        graph.teacher_score = teacher_score
        for optional_key in ("is_corrupted", "corruption_name", "pair_group_id", "source_song_id"):
            if optional_key in target:
                setattr(graph, optional_key, target[optional_key])
        return graph


def _load_teacher_targets_jsonl(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    rows_by_song_id: dict[str, dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ObserverDatasetValidationError(f"Invalid target JSON at line {line_idx}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ObserverDatasetValidationError(f"Target line {line_idx}: row must be a JSON object")
            for field in _REQUIRED_TARGET_FIELDS:
                if field not in payload:
                    raise ObserverDatasetValidationError(f"Target line {line_idx}: missing required field '{field}'")

            song_id = payload["song_id"]
            if not isinstance(song_id, str) or not song_id:
                raise ObserverDatasetValidationError(f"Target line {line_idx}: song_id must be non-empty string")
            if song_id in rows_by_song_id:
                raise ObserverDatasetValidationError(f"Duplicate song_id in target dump: '{song_id}'")

            teacher_score = float(payload["teacher_score"])
            if not math.isfinite(teacher_score):
                raise ObserverDatasetValidationError(
                    f"Target line {line_idx}: teacher_score for song_id='{song_id}' must be finite"
                )

            payload["teacher_score"] = teacher_score
            rows_by_song_id[song_id] = payload
    return rows_by_song_id


def _join_samples_and_targets(
    samples: list[dict[str, Any]],
    targets_by_song_id: dict[str, dict[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    seen_song_ids: set[str] = set()
    joined: list[dict[str, dict[str, Any]]] = []

    for sample in samples:
        song_id = sample["song_id"]
        if song_id in seen_song_ids:
            raise ObserverDatasetValidationError(f"Duplicate song_id in input manifest: '{song_id}'")
        seen_song_ids.add(song_id)

        target = targets_by_song_id.get(song_id)
        if target is None:
            raise ObserverDatasetValidationError(f"Missing teacher_score for song_id='{song_id}'")
        joined.append({"sample": sample, "target": target})

    extra_target_song_ids = sorted(set(targets_by_song_id.keys()) - seen_song_ids)
    if extra_target_song_ids:
        raise ObserverDatasetValidationError(
            f"Teacher target dump contains song_ids absent from input manifest: {extra_target_song_ids}"
        )

    return joined
