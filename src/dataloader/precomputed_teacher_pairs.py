from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .utils_graph import build_graph_from_encoded, mask_graph


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def _load_song(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected encoded song JSON object at {path}")
    return payload


def _load_cached_graph(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Cached teacher graph not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _corruption_metadata_from_manifest(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "applied": bool(row.get("is_corrupted", False)),
        "corruption_name": row.get("corruption_name", "identity"),
        "corruption_params": row.get("corruption_params", {}),
        "topology_changed": bool(row.get("topology_changed", False)),
        "note_corrupted_indices": row.get("note_corrupted_indices", []),
        "chord_corrupted_indices": row.get("chord_corrupted_indices", []),
        "onset_corrupted_indices": row.get("onset_corrupted_indices", []),
        "attempted_corruption_modes": row.get("attempted_corruption_modes", []),
        "skipped_corruption_attempts": row.get("skipped_corruption_attempts", []),
        "source_song_id": row.get("source_song_id"),
        "pair_group_id": row.get("pair_group_id"),
    }


class PrecomputedTeacherPairDataset(Dataset):
    """Teacher SSL dataset backed by a fixed clean/corrupted PairCorpus.

    The expected layout is the one produced by ``src.observer.build_observer_pair_dataset``:
    a split manifest with sample rows and a split pair index with aligned clean/corrupted
    sample ids.  Returned items intentionally match ``HookTheoryDataset`` so the existing
    teacher training loop and collate function can be reused unchanged.
    """

    def __init__(
        self,
        pair_index_jsonl: str | Path,
        manifest_jsonl: str | Path,
        *,
        mask_prob: float = 0.15,
        mask_min_nodes: int = 1,
        optional_mask_field_prob: float = 0.5,
        base_dir: str | Path | None = None,
        graph_index_jsonl: str | Path | None = None,
    ) -> None:
        self.pair_index_jsonl = Path(pair_index_jsonl)
        self.manifest_jsonl = Path(manifest_jsonl)
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.mask_prob = float(mask_prob)
        self.mask_min_nodes = int(mask_min_nodes)
        self.optional_mask_field_prob = float(optional_mask_field_prob)
        self.graph_index_jsonl = Path(graph_index_jsonl) if graph_index_jsonl is not None else None

        if not self.pair_index_jsonl.exists():
            raise FileNotFoundError(f"Pair index not found: {self.pair_index_jsonl}")
        if not self.manifest_jsonl.exists():
            raise FileNotFoundError(f"Pair manifest not found: {self.manifest_jsonl}")

        manifest_rows = _read_jsonl(self.manifest_jsonl)
        self.manifest_by_sample_id = {str(row["sample_id"]): row for row in manifest_rows}
        self.graph_by_sample_id: dict[str, dict[str, Any]] = {}
        if self.graph_index_jsonl is not None:
            if not self.graph_index_jsonl.exists():
                raise FileNotFoundError(f"Teacher graph index not found: {self.graph_index_jsonl}")
            graph_rows = _read_jsonl(self.graph_index_jsonl)
            self.graph_by_sample_id = {str(row["sample_id"]): row for row in graph_rows}

        pair_rows = _read_jsonl(self.pair_index_jsonl)
        self.pair_rows: list[dict[str, Any]] = []
        for row in pair_rows:
            if not bool(row.get("is_valid_pair_for_rank", True)):
                continue
            clean_id = str(row.get("clean_sample_id", ""))
            corrupted_id = str(row.get("corrupted_sample_id", ""))
            if clean_id not in self.manifest_by_sample_id or corrupted_id not in self.manifest_by_sample_id:
                continue
            if self.graph_by_sample_id and (clean_id not in self.graph_by_sample_id or corrupted_id not in self.graph_by_sample_id):
                continue
            self.pair_rows.append(row)

        if not self.pair_rows:
            raise ValueError(f"No valid precomputed teacher pairs found in {self.pair_index_jsonl}")

    def __len__(self) -> int:
        return len(self.pair_rows)

    def _load_graph(self, row: dict[str, Any]):
        sample_id = str(row["sample_id"])
        graph_row = self.graph_by_sample_id.get(sample_id)
        if graph_row is not None:
            graph_path = _resolve_path(graph_row["graph_path"], self.base_dir)
            return _load_cached_graph(graph_path)

        encoded_path = _resolve_path(row["encoded_song_path"], self.base_dir)
        return build_graph_from_encoded(_load_song(encoded_path))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self.pair_rows[idx]
        clean_row = self.manifest_by_sample_id[str(pair["clean_sample_id"])]
        corrupted_row = self.manifest_by_sample_id[str(pair["corrupted_sample_id"])]

        graph_real = self._load_graph(clean_row)
        graph_masked, masked_labels = mask_graph(
            graph_real,
            mask_prob=self.mask_prob,
            min_nodes_to_mask=self.mask_min_nodes,
            optional_mask_field_prob=self.optional_mask_field_prob,
        )
        graph_corrupted = self._load_graph(corrupted_row)

        return {
            "graph_real": graph_real,
            "graph_masked": graph_masked,
            "graph_corrupted": graph_corrupted,
            "masked_labels": masked_labels,
            "corruption_metadata": _corruption_metadata_from_manifest(corrupted_row),
            "graph_score_label": 1.0,
        }
