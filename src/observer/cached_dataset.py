from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class ObserverPairCachedDataset(Dataset):
    def __init__(
        self,
        graph_index_jsonl: str | Path,
        pair_target_index_jsonl: str | Path,
        mode: str = "pair",
    ) -> None:
        self.mode = mode
        if self.mode not in {"sample", "pair"}:
            raise ValueError("mode must be 'sample' or 'pair'")

        self.graph_rows = self._read_jsonl(Path(graph_index_jsonl))
        if not self.graph_rows:
            raise ValueError(f"Graph index is empty: {graph_index_jsonl}")
        self.graph_by_sample_id = {row["sample_id"]: row for row in self.graph_rows}

        self.pair_rows = self._read_jsonl(Path(pair_target_index_jsonl)) if Path(pair_target_index_jsonl).exists() else []
        self._validate_graph_rows()
        self.distillation_target_dims = self._infer_distillation_target_dims()
        if self.mode == "pair":
            self._validate_pair_rows()

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _validate_graph_rows(self) -> None:
        for row in self.graph_rows:
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise ValueError("Graph index row missing sample_id")
            graph_path = Path(str(row.get("graph_path", "")))
            if not graph_path.exists():
                raise ValueError(f"Missing graph file for sample_id='{sample_id}': {graph_path}")

    @staticmethod
    def _vector_dim(value: Any) -> int | None:
        if not isinstance(value, list) or not value:
            return None
        try:
            for item in value:
                scalar = float(item)
                if not math.isfinite(scalar):
                    return None
        except (TypeError, ValueError):
            return None
        return len(value)

    def _infer_distillation_target_dims(self) -> dict[str, Any]:
        dims: dict[str, Any] = {
            "teacher_graph_embedding": None,
            "teacher_local_score_summaries": None,
            "teacher_pooled_by_type": {},
        }
        pooled_dims: dict[str, int] = {}
        for row in self.graph_rows:
            if dims["teacher_graph_embedding"] is None:
                dims["teacher_graph_embedding"] = self._vector_dim(row.get("teacher_graph_embedding"))
            if dims["teacher_local_score_summaries"] is None:
                dims["teacher_local_score_summaries"] = self._vector_dim(row.get("teacher_local_score_summaries"))

            pooled = row.get("teacher_pooled_by_type")
            if isinstance(pooled, dict):
                for node_type, values in pooled.items():
                    if str(node_type) in pooled_dims:
                        continue
                    dim = self._vector_dim(values)
                    if dim is not None:
                        pooled_dims[str(node_type)] = dim
        dims["teacher_pooled_by_type"] = pooled_dims
        return dims

    @staticmethod
    def _teacher_distillation_targets(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: row[key]
            for key in ("teacher_graph_embedding", "teacher_pooled_by_type", "teacher_local_score_summaries")
            if key in row
        }

    def _validate_pair_rows(self) -> None:
        if not self.pair_rows:
            raise ValueError("Pair target index is empty for pair mode")

        valid_pairs: list[dict[str, Any]] = []
        skipped = 0

        for pair in self.pair_rows:
            pair_group_id = str(pair.get("pair_group_id", "<unknown>"))
            clean_id = str(pair.get("clean_sample_id", ""))
            corr_id = str(pair.get("corrupted_sample_id", ""))

            if clean_id not in self.graph_by_sample_id or corr_id not in self.graph_by_sample_id:
                skipped += 1
                continue

            clean_score = float(pair.get("teacher_score_clean"))
            corr_score = float(pair.get("teacher_score_corrupted"))
            gap = float(pair.get("teacher_score_gap", clean_score - corr_score))
            if not all(math.isfinite(v) for v in (clean_score, corr_score, gap)):
                raise ValueError(
                    f"pair_group_id='{pair_group_id}' has non-finite target values "
                    f"(clean={clean_score}, corrupted={corr_score}, gap={gap})"
                )

            valid_pairs.append(pair)

        if not valid_pairs:
            raise ValueError("No valid pairs remain after filtering by graph index")

        self.pair_rows = valid_pairs

    def __len__(self) -> int:
        return len(self.graph_rows) if self.mode == "sample" else len(self.pair_rows)

    def __getitem__(self, idx: int) -> Any:
        if self.mode == "sample":
            row = self.graph_rows[idx]
            graph = torch.load(row["graph_path"], map_location="cpu", weights_only=False)
            graph.y = torch.tensor([float(row["teacher_score"])], dtype=torch.float)
            return graph

        pair = self.pair_rows[idx]
        clean_row = self.graph_by_sample_id[pair["clean_sample_id"]]
        corr_row = self.graph_by_sample_id[pair["corrupted_sample_id"]]
        graph_clean = torch.load(clean_row["graph_path"], map_location="cpu", weights_only=False)
        graph_corrupted = torch.load(corr_row["graph_path"], map_location="cpu", weights_only=False)
        teacher_score_clean = float(pair["teacher_score_clean"])
        teacher_score_corrupted = float(pair["teacher_score_corrupted"])
        graph_clean.y = torch.tensor([teacher_score_clean], dtype=torch.float)
        graph_corrupted.y = torch.tensor([teacher_score_corrupted], dtype=torch.float)
        return {
            "graph_clean": graph_clean,
            "graph_corrupted": graph_corrupted,
            "teacher_score_clean": teacher_score_clean,
            "teacher_score_corrupted": teacher_score_corrupted,
            "teacher_distill_clean": self._teacher_distillation_targets(clean_row),
            "teacher_distill_corrupted": self._teacher_distillation_targets(corr_row),
            "pair_metadata": pair,
        }
