# src/dataloader/hooktheory_dataset.py
import json
import random
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from .corruption_balancer import CorruptionModeBalancer
from .song_corruptions import corrupt_song_obj
from .theory_helpers import build_theory_context
from .utils_graph import build_graph_from_encoded, corrupt_graph, mask_graph


_SECTION_CORRUPTION_MODES = {
    "adjacent_section_swap",
    "non_adjacent_section_swap",
    "section_duplicate",
    "section_drop_keep_silence",
    "section_drop_and_close_gap",
    "section_entry_non_tonic_substitution",
    "section_exit_non_dominant_substitution",
}


def _positive_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0.0 else None


def _family_weight(config: dict, aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        if alias in config:
            return _positive_float(config.get(alias))
    return None


def _build_corruption_mode_weights(modes: Sequence[str], theory_aware_cfg: dict) -> dict[str, float] | None:
    weights: dict[str, float] = {}

    family_weights_cfg = theory_aware_cfg.get("corruption_family_weights")
    if isinstance(family_weights_cfg, dict) and family_weights_cfg:
        section_modes = [str(mode) for mode in modes if str(mode) in _SECTION_CORRUPTION_MODES]
        local_modes = [str(mode) for mode in modes if str(mode) not in _SECTION_CORRUPTION_MODES]
        section_weight = _family_weight(family_weights_cfg, ("section", "sections", "structural", "structure"))
        local_weight = _family_weight(family_weights_cfg, ("local", "theory", "harmonic", "harmony"))

        if section_modes:
            section_weight = 1.0 if section_weight is None else section_weight
            per_mode = section_weight / max(1, len(section_modes))
            weights.update({mode: per_mode for mode in section_modes})
        if local_modes:
            local_weight = 1.0 if local_weight is None else local_weight
            per_mode = local_weight / max(1, len(local_modes))
            weights.update({mode: per_mode for mode in local_modes})

    mode_weights_cfg = theory_aware_cfg.get("corruption_mode_weights")
    if isinstance(mode_weights_cfg, dict) and mode_weights_cfg:
        for mode in modes:
            weight = _positive_float(mode_weights_cfg.get(mode))
            if weight is not None:
                weights[str(mode)] = weight

    return weights or None


class HookTheoryDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        mask_prob: float = 0.15,
        mask_min_nodes: int = 1,
        optional_mask_field_prob: float = 0.5,
        corruption_modes: Sequence[str] | None = None,
        corruption_backend: str = "graph",
        theory_aware_cfg: dict | None = None,
    ):
        self.json_path = Path(json_path)
        self.mask_prob = mask_prob
        self.mask_min_nodes = mask_min_nodes
        self.optional_mask_field_prob = optional_mask_field_prob
        self.corruption_modes = tuple(corruption_modes) if corruption_modes is not None else None
        self.corruption_backend = corruption_backend
        self.theory_aware_cfg = theory_aware_cfg or {}
        self.deterministic_per_sample = bool(self.theory_aware_cfg.get("deterministic_per_sample", False))
        self.balance_mode_usage = (
            self.corruption_backend == "song_theory"
            and bool(self.theory_aware_cfg.get("balance_mode_usage", False))
            and not self.deterministic_per_sample
            and self.corruption_modes is not None
            and len(self.corruption_modes) > 1
        )
        mode_weights = (
            _build_corruption_mode_weights(self.corruption_modes or (), self.theory_aware_cfg)
            if self.balance_mode_usage
            else None
        )
        self.corruption_mode_balancer = (
            CorruptionModeBalancer(self.corruption_modes or (), mode_weights=mode_weights) if self.balance_mode_usage else None
        )
        self.theory_ctx = build_theory_context() if self.corruption_backend == "song_theory" else None
        self.return_masked_graph = True
        self.return_corrupted_graph = True

        with open(self.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            self.data = list(raw.values())
        elif isinstance(raw, list):
            self.data = raw
        else:
            raise ValueError("Unsupported JSON format: expected dict or list")

    def __len__(self):
        return len(self.data)

    def set_stage_outputs(self, *, masked: bool = True, corrupted: bool = True) -> None:
        self.return_masked_graph = bool(masked)
        self.return_corrupted_graph = bool(corrupted)

    def __getitem__(self, idx):
        song_obj = self.data[idx]
        graph_real = build_graph_from_encoded(song_obj)
        if self.return_masked_graph:
            graph_masked, masked_labels = mask_graph(
                graph_real,
                mask_prob=self.mask_prob,
                min_nodes_to_mask=self.mask_min_nodes,
                optional_mask_field_prob=self.optional_mask_field_prob,
            )
        else:
            graph_masked, masked_labels = None, {}

        if not self.return_corrupted_graph:
            graph_corrupted = None
            corruption_metadata = None
        elif self.corruption_backend == "song_theory":
            per_sample_rng = random
            if self.deterministic_per_sample:
                base_seed = int(self.theory_aware_cfg.get("deterministic_seed", 0))
                per_sample_rng = random.Random(base_seed + int(idx))
            corruption_modes = self.corruption_modes
            shuffle_modes = True
            if self.corruption_mode_balancer is not None:
                corruption_modes = self.corruption_mode_balancer.ordered_modes(per_sample_rng)
                shuffle_modes = False
            song_corrupted, corruption_metadata = corrupt_song_obj(
                song_obj,
                corruption_modes=corruption_modes,
                corruption_cfg=self.theory_aware_cfg,
                theory_ctx=self.theory_ctx,
                rng=per_sample_rng,
                shuffle_modes=shuffle_modes,
            )
            if self.corruption_mode_balancer is not None and bool((corruption_metadata or {}).get("applied", False)):
                self.corruption_mode_balancer.record_applied(str(corruption_metadata.get("corruption_name", "")))
            graph_corrupted = build_graph_from_encoded(song_corrupted)
        else:
            graph_corrupted = corrupt_graph(graph_real, corruption_modes=self.corruption_modes)
            corruption_metadata = getattr(graph_corrupted, "corruption_metadata", None)

        return {
            "graph_real": graph_real,
            "graph_masked": graph_masked,
            "graph_corrupted": graph_corrupted,
            "masked_labels": masked_labels,
            "corruption_metadata": corruption_metadata,
            "graph_score_label": 1.0,
        }

    def get_corruption_usage_counts(self) -> dict[str, int]:
        if self.corruption_mode_balancer is None:
            return {}
        return self.corruption_mode_balancer.usage_counts()


def _batch_graphs(graphs):
    if all(graph is None for graph in graphs):
        return None
    if any(graph is None for graph in graphs):
        raise ValueError("Cannot collate a batch with mixed present/missing graph objects.")
    return Batch.from_data_list(graphs)


def collate_fn(batch):
    graphs_real = [item["graph_real"] for item in batch]
    graphs_masked = [item["graph_masked"] for item in batch]
    graphs_corrupted = [item["graph_corrupted"] for item in batch]
    masked_labels = [item["masked_labels"] for item in batch]
    corruption_metadata = [item.get("corruption_metadata") for item in batch]
    score_labels = torch.tensor([item["graph_score_label"] for item in batch], dtype=torch.float)

    return {
        "graph_real": Batch.from_data_list(graphs_real),
        "graph_masked": _batch_graphs(graphs_masked),
        "graph_corrupted": _batch_graphs(graphs_corrupted),
        "masked_labels": masked_labels,
        "corruption_metadata": corruption_metadata,
        "graph_score_label": score_labels,
    }
