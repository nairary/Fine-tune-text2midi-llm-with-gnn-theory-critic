# src/dataloader/hooktheory_dataset.py
import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from .utils_graph import build_graph_from_encoded, corrupt_graph, mask_graph


class HookTheoryDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        mask_prob: float = 0.15,
        mask_min_nodes: int = 1,
        optional_mask_field_prob: float = 0.5,
        corruption_modes: Sequence[str] | None = None,
    ):
        self.json_path = Path(json_path)
        self.mask_prob = mask_prob
        self.mask_min_nodes = mask_min_nodes
        self.optional_mask_field_prob = optional_mask_field_prob
        self.corruption_modes = tuple(corruption_modes) if corruption_modes is not None else None

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

    def __getitem__(self, idx):
        song_obj = self.data[idx]
        graph_real = build_graph_from_encoded(song_obj)
        graph_masked, masked_labels = mask_graph(
            graph_real,
            mask_prob=self.mask_prob,
            min_nodes_to_mask=self.mask_min_nodes,
            optional_mask_field_prob=self.optional_mask_field_prob,
        )
        graph_corrupted = corrupt_graph(graph_real, corruption_modes=self.corruption_modes)

        return {
            "graph_real": graph_real,
            "graph_masked": graph_masked,
            "graph_corrupted": graph_corrupted,
            "masked_labels": masked_labels,
            "graph_score_label": 1.0,
        }


def collate_fn(batch):
    graphs_real = [item["graph_real"] for item in batch]
    graphs_masked = [item["graph_masked"] for item in batch]
    graphs_corrupted = [item["graph_corrupted"] for item in batch]
    masked_labels = [item["masked_labels"] for item in batch]
    score_labels = torch.tensor([item["graph_score_label"] for item in batch], dtype=torch.float)

    return {
        "graph_real": Batch.from_data_list(graphs_real),
        "graph_masked": Batch.from_data_list(graphs_masked),
        "graph_corrupted": Batch.from_data_list(graphs_corrupted),
        "masked_labels": masked_labels,
        "graph_score_label": score_labels,
    }
