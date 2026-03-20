# src/dataloader/hooktheory_dataset.py
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from .utils_graph import (
    build_graph_from_encoded,
    mask_graph,
    corrupt_graph,
)


class HookTheoryDataset(Dataset):
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)

        with open(self.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # teacher_encoded.json у тебя top-level dict: song_id -> object
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

        g_real = build_graph_from_encoded(song_obj)
        g_masked, masked_labels = mask_graph(g_real, mask_prob=0.15)
        g_corrupted = corrupt_graph(g_real)

        return {
            "graph_real": g_real,
            "graph_masked": g_masked,
            "graph_corrupted": g_corrupted,
            "masked_labels": masked_labels,
            "graph_score_label": 1.0,
        }


def collate_fn(batch):
    graphs_real = [x["graph_real"] for x in batch]
    graphs_masked = [x["graph_masked"] for x in batch]
    graphs_corrupted = [x["graph_corrupted"] for x in batch]
    masked_labels = [x["masked_labels"] for x in batch]
    score_labels = torch.tensor(
        [x["graph_score_label"] for x in batch],
        dtype=torch.float
    )

    return {
        "graph_real": Batch.from_data_list(graphs_real),
        "graph_masked": Batch.from_data_list(graphs_masked),
        "graph_corrupted": Batch.from_data_list(graphs_corrupted),
        "masked_labels": masked_labels,
        "graph_score_label": score_labels,
    }


if __name__ == "__main__":
    dataset = HookTheoryDataset("data/HTCanon/HK_processed/encoded_full/teacher_encoded.json")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    for batch in loader:
        print(batch["graph_real"])
        break