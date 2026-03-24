from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn

from src.dataloader.graph_layouts import VALID_ID_SETS


RECONSTRUCTION_SPECS = {
    "note_sd": {
        "node_type": "note",
        "field_name": "sd_id",
        "valid_ids": VALID_ID_SETS["note_sd_id"],
        "default_loss_weight": 1.0,
    },
    "chord_root": {
        "node_type": "chord",
        "field_name": "root_id",
        "valid_ids": VALID_ID_SETS["chord_root_id"],
        "default_loss_weight": 1.0,
    },
    "chord_type": {
        "node_type": "chord",
        "field_name": "type_id",
        "valid_ids": VALID_ID_SETS["chord_type_id"],
        "default_loss_weight": 1.0,
    },
    "chord_applied": {
        "node_type": "chord",
        "field_name": "applied_id",
        "valid_ids": VALID_ID_SETS["chord_applied_id"],
        "default_loss_weight": 0.5,
    },
    "chord_borrowed_kind": {
        "node_type": "chord",
        "field_name": "borrowed_kind_id",
        "valid_ids": VALID_ID_SETS["chord_borrowed_kind_id"],
        "default_loss_weight": 0.25,
    },
}


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphScoreHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LocalScoreHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ReconstructionHeads(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        head_hidden_dim: int,
        enabled_heads: Mapping[str, bool] | None = None,
    ):
        super().__init__()
        self.enabled_heads = {
            head_name: bool(enabled_heads.get(head_name, True)) if enabled_heads is not None else True
            for head_name in RECONSTRUCTION_SPECS
        }
        self.heads = nn.ModuleDict(
            {
                head_name: MLPHead(hidden_dim, head_hidden_dim, len(spec["valid_ids"]))
                for head_name, spec in RECONSTRUCTION_SPECS.items()
                if self.enabled_heads[head_name]
            }
        )

    def forward(self, node_embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        note_embeddings = node_embeddings.get("note")
        chord_embeddings = node_embeddings.get("chord")
        if note_embeddings is None or chord_embeddings is None:
            raise KeyError("Expected note and chord embeddings to be present for reconstruction heads.")

        outputs: Dict[str, torch.Tensor] = {}
        if "note_sd" in self.heads:
            outputs["note_sd"] = self.heads["note_sd"](note_embeddings)
        if "chord_root" in self.heads:
            outputs["chord_root"] = self.heads["chord_root"](chord_embeddings)
        if "chord_type" in self.heads:
            outputs["chord_type"] = self.heads["chord_type"](chord_embeddings)
        if "chord_applied" in self.heads:
            outputs["chord_applied"] = self.heads["chord_applied"](chord_embeddings)
        if "chord_borrowed_kind" in self.heads:
            outputs["chord_borrowed_kind"] = self.heads["chord_borrowed_kind"](chord_embeddings)
        return outputs
