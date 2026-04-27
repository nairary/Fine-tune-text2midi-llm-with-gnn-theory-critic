from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax


class _GatedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, attention_hidden_dim: int | None = None):
        super().__init__()
        gate_hidden_dim = int(attention_hidden_dim or hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.Tanh(),
            nn.Linear(gate_hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        logits = self.gate(embeddings).squeeze(-1)
        weights = softmax(logits, batch, num_nodes=num_graphs)
        return global_add_pool(weights.unsqueeze(-1) * embeddings, batch, size=num_graphs)


class MultiTypeMeanPooling(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        node_types: Iterable[str],
        output_dim: int | None = None,
        pooling_mode: str = "mean",
        attention_hidden_dim: int | None = None,
        use_type_attention: bool = False,
    ):
        super().__init__()
        if pooling_mode not in {"mean", "mean_max", "attention"}:
            raise ValueError(
                f"Unsupported pooling_mode='{pooling_mode}'. Supported modes are 'mean', 'mean_max', and 'attention'."
            )
        self.hidden_dim = hidden_dim
        self.node_types = tuple(node_types)
        self.pooling_mode = pooling_mode
        self.output_dim = output_dim or hidden_dim
        self.use_type_attention = bool(use_type_attention)
        self.attention_hidden_dim = int(attention_hidden_dim or hidden_dim)
        self.per_type_dim = hidden_dim if pooling_mode in {"mean", "attention"} else 2 * hidden_dim
        self.attention_pools = nn.ModuleDict()
        if self.pooling_mode == "attention":
            self.attention_pools = nn.ModuleDict(
                {
                    node_type: _GatedAttentionPool(hidden_dim=hidden_dim, attention_hidden_dim=self.attention_hidden_dim)
                    for node_type in self.node_types
                }
            )
        self.type_gate = None
        if self.use_type_attention:
            self.type_gate = nn.Sequential(
                nn.Linear(self.per_type_dim, self.attention_hidden_dim),
                nn.Tanh(),
                nn.Linear(self.attention_hidden_dim, 1),
            )
        self.proj = nn.Linear(len(self.node_types) * self.per_type_dim, self.output_dim)

    def _infer_num_graphs(self, batch_dict: Dict[str, torch.Tensor]) -> int:
        max_graph_index = -1
        for batch in batch_dict.values():
            if batch.numel() > 0:
                max_graph_index = max(max_graph_index, int(batch.max().item()))
        return max_graph_index + 1 if max_graph_index >= 0 else 1

    def forward(
        self,
        node_embeddings: Dict[str, torch.Tensor],
        batch_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        num_graphs = self._infer_num_graphs(batch_dict)
        pooled_by_type = {}

        reference_tensor = next(iter(node_embeddings.values()))
        for node_type in self.node_types:
            embeddings = node_embeddings[node_type]
            batch = batch_dict[node_type]
            if embeddings.size(0) == 0:
                pooled = reference_tensor.new_zeros((num_graphs, self.per_type_dim))
            else:
                if self.pooling_mode == "attention":
                    pooled = self.attention_pools[node_type](embeddings, batch, num_graphs)
                else:
                    pooled_mean = global_mean_pool(embeddings, batch, size=num_graphs)
                    if self.pooling_mode == "mean":
                        pooled = pooled_mean
                    else:
                        pooled_max = global_max_pool(embeddings, batch, size=num_graphs)
                        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
            pooled_by_type[node_type] = pooled

        if self.type_gate is not None:
            type_stack = torch.stack([pooled_by_type[node_type] for node_type in self.node_types], dim=1)
            type_logits = self.type_gate(type_stack).squeeze(-1)
            type_weights = torch.softmax(type_logits, dim=1).unsqueeze(-1)
            type_stack = type_stack * type_weights
            pooled_by_type = {
                node_type: type_stack[:, idx, :]
                for idx, node_type in enumerate(self.node_types)
            }

        graph_embedding = torch.cat([pooled_by_type[node_type] for node_type in self.node_types], dim=-1)
        graph_embedding = self.proj(graph_embedding)
        return graph_embedding, pooled_by_type
