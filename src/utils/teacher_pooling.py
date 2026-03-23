from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool


class MultiTypeMeanPooling(nn.Module):
    def __init__(self, hidden_dim: int, node_types: Iterable[str]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_types = tuple(node_types)
        self.proj = nn.Linear(len(self.node_types) * hidden_dim, hidden_dim)

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
                pooled = reference_tensor.new_zeros((num_graphs, self.hidden_dim))
            else:
                pooled = global_mean_pool(embeddings, batch, size=num_graphs)
            pooled_by_type[node_type] = pooled

        graph_embedding = torch.cat([pooled_by_type[node_type] for node_type in self.node_types], dim=-1)
        graph_embedding = self.proj(graph_embedding)
        return graph_embedding, pooled_by_type
