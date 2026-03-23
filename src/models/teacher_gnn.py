from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from src.models.teacher_heads import GraphScoreHead, ReconstructionHeads
from src.utils.teacher_pooling import MultiTypeMeanPooling


class NodeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TeacherGNN(nn.Module):
    def __init__(
        self,
        input_dims: Mapping[str, int],
        edge_types: Sequence[Tuple[str, str, str]],
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        residual: bool = True,
        node_types: Iterable[str] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.node_types = tuple(node_types or input_dims.keys())
        self.edge_types = list(edge_types)

        self.encoders = nn.ModuleDict(
            {
                node_type: NodeEncoder(input_dim=input_dims[node_type], hidden_dim=hidden_dim)
                for node_type in self.node_types
            }
        )
        self.encoder_norms = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types})

        self.convs = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv((-1, -1), hidden_dim)
                        for edge_type in self.edge_types
                    },
                    aggr="sum",
                )
            )
            self.conv_norms.append(nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types}))

        self.pool = MultiTypeMeanPooling(hidden_dim=hidden_dim, node_types=self.node_types)
        self.reconstruction_heads = ReconstructionHeads(hidden_dim=hidden_dim)
        self.graph_score_head = GraphScoreHead(hidden_dim=hidden_dim)

    def encode_nodes(self, batch) -> Dict[str, torch.Tensor]:
        encoded = {}
        for node_type in self.node_types:
            x = batch[node_type].x.float()
            encoded[node_type] = self.encoder_norms[node_type](self.encoders[node_type](x))
        return encoded

    def backbone(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]):
        for conv, norms in zip(self.convs, self.conv_norms):
            updated = conv(x_dict, edge_index_dict)
            next_x_dict = {}
            for node_type in self.node_types:
                node_embeddings = updated.get(node_type, x_dict[node_type])
                if self.residual and node_embeddings.shape == x_dict[node_type].shape:
                    node_embeddings = node_embeddings + x_dict[node_type]
                node_embeddings = norms[node_type](node_embeddings)
                node_embeddings = F.relu(node_embeddings)
                node_embeddings = F.dropout(node_embeddings, p=self.dropout, training=self.training)
                next_x_dict[node_type] = node_embeddings
            x_dict = next_x_dict
        return x_dict

    def _get_batch_dict(self, batch) -> Dict[str, torch.Tensor]:
        batch_dict = {}
        for node_type in self.node_types:
            node_store = batch[node_type]
            if hasattr(node_store, "batch") and node_store.batch is not None:
                batch_dict[node_type] = node_store.batch
            else:
                batch_dict[node_type] = torch.zeros(
                    node_store.x.size(0),
                    dtype=torch.long,
                    device=node_store.x.device,
                )
        return batch_dict

    def forward(self, batch):
        x_dict = self.encode_nodes(batch)
        node_embeddings = self.backbone(x_dict, batch.edge_index_dict)
        batch_dict = self._get_batch_dict(batch)
        graph_embedding, pooled_by_type = self.pool(node_embeddings=node_embeddings, batch_dict=batch_dict)
        recon_logits = self.reconstruction_heads(node_embeddings)
        graph_score = self.graph_score_head(graph_embedding)

        return {
            "node_embeddings": node_embeddings,
            "graph_embedding": graph_embedding,
            "graph_score": graph_score,
            "recon_logits": recon_logits,
            "pooled_by_type": pooled_by_type,
        }

    @classmethod
    def from_hetero_data(
        cls,
        hetero_data,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        residual: bool = True,
    ) -> "TeacherGNN":
        input_dims = {node_type: hetero_data[node_type].x.size(-1) for node_type in hetero_data.node_types}
        return cls(
            input_dims=input_dims,
            edge_types=hetero_data.edge_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            residual=residual,
            node_types=hetero_data.node_types,
        )
