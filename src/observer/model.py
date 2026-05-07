from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from src.observer.schema import OBSERVER_NODE_TYPES
from src.utils.teacher_pooling import MultiTypeMeanPooling


class ObserverNodeFeaturizer(nn.Module):
    def __init__(
        self,
        cat_vocab_sizes: Mapping[str, Sequence[int]],
        num_dims: Mapping[str, int],
        cat_embedding_dim: int,
        hidden_dim: int,
        encoder_hidden_dims: Sequence[int] | None = None,
    ):
        super().__init__()
        self.node_types = tuple(OBSERVER_NODE_TYPES)
        self.cat_embeddings = nn.ModuleDict()
        self.input_mlps = nn.ModuleDict()
        self.norms = nn.ModuleDict()

        for node_type in self.node_types:
            emb_list = nn.ModuleList([nn.Embedding(max(1, int(v)), cat_embedding_dim) for v in cat_vocab_sizes[node_type]])
            self.cat_embeddings[node_type] = emb_list
            cat_dim = len(emb_list) * cat_embedding_dim
            input_dim = cat_dim + int(num_dims[node_type])
            hidden_stack = list(encoder_hidden_dims or [hidden_dim])
            dims = [input_dim, *hidden_stack, hidden_dim]
            layers = []
            for in_dim, out_dim in zip(dims[:-1], dims[1:]):
                layers.append(nn.Linear(in_dim, out_dim))
                if (in_dim, out_dim) != (dims[-2], dims[-1]):
                    layers.append(nn.ReLU())
            if layers and isinstance(layers[-1], nn.ReLU):
                layers.pop()
            self.input_mlps[node_type] = nn.Sequential(*layers)
            self.norms[node_type] = nn.LayerNorm(hidden_dim)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        encoded: Dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            num_x = batch[node_type].x_num.float()
            cat_x = batch[node_type].x_cat.long()
            cat_embeds = []
            for idx, embedding in enumerate(self.cat_embeddings[node_type]):
                cat_embeds.append(embedding(cat_x[:, idx]))
            parts = cat_embeds + [num_x] if cat_embeds else [num_x]
            merged = torch.cat(parts, dim=-1)
            encoded[node_type] = self.norms[node_type](self.input_mlps[node_type](merged))
        return encoded


class ObserverGNN(nn.Module):
    """Teacher-style hetero GNN that outputs one scalar per song graph."""

    def __init__(
        self,
        cat_vocab_sizes: Mapping[str, Sequence[int]],
        num_feature_dims: Mapping[str, int],
        edge_types: Sequence[Tuple[str, str, str]],
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        residual: bool = True,
        cat_embedding_dim: int = 16,
        encoder_hidden_dims: Sequence[int] | None = None,
        pooling_mode: str = "mean",
        pooling_output_dim: int | None = None,
        score_head_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.node_types = tuple(OBSERVER_NODE_TYPES)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.residual = bool(residual)

        self.featurizer = ObserverNodeFeaturizer(
            cat_vocab_sizes=cat_vocab_sizes,
            num_dims=num_feature_dims,
            cat_embedding_dim=int(cat_embedding_dim),
            hidden_dim=self.hidden_dim,
            encoder_hidden_dims=encoder_hidden_dims,
        )

        self.convs = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.convs.append(HeteroConv({edge_type: SAGEConv((-1, -1), self.hidden_dim) for edge_type in edge_types}, aggr="sum"))
            self.conv_norms.append(nn.ModuleDict({node_type: nn.LayerNorm(self.hidden_dim) for node_type in self.node_types}))

        pool_out_dim = pooling_output_dim or self.hidden_dim
        self.pooling_output_dim = int(pool_out_dim)
        self.pool = MultiTypeMeanPooling(
            hidden_dim=self.hidden_dim,
            node_types=self.node_types,
            output_dim=pool_out_dim,
            pooling_mode=pooling_mode,
        )
        score_hidden = score_head_hidden_dim or max(1, pool_out_dim // 2)
        self.graph_head = nn.Sequential(
            nn.Linear(pool_out_dim, score_hidden),
            nn.ReLU(),
            nn.Linear(score_hidden, 1),
        )

    def backbone(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]):
        for conv, norms in zip(self.convs, self.conv_norms):
            updated = conv(x_dict, edge_index_dict)
            next_x = {}
            for node_type in self.node_types:
                node_embeddings = updated.get(node_type, x_dict[node_type])
                if self.residual and node_embeddings.shape == x_dict[node_type].shape:
                    node_embeddings = node_embeddings + x_dict[node_type]
                node_embeddings = norms[node_type](node_embeddings)
                node_embeddings = F.relu(node_embeddings)
                node_embeddings = F.dropout(node_embeddings, p=self.dropout, training=self.training)
                next_x[node_type] = node_embeddings
            x_dict = next_x
        return x_dict

    def _get_batch_dict(self, batch) -> Dict[str, torch.Tensor]:
        out = {}
        for node_type in self.node_types:
            node_store = batch[node_type]
            if hasattr(node_store, "batch") and node_store.batch is not None:
                out[node_type] = node_store.batch
            else:
                out[node_type] = torch.zeros(node_store.num_nodes, dtype=torch.long, device=node_store.x.device)
        return out

    def forward(self, batch, *, return_outputs: bool = False):
        x_dict = self.featurizer(batch)
        x_dict = self.backbone(x_dict, batch.edge_index_dict)
        batch_dict = self._get_batch_dict(batch)
        graph_embedding, pooled_by_type = self.pool(x_dict, batch_dict)
        score = self.graph_head(graph_embedding).squeeze(-1)
        if not return_outputs:
            return score
        return {
            "score": score,
            "graph_embedding": graph_embedding,
            "pooled_by_type": pooled_by_type,
        }
