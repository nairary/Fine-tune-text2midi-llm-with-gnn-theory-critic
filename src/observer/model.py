from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from src.observer.schema import OBSERVER_NODE_TYPES
from src.utils.teacher_pooling import MultiTypeMeanPooling


def _make_activation(name: str) -> nn.Module:
    name = str(name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError("score_head_activation must be one of: relu, leaky_relu, gelu, silu.")


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


class BarSequenceTransformer(nn.Module):
    """Encode the ordered bar sequence so the observer can model long-form structure."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int | None = None,
        dropout: float = 0.1,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by bar_transformer_num_heads ({num_heads}).")
        pooling = str(pooling or "cls").lower()
        if pooling not in {"cls", "mean"}:
            raise ValueError("bar_transformer_pooling must be either 'cls' or 'mean'.")

        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if pooling == "cls" else None
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(ff_dim or 4 * hidden_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.output_norm = nn.LayerNorm(hidden_dim)

    def _sinusoidal_positions(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / float(self.hidden_dim))
        )
        pe = torch.zeros(length, self.hidden_dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(positions * div_term)
        if self.hidden_dim > 1:
            pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].size(1)])
        return pe

    def _pack_bars(
        self,
        bar_embeddings: torch.Tensor,
        bar_batch: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_graphs = int(num_graphs)
        hidden_dim = int(bar_embeddings.size(-1))
        if bar_embeddings.numel() == 0:
            empty = bar_embeddings.new_zeros((num_graphs, 1, hidden_dim))
            mask = torch.ones((num_graphs, 1), dtype=torch.bool, device=bar_embeddings.device)
            counts = torch.zeros((num_graphs,), dtype=torch.long, device=bar_embeddings.device)
            return empty, mask, counts

        bar_batch = bar_batch.to(device=bar_embeddings.device, dtype=torch.long).view(-1)
        counts = torch.bincount(bar_batch.clamp_min(0), minlength=num_graphs)[:num_graphs]
        max_bars = max(1, int(counts.max().item()))
        packed = bar_embeddings.new_zeros((num_graphs, max_bars, hidden_dim))
        mask = torch.ones((num_graphs, max_bars), dtype=torch.bool, device=bar_embeddings.device)

        for graph_idx in range(num_graphs):
            indices = torch.nonzero(bar_batch == graph_idx, as_tuple=False).view(-1)
            if indices.numel() == 0:
                continue
            graph_bars = bar_embeddings.index_select(0, indices)
            packed[graph_idx, : graph_bars.size(0)] = graph_bars
            mask[graph_idx, : graph_bars.size(0)] = False
        return packed, mask, counts

    def forward(self, bar_embeddings: torch.Tensor, bar_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        packed, mask, counts = self._pack_bars(bar_embeddings, bar_batch, num_graphs)
        bar_positions = self._sinusoidal_positions(packed.size(1), packed.device, packed.dtype)
        packed = packed + bar_positions.unsqueeze(0)

        if self.cls_token is not None:
            cls = self.cls_token.to(dtype=packed.dtype).expand(packed.size(0), -1, -1)
            packed = torch.cat([cls, packed], dim=1)
            cls_mask = torch.zeros((mask.size(0), 1), dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        encoded = self.encoder(packed, src_key_padding_mask=mask)
        if self.cls_token is not None:
            out = encoded[:, 0]
        else:
            valid = (~mask).to(dtype=encoded.dtype).unsqueeze(-1)
            out = (encoded * valid).sum(dim=1) / counts.clamp_min(1).to(dtype=encoded.dtype).unsqueeze(-1)
        return self.output_norm(out)


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
        use_bar_sequence_transformer: bool = False,
        bar_transformer_num_layers: int = 2,
        bar_transformer_num_heads: int = 4,
        bar_transformer_ff_dim: int | None = None,
        bar_transformer_dropout: float | None = None,
        bar_transformer_pooling: str = "cls",
        bar_transformer_combine: str = "concat",
        score_head_activation: str = "relu",
        score_head_layer_norm: bool = False,
    ):
        super().__init__()
        self.node_types = tuple(OBSERVER_NODE_TYPES)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.residual = bool(residual)
        self.use_bar_sequence_transformer = bool(use_bar_sequence_transformer)
        self.bar_transformer_combine = str(bar_transformer_combine or "concat").lower()
        if self.bar_transformer_combine not in {"concat", "replace"}:
            raise ValueError("bar_transformer_combine must be either 'concat' or 'replace'.")

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
        self.pool_per_type_dim = int(self.pool.per_type_dim)
        self.bar_sequence_encoder = None
        graph_dim = int(pool_out_dim)
        if self.use_bar_sequence_transformer:
            self.bar_sequence_encoder = BarSequenceTransformer(
                hidden_dim=self.hidden_dim,
                num_layers=int(bar_transformer_num_layers),
                num_heads=int(bar_transformer_num_heads),
                ff_dim=None if bar_transformer_ff_dim is None else int(bar_transformer_ff_dim),
                dropout=float(self.dropout if bar_transformer_dropout is None else bar_transformer_dropout),
                pooling=str(bar_transformer_pooling),
            )
            graph_dim = self.hidden_dim if self.bar_transformer_combine == "replace" else int(pool_out_dim) + self.hidden_dim
        self.pooling_output_dim = int(graph_dim)
        score_hidden = score_head_hidden_dim or max(1, graph_dim // 2)
        score_head_layers: list[nn.Module] = []
        if bool(score_head_layer_norm):
            score_head_layers.append(nn.LayerNorm(graph_dim))
        score_head_layers.extend(
            [
                nn.Linear(graph_dim, score_hidden),
                _make_activation(score_head_activation),
                nn.Linear(score_hidden, 1),
            ]
        )
        self.graph_head = nn.Sequential(*score_head_layers)

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

    @staticmethod
    def _infer_num_graphs(batch_dict: Dict[str, torch.Tensor]) -> int:
        max_graph_index = -1
        for batch in batch_dict.values():
            if batch.numel() > 0:
                max_graph_index = max(max_graph_index, int(batch.max().item()))
        return max_graph_index + 1 if max_graph_index >= 0 else 1

    def forward(self, batch, *, return_outputs: bool = False):
        x_dict = self.featurizer(batch)
        x_dict = self.backbone(x_dict, batch.edge_index_dict)
        batch_dict = self._get_batch_dict(batch)
        pooled_embedding, pooled_by_type = self.pool(x_dict, batch_dict)
        graph_embedding = pooled_embedding
        if self.bar_sequence_encoder is not None:
            bar_embedding = self.bar_sequence_encoder(
                x_dict["bar"],
                batch_dict["bar"],
                num_graphs=self._infer_num_graphs(batch_dict),
            )
            if self.bar_transformer_combine == "replace":
                graph_embedding = bar_embedding
            else:
                graph_embedding = torch.cat([pooled_embedding, bar_embedding], dim=-1)
        score = self.graph_head(graph_embedding).squeeze(-1)
        if not return_outputs:
            return score
        return {
            "score": score,
            "graph_embedding": graph_embedding,
            "pooled_by_type": pooled_by_type,
        }
