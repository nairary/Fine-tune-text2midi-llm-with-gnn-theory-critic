from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HGTConv, HeteroConv, SAGEConv

from src.models.teacher_heads import GraphScoreHead, LocalScoreHead, ReconstructionHeads, SlotContextAttention
from src.utils.teacher_pooling import MultiTypeMeanPooling


class NodeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, encoder_hidden_dims: Sequence[int] | None = None):
        super().__init__()
        hidden_stack = list(encoder_hidden_dims or [hidden_dim])
        dims = [input_dim, *hidden_stack, hidden_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            # if out_dim != hidden_dim or (in_dim, out_dim) != (dims[-2], dims[-1]): было так, но будто первое условие избыточное
            if (in_dim, out_dim) != (dims[-2], dims[-1]):
                layers.append(nn.ReLU())
        if layers and isinstance(layers[-1], nn.ReLU):
            layers.pop()
        self.net = nn.Sequential(*layers)

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
        backbone: str = "sage",
        hgt_num_heads: int = 4,
        node_types: Iterable[str] | None = None,
        encoder_hidden_dims: Sequence[int] | None = None,
        pooling_mode: str = "mean",
        pooling_attention_hidden_dim: int | None = None,
        pooling_type_attention: bool = False,
        pooling_output_dim: int | None = None,
        score_head_hidden_dim: int | None = None,
        reconstruction_head_hidden_dim: int | None = None,
        enabled_heads: Mapping[str, bool] | None = None,
        use_note_score_head: bool = True,
        use_chord_score_head: bool = True,
        use_onset_score_head: bool = True,
        local_score_head_hidden_dim: int | None = None,
        local_context_mode: str = "mean",
        local_context_num_heads: int = 4,
        use_hybrid_graph_scorer: bool = False,
        score_fusion_mode: str = "none",
        score_fusion_hidden_dim: int | None = None,
        local_summary_use_mean: bool = True,
        local_summary_use_max: bool = True,
        local_summary_use_topk_mean: bool = False,
        local_summary_topk: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.backbone_type = str(backbone).lower()
        self.hgt_num_heads = int(hgt_num_heads)
        self.node_types = tuple(node_types or input_dims.keys())
        self.edge_types = list(edge_types)
        self.pooling_output_dim = pooling_output_dim or hidden_dim
        self.score_head_hidden_dim = score_head_hidden_dim or max(1, self.pooling_output_dim // 2)
        self.reconstruction_head_hidden_dim = reconstruction_head_hidden_dim or hidden_dim
        self.local_score_head_hidden_dim = local_score_head_hidden_dim or max(1, hidden_dim // 2)
        self.use_note_score_head = bool(use_note_score_head)
        self.use_chord_score_head = bool(use_chord_score_head)
        self.use_onset_score_head = bool(use_onset_score_head)
        self.local_context_mode = str(local_context_mode)
        self.local_context_num_heads = int(local_context_num_heads)
        self.use_hybrid_graph_scorer = bool(use_hybrid_graph_scorer)
        self.score_fusion_mode = str(score_fusion_mode or "none")
        self.local_summary_use_mean = bool(local_summary_use_mean)
        self.local_summary_use_max = bool(local_summary_use_max)
        self.local_summary_use_topk_mean = bool(local_summary_use_topk_mean)
        self.local_summary_topk = int(local_summary_topk)
        if self.local_summary_topk < 1:
            raise ValueError("local_summary_topk must be >= 1.")
        if self.backbone_type not in {"sage", "hgt"}:
            raise ValueError(f"Unsupported teacher backbone '{backbone}'. Supported values are 'sage' and 'hgt'.")
        if self.backbone_type == "hgt":
            if self.hgt_num_heads < 1:
                raise ValueError("hgt_num_heads must be >= 1.")
            if hidden_dim % self.hgt_num_heads != 0:
                raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by hgt_num_heads ({self.hgt_num_heads}).")
        if not any([self.local_summary_use_mean, self.local_summary_use_max, self.local_summary_use_topk_mean]):
            raise ValueError("At least one local score summary statistic must be enabled.")
        if self.local_context_mode not in {"mean", "attention"}:
            raise ValueError(
                f"Unsupported local_context_mode='{self.local_context_mode}'. Supported modes are 'mean' and 'attention'."
            )
        if self.score_fusion_mode not in {"none", "learned_logit_fusion"}:
            raise ValueError(
                f"Unsupported score_fusion_mode='{self.score_fusion_mode}'. Supported modes are 'none' and 'learned_logit_fusion'."
            )

        self.encoders = nn.ModuleDict(
            {
                node_type: NodeEncoder(
                    input_dim=input_dims[node_type],
                    hidden_dim=hidden_dim,
                    encoder_hidden_dims=encoder_hidden_dims,
                )
                for node_type in self.node_types
            }
        )
        self.encoder_norms = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types})

        self.convs = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        for _ in range(num_layers):
            if self.backbone_type == "sage":
                self.convs.append(
                    HeteroConv(
                        {edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in self.edge_types},
                        aggr="sum",
                    )
                )
            else:
                self.convs.append(
                    HGTConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        metadata=(list(self.node_types), list(self.edge_types)),
                        heads=self.hgt_num_heads,
                    )
                )
            self.conv_norms.append(nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types}))

        self.pool = MultiTypeMeanPooling(
            hidden_dim=hidden_dim,
            node_types=self.node_types,
            output_dim=self.pooling_output_dim,
            pooling_mode=pooling_mode,
            attention_hidden_dim=pooling_attention_hidden_dim,
            use_type_attention=bool(pooling_type_attention),
        )
        self.reconstruction_heads = ReconstructionHeads(
            hidden_dim=hidden_dim,
            head_hidden_dim=self.reconstruction_head_hidden_dim,
            enabled_heads=enabled_heads,
        )
        self.local_score_heads = nn.ModuleDict()
        self.local_context_attn = nn.ModuleDict()
        if self.use_note_score_head:
            self.local_score_heads["note"] = LocalScoreHead(hidden_dim, self.local_score_head_hidden_dim)
        if self.use_chord_score_head:
            self.local_score_heads["chord"] = LocalScoreHead(hidden_dim, self.local_score_head_hidden_dim)
        if self.use_onset_score_head:
            self.local_score_heads["onset"] = LocalScoreHead(hidden_dim, self.local_score_head_hidden_dim)
        self.active_local_head_types = tuple(node_type for node_type in ("note", "chord", "onset") if node_type in self.local_score_heads)
        if self.local_context_mode == "attention":
            self.local_context_attn = nn.ModuleDict(
                {
                    node_type: SlotContextAttention(
                        hidden_dim=hidden_dim,
                        num_heads=self.local_context_num_heads,
                        dropout=self.dropout,
                    )
                    for node_type in self.active_local_head_types
                }
            )
        self.local_summary_stats_count = sum(
            [self.local_summary_use_mean, self.local_summary_use_max, self.local_summary_use_topk_mean]
        )
        self.local_summary_dim = len(self.active_local_head_types) * self.local_summary_stats_count
        graph_score_input_dim = (
            self.pooling_output_dim
            if self.score_fusion_mode == "learned_logit_fusion"
            else self.pooling_output_dim + (self.local_summary_dim if self.use_hybrid_graph_scorer else 0)
        )
        self.graph_score_head = GraphScoreHead(
            input_dim=graph_score_input_dim,
            hidden_dim=self.score_head_hidden_dim,
        )
        self.score_fusion_head = None
        if self.score_fusion_mode == "learned_logit_fusion":
            self.score_fusion_hidden_dim = score_fusion_hidden_dim or self.score_head_hidden_dim
            self.score_fusion_head = GraphScoreHead(
                input_dim=1 + self.local_summary_dim,
                hidden_dim=self.score_fusion_hidden_dim,
            )
        else:
            self.score_fusion_hidden_dim = score_fusion_hidden_dim

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
                node_embeddings = updated.get(node_type)
                if node_embeddings is None:
                    node_embeddings = x_dict[node_type]
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

    @staticmethod
    def _build_neighbor_map(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
        neighbors = [[] for _ in range(num_nodes)]
        if edge_index.numel() == 0:
            return neighbors
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for a, b in zip(src, dst):
            if 0 <= a < num_nodes and 0 <= b < num_nodes:
                neighbors[a].append(b)
                neighbors[b].append(a)
        return neighbors

    @staticmethod
    def _gather_mean(embeddings: torch.Tensor, indices: list[int], fallback: torch.Tensor) -> torch.Tensor:
        if not indices:
            return fallback
        index_tensor = torch.tensor(indices, dtype=torch.long, device=embeddings.device)
        return embeddings.index_select(0, index_tensor).mean(dim=0)

    def _aggregate_local_context(self, node_type: str, query_embeddings: torch.Tensor, slot_tensor: torch.Tensor) -> torch.Tensor:
        if self.local_context_mode == "attention":
            return self.local_context_attn[node_type](query_embeddings, slot_tensor)
        return slot_tensor.mean(dim=1)

    def _prepare_edge_maps(self, batch) -> dict:
        edge_maps = {}
        edge_maps["note_neighbors"] = self._build_neighbor_map(
            batch[("note", "next_note", "note")].edge_index,
            batch["note"].x.size(0),
        )
        edge_maps["chord_neighbors"] = self._build_neighbor_map(
            batch[("chord", "next_chord", "chord")].edge_index,
            batch["chord"].x.size(0),
        )
        edge_maps["onset_neighbors"] = self._build_neighbor_map(
            batch[("onset", "next_onset", "onset")].edge_index,
            batch["onset"].x.size(0),
        )
        edge_maps["onset_to_notes"] = [[] for _ in range(batch["onset"].x.size(0))]
        edge_maps["onset_to_chords"] = [[] for _ in range(batch["onset"].x.size(0))]
        edge_maps["note_to_onset"] = [None for _ in range(batch["note"].x.size(0))]
        edge_maps["chord_to_onset"] = [None for _ in range(batch["chord"].x.size(0))]
        edge_maps["note_to_chords"] = [[] for _ in range(batch["note"].x.size(0))]
        edge_maps["chord_to_notes"] = [[] for _ in range(batch["chord"].x.size(0))]

        starts_note = batch[("onset", "starts_note", "note")].edge_index
        for onset_idx, note_idx in zip(starts_note[0].tolist(), starts_note[1].tolist()):
            edge_maps["onset_to_notes"][onset_idx].append(note_idx)
            edge_maps["note_to_onset"][note_idx] = onset_idx

        starts_chord = batch[("onset", "starts_chord", "chord")].edge_index
        for onset_idx, chord_idx in zip(starts_chord[0].tolist(), starts_chord[1].tolist()):
            edge_maps["onset_to_chords"][onset_idx].append(chord_idx)
            edge_maps["chord_to_onset"][chord_idx] = onset_idx

        covers_note = batch[("chord", "covers_note", "note")].edge_index
        for chord_idx, note_idx in zip(covers_note[0].tolist(), covers_note[1].tolist()):
            edge_maps["note_to_chords"][note_idx].append(chord_idx)
            edge_maps["chord_to_notes"][chord_idx].append(note_idx)
        return edge_maps

    def compute_contextual_local_scores(self, batch, node_embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.local_score_heads:
            return {}

        edge_maps = self._prepare_edge_maps(batch)
        batch_dict = self._get_batch_dict(batch)
        song_embeddings = node_embeddings["song"]
        hidden_dim = song_embeddings.size(-1)
        zero = song_embeddings.new_zeros((hidden_dim,))

        def song_context_for(node_type: str, node_idx: int) -> torch.Tensor:
            graph_idx = int(batch_dict[node_type][node_idx].item()) if batch_dict[node_type].numel() > 0 else 0
            if 0 <= graph_idx < song_embeddings.size(0):
                return song_embeddings[graph_idx]
            return zero

        contextual_scores: Dict[str, torch.Tensor] = {}

        if "note" in self.local_score_heads:
            note_slots = []
            for idx in range(node_embeddings["note"].size(0)):
                note_emb = node_embeddings["note"][idx]
                note_neighbors = self._gather_mean(node_embeddings["note"], edge_maps["note_neighbors"][idx], zero)
                onset_idx = edge_maps["note_to_onset"][idx]
                onset_emb = node_embeddings["onset"][onset_idx] if onset_idx is not None else zero
                cover_chords = self._gather_mean(node_embeddings["chord"], edge_maps["note_to_chords"][idx], zero)
                song_context = song_context_for("note", idx)
                note_slots.append(torch.stack([note_emb, note_neighbors, onset_emb, cover_chords, song_context], dim=0))
            if note_slots:
                slot_tensor = torch.stack(note_slots, dim=0)
                note_context = self._aggregate_local_context("note", node_embeddings["note"], slot_tensor)
                contextual_scores["note"] = self.local_score_heads["note"](note_context)

        if "chord" in self.local_score_heads:
            chord_slots = []
            for idx in range(node_embeddings["chord"].size(0)):
                chord_emb = node_embeddings["chord"][idx]
                chord_neighbors = self._gather_mean(node_embeddings["chord"], edge_maps["chord_neighbors"][idx], zero)
                covered_notes = self._gather_mean(node_embeddings["note"], edge_maps["chord_to_notes"][idx], zero)
                onset_idx = edge_maps["chord_to_onset"][idx]
                onset_emb = node_embeddings["onset"][onset_idx] if onset_idx is not None else zero
                song_context = song_context_for("chord", idx)
                chord_slots.append(torch.stack([chord_emb, chord_neighbors, covered_notes, onset_emb, song_context], dim=0))
            if chord_slots:
                slot_tensor = torch.stack(chord_slots, dim=0)
                chord_context = self._aggregate_local_context("chord", node_embeddings["chord"], slot_tensor)
                contextual_scores["chord"] = self.local_score_heads["chord"](chord_context)

        if "onset" in self.local_score_heads:
            onset_slots = []
            for idx in range(node_embeddings["onset"].size(0)):
                onset_emb = node_embeddings["onset"][idx]
                onset_neighbors = self._gather_mean(node_embeddings["onset"], edge_maps["onset_neighbors"][idx], zero)
                onset_notes = self._gather_mean(node_embeddings["note"], edge_maps["onset_to_notes"][idx], zero)
                onset_chords = self._gather_mean(node_embeddings["chord"], edge_maps["onset_to_chords"][idx], zero)
                song_context = song_context_for("onset", idx)
                onset_slots.append(torch.stack([onset_emb, onset_notes, onset_chords, onset_neighbors, song_context], dim=0))
            if onset_slots:
                slot_tensor = torch.stack(onset_slots, dim=0)
                onset_context = self._aggregate_local_context("onset", node_embeddings["onset"], slot_tensor)
                contextual_scores["onset"] = self.local_score_heads["onset"](onset_context)

        return contextual_scores

    def _summarize_type_scores(self, scores: torch.Tensor, batch_vector: torch.Tensor, num_graphs: int) -> torch.Tensor:
        summary_rows = []
        for graph_idx in range(num_graphs):
            graph_mask = batch_vector == graph_idx
            graph_scores = scores[graph_mask]
            stats = []
            if graph_scores.numel() == 0:
                if self.local_summary_use_mean:
                    stats.append(scores.new_zeros(()))
                if self.local_summary_use_max:
                    stats.append(scores.new_zeros(()))
                if self.local_summary_use_topk_mean:
                    stats.append(scores.new_zeros(()))
            else:
                if self.local_summary_use_mean:
                    stats.append(graph_scores.mean())
                if self.local_summary_use_max:
                    stats.append(graph_scores.max())
                if self.local_summary_use_topk_mean:
                    top_k = min(self.local_summary_topk, graph_scores.numel())
                    stats.append(torch.topk(graph_scores, k=top_k).values.mean())
            summary_rows.append(torch.stack(stats, dim=0))
        return torch.stack(summary_rows, dim=0)

    def summarize_local_scores(self, batch, local_scores: Dict[str, torch.Tensor], batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        num_graphs = self.pool._infer_num_graphs(batch_dict)
        if not self.active_local_head_types:
            song_x = batch["song"].x
            return song_x.new_zeros((num_graphs, 0))

        summaries = []
        for node_type in self.active_local_head_types:
            scores = local_scores.get(node_type)
            if scores is None:
                scores = batch[node_type].x.new_zeros((batch[node_type].x.size(0),), dtype=batch["song"].x.dtype)
            summaries.append(self._summarize_type_scores(scores, batch_dict[node_type], num_graphs))
        return torch.cat(summaries, dim=-1)

    def forward(self, batch):
        x_dict = self.encode_nodes(batch)
        node_embeddings = self.backbone(x_dict, batch.edge_index_dict)
        batch_dict = self._get_batch_dict(batch)
        local_scores = self.compute_contextual_local_scores(batch, node_embeddings)
        graph_embedding, pooled_by_type = self.pool(node_embeddings=node_embeddings, batch_dict=batch_dict)
        local_score_summaries = self.summarize_local_scores(batch, local_scores, batch_dict)
        if self.score_fusion_mode == "learned_logit_fusion":
            graph_score_features = graph_embedding
        elif self.use_hybrid_graph_scorer:
            graph_score_features = torch.cat([graph_embedding, local_score_summaries], dim=-1)
        else:
            graph_score_features = graph_embedding
        recon_logits = self.reconstruction_heads(node_embeddings)
        graph_score_base = self.graph_score_head(graph_score_features)
        graph_score_fusion_features = graph_score_features.new_zeros((graph_score_features.size(0), 0))
        if self.score_fusion_mode == "learned_logit_fusion":
            graph_score_base = self.graph_score_head(graph_embedding)
            graph_score_fusion_features = torch.cat([graph_score_base.unsqueeze(-1), local_score_summaries], dim=-1)
            graph_score = self.score_fusion_head(graph_score_fusion_features)
            graph_score_features = graph_score_fusion_features
        else:
            graph_score = graph_score_base

        return {
            "node_embeddings": node_embeddings,
            "graph_embedding": graph_embedding,
            "graph_score": graph_score,
            "graph_score_base": graph_score_base,
            "recon_logits": recon_logits,
            "local_scores": local_scores,
            "pooled_by_type": pooled_by_type,
            "local_score_summaries": local_score_summaries,
            "graph_score_features": graph_score_features,
            "graph_score_fusion_features": graph_score_fusion_features,
        }

    @classmethod
    def from_hetero_data(
        cls,
        hetero_data,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        residual: bool = True,
        backbone: str = "sage",
        hgt_num_heads: int = 4,
        encoder_hidden_dims: Sequence[int] | None = None,
        pooling_mode: str = "mean",
        pooling_attention_hidden_dim: int | None = None,
        pooling_type_attention: bool = False,
        pooling_output_dim: int | None = None,
        score_head_hidden_dim: int | None = None,
        reconstruction_head_hidden_dim: int | None = None,
        enabled_heads: Mapping[str, bool] | None = None,
        use_note_score_head: bool = True,
        use_chord_score_head: bool = True,
        use_onset_score_head: bool = True,
        local_score_head_hidden_dim: int | None = None,
        local_context_mode: str = "mean",
        local_context_num_heads: int = 4,
        use_hybrid_graph_scorer: bool = False,
        score_fusion_mode: str = "none",
        score_fusion_hidden_dim: int | None = None,
        local_summary_use_mean: bool = True,
        local_summary_use_max: bool = True,
        local_summary_use_topk_mean: bool = False,
        local_summary_topk: int = 3,
    ) -> "TeacherGNN":
        input_dims = {node_type: hetero_data[node_type].x.size(-1) for node_type in hetero_data.node_types}
        return cls(
            input_dims=input_dims,
            edge_types=hetero_data.edge_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            residual=residual,
            backbone=backbone,
            hgt_num_heads=hgt_num_heads,
            node_types=hetero_data.node_types,
            encoder_hidden_dims=encoder_hidden_dims,
            pooling_mode=pooling_mode,
            pooling_attention_hidden_dim=pooling_attention_hidden_dim,
            pooling_type_attention=pooling_type_attention,
            pooling_output_dim=pooling_output_dim,
            score_head_hidden_dim=score_head_hidden_dim,
            reconstruction_head_hidden_dim=reconstruction_head_hidden_dim,
            enabled_heads=enabled_heads,
            use_note_score_head=use_note_score_head,
            use_chord_score_head=use_chord_score_head,
            use_onset_score_head=use_onset_score_head,
            local_score_head_hidden_dim=local_score_head_hidden_dim,
            local_context_mode=local_context_mode,
            local_context_num_heads=local_context_num_heads,
            use_hybrid_graph_scorer=use_hybrid_graph_scorer,
            score_fusion_mode=score_fusion_mode,
            score_fusion_hidden_dim=score_fusion_hidden_dim,
            local_summary_use_mean=local_summary_use_mean,
            local_summary_use_max=local_summary_use_max,
            local_summary_use_topk_mean=local_summary_use_topk_mean,
            local_summary_topk=local_summary_topk,
        )
