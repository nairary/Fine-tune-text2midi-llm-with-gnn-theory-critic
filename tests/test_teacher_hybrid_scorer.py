from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch, HeteroData

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.teacher_gnn import TeacherGNN


def _empty_edge() -> torch.Tensor:
    return torch.empty((2, 0), dtype=torch.long)


def _chain_edges(num_nodes: int) -> torch.Tensor:
    if num_nodes <= 1:
        return _empty_edge()
    src = torch.arange(0, num_nodes - 1, dtype=torch.long)
    dst = torch.arange(1, num_nodes, dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def _build_graph(note_count: int = 3, chord_count: int = 2, onset_count: int = 2) -> HeteroData:
    data = HeteroData()
    data["song"].x = torch.randn(1, 4)
    data["bar"].x = torch.randn(1, 3)
    data["onset"].x = torch.randn(onset_count, 5)
    data["note"].x = torch.randn(note_count, 6)
    data["chord"].x = torch.randn(chord_count, 7)

    data[("song", "contains_bar", "bar")].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    if onset_count > 0:
        onset_ids = torch.arange(onset_count, dtype=torch.long)
        data[("bar", "contains_onset", "onset")].edge_index = torch.stack([torch.zeros(onset_count, dtype=torch.long), onset_ids], dim=0)
    else:
        data[("bar", "contains_onset", "onset")].edge_index = _empty_edge()

    data[("onset", "next_onset", "onset")].edge_index = _chain_edges(onset_count)
    data[("note", "next_note", "note")].edge_index = _chain_edges(note_count)
    data[("chord", "next_chord", "chord")].edge_index = _chain_edges(chord_count)

    if onset_count > 0 and note_count > 0:
        note_targets = torch.arange(note_count, dtype=torch.long)
        onset_sources = note_targets % onset_count
        data[("onset", "starts_note", "note")].edge_index = torch.stack([onset_sources, note_targets], dim=0)
    else:
        data[("onset", "starts_note", "note")].edge_index = _empty_edge()

    if onset_count > 0 and chord_count > 0:
        chord_targets = torch.arange(chord_count, dtype=torch.long)
        onset_sources = chord_targets % onset_count
        data[("onset", "starts_chord", "chord")].edge_index = torch.stack([onset_sources, chord_targets], dim=0)
    else:
        data[("onset", "starts_chord", "chord")].edge_index = _empty_edge()

    if chord_count > 0 and note_count > 0:
        note_targets = torch.arange(note_count, dtype=torch.long)
        chord_sources = note_targets % chord_count
        data[("chord", "covers_note", "note")].edge_index = torch.stack([chord_sources, note_targets], dim=0)
    else:
        data[("chord", "covers_note", "note")].edge_index = _empty_edge()

    return data


def _build_model(
    sample_graph: HeteroData,
    pooling_mode: str = "mean",
    pooling_type_attention: bool = False,
    use_hybrid_graph_scorer: bool = True,
    local_context_mode: str = "mean",
    local_summary_use_topk_mean: bool = False,
    backbone: str = "sage",
    score_fusion_mode: str = "none",
) -> TeacherGNN:
    return TeacherGNN.from_hetero_data(
        sample_graph,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        residual=True,
        backbone=backbone,
        hgt_num_heads=4,
        pooling_mode=pooling_mode,
        pooling_type_attention=pooling_type_attention,
        pooling_output_dim=20,
        score_head_hidden_dim=10,
        local_context_mode=local_context_mode,
        local_context_num_heads=4,
        use_hybrid_graph_scorer=use_hybrid_graph_scorer,
        score_fusion_mode=score_fusion_mode,
        score_fusion_hidden_dim=8,
        local_summary_use_mean=True,
        local_summary_use_max=True,
        local_summary_use_topk_mean=local_summary_use_topk_mean,
        local_summary_topk=3,
    )


def test_forward_shapes_mean_pooling():
    batch = Batch.from_data_list([_build_graph(), _build_graph()])
    model = _build_model(batch, pooling_mode="mean", use_hybrid_graph_scorer=True)

    outputs = model(batch)

    assert outputs["graph_embedding"].shape == (2, model.pooling_output_dim)
    assert outputs["graph_score"].shape == (2,)
    assert outputs["local_score_summaries"].shape == (2, model.local_summary_dim)
    assert outputs["graph_score_features"].shape == (2, model.pooling_output_dim + model.local_summary_dim)


def test_hgt_forward_shapes_match_teacher_output_contract():
    batch = Batch.from_data_list([_build_graph(), _build_graph()])
    model = _build_model(batch, pooling_mode="mean_max", use_hybrid_graph_scorer=True, backbone="hgt")

    outputs = model(batch)

    assert model.backbone_type == "hgt"
    assert outputs["graph_embedding"].shape == (2, model.pooling_output_dim)
    assert outputs["graph_score"].shape == (2,)
    assert set(outputs["local_scores"]) == {"note", "chord", "onset"}
    assert outputs["recon_logits"]["note_sd"].shape[0] == batch["note"].x.size(0)
    assert outputs["recon_logits"]["chord_root"].shape[0] == batch["chord"].x.size(0)
    assert outputs["local_score_summaries"].shape == (2, model.local_summary_dim)
    assert outputs["graph_score_features"].shape == (2, model.pooling_output_dim + model.local_summary_dim)


def test_hgt_with_learned_logit_fusion_keeps_output_contract():
    batch = Batch.from_data_list([_build_graph(), _build_graph()])
    model = _build_model(batch, pooling_mode="mean_max", backbone="hgt", score_fusion_mode="learned_logit_fusion")

    outputs = model(batch)

    assert model.backbone_type == "hgt"
    assert model.score_fusion_mode == "learned_logit_fusion"
    assert outputs["graph_score"].shape == (2,)
    assert outputs["graph_score_base"].shape == (2,)
    assert outputs["graph_score_fusion_features"].shape == (2, 1 + model.local_summary_dim)


def test_hgt_rejects_hidden_dim_not_divisible_by_heads():
    batch = Batch.from_data_list([_build_graph()])

    try:
        TeacherGNN.from_hetero_data(batch, hidden_dim=10, backbone="hgt", hgt_num_heads=4)
    except ValueError as exc:
        assert "must be divisible by hgt_num_heads" in str(exc)
    else:
        raise AssertionError("Expected HGT hidden-dim validation to fail.")


def test_forward_shapes_mean_max_pooling():
    batch = Batch.from_data_list([_build_graph(), _build_graph()])
    model = _build_model(batch, pooling_mode="mean_max", use_hybrid_graph_scorer=True)

    outputs = model(batch)

    assert model.pool.per_type_dim == 2 * model.hidden_dim
    for pooled in outputs["pooled_by_type"].values():
        assert pooled.shape == (2, 2 * model.hidden_dim)
    assert outputs["graph_embedding"].shape == (2, model.pooling_output_dim)


def test_forward_shapes_attention_pooling_with_type_attention():
    batch = Batch.from_data_list([_build_graph(), _build_graph()])
    model = _build_model(
        batch,
        pooling_mode="attention",
        pooling_type_attention=True,
        use_hybrid_graph_scorer=True,
    )

    outputs = model(batch)

    assert model.pool.per_type_dim == model.hidden_dim
    for pooled in outputs["pooled_by_type"].values():
        assert pooled.shape == (2, model.hidden_dim)
    assert outputs["graph_embedding"].shape == (2, model.pooling_output_dim)


def test_hybrid_off_uses_graph_embedding_only():
    batch = Batch.from_data_list([_build_graph(), _build_graph()])
    model = _build_model(batch, pooling_mode="mean", use_hybrid_graph_scorer=False)

    outputs = model(batch)

    assert outputs["graph_score_features"].shape == outputs["graph_embedding"].shape
    assert torch.allclose(outputs["graph_score_features"], outputs["graph_embedding"])


def test_hybrid_on_expands_graph_score_features():
    batch = Batch.from_data_list([_build_graph(), _build_graph()])
    model = _build_model(
        batch,
        pooling_mode="mean_max",
        use_hybrid_graph_scorer=True,
        local_summary_use_topk_mean=True,
    )

    outputs = model(batch)

    expected_dim = model.pooling_output_dim + model.local_summary_dim
    assert outputs["graph_score_features"].shape == (2, expected_dim)


def test_score_fusion_disabled_exposes_final_score_as_base_score():
    batch = Batch.from_data_list([_build_graph(), _build_graph()])
    model = _build_model(batch, pooling_mode="mean", use_hybrid_graph_scorer=False, score_fusion_mode="none")

    outputs = model(batch)

    assert outputs["graph_score"].shape == (2,)
    assert torch.allclose(outputs["graph_score"], outputs["graph_score_base"])
    assert outputs["graph_score_fusion_features"].shape == (2, 0)
    assert outputs["graph_score_features"].shape == outputs["graph_embedding"].shape


def test_learned_logit_fusion_uses_base_logit_and_local_summary_features():
    batch = Batch.from_data_list([_build_graph(), _build_graph()])
    model = _build_model(
        batch,
        pooling_mode="mean_max",
        use_hybrid_graph_scorer=True,
        local_summary_use_topk_mean=True,
        score_fusion_mode="learned_logit_fusion",
    )

    outputs = model(batch)

    assert outputs["graph_score"].shape == (2,)
    assert outputs["graph_score_base"].shape == (2,)
    assert outputs["graph_score_fusion_features"].shape == (2, 1 + model.local_summary_dim)
    assert outputs["graph_score_features"].shape == outputs["graph_score_fusion_features"].shape
    assert torch.allclose(outputs["graph_score_fusion_features"][:, 0], outputs["graph_score_base"])
    assert torch.allclose(outputs["graph_score_fusion_features"][:, 1:], outputs["local_score_summaries"])


def test_local_context_attention_emits_local_scores_and_hybrid_features():
    batch = Batch.from_data_list([_build_graph(), _build_graph()])
    model = _build_model(
        batch,
        pooling_mode="attention",
        use_hybrid_graph_scorer=True,
        local_context_mode="attention",
        local_summary_use_topk_mean=True,
    )

    outputs = model(batch)

    assert set(outputs["local_scores"]) == {"note", "chord", "onset"}
    assert outputs["local_scores"]["note"].shape[0] == batch["note"].x.size(0)
    assert outputs["local_scores"]["chord"].shape[0] == batch["chord"].x.size(0)
    assert outputs["local_scores"]["onset"].shape[0] == batch["onset"].x.size(0)
    assert outputs["local_score_summaries"].shape == (2, model.local_summary_dim)
    assert outputs["graph_score_features"].shape == (2, model.pooling_output_dim + model.local_summary_dim)


def test_summary_handles_empty_node_type_per_graph():
    g1 = _build_graph(note_count=2, chord_count=1, onset_count=0)
    g2 = _build_graph(note_count=2, chord_count=2, onset_count=2)
    batch = Batch.from_data_list([g1, g2])
    model = _build_model(batch, pooling_mode="mean", use_hybrid_graph_scorer=True)

    outputs = model(batch)

    assert outputs["local_score_summaries"].shape == (2, model.local_summary_dim)
    onset_block_start = 2 * model.local_summary_stats_count
    onset_block = outputs["local_score_summaries"][0, onset_block_start : onset_block_start + model.local_summary_stats_count]
    assert torch.allclose(onset_block, torch.zeros_like(onset_block))
