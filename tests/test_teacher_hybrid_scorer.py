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


def _slow_mean_contextual_local_scores(model: TeacherGNN, batch: Batch, node_embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    edge_maps = model._prepare_edge_maps(batch)
    batch_dict = model._get_batch_dict(batch)
    song_embeddings = node_embeddings["song"]
    zero = song_embeddings.new_zeros((song_embeddings.size(-1),))

    def song_context_for(node_type: str, node_idx: int) -> torch.Tensor:
        graph_idx = int(batch_dict[node_type][node_idx].item()) if batch_dict[node_type].numel() > 0 else 0
        if 0 <= graph_idx < song_embeddings.size(0):
            return song_embeddings[graph_idx]
        return zero

    contextual_scores = {}
    if "note" in model.local_score_heads:
        note_contexts = []
        for idx in range(node_embeddings["note"].size(0)):
            onset_idx = edge_maps["note_to_onset"][idx]
            onset_emb = node_embeddings["onset"][onset_idx] if onset_idx is not None else zero
            note_contexts.append(
                torch.stack(
                    [
                        node_embeddings["note"][idx],
                        model._gather_mean(node_embeddings["note"], edge_maps["note_neighbors"][idx], zero),
                        onset_emb,
                        model._gather_mean(node_embeddings["chord"], edge_maps["note_to_chords"][idx], zero),
                        song_context_for("note", idx),
                    ],
                    dim=0,
                ).mean(dim=0)
            )
        contextual_scores["note"] = model.local_score_heads["note"](torch.stack(note_contexts, dim=0))

    if "chord" in model.local_score_heads:
        chord_contexts = []
        for idx in range(node_embeddings["chord"].size(0)):
            onset_idx = edge_maps["chord_to_onset"][idx]
            onset_emb = node_embeddings["onset"][onset_idx] if onset_idx is not None else zero
            chord_contexts.append(
                torch.stack(
                    [
                        node_embeddings["chord"][idx],
                        model._gather_mean(node_embeddings["chord"], edge_maps["chord_neighbors"][idx], zero),
                        model._gather_mean(node_embeddings["note"], edge_maps["chord_to_notes"][idx], zero),
                        onset_emb,
                        song_context_for("chord", idx),
                    ],
                    dim=0,
                ).mean(dim=0)
            )
        contextual_scores["chord"] = model.local_score_heads["chord"](torch.stack(chord_contexts, dim=0))

    if "onset" in model.local_score_heads:
        onset_contexts = []
        for idx in range(node_embeddings["onset"].size(0)):
            onset_contexts.append(
                torch.stack(
                    [
                        node_embeddings["onset"][idx],
                        model._gather_mean(node_embeddings["note"], edge_maps["onset_to_notes"][idx], zero),
                        model._gather_mean(node_embeddings["chord"], edge_maps["onset_to_chords"][idx], zero),
                        model._gather_mean(node_embeddings["onset"], edge_maps["onset_neighbors"][idx], zero),
                        song_context_for("onset", idx),
                    ],
                    dim=0,
                ).mean(dim=0)
            )
        contextual_scores["onset"] = model.local_score_heads["onset"](torch.stack(onset_contexts, dim=0))
    return contextual_scores


def test_fast_mean_local_context_matches_reference_loop():
    batch = Batch.from_data_list([_build_graph(note_count=5, chord_count=3, onset_count=4), _build_graph(note_count=4, chord_count=4, onset_count=3)])
    model = _build_model(batch, pooling_mode="mean", use_hybrid_graph_scorer=True, local_context_mode="mean")
    model.eval()

    with torch.no_grad():
        node_embeddings = model.backbone(model.encode_nodes(batch), batch.edge_index_dict)
        fast_scores = model.compute_contextual_local_scores(batch, node_embeddings)
        slow_scores = _slow_mean_contextual_local_scores(model, batch, node_embeddings)

    assert set(fast_scores) == set(slow_scores)
    for node_type in fast_scores:
        assert torch.allclose(fast_scores[node_type], slow_scores[node_type], atol=1e-6)


def test_fast_local_summary_matches_reference_loop():
    batch = Batch.from_data_list([_build_graph(note_count=5), _build_graph(note_count=3), _build_graph(note_count=0)])
    model = _build_model(batch, pooling_mode="mean", use_hybrid_graph_scorer=True, local_summary_use_topk_mean=True)
    scores = torch.tensor([0.1, -0.3, 0.7, 0.4, 0.2, 2.0, -1.0, 0.5])

    summary = model._summarize_type_scores(scores, batch["note"].batch, num_graphs=3)
    expected_rows = []
    for graph_idx in range(3):
        graph_scores = scores[batch["note"].batch == graph_idx]
        if graph_scores.numel() == 0:
            expected_rows.append(torch.zeros(3))
            continue
        expected_rows.append(
            torch.tensor(
                [
                    graph_scores.mean(),
                    graph_scores.max(),
                    torch.topk(graph_scores, k=min(model.local_summary_topk, graph_scores.numel())).values.mean(),
                ]
            )
        )
    expected = torch.stack(expected_rows, dim=0)

    assert torch.allclose(summary, expected, atol=1e-6)


def test_forward_can_skip_unused_heads_for_mlm_stage():
    batch = Batch.from_data_list([_build_graph(), _build_graph()])
    model = _build_model(batch, pooling_mode="mean", use_hybrid_graph_scorer=True)

    outputs = model(batch, compute_recon=True, compute_graph_score=False, compute_local_scores=False)

    assert outputs["recon_logits"]
    assert outputs["local_scores"] == {}
    assert outputs["graph_score"].shape == (2,)
    assert torch.count_nonzero(outputs["graph_score"]).item() == 0
    assert outputs["graph_score_features"].shape == (2, 0)


def test_forward_skips_reconstruction_for_corruption_stage():
    batch = Batch.from_data_list([_build_graph(), _build_graph()])
    model = _build_model(batch, pooling_mode="mean", use_hybrid_graph_scorer=True)

    outputs = model(batch, compute_recon=False, compute_graph_score=True, compute_local_scores=True)

    assert outputs["recon_logits"] == {}
    assert set(outputs["local_scores"]) == {"note", "chord", "onset"}
    assert outputs["graph_score"].shape == (2,)
    assert outputs["graph_score_features"].shape == (2, model.pooling_output_dim + model.local_summary_dim)


def test_graph_score_forces_local_summary_when_hybrid_needs_it():
    batch = Batch.from_data_list([_build_graph(), _build_graph()])
    model = _build_model(batch, pooling_mode="mean", use_hybrid_graph_scorer=True)

    outputs = model(batch, compute_recon=False, compute_graph_score=True, compute_local_scores=False)

    assert outputs["recon_logits"] == {}
    assert set(outputs["local_scores"]) == {"note", "chord", "onset"}
    assert outputs["graph_score_features"].shape == (2, model.pooling_output_dim + model.local_summary_dim)


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
