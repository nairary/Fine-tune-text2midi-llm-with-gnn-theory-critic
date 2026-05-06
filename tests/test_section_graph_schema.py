from __future__ import annotations

import sys
from pathlib import Path

from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader.graph_layouts import NODE_DIMS, SECTION_LABEL_IDS, SECTION_LAYOUT
from src.dataloader.utils_graph import MANDATORY_EDGE_TYPES, MANDATORY_NODE_TYPES, build_graph_from_encoded
from src.models.teacher_gnn import TeacherGNN


def _chord(beat: float, duration: float = 4.0) -> dict:
    return {
        "beat": beat,
        "duration": duration,
        "root_id": 1,
        "type_id": 1,
        "inversion_id": 1,
        "applied_id": 1,
        "borrowed_kind_id": 2,
        "borrowed_mode_name_id": 2,
        "adds_vec": [0, 0, 0, 0, 0, 0],
        "omits_vec": [0, 0],
        "suspensions_vec": [0, 0],
        "alterations_vec": [0, 0, 0, 0, 0, 0],
        "borrowed_pcset_vec": [0] * 12,
        "is_rest": 0,
    }


def _song_with_section_spans() -> dict:
    return {
        "song_id": "assembled_test",
        "meta": {
            "end_beat": 17.0,
            "main_num_beats": 4.0,
            "main_beat_unit": 1.0,
            "main_bpm": 120.0,
            "main_key_tonic_pc_id": 1,
            "main_key_scale_id": 1,
            "main_num_beats_id": 1,
            "main_beat_unit_id": 1,
            "section_spans": [
                {
                    "section_index": 0,
                    "label": "verse",
                    "labels": ["verse"],
                    "source_clip_song_ids": ["clip_a"],
                    "target_start_beat": 1.0,
                    "target_end_beat": 9.0,
                    "inserted_gap_beats_before": 0.0,
                },
                {
                    "section_index": 1,
                    "label": "chorus",
                    "labels": ["chorus"],
                    "source_clip_song_ids": ["clip_b"],
                    "target_start_beat": 13.0,
                    "target_end_beat": 17.0,
                    "inserted_gap_beats_before": 4.0,
                    "positive_gap_seconds_from_previous": 2.0,
                },
            ],
        },
        "melody": [
            {"beat": 2.0, "duration": 1.0, "sd_id": 4, "octave_id": 5, "is_rest": 0},
            {"beat": 14.0, "duration": 1.0, "sd_id": 5, "octave_id": 5, "is_rest": 0},
        ],
        "chords": [_chord(1.0, 8.0), _chord(13.0, 4.0)],
    }


def _legacy_clip_song() -> dict:
    return {
        "song_id": "legacy_clip",
        "meta": {
            "end_beat": 9.0,
            "main_num_beats": 4.0,
            "main_beat_unit": 1.0,
        },
        "sections": [
            {
                "label_ids": [SECTION_LABEL_IDS["chorus"]],
                "duration_seconds": 10.0,
                "segment_start_seconds": 0.0,
                "segment_end_seconds": 10.0,
            }
        ],
        "melody": [
            {"beat": 1.0, "duration": 1.0, "sd_id": 4, "octave_id": 5, "is_rest": 0},
        ],
        "chords": [],
    }


def _edge_pairs(graph, edge_type: tuple[str, str, str]) -> list[tuple[int, int]]:
    return [tuple(pair) for pair in graph[edge_type].edge_index.t().tolist()]


def test_section_spans_create_section_nodes_and_edges():
    graph = build_graph_from_encoded(_song_with_section_spans())

    assert set(MANDATORY_NODE_TYPES).issubset(set(graph.node_types))
    assert graph["section"].x.shape == (2, NODE_DIMS["section"])
    assert graph["section"].x[:, SECTION_LAYOUT["label_id"]].tolist() == [
        float(SECTION_LABEL_IDS["verse"]),
        float(SECTION_LABEL_IDS["chorus"]),
    ]
    assert graph["section"].x[:, SECTION_LAYOUT["start_beat"]].tolist() == [1.0, 13.0]
    assert graph["section"].x[:, SECTION_LAYOUT["end_beat"]].tolist() == [9.0, 17.0]
    assert graph["section"].x[1, SECTION_LAYOUT["inserted_gap_beats_before"]].item() == 4.0

    assert _edge_pairs(graph, ("song", "contains_section", "section")) == [(0, 0), (0, 1)]
    assert _edge_pairs(graph, ("section", "next_section", "section")) == [(0, 1)]
    assert _edge_pairs(graph, ("section", "contains_note", "note")) == [(0, 0), (1, 1)]
    assert _edge_pairs(graph, ("section", "contains_chord", "chord")) == [(0, 0), (1, 1)]

    section_bar_pairs = _edge_pairs(graph, ("section", "contains_bar", "bar"))
    assert (0, 0) in section_bar_pairs
    assert (0, 1) in section_bar_pairs
    assert (1, 3) in section_bar_pairs
    assert all(bar_idx != 2 for _, bar_idx in section_bar_pairs)
    assert graph.graph_metadata["n_sections"] == 2


def test_legacy_clip_gets_one_dummy_section_with_existing_label():
    graph = build_graph_from_encoded(_legacy_clip_song())

    assert graph["section"].x.shape == (1, NODE_DIMS["section"])
    assert graph["section"].x[0, SECTION_LAYOUT["label_id"]].item() == float(SECTION_LABEL_IDS["chorus"])
    assert _edge_pairs(graph, ("song", "contains_section", "section")) == [(0, 0)]
    assert _edge_pairs(graph, ("section", "contains_note", "note")) == [(0, 0)]
    assert graph.graph_metadata["n_sections"] == 1

    for edge_type in MANDATORY_EDGE_TYPES:
        assert graph[edge_type].edge_index.size(0) == 2

    batch = Batch.from_data_list([graph, build_graph_from_encoded(_song_with_section_spans())])
    assert batch["section"].x.size(0) == 3


def test_hgt_teacher_forward_supports_dummy_and_real_section_graphs():
    batch = Batch.from_data_list([build_graph_from_encoded(_legacy_clip_song()), build_graph_from_encoded(_song_with_section_spans())])
    model = TeacherGNN.from_hetero_data(
        batch,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        residual=True,
        backbone="hgt",
        hgt_num_heads=4,
        pooling_output_dim=16,
        score_head_hidden_dim=8,
        reconstruction_head_hidden_dim=16,
        local_score_head_hidden_dim=8,
    )

    outputs = model(batch)

    assert outputs["graph_score"].shape == (2,)
    assert outputs["node_embeddings"]["section"].shape[0] == batch["section"].x.size(0)
    assert outputs["local_scores"]["note"].shape[0] == batch["note"].x.size(0)
    assert outputs["recon_logits"]["note_sd"].shape[0] == batch["note"].x.size(0)
