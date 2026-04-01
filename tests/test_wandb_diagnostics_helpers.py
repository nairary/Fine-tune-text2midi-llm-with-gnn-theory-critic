from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.wandb_diagnostics import (
    corruption_applied,
    graph_diff,
    graph_summary,
    model_output_diff,
    prioritize_diagnostic_examples,
    serialize_graph,
    serialize_graph_nodes,
)


class FakeModel:
    def eval(self):
        return self

    def __call__(self, graph):
        note_x = graph["note"].x.float()
        emb = note_x.mean(dim=1, keepdim=True)
        return {
            "graph_score": emb.mean().reshape(1),
            "graph_embedding": emb.mean(dim=0, keepdim=True),
            "graph_score_features": emb.mean(dim=0, keepdim=True),
            "local_score_summaries": emb.mean(dim=0, keepdim=True),
            "node_embeddings": {"note": emb},
            "local_scores": {"note": emb.squeeze(-1)},
        }


class NodeStore:
    def __init__(self, x):
        self.x = x


class EdgeStore:
    def __init__(self, edge_index):
        self.edge_index = edge_index


class MiniGraph:
    def __init__(self, note_x, edge_index):
        self.node_types = ["note"]
        self.edge_types = [("note", "next_note", "note")]
        self._nodes = {"note": NodeStore(note_x)}
        self._edges = {("note", "next_note", "note"): EdgeStore(edge_index)}

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self._edges[item]
        return self._nodes[item]

    def to(self, _device):
        return self


class MultiTypeGraph(MiniGraph):
    def __init__(self, nodes_by_type, edges_by_type):
        self.node_types = list(nodes_by_type.keys())
        self.edge_types = list(edges_by_type.keys())
        self._nodes = {key: NodeStore(value) for key, value in nodes_by_type.items()}
        self._edges = {key: EdgeStore(value) for key, value in edges_by_type.items()}


def test_graph_summary_and_diff():
    real = MiniGraph(
        note_x=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    corrupted = MiniGraph(
        note_x=torch.tensor([[1.0, 0.0], [4.0, 0.0]]),
        edge_index=torch.tensor([[1], [0]], dtype=torch.long),
    )

    summary = graph_summary(real)
    assert summary["node_counts"]["note"] == 2
    assert summary["edge_counts"]["note__next_note__note"] == 1

    diff = graph_diff(real, corrupted)
    assert diff["nodes"]["note"]["changed_row_indices"] == [1]
    assert diff["edges"]["note__next_note__note"]["added_edges_count"] == 1
    assert diff["edges"]["note__next_note__note"]["removed_edges_count"] == 1


def test_graph_diff_contains_before_after():
    real = MiniGraph(
        note_x=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    corrupted = MiniGraph(
        note_x=torch.tensor([[1.0, 0.0], [3.0, 1.0]]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    diff = graph_diff(real, corrupted)
    changed = diff["nodes"]["note"]["changed_rows"][0]
    assert "before" in changed
    assert "after" in changed


def test_graph_diff_handles_missing_node_and_edge_types():
    real = MultiTypeGraph(
        nodes_by_type={"note": torch.tensor([[1.0, 0.0]])},
        edges_by_type={("note", "next_note", "note"): torch.tensor([[0], [0]], dtype=torch.long)},
    )
    corrupted = MultiTypeGraph(
        nodes_by_type={
            "note": torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
            "chord": torch.tensor([[3.0, 1.0]]),
        },
        edges_by_type={("chord", "next_chord", "chord"): torch.tensor([[0], [0]], dtype=torch.long)},
    )

    diff = graph_diff(real, corrupted)
    assert "chord" in diff["nodes"]
    assert diff["nodes"]["chord"]["count_before"] == 0
    assert diff["nodes"]["chord"]["count_after"] == 1
    assert diff["edges"]["note__next_note__note"]["count_after"] == 0
    assert diff["edges"]["chord__next_chord__chord"]["count_before"] == 0


def test_graph_diff_counts_added_and_removed_rows():
    real = MiniGraph(
        note_x=torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    corrupted = MiniGraph(
        note_x=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    diff = graph_diff(real, corrupted)
    assert diff["nodes"]["note"]["added_rows_count"] == 0
    assert diff["nodes"]["note"]["removed_rows_count"] == 1


def test_model_output_diff_and_corruption_applied_field():
    real = MiniGraph(
        note_x=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    corrupted = MiniGraph(
        note_x=torch.tensor([[1.0, 0.0], [4.0, 0.0]]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    payload = model_output_diff(FakeModel(), torch.device("cpu"), real, corrupted)
    assert payload["graph"]["graph_score_delta"] != 0.0
    assert payload["nodes"]["note"]["top_changed_node_indices"]
    assert corruption_applied({"applied": True}) is True
    assert corruption_applied({"applied": False}) is False
    assert corruption_applied({}) is False
    assert corruption_applied(None) is False


def test_prioritize_diagnostic_examples_prefers_applied():
    examples = [
        {"id": 1, "corruption_metadata": {"applied": False}},
        {"id": 2, "corruption_metadata": {"applied": True}},
        {"id": 3, "corruption_metadata": {"applied": False}},
    ]
    prioritized = prioritize_diagnostic_examples(examples, max_samples=2)
    assert [item["id"] for item in prioritized] == [2, 1]


def test_serialize_graph_contains_nodes_and_edges():
    graph = MiniGraph(
        note_x=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    serialized = serialize_graph(graph)
    assert "nodes" in serialized
    assert "edges" in serialized


def test_serialize_graph_nodes_contains_raw_rows():
    graph = MiniGraph(
        note_x=torch.tensor([[1.0, 0.0]]),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
    )
    nodes = serialize_graph_nodes(graph)
    row = nodes["note"]["rows"][0]
    assert "node_index" in row
    assert "raw" in row
