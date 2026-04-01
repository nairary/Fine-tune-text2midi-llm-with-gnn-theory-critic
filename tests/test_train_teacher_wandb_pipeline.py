from pathlib import Path
import sys

import pytest
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train_teacher import (
    ARTIFACT_SAMPLE_NOT_SAVED,
    build_artifact_sample_dir_map,
    collect_eval_diagnostics,
    finish_wandb_run,
    log_diagnostics_to_wandb,
)


class FakeGraphBatch:
    def __init__(self, graphs):
        self.graphs = graphs

    def to(self, _device):
        return self

    def to_data_list(self):
        return self.graphs


class DummyRun:
    def __init__(self):
        self.finish_calls = 0

    def finish(self):
        self.finish_calls += 1


class DummyNodeStore:
    def __init__(self, x):
        self.x = x


class DummyEdgeStore:
    def __init__(self, edge_index):
        self.edge_index = edge_index


class DummyGraph:
    def __init__(self):
        self.node_types = ["note"]
        self.edge_types = [("note", "next_note", "note")]
        self._nodes = {"note": DummyNodeStore(torch.tensor([[1.0, 0.0]]))}
        self._edges = {("note", "next_note", "note"): DummyEdgeStore(torch.tensor([[0], [0]], dtype=torch.long))}

    def __getitem__(self, item):
        return self._edges[item] if isinstance(item, tuple) else self._nodes[item]


class FakeWandbTable:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def add_data(self, *args):
        self.rows.append(args)


class FakeWandbArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.sample_dirs = []

    def add_dir(self, dir_path):
        base = Path(dir_path)
        self.sample_dirs = sorted([path.name for path in base.iterdir() if path.is_dir()])


class FakeWandb:
    def __init__(self):
        self.logged_payloads = []
        self.logged_artifacts = []
        self.last_table = None

    def Table(self, columns):
        table = FakeWandbTable(columns)
        self.last_table = table
        return table

    def log(self, payload, step=None):
        self.logged_payloads.append((payload, step))

    def Artifact(self, name, type):
        return FakeWandbArtifact(name, type)

    def log_artifact(self, artifact):
        self.logged_artifacts.append(artifact)


def _diagnostic_item(sample_idx: int) -> dict:
    return {
        "sample_local_index": sample_idx,
        "global_sample_index_in_loader": sample_idx,
        "raw_dataset_index": 100 + sample_idx,
        "song_id": f"song_{sample_idx}",
        "corruption_metadata": {"applied": True, "mode": "x"},
        "model_diff": {
            "graph": {
                "graph_score_real": 0.1,
                "graph_score_corrupted": 0.2,
                "graph_score_delta": 0.1,
                "graph_embedding_delta_l2": 0.3,
            },
            "nodes": {},
            "local_scores": {},
        },
        "summary_before": {"node_counts": {"note": 1}, "edge_counts": {"note__next_note__note": 1}},
        "summary_after": {"node_counts": {"note": 1}, "edge_counts": {"note__next_note__note": 1}},
        "diff": {"nodes": {"note": {"changed_row_indices": [0]}}, "edges": {}},
        "real_graph": DummyGraph(),
        "corrupted_graph": DummyGraph(),
    }


def test_collect_eval_diagnostics_computes_heavy_only_for_selected(monkeypatch):
    calls = {"model": 0, "summary": 0, "diff": 0}

    def fake_model_output_diff(**_kwargs):
        calls["model"] += 1
        return {"graph": {"graph_score_real": 0.0, "graph_score_corrupted": 0.0, "graph_score_delta": 0.0, "graph_embedding_delta_l2": 0.0}, "nodes": {}, "local_scores": {}}

    def fake_graph_summary(_graph):
        calls["summary"] += 1
        return {"node_counts": {}, "edge_counts": {}}

    def fake_graph_diff(_real, _corrupted):
        calls["diff"] += 1
        return {"nodes": {}, "edges": {}}

    monkeypatch.setattr("src.training.train_teacher.model_output_diff", fake_model_output_diff)
    monkeypatch.setattr("src.training.train_teacher.graph_summary", fake_graph_summary)
    monkeypatch.setattr("src.training.train_teacher.graph_diff", fake_graph_diff)

    batch = {
        "graph_real": FakeGraphBatch([object(), object(), object(), object()]),
        "graph_masked": FakeGraphBatch([]),
        "graph_corrupted": FakeGraphBatch([object(), object(), object(), object()]),
        "masked_labels": [],
        "corruption_metadata": [
            {"applied": False},
            {"applied": True},
            {"applied": False},
            {"applied": True},
        ],
        "graph_score_label": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        "raw_dataset_index": [10, 11, 12, 13],
        "song_id": ["s10", "s11", "s12", "s13"],
    }

    diagnostics = collect_eval_diagnostics(
        model=None,
        loader=[batch],
        device=torch.device("cpu"),
        max_samples=2,
        max_batches=1,
    )

    assert len(diagnostics) == 2
    assert [item["raw_dataset_index"] for item in diagnostics] == [11, 13]
    assert [item["song_id"] for item in diagnostics] == ["s11", "s13"]
    assert calls["model"] == 2
    assert calls["summary"] == 4
    assert calls["diff"] == 2


def test_collect_eval_diagnostics_does_not_keep_unbounded_candidates(monkeypatch):
    observed_candidates = {"count": 0}

    def fake_prioritize(candidates, max_samples):
        observed_candidates["count"] = len(candidates)
        return candidates[:max_samples]

    monkeypatch.setattr("src.training.train_teacher.prioritize_diagnostic_examples", fake_prioritize)
    monkeypatch.setattr(
        "src.training.train_teacher.model_output_diff",
        lambda **_kwargs: {"graph": {"graph_score_real": 0.0, "graph_score_corrupted": 0.0, "graph_score_delta": 0.0, "graph_embedding_delta_l2": 0.0}, "nodes": {}, "local_scores": {}},
    )
    monkeypatch.setattr("src.training.train_teacher.graph_summary", lambda _graph: {"node_counts": {}, "edge_counts": {}})
    monkeypatch.setattr("src.training.train_teacher.graph_diff", lambda _real, _corrupted: {"nodes": {}, "edges": {}})

    many_graphs = [object() for _ in range(50)]
    batch = {
        "graph_real": FakeGraphBatch(many_graphs),
        "graph_masked": FakeGraphBatch([]),
        "graph_corrupted": FakeGraphBatch(many_graphs),
        "masked_labels": [],
        "corruption_metadata": [{"applied": False} for _ in range(50)],
        "graph_score_label": torch.ones(50),
        "raw_dataset_index": list(range(50)),
        "song_id": [f"s{i}" for i in range(50)],
    }

    collect_eval_diagnostics(
        model=None,
        loader=[batch],
        device=torch.device("cpu"),
        max_samples=2,
        max_batches=1,
    )

    assert observed_candidates["count"] <= 16


def test_diagnostics_max_scan_batches_respected(monkeypatch):
    def fake_model_output_diff(**_kwargs):
        return {"graph": {"graph_score_real": 0.0, "graph_score_corrupted": 0.0, "graph_score_delta": 0.0, "graph_embedding_delta_l2": 0.0}, "nodes": {}, "local_scores": {}}

    monkeypatch.setattr("src.training.train_teacher.model_output_diff", fake_model_output_diff)
    monkeypatch.setattr("src.training.train_teacher.graph_summary", lambda _graph: {"node_counts": {}, "edge_counts": {}})
    monkeypatch.setattr("src.training.train_teacher.graph_diff", lambda _real, _corrupted: {"nodes": {}, "edges": {}})

    batch1 = {
        "graph_real": FakeGraphBatch([object()]),
        "graph_masked": FakeGraphBatch([]),
        "graph_corrupted": FakeGraphBatch([object()]),
        "masked_labels": [],
        "corruption_metadata": [{"applied": True}],
        "graph_score_label": torch.tensor([1.0]),
        "raw_dataset_index": [1],
        "song_id": ["song_1"],
    }
    batch2 = {
        "graph_real": FakeGraphBatch([object()]),
        "graph_masked": FakeGraphBatch([]),
        "graph_corrupted": FakeGraphBatch([object()]),
        "masked_labels": [],
        "corruption_metadata": [{"applied": True}],
        "graph_score_label": torch.tensor([1.0]),
        "raw_dataset_index": [2],
        "song_id": ["song_2"],
    }

    diagnostics = collect_eval_diagnostics(
        model=None,
        loader=[batch1, batch2],
        device=torch.device("cpu"),
        max_samples=2,
        max_batches=1,
    )
    assert [item["raw_dataset_index"] for item in diagnostics] == [1]


def test_finish_wandb_run_called_in_finally():
    run = DummyRun()
    with pytest.raises(RuntimeError):
        try:
            raise RuntimeError("boom")
        finally:
            finish_wandb_run({"run": run})
    assert run.finish_calls == 1


def test_artifact_sample_dir_map_when_limit_smaller_than_diagnostics():
    mapping = build_artifact_sample_dir_map(num_diagnostics=5, max_artifact_examples=2)
    assert mapping[0] == "sample_0"
    assert mapping[1] == "sample_1"
    assert mapping[2] is None
    assert mapping[3] is None
    assert mapping[4] is None


def test_artifact_sample_dir_map_when_limit_covers_all_diagnostics():
    mapping = build_artifact_sample_dir_map(num_diagnostics=2, max_artifact_examples=5)
    assert mapping[0] == "sample_0"
    assert mapping[1] == "sample_1"
    assert sorted([value for value in mapping.values() if value is not None]) == ["sample_0", "sample_1"]


def test_log_diagnostics_to_wandb_syncs_table_and_artifacts_when_limited(tmp_path):
    fake_wandb = FakeWandb()
    cfg = OmegaConf.create({"wandb": {"log_artifacts": True, "artifact_examples_limit": 2}})
    diagnostics = [_diagnostic_item(i) for i in range(5)]

    log_diagnostics_to_wandb(
        wandb_state={"wandb": fake_wandb},
        cfg=cfg,
        epoch=1,
        split="val",
        diagnostics=diagnostics,
        output_dir=tmp_path,
    )

    assert len(fake_wandb.last_table.rows) == 5
    artifact_idx = fake_wandb.last_table.columns.index("artifact_sample_dir")
    assert fake_wandb.last_table.rows[0][artifact_idx] == "sample_0"
    assert fake_wandb.last_table.rows[1][artifact_idx] == "sample_1"
    assert fake_wandb.last_table.rows[2][artifact_idx] == ARTIFACT_SAMPLE_NOT_SAVED
    assert fake_wandb.last_table.rows[4][artifact_idx] == ARTIFACT_SAMPLE_NOT_SAVED
    assert len(fake_wandb.logged_artifacts) == 1
    assert fake_wandb.logged_artifacts[0].sample_dirs == ["sample_0", "sample_1"]


def test_log_diagnostics_to_wandb_all_saved_when_limit_covers_count(tmp_path):
    fake_wandb = FakeWandb()
    cfg = OmegaConf.create({"wandb": {"log_artifacts": True, "artifact_examples_limit": 5}})
    diagnostics = [_diagnostic_item(i) for i in range(2)]

    log_diagnostics_to_wandb(
        wandb_state={"wandb": fake_wandb},
        cfg=cfg,
        epoch=1,
        split="val",
        diagnostics=diagnostics,
        output_dir=tmp_path,
    )

    artifact_idx = fake_wandb.last_table.columns.index("artifact_sample_dir")
    assert [row[artifact_idx] for row in fake_wandb.last_table.rows] == ["sample_0", "sample_1"]
    assert fake_wandb.logged_artifacts[0].sample_dirs == ["sample_0", "sample_1"]


def test_log_diagnostics_to_wandb_does_not_log_artifact_when_limit_zero(tmp_path):
    fake_wandb = FakeWandb()
    cfg = OmegaConf.create({"wandb": {"log_artifacts": True, "artifact_examples_limit": 0}})
    diagnostics = [_diagnostic_item(i) for i in range(3)]

    log_diagnostics_to_wandb(
        wandb_state={"wandb": fake_wandb},
        cfg=cfg,
        epoch=1,
        split="val",
        diagnostics=diagnostics,
        output_dir=tmp_path,
    )

    artifact_idx = fake_wandb.last_table.columns.index("artifact_sample_dir")
    assert all(row[artifact_idx] == ARTIFACT_SAMPLE_NOT_SAVED for row in fake_wandb.last_table.rows)
    assert fake_wandb.logged_artifacts == []


def test_log_diagnostics_to_wandb_no_artifacts_when_disabled(tmp_path):
    fake_wandb = FakeWandb()
    cfg = OmegaConf.create({"wandb": {"log_artifacts": False, "artifact_examples_limit": 2}})
    diagnostics = [_diagnostic_item(i) for i in range(2)]

    log_diagnostics_to_wandb(
        wandb_state={"wandb": fake_wandb},
        cfg=cfg,
        epoch=1,
        split="val",
        diagnostics=diagnostics,
        output_dir=tmp_path,
    )

    artifact_idx = fake_wandb.last_table.columns.index("artifact_sample_dir")
    assert all(row[artifact_idx] == ARTIFACT_SAMPLE_NOT_SAVED for row in fake_wandb.last_table.rows)
    assert fake_wandb.logged_artifacts == []
