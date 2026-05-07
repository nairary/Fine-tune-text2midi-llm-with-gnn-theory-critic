from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.observer.build_observer_pair_dataset import build_pairs
from src.observer.build_observer_pair_targets import PairTargetJoinError, _join_pair_targets, build_pair_targets
from src.observer.cached_dataset import ObserverPairCachedDataset
from src.observer.train_observer_distill import ObserverDistillationAdapters, _collate_pairs, _run_epoch


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _build_cfg(tmp_path: Path, dataset_path: Path):
    return OmegaConf.create(
        {
            "data": {"json_path": str(dataset_path), "split": {"train": "train", "val": "val", "test": "test"}},
            "dataloader": {
                "corruption_backend": "song_theory",
                "corruption_modes": ["strong_weak_beat_flip"],
                "theory_aware": {"deterministic_per_sample": True, "deterministic_seed": 11},
                "pairs_per_song": 1,
                "batch_size": 2,
                "num_workers": 0,
            },
            "observer_pipeline": {
                "output_root": str(tmp_path / "out"),
                "overwrite": True,
                "skip_render_failures": True,
                "skip_graph_build_failures": True,
            },
            "observer_training": {
                "teacher_checkpoint": "models/best_rank_acc.pt",
                "teacher_config": "configs/config.yaml",
                "device": "cpu",
            },
            "losses": {"lambda_reg": 1.0, "lambda_rank": 0.5, "min_teacher_gap_for_rank": 0.25, "use_pair_rank": True},
        }
    )


def test_default_config_is_self_consistent():
    cfg = OmegaConf.load(REPO_ROOT / "configs/observer_distill.yaml")
    dl = OmegaConf.load(REPO_ROOT / "configs/dataloader/observer_pairs_song_theory.yaml")
    pipe = OmegaConf.load(REPO_ROOT / "configs/observer_pipeline/default.yaml")
    assert not (pipe.overwrite is False and dl.theory_aware.deterministic_per_sample is False)


def _song(song_id: str = "s1", split: str = "train", beat_origin: float | None = None):
    meta = {
        "song_id": song_id,
        "split": split,
        "main_key_tonic_pc": 0,
        "main_key_scale_id": 3,
        "main_bpm": 120,
        "main_num_beats": 4,
        "main_beat_unit": 1,
    }
    if beat_origin is not None:
        meta["beat_origin"] = beat_origin
    return {
        "meta": meta,
        "melody": [{"beat": 1.0, "duration": 1.0, "sd_id": 4, "octave_id": 6, "is_rest": 0}],
        "chords": [{"beat": 1.0, "duration": 1.0, "root_id": 1, "type_id": 1, "inversion_id": 1, "is_rest": 0}],
    }


def _read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_pair_manifest_numeric_beat_origin_and_no_dangling_refs(tmp_path: Path):
    ds_path = tmp_path / "encoded.json"
    _write_json(ds_path, {"s1": _song("s1", "train", beat_origin=2.0)})

    cfg = _build_cfg(tmp_path, ds_path)
    build_pairs(cfg)

    out_root = Path(cfg.observer_pipeline.output_root)
    manifest_rows = _read_jsonl(out_root / "pairs/manifests/train.jsonl")
    pair_rows = _read_jsonl(out_root / "pairs/index/train_pairs.jsonl")

    assert len(manifest_rows) == 2
    assert len(pair_rows) == 1
    sample_ids = {r["sample_id"] for r in manifest_rows}

    for row in manifest_rows:
        assert isinstance(row["beat_origin"], (int, float))
        assert Path(row["midi_path"]).exists()
        assert Path(row["encoded_song_path"]).exists()
        assert "tonal_group" in row
        assert "corruption_group" in row
        assert "mode_group" not in row

    pair = pair_rows[0]
    assert pair["clean_sample_id"] in sample_ids
    assert pair["corrupted_sample_id"] in sample_ids


def test_stable_deterministic_pair_ids_across_dataset_order(tmp_path: Path):
    cfg = _build_cfg(tmp_path, tmp_path / "encoded_a.json")
    _write_json(tmp_path / "encoded_a.json", {"a": _song("a", "train"), "b": _song("b", "train")})
    build_pairs(cfg)
    ids_a = sorted(x["pair_group_id"] for x in _read_jsonl(Path(cfg.observer_pipeline.output_root) / "pairs/index/train_pairs.jsonl"))

    cfg2 = _build_cfg(tmp_path, tmp_path / "encoded_b.json")
    cfg2.observer_pipeline.output_root = str(tmp_path / "out2")
    _write_json(tmp_path / "encoded_b.json", {"b": _song("b", "train"), "a": _song("a", "train")})
    build_pairs(cfg2)
    ids_b = sorted(x["pair_group_id"] for x in _read_jsonl(Path(cfg2.observer_pipeline.output_root) / "pairs/index/train_pairs.jsonl"))

    assert ids_a == ids_b


def test_bad_meta_counted_once(tmp_path: Path):
    cfg = _build_cfg(tmp_path, tmp_path / "encoded.json")
    bad = {"meta": {"song_id": "bad"}, "melody": [], "chords": []}
    _write_json(tmp_path / "encoded.json", {"bad": bad, "good": _song("good", "train")})
    build_pairs(cfg)
    skip_rows = _read_jsonl(Path(cfg.observer_pipeline.output_root) / "pairs/skipped_manifest_rows.jsonl")
    bad_rows = [x for x in skip_rows if x.get("index") == 0 or x.get("source_song_id") == "bad"]
    assert len(bad_rows) == 1


def test_pair_row_created_only_if_both_written_and_no_duplicates_on_rerun(tmp_path: Path, monkeypatch):
    ds_path = tmp_path / "encoded.json"
    _write_json(ds_path, {"s1": _song("s1", "train")})
    cfg = _build_cfg(tmp_path, ds_path)

    import src.observer.build_observer_pair_dataset as builder

    # fail corrupted render only -> no pair row and no orphan rows
    real_render = builder._render_midi

    def selective_render(song_obj, midi_path, theory_ctx, octave_id_map):
        if str(midi_path).endswith("corrupted.mid"):
            raise RuntimeError("forced corrupted render fail")
        return real_render(song_obj, midi_path, theory_ctx, octave_id_map)

    monkeypatch.setattr(builder, "_render_midi", selective_render)
    stats = build_pairs(cfg)

    out_root = Path(cfg.observer_pipeline.output_root)
    manifest_rows = _read_jsonl(out_root / "pairs/manifests/train.jsonl")
    pair_rows = _read_jsonl(out_root / "pairs/index/train_pairs.jsonl")
    assert manifest_rows == []
    assert pair_rows == []
    assert stats.skipped_rows == 1

    # restore render and run twice with overwrite=False -> no duplication
    monkeypatch.setattr(builder, "_render_midi", real_render)
    cfg.observer_pipeline.overwrite = False
    build_pairs(cfg)
    build_pairs(cfg)
    manifest_rows = _read_jsonl(out_root / "pairs/manifests/train.jsonl")
    pair_rows = _read_jsonl(out_root / "pairs/index/train_pairs.jsonl")
    assert len({x["sample_id"] for x in manifest_rows}) == len(manifest_rows)
    assert len({x["pair_group_id"] for x in pair_rows}) == len(pair_rows)


def test_overwrite_false_recovers_missing_artifact(tmp_path: Path):
    cfg = _build_cfg(tmp_path, tmp_path / "encoded.json")
    _write_json(tmp_path / "encoded.json", {"s1": _song("s1", "train")})
    build_pairs(cfg)
    out_root = Path(cfg.observer_pipeline.output_root)
    manifest_rows = _read_jsonl(out_root / "pairs/manifests/train.jsonl")
    # delete one artifact and rerun with overwrite=false
    Path(manifest_rows[0]["midi_path"]).unlink()
    cfg.observer_pipeline.overwrite = False
    build_pairs(cfg)
    manifest_rows2 = _read_jsonl(out_root / "pairs/manifests/train.jsonl")
    assert all(Path(r["midi_path"]).exists() and Path(r["encoded_song_path"]).exists() for r in manifest_rows2)


def test_overwrite_false_rebuild_cleans_stale_incomplete_pair_files(tmp_path: Path, monkeypatch):
    cfg = _build_cfg(tmp_path, tmp_path / "encoded.json")
    _write_json(tmp_path / "encoded.json", {"s1": _song("s1", "train")})
    build_pairs(cfg)
    out_root = Path(cfg.observer_pipeline.output_root)
    manifest = _read_jsonl(out_root / "pairs/manifests/train.jsonl")
    pair = _read_jsonl(out_root / "pairs/index/train_pairs.jsonl")[0]

    # emulate stale incomplete pair on disk/index
    clean_row = next(x for x in manifest if x["sample_id"] == pair["clean_sample_id"])
    corr_row = next(x for x in manifest if x["sample_id"] == pair["corrupted_sample_id"])
    Path(clean_row["encoded_song_path"]).write_text("stale", encoding="utf-8")
    Path(clean_row["midi_path"]).write_text("stale", encoding="utf-8")
    Path(corr_row["encoded_song_path"]).unlink()

    cfg.observer_pipeline.overwrite = False

    import src.observer.build_observer_pair_dataset as builder
    real_render = builder._render_midi

    def fail_all(*args, **kwargs):
        raise RuntimeError("forced fail")

    monkeypatch.setattr(builder, "_render_midi", fail_all)
    build_pairs(cfg)
    monkeypatch.setattr(builder, "_render_midi", real_render)

    # old stale files should be removed during forced rebuild path
    assert not Path(clean_row["encoded_song_path"]).exists()
    assert not Path(clean_row["midi_path"]).exists()


def test_no_orphan_rows_when_clean_render_fails(tmp_path: Path, monkeypatch):
    ds_path = tmp_path / "encoded.json"
    _write_json(ds_path, {"s1": _song("s1", "train")})
    cfg = _build_cfg(tmp_path, ds_path)

    import src.observer.build_observer_pair_dataset as builder

    real_render = builder._render_midi

    def selective_render(song_obj, midi_path, theory_ctx, octave_id_map):
        if str(midi_path).endswith("clean.mid"):
            raise RuntimeError("forced clean render fail")
        return real_render(song_obj, midi_path, theory_ctx, octave_id_map)

    monkeypatch.setattr(builder, "_render_midi", selective_render)
    build_pairs(cfg)

    out_root = Path(cfg.observer_pipeline.output_root)
    assert _read_jsonl(out_root / "pairs/manifests/train.jsonl") == []
    assert _read_jsonl(out_root / "pairs/index/train_pairs.jsonl") == []
    # no partial artifacts should remain
    assert list((out_root / "pairs/encoded/train").glob("*.json")) == []
    assert list((out_root / "pairs/midi/train").glob("*.mid")) == []


def test_overwrite_false_requires_deterministic_pair_ids(tmp_path: Path):
    ds_path = tmp_path / "encoded.json"
    _write_json(ds_path, {"s1": _song("s1", "train")})
    cfg = _build_cfg(tmp_path, ds_path)
    cfg.observer_pipeline.overwrite = False
    cfg.dataloader.theory_aware.deterministic_per_sample = False

    with pytest.raises(Exception):
        build_pairs(cfg)


def test_pair_targets_join_strict_validation_and_finite_gap(tmp_path: Path):
    pair_path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        pair_path,
        [{"pair_group_id": "p", "clean_sample_id": "p::clean", "corrupted_sample_id": "p::corrupted", "is_valid_pair_for_rank": True}],
    )

    target_rows = [
        {"sample_id": "p::clean", "teacher_score": 1.0},
        {"sample_id": "p::corrupted", "teacher_score": 0.4},
    ]
    out = tmp_path / "out.jsonl"
    built, skipped = _join_pair_targets(target_rows, pair_path, out)
    assert built == 1 and skipped == 0
    assert _read_jsonl(out)[0]["teacher_score_gap"] == pytest.approx(0.6)

    with pytest.raises(PairTargetJoinError):
        _join_pair_targets([{"sample_id": "p::clean", "teacher_score": 1.0}], pair_path, out)


def test_build_pair_targets_uses_sample_id_passthrough_without_zip(tmp_path: Path, monkeypatch):
    # check source code invariant (no zip-based recovery)
    src = (REPO_ROOT / "src/observer/build_observer_pair_targets.py").read_text(encoding="utf-8")
    assert "zip(target_rows, rows)" not in src

    # minimal functional check with mocked teacher scorer
    out_root = tmp_path / "out"
    manifests = out_root / "pairs/manifests"
    index = out_root / "pairs/index"
    _write_jsonl(
        manifests / "train.jsonl",
        [
            {"sample_id": "p::clean", "song_id": "p::clean", "encoded_song_path": str(tmp_path / "c.json"), "pair_group_id": "p", "split": "train"},
            {"sample_id": "p::corrupted", "song_id": "p::corrupted", "encoded_song_path": str(tmp_path / "x.json"), "pair_group_id": "p", "split": "train"},
        ],
    )
    _write_jsonl(index / "train_pairs.jsonl", [{"pair_group_id": "p", "clean_sample_id": "p::clean", "corrupted_sample_id": "p::corrupted", "is_valid_pair_for_rank": True}])
    _write_json(tmp_path / "c.json", {"meta": {}})
    _write_json(tmp_path / "x.json", {"meta": {}})

    cfg = OmegaConf.create(
        {
            "observer_pipeline": {"output_root": str(out_root)},
            "observer_training": {"teacher_checkpoint": "models/best_rank_acc.pt", "teacher_config": "configs/config.yaml", "device": "cpu"},
            "data": {"split": {"train": "train"}},
        }
    )

    import src.observer.build_observer_pair_targets as module

    def fake_build_teacher_targets(rows, **kwargs):
        return [{"sample_id": r["sample_id"], "song_id": r["song_id"], "teacher_score": 1.0 if r["sample_id"].endswith("clean") else 0.5} for r in rows]

    monkeypatch.setattr(module, "build_teacher_targets", fake_build_teacher_targets)
    build_pair_targets(cfg)
    assert _read_jsonl(out_root / "targets/train.jsonl")[0]["sample_id"] == "p::clean"


def test_stale_target_cleanup(tmp_path: Path):
    out_root = tmp_path / "out"
    (out_root / "targets").mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_root / "targets/train.jsonl", [{"sample_id": "x", "teacher_score": 1.0}])
    _write_jsonl(out_root / "targets/train_pairs.jsonl", [{"pair_group_id": "x"}])
    cfg = OmegaConf.create({"observer_pipeline": {"output_root": str(out_root)}, "observer_training": {"teacher_checkpoint": "m.pt", "teacher_config": "c.yaml", "device": "cpu"}, "data": {"split": {"train": "train"}}})
    build_pair_targets(cfg)
    assert not (out_root / "targets/train.jsonl").exists()
    assert not (out_root / "targets/train_pairs.jsonl").exists()


def test_skip_log_dedup_on_rerun(tmp_path: Path):
    cfg = _build_cfg(tmp_path, tmp_path / "encoded.json")
    _write_json(tmp_path / "encoded.json", {"bad": {"meta": {"song_id": "bad"}, "melody": [], "chords": []}})
    build_pairs(cfg)
    build_pairs(cfg)
    rows = _read_jsonl(Path(cfg.observer_pipeline.output_root) / "pairs/skipped_manifest_rows.jsonl")
    assert len(rows) == 1


def _tiny_graph(y: float):
    g = HeteroData()
    g["song"].x_cat = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    g["song"].x_num = torch.tensor([[y]], dtype=torch.float)
    g["song"].num_nodes = 1
    g["note"].x_cat = torch.tensor([[1, 1]], dtype=torch.long)
    g["note"].x_num = torch.zeros((1, 4), dtype=torch.float)
    g["note"].num_nodes = 1
    g["chord"].x_cat = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
    g["chord"].x_num = torch.zeros((1, 18), dtype=torch.float)
    g["chord"].num_nodes = 1
    g["bar"].x_cat = torch.zeros((1, 0), dtype=torch.long)
    g["bar"].x_num = torch.zeros((1, 2), dtype=torch.float)
    g["bar"].num_nodes = 1
    g["onset"].x_cat = torch.zeros((1, 0), dtype=torch.long)
    g["onset"].x_num = torch.zeros((1, 2), dtype=torch.float)
    g["onset"].num_nodes = 1
    g[("song", "to", "song")].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    return g


def test_cached_dataset_pair_validation_and_rank_toggle_smoke(tmp_path: Path):
    g1 = _tiny_graph(1.0)
    p1 = tmp_path / "g1.pt"
    torch.save(g1, p1)
    _write_jsonl(tmp_path / "graph_index.jsonl", [{"sample_id": "p::clean", "graph_path": str(p1), "teacher_score": 1.0}])
    _write_jsonl(
        tmp_path / "pair_targets.jsonl",
        [{"pair_group_id": "p", "clean_sample_id": "p::clean", "corrupted_sample_id": "p::corrupted", "teacher_score_clean": 1.0, "teacher_score_corrupted": 0.0}],
    )

    with pytest.raises(ValueError):
        ObserverPairCachedDataset(tmp_path / "graph_index.jsonl", tmp_path / "pair_targets.jsonl", mode="pair")


def test_train_metrics_file_is_reset_on_rerun(tmp_path: Path, monkeypatch):
    import src.observer.train_observer_distill as train_mod

    out_root = tmp_path / "out"
    (out_root / "training").mkdir(parents=True, exist_ok=True)
    old_metrics = out_root / "training/metrics.jsonl"
    old_metrics.write_text("{\"epoch\":999}\n", encoding="utf-8")
    (out_root / "training/best.pt").write_text("old", encoding="utf-8")
    (out_root / "training/last.pt").write_text("old", encoding="utf-8")

    class FakeDataset:
        def __len__(self):
            return 1

    class FakeModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(1.0))

        def to(self, device):
            return self

    metrics_rows = [
        {"loss": 2.0, "reg_loss": 2.0, "rank_loss": 0.0, "mae": 0.0, "rmse": 0.0, "pearson": 0.0, "spearman": 0.0, "pair_rank_acc": 0.0, "mean_pred_margin": 0.0, "mean_teacher_margin": 0.0},
        {"loss": 1.0, "reg_loss": 1.0, "rank_loss": 0.0, "mae": 0.0, "rmse": 0.0, "pearson": 0.0, "spearman": 0.0, "pair_rank_acc": 0.0, "mean_pred_margin": 0.0, "mean_teacher_margin": 0.0},
    ]
    counter = {"i": 0}

    def fake_run_epoch(*args, **kwargs):
        row = metrics_rows[counter["i"] % len(metrics_rows)]
        counter["i"] += 1
        return row

    monkeypatch.setattr(train_mod, "ObserverPairCachedDataset", lambda *a, **k: FakeDataset())
    monkeypatch.setattr(train_mod, "DataLoader", lambda *a, **k: [object()])
    monkeypatch.setattr(train_mod, "ObserverGNN", FakeModel)
    monkeypatch.setattr(train_mod, "build_theory_context", lambda: {})
    monkeypatch.setattr(train_mod, "build_observer_vocab_sizes", lambda *a, **k: {"song": [2], "note": [2], "chord": [2], "bar": [], "onset": []})
    monkeypatch.setattr(train_mod, "_run_epoch", fake_run_epoch)

    cfg = OmegaConf.create(
        {
            "observer_training": {"seed": 1, "device": "cpu", "epochs": 1},
            "observer_pipeline": {"output_root": str(out_root)},
            "dataloader": {"batch_size": 1, "num_workers": 0},
            "observer_model": {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0},
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
            "losses": {"lambda_reg": 1.0, "lambda_rank": 0.0, "min_teacher_gap_for_rank": 0.25, "use_pair_rank": False},
        }
    )

    train_mod.train(cfg)
    lines = old_metrics.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "999" not in lines[0]
    assert (out_root / "training/best.pt").exists()
    assert (out_root / "training/last.pt").exists()


def test_stale_graph_cache_cleanup(tmp_path: Path):
    import src.observer.build_observer_graph_cache as cache_mod

    out_root = tmp_path / "out"
    stale_dir = out_root / "cache/graphs/train"
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "old.pt").write_text("x", encoding="utf-8")
    (stale_dir / "junk.txt").write_text("x", encoding="utf-8")
    (out_root / "cache/index").mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_root / "cache/index/train.jsonl", [{"sample_id": "old"}])

    cfg = OmegaConf.create({"observer_pipeline": {"output_root": str(out_root)}, "observer_training": {}, "data": {"split": {"train": "train"}}})
    cache_mod.build_graph_cache(cfg)
    assert not (out_root / "cache/index/train.jsonl").exists()
    assert not stale_dir.exists()


def test_stable_sorted_write_order(tmp_path: Path):
    cfg = _build_cfg(tmp_path, tmp_path / "encoded.json")
    _write_json(tmp_path / "encoded.json", {"b": _song("b", "train"), "a": _song("a", "train")})
    build_pairs(cfg)
    out_root = Path(cfg.observer_pipeline.output_root)
    manifest = _read_jsonl(out_root / "pairs/manifests/train.jsonl")
    pair_rows = _read_jsonl(out_root / "pairs/index/train_pairs.jsonl")
    assert [r["sample_id"] for r in manifest] == sorted(r["sample_id"] for r in manifest)
    assert [r["pair_group_id"] for r in pair_rows] == sorted(r["pair_group_id"] for r in pair_rows)


def test_custom_pipeline_paths_are_used(tmp_path: Path):
    cfg = _build_cfg(tmp_path, tmp_path / "encoded.json")
    cfg.observer_pipeline.encoded_output_dir = "custom_pairs/enc"
    cfg.observer_pipeline.midi_output_dir = "custom_pairs/mid"
    cfg.observer_pipeline.manifest_output_dir = "custom_pairs/mf"
    _write_json(tmp_path / "encoded.json", {"s1": _song("s1", "train")})
    build_pairs(cfg)
    out_root = Path(cfg.observer_pipeline.output_root)
    assert (out_root / "custom_pairs/mf/train.jsonl").exists()
    rows = _read_jsonl(out_root / "custom_pairs/mf/train.jsonl")
    assert all("custom_pairs/enc" in r["encoded_song_path"] for r in rows)
    assert all("custom_pairs/mid" in r["midi_path"] for r in rows)


def test_dataloader_flags_are_forwarded(tmp_path: Path, monkeypatch):
    import src.observer.train_observer_distill as train_mod

    captured = {"train": None, "val": None}

    class FakeDataset:
        def __len__(self):
            return 1

    class FakeModel(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(1.0))

        def to(self, d):
            return self

    def fake_loader(dataset, **kwargs):
        if captured["train"] is None:
            captured["train"] = kwargs
        else:
            captured["val"] = kwargs
        return [object()]

    monkeypatch.setattr(train_mod, "ObserverPairCachedDataset", lambda *a, **k: FakeDataset())
    monkeypatch.setattr(train_mod, "ObserverGNN", FakeModel)
    monkeypatch.setattr(train_mod, "DataLoader", fake_loader)
    monkeypatch.setattr(train_mod, "build_theory_context", lambda: {})
    monkeypatch.setattr(train_mod, "build_observer_vocab_sizes", lambda *a, **k: {"song": [2], "note": [2], "chord": [2], "bar": [], "onset": []})
    monkeypatch.setattr(train_mod, "_run_epoch", lambda *a, **k: {"loss": 1.0, "reg_loss": 1.0, "rank_loss": 0.0, "mae": 0.0, "rmse": 0.0, "pearson": 0.0, "spearman": 0.0, "pair_rank_acc": 0.0, "mean_pred_margin": 0.0, "mean_teacher_margin": 0.0})

    cfg = OmegaConf.create(
        {
            "observer_training": {"seed": 1, "device": "cpu", "epochs": 1, "resume": False},
            "observer_pipeline": {"output_root": str(tmp_path / "out"), "cache_output_dir": "cache", "targets_output_dir": "targets"},
            "dataloader": {"batch_size": 1, "num_workers": 0, "shuffle": True, "pin_memory": True, "drop_last": True},
            "observer_model": {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0},
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
            "losses": {"lambda_reg": 1.0, "lambda_rank": 0.0, "min_teacher_gap_for_rank": 0.25, "use_pair_rank": False},
        }
    )
    train_mod.train(cfg)
    assert captured["train"]["batch_size"] == 1
    assert captured["train"]["shuffle"] is True
    assert captured["train"]["num_workers"] == 0
    assert captured["train"]["pin_memory"] is True
    assert captured["train"]["drop_last"] is True
    assert captured["val"]["batch_size"] == 1
    assert captured["val"]["shuffle"] is False
    assert captured["val"]["num_workers"] == 0
    assert captured["val"]["pin_memory"] is True
    assert captured["val"]["drop_last"] is False


def test_sample_weighted_metrics_not_batch_weighted():
    class DummyModel(torch.nn.Module):
        def forward(self, batch):
            return batch["song"].x_num.view(-1)

    def mk_batch(clean_vals: list[float], corrupted_vals: list[float], teacher_clean: list[float], teacher_corrupted: list[float]):
        items = []
        for c_val, x_val, t_c, t_x in zip(clean_vals, corrupted_vals, teacher_clean, teacher_corrupted, strict=True):
            g_clean = _tiny_graph(c_val)
            g_corr = _tiny_graph(x_val)
            items.append(
                {
                    "graph_clean": g_clean,
                    "graph_corrupted": g_corr,
                    "teacher_score_clean": t_c,
                    "teacher_score_corrupted": t_x,
                    "pair_metadata": {},
                }
            )
        return _collate_pairs(items)

    batch1 = mk_batch(
        clean_vals=[0.0],
        corrupted_vals=[0.0],
        teacher_clean=[4.0],
        teacher_corrupted=[0.0],
    )
    batch2 = mk_batch(
        clean_vals=[0.0, 0.0, 0.0],
        corrupted_vals=[0.0, 0.0, 0.0],
        teacher_clean=[0.0, 0.0, 0.0],
        teacher_corrupted=[0.0, 0.0, 0.0],
    )

    metrics = _run_epoch(
        DummyModel(),
        [batch1, batch2],
        optimizer=None,
        device=torch.device("cpu"),
        cfg_losses=OmegaConf.create({"lambda_reg": 1.0, "lambda_rank": 0.0, "min_teacher_gap_for_rank": 0.25, "use_pair_rank": False}),
    )
    # Batch means are [3.5, 0.0]; old buggy averaging yields 1.75.
    # Correct sample-weighted average: (3.5*1 + 0*3)/4 = 0.875.
    assert metrics["reg_loss"] == pytest.approx(0.875)
    assert metrics["loss"] == pytest.approx(0.875)
    assert metrics["pair_rank_acc"] != metrics["pair_rank_acc"]  # nan


def test_run_epoch_uses_cached_graph_embedding_distillation_targets():
    class DummyDistillModel(torch.nn.Module):
        def forward(self, batch, *, return_outputs: bool = False):
            score = batch["song"].x_num.view(-1)
            if not return_outputs:
                return score
            return {
                "score": score,
                "graph_embedding": torch.zeros((score.numel(), 2), dtype=torch.float, device=score.device),
                "pooled_by_type": {},
            }

    items = []
    for _ in range(2):
        items.append(
            {
                "graph_clean": _tiny_graph(0.0),
                "graph_corrupted": _tiny_graph(0.0),
                "teacher_score_clean": 0.0,
                "teacher_score_corrupted": 0.0,
                "teacher_distill_clean": {"teacher_graph_embedding": [1.0, 1.0]},
                "teacher_distill_corrupted": {"teacher_graph_embedding": [1.0, 1.0]},
                "pair_metadata": {},
            }
        )

    adapters = ObserverDistillationAdapters(
        observer_graph_dim=2,
        observer_node_dim=2,
        target_dims={"teacher_graph_embedding": 2, "teacher_pooled_by_type": {}, "teacher_local_score_summaries": None},
    )
    metrics = _run_epoch(
        DummyDistillModel(),
        [_collate_pairs(items)],
        optimizer=None,
        device=torch.device("cpu"),
        cfg_losses=OmegaConf.create(
            {
                "lambda_reg": 0.0,
                "lambda_rank": 0.0,
                "lambda_graph_embedding_distill": 1.0,
                "lambda_node_type_embedding_distill": 0.0,
                "lambda_local_summary_distill": 0.0,
                "min_teacher_gap_for_rank": 0.25,
                "use_pair_rank": False,
            }
        ),
        adapters=adapters,
    )
    assert metrics["graph_embedding_distill_loss"] == pytest.approx(1.0)
    assert metrics["loss"] == pytest.approx(1.0)


def test_early_fail_on_empty_pairs(tmp_path: Path):
    import src.observer.run_observer_pipeline as pipe_mod
    cfg = OmegaConf.create({"observer_pipeline": {"output_root": str(tmp_path), "manifest_output_dir": "pairs/manifests"}, "data": {"split": {"train": "train", "val": "val"}}})
    with pytest.raises(ValueError):
        pipe_mod._validate_after_pairs(cfg, tmp_path)


def test_resume_mode_loads_checkpoint(tmp_path: Path, monkeypatch):
    import src.observer.train_observer_distill as train_mod
    out_root = tmp_path / "out"
    (out_root / "training").mkdir(parents=True, exist_ok=True)
    last = out_root / "training/last.pt"
    torch.save({"model_state_dict": {}, "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 1e-3, "params": [0]}]}, "epoch": 1, "best_val_loss": 1.0}, last)
    metrics = out_root / "training/metrics.jsonl"
    metrics.write_text('{"epoch":1,"train":{"loss":1.2},"val":{"loss":1.1}}\n', encoding="utf-8")

    class FakeDataset:
        def __len__(self):
            return 1

    class FakeModel(torch.nn.Module):
        loaded_payload = None

        def __init__(self, *a, **k):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(1.0))

        def to(self, d):
            return self

        def load_state_dict(self, state):
            FakeModel.loaded_payload = state
            return None

    loaded_optimizer_state = {"called": False}

    monkeypatch.setattr(train_mod, "ObserverPairCachedDataset", lambda *a, **k: FakeDataset())
    monkeypatch.setattr(train_mod, "ObserverGNN", FakeModel)
    monkeypatch.setattr(train_mod, "DataLoader", lambda *a, **k: [object()])
    monkeypatch.setattr(train_mod, "build_theory_context", lambda: {})
    monkeypatch.setattr(train_mod, "build_observer_vocab_sizes", lambda *a, **k: {"song": [2], "note": [2], "chord": [2], "bar": [], "onset": []})
    monkeypatch.setattr(
        train_mod,
        "_run_epoch",
        lambda *a, **k: {"loss": 1.0, "reg_loss": 1.0, "rank_loss": 0.0, "mae": 0.0, "rmse": 0.0, "pearson": 0.0, "spearman": 0.0, "pair_rank_acc": 0.0, "mean_pred_margin": 0.0, "mean_teacher_margin": 0.0},
    )

    real_adamw = torch.optim.AdamW

    class TrackingAdamW(real_adamw):
        def load_state_dict(self, state_dict):
            loaded_optimizer_state["called"] = True
            return super().load_state_dict(state_dict)

    monkeypatch.setattr(train_mod.torch.optim, "AdamW", TrackingAdamW)
    monkeypatch.setattr(train_mod, "_save_checkpoint", lambda *a, **k: None)

    cfg = OmegaConf.create({"observer_training": {"seed": 1, "device": "cpu", "epochs": 3, "resume": True}, "observer_pipeline": {"output_root": str(out_root), "cache_output_dir": "cache", "targets_output_dir": "targets"}, "dataloader": {"batch_size": 1, "num_workers": 0, "shuffle": True}, "observer_model": {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0}, "optimizer": {"lr": 1e-3, "weight_decay": 0.0}, "losses": {"lambda_reg": 1.0, "lambda_rank": 0.0, "min_teacher_gap_for_rank": 0.25, "use_pair_rank": False}})
    train_mod.train(cfg)
    lines = metrics.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    rows = [json.loads(x) for x in lines]
    assert rows[1]["epoch"] == 2
    assert rows[2]["epoch"] == 3
    assert loaded_optimizer_state["called"] is True
    assert FakeModel.loaded_payload == {}


def test_resume_mode_requires_last_checkpoint(tmp_path: Path):
    import src.observer.train_observer_distill as train_mod

    out_root = tmp_path / "out"
    (out_root / "training").mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create({"observer_training": {"seed": 1, "device": "cpu", "epochs": 3, "resume": True}, "observer_pipeline": {"output_root": str(out_root), "cache_output_dir": "cache", "targets_output_dir": "targets"}, "dataloader": {"batch_size": 1, "num_workers": 0, "shuffle": True}, "observer_model": {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0}, "optimizer": {"lr": 1e-3, "weight_decay": 0.0}, "losses": {"lambda_reg": 1.0, "lambda_rank": 0.0, "min_teacher_gap_for_rank": 0.25, "use_pair_rank": False}})

    with pytest.raises(ValueError, match="last\\.pt does not exist"):
        train_mod.train(cfg)


def test_resume_mode_overflow_epoch_fails(tmp_path: Path, monkeypatch):
    import src.observer.train_observer_distill as train_mod

    out_root = tmp_path / "out"
    (out_root / "training").mkdir(parents=True, exist_ok=True)
    last = out_root / "training/last.pt"
    torch.save({"model_state_dict": {}, "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 1e-3, "params": [0]}]}, "epoch": 3, "best_val_loss": 1.0}, last)

    class FakeDataset:
        def __len__(self):
            return 1

    class FakeModel(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(1.0))

        def to(self, d):
            return self

        def load_state_dict(self, state):
            return None

    monkeypatch.setattr(train_mod, "ObserverPairCachedDataset", lambda *a, **k: FakeDataset())
    monkeypatch.setattr(train_mod, "ObserverGNN", FakeModel)
    monkeypatch.setattr(train_mod, "DataLoader", lambda *a, **k: [object()])
    monkeypatch.setattr(train_mod, "build_theory_context", lambda: {})
    monkeypatch.setattr(train_mod, "build_observer_vocab_sizes", lambda *a, **k: {"song": [2], "note": [2], "chord": [2], "bar": [], "onset": []})

    real_adamw = torch.optim.AdamW

    class TrackingAdamW(real_adamw):
        pass

    monkeypatch.setattr(train_mod.torch.optim, "AdamW", TrackingAdamW)

    cfg = OmegaConf.create({"observer_training": {"seed": 1, "device": "cpu", "epochs": 3, "resume": True}, "observer_pipeline": {"output_root": str(out_root), "cache_output_dir": "cache", "targets_output_dir": "targets"}, "dataloader": {"batch_size": 1, "num_workers": 0, "shuffle": True}, "observer_model": {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0}, "optimizer": {"lr": 1e-3, "weight_decay": 0.0}, "losses": {"lambda_reg": 1.0, "lambda_rank": 0.0, "min_teacher_gap_for_rank": 0.25, "use_pair_rank": False}})
    with pytest.raises(ValueError, match="resume checkpoint epoch already exceeds configured epochs"):
        train_mod.train(cfg)


def test_cached_dataset_rank_toggle_and_nonfinite_guard(tmp_path: Path):
    g1 = _tiny_graph(1.0)
    g2 = _tiny_graph(0.0)
    p1 = tmp_path / "g1.pt"
    p2 = tmp_path / "g2.pt"
    torch.save(g1, p1)
    torch.save(g2, p2)
    _write_jsonl(
        tmp_path / "graph_index.jsonl",
        [
            {"sample_id": "p::clean", "graph_path": str(p1), "teacher_score": 1.0},
            {"sample_id": "p::corrupted", "graph_path": str(p2), "teacher_score": 0.0},
        ],
    )
    _write_jsonl(
        tmp_path / "pair_targets.jsonl",
        [{"pair_group_id": "p", "clean_sample_id": "p::clean", "corrupted_sample_id": "p::corrupted", "teacher_score_clean": 1.0, "teacher_score_corrupted": 0.0}],
    )
    ds = ObserverPairCachedDataset(tmp_path / "graph_index.jsonl", tmp_path / "pair_targets.jsonl", mode="pair")

    class Dummy(torch.nn.Module):
        def forward(self, batch):
            return batch["song"].x_num.view(-1)

    loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=_collate_pairs)
    cfg_losses = OmegaConf.create({"lambda_reg": 1.0, "lambda_rank": 0.5, "min_teacher_gap_for_rank": 0.25, "use_pair_rank": False})
    metrics = _run_epoch(Dummy(), loader, optimizer=None, device=torch.device("cpu"), cfg_losses=cfg_losses)
    assert metrics["rank_loss"] == 0.0

    _write_jsonl(
        tmp_path / "pair_targets.jsonl",
        [{"pair_group_id": "p", "clean_sample_id": "p::clean", "corrupted_sample_id": "p::corrupted", "teacher_score_clean": float("nan"), "teacher_score_corrupted": 0.0}],
    )
    with pytest.raises(ValueError):
        ObserverPairCachedDataset(tmp_path / "graph_index.jsonl", tmp_path / "pair_targets.jsonl", mode="pair")
