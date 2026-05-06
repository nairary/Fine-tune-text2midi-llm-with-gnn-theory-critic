from __future__ import annotations

import json
from pathlib import Path

from src.dataloader.build_teacher_pair_graph_cache import build_teacher_pair_graph_cache
from src.dataloader.precomputed_teacher_pairs import PrecomputedTeacherPairDataset


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _song(song_id: str, *, sd_id: int = 1) -> dict:
    return {
        "meta": {
            "song_id": song_id,
            "split": "train",
            "main_key_tonic_pc": 0,
            "main_key_scale_id": 3,
            "main_bpm": 120,
            "main_num_beats": 4,
            "main_beat_unit": 1,
        },
        "melody": [{"beat": 1.0, "duration": 1.0, "sd_id": sd_id, "octave_id": 6, "is_rest": 0}],
        "chords": [{"beat": 1.0, "duration": 1.0, "root_id": 1, "type_id": 1, "inversion_id": 1, "is_rest": 0}],
    }


def _build_pair_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "pair_corpus"
    clean_path = root / "pairs/encoded/train/pair1::clean.json"
    corr_path = root / "pairs/encoded/train/pair1::corrupted.json"
    _write_json(clean_path, _song("pair1::clean", sd_id=1))
    _write_json(corr_path, _song("pair1::corrupted", sd_id=2))

    manifest_rows = [
        {
            "sample_id": "pair1::clean",
            "song_id": "pair1::clean",
            "source_song_id": "s1",
            "pair_group_id": "pair1",
            "split": "train",
            "is_corrupted": False,
            "corruption_name": "identity",
            "encoded_song_path": str(clean_path),
            "midi_path": str(root / "pairs/midi/train/pair1::clean.mid"),
        },
        {
            "sample_id": "pair1::corrupted",
            "song_id": "pair1::corrupted",
            "source_song_id": "s1",
            "pair_group_id": "pair1",
            "split": "train",
            "is_corrupted": True,
            "corruption_name": "strongbeat_nonchord_note",
            "encoded_song_path": str(corr_path),
            "midi_path": str(root / "pairs/midi/train/pair1::corrupted.mid"),
            "note_corrupted_indices": [0],
            "attempted_corruption_modes": ["strongbeat_nonchord_note"],
            "skipped_corruption_attempts": [],
        },
    ]
    _write_jsonl(root / "pairs/manifests/train.jsonl", manifest_rows)
    _write_jsonl(
        root / "pairs/index/train_pairs.jsonl",
        [
            {
                "pair_group_id": "pair1",
                "split": "train",
                "source_song_id": "s1",
                "clean_sample_id": "pair1::clean",
                "corrupted_sample_id": "pair1::corrupted",
                "corruption_name": "strongbeat_nonchord_note",
                "is_valid_pair_for_rank": True,
            }
        ],
    )
    return root


def test_build_teacher_pair_graph_cache_and_train_dataset_loads_cached_graphs(tmp_path: Path, monkeypatch):
    root = _build_pair_corpus(tmp_path)

    stats = build_teacher_pair_graph_cache(pair_corpus_root=root, splits=["train"], overwrite=True)

    assert stats[0].built == 2
    index_path = root / "teacher_graphs/index/train.jsonl"
    assert index_path.exists()

    import src.dataloader.precomputed_teacher_pairs as pair_dataset_module

    def fail_build_graph_from_encoded(*args, **kwargs):
        raise AssertionError("dataset should load cached graphs, not rebuild from encoded JSON")

    monkeypatch.setattr(pair_dataset_module, "build_graph_from_encoded", fail_build_graph_from_encoded)
    dataset = PrecomputedTeacherPairDataset(
        pair_index_jsonl=root / "pairs/index/train_pairs.jsonl",
        manifest_jsonl=root / "pairs/manifests/train.jsonl",
        graph_index_jsonl=index_path,
    )

    item = dataset[0]
    assert item["graph_real"]["song"].x.size(0) == 1
    assert item["graph_corrupted"]["song"].x.size(0) == 1
    assert item["corruption_metadata"]["corruption_name"] == "strongbeat_nonchord_note"
    assert item["corruption_metadata"]["attempted_corruption_modes"] == ["strongbeat_nonchord_note"]
