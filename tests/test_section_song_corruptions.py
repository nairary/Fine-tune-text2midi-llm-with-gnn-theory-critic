from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader.song_corruptions import classify_chord_function_root_only, corrupt_song_obj
from src.dataloader.theory_helpers import build_theory_context
from src.dataloader.utils_graph import build_graph_from_encoded


def _note(beat: float, sd_id: int) -> dict:
    return {"beat": beat, "duration": 1.0, "sd_id": sd_id, "octave_id": 5, "is_rest": 0}


def _chord(beat: float, root_id: int, root_raw: int) -> dict:
    return {
        "beat": beat,
        "duration": 2.0,
        "root_id": root_id,
        "root_degree_raw": root_raw,
        "type_id": 1,
        "type_raw": 5,
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


def _section_song() -> dict:
    return {
        "song_id": "assembled_test",
        "meta": {
            "main_key_scale_id": 2,
            "main_num_beats": 4,
            "main_beat_unit": 1,
            "main_bpm": 120.0,
            "end_beat": 13.0,
            "section_spans": [
                {"section_index": 0, "label": "verse", "labels": ["verse"], "target_start_beat": 1.0, "target_end_beat": 5.0, "duration_beats": 4.0},
                {"section_index": 1, "label": "pre-chorus", "labels": ["pre-chorus"], "target_start_beat": 5.0, "target_end_beat": 9.0, "duration_beats": 4.0},
                {"section_index": 2, "label": "chorus", "labels": ["chorus"], "target_start_beat": 9.0, "target_end_beat": 13.0, "duration_beats": 4.0},
            ],
        },
        "melody": [
            _note(1.0, 4),
            _note(5.0, 5),
            _note(9.0, 6),
        ],
        "chords": [
            _chord(1.0, 1, 0),
            _chord(3.0, 5, 4),
            _chord(5.0, 1, 0),
            _chord(7.0, 5, 4),
            _chord(9.0, 1, 0),
            _chord(11.0, 5, 4),
        ],
    }


def _apply(song: dict, mode: str, cfg: dict | None = None) -> tuple[dict, dict]:
    return corrupt_song_obj(
        song_obj=copy.deepcopy(song),
        corruption_modes=[mode],
        corruption_cfg=cfg or {},
        theory_ctx=build_theory_context(),
        rng=random.Random(3),
    )


def _labels(song: dict) -> list[str]:
    return [span["label"] for span in song["meta"]["section_spans"]]


def test_adjacent_section_swap_rebuilds_order_and_event_beats():
    corrupted, metadata = _apply(_section_song(), "adjacent_section_swap", {"section_swap_left_index": 0})

    assert metadata["applied"]
    assert _labels(corrupted) == ["pre-chorus", "verse", "chorus"]
    assert [note["sd_id"] for note in corrupted["melody"]] == [5, 4, 6]
    assert [note["beat"] for note in corrupted["melody"]] == [1.0, 5.0, 9.0]
    assert corrupted["meta"]["end_beat"] == 13.0


def test_non_adjacent_section_swap_rebuilds_order():
    corrupted, metadata = _apply(_section_song(), "non_adjacent_section_swap", {"section_swap_indices": [0, 2]})

    assert metadata["applied"]
    assert _labels(corrupted) == ["chorus", "pre-chorus", "verse"]
    assert [note["sd_id"] for note in corrupted["melody"]] == [6, 5, 4]


def test_section_duplicate_can_repeat_selected_section_multiple_times():
    corrupted, metadata = _apply(
        _section_song(),
        "section_duplicate",
        {"section_duplicate_index": 1, "section_duplicate_times": 2, "section_duplicate_max_times": 2},
    )

    assert metadata["applied"]
    assert _labels(corrupted) == ["verse", "pre-chorus", "pre-chorus", "pre-chorus", "chorus"]
    assert [note["sd_id"] for note in corrupted["melody"]] == [4, 5, 5, 5, 6]
    assert [note["beat"] for note in corrupted["melody"]] == [1.0, 5.0, 9.0, 13.0, 17.0]
    assert corrupted["meta"]["end_beat"] == 21.0


def test_section_drop_keep_silence_removes_events_without_shifting_later_sections():
    corrupted, metadata = _apply(_section_song(), "section_drop_keep_silence", {"section_drop_index": 1})

    assert metadata["applied"]
    assert _labels(corrupted) == ["verse", "pre-chorus", "chorus"]
    assert [note["sd_id"] for note in corrupted["melody"]] == [4, 6]
    assert [note["beat"] for note in corrupted["melody"]] == [1.0, 9.0]
    assert [chord["beat"] for chord in corrupted["chords"]] == [1.0, 3.0, 9.0, 11.0]
    assert corrupted["meta"]["end_beat"] == 13.0


def test_section_drop_and_close_gap_removes_section_and_shifts_later_sections():
    corrupted, metadata = _apply(_section_song(), "section_drop_and_close_gap", {"section_drop_index": 1})

    assert metadata["applied"]
    assert _labels(corrupted) == ["verse", "chorus"]
    assert [note["sd_id"] for note in corrupted["melody"]] == [4, 6]
    assert [note["beat"] for note in corrupted["melody"]] == [1.0, 5.0]
    assert corrupted["meta"]["section_spans"][1]["target_start_beat"] == 5.0
    assert corrupted["meta"]["end_beat"] == 9.0


def test_section_entry_non_tonic_substitution_changes_first_tonic_chord():
    song = _section_song()
    corrupted, metadata = _apply(song, "section_entry_non_tonic_substitution", {"section_boundary_index": 0})
    ctx = build_theory_context()

    assert metadata["applied"]
    assert metadata["chord_corrupted_indices"] == [0]
    assert corrupted["chords"][0]["root_id"] != song["chords"][0]["root_id"]
    assert classify_chord_function_root_only(corrupted, corrupted["chords"][0], ctx)["slot"] != "T"


def test_section_exit_non_dominant_substitution_changes_last_dominant_chord():
    song = _section_song()
    corrupted, metadata = _apply(song, "section_exit_non_dominant_substitution", {"section_boundary_index": 0})
    ctx = build_theory_context()

    assert metadata["applied"]
    assert metadata["chord_corrupted_indices"] == [1]
    assert corrupted["chords"][1]["root_id"] != song["chords"][1]["root_id"]
    assert classify_chord_function_root_only(corrupted, corrupted["chords"][1], ctx)["slot"] != "D"


def test_section_corruptions_change_teacher_graph_schema_surface():
    song = _section_song()
    original = build_graph_from_encoded(song)

    for mode, cfg in {
        "adjacent_section_swap": {"section_swap_left_index": 0},
        "section_duplicate": {"section_duplicate_index": 1, "section_duplicate_times": 1},
        "section_drop_keep_silence": {"section_drop_index": 1},
        "section_drop_and_close_gap": {"section_drop_index": 1},
        "section_entry_non_tonic_substitution": {"section_boundary_index": 0},
        "section_exit_non_dominant_substitution": {"section_boundary_index": 0},
    }.items():
        corrupted, metadata = _apply(song, mode, cfg)
        assert metadata["applied"], mode
        mutated = build_graph_from_encoded(corrupted)
        changed = False
        for node_type in ("section", "note", "chord"):
            if original[node_type].x.shape != mutated[node_type].x.shape:
                changed = True
            elif not torch.equal(original[node_type].x, mutated[node_type].x):
                changed = True
        assert changed, mode
