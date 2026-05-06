from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.assemble_timeline_songs import assemble_from_audit_rows, write_assembly_outputs


def _clip(song_id: str, split: str = "train", end_beat: float = 9.0) -> dict:
    return {
        "song_id": song_id,
        "meta": {
            "song_id": song_id,
            "split": split,
            "ori_uid": "ori_a",
            "end_beat": end_beat,
            "main_bpm": 120.0,
            "main_num_beats": 4,
            "main_beat_unit": 1,
            "key_regions": [{"beat": 1.0, "tonic_pc": 0}],
            "tempo_regions": [{"beat": 1.0, "bpm": 120.0}],
            "meter_regions": [{"beat": 1.0, "num_beats": 4, "beat_unit": 1}],
        },
        "melody": [
            {"beat": 1.0, "duration": 1.0, "sd_id": 4, "octave_id": 5, "is_rest": 0},
            {"beat": 2.0, "duration": 1.0, "sd_id": 5, "octave_id": 5, "is_rest": 0},
        ],
        "chords": [
            {
                "beat": 1.0,
                "duration": 4.0,
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
        ],
    }


def _audit_row(ori_uid: str, clip_ids: list[str], labels: list[str], gap_seconds: float = 0.0) -> dict:
    timeline = []
    cursor = 0.0
    for index, (clip_id, label) in enumerate(zip(clip_ids, labels)):
        if index > 0:
            cursor += gap_seconds
        timeline.append(
            {
                "index": index,
                "segment_start_seconds": cursor,
                "segment_end_seconds": cursor + 10.0,
                "duration_seconds": 10.0,
                "labels": [label],
                "clip_song_ids": [clip_id],
                "splits": ["train"],
            }
        )
        cursor += 10.0
    return {
        "ori_uid": ori_uid,
        "section_count": len(timeline),
        "usable_strict": True,
        "usable_compact_gap": True,
        "timeline": timeline,
    }


def test_compact_assembly_shifts_events_and_writes_section_spans(tmp_path: Path):
    encoded_payload = {"clip_a": _clip("clip_a"), "clip_b": _clip("clip_b")}
    audit_rows = [_audit_row("ori_a", ["clip_a", "clip_b"], ["verse", "chorus"])]

    assembled, manifest_rows, skipped_rows, summary = assemble_from_audit_rows(audit_rows, encoded_payload)

    assert not skipped_rows
    assert summary["assembled_song_count"] == 1
    assert summary["transition_counts"]["verse->chorus"] == 1
    assert manifest_rows[0]["labels"] == ["verse", "chorus"]

    song = assembled["assembled_ori_a"]
    assert song["song_id"] == "assembled_ori_a"
    assert song["meta"]["assembled_from_timeline"] is True
    assert song["meta"]["end_beat"] == 17.0
    assert [span["label"] for span in song["meta"]["section_spans"]] == ["verse", "chorus"]
    assert song["meta"]["section_spans"][0]["target_start_beat"] == 1.0
    assert song["meta"]["section_spans"][0]["target_end_beat"] == 9.0
    assert song["meta"]["section_spans"][1]["target_start_beat"] == 9.0
    assert song["meta"]["section_spans"][1]["target_end_beat"] == 17.0

    assert [event["beat"] for event in song["melody"]] == [1.0, 2.0, 9.0, 10.0]
    assert [event["source_clip_song_id"] for event in song["melody"]] == ["clip_a", "clip_a", "clip_b", "clip_b"]
    assert [chord["beat"] for chord in song["chords"]] == [1.0, 9.0]
    assert [region["beat"] for region in song["meta"]["key_regions"]] == [1.0, 9.0]

    outdir = tmp_path / "assembled"
    write_assembly_outputs(assembled, manifest_rows, skipped_rows, summary, outdir, usable_mode="strict")
    written = json.loads((outdir / "teacher_encoded_assembled_strict.json").read_text(encoding="utf-8"))
    assert list(written) == ["assembled_ori_a"]
    assert (outdir / "assembled_manifest_strict.jsonl").exists()
    assert (outdir / "summary_strict.json").exists()


def test_assembly_skips_multi_clip_segment_by_default():
    encoded_payload = {"clip_a": _clip("clip_a"), "clip_b": _clip("clip_b"), "clip_c": _clip("clip_c")}
    row = _audit_row("ori_multi", ["clip_a", "clip_c"], ["verse", "chorus"])
    row["timeline"][1]["clip_song_ids"] = ["clip_b", "clip_c"]

    assembled, manifest_rows, skipped_rows, summary = assemble_from_audit_rows([row], encoded_payload)

    assert assembled == {}
    assert manifest_rows == []
    assert summary["assembled_song_count"] == 0
    assert skipped_rows[0]["reason"] == "segment has 2 clip_song_ids"


def test_next_bar_gap_aligns_short_gap_to_next_bar():
    encoded_payload = {
        "clip_a": _clip("clip_a", end_beat=8.0),
        "clip_b": _clip("clip_b", end_beat=9.0),
    }
    audit_rows = [_audit_row("ori_short_gap", ["clip_a", "clip_b"], ["verse", "chorus"], gap_seconds=0.2)]

    assembled, _, skipped_rows, _ = assemble_from_audit_rows(
        audit_rows,
        encoded_payload,
        section_start_policy="next_bar_gap",
        max_gap_sec=10.0,
    )

    assert not skipped_rows
    spans = assembled["assembled_ori_short_gap"]["meta"]["section_spans"]
    assert spans[0]["target_end_beat"] == 8.0
    assert spans[1]["target_start_beat"] == 9.0
    assert spans[1]["inserted_gap_beats_before"] == 1.0
    assert spans[1]["extra_full_gap_bars_before"] == 0


def test_next_bar_gap_preserves_long_gap_as_whole_empty_bars():
    encoded_payload = {"clip_a": _clip("clip_a"), "clip_b": _clip("clip_b")}
    audit_rows = [_audit_row("ori_long_gap", ["clip_a", "clip_b"], ["verse", "chorus"], gap_seconds=8.0)]

    assembled, _, skipped_rows, summary = assemble_from_audit_rows(
        audit_rows,
        encoded_payload,
        usable_mode="compact_gap",
        section_start_policy="next_bar_gap",
        max_gap_sec=10.0,
    )

    assert not skipped_rows
    spans = assembled["assembled_ori_long_gap"]["meta"]["section_spans"]
    assert spans[0]["target_end_beat"] == 9.0
    assert spans[1]["target_start_beat"] == 25.0
    assert spans[1]["extra_full_gap_bars_before"] == 4
    assert spans[1]["inserted_gap_beats_before"] == 16.0
    assert assembled["assembled_ori_long_gap"]["meta"]["end_beat"] == 33.0
    assert summary["extra_full_gap_bars"] == 4


def test_assembly_skips_gap_above_max_gap_sec():
    encoded_payload = {"clip_a": _clip("clip_a"), "clip_b": _clip("clip_b")}
    audit_rows = [_audit_row("ori_too_long_gap", ["clip_a", "clip_b"], ["verse", "chorus"], gap_seconds=10.5)]

    assembled, manifest_rows, skipped_rows, summary = assemble_from_audit_rows(
        audit_rows,
        encoded_payload,
        usable_mode="compact_gap",
        section_start_policy="next_bar_gap",
        max_gap_sec=10.0,
    )

    assert assembled == {}
    assert manifest_rows == []
    assert summary["assembled_song_count"] == 0
    assert skipped_rows[0]["reason"] == "gap exceeds max_gap_sec"
