from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_original_song_timeline import audit_timeline, write_audit_outputs


def _segment(start: float, end: float, label: str, clip_id: str, split: str = "train") -> dict:
    return {
        "segment_start_seconds": start,
        "segment_end_seconds": end,
        "duration_seconds": round(end - start, 3),
        "labels": [label],
        "clip_song_ids": [clip_id],
        "splits": [split],
    }


def test_audit_classifies_gaps_overlaps_and_usable_sets(tmp_path: Path):
    timeline_payload = {
        "safe_song": {
            "timeline": [
                _segment(0.0, 10.0, "verse", "clip_a"),
                _segment(10.0, 20.0, "chorus", "clip_b"),
            ]
        },
        "large_gap_song": {
            "timeline": [
                _segment(0.0, 10.0, "verse", "clip_c"),
                _segment(15.0, 25.0, "chorus", "clip_d"),
            ]
        },
        "large_overlap_song": {
            "timeline": [
                _segment(0.0, 10.0, "intro", "clip_e"),
                _segment(8.0, 18.0, "verse", "clip_f"),
            ]
        },
        "missing_clip_song": {"timeline": [_segment(0.0, 8.0, "verse", "missing_clip")]},
    }
    encoded_payload = {f"clip_{letter}": {"song_id": f"clip_{letter}"} for letter in "abcdef"}

    rows, summary = audit_timeline(
        timeline_payload,
        encoded_payload,
        near_touch_sec=0.01,
        small_gap_sec=2.0,
        small_overlap_sec=0.25,
    )

    by_uid = {row["ori_uid"]: row for row in rows}

    assert by_uid["safe_song"]["usable_strict"] is True
    assert by_uid["safe_song"]["usable_compact_gap"] is True
    assert "safe_multisection" in by_uid["safe_song"]["buckets"]
    assert by_uid["safe_song"]["neighbor_relations"][0]["transition"] == "verse->chorus"

    assert by_uid["large_gap_song"]["usable_strict"] is False
    assert by_uid["large_gap_song"]["usable_compact_gap"] is True
    assert "large_gap" in by_uid["large_gap_song"]["buckets"]

    assert by_uid["large_overlap_song"]["usable_strict"] is False
    assert by_uid["large_overlap_song"]["usable_compact_gap"] is False
    assert "large_overlap" in by_uid["large_overlap_song"]["buckets"]

    assert by_uid["missing_clip_song"]["missing_clip_ids"] == ["missing_clip"]
    assert "missing_clip" in by_uid["missing_clip_song"]["buckets"]

    assert summary["original_song_count"] == 4
    assert summary["usable_strict_count"] == 1
    assert summary["usable_compact_gap_count"] == 2
    assert summary["transition_counts"]["verse->chorus"] == 2

    outdir = tmp_path / "audit"
    write_audit_outputs(rows, summary, outdir)

    summary_payload = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["usable_strict_count"] == 1
    assert (outdir / "original_song_audit.jsonl").exists()
    assert (outdir / "usable_multisection_strict_ori_uids.txt").read_text(encoding="utf-8").strip() == "safe_song"
    assert set((outdir / "usable_multisection_compact_gap_ori_uids.txt").read_text(encoding="utf-8").split()) == {
        "safe_song",
        "large_gap_song",
    }
