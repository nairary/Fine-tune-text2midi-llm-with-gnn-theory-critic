#!/usr/bin/env python3
"""Audit original-song section timelines before assembling multi-section songs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TIMELINE_JSON = Path("data/HTCanon/HK_processed/original_songs_timeline.json")
DEFAULT_ENCODED_JSON = Path("data/HTCanon/encoded_full/teacher_encoded.json")
DEFAULT_OUTDIR = Path("outputs/timeline_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-json", type=Path, default=DEFAULT_TIMELINE_JSON)
    parser.add_argument("--encoded-json", type=Path, default=DEFAULT_ENCODED_JSON)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--near-touch-sec",
        type=float,
        default=0.01,
        help="Absolute gap/overlap <= this is treated as touching.",
    )
    parser.add_argument(
        "--small-gap-sec",
        type=float,
        default=2.0,
        help="Positive gap <= this is treated as small_gap; larger gaps are large_gap.",
    )
    parser.add_argument(
        "--small-overlap-sec",
        type=float,
        default=0.25,
        help="Overlap <= this is treated as small_overlap; larger overlaps are large_overlap.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _segment_row(index: int, segment: dict[str, Any]) -> dict[str, Any]:
    start = float(segment["segment_start_seconds"])
    end = float(segment["segment_end_seconds"])
    return {
        "index": index,
        "segment_start_seconds": start,
        "segment_end_seconds": end,
        "duration_seconds": float(segment.get("duration_seconds", end - start)),
        "labels": _as_str_list(segment.get("labels")),
        "clip_song_ids": _as_str_list(segment.get("clip_song_ids")),
        "splits": _as_str_list(segment.get("splits")),
    }


def _label_key(labels: list[str]) -> str:
    return "+".join(labels) if labels else "unknown"


def classify_relation(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    near_touch_sec: float,
    small_gap_sec: float,
    small_overlap_sec: float,
) -> dict[str, Any]:
    gap = float(current["segment_start_seconds"]) - float(previous["segment_end_seconds"])
    overlap = max(0.0, -gap)
    if abs(gap) <= near_touch_sec:
        relation = "touching"
    elif gap > 0:
        relation = "small_gap" if gap <= small_gap_sec else "large_gap"
    else:
        relation = "small_overlap" if overlap <= small_overlap_sec else "large_overlap"

    return {
        "from_index": previous["index"],
        "to_index": current["index"],
        "from_clip_song_ids": previous["clip_song_ids"],
        "to_clip_song_ids": current["clip_song_ids"],
        "from_labels": previous["labels"],
        "to_labels": current["labels"],
        "transition": f"{_label_key(previous['labels'])}->{_label_key(current['labels'])}",
        "gap_seconds": round(gap, 6),
        "overlap_seconds": round(overlap, 6),
        "relation": relation,
    }


def audit_original_song(
    ori_uid: str,
    original_song: dict[str, Any],
    encoded_song_ids: set[str],
    *,
    near_touch_sec: float,
    small_gap_sec: float,
    small_overlap_sec: float,
) -> dict[str, Any]:
    raw_timeline = original_song.get("timeline") or []
    invalid_segments: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []

    for index, segment in enumerate(raw_timeline):
        try:
            row = _segment_row(index, segment)
            if row["segment_end_seconds"] < row["segment_start_seconds"]:
                raise ValueError("segment_end_seconds is before segment_start_seconds")
            timeline.append(row)
        except Exception as exc:  # noqa: BLE001
            invalid_segments.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})

    timeline.sort(key=lambda row: (row["segment_start_seconds"], row["segment_end_seconds"], row["index"]))

    labels = sorted({label for segment in timeline for label in segment["labels"]})
    clip_song_ids = [clip_id for segment in timeline for clip_id in segment["clip_song_ids"]]
    splits = sorted({split for segment in timeline for split in segment["splits"]})
    missing_clip_ids = sorted({clip_id for clip_id in clip_song_ids if clip_id not in encoded_song_ids})

    neighbor_relations = [
        classify_relation(
            previous,
            current,
            near_touch_sec=near_touch_sec,
            small_gap_sec=small_gap_sec,
            small_overlap_sec=small_overlap_sec,
        )
        for previous, current in zip(timeline, timeline[1:])
    ]

    relation_counts = Counter(relation["relation"] for relation in neighbor_relations)
    max_gap_seconds = max((max(0.0, relation["gap_seconds"]) for relation in neighbor_relations), default=0.0)
    max_overlap_seconds = max((relation["overlap_seconds"] for relation in neighbor_relations), default=0.0)

    buckets: list[str] = []
    if len(timeline) <= 1:
        buckets.append("single_section")
    else:
        if relation_counts["touching"]:
            buckets.append("touching")
        if relation_counts["small_gap"]:
            buckets.append("small_gap")
        if relation_counts["large_gap"]:
            buckets.append("large_gap")
        if relation_counts["small_overlap"]:
            buckets.append("small_overlap")
        if relation_counts["large_overlap"]:
            buckets.append("large_overlap")
        if not any(bucket in buckets for bucket in ("small_gap", "large_gap", "small_overlap", "large_overlap")):
            buckets.append("safe_multisection")

    if len(splits) > 1:
        buckets.append("mixed_split")
    if missing_clip_ids:
        buckets.append("missing_clip")
    if invalid_segments:
        buckets.append("invalid_segment")

    usable_base = len(timeline) >= 2 and not missing_clip_ids and not invalid_segments and len(splits) <= 1
    usable_strict = usable_base and relation_counts["large_gap"] == 0 and relation_counts["large_overlap"] == 0
    usable_compact_gap = usable_base and relation_counts["large_overlap"] == 0

    return {
        "ori_uid": ori_uid,
        "section_count": len(timeline),
        "clip_ref_count": len(clip_song_ids),
        "unique_clip_ref_count": len(set(clip_song_ids)),
        "splits": splits,
        "labels": labels,
        "buckets": buckets,
        "missing_clip_ids": missing_clip_ids,
        "invalid_segments": invalid_segments,
        "relation_counts": dict(sorted(relation_counts.items())),
        "max_gap_seconds": round(max_gap_seconds, 6),
        "max_overlap_seconds": round(max_overlap_seconds, 6),
        "usable_strict": usable_strict,
        "usable_compact_gap": usable_compact_gap,
        "timeline": timeline,
        "neighbor_relations": neighbor_relations,
    }


def build_summary(rows: list[dict[str, Any]], *, thresholds: dict[str, float]) -> dict[str, Any]:
    bucket_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    section_count_distribution: Counter[str] = Counter()

    for row in rows:
        bucket_counts.update(row["buckets"])
        label_counts.update(row["labels"])
        transition_counts.update(relation["transition"] for relation in row["neighbor_relations"])
        relation_counts.update(relation["relation"] for relation in row["neighbor_relations"])
        section_count_distribution[str(row["section_count"])] += 1

    return {
        "thresholds": thresholds,
        "original_song_count": len(rows),
        "timeline_segment_count": sum(row["section_count"] for row in rows),
        "clip_ref_count": sum(row["clip_ref_count"] for row in rows),
        "unique_clip_ref_count": len({clip_id for row in rows for segment in row["timeline"] for clip_id in segment["clip_song_ids"]}),
        "missing_clip_ref_count": sum(len(row["missing_clip_ids"]) for row in rows),
        "invalid_segment_count": sum(len(row["invalid_segments"]) for row in rows),
        "single_section_count": sum(1 for row in rows if row["section_count"] <= 1),
        "multisection_count": sum(1 for row in rows if row["section_count"] >= 2),
        "usable_strict_count": sum(1 for row in rows if row["usable_strict"]),
        "usable_compact_gap_count": sum(1 for row in rows if row["usable_compact_gap"]),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "label_counts": dict(label_counts.most_common()),
        "transition_counts": dict(transition_counts.most_common()),
        "section_count_distribution": dict(sorted(section_count_distribution.items(), key=lambda item: int(item[0]))),
    }


def audit_timeline(
    timeline_payload: dict[str, Any],
    encoded_payload: dict[str, Any],
    *,
    near_touch_sec: float = 0.01,
    small_gap_sec: float = 2.0,
    small_overlap_sec: float = 0.25,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded_song_ids = set(str(song_id) for song_id in encoded_payload.keys())
    rows = [
        audit_original_song(
            str(ori_uid),
            original_song,
            encoded_song_ids,
            near_touch_sec=near_touch_sec,
            small_gap_sec=small_gap_sec,
            small_overlap_sec=small_overlap_sec,
        )
        for ori_uid, original_song in sorted(timeline_payload.items())
        if isinstance(original_song, dict)
    ]
    thresholds = {
        "near_touch_sec": near_touch_sec,
        "small_gap_sec": small_gap_sec,
        "small_overlap_sec": small_overlap_sec,
    }
    return rows, build_summary(rows, thresholds=thresholds)


def write_audit_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    strict_uids = [row["ori_uid"] for row in rows if row["usable_strict"]]
    compact_gap_uids = [row["ori_uid"] for row in rows if row["usable_compact_gap"]]

    write_json(outdir / "summary.json", summary)
    write_jsonl(outdir / "original_song_audit.jsonl", rows)
    write_lines(outdir / "usable_multisection_strict_ori_uids.txt", strict_uids)
    write_lines(outdir / "usable_multisection_compact_gap_ori_uids.txt", compact_gap_uids)


def main() -> None:
    args = parse_args()
    timeline_payload = load_json(args.timeline_json)
    encoded_payload = load_json(args.encoded_json)
    if not isinstance(timeline_payload, dict):
        raise ValueError(f"Timeline JSON must be an object keyed by ori_uid: {args.timeline_json}")
    if not isinstance(encoded_payload, dict):
        raise ValueError(f"Encoded JSON must be an object keyed by song_id: {args.encoded_json}")

    rows, summary = audit_timeline(
        timeline_payload,
        encoded_payload,
        near_touch_sec=args.near_touch_sec,
        small_gap_sec=args.small_gap_sec,
        small_overlap_sec=args.small_overlap_sec,
    )
    write_audit_outputs(rows, summary, args.outdir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
