#!/usr/bin/env python3
"""Assemble compact multi-section encoded songs from audited original-song timelines."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_JSONL = Path("outputs/timeline_audit/original_song_audit.jsonl")
DEFAULT_ENCODED_JSON = Path("data/HTCanon/encoded_full/teacher_encoded.json")
DEFAULT_OUTDIR = Path("outputs/assembled_sections")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-jsonl", type=Path, default=DEFAULT_AUDIT_JSONL)
    parser.add_argument("--encoded-json", type=Path, default=DEFAULT_ENCODED_JSON)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--usable-mode",
        choices=["strict", "compact_gap"],
        default="compact_gap",
        help="Which audited usable flag to assemble.",
    )
    parser.add_argument(
        "--section-start-policy",
        choices=["compact", "next_bar", "next_bar_gap"],
        default="next_bar_gap",
        help=(
            "How to place each section after the first. compact appends immediately; "
            "next_bar aligns to the next barline; next_bar_gap aligns to a barline and preserves real timeline gaps as whole empty bars."
        ),
    )
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=10.0,
        help="Skip an original song if any positive timeline gap exceeds this many seconds.",
    )
    parser.add_argument(
        "--multi-clip-segment-policy",
        choices=["skip", "first"],
        default="skip",
        help="How to handle a timeline segment with multiple clip_song_ids.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of assembled songs.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip_meta_value(song_obj: dict[str, Any], key: str, default: float) -> float:
    meta = song_obj.get("meta", {}) if isinstance(song_obj.get("meta"), dict) else {}
    return _safe_float(meta.get(key), default)


def _next_bar_start(beat: float, num_beats: float) -> float:
    num_beats = max(1.0, float(num_beats))
    bar_index = math.ceil(((float(beat) - 1.0) / num_beats) - 1e-9)
    return 1.0 + max(0, bar_index) * num_beats


def _beats_to_seconds(beats: float, bpm: float) -> float:
    if bpm <= 0:
        return 0.0
    return max(0.0, float(beats)) * 60.0 / float(bpm)


def _section_start_after_previous(
    *,
    previous_target_end_beat: float,
    previous_segment: dict[str, Any],
    current_segment: dict[str, Any],
    previous_clip: dict[str, Any],
    section_start_policy: str,
) -> tuple[float, dict[str, Any]]:
    raw_gap_seconds = _safe_float(current_segment.get("segment_start_seconds"), 0.0) - _safe_float(
        previous_segment.get("segment_end_seconds"), 0.0
    )
    positive_gap_seconds = max(0.0, raw_gap_seconds)
    num_beats = _clip_meta_value(previous_clip, "main_num_beats", 4.0)
    bpm = _clip_meta_value(previous_clip, "main_bpm", 120.0)

    if section_start_policy == "compact":
        target_start_beat = previous_target_end_beat
        gap_reason = "compact"
        extra_full_gap_bars = 0
        min_barline_gap_seconds = 0.0
    else:
        barline_start_beat = _next_bar_start(previous_target_end_beat, num_beats)
        min_barline_gap_beats = max(0.0, barline_start_beat - previous_target_end_beat)
        min_barline_gap_seconds = _beats_to_seconds(min_barline_gap_beats, bpm)
        extra_full_gap_bars = 0

        if section_start_policy == "next_bar_gap" and positive_gap_seconds > min_barline_gap_seconds:
            bar_seconds = _beats_to_seconds(num_beats, bpm)
            remaining_gap_seconds = max(0.0, positive_gap_seconds - min_barline_gap_seconds)
            if bar_seconds > 0:
                # Quantize preserved silence to whole bars; floor(x + 0.5) avoids Python's bankers rounding.
                extra_full_gap_bars = max(0, int(math.floor((remaining_gap_seconds / bar_seconds) + 0.5)))
        target_start_beat = barline_start_beat + extra_full_gap_bars * num_beats
        gap_reason = "barline" if extra_full_gap_bars == 0 else "barline_plus_gap_bars"

    inserted_gap_beats = max(0.0, target_start_beat - previous_target_end_beat)
    return target_start_beat, {
        "section_start_policy": section_start_policy,
        "timeline_gap_seconds_from_previous": round(raw_gap_seconds, 6),
        "positive_gap_seconds_from_previous": round(positive_gap_seconds, 6),
        "min_barline_gap_seconds": round(min_barline_gap_seconds, 6),
        "inserted_gap_beats_before": round(inserted_gap_beats, 6),
        "inserted_gap_seconds_before": round(_beats_to_seconds(inserted_gap_beats, bpm), 6),
        "inserted_gap_bars_before": round(inserted_gap_beats / max(1.0, num_beats), 6),
        "extra_full_gap_bars_before": extra_full_gap_bars,
        "gap_placement_reason": gap_reason,
    }


def _event_start(event: dict[str, Any]) -> float | None:
    if event.get("beat") is None:
        return None
    return _safe_float(event.get("beat"), 1.0)


def _event_end(event: dict[str, Any]) -> float | None:
    start = _event_start(event)
    if start is None:
        return None
    return start + max(0.0, _safe_float(event.get("duration"), 0.0))


def infer_clip_bounds(song_obj: dict[str, Any]) -> tuple[float, float]:
    """Infer 1-based clip beat bounds as [start, end)."""
    meta = song_obj.get("meta", {}) if isinstance(song_obj.get("meta"), dict) else {}
    starts: list[float] = []
    ends: list[float] = []
    for collection_name in ("melody", "chords"):
        for event in song_obj.get(collection_name, []) or []:
            if not isinstance(event, dict):
                continue
            start = _event_start(event)
            end = _event_end(event)
            if start is not None:
                starts.append(start)
            if end is not None:
                ends.append(end)

    source_start = min(starts) if starts else 1.0
    source_start = min(1.0, source_start)
    source_end = max(ends) if ends else source_start
    if meta.get("end_beat") is not None:
        source_end = max(source_end, _safe_float(meta.get("end_beat"), source_end))
    return source_start, max(source_start, source_end)


def _shift_event(event: dict[str, Any], *, offset_beats: float, source_clip_song_id: str, section_index: int, source_event_index: int) -> dict[str, Any]:
    shifted = copy.deepcopy(event)
    if shifted.get("beat") is not None:
        shifted["beat"] = _safe_float(shifted["beat"], 1.0) + offset_beats
    shifted["source_clip_song_id"] = source_clip_song_id
    shifted["source_section_index"] = section_index
    shifted["source_event_index"] = source_event_index
    return shifted


def _shift_regions(regions: Any, *, offset_beats: float, source_clip_song_id: str, section_index: int) -> list[dict[str, Any]]:
    shifted_regions: list[dict[str, Any]] = []
    if not isinstance(regions, list):
        return shifted_regions
    for region_index, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        shifted = copy.deepcopy(region)
        if shifted.get("beat") is not None:
            shifted["beat"] = _safe_float(shifted["beat"], 1.0) + offset_beats
        shifted["source_clip_song_id"] = source_clip_song_id
        shifted["source_section_index"] = section_index
        shifted["source_region_index"] = region_index
        shifted_regions.append(shifted)
    return shifted_regions


def _section_label(labels: list[str]) -> str:
    return "+".join(labels) if labels else "unknown"


def _selected_clip_ids(segment: dict[str, Any], multi_clip_segment_policy: str) -> tuple[list[str], str | None]:
    clip_ids = list(segment.get("clip_song_ids") or [])
    if len(clip_ids) == 1:
        return clip_ids, None
    if not clip_ids:
        return [], "segment has no clip_song_ids"
    if multi_clip_segment_policy == "first":
        return [clip_ids[0]], None
    return [], f"segment has {len(clip_ids)} clip_song_ids"


def assemble_timeline_row(
    row: dict[str, Any],
    encoded_payload: dict[str, Any],
    *,
    multi_clip_segment_policy: str = "skip",
    section_start_policy: str = "next_bar_gap",
    max_gap_sec: float = 10.0,
) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Assemble one audited original-song row.

    Returns (assembled_song_id, assembled_song, skip_row).
    """
    ori_uid = str(row.get("ori_uid"))
    timeline = list(row.get("timeline") or [])
    if len(timeline) < 2:
        return None, None, {"ori_uid": ori_uid, "reason": "not_multisection", "section_count": len(timeline)}

    max_positive_gap = max(
        (
            _safe_float(current.get("segment_start_seconds"), 0.0) - _safe_float(previous.get("segment_end_seconds"), 0.0)
            for previous, current in zip(timeline, timeline[1:])
        ),
        default=0.0,
    )
    if max_positive_gap > max_gap_sec:
        return None, None, {
            "ori_uid": ori_uid,
            "reason": "gap exceeds max_gap_sec",
            "max_gap_seconds": round(max_positive_gap, 6),
            "max_gap_sec": max_gap_sec,
        }

    sections: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for segment in timeline:
        selected_clip_ids, skip_reason = _selected_clip_ids(segment, multi_clip_segment_policy)
        if skip_reason is not None:
            return None, None, {
                "ori_uid": ori_uid,
                "reason": skip_reason,
                "section_index": segment.get("index"),
                "clip_song_ids": segment.get("clip_song_ids") or [],
            }
        for clip_id in selected_clip_ids:
            clip = encoded_payload.get(clip_id)
            if not isinstance(clip, dict):
                return None, None, {
                    "ori_uid": ori_uid,
                    "reason": "clip missing from encoded payload",
                    "clip_song_id": clip_id,
                    "section_index": segment.get("index"),
                }
            sections.append((segment, clip_id, clip))

    if len(sections) < 2:
        return None, None, {"ori_uid": ori_uid, "reason": "not_enough_selected_sections", "section_count": len(sections)}

    assembled_song_id = f"assembled_{ori_uid}"
    first_clip = sections[0][2]
    first_meta = copy.deepcopy(first_clip.get("meta", {}) if isinstance(first_clip.get("meta"), dict) else {})
    source_splits = sorted({split for segment, _, _ in sections for split in (segment.get("splits") or [])})
    source_clip_ids = [clip_id for _, clip_id, _ in sections]

    assembled_melody: list[dict[str, Any]] = []
    assembled_chords: list[dict[str, Any]] = []
    key_regions: list[dict[str, Any]] = []
    tempo_regions: list[dict[str, Any]] = []
    meter_regions: list[dict[str, Any]] = []
    section_spans: list[dict[str, Any]] = []
    previous_target_end_beat: float | None = None
    previous_segment: dict[str, Any] | None = None
    previous_clip: dict[str, Any] | None = None

    for section_index, (segment, clip_id, clip) in enumerate(sections):
        source_start_beat, source_end_beat = infer_clip_bounds(clip)
        duration_beats = max(0.0, source_end_beat - source_start_beat)
        if section_index == 0:
            target_start_beat = 1.0
            gap_metadata = {
                "section_start_policy": section_start_policy,
                "timeline_gap_seconds_from_previous": None,
                "positive_gap_seconds_from_previous": None,
                "min_barline_gap_seconds": None,
                "inserted_gap_beats_before": 0.0,
                "inserted_gap_seconds_before": 0.0,
                "inserted_gap_bars_before": 0.0,
                "extra_full_gap_bars_before": 0,
                "gap_placement_reason": "first_section",
            }
        else:
            assert previous_target_end_beat is not None
            assert previous_segment is not None
            assert previous_clip is not None
            target_start_beat, gap_metadata = _section_start_after_previous(
                previous_target_end_beat=previous_target_end_beat,
                previous_segment=previous_segment,
                current_segment=segment,
                previous_clip=previous_clip,
                section_start_policy=section_start_policy,
            )
        target_end_beat = target_start_beat + duration_beats
        offset_beats = target_start_beat - source_start_beat
        labels = list(segment.get("labels") or [])

        for event_index, event in enumerate(clip.get("melody", []) or []):
            if isinstance(event, dict):
                assembled_melody.append(
                    _shift_event(
                        event,
                        offset_beats=offset_beats,
                        source_clip_song_id=clip_id,
                        section_index=section_index,
                        source_event_index=event_index,
                    )
                )
        for event_index, event in enumerate(clip.get("chords", []) or []):
            if isinstance(event, dict):
                assembled_chords.append(
                    _shift_event(
                        event,
                        offset_beats=offset_beats,
                        source_clip_song_id=clip_id,
                        section_index=section_index,
                        source_event_index=event_index,
                    )
                )

        clip_meta = clip.get("meta", {}) if isinstance(clip.get("meta"), dict) else {}
        key_regions.extend(_shift_regions(clip_meta.get("key_regions"), offset_beats=offset_beats, source_clip_song_id=clip_id, section_index=section_index))
        tempo_regions.extend(_shift_regions(clip_meta.get("tempo_regions"), offset_beats=offset_beats, source_clip_song_id=clip_id, section_index=section_index))
        meter_regions.extend(_shift_regions(clip_meta.get("meter_regions"), offset_beats=offset_beats, source_clip_song_id=clip_id, section_index=section_index))

        section_spans.append(
            {
                "section_index": section_index,
                "label": _section_label(labels),
                "labels": labels,
                "source_clip_song_id": clip_id,
                "source_clip_song_ids": list(segment.get("clip_song_ids") or []),
                "source_timeline_index": segment.get("index"),
                "source_start_seconds": segment.get("segment_start_seconds"),
                "source_end_seconds": segment.get("segment_end_seconds"),
                "source_duration_seconds": segment.get("duration_seconds"),
                "source_start_beat": source_start_beat,
                "source_end_beat": source_end_beat,
                "target_start_beat": target_start_beat,
                "target_end_beat": target_end_beat,
                "duration_beats": duration_beats,
                "splits": list(segment.get("splits") or []),
                **gap_metadata,
            }
        )
        previous_target_end_beat = target_end_beat
        previous_segment = segment
        previous_clip = clip

    assembled_melody.sort(key=lambda event: (_safe_float(event.get("beat"), 1.0), _safe_float(event.get("duration"), 0.0)))
    assembled_chords.sort(key=lambda event: (_safe_float(event.get("beat"), 1.0), _safe_float(event.get("duration"), 0.0)))

    meta = first_meta
    meta["song_id"] = assembled_song_id
    meta["ori_uid"] = ori_uid
    meta["split"] = source_splits[0] if len(source_splits) == 1 else "mixed"
    meta["source_splits"] = source_splits
    meta["assembled_from_timeline"] = True
    meta["assembly_mode"] = section_start_policy
    meta["source_clip_song_ids"] = source_clip_ids
    meta["section_spans"] = section_spans
    meta["end_beat"] = previous_target_end_beat or 1.0
    if key_regions:
        meta["key_regions"] = sorted(key_regions, key=lambda region: _safe_float(region.get("beat"), 1.0))
    if tempo_regions:
        meta["tempo_regions"] = sorted(tempo_regions, key=lambda region: _safe_float(region.get("beat"), 1.0))
    if meter_regions:
        meta["meter_regions"] = sorted(meter_regions, key=lambda region: _safe_float(region.get("beat"), 1.0))

    return assembled_song_id, {"song_id": assembled_song_id, "meta": meta, "melody": assembled_melody, "chords": assembled_chords}, None


def assemble_from_audit_rows(
    audit_rows: list[dict[str, Any]],
    encoded_payload: dict[str, Any],
    *,
    usable_mode: str = "strict",
    multi_clip_segment_policy: str = "skip",
    section_start_policy: str = "next_bar_gap",
    max_gap_sec: float = 10.0,
    limit: int | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    usable_key = "usable_strict" if usable_mode == "strict" else "usable_compact_gap"
    assembled: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for row in audit_rows:
        if not bool(row.get(usable_key)):
            continue
        if limit is not None and len(assembled) >= limit:
            break

        assembled_song_id, assembled_song, skipped = assemble_timeline_row(
            row,
            encoded_payload,
            multi_clip_segment_policy=multi_clip_segment_policy,
            section_start_policy=section_start_policy,
            max_gap_sec=max_gap_sec,
        )
        if skipped is not None:
            skipped_rows.append(skipped)
            continue
        if assembled_song_id is None or assembled_song is None:
            continue

        section_spans = assembled_song["meta"]["section_spans"]
        labels = [span["label"] for span in section_spans]
        label_counts.update(labels)
        transition_counts.update(f"{left}->{right}" for left, right in zip(labels, labels[1:]))

        assembled[assembled_song_id] = assembled_song
        manifest_rows.append(
            {
                "song_id": assembled_song_id,
                "ori_uid": row["ori_uid"],
                "split": assembled_song["meta"].get("split"),
                "section_count": len(section_spans),
                "source_clip_song_ids": assembled_song["meta"]["source_clip_song_ids"],
                "labels": labels,
                "end_beat": assembled_song["meta"]["end_beat"],
            }
        )

    summary = {
        "usable_mode": usable_mode,
        "multi_clip_segment_policy": multi_clip_segment_policy,
        "section_start_policy": section_start_policy,
        "max_gap_sec": max_gap_sec,
        "assembled_song_count": len(assembled),
        "skipped_count": len(skipped_rows),
        "section_count": sum(row["section_count"] for row in manifest_rows),
        "source_clip_ref_count": sum(len(row["source_clip_song_ids"]) for row in manifest_rows),
        "inserted_gap_beats": round(
            sum(
                float(span.get("inserted_gap_beats_before") or 0.0)
                for song in assembled.values()
                for span in song["meta"].get("section_spans", [])
            ),
            6,
        ),
        "extra_full_gap_bars": sum(
            int(span.get("extra_full_gap_bars_before") or 0)
            for song in assembled.values()
            for span in song["meta"].get("section_spans", [])
        ),
        "label_counts": dict(label_counts.most_common()),
        "transition_counts": dict(transition_counts.most_common()),
        "skip_reason_counts": dict(Counter(str(row.get("reason")) for row in skipped_rows).most_common()),
    }
    return assembled, manifest_rows, skipped_rows, summary


def write_assembly_outputs(
    assembled: dict[str, dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    outdir: Path,
    *,
    usable_mode: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / f"teacher_encoded_assembled_{usable_mode}.json", assembled)
    write_jsonl(outdir / f"assembled_manifest_{usable_mode}.jsonl", manifest_rows)
    write_jsonl(outdir / f"skipped_assembly_{usable_mode}.jsonl", skipped_rows)
    write_json(outdir / f"summary_{usable_mode}.json", summary)


def main() -> None:
    args = parse_args()
    audit_rows = load_jsonl(args.audit_jsonl)
    encoded_payload = load_json(args.encoded_json)
    if not isinstance(encoded_payload, dict):
        raise ValueError(f"Encoded JSON must be an object keyed by song_id: {args.encoded_json}")

    assembled, manifest_rows, skipped_rows, summary = assemble_from_audit_rows(
        audit_rows,
        encoded_payload,
        usable_mode=args.usable_mode,
        multi_clip_segment_policy=args.multi_clip_segment_policy,
        section_start_policy=args.section_start_policy,
        max_gap_sec=args.max_gap_sec,
        limit=args.limit,
    )
    write_assembly_outputs(assembled, manifest_rows, skipped_rows, summary, args.outdir, usable_mode=args.usable_mode)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
