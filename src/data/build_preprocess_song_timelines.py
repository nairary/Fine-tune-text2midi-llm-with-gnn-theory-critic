import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_processed_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_labels(labels):
    if labels is None:
        return []
    if isinstance(labels, str):
        labels = [labels]

    out = []
    for x in labels:
        if x is None:
            continue
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def round_time(x, ndigits=6):
    x = safe_float(x)
    if x is None:
        return None
    return round(x, ndigits)


def merge_duplicate_segments(segments, ndigits=6):
    """
    Мёрджит сегменты с одинаковыми start/end как шумные дубликаты.
    """
    merged = {}

    for seg in segments:
        start = round_time(seg.get("segment_start_seconds"), ndigits)
        end = round_time(seg.get("segment_end_seconds"), ndigits)
        key = (start, end)

        if key not in merged:
            merged[key] = {
                "segment_start_seconds": seg.get("segment_start_seconds"),
                "segment_end_seconds": seg.get("segment_end_seconds"),
                "duration_seconds": seg.get("duration_seconds"),
                "labels": [],
                "clip_song_ids": [],
                "splits": [],
            }

        merged[key]["labels"].extend(normalize_labels(seg.get("labels")))
        merged[key]["clip_song_ids"].extend(seg.get("clip_song_ids", []))
        merged[key]["splits"].extend(seg.get("splits", []))

        if merged[key]["duration_seconds"] is None and seg.get("duration_seconds") is not None:
            merged[key]["duration_seconds"] = seg.get("duration_seconds")

    out = []
    for _, seg in merged.items():
        seg["labels"] = sorted(set(seg["labels"]))
        seg["clip_song_ids"] = sorted(set(seg["clip_song_ids"]))
        seg["splits"] = sorted(set([x for x in seg["splits"] if x is not None]))
        out.append(seg)

    out.sort(
        key=lambda s: (
            safe_float(s.get("segment_start_seconds")) if s.get("segment_start_seconds") is not None else 1e18,
            safe_float(s.get("segment_end_seconds")) if s.get("segment_end_seconds") is not None else 1e18,
        )
    )
    return out


def build_original_song_timelines(processed):
    """
    processed: dict[song_id] -> processed clip object
    Требует наличия meta.ori_uid
    """
    by_ori_uid = {}
    skipped_no_ori_uid = 0
    clips_without_sections = 0

    for song_id, clip in processed.items():
        meta = clip.get("meta", {})
        ori_uid = meta.get("ori_uid")
        split = meta.get("split")

        if not ori_uid:
            skipped_no_ori_uid += 1
            continue

        if ori_uid not in by_ori_uid:
            by_ori_uid[ori_uid] = {
                "ori_uid": ori_uid,
                "clip_song_ids": [],
                "splits": [],
                "timeline": [],
                "clips_without_sections": [],
            }

        song_entry = by_ori_uid[ori_uid]
        song_entry["clip_song_ids"].append(song_id)
        if split is not None:
            song_entry["splits"].append(split)

        sections = clip.get("sections", []) or []
        if len(sections) == 0:
            clips_without_sections += 1
            song_entry["clips_without_sections"].append(song_id)
            continue

        for sec in sections:
            song_entry["timeline"].append({
                "segment_start_seconds": sec.get("segment_start_seconds"),
                "segment_end_seconds": sec.get("segment_end_seconds"),
                "duration_seconds": sec.get("duration_seconds"),
                "labels": normalize_labels(sec.get("labels")),
                "clip_song_ids": [song_id],
                "splits": [split] if split is not None else [],
            })

    # постобработка
    for ori_uid, entry in by_ori_uid.items():
        entry["clip_song_ids"] = sorted(set(entry["clip_song_ids"]))
        entry["splits"] = sorted(set([x for x in entry["splits"] if x is not None]))
        entry["clips_without_sections"] = sorted(set(entry["clips_without_sections"]))

        # сначала сортировка
        entry["timeline"].sort(
            key=lambda s: (
                safe_float(s.get("segment_start_seconds")) if s.get("segment_start_seconds") is not None else 1e18,
                safe_float(s.get("segment_end_seconds")) if s.get("segment_end_seconds") is not None else 1e18,
            )
        )

        # затем merge точных дублей
        entry["timeline"] = merge_duplicate_segments(entry["timeline"])

    aggregate_stats = {
        "n_original_songs": len(by_ori_uid),
        "skipped_clips_without_ori_uid": skipped_no_ori_uid,
        "clips_without_sections": clips_without_sections,
    }

    return by_ori_uid, aggregate_stats


def compute_stats(original_songs, eps=1e-6):
    global_stats = {
        "n_original_songs": len(original_songs),
        "n_total_segments": 0,
        "n_songs_with_segments": 0,
        "n_songs_with_gaps": 0,
        "n_songs_with_overlaps": 0,
        "n_multilabel_segments": 0,
    }

    per_song = {}

    for ori_uid, song in original_songs.items():
        timeline = song.get("timeline", [])
        global_stats["n_total_segments"] += len(timeline)

        if timeline:
            global_stats["n_songs_with_segments"] += 1

        gaps = []
        overlaps = []
        multilabel_count = 0

        prev = None
        for seg in timeline:
            if len(seg.get("labels", [])) > 1:
                multilabel_count += 1

            if prev is not None:
                prev_end = safe_float(prev.get("segment_end_seconds"))
                cur_start = safe_float(seg.get("segment_start_seconds"))

                if prev_end is not None and cur_start is not None:
                    if cur_start > prev_end + eps:
                        gaps.append({
                            "prev_end": prev_end,
                            "cur_start": cur_start,
                            "gap_seconds": cur_start - prev_end,
                        })
                    elif cur_start < prev_end - eps:
                        overlaps.append({
                            "prev_end": prev_end,
                            "cur_start": cur_start,
                            "overlap_seconds": prev_end - cur_start,
                        })

            prev = seg

        if gaps:
            global_stats["n_songs_with_gaps"] += 1
        if overlaps:
            global_stats["n_songs_with_overlaps"] += 1
        global_stats["n_multilabel_segments"] += multilabel_count

        per_song[ori_uid] = {
            "n_clip_song_ids": len(song.get("clip_song_ids", [])),
            "n_segments": len(timeline),
            "n_multilabel_segments": multilabel_count,
            "n_gaps": len(gaps),
            "n_overlaps": len(overlaps),
            "gaps": gaps[:20],
            "overlaps": overlaps[:20],
        }

    return {
        "global": global_stats,
        "per_song": per_song,
    }


def save_outputs(out_dir, original_songs, aggregate_stats, detailed_stats=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path_main = out_dir / "original_songs_timeline.json"
    with open(path_main, "w", encoding="utf-8") as f:
        json.dump(original_songs, f, ensure_ascii=False, indent=2)

    path_agg = out_dir / "original_songs_aggregate.stats.json"
    with open(path_agg, "w", encoding="utf-8") as f:
        json.dump(aggregate_stats, f, ensure_ascii=False, indent=2)

    if detailed_stats is not None:
        path_stats = out_dir / "original_songs_timeline.stats.json"
        with open(path_stats, "w", encoding="utf-8") as f:
            json.dump(detailed_stats, f, ensure_ascii=False, indent=2)

    print(f"[INFO] saved: {path_main}")
    print(f"[INFO] saved: {path_agg}")
    if detailed_stats is not None:
        print(f"[INFO] saved: {out_dir / 'original_songs_timeline.stats.json'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-json", type=str, required=True, help="Path to hooktheory_processed.json")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for outputs")
    parser.add_argument("--compute-stats", action="store_true", help="Compute additional timeline statistics")
    return parser.parse_args()


def main():
    args = parse_args()

    processed = load_processed_json(args.processed_json)

    original_songs, aggregate_stats = build_original_song_timelines(processed)

    detailed_stats = None
    if args.compute_stats:
        detailed_stats = compute_stats(original_songs)

    save_outputs(
        out_dir=args.out_dir,
        original_songs=original_songs,
        aggregate_stats=aggregate_stats,
        detailed_stats=detailed_stats,
    )

    print("[INFO] done")


if __name__ == "__main__":
    main()