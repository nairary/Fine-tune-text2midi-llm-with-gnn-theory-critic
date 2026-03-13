import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional


# =========================================================
# I/O
# =========================================================

def load_top_level_dict(path: str) -> Dict[str, Any]:
    """
    Загружает JSON, где верхний уровень — dict:
    {
      "A": {...},
      "B": {...}
    }

    Поддерживает и случай, когда файл является фрагментом без внешних {}.
    """
    text = Path(path).read_text(encoding="utf-8").strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    wrapped = "{\n" + text.rstrip(", \n") + "\n}"
    obj = json.loads(wrapped)
    if not isinstance(obj, dict):
        raise ValueError("Top-level object is not a dict.")
    return obj


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] bad json at line {line_idx} in {path}")
    return rows


# =========================================================
# Raw main JSON -> processed song_record
# =========================================================

def simplify_key_obj(x: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "beat": x.get("beat"),
        "tonic": x.get("tonic"),
        "scale": x.get("scale"),
    }


def simplify_tempo_obj(x: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "beat": x.get("beat"),
        "bpm": x.get("bpm"),
    }


def simplify_meter_obj(x: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "beat": x.get("beat"),
        "num_beats": x.get("numBeats"),
        "beat_unit": x.get("beatUnit"),
    }


def simplify_note_obj(x: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "beat": x.get("beat"),
        "duration": x.get("duration"),
        "sd": x.get("sd"),
        "octave": x.get("octave"),
        "is_rest": x.get("isRest", x.get("is_rest")),
    }


def simplify_chord_obj(x: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "beat": x.get("beat"),
        "duration": x.get("duration"),
        "root": x.get("root"),
        "type": x.get("type"),
        "inversion": x.get("inversion"),
        "applied": x.get("applied"),
        "adds": x.get("adds", []),
        "omits": x.get("omits", []),
        "alterations": x.get("alterations", []),
        "suspensions": x.get("suspensions", []),
        "borrowed": x.get("borrowed"),
        "alternate": x.get("alternate"),
        "is_rest": x.get("isRest", x.get("is_rest")),
    }


def parse_raw_record(track_id: str, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Преобразует одну raw HookTheory запись в твой упрощённый song_record.
    Если json отсутствует или битый — возвращает None.
    """
    j = record.get("json")
    if not isinstance(j, dict):
        return None

    keys = j.get("keys") or []
    tempos = j.get("tempos") or []
    meters = j.get("meters") or []

    song = {
        "song_id": record.get("hash", track_id),
        "meta": {
            "ori_uid": None,
            "split": record.get("split"),
            "end_beat": j.get("endBeat"),
            "keys": [simplify_key_obj(x) for x in keys if isinstance(x, dict)],
            "tempos": [simplify_tempo_obj(x) for x in tempos if isinstance(x, dict)],
            "meters": [simplify_meter_obj(x) for x in meters if isinstance(x, dict)],
        },
        "melody": [simplify_note_obj(x) for x in (j.get("notes") or []) if isinstance(x, dict)],
        "chords": [simplify_chord_obj(x) for x in (j.get("chords") or []) if isinstance(x, dict)],
        "sections": [],
    }
    return song


# =========================================================
# HookTheoryStructure parsing
# =========================================================

def canonical_split(split_value: Optional[str]) -> Optional[str]:
    if split_value is None:
        return None
    s = str(split_value).strip().lower()
    if s == "valid":
        return "val"
    return s


def extract_song_id_from_structure_obj(obj: Dict[str, Any]) -> Optional[str]:
    """
    В HookTheoryStructure.*.jsonl song_id берём из audio_path:
    data/HookTheory/hooktheory_clips/BbWgMGXzolX.mp3 -> BbWgMGXzolX
    """
    audio_path = obj.get("audio_path")
    if not audio_path:
        return None
    return Path(audio_path).stem


def simplify_section_obj(section_obj):
    """
    Одна строка HookTheoryStructure = один аудио-клип секции.
    """
    label = section_obj.get("label")
    if isinstance(label, str):
        labels = [label]
    elif isinstance(label, list):
        labels = label
    elif label is None:
        labels = []
    else:
        labels = [str(label)]

    return {
        "labels": labels,
        "duration_seconds": section_obj.get("duration"),
        "segment_start_seconds": section_obj.get("segment_start"),
        "segment_end_seconds": section_obj.get("segment_end"),
        "ori_uid": section_obj.get("ori_uid"),   # временно сохраняем, чтобы потом поднять в meta
    }


def extract_sections_from_structure_obj(obj):
    """
    Для HookTheoryStructure.*.jsonl:
    одна строка = один section-клип.
    """
    if not isinstance(obj, dict):
        return []
    if "audio_path" not in obj:
        return []
    return [simplify_section_obj(obj)]


def load_structure_map(path: str) -> Dict[str, List[Dict[str, Any]]]:
    rows = load_jsonl(path)
    by_song = defaultdict(list)

    for obj in rows:
        if not isinstance(obj, dict):
            continue

        song_id = extract_song_id_from_structure_obj(obj)
        if song_id is None:
            continue

        sections = extract_sections_from_structure_obj(obj)
        if sections:
            by_song[song_id].extend(sections)

    for song_id, sections in by_song.items():
        sections.sort(
            key=lambda x: (
                float(x["segment_start_seconds"])
                if x.get("segment_start_seconds") is not None
                else 1e18
            )
        )

    return dict(by_song)

# =========================================================
# Merge
# =========================================================

def build_structure_maps(
    structure_train_path: str,
    structure_val_path: str,
    structure_test_path: str,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    return {
        "train": load_structure_map(structure_train_path),
        "val": load_structure_map(structure_val_path),
        "test": load_structure_map(structure_test_path),
    }


def attach_sections(songs, structure_maps):
    """
    Подтягивает секции по split, сохраняет ori_uid в meta
    и возвращает подробную статистику по unmatched song_id.
    """
    stats = {
        "songs_with_attached_sections": 0,
        "songs_without_attached_sections": 0,
        "songs_with_unknown_split": 0,
        "unknown_split_song_ids": [],
        "unmatched_song_ids": [],
    }

    for song_id, song in songs.items():
        split = canonical_split(song.get("meta", {}).get("split"))

        if split not in structure_maps:
            stats["songs_with_unknown_split"] += 1
            stats["unknown_split_song_ids"].append({
                "song_id": song_id,
                "raw_split": song.get("meta", {}).get("split"),
                "canonical_split": split,
            })
            song["meta"]["ori_uid"] = None
            song["sections"] = []
            continue

        raw_sections = structure_maps[split].get(song_id, [])

        if raw_sections:
            first = raw_sections[0]
            song["meta"]["ori_uid"] = first.get("ori_uid")

            song["sections"] = [
                {
                    "labels": sec.get("labels", []),
                    "duration_seconds": sec.get("duration_seconds"),
                    "segment_start_seconds": sec.get("segment_start_seconds"),
                    "segment_end_seconds": sec.get("segment_end_seconds"),
                }
                for sec in raw_sections
            ]

            stats["songs_with_attached_sections"] += 1
        else:
            song["meta"]["ori_uid"] = None
            song["sections"] = []
            stats["songs_without_attached_sections"] += 1
            stats["unmatched_song_ids"].append({
                "song_id": song_id,
                "split": split,
            })

    return stats


# =========================================================
# Stats
# =========================================================

def compute_stats(songs, attach_stats):
    stats = {
        "n_tracks": 0,
        "n_total_notes": 0,
        "n_total_chords": 0,
        "n_tracks_with_sections": 0,
        "n_tracks_with_multiple_keys": 0,
        "n_tracks_with_multiple_tempos": 0,
        "n_tracks_with_multiple_meters": 0,
        "splits": {
            "train": {"n_tracks": 0, "n_with_sections": 0},
            "val": {"n_tracks": 0, "n_with_sections": 0},
            "test": {"n_tracks": 0, "n_with_sections": 0},
            "unknown": {"n_tracks": 0, "n_with_sections": 0},
        },
    }

    for song in songs.values():
        stats["n_tracks"] += 1

        split = canonical_split(song.get("meta", {}).get("split"))
        split_key = split if split in {"train", "val", "test"} else "unknown"
        stats["splits"][split_key]["n_tracks"] += 1

        melody = song.get("melody", [])
        chords = song.get("chords", [])
        sections = song.get("sections", [])

        stats["n_total_notes"] += len(melody)
        stats["n_total_chords"] += len(chords)

        if sections:
            stats["n_tracks_with_sections"] += 1
            stats["splits"][split_key]["n_with_sections"] += 1

        keys = song.get("meta", {}).get("keys") or []
        tempos = song.get("meta", {}).get("tempos") or []
        meters = song.get("meta", {}).get("meters") or []

        if len(keys) > 1:
            stats["n_tracks_with_multiple_keys"] += 1
        if len(tempos) > 1:
            stats["n_tracks_with_multiple_tempos"] += 1
        if len(meters) > 1:
            stats["n_tracks_with_multiple_meters"] += 1

    # кладём только агрегаты, без дублирования больших списков
    stats["section_attach"] = {
        "songs_with_attached_sections": attach_stats["songs_with_attached_sections"],
        "songs_without_attached_sections": attach_stats["songs_without_attached_sections"],
        "songs_with_unknown_split": attach_stats["songs_with_unknown_split"],
        "n_unmatched_song_ids": len(attach_stats["unmatched_song_ids"]),
        "n_unknown_split_song_ids": len(attach_stats["unknown_split_song_ids"]),
    }

    return stats


# =========================================================
# Main pipeline
# =========================================================

def build_processed_dataset(
    raw_json_path,
    structure_train_path,
    structure_val_path,
    structure_test_path,
    do_compute_stats=False,
):
    raw = load_top_level_dict(raw_json_path)

    songs = {}
    skipped = 0

    for track_id, record in raw.items():
        song = parse_raw_record(track_id, record)
        if song is None:
            skipped += 1
            continue
        songs[song["song_id"]] = song

    structure_maps = build_structure_maps(
        structure_train_path=structure_train_path,
        structure_val_path=structure_val_path,
        structure_test_path=structure_test_path,
    )

    # полезная диагностика покрытия
    for split_name in ["train", "val", "test"]:
        processed_ids = {
            sid for sid, song in songs.items()
            if canonical_split(song.get("meta", {}).get("split")) == split_name
        }
        structure_ids = set(structure_maps[split_name].keys())

        print(f"[DEBUG] split={split_name}")
        print(f"  processed_ids:  {len(processed_ids)}")
        print(f"  structure_ids:  {len(structure_ids)}")
        print(f"  intersection:   {len(processed_ids & structure_ids)}")
        print(f"  processed_only: {len(processed_ids - structure_ids)}")
        print(f"  structure_only: {len(structure_ids - processed_ids)}")

    attach_stats = attach_sections(songs, structure_maps)

    stats = None
    if do_compute_stats:
        stats = compute_stats(songs, attach_stats)
        stats["skipped_records_without_valid_json"] = skipped

    return songs, stats, attach_stats


def save_outputs(out_dir: str, songs: Dict[str, Any], stats: Optional[Dict[str, Any]]):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    processed_path = out / "hooktheory_processed.json"
    processed_path.write_text(
        json.dumps(songs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    if stats is not None:
        stats_path = out / "hooktheory_processed.stats.json"
        stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    print(f"[INFO] saved processed dataset to: {processed_path}")
    if stats is not None:
        print(f"[INFO] saved stats to: {out / 'hooktheory_processed.stats.json'}")

def save_auxiliary_outputs(out_dir, songs, attach_stats):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Только клипы, у которых есть structure match
    structured_only = {
        song_id: song
        for song_id, song in songs.items()
        if song.get("meta", {}).get("ori_uid") is not None and len(song.get("sections", [])) > 0
    }

    # Только unmatched ids
    unmatched_ids = attach_stats.get("unmatched_song_ids", [])
    unknown_split_ids = attach_stats.get("unknown_split_song_ids", [])

    structured_path = out / "hooktheory_processed_structured_only.json"
    unmatched_path = out / "hooktheory_processed_unmatched_ids.json"
    unknown_split_path = out / "hooktheory_processed_unknown_split_ids.json"

    structured_path.write_text(
        json.dumps(structured_only, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    unmatched_path.write_text(
        json.dumps(unmatched_ids, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    unknown_split_path.write_text(
        json.dumps(unknown_split_ids, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"[INFO] saved structured-only dataset to: {structured_path}")
    print(f"[INFO] saved unmatched ids to: {unmatched_path}")
    print(f"[INFO] saved unknown split ids to: {unknown_split_path}")

# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw-json", type=str, required=True, help="Path to raw big HookTheory JSON")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save processed files")

    parser.add_argument("--structure-train", type=str, required=True, help="Path to HookTheoryStructure.train.jsonl")
    parser.add_argument("--structure-val", type=str, required=True, help="Path to HookTheoryStructure.val.jsonl")
    parser.add_argument("--structure-test", type=str, required=True, help="Path to HookTheoryStructure.test.jsonl")

    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="If set, compute and save dataset statistics"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    songs, stats, attach_stats = build_processed_dataset(
        raw_json_path=args.raw_json,
        structure_train_path=args.structure_train,
        structure_val_path=args.structure_val,
        structure_test_path=args.structure_test,
        do_compute_stats=args.compute_stats,
    )

    save_outputs(args.out_dir, songs, stats)
    save_auxiliary_outputs(args.out_dir, songs, attach_stats)

    print("[INFO] done")


if __name__ == "__main__":
    main()