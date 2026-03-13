import argparse
import json
from pathlib import Path

from preprocess_hooktheory import build_processed_dataset
from build_preprocess_song_timelines import build_original_song_timelines, compute_stats as compute_timeline_stats
from canonicalize_hooktheory import Reporter, normalize_song, compute_stats as compute_canonical_stats
from encode_teacher_features import load_metadata, build_runtime_maps, encode_song, compute_stats as compute_encoded_stats


def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_structured_only_songs(songs):
    return {
        song_id: song
        for song_id, song in songs.items()
        if song.get("meta", {}).get("ori_uid") is not None and len(song.get("sections", [])) > 0
    }


def canonicalize_dataset(dataset, keep_raw=False):
    reporter = Reporter()
    canonical = {
        song_id: normalize_song(song_id, song, reporter=reporter, keep_raw=keep_raw)
        for song_id, song in dataset.items()
    }
    return canonical, reporter.to_dict()


def encode_dataset(canonical, metadata_dir):
    vocabs, specs = load_metadata(metadata_dir)
    runtime_maps = build_runtime_maps(specs)
    encoded = {
        song_id: encode_song(song_id, song, vocabs=vocabs, specs=specs, runtime_maps=runtime_maps)
        for song_id, song in canonical.items()
    }
    return encoded, runtime_maps


def parse_args():
    parser = argparse.ArgumentParser(description="Run full HookTheory data pipeline.")
    parser.add_argument("--raw-json", required=True, help="Path to raw HookTheory JSON")
    parser.add_argument("--structure-train", required=True, help="Path to HookTheoryStructure.train.jsonl")
    parser.add_argument("--structure-val", required=True, help="Path to HookTheoryStructure.val.jsonl")
    parser.add_argument("--structure-test", required=True, help="Path to HookTheoryStructure.test.jsonl")
    parser.add_argument("--metadata-dir", default="metadata", help="Path to metadata directory")
    parser.add_argument(
        "--out-dir",
        default="data/ProcessedHookTheory",
        help="Output directory for processed artifacts (default: data/ProcessedHookTheory)",
    )
    parser.add_argument("--compute-stats", action="store_true", help="Save extra stats files for all stages")
    parser.add_argument("--keep-raw", action="store_true", help="Keep *_raw fields during canonicalization")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    canonical_full_dir = out_dir / "canonical_full"
    canonical_structured_dir = out_dir / "canonical_structured_only"

    print("[PIPELINE] Step 1/4: preprocess")
    songs, processed_stats, attach_stats = build_processed_dataset(
        raw_json_path=args.raw_json,
        structure_train_path=args.structure_train,
        structure_val_path=args.structure_val,
        structure_test_path=args.structure_test,
        do_compute_stats=args.compute_stats,
    )
    structured_only = build_structured_only_songs(songs)

    # exactly as in hk_data_pipeline
    dump_json(songs, out_dir / "hooktheory_processed.json")
    dump_json(structured_only, out_dir / "hooktheory_processed_structured_only.json")
    dump_json(attach_stats.get("unmatched_song_ids", []), out_dir / "hooktheory_processed_unmatched_ids.json")
    dump_json(attach_stats.get("unknown_split_song_ids", []), out_dir / "hooktheory_processed_unknown_split_ids.json")
    if args.compute_stats:
        dump_json(processed_stats, out_dir / "hooktheory_processed.stats.json")

    print("[PIPELINE] Step 2/4: timelines")
    timeline, timeline_aggregate = build_original_song_timelines(structured_only)
    dump_json(timeline, out_dir / "original_songs_timeline.json")
    if args.compute_stats:
        timeline_stats = compute_timeline_stats(timeline)
        dump_json(timeline_aggregate, out_dir / "original_songs_aggregate.stats.json")
        dump_json(timeline_stats, out_dir / "original_songs_timeline.stats.json")

    print("[PIPELINE] Step 3/4: canonicalize")
    canonical_full, canonical_full_report = canonicalize_dataset(songs, keep_raw=args.keep_raw)
    canonical_structured, canonical_structured_report = canonicalize_dataset(structured_only, keep_raw=args.keep_raw)

    dump_json(canonical_full, canonical_full_dir / "hooktheory_canonical.json")
    dump_json(canonical_structured, canonical_structured_dir / "hooktheory_canonical.json")

    canonical_full_stats = compute_canonical_stats(canonical_full)
    canonical_structured_stats = compute_canonical_stats(canonical_structured)
    dump_json(canonical_full_stats, canonical_full_dir / "hooktheory_canonical.stats.json")
    dump_json(canonical_full_report, canonical_full_dir / "hooktheory_canonical.report.json")
    dump_json(canonical_structured_stats, canonical_structured_dir / "hooktheory_canonical.stats.json")
    dump_json(canonical_structured_report, canonical_structured_dir / "hooktheory_canonical.report.json")

    print("[PIPELINE] Step 4/4: encode")
    encoded_full, runtime_maps = encode_dataset(canonical_full, metadata_dir=args.metadata_dir)
    encoded_structured, _ = encode_dataset(canonical_structured, metadata_dir=args.metadata_dir)

    # two encoded jsons for canonicalized full vs structured-only
    dump_json(encoded_full, canonical_full_dir / "teacher_encoded.json")
    dump_json(encoded_structured, canonical_structured_dir / "teacher_encoded.json")

    if args.compute_stats:
        dump_json(compute_encoded_stats(encoded_full), canonical_full_dir / "teacher_encoded.stats.json")
        dump_json(compute_encoded_stats(encoded_structured), canonical_structured_dir / "teacher_encoded.stats.json")
        dump_json(runtime_maps, canonical_full_dir / "teacher_encoder_maps.json")
        dump_json(runtime_maps, canonical_structured_dir / "teacher_encoder_maps.json")

    print("[PIPELINE] done")
    print(f"[PIPELINE] outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
