import argparse
import json
from pathlib import Path

from preprocess_hooktheory import build_processed_dataset
from build_preprocess_song_timelines import build_original_song_timelines, compute_stats as compute_timeline_stats
from canonicalize_hooktheory import Reporter, normalize_song, compute_stats as compute_canonical_stats
from encode_teacher_features import load_metadata, build_runtime_maps, encode_song, compute_stats as compute_encoded_stats


def dump_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[PIPELINE] Step 1/4: preprocess")
    songs, processed_stats, attach_stats = build_processed_dataset(
        raw_json_path=args.raw_json,
        structure_train_path=args.structure_train,
        structure_val_path=args.structure_val,
        structure_test_path=args.structure_test,
        do_compute_stats=args.compute_stats,
    )
    dump_json(songs, out_dir / "processed.json")
    if args.compute_stats:
        dump_json(processed_stats, out_dir / "processed.stats.json")
        dump_json(attach_stats, out_dir / "processed.attach.stats.json")

    print("[PIPELINE] Step 2/4: timelines")
    timeline, timeline_aggregate = build_original_song_timelines(songs)
    dump_json(timeline, out_dir / "timeline.json")
    if args.compute_stats:
        dump_json(timeline_aggregate, out_dir / "timeline.aggregate.stats.json")
        dump_json(compute_timeline_stats(timeline), out_dir / "timeline.stats.json")

    print("[PIPELINE] Step 3/4: canonicalize")
    reporter = Reporter()
    canonical = {
        song_id: normalize_song(song_id, song, reporter=reporter, keep_raw=args.keep_raw)
        for song_id, song in songs.items()
    }
    dump_json(canonical, out_dir / "canonical.json")
    if args.compute_stats:
        dump_json(compute_canonical_stats(canonical), out_dir / "canonical.stats.json")
        dump_json(reporter.to_dict(), out_dir / "canonical.report.json")

    print("[PIPELINE] Step 4/4: encode")
    vocabs, specs = load_metadata(args.metadata_dir)
    runtime_maps = build_runtime_maps(specs)
    encoded = {
        song_id: encode_song(song_id, song, vocabs=vocabs, specs=specs, runtime_maps=runtime_maps)
        for song_id, song in canonical.items()
    }
    dump_json(encoded, out_dir / "encoded.json")
    if args.compute_stats:
        dump_json(compute_encoded_stats(encoded), out_dir / "encoded.stats.json")
        dump_json(runtime_maps, out_dir / "encoded.maps.json")

    print("[PIPELINE] done")
    print(f"[PIPELINE] outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
