#!/usr/bin/env python3
"""Run the section-aware TeacherGNN training pipeline end to end.

The script intentionally launches normal project entrypoints instead of
reimplementing training logic:

1. audit original-song timelines;
2. assemble multi-section encoded songs;
3. train Stage 1 on short clips;
4. fine-tune Stage 2 on assembled section songs;
5. optionally build a mixed JSON and fine-tune Stage 3.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


LOCAL_THEORY_MODES = [
    "strongbeat_nonchord_note",
    "borrowed_melody_conflict",
    "borrowed_kind_toggle_without_melody_change",
    "melody_semitone_add_clash",
    "melody_suspension_clash",
    "melody_alteration_clash",
    "melody_omit_core_tone_conflict",
    "inversion_bass_continuity_conflict",
    "note_onset_shift",
    "chord_onset_shift",
    "strong_weak_beat_flip",
    "duration_stretch_shrink_note",
    "duration_stretch_shrink_chord",
    "functional_progression_violation_strict",
]

SECTION_MODES = [
    "adjacent_section_swap",
    "non_adjacent_section_swap",
    "section_duplicate",
    "section_drop_keep_silence",
    "section_drop_and_close_gap",
    "section_entry_non_tonic_substitution",
    "section_exit_non_dominant_substitution",
]

STAGE2_MODES = SECTION_MODES + [
    "strongbeat_nonchord_note",
    "borrowed_melody_conflict",
    "melody_semitone_add_clash",
    "melody_suspension_clash",
    "melody_alteration_clash",
    "melody_omit_core_tone_conflict",
    "inversion_bass_continuity_conflict",
    "note_onset_shift",
    "chord_onset_shift",
    "strong_weak_beat_flip",
    "duration_stretch_shrink_note",
    "duration_stretch_shrink_chord",
    "functional_progression_violation_strict",
]

STAGE3_MODES = SECTION_MODES + LOCAL_THEORY_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child commands.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Training device override.")
    parser.add_argument("--run-root", type=Path, default=None, help="Root output directory for this multistage run.")
    parser.add_argument("--run-prefix", default="teacher_sections", help="Prefix used in Hydra run names.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--smoke", action="store_true", help="Use tiny epochs/sample limits for pipeline validation.")

    parser.add_argument("--encoded-json", type=Path, default=Path("data/HTCanon/encoded_full/teacher_encoded.json"))
    parser.add_argument("--timeline-json", type=Path, default=Path("data/HTCanon/HK_processed/original_songs_timeline.json"))
    parser.add_argument("--audit-outdir", type=Path, default=Path("outputs/timeline_audit"))
    parser.add_argument("--assembled-outdir", type=Path, default=Path("outputs/assembled_sections"))
    parser.add_argument(
        "--assembled-json",
        type=Path,
        default=Path("outputs/assembled_sections/teacher_encoded_assembled_compact_gap.json"),
    )
    parser.add_argument(
        "--mixed-json",
        type=Path,
        default=None,
        help="Mixed short+assembled JSON. Defaults to <run-root>/prepared_data/teacher_encoded_mixed_short_assembled.json.",
    )
    parser.add_argument("--assembled-repeats", type=int, default=12)
    parser.add_argument("--skip-assembly", action="store_true", help="Reuse existing audit/assembled JSON.")
    parser.add_argument("--render-assembled-midi-smoke", action="store_true")

    parser.add_argument("--stage1-checkpoint", type=Path, default=None, help="Skip Stage 1 and reuse this checkpoint.")
    parser.add_argument("--stage2-checkpoint", type=Path, default=None, help="Skip Stage 2 and reuse this checkpoint.")
    parser.add_argument("--skip-stage3", action="store_true")

    parser.add_argument("--stage1-epochs", type=int, default=500)
    parser.add_argument("--stage1-mlm-epochs", type=int, default=300)
    parser.add_argument("--stage1-corruption-epochs", type=int, default=200)
    parser.add_argument("--stage1-batch-size", type=int, default=32)
    parser.add_argument("--stage1-lr", type=float, default=3e-4)
    parser.add_argument("--stage1-limit-train-samples", type=int, default=None)
    parser.add_argument("--stage1-limit-val-samples", type=int, default=None)

    parser.add_argument("--stage2-epochs", type=int, default=120)
    parser.add_argument("--stage2-batch-size", type=int, default=16)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--stage2-limit-train-samples", type=int, default=None)
    parser.add_argument("--stage2-limit-val-samples", type=int, default=None)
    parser.add_argument("--stage2-section-weight", type=float, default=0.25)
    parser.add_argument("--stage2-local-weight", type=float, default=0.75)

    parser.add_argument("--stage3-epochs", type=int, default=60)
    parser.add_argument("--stage3-batch-size", type=int, default=16)
    parser.add_argument("--stage3-lr", type=float, default=5e-5)
    parser.add_argument("--stage3-limit-train-samples", type=int, default=None)
    parser.add_argument("--stage3-limit-val-samples", type=int, default=None)
    parser.add_argument("--stage3-section-weight", type=float, default=0.20)
    parser.add_argument("--stage3-local-weight", type=float, default=0.80)
    return parser.parse_args()


def hydra_list(values: list[str]) -> str:
    return "[" + ",".join(values) + "]"


def command_to_text(cmd: list[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(f"\n$ {command_to_text(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], check=True)


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def checkpoint_from_run(run_dir: Path) -> Path:
    candidates = [
        run_dir / "checkpoints" / "best_rank_acc.pt",
        run_dir / "checkpoints" / "last.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No usable checkpoint found under {run_dir / 'checkpoints'}")


def train_command(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    run_name: str,
    json_path: Path,
    batch_size: int,
    lr: float,
    epochs: int,
    mlm_epochs: int,
    corruption_epochs: int,
    corruption_modes: list[str] | None = None,
    init_checkpoint: Path | None = None,
    limit_train_samples: int | None = None,
    limit_val_samples: int | None = None,
    section_weight: float | None = None,
    local_weight: float | None = None,
) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "src.training.train_teacher",
        f"hydra.run.dir={run_dir}",
        f"run_name={run_name}",
        f"data.json_path={json_path}",
        f"dataloader.batch_size={batch_size}",
        f"optimizer.lr={lr}",
        f"training.epochs={epochs}",
        f"experiment.epochs={epochs}",
        f"scheduler.t_max={max(1, epochs)}",
        f"training.mlm_ssl_epochs={mlm_epochs}",
        f"training.corruption_epochs={corruption_epochs}",
        f"device={args.device}",
        f"training.device={args.device}",
    ]
    if corruption_modes is not None:
        cmd.append(f"dataloader.corruption_modes={hydra_list(corruption_modes)}")
    if limit_train_samples is not None:
        cmd.append(f"training.limit_train_samples={int(limit_train_samples)}")
    if limit_val_samples is not None:
        cmd.append(f"training.limit_val_samples={int(limit_val_samples)}")
    if section_weight is not None:
        cmd.append(f"+dataloader.theory_aware.corruption_family_weights.section={float(section_weight)}")
    if local_weight is not None:
        cmd.append(f"+dataloader.theory_aware.corruption_family_weights.local={float(local_weight)}")
    if init_checkpoint is not None:
        cmd.extend(
            [
                f"training.init_checkpoint={init_checkpoint}",
                "training.init_checkpoint_strict=true",
            ]
        )
    return cmd


def build_mixed_json(original_path: Path, assembled_path: Path, out_path: Path, assembled_repeats: int) -> None:
    with original_path.open("r", encoding="utf-8") as handle:
        original = json.load(handle)
    with assembled_path.open("r", encoding="utf-8") as handle:
        assembled = json.load(handle)
    if not isinstance(original, dict) or not isinstance(assembled, dict):
        raise ValueError("Expected both original and assembled encoded JSON files to be dicts.")

    mixed: dict[str, dict] = {}
    for song_id, song in original.items():
        item = copy.deepcopy(song)
        item["song_id"] = song_id
        if isinstance(item.get("meta"), dict):
            item["meta"]["mixed_dataset_source"] = "original_short"
        mixed[f"orig_{song_id}"] = item

    for repeat_idx in range(max(1, int(assembled_repeats))):
        for song_id, song in assembled.items():
            new_id = f"assembled_r{repeat_idx}_{song_id}"
            item = copy.deepcopy(song)
            item["song_id"] = new_id
            if isinstance(item.get("meta"), dict):
                item["meta"]["song_id"] = new_id
                item["meta"]["mixed_dataset_source"] = "assembled"
                item["meta"]["mixed_dataset_repeat"] = repeat_idx
            mixed[new_id] = item

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mixed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote mixed dataset: {out_path} ({len(mixed)} songs)", flush=True)


def _iter_items(data: Any) -> list[tuple[str, dict]]:
    if isinstance(data, dict):
        return [(str(song_id), song) for song_id, song in data.items() if isinstance(song, dict)]
    if isinstance(data, list):
        return [(str(index), song) for index, song in enumerate(data) if isinstance(song, dict)]
    raise ValueError("Expected encoded JSON to be a dict or list.")


def _normalize_split_name(value: Any) -> str | None:
    if value is None:
        return None
    split = str(value).strip().lower()
    if split in {"valid", "validation", "dev"}:
        return "val"
    return split


def prepare_training_json(input_path: Path, out_path: Path, *, dry_run: bool) -> Path:
    if dry_run:
        print(f"Would prepare train/val JSON: {input_path} -> {out_path}", flush=True)
        return out_path

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    items = _iter_items(data)
    for _, song in items:
        meta = song.setdefault("meta", {})
        if not isinstance(meta, dict):
            meta = {}
            song["meta"] = meta
        split = _normalize_split_name(meta.get("split", song.get("split")))
        if split is not None:
            meta["split"] = split

    train_items = [(song_id, song) for song_id, song in items if song.get("meta", {}).get("split") == "train"]
    if not train_items:
        for _, song in items:
            song.setdefault("meta", {})["split"] = "train"

    train_items = [(song_id, song) for song_id, song in items if song.get("meta", {}).get("split") == "train"]
    val_items = [(song_id, song) for song_id, song in items if song.get("meta", {}).get("split") == "val"]
    if not val_items and len(train_items) > 1:
        val_count = max(1, min(len(train_items) - 1, int(round(len(train_items) * 0.1))))
        for _, song in train_items[-val_count:]:
            song["meta"]["split"] = "val"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    split_counts: dict[str, int] = {}
    for _, song in items:
        split = str(song.get("meta", {}).get("split", "unknown"))
        split_counts[split] = split_counts.get(split, 0) + 1
    print(f"Wrote train/val-ready JSON: {out_path} splits={split_counts}", flush=True)
    return out_path


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.stage1_epochs = 4
    args.stage1_mlm_epochs = 2
    args.stage1_corruption_epochs = 2
    args.stage1_batch_size = 2
    args.stage1_limit_train_samples = 32
    args.stage1_limit_val_samples = 16
    args.stage2_epochs = 3
    args.stage2_batch_size = 2
    args.stage2_limit_train_samples = 16
    args.stage2_limit_val_samples = 8
    args.stage3_epochs = 2
    args.stage3_batch_size = 2
    args.stage3_limit_train_samples = 16
    args.stage3_limit_val_samples = 8
    args.assembled_repeats = 1


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)

    require_path(args.encoded_json, "encoded JSON")
    if not args.skip_assembly:
        require_path(args.timeline_json, "timeline JSON")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.run_root or Path("outputs/teacher_section_multistage") / f"{timestamp}_{args.run_prefix}"
    run_root.mkdir(parents=True, exist_ok=True)
    if args.mixed_json is None:
        args.mixed_json = run_root / "prepared_data" / "teacher_encoded_mixed_short_assembled.json"

    if not args.skip_assembly:
        run_command(
            [
                args.python,
                "scripts/audit_original_song_timeline.py",
                "--timeline-json",
                args.timeline_json,
                "--encoded-json",
                args.encoded_json,
                "--outdir",
                args.audit_outdir,
            ],
            dry_run=args.dry_run,
        )
        run_command(
            [
                args.python,
                "scripts/assemble_timeline_songs.py",
                "--audit-jsonl",
                args.audit_outdir / "original_song_audit.jsonl",
                "--encoded-json",
                args.encoded_json,
                "--outdir",
                args.assembled_outdir,
                "--usable-mode",
                "compact_gap",
                "--section-start-policy",
                "next_bar_gap",
                "--max-gap-sec",
                "10.0",
                "--multi-clip-segment-policy",
                "skip",
            ],
            dry_run=args.dry_run,
        )

    if args.render_assembled_midi_smoke:
        run_command(
            [
                args.python,
                "-m",
                "src.data.render_encoded_song_to_midi",
                "--encoded-json",
                args.assembled_json,
                "--output-root",
                run_root / "assembled_midi_smoke",
                "--limit",
                "32",
                "--overwrite",
                "--verbose",
            ],
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        require_path(args.assembled_json, "assembled encoded JSON")

    stage1_run_dir = run_root / "stage1_short_local"
    stage2_run_dir = run_root / "stage2_assembled_sections"
    stage3_run_dir = run_root / "stage3_mixed_sections"
    prepared_dir = run_root / "prepared_data"
    assembled_training_json = prepare_training_json(
        args.assembled_json,
        prepared_dir / "teacher_encoded_assembled_trainval.json",
        dry_run=args.dry_run,
    )

    if args.stage1_checkpoint is not None:
        stage1_checkpoint = args.stage1_checkpoint
        require_path(stage1_checkpoint, "Stage 1 checkpoint")
        print(f"Reusing Stage 1 checkpoint: {stage1_checkpoint}", flush=True)
    else:
        run_command(
            train_command(
                args,
                run_dir=stage1_run_dir,
                run_name=f"{args.run_prefix}_stage1_short_local",
                json_path=args.encoded_json,
                batch_size=args.stage1_batch_size,
                lr=args.stage1_lr,
                epochs=args.stage1_epochs,
                mlm_epochs=args.stage1_mlm_epochs,
                corruption_epochs=args.stage1_corruption_epochs,
                corruption_modes=LOCAL_THEORY_MODES,
                limit_train_samples=args.stage1_limit_train_samples,
                limit_val_samples=args.stage1_limit_val_samples,
            ),
            dry_run=args.dry_run,
        )
        stage1_checkpoint = checkpoint_from_run(stage1_run_dir) if not args.dry_run else stage1_run_dir / "checkpoints" / "best_rank_acc.pt"
        print(f"Stage 1 checkpoint: {stage1_checkpoint}", flush=True)

    if args.stage2_checkpoint is not None:
        stage2_checkpoint = args.stage2_checkpoint
        require_path(stage2_checkpoint, "Stage 2 checkpoint")
        print(f"Reusing Stage 2 checkpoint: {stage2_checkpoint}", flush=True)
    else:
        run_command(
            train_command(
                args,
                run_dir=stage2_run_dir,
                run_name=f"{args.run_prefix}_stage2_assembled_sections",
                json_path=assembled_training_json,
                batch_size=args.stage2_batch_size,
                lr=args.stage2_lr,
                epochs=args.stage2_epochs,
                mlm_epochs=0,
                corruption_epochs=args.stage2_epochs,
                corruption_modes=STAGE2_MODES,
                init_checkpoint=stage1_checkpoint,
                limit_train_samples=args.stage2_limit_train_samples,
                limit_val_samples=args.stage2_limit_val_samples,
                section_weight=args.stage2_section_weight,
                local_weight=args.stage2_local_weight,
            ),
            dry_run=args.dry_run,
        )
        stage2_checkpoint = checkpoint_from_run(stage2_run_dir) if not args.dry_run else stage2_run_dir / "checkpoints" / "best_rank_acc.pt"
        print(f"Stage 2 checkpoint: {stage2_checkpoint}", flush=True)

    if args.skip_stage3:
        print("Skipping Stage 3 mixed fine-tune.", flush=True)
        print(f"Final checkpoint: {stage2_checkpoint}", flush=True)
        return

    if args.dry_run:
        print(f"Would build mixed JSON: {args.mixed_json}", flush=True)
    else:
        build_mixed_json(args.encoded_json, assembled_training_json, args.mixed_json, args.assembled_repeats)

    run_command(
        train_command(
            args,
            run_dir=stage3_run_dir,
            run_name=f"{args.run_prefix}_stage3_mixed_sections",
            json_path=args.mixed_json,
            batch_size=args.stage3_batch_size,
            lr=args.stage3_lr,
            epochs=args.stage3_epochs,
            mlm_epochs=0,
            corruption_epochs=args.stage3_epochs,
            corruption_modes=STAGE3_MODES,
            init_checkpoint=stage2_checkpoint,
            limit_train_samples=args.stage3_limit_train_samples,
            limit_val_samples=args.stage3_limit_val_samples,
            section_weight=args.stage3_section_weight,
            local_weight=args.stage3_local_weight,
        ),
        dry_run=args.dry_run,
    )
    stage3_checkpoint = checkpoint_from_run(stage3_run_dir) if not args.dry_run else stage3_run_dir / "checkpoints" / "best_rank_acc.pt"
    print(f"Final checkpoint: {stage3_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
