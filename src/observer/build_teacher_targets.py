from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from src.inference.infer_teacher_score import build_model_from_config, score_song


class TeacherTargetBuildError(ValueError):
    """Raised when teacher target dump cannot be built due to invalid inputs."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline teacher scalar targets for observer training.")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Observer input JSONL manifest.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL with teacher scores.")
    parser.add_argument("--split", type=str, default=None, help="Optional split name to store in output rows.")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True, help="Teacher checkpoint path (.pt).")
    parser.add_argument("--teacher-config", type=Path, required=True, help="Teacher composed config path (.yaml).")
    parser.add_argument(
        "--encoded-song-field",
        type=str,
        default="encoded_song_path",
        help="Input JSONL field containing path to encoded song JSON used by teacher.",
    )
    parser.add_argument(
        "--encoded-song-root",
        type=Path,
        default=None,
        help="Fallback root for encoded songs, resolved as <root>/<split>/<song_id>.json when field is absent.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TeacherTargetBuildError(f"Invalid JSON at line {line_idx}: {exc}") from exc
            if not isinstance(payload, dict):
                raise TeacherTargetBuildError(f"Line {line_idx}: row must be a JSON object")
            rows.append(payload)
    return rows


def _resolve_encoded_song_path(
    sample: dict[str, Any],
    encoded_song_field: str,
    encoded_song_root: Path | None,
    split: str | None,
) -> Path:
    if encoded_song_field in sample and sample[encoded_song_field]:
        return Path(str(sample[encoded_song_field]))
    if encoded_song_root is not None:
        folder_split = split or str(sample.get("split") or "")
        if folder_split:
            return encoded_song_root / folder_split / f"{sample['song_id']}.json"
        return encoded_song_root / f"{sample['song_id']}.json"
    raise TeacherTargetBuildError(
        f"song_id='{sample.get('song_id')}' is missing '{encoded_song_field}' and --encoded-song-root is not set"
    )


def _load_encoded_song(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise TeacherTargetBuildError(f"Encoded song JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TeacherTargetBuildError(f"Encoded song must be a JSON object: {path}")
    return payload


def build_teacher_targets(
    rows: list[dict[str, Any]],
    teacher_checkpoint: Path,
    teacher_config: Path,
    encoded_song_field: str,
    encoded_song_root: Path | None,
    split: str | None,
    device: str,
) -> list[dict[str, Any]]:
    """Build offline teacher scalar targets.

    Note:
        Teacher model bootstrap is performed using the first encoded song resolved
        from the input manifest because `build_model_from_config(...)` requires a
        representative sample graph to infer hetero input dimensions.
    """
    device_t = torch.device(device)

    if not rows:
        return []

    seen_song_ids: set[str] = set()
    for row_idx, row in enumerate(rows, start=1):
        song_id = row.get("song_id")
        if not isinstance(song_id, str) or not song_id:
            raise TeacherTargetBuildError(f"Line {row_idx}: song_id is required and must be non-empty string")
        if song_id in seen_song_ids:
            raise TeacherTargetBuildError(f"Duplicate song_id in input manifest: '{song_id}'")
        seen_song_ids.add(song_id)

    first = rows[0]
    first_song_id = str(first["song_id"])
    first_encoded_path: Path | None = None
    try:
        first_encoded_path = _resolve_encoded_song_path(first, encoded_song_field, encoded_song_root, split)
        first_encoded = _load_encoded_song(first_encoded_path)
        # build_model_from_config expects OmegaConf config object. Import lazily to avoid
        # making it a hard dependency for module import.
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(teacher_config)
        model = build_model_from_config(cfg, first_encoded, teacher_checkpoint, device_t)
    except Exception as exc:  # noqa: BLE001
        raise TeacherTargetBuildError(
            "Failed to bootstrap teacher model on the first sample "
            f"(song_id='{first_song_id}', encoded_song_path='{first_encoded_path}'): {exc}"
        ) from exc

    out_rows: list[dict[str, Any]] = []
    for sample in rows:
        song_id = str(sample["song_id"])

        encoded_path = _resolve_encoded_song_path(sample, encoded_song_field, encoded_song_root, split)
        encoded_song = _load_encoded_song(encoded_path)
        score_payload = score_song(model, encoded_song, device_t)
        teacher_score = float(score_payload["graph_score"])
        if not math.isfinite(teacher_score):
            raise TeacherTargetBuildError(f"Teacher score for song_id='{song_id}' is not finite: {teacher_score}")

        row_out: dict[str, Any] = {
            "song_id": song_id,
            "teacher_score": teacher_score,
        }
        if split is not None:
            row_out["split"] = split
        for passthrough_key in ("sample_id", "midi_path", "tonic_pc", "mode_name", "is_corrupted", "corruption_name", "pair_group_id", "source_song_id", "tonal_group", "corruption_group", "is_valid_pair_for_rank"):
            if passthrough_key in sample:
                row_out[passthrough_key] = sample[passthrough_key]
        out_rows.append(row_out)
    return out_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rows = load_jsonl_rows(args.input_jsonl)
    output_rows = build_teacher_targets(
        rows=rows,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_config=args.teacher_config,
        encoded_song_field=args.encoded_song_field,
        encoded_song_root=args.encoded_song_root,
        split=args.split,
        device=args.device,
    )
    write_jsonl(args.output_jsonl, output_rows)


if __name__ == "__main__":
    main()
