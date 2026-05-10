from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch

from src.inference.infer_teacher_score import build_model_from_config, score_song


LOGGER = logging.getLogger(__name__)


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
    parser.add_argument(
        "--include-intermediates",
        action="store_true",
        help="Store detached teacher graph/pool/local-summary features for observer distillation.",
    )
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


def _finite_float_vector(value: Any, *, field_name: str) -> list[float]:
    if not isinstance(value, list):
        raise TeacherTargetBuildError(f"Teacher intermediate '{field_name}' must be a list")
    out: list[float] = []
    for idx, item in enumerate(value):
        try:
            scalar = float(item)
        except (TypeError, ValueError) as exc:
            raise TeacherTargetBuildError(f"Teacher intermediate '{field_name}[{idx}]' is not numeric: {item!r}") from exc
        if not math.isfinite(scalar):
            raise TeacherTargetBuildError(f"Teacher intermediate '{field_name}[{idx}]' is not finite: {scalar}")
        out.append(scalar)
    return out


def _copy_teacher_intermediates(row_out: dict[str, Any], score_payload: dict[str, Any]) -> None:
    if "graph_embedding" in score_payload:
        row_out["teacher_graph_embedding"] = _finite_float_vector(
            score_payload["graph_embedding"],
            field_name="graph_embedding",
        )
    if "local_score_summaries" in score_payload:
        row_out["teacher_local_score_summaries"] = _finite_float_vector(
            score_payload["local_score_summaries"],
            field_name="local_score_summaries",
        )

    pooled_by_type = score_payload.get("pooled_by_type")
    if pooled_by_type is None:
        return
    if not isinstance(pooled_by_type, dict):
        raise TeacherTargetBuildError("Teacher intermediate 'pooled_by_type' must be a mapping")
    row_out["teacher_pooled_by_type"] = {
        str(node_type): _finite_float_vector(values, field_name=f"pooled_by_type.{node_type}")
        for node_type, values in pooled_by_type.items()
    }


def _validate_unique_sample_ids(rows: list[dict[str, Any]]) -> None:
    seen_sample_ids: set[str] = set()
    for row_idx, row in enumerate(rows, start=1):
        song_id = row.get("song_id")
        if not isinstance(song_id, str) or not song_id:
            raise TeacherTargetBuildError(f"Line {row_idx}: song_id is required and must be non-empty string")
        sample_id = str(row.get("sample_id", song_id))
        if sample_id in seen_sample_ids:
            raise TeacherTargetBuildError(f"Duplicate sample_id in input manifest: '{sample_id}'")
        seen_sample_ids.add(sample_id)


def _build_teacher_model(
    rows: list[dict[str, Any]],
    teacher_checkpoint: Path,
    teacher_config: Path,
    encoded_song_field: str,
    encoded_song_root: Path | None,
    split: str | None,
    device: str,
) -> tuple[Any, torch.device]:
    device_t = torch.device(device)

    if not rows:
        raise TeacherTargetBuildError("Cannot build teacher model from an empty manifest")

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
    return model, device_t


def _build_target_row(
    *,
    model: Any,
    sample: dict[str, Any],
    encoded_song_field: str,
    encoded_song_root: Path | None,
    split: str | None,
    device_t: torch.device,
    include_intermediates: bool = False,
) -> dict[str, Any]:
    song_id = str(sample["song_id"])
    encoded_path = _resolve_encoded_song_path(sample, encoded_song_field, encoded_song_root, split)
    encoded_song = _load_encoded_song(encoded_path)
    score_payload = score_song(
        model,
        encoded_song,
        device_t,
        include_intermediates=bool(include_intermediates),
    )
    teacher_score = float(score_payload["graph_score"])
    if not math.isfinite(teacher_score):
        raise TeacherTargetBuildError(f"Teacher score for song_id='{song_id}' is not finite: {teacher_score}")

    row_out: dict[str, Any] = {
        "song_id": song_id,
        "teacher_score": teacher_score,
    }
    if split is not None:
        row_out["split"] = split
    for passthrough_key in (
        "sample_id",
        "midi_path",
        "tonic_pc",
        "mode_name",
        "is_corrupted",
        "corruption_name",
        "pair_group_id",
        "source_song_id",
        "tonal_group",
        "corruption_group",
        "is_valid_pair_for_rank",
    ):
        if passthrough_key in sample:
            row_out[passthrough_key] = sample[passthrough_key]
    if include_intermediates:
        _copy_teacher_intermediates(row_out, score_payload)
    return row_out


def build_teacher_targets(
    rows: list[dict[str, Any]],
    teacher_checkpoint: Path,
    teacher_config: Path,
    encoded_song_field: str,
    encoded_song_root: Path | None,
    split: str | None,
    device: str,
    include_intermediates: bool = False,
) -> list[dict[str, Any]]:
    """Build offline teacher scalar targets.

    Note:
        Teacher model bootstrap is performed using the first encoded song resolved
        from the input manifest because `build_model_from_config(...)` requires a
        representative sample graph to infer hetero input dimensions.
    """
    if not rows:
        return []

    _validate_unique_sample_ids(rows)
    model, device_t = _build_teacher_model(
        rows=rows,
        teacher_checkpoint=teacher_checkpoint,
        teacher_config=teacher_config,
        encoded_song_field=encoded_song_field,
        encoded_song_root=encoded_song_root,
        split=split,
        device=device,
    )

    out_rows: list[dict[str, Any]] = []
    for sample in rows:
        out_rows.append(
            _build_target_row(
                model=model,
                sample=sample,
                encoded_song_field=encoded_song_field,
                encoded_song_root=encoded_song_root,
                split=split,
                device_t=device_t,
                include_intermediates=bool(include_intermediates),
            )
        )
    return out_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_existing_sample_ids(path: Path) -> set[str]:
    sample_ids: set[str] = set()
    if not path.exists():
        return sample_ids
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TeacherTargetBuildError(f"Invalid existing target JSON at {path}:{line_idx}: {exc}") from exc
            sample_ids.add(str(row.get("sample_id", row.get("song_id"))))
    return sample_ids


def build_teacher_targets_jsonl(
    rows: list[dict[str, Any]],
    output_jsonl: Path,
    teacher_checkpoint: Path,
    teacher_config: Path,
    encoded_song_field: str,
    encoded_song_root: Path | None,
    split: str | None,
    device: str,
    include_intermediates: bool = False,
    resume: bool = True,
    log_every: int = 100,
) -> int:
    """Stream teacher targets to JSONL with progress logging and resume support."""
    if not rows:
        write_jsonl(output_jsonl, [])
        return 0

    _validate_unique_sample_ids(rows)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    existing_sample_ids = _load_existing_sample_ids(output_jsonl) if resume else set()
    mode = "a" if resume and output_jsonl.exists() else "w"
    total = len(rows)
    done_existing = len(existing_sample_ids)
    if done_existing:
        LOGGER.info(
            "Teacher targets split=%s resuming from %s with %d/%d existing rows",
            split,
            output_jsonl,
            done_existing,
            total,
        )

    model, device_t = _build_teacher_model(
        rows=rows,
        teacher_checkpoint=teacher_checkpoint,
        teacher_config=teacher_config,
        encoded_song_field=encoded_song_field,
        encoded_song_root=encoded_song_root,
        split=split,
        device=device,
    )

    written = 0
    started = time.perf_counter()
    last_logged = started
    with output_jsonl.open(mode, encoding="utf-8") as handle:
        for idx, sample in enumerate(rows, start=1):
            sample_id = str(sample.get("sample_id", sample["song_id"]))
            if sample_id in existing_sample_ids:
                continue

            row_out = _build_target_row(
                model=model,
                sample=sample,
                encoded_song_field=encoded_song_field,
                encoded_song_root=encoded_song_root,
                split=split,
                device_t=device_t,
                include_intermediates=bool(include_intermediates),
            )
            handle.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            written += 1

            processed = done_existing + written
            if log_every > 0 and (written % log_every == 0 or processed == total):
                handle.flush()
                now = time.perf_counter()
                elapsed = max(1e-9, now - started)
                recent = max(1e-9, now - last_logged)
                rate = written / elapsed
                remaining = max(0, total - processed)
                eta_seconds = remaining / max(rate, 1e-9)
                LOGGER.info(
                    "Teacher targets split=%s progress=%d/%d written=%d rate=%.2f samples/s eta=%.1f min",
                    split,
                    processed,
                    total,
                    written,
                    rate,
                    eta_seconds / 60.0,
                )
                last_logged = now

    LOGGER.info(
        "Teacher targets split=%s complete existing=%d written=%d total=%d output=%s",
        split,
        done_existing,
        written,
        total,
        output_jsonl,
    )
    return done_existing + written


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    rows = load_jsonl_rows(args.input_jsonl)
    build_teacher_targets_jsonl(
        rows=rows,
        output_jsonl=args.output_jsonl,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_config=args.teacher_config,
        encoded_song_field=args.encoded_song_field,
        encoded_song_root=args.encoded_song_root,
        split=args.split,
        device=args.device,
        include_intermediates=bool(args.include_intermediates),
        resume=True,
    )


if __name__ == "__main__":
    main()
