#!/usr/bin/env python3
"""OOD evaluation for TeacherGNN on unseen theory-aware corruption modes."""



from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from src.dataloader.song_corruptions import (
    NEAR_BENIGN_CORRUPTIONS,
    STRICT_BENIGN_CORRUPTIONS,
    corrupt_song_obj,
)
from src.dataloader.theory_helpers import build_theory_context
from src.dataloader.utils_graph import build_graph_from_encoded
from src.models.teacher_gnn import TeacherGNN

OOD_MODES = [
    "out_of_key_note",
    "local_semitone_fragment_shift",
    "octave_leap_violation",
    "semitone_from_bass_or_chord_tone",
]

DEFAULT_MODES = [
  "strongbeat_nonchord_note",
  "borrowed_melody_conflict",
  "borrowed_kind_toggle_without_melody_change",
  "note_onset_shift",
  "strong_weak_beat_flip",
  "functional_progression_violation_strict"
]
BENIGN_ALL_MODES = STRICT_BENIGN_CORRUPTIONS + NEAR_BENIGN_CORRUPTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True, help="Path to encoded dataset JSON (list or dict).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to TeacherGNN checkpoint (.pt).")
    parser.add_argument("--config", type=Path, required=True, help="Path to composed_config.yaml matching checkpoint.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate (default: test).")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Corruption modes to evaluate independently.",
    )
    parser.add_argument(
        "--mode-set",
        choices=["default", "ood", "strict_benign", "near_benign", "benign_all"],
        default="default",
        help="Predefined mode set; used only when --modes is not provided.",
    )
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed for reproducible corruption sampling.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of split songs to evaluate.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/teacher_ood_eval"),
        help="Directory for CSV/XLSX outputs.",
    )
    return parser.parse_args()


def resolve_modes(args: argparse.Namespace) -> list[str]:
    if args.modes:
        return list(args.modes)
    if args.mode_set == "ood":
        return list(OOD_MODES)
    if args.mode_set == "strict_benign":
        return list(STRICT_BENIGN_CORRUPTIONS)
    if args.mode_set == "near_benign":
        return list(NEAR_BENIGN_CORRUPTIONS)
    if args.mode_set == "benign_all":
        return list(BENIGN_ALL_MODES)
    return list(DEFAULT_MODES)


def load_dataset_json(path: Path) -> list[tuple[str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        songs: list[tuple[str, dict[str, Any]]] = []
        for index, song_obj in enumerate(payload):
            if not isinstance(song_obj, dict):
                raise ValueError(f"dataset list item at index={index} is not a JSON object")
            meta = song_obj.get("meta", {})
            song_id = str(meta.get("song_id") or meta.get("id") or f"song_{index}")
            songs.append((song_id, song_obj))
        return songs

    if isinstance(payload, dict):
        songs_dict: list[tuple[str, dict[str, Any]]] = []
        for key, song_obj in payload.items():
            if not isinstance(song_obj, dict):
                raise ValueError(f"dataset entry for key={key!r} is not a JSON object")
            songs_dict.append((str(key), song_obj))
        return songs_dict

    raise ValueError("Dataset JSON must be either a list[song_obj] or dict[song_id, song_obj].")


def iter_split_songs(
    songs: list[tuple[str, dict[str, Any]]],
    split: str,
    limit: int | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    selected = [
        (song_id, song_obj)
        for song_id, song_obj in songs
        if song_obj.get("meta", {}).get("split") == split
    ]
    if limit is not None:
        return selected[: max(0, limit)]
    return selected


def load_model_from_config_and_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    sample_song_obj: dict[str, Any],
    device: torch.device,
) -> TeacherGNN:
    cfg = OmegaConf.load(config_path)
    sample_graph = build_graph_from_encoded(sample_song_obj)

    model = TeacherGNN.from_hetero_data(
        sample_graph,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        residual=cfg.model.use_residual,
        encoder_hidden_dims=list(cfg.model.encoder_hidden_dims),
        pooling_mode=cfg.model.pooling_mode,
        pooling_output_dim=cfg.model.pooling_output_dim,
        score_head_hidden_dim=cfg.model.score_head_hidden_dim,
        reconstruction_head_hidden_dim=cfg.model.reconstruction_head_hidden_dim,
        enabled_heads=OmegaConf.to_container(cfg.losses.enabled_heads, resolve=True),
        use_note_score_head=bool(cfg.model.use_note_score_head),
        use_chord_score_head=bool(cfg.model.use_chord_score_head),
        use_onset_score_head=bool(cfg.model.use_onset_score_head),
        local_score_head_hidden_dim=cfg.model.local_score_head_hidden_dim,
        use_hybrid_graph_scorer=bool(cfg.model.use_hybrid_graph_scorer),
        local_summary_use_mean=bool(cfg.model.local_summary_use_mean),
        local_summary_use_max=bool(cfg.model.local_summary_use_max),
        local_summary_use_topk_mean=bool(cfg.model.local_summary_use_topk_mean),
        local_summary_topk=int(cfg.model.local_summary_topk),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def score_song(model: TeacherGNN, song_obj: dict[str, Any], device: torch.device) -> float:
    graph = build_graph_from_encoded(song_obj)
    batch = Batch.from_data_list([graph]).to(device)
    outputs = model(batch)
    return float(outputs["graph_score"].view(-1)[0].item())


def apply_single_mode_corruption(
    song_obj: dict[str, Any],
    mode: str,
    theory_ctx: dict[str, Any],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    corrupted_song_obj, metadata = corrupt_song_obj(
        copy.deepcopy(song_obj),
        corruption_modes=[mode],
        corruption_cfg={},
        theory_ctx=theory_ctx,
        rng=rng,
    )
    return corrupted_song_obj, metadata


def build_rows_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        return pd.DataFrame(
            columns=[
                "song_id",
                "split",
                "mode",
                "applied",
                "topology_changed",
                "note_corrupted_indices",
                "chord_corrupted_indices",
                "onset_corrupted_indices",
                "score_real",
                "score_corrupted",
                "score_gap",
                "rank_success",
                "metadata_json",
            ]
        )

    base_columns = [
        "song_id",
        "split",
        "mode",
        "applied",
        "topology_changed",
        "note_corrupted_indices",
        "chord_corrupted_indices",
        "onset_corrupted_indices",
        "score_real",
        "score_corrupted",
        "score_gap",
        "rank_success",
        "metadata_json",
    ]
    tail_columns = [col for col in rows_df.columns if col not in base_columns]
    return rows_df[base_columns + tail_columns]


def build_summary_dataframe(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame(
            columns=[
                "mode",
                "n_total",
                "n_applied",
                "applied_rate",
                "success_count",
                "success_rate",
                "mean_gap",
                "median_gap",
                "std_gap",
                "mean_score_real",
                "mean_score_corrupted",
                "median_score_real",
                "median_score_corrupted",
                "positive_gap_count",
                "non_positive_gap_count",
            ]
        )

    summary_rows: list[dict[str, Any]] = []

    for mode, mode_df in rows_df.groupby("mode", sort=False):
        n_total = int(len(mode_df))
        applied_df = mode_df[mode_df["applied"] == True]  # noqa: E712
        n_applied = int(len(applied_df))

        success_count = int(applied_df["rank_success"].fillna(0).sum()) if n_applied else 0
        success_rate = (success_count / n_applied) if n_applied else float("nan")
        applied_rate = (n_applied / n_total) if n_total else float("nan")

        gap_series = applied_df["score_gap"] if n_applied else pd.Series(dtype=float)
        positive_gap_count = int((gap_series > 0).sum()) if n_applied else 0
        non_positive_gap_count = int((gap_series <= 0).sum()) if n_applied else 0

        summary_rows.append(
            {
                "mode": mode,
                "n_total": n_total,
                "n_applied": n_applied,
                "applied_rate": applied_rate,
                "success_count": success_count,
                "success_rate": success_rate,
                "mean_gap": float(gap_series.mean()) if n_applied else float("nan"),
                "median_gap": float(gap_series.median()) if n_applied else float("nan"),
                "std_gap": float(gap_series.std()) if n_applied else float("nan"),
                "mean_score_real": float(applied_df["score_real"].mean()) if n_applied else float("nan"),
                "mean_score_corrupted": float(applied_df["score_corrupted"].mean()) if n_applied else float("nan"),
                "median_score_real": float(applied_df["score_real"].median()) if n_applied else float("nan"),
                "median_score_corrupted": float(applied_df["score_corrupted"].median()) if n_applied else float("nan"),
                "positive_gap_count": positive_gap_count,
                "non_positive_gap_count": non_positive_gap_count,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df.sort_values(
        by=["success_rate", "mean_gap", "applied_rate"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def write_outputs(rows_df: pd.DataFrame, summary_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows_path = outdir / "teacher_ood_eval_rows.csv"
    summary_path = outdir / "teacher_ood_eval_summary.csv"
    excel_path = outdir / "teacher_ood_eval.xlsx"

    rows_df.to_csv(rows_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        rows_df.to_excel(writer, index=False, sheet_name="rows")

    print(f"Saved rows CSV: {rows_path}")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved Excel workbook: {excel_path}")


def run_eval(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device(args.device)
    all_songs = load_dataset_json(args.dataset_json)
    split_songs = iter_split_songs(all_songs, split=args.split, limit=args.limit)
    if not split_songs:
        raise ValueError(f"No songs found for split={args.split!r} in {args.dataset_json}")

    model = load_model_from_config_and_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        sample_song_obj=split_songs[0][1],
        device=device,
    )
    theory_ctx = build_theory_context()
    modes = resolve_modes(args)

    rows: list[dict[str, Any]] = []
    for song_idx, (song_id, song_obj) in enumerate(split_songs):
        score_real = score_song(model, song_obj, device)
        for mode_idx, mode in enumerate(modes):
            rng = random.Random(args.seed + song_idx * 1000 + mode_idx)
            corrupted_song_obj, metadata = apply_single_mode_corruption(song_obj, mode, theory_ctx, rng)
            applied = bool(metadata.get("applied", False))

            score_corrupted: float | None = None
            score_gap: float | None = None
            rank_success: int | None = None
            if applied:
                score_corrupted = score_song(model, corrupted_song_obj, device)
                score_gap = score_real - score_corrupted
                rank_success = int(score_real > score_corrupted)

            details = metadata.get("details", {})
            row = {
                "song_id": song_id,
                "split": song_obj.get("meta", {}).get("split"),
                "mode": mode,
                "applied": applied,
                "topology_changed": bool(metadata.get("topology_changed", False)),
                "note_corrupted_indices": metadata.get("note_corrupted_indices", []),
                "chord_corrupted_indices": metadata.get("chord_corrupted_indices", []),
                "onset_corrupted_indices": metadata.get("onset_corrupted_indices", []),
                "score_real": score_real,
                "score_corrupted": score_corrupted,
                "score_gap": score_gap,
                "rank_success": rank_success,
                "metadata_json": json.dumps(details, ensure_ascii=False, sort_keys=True),
                "mode_family": metadata.get("mode_family"),
                "corruption_name": metadata.get("corruption_name", mode),
                "corruption_params_json": json.dumps(metadata.get("corruption_params", {}), ensure_ascii=False, sort_keys=True),
                "reason_skipped": metadata.get("reason_skipped"),
                "n_notes_modified": (
                    int(metadata.get("n_notes_modified", 0) or 0)
                    if mode in BENIGN_ALL_MODES or int(metadata.get("n_notes_modified", 0) or 0) > 0
                    else None
                ),
                "n_chords_modified": (
                    int(metadata.get("n_chords_modified", 0) or 0)
                    if mode in BENIGN_ALL_MODES or int(metadata.get("n_chords_modified", 0) or 0) > 0
                    else None
                ),
                "mode_group": (
                    "strict_benign"
                    if mode in STRICT_BENIGN_CORRUPTIONS
                    else "near_benign" if mode in NEAR_BENIGN_CORRUPTIONS else "other"
                ),
            }
            if "reference_role" in details:
                row["reference_role"] = details.get("reference_role")
            if "covering_chord_index" in details:
                row["covering_chord_index"] = details.get("covering_chord_index")
            if "active_mode_name" in details:
                row["active_mode_name"] = details.get("active_mode_name")
            rows.append(row)

    rows_df = build_rows_dataframe(rows)
    summary_df = build_summary_dataframe(rows_df)
    return rows_df, summary_df


def main() -> None:
    args = parse_args()
    rows_df, summary_df = run_eval(args)
    write_outputs(rows_df, summary_df, args.outdir)


if __name__ == "__main__":
    main()
