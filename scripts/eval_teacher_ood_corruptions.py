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
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=None,
        help="Path to encoded dataset JSON (list or dict). Required unless --pair-corpus-root is used.",
    )
    parser.add_argument(
        "--pair-corpus-root",
        type=Path,
        default=None,
        help="Optional fixed PairCorpus root produced by src.observer.build_observer_pair_dataset.",
    )
    parser.add_argument(
        "--pair-corpus-split",
        default=None,
        help="PairCorpus split file prefix to evaluate. Defaults to --split.",
    )
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
        "--rank-probe-size",
        type=int,
        default=0,
        help=(
            "If > 0, build an extra per-mode ranking probe from the first N applied songs: "
            "N clean versions plus N corrupted versions sorted by TeacherGNN score."
        ),
    )
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_encoded_song(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected encoded song JSON object at {path}")
    return payload


def resolve_manifest_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


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
        backbone=str(cfg.model.get("backbone", "sage")),
        hgt_num_heads=int(cfg.model.get("hgt_num_heads", 4)),
        encoder_hidden_dims=list(cfg.model.encoder_hidden_dims),
        pooling_mode=cfg.model.pooling_mode,
        pooling_attention_hidden_dim=cfg.model.get("pooling_attention_hidden_dim"),
        pooling_type_attention=bool(cfg.model.get("pooling_type_attention", False)),
        pooling_output_dim=cfg.model.pooling_output_dim,
        score_head_hidden_dim=cfg.model.score_head_hidden_dim,
        reconstruction_head_hidden_dim=cfg.model.reconstruction_head_hidden_dim,
        enabled_heads=OmegaConf.to_container(cfg.losses.enabled_heads, resolve=True),
        use_note_score_head=bool(cfg.model.use_note_score_head),
        use_chord_score_head=bool(cfg.model.use_chord_score_head),
        use_onset_score_head=bool(cfg.model.use_onset_score_head),
        local_score_head_hidden_dim=cfg.model.local_score_head_hidden_dim,
        local_context_mode=str(cfg.model.get("local_context_mode", "mean")),
        local_context_num_heads=int(cfg.model.get("local_context_num_heads", 4)),
        use_hybrid_graph_scorer=bool(cfg.model.use_hybrid_graph_scorer),
        local_summary_use_mean=bool(cfg.model.local_summary_use_mean),
        local_summary_use_max=bool(cfg.model.local_summary_use_max),
        local_summary_use_topk_mean=bool(cfg.model.local_summary_use_topk_mean),
        local_summary_topk=int(cfg.model.local_summary_topk),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def empty_rank_probe_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_rows_columns = [
        "mode",
        "rank_probe_size",
        "rank",
        "variant",
        "song_id",
        "score",
        "paired_score",
        "pair_gap",
        "clean_above_own_corrupt",
        "all_clean_above_all_corrupt",
        "source_row_index",
        "song_order",
    ]
    rank_summary_columns = [
        "mode",
        "requested_probe_size",
        "n_applied",
        "n_items",
        "all_clean_above_all_corrupt",
        "global_pair_success_count",
        "global_pair_count",
        "global_rank_acc",
        "own_pair_success_count",
        "own_pair_success_rate",
        "min_clean_score",
        "min_clean_song_id",
        "max_corrupted_score",
        "max_corrupted_song_id",
        "minmax_gap",
        "weakest_clean_rank",
        "best_corrupted_rank",
    ]
    return pd.DataFrame(columns=rank_rows_columns), pd.DataFrame(columns=rank_summary_columns)


def build_rank_probe_dataframes(rows_df: pd.DataFrame, probe_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if probe_size <= 0:
        return empty_rank_probe_dataframes()
    if rows_df.empty:
        return empty_rank_probe_dataframes()

    rank_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for mode, mode_df in rows_df.groupby("mode", sort=False):
        applied_df = mode_df[
            (mode_df["applied"] == True)  # noqa: E712
            & mode_df["score_real"].notna()
            & mode_df["score_corrupted"].notna()
        ].head(probe_size)
        n_applied = int(len(applied_df))

        if n_applied == 0:
            summary_rows.append(
                {
                    "mode": mode,
                    "requested_probe_size": int(probe_size),
                    "n_applied": 0,
                    "n_items": 0,
                    "all_clean_above_all_corrupt": False,
                    "global_pair_success_count": 0,
                    "global_pair_count": 0,
                    "global_rank_acc": float("nan"),
                    "own_pair_success_count": 0,
                    "own_pair_success_rate": float("nan"),
                    "min_clean_score": float("nan"),
                    "min_clean_song_id": None,
                    "max_corrupted_score": float("nan"),
                    "max_corrupted_song_id": None,
                    "minmax_gap": float("nan"),
                    "weakest_clean_rank": None,
                    "best_corrupted_rank": None,
                }
            )
            continue

        song_ids: list[str] = []
        clean_scores: list[float] = []
        corrupted_scores: list[float] = []
        items: list[dict[str, Any]] = []

        for song_order, (source_row_index, row) in enumerate(applied_df.iterrows()):
            song_id = str(row["song_id"])
            score_real = float(row["score_real"])
            score_corrupted = float(row["score_corrupted"])
            pair_gap = score_real - score_corrupted
            own_pair_success = bool(score_real > score_corrupted)

            song_ids.append(song_id)
            clean_scores.append(score_real)
            corrupted_scores.append(score_corrupted)

            items.append(
                {
                    "mode": mode,
                    "rank_probe_size": n_applied,
                    "variant": "clean",
                    "song_id": song_id,
                    "score": score_real,
                    "paired_score": score_corrupted,
                    "pair_gap": pair_gap,
                    "clean_above_own_corrupt": own_pair_success,
                    "source_row_index": int(source_row_index),
                    "song_order": song_order,
                }
            )
            items.append(
                {
                    "mode": mode,
                    "rank_probe_size": n_applied,
                    "variant": "corrupted",
                    "song_id": song_id,
                    "score": score_corrupted,
                    "paired_score": score_real,
                    "pair_gap": pair_gap,
                    "clean_above_own_corrupt": own_pair_success,
                    "source_row_index": int(source_row_index),
                    "song_order": song_order,
                }
            )

        global_pair_success_count = sum(
            1 for clean_score in clean_scores for corrupted_score in corrupted_scores if clean_score > corrupted_score
        )
        global_pair_count = len(clean_scores) * len(corrupted_scores)
        all_clean_above_all_corrupt = bool(global_pair_count > 0 and global_pair_success_count == global_pair_count)
        own_pair_success_count = sum(
            1 for clean_score, corrupted_score in zip(clean_scores, corrupted_scores) if clean_score > corrupted_score
        )

        ranked_items = sorted(
            items,
            key=lambda item: (
                -float(item["score"]),
                0 if item["variant"] == "clean" else 1,
                int(item["song_order"]),
            ),
        )
        for rank, item in enumerate(ranked_items, start=1):
            item["rank"] = rank
            item["all_clean_above_all_corrupt"] = all_clean_above_all_corrupt
            rank_rows.append(item)

        clean_ranks = [int(item["rank"]) for item in ranked_items if item["variant"] == "clean"]
        corrupted_ranks = [int(item["rank"]) for item in ranked_items if item["variant"] == "corrupted"]
        min_clean_index = min(range(len(clean_scores)), key=lambda idx: clean_scores[idx])
        max_corrupted_index = max(range(len(corrupted_scores)), key=lambda idx: corrupted_scores[idx])
        min_clean_score = clean_scores[min_clean_index]
        max_corrupted_score = corrupted_scores[max_corrupted_index]

        summary_rows.append(
            {
                "mode": mode,
                "requested_probe_size": int(probe_size),
                "n_applied": n_applied,
                "n_items": int(len(ranked_items)),
                "all_clean_above_all_corrupt": all_clean_above_all_corrupt,
                "global_pair_success_count": int(global_pair_success_count),
                "global_pair_count": int(global_pair_count),
                "global_rank_acc": (
                    float(global_pair_success_count / global_pair_count) if global_pair_count else float("nan")
                ),
                "own_pair_success_count": int(own_pair_success_count),
                "own_pair_success_rate": float(own_pair_success_count / n_applied) if n_applied else float("nan"),
                "min_clean_score": min_clean_score,
                "min_clean_song_id": song_ids[min_clean_index],
                "max_corrupted_score": max_corrupted_score,
                "max_corrupted_song_id": song_ids[max_corrupted_index],
                "minmax_gap": float(min_clean_score - max_corrupted_score),
                "weakest_clean_rank": max(clean_ranks) if clean_ranks else None,
                "best_corrupted_rank": min(corrupted_ranks) if corrupted_ranks else None,
            }
        )

    rank_rows_df = pd.DataFrame(rank_rows)
    rank_summary_df = pd.DataFrame(summary_rows)
    if rank_rows_df.empty or rank_summary_df.empty:
        return empty_rank_probe_dataframes()
    empty_rank_rows_df, empty_rank_summary_df = empty_rank_probe_dataframes()
    return (
        rank_rows_df.reindex(columns=empty_rank_rows_df.columns).reset_index(drop=True),
        rank_summary_df.reindex(columns=empty_rank_summary_df.columns).reset_index(drop=True),
    )


def write_outputs(
    rows_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    outdir: Path,
    rank_probe_rows_df: pd.DataFrame | None = None,
    rank_probe_summary_df: pd.DataFrame | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows_path = outdir / "teacher_ood_eval_rows.csv"
    summary_path = outdir / "teacher_ood_eval_summary.csv"
    excel_path = outdir / "teacher_ood_eval.xlsx"

    rows_df.to_csv(rows_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        rows_df.to_excel(writer, index=False, sheet_name="rows")
        if rank_probe_summary_df is not None and rank_probe_rows_df is not None:
            rank_probe_summary_df.to_excel(writer, index=False, sheet_name="rank_probe_summary")
            rank_probe_rows_df.to_excel(writer, index=False, sheet_name="rank_probe_rows")

    print(f"Saved rows CSV: {rows_path}")
    print(f"Saved summary CSV: {summary_path}")
    if rank_probe_summary_df is not None and rank_probe_rows_df is not None:
        rank_probe_rows_path = outdir / "teacher_ood_rank_probe_rows.csv"
        rank_probe_summary_path = outdir / "teacher_ood_rank_probe_summary.csv"
        rank_probe_rows_df.to_csv(rank_probe_rows_path, index=False)
        rank_probe_summary_df.to_csv(rank_probe_summary_path, index=False)
        print(f"Saved rank probe rows CSV: {rank_probe_rows_path}")
        print(f"Saved rank probe summary CSV: {rank_probe_summary_path}")
    print(f"Saved Excel workbook: {excel_path}")


def run_eval(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.pair_corpus_root is not None:
        return run_pair_corpus_eval(args)
    if args.dataset_json is None:
        raise ValueError("--dataset-json is required unless --pair-corpus-root is provided")

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


def run_pair_corpus_eval(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device(args.device)
    corpus_root = args.pair_corpus_root
    if corpus_root is None:
        raise ValueError("--pair-corpus-root is required for fixed corpus eval")
    if not corpus_root.is_absolute():
        corpus_root = ROOT / corpus_root
    split = str(args.pair_corpus_split or args.split)
    manifest_path = corpus_root / "pairs" / "manifests" / f"{split}.jsonl"
    pair_index_path = corpus_root / "pairs" / "index" / f"{split}_pairs.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"PairCorpus manifest not found: {manifest_path}")
    if not pair_index_path.exists():
        raise FileNotFoundError(f"PairCorpus pair index not found: {pair_index_path}")

    manifest_rows = load_jsonl(manifest_path)
    manifest_by_sample_id = {str(row["sample_id"]): row for row in manifest_rows}
    pair_rows = [
        row
        for row in load_jsonl(pair_index_path)
        if bool(row.get("is_valid_pair_for_rank", True))
        and str(row.get("clean_sample_id", "")) in manifest_by_sample_id
        and str(row.get("corrupted_sample_id", "")) in manifest_by_sample_id
    ]
    if args.limit is not None:
        pair_rows = pair_rows[: max(0, int(args.limit))]
    if not pair_rows:
        raise ValueError(f"No valid pairs found in {pair_index_path}")

    first_clean = manifest_by_sample_id[str(pair_rows[0]["clean_sample_id"])]
    sample_song_obj = load_encoded_song(resolve_manifest_path(first_clean["encoded_song_path"]))
    model = load_model_from_config_and_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        sample_song_obj=sample_song_obj,
        device=device,
    )
    requested_modes = set(resolve_modes(args)) if args.modes else None

    rows: list[dict[str, Any]] = []
    for row_index, pair in enumerate(pair_rows):
        clean_row = manifest_by_sample_id[str(pair["clean_sample_id"])]
        corrupted_row = manifest_by_sample_id[str(pair["corrupted_sample_id"])]
        mode = str(pair.get("corruption_name") or corrupted_row.get("corruption_name", "identity"))
        if requested_modes is not None and mode not in requested_modes:
            continue

        clean_song_obj = load_encoded_song(resolve_manifest_path(clean_row["encoded_song_path"]))
        corrupted_song_obj = load_encoded_song(resolve_manifest_path(corrupted_row["encoded_song_path"]))
        score_real = score_song(model, clean_song_obj, device)
        score_corrupted = score_song(model, corrupted_song_obj, device)
        score_gap = score_real - score_corrupted

        rows.append(
            {
                "song_id": pair.get("source_song_id") or clean_row.get("source_song_id") or pair.get("pair_group_id"),
                "split": split,
                "mode": mode,
                "applied": True,
                "topology_changed": bool(pair.get("topology_changed", corrupted_row.get("topology_changed", False))),
                "note_corrupted_indices": corrupted_row.get("note_corrupted_indices", []),
                "chord_corrupted_indices": corrupted_row.get("chord_corrupted_indices", []),
                "onset_corrupted_indices": corrupted_row.get("onset_corrupted_indices", []),
                "score_real": score_real,
                "score_corrupted": score_corrupted,
                "score_gap": score_gap,
                "rank_success": int(score_real > score_corrupted),
                "metadata_json": json.dumps(
                    {
                        "pair_group_id": pair.get("pair_group_id"),
                        "clean_sample_id": pair.get("clean_sample_id"),
                        "corrupted_sample_id": pair.get("corrupted_sample_id"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "mode_family": corrupted_row.get("corruption_group"),
                "corruption_name": mode,
                "corruption_params_json": json.dumps(corrupted_row.get("corruption_params", {}), ensure_ascii=False, sort_keys=True),
                "reason_skipped": None,
                "n_notes_modified": len(corrupted_row.get("note_corrupted_indices", []) or []),
                "n_chords_modified": len(corrupted_row.get("chord_corrupted_indices", []) or []),
                "mode_group": corrupted_row.get("corruption_group", "other"),
                "pair_group_id": pair.get("pair_group_id"),
                "source_row_index": row_index,
            }
        )

    rows_df = build_rows_dataframe(rows)
    summary_df = build_summary_dataframe(rows_df)
    return rows_df, summary_df


def main() -> None:
    args = parse_args()
    rows_df, summary_df = run_eval(args)
    rank_probe_rows_df = None
    rank_probe_summary_df = None
    if int(args.rank_probe_size) > 0:
        rank_probe_rows_df, rank_probe_summary_df = build_rank_probe_dataframes(rows_df, int(args.rank_probe_size))
    write_outputs(rows_df, summary_df, args.outdir, rank_probe_rows_df, rank_probe_summary_df)


if __name__ == "__main__":
    main()
