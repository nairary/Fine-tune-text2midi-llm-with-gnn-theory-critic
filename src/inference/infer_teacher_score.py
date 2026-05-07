#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from src.dataloader.song_corruptions import corrupt_song_obj
from src.dataloader.theory_helpers import build_theory_context
from src.dataloader.utils_graph import build_graph_from_encoded
from src.models.teacher_gnn import TeacherGNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one encoded song object with a trained TeacherGNN and optional song-level corruptions."
    )
    parser.add_argument("--encoded-json", type=Path, required=True, help="Path to a single encoded song JSON object.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a model checkpoint (.pt).")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the matching composed_config.yaml used to train the checkpoint.",
    )
    parser.add_argument(
        "--backend",
        choices=["none", "song_theory"],
        default="none",
        help="'none' scores the input as-is. 'song_theory' applies the requested corruption modes in order.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=[],
        help="Ordered list of corruption modes to apply sequentially during inference.",
    )
    parser.add_argument("--device", default="cpu", help="Device for inference, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed used for corruption application.")
    parser.add_argument(
        "--save-corrupted-json",
        type=Path,
        default=None,
        help="Optional path to save the corrupted encoded song JSON.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON result.")
    return parser.parse_args()


def load_song_obj(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} does not contain a single song object. Expected a JSON object with keys like "
            "'meta', 'melody', and 'chords'."
        )

    required_keys = {"meta", "melody", "chords"}
    if not required_keys.issubset(payload.keys()):
        raise ValueError(
            f"{path} does not look like a single encoded song object. Missing any of: {sorted(required_keys)}."
        )
    return payload


def build_model_from_config(
    cfg,
    sample_song_obj: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> TeacherGNN:
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
        score_fusion_mode=str(cfg.model.get("score_fusion_mode", "none")),
        score_fusion_hidden_dim=cfg.model.get("score_fusion_hidden_dim"),
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
def score_song(
    model: TeacherGNN,
    song_obj: dict[str, Any],
    device: torch.device,
    *,
    include_intermediates: bool = False,
) -> dict[str, Any]:
    graph = build_graph_from_encoded(song_obj)
    batch = Batch.from_data_list([graph]).to(device)
    outputs = model(batch)

    result: dict[str, Any] = {
        "graph_score": float(outputs["graph_score"].view(-1)[0].item()),
    }

    summaries = outputs.get("local_score_summaries")
    if summaries is not None and summaries.numel() > 0:
        result["local_score_summaries"] = summaries[0].detach().cpu().tolist()

    if include_intermediates:
        graph_embedding = outputs.get("graph_embedding")
        if graph_embedding is not None and graph_embedding.numel() > 0:
            result["graph_embedding"] = graph_embedding[0].detach().cpu().tolist()

        pooled_by_type = outputs.get("pooled_by_type") or {}
        if pooled_by_type:
            result["pooled_by_type"] = {
                str(node_type): pooled[0].detach().cpu().tolist()
                for node_type, pooled in pooled_by_type.items()
                if pooled is not None and pooled.numel() > 0
            }

    return result


def apply_song_theory_corruptions(
    song_obj: dict[str, Any],
    modes: list[str],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    theory_ctx = build_theory_context()
    working = copy.deepcopy(song_obj)
    metadata_list: list[dict[str, Any]] = []

    for offset, mode in enumerate(modes):
        rng = random.Random(seed + offset)
        working, metadata = corrupt_song_obj(
            working,
            corruption_modes=[mode],
            corruption_cfg={},
            theory_ctx=theory_ctx,
            rng=rng,
        )
        metadata_list.append(metadata)

    return working, metadata_list


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    song_obj = load_song_obj(args.encoded_json)
    cfg = OmegaConf.load(args.config)
    model = build_model_from_config(cfg, song_obj, args.checkpoint, device)

    original = score_song(model, song_obj, device)

    result: dict[str, Any] = {
        "input_json": str(args.encoded_json),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "backend": args.backend,
        "modes": list(args.modes),
        "original_score": original["graph_score"],
        "original_local_score_summaries": original.get("local_score_summaries"),
    }

    if args.backend == "none":
        result["corrupted_score"] = None
        result["score_gap"] = None
        result["applied_corruptions"] = []
    elif args.backend == "song_theory":
        corrupted_song_obj, metadata_list = apply_song_theory_corruptions(song_obj, args.modes, args.seed)
        corrupted = score_song(model, corrupted_song_obj, device)

        result["corrupted_score"] = corrupted["graph_score"]
        result["corrupted_local_score_summaries"] = corrupted.get("local_score_summaries")
        result["score_gap"] = result["original_score"] - result["corrupted_score"]
        result["applied_corruptions"] = metadata_list

        if args.save_corrupted_json is not None:
            args.save_corrupted_json.parent.mkdir(parents=True, exist_ok=True)
            with args.save_corrupted_json.open("w", encoding="utf-8") as handle:
                json.dump(corrupted_song_obj, handle, ensure_ascii=False, indent=2)
            result["saved_corrupted_json"] = str(args.save_corrupted_json)
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
