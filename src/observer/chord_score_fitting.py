from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataloader.theory_helpers import build_theory_context, select_active_mode_name
from src.observer.chord_parser import (
    _candidate_sort_key,
    build_sounding_sonority,
    explain_score_candidate,
    generate_all_candidates,
    select_target_instrument,
)

LOGGER = logging.getLogger(__name__)

POSITIVE_FEATURES = [
    "body_match_count",
    "extras_explained_count",
    "bass_matches_body",
    "mode_equals_main",
]
NEGATIVE_FEATURES = [
    "unexplained_pcs_count",
    "missing_core_pcs_count",
    "borrowed_mode_penalty",
    "mode_distance_penalty",
    "add_penalty",
    "suspension_penalty",
    "alteration_penalty",
    "omit_penalty",
    "body_size_penalty",
]
ALL_FEATURES = POSITIVE_FEATURES + NEGATIVE_FEATURES


def beat_to_seconds(beat: float, bpm: float) -> float:
    seconds_per_beat = 60.0 / max(float(bpm), 1e-6)
    return max(0.0, (float(beat) - 1.0) * seconds_per_beat)


def _decode_set_vec(vec: list[int] | None, allowed_values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for idx, bit in enumerate(vec or []):
        if bit and idx < len(allowed_values):
            out.append(allowed_values[idx])
    return out


def decode_ground_truth_chord(song_obj: dict[str, Any], chord: dict[str, Any], theory_ctx: dict[str, Any]) -> dict[str, Any] | None:
    root_raw = theory_ctx["root_id_to_raw"].get(int(chord.get("root_id", 0)))
    type_raw = theory_ctx["type_id_to_raw"].get(int(chord.get("type_id", 0)))
    inversion_raw = theory_ctx["inversion_id_to_raw"].get(int(chord.get("inversion_id", 0)))
    if root_raw is None or type_raw is None:
        return None

    mode_name = select_active_mode_name(song_obj, chord, theory_ctx)
    add_degrees = sorted(int(v) for v in _decode_set_vec(chord.get("adds_vec"), theory_ctx["chord_add_allowed_values"]))
    suspension_degrees = sorted(
        int(v) for v in _decode_set_vec(chord.get("suspensions_vec"), theory_ctx["chord_susp_allowed_values"])
    )
    omit_degrees = sorted(int(v) for v in _decode_set_vec(chord.get("omits_vec"), theory_ctx["chord_omit_allowed_values"]))
    alteration_tokens = sorted(str(v) for v in _decode_set_vec(chord.get("alterations_vec"), theory_ctx["chord_alter_allowed_values"]))

    return {
        "mode_name": mode_name,
        "root_degree_raw": int(root_raw),
        "type_raw": int(type_raw),
        "inversion_raw": None if inversion_raw is None else int(inversion_raw),
        "add_degrees": add_degrees,
        "suspension_degrees": suspension_degrees,
        "omit_degrees": omit_degrees,
        "alteration_tokens": alteration_tokens,
    }


def extract_candidate_feature_dict(
    candidate: Any,
    observed_pcs: list[int],
    bass_pc: int | None,
    main_mode: str,
    theory_ctx: dict[str, Any],
) -> dict[str, float]:
    score_terms = explain_score_candidate(candidate, observed_pcs, bass_pc, main_mode, theory_ctx)
    features: dict[str, float] = {}
    features.update({name: float(score_terms["positive_terms"].get(name, 0.0)) for name in POSITIVE_FEATURES})
    features.update({name: float(score_terms["negative_terms"].get(name, 0.0)) for name in NEGATIVE_FEATURES})
    return features


def compute_weighted_candidate_score(feature_dict: dict[str, float], weights: dict[str, Any]) -> float:
    bias = float(weights.get("bias", 0.0))
    positive_weights = weights.get("positive", {})
    negative_weights = weights.get("negative", {})

    score = bias
    for name in POSITIVE_FEATURES:
        score += float(positive_weights.get(name, 0.0)) * float(feature_dict.get(name, 0.0))
    for name in NEGATIVE_FEATURES:
        score -= float(negative_weights.get(name, 0.0)) * float(feature_dict.get(name, 0.0))
    return float(score)


def match_candidate_to_ground_truth(candidate: Any, gt_chord: dict[str, Any], theory_ctx: dict[str, Any] | None = None) -> bool:
    _ = theory_ctx
    return (
        candidate.mode_name == gt_chord["mode_name"]
        and int(candidate.root_degree_raw) == int(gt_chord["root_degree_raw"])
        and int(candidate.type_raw) == int(gt_chord["type_raw"])
        and candidate.inversion_raw == gt_chord["inversion_raw"]
        and sorted(int(v) for v in candidate.add_degrees) == sorted(int(v) for v in gt_chord["add_degrees"])
        and sorted(int(v) for v in candidate.suspension_degrees) == sorted(int(v) for v in gt_chord["suspension_degrees"])
        and sorted(int(v) for v in candidate.omit_degrees) == sorted(int(v) for v in gt_chord["omit_degrees"])
        and sorted(str(v) for v in candidate.alteration_tokens) == sorted(str(v) for v in gt_chord["alteration_tokens"])
    )


def _serialize_candidate(candidate: Any) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["score"] = float(payload.get("score", 0.0))
    return payload


def build_training_groups(
    encoded_data: dict[str, Any],
    midi_root: str | Path,
    split: str,
    instrument_name: str = "chords",
    limit: int | None = None,
    theory_ctx: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    song_subset = list(iter_split_song_items(encoded_data=encoded_data, split=split, limit=limit))
    return build_groups_for_song_subset(
        song_subset=song_subset,
        midi_root=midi_root,
        instrument_name=instrument_name,
        theory_ctx=theory_ctx,
        include_candidate_metadata=True,
        drop_groups_without_positives=False,
    )


def iter_split_song_items(
    encoded_data: dict[str, Any],
    split: str,
    limit: int | None = None,
):
    count = 0
    for song_id, song_obj in encoded_data.items():
        if song_obj.get("meta", {}).get("split") != split:
            continue
        if limit is not None and count >= int(limit):
            break
        count += 1
        yield song_id, song_obj


def iter_chunked_song_items(song_items: Any, chunk_size: int = 16):
    chunk_n = max(1, int(chunk_size))
    iterator = iter(song_items)
    while True:
        chunk = list(islice(iterator, chunk_n))
        if not chunk:
            break
        yield chunk


def build_groups_for_song_subset(
    song_subset: list[tuple[str, dict[str, Any]]],
    midi_root: str | Path,
    instrument_name: str = "chords",
    theory_ctx: dict[str, Any] | None = None,
    include_candidate_metadata: bool = True,
    drop_groups_without_positives: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import pretty_midi

    theory_ctx = theory_ctx or build_theory_context()
    midi_root = Path(midi_root)

    groups: list[dict[str, Any]] = []
    stats = {
        "songs_total": 0,
        "songs_missing_midi": 0,
        "songs_bad_midi": 0,
        "songs_no_instrument": 0,
        "events_total": 0,
        "events_skipped_rest": 0,
        "events_skipped_bad_gt": 0,
        "events_skipped_sparse_sonority": 0,
        "events_skipped_no_candidates": 0,
        "events_positive_missing": 0,
        "groups_kept": 0,
    }

    for song_id, song_obj in song_subset:
        split = str(song_obj.get("meta", {}).get("split") or "")
        stats["songs_total"] += 1
        midi_path = midi_root / split / f"{song_id}.mid"
        if not midi_path.exists():
            stats["songs_missing_midi"] += 1
            LOGGER.warning("[%s] missing MIDI: %s", song_id, midi_path)
            continue

        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as exc:  # noqa: BLE001
            stats["songs_bad_midi"] += 1
            LOGGER.warning("[%s] failed to open MIDI (%s): %s", song_id, midi_path, exc)
            continue

        try:
            instrument = select_target_instrument(pm, instrument_name=instrument_name)
        except Exception as exc:  # noqa: BLE001
            stats["songs_no_instrument"] += 1
            LOGGER.warning("[%s] target instrument missing: %s", song_id, exc)
            continue

        main_mode = select_active_mode_name(song_obj, None, theory_ctx)
        tonic_pc = int(song_obj.get("meta", {}).get("main_key_tonic_pc", 0) or 0) % 12
        bpm = float(song_obj.get("meta", {}).get("main_bpm", 120.0) or 120.0)

        for chord_event in song_obj.get("chords", []):
            stats["events_total"] += 1
            if int(chord_event.get("is_rest", 0) or 0) == 1:
                stats["events_skipped_rest"] += 1
                continue

            gt_chord = decode_ground_truth_chord(song_obj, chord_event, theory_ctx)
            if gt_chord is None:
                stats["events_skipped_bad_gt"] += 1
                continue

            onset_sec = beat_to_seconds(float(chord_event.get("beat", 1.0) or 1.0), bpm)
            sonority = build_sounding_sonority(instrument, onset_sec)
            observed_pcs_abs = sonority["observed_pcs"]
            if len(observed_pcs_abs) < 3:
                stats["events_skipped_sparse_sonority"] += 1
                continue

            observed_pcs = sorted({(int(pc) - tonic_pc) % 12 for pc in observed_pcs_abs})
            bass_pc_abs = sonority["bass_pc"]
            bass_pc = None if bass_pc_abs is None else (int(bass_pc_abs) - tonic_pc) % 12

            candidates = generate_all_candidates(observed_pcs, bass_pc, main_mode, theory_ctx)
            if not candidates:
                stats["events_skipped_no_candidates"] += 1
                continue
            candidates = sorted(candidates, key=lambda c: _candidate_sort_key(c, main_mode, theory_ctx))

            positive_mask = [match_candidate_to_ground_truth(candidate, gt_chord, theory_ctx) for candidate in candidates]
            has_positive = any(positive_mask)
            if not has_positive:
                stats["events_positive_missing"] += 1
                if drop_groups_without_positives:
                    continue

            features = [extract_candidate_feature_dict(c, observed_pcs, bass_pc, main_mode, theory_ctx) for c in candidates]
            payload = {
                "features": features,
                "positive_mask": positive_mask,
            }
            if include_candidate_metadata:
                payload.update(
                    {
                        "song_id": song_id,
                        "split": split,
                        "onset_time": float(onset_sec),
                        "candidates": [_serialize_candidate(c) for c in candidates],
                        "ground_truth": gt_chord,
                        "observed_pcs": observed_pcs,
                        "bass_pc": bass_pc,
                        "main_mode": main_mode,
                    }
                )
            groups.append(payload)
            stats["groups_kept"] += 1

    return groups, stats


def iter_training_group_chunks(
    encoded_data: dict[str, Any],
    midi_root: str | Path,
    split: str,
    instrument_name: str = "chords",
    limit: int | None = None,
    chunk_size: int = 16,
    theory_ctx: dict[str, Any] | None = None,
    include_candidate_metadata: bool = False,
    drop_groups_without_positives: bool = False,
):
    song_items = iter_split_song_items(encoded_data=encoded_data, split=split, limit=limit)
    for song_chunk in iter_chunked_song_items(song_items, chunk_size=chunk_size):
        yield build_groups_for_song_subset(
            song_subset=song_chunk,
            midi_root=midi_root,
            instrument_name=instrument_name,
            theory_ctx=theory_ctx,
            include_candidate_metadata=include_candidate_metadata,
            drop_groups_without_positives=drop_groups_without_positives,
        )


class LearnableChordScore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.theta_positive = nn.ParameterDict({name: nn.Parameter(torch.tensor(0.0, dtype=torch.float32)) for name in POSITIVE_FEATURES})
        self.theta_negative = nn.ParameterDict({name: nn.Parameter(torch.tensor(0.0, dtype=torch.float32)) for name in NEGATIVE_FEATURES})

    def positive_weights(self) -> dict[str, torch.Tensor]:
        return {name: F.softplus(param) for name, param in self.theta_positive.items()}

    def negative_weights(self) -> dict[str, torch.Tensor]:
        return {name: F.softplus(param) for name, param in self.theta_negative.items()}

    def score_feature_dict(self, feature_dict: dict[str, float], device: str | torch.device) -> torch.Tensor:
        positive_weights = self.positive_weights()
        negative_weights = self.negative_weights()
        score = self.bias.to(device=device)
        for name in POSITIVE_FEATURES:
            score = score + positive_weights[name] * float(feature_dict.get(name, 0.0))
        for name in NEGATIVE_FEATURES:
            score = score - negative_weights[name] * float(feature_dict.get(name, 0.0))
        return score

    def score_group(self, feature_dicts: list[dict[str, float]], device: str | torch.device) -> torch.Tensor:
        return torch.stack([self.score_feature_dict(feature_dict, device=device) for feature_dict in feature_dicts], dim=0)

    def export_weights(self) -> dict[str, Any]:
        return {
            "bias": float(self.bias.detach().cpu().item()),
            "positive": {name: float(F.softplus(param).detach().cpu().item()) for name, param in self.theta_positive.items()},
            "negative": {name: float(F.softplus(param).detach().cpu().item()) for name, param in self.theta_negative.items()},
        }


def multi_positive_softmax_loss(scores: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 1 or positive_mask.ndim != 1 or scores.shape[0] != positive_mask.shape[0]:
        raise ValueError("scores and positive_mask must be 1D tensors with same shape")
    if int(positive_mask.sum().item()) == 0:
        return torch.tensor(0.0, dtype=scores.dtype, device=scores.device)

    log_denom = torch.logsumexp(scores, dim=0)
    log_num = torch.logsumexp(scores[positive_mask], dim=0)
    return log_denom - log_num


def evaluate_groups(
    model: LearnableChordScore,
    groups: list[dict[str, Any]],
    device: str | torch.device = "cpu",
    topk: int = 5,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    valid_groups = 0
    top1_hits = 0
    topk_hits = 0
    root_hits = 0
    type_hits = 0
    coverage_hits = 0
    candidate_meta_groups = 0

    with torch.no_grad():
        for group in groups:
            scores = model.score_group(group["features"], device=device)
            positive_mask = torch.tensor(group["positive_mask"], dtype=torch.bool, device=device)
            if int(positive_mask.sum().item()) == 0:
                continue

            coverage_hits += 1
            valid_groups += 1
            losses.append(float(multi_positive_softmax_loss(scores, positive_mask).item()))

            top_idx = int(torch.argmax(scores).item())
            top1_hits += int(bool(positive_mask[top_idx].item()))

            k = min(int(topk), int(scores.shape[0]))
            topk_idx = torch.topk(scores, k=k).indices
            topk_hits += int(bool(torch.any(positive_mask[topk_idx]).item()))

            if "candidates" in group:
                candidate_meta_groups += 1
                gt_ref = next(
                    candidate
                    for candidate, is_positive in zip(group["candidates"], group["positive_mask"], strict=False)
                    if is_positive
                )
                pred_ref = group["candidates"][top_idx]
                root_hits += int(int(pred_ref["root_degree_raw"]) == int(gt_ref["root_degree_raw"]))
                type_hits += int(int(pred_ref["type_raw"]) == int(gt_ref["type_raw"]))

    group_count = len(groups)
    return {
        "loss": float(sum(losses) / len(losses)) if losses else math.nan,
        "top1_exact_acc": float(top1_hits / valid_groups) if valid_groups else 0.0,
        "topk_contains_gt_acc": float(topk_hits / valid_groups) if valid_groups else 0.0,
        "root_acc": float(root_hits / candidate_meta_groups) if candidate_meta_groups else math.nan,
        "type_acc": float(type_hits / candidate_meta_groups) if candidate_meta_groups else math.nan,
        "group_count": float(group_count),
        "positive_coverage": float(coverage_hits / group_count) if group_count else 0.0,
        "valid_group_count": float(valid_groups),
    }


def evaluate_group_chunks(
    model: LearnableChordScore,
    group_chunks: Any,
    device: str | torch.device = "cpu",
    topk: int = 5,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    valid_groups = 0
    top1_hits = 0
    topk_hits = 0
    root_hits = 0
    type_hits = 0
    coverage_hits = 0
    group_count = 0
    candidate_meta_groups = 0

    with torch.no_grad():
        for chunk_groups, _ in group_chunks:
            for group in chunk_groups:
                group_count += 1
                scores = model.score_group(group["features"], device=device)
                positive_mask = torch.tensor(group["positive_mask"], dtype=torch.bool, device=device)
                if int(positive_mask.sum().item()) == 0:
                    continue

                coverage_hits += 1
                valid_groups += 1
                losses.append(float(multi_positive_softmax_loss(scores, positive_mask).item()))

                top_idx = int(torch.argmax(scores).item())
                top1_hits += int(bool(positive_mask[top_idx].item()))

                k = min(int(topk), int(scores.shape[0]))
                topk_idx = torch.topk(scores, k=k).indices
                topk_hits += int(bool(torch.any(positive_mask[topk_idx]).item()))

                if "candidates" in group:
                    candidate_meta_groups += 1
                    gt_ref = next(
                        candidate
                        for candidate, is_positive in zip(group["candidates"], group["positive_mask"], strict=False)
                        if is_positive
                    )
                    pred_ref = group["candidates"][top_idx]
                    root_hits += int(int(pred_ref["root_degree_raw"]) == int(gt_ref["root_degree_raw"]))
                    type_hits += int(int(pred_ref["type_raw"]) == int(gt_ref["type_raw"]))

    return {
        "loss": float(sum(losses) / len(losses)) if losses else math.nan,
        "top1_exact_acc": float(top1_hits / valid_groups) if valid_groups else 0.0,
        "topk_contains_gt_acc": float(topk_hits / valid_groups) if valid_groups else 0.0,
        "root_acc": float(root_hits / candidate_meta_groups) if candidate_meta_groups else math.nan,
        "type_acc": float(type_hits / candidate_meta_groups) if candidate_meta_groups else math.nan,
        "group_count": float(group_count),
        "positive_coverage": float(coverage_hits / group_count) if group_count else 0.0,
        "valid_group_count": float(valid_groups),
    }


def train_learnable_chord_score(
    train_groups: list[dict[str, Any]] | None = None,
    val_groups: list[dict[str, Any]] | None = None,
    train_group_chunks: Any = None,
    val_group_chunks: Any = None,
    epochs: int = 200,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    seed: int = 123,
    device: str = "cpu",
    topk: int = 5,
    log_every: int = 10,
    eval_every: int = 1,
) -> tuple[LearnableChordScore, dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(int(seed))
    model = LearnableChordScore().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = None
    best_val_top1 = -1.0
    best_row: dict[str, Any] | None = None
    metrics_log: list[dict[str, Any]] = []
    log_step = max(1, int(log_every))
    eval_step = max(1, int(eval_every))

    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_chunk_losses: list[float] = []
        chunk_source = train_group_chunks() if callable(train_group_chunks) else [((train_groups or []), {"songs_total": 0})]
        for chunk_idx, (chunk_groups, chunk_stats) in enumerate(chunk_source, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss_terms: list[torch.Tensor] = []
            for group in chunk_groups:
                scores = model.score_group(group["features"], device=device)
                positive_mask = torch.tensor(group["positive_mask"], dtype=torch.bool, device=device)
                if int(positive_mask.sum().item()) == 0:
                    continue
                loss_terms.append(multi_positive_softmax_loss(scores, positive_mask))
            if loss_terms:
                chunk_loss_tensor = torch.stack(loss_terms).mean()
                chunk_loss_tensor.backward()
                optimizer.step()
                chunk_loss = float(chunk_loss_tensor.detach().cpu().item())
                epoch_chunk_losses.append(chunk_loss)
            else:
                chunk_loss = math.nan
            LOGGER.info(
                "epoch=%d/%d train_chunk=%d chunk_songs=%d chunk_groups=%d chunk_loss=%.4f",
                epoch,
                int(epochs),
                chunk_idx,
                int(chunk_stats.get("songs_total", 0)),
                len(chunk_groups),
                float(chunk_loss),
            )
            del chunk_groups

        train_loss = float(sum(epoch_chunk_losses) / len(epoch_chunk_losses)) if epoch_chunk_losses else math.nan
        should_eval = epoch == 1 or epoch == int(epochs) or (epoch % eval_step == 0)
        if should_eval:
            train_metrics = (
                evaluate_group_chunks(model, train_group_chunks(), device=device, topk=topk)
                if callable(train_group_chunks)
                else evaluate_groups(model, train_groups or [], device=device, topk=topk)
            )
            val_metrics = (
                evaluate_group_chunks(model, val_group_chunks(), device=device, topk=topk)
                if callable(val_group_chunks)
                else evaluate_groups(model, val_groups or [], device=device, topk=topk)
            )
            train_metrics["loss"] = train_loss if not math.isnan(train_loss) else train_metrics["loss"]
        else:
            train_metrics = {
                "loss": train_loss,
                "top1_exact_acc": math.nan,
                "topk_contains_gt_acc": math.nan,
                "root_acc": math.nan,
                "type_acc": math.nan,
                "group_count": 0.0,
                "positive_coverage": math.nan,
            }
            val_metrics = {
                "loss": math.nan,
                "top1_exact_acc": math.nan,
                "topk_contains_gt_acc": math.nan,
                "root_acc": math.nan,
                "type_acc": math.nan,
                "group_count": 0.0,
                "positive_coverage": math.nan,
            }

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_top1_exact_acc": train_metrics["top1_exact_acc"],
            "val_top1_exact_acc": val_metrics["top1_exact_acc"],
            "train_topk_contains_gt_acc": train_metrics["topk_contains_gt_acc"],
            "val_topk_contains_gt_acc": val_metrics["topk_contains_gt_acc"],
            "train_root_acc": train_metrics["root_acc"],
            "val_root_acc": val_metrics["root_acc"],
            "train_type_acc": train_metrics["type_acc"],
            "val_type_acc": val_metrics["type_acc"],
            "train_group_count": int(train_metrics["group_count"]),
            "val_group_count": int(val_metrics["group_count"]),
            "train_positive_coverage": train_metrics["positive_coverage"],
            "val_positive_coverage": val_metrics["positive_coverage"],
        }
        metrics_log.append(row)

        should_log_epoch = epoch == 1 or epoch == int(epochs) or (epoch % log_step == 0)
        if should_log_epoch:
            LOGGER.info(
                (
                    "epoch=%d/%d train_loss=%.4f val_loss=%.4f "
                    "train_top1=%.4f val_top1=%.4f train_top%d=%.4f val_top%d=%.4f "
                    "train_root=%.4f val_root=%.4f train_type=%.4f val_type=%.4f"
                ),
                epoch,
                int(epochs),
                float(row["train_loss"]),
                float(row["val_loss"]),
                float(row["train_top1_exact_acc"]),
                float(row["val_top1_exact_acc"]),
                int(topk),
                float(row["train_topk_contains_gt_acc"]),
                int(topk),
                float(row["val_topk_contains_gt_acc"]),
                float(row["train_root_acc"]),
                float(row["val_root_acc"]),
                float(row["train_type_acc"]),
                float(row["val_type_acc"]),
            )

        if should_eval and val_metrics["top1_exact_acc"] >= best_val_top1:
            best_val_top1 = val_metrics["top1_exact_acc"]
            best_state = {
                "epoch": epoch,
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "best_val_top1_exact_acc": best_val_top1,
            }
            best_row = dict(row)
            LOGGER.info(
                "new_best epoch=%d val_top1=%.4f val_loss=%.4f",
                epoch,
                float(row["val_top1_exact_acc"]),
                float(row["val_loss"]),
            )

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    # Summary is intentionally tied to the best checkpoint (not the last epoch)
    # so metrics and learned weights describe the same selected model.
    summary_base = best_row if best_row is not None else (metrics_log[-1] if metrics_log else {})
    summary = {
        **summary_base,
        "learned_weights": model.export_weights(),
    }
    return model, summary, metrics_log


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
