from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

import torch
import torch.nn.functional as F

from src.models.teacher_heads import RECONSTRUCTION_SPECS


def _zero_like_reference(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _mean_or_zero(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return _zero_like_reference(reference)
    return values.mean()


def _find_reference_tensor(*candidates) -> torch.Tensor:
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, torch.Tensor):
            return candidate
        if isinstance(candidate, Mapping):
            nested = _find_reference_tensor(*candidate.values())
            if nested is not None:
                return nested
    raise ValueError("Could not infer a reference tensor for zero-valued loss construction.")


def _filter_and_encode_targets(
    selected_logits: torch.Tensor,
    target_values: torch.Tensor,
    valid_ids: Iterable[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_ids = [int(value) for value in valid_ids if int(value) >= 0]
    if not valid_ids:
        return selected_logits[:0], torch.empty((0,), dtype=torch.long, device=selected_logits.device)

    target_long = target_values.view(-1).to(device=selected_logits.device, dtype=torch.long)
    max_valid_id = max(int(value) for value in valid_ids)
    if max_valid_id < 0:
        return selected_logits[:0], torch.empty((0,), dtype=torch.long, device=selected_logits.device)

    lookup = torch.full((max_valid_id + 1,), -1, dtype=torch.long, device=selected_logits.device)
    lookup_ids = torch.tensor(valid_ids, dtype=torch.long, device=selected_logits.device)
    lookup[lookup_ids] = torch.arange(len(valid_ids), dtype=torch.long, device=selected_logits.device)

    in_range = (target_long >= 0) & (target_long <= max_valid_id)
    encoded_all = torch.full_like(target_long, -1)
    encoded_all[in_range] = lookup[target_long[in_range]]
    valid_mask = encoded_all >= 0
    if not bool(valid_mask.any()):
        return selected_logits[:0], torch.empty((0,), dtype=torch.long, device=selected_logits.device)
    filtered_logits = selected_logits[valid_mask]
    encoded_targets = encoded_all[valid_mask]
    return filtered_logits, encoded_targets


def _batched_indices(masked_batch, masked_labels: List[dict], node_type: str, field_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    ptr = masked_batch[node_type].ptr
    global_indices = []
    target_values = []
    for graph_index, per_graph_labels in enumerate(masked_labels):
        node_labels = per_graph_labels.get(node_type, {})
        field_names = node_labels.get("field_names", [])
        if field_name not in field_names:
            continue
        local_indices = node_labels.get("indices")
        if local_indices is None or local_indices.numel() == 0:
            continue
        offset = int(ptr[graph_index].item())
        global_indices.append(local_indices + offset)
        target_values.append(node_labels["target_values"][field_name].view(-1))

    if not global_indices:
        empty_indices = torch.empty((0,), dtype=torch.long, device=masked_batch[node_type].x.device)
        empty_targets = torch.empty((0,), dtype=torch.float, device=masked_batch[node_type].x.device)
        return empty_indices, empty_targets

    return torch.cat(global_indices).to(masked_batch[node_type].x.device), torch.cat(target_values).to(masked_batch[node_type].x.device)


def compute_reconstruction_losses(
    masked_outputs: Mapping[str, Dict[str, torch.Tensor]],
    masked_batch,
    masked_labels: List[dict],
    recon_weights: Mapping[str, float] | None = None,
    enabled_heads: Mapping[str, bool] | None = None,
):
    recon_logits = masked_outputs["recon_logits"]
    losses = {}
    metrics = {}
    total_recon_loss = None
    recon_weights = recon_weights or {}
    enabled_heads = enabled_heads or {}

    for head_name, spec in RECONSTRUCTION_SPECS.items():
        if not enabled_heads.get(head_name, True):
            continue
        if head_name not in recon_logits:
            continue

        logits = recon_logits[head_name]
        node_type = spec["node_type"]
        field_name = spec["field_name"]
        valid_ids = spec["valid_ids"]
        loss_weight = float(recon_weights.get(head_name, spec["default_loss_weight"]))
        global_indices, target_values = _batched_indices(masked_batch, masked_labels, node_type, field_name)

        metric_prefix = head_name.replace("note_", "note_").replace("chord_", "chord_")
        acc_key = f"{metric_prefix}_acc"
        loss_key = f"{head_name}_loss"

        if global_indices.numel() == 0:
            loss_value = _zero_like_reference(logits)
            accuracy = logits.new_tensor(0.0)
            count = logits.new_tensor(0.0)
        else:
            selected_logits = logits[global_indices]
            selected_logits, encoded_targets = _filter_and_encode_targets(selected_logits, target_values, valid_ids)
            if encoded_targets.numel() == 0:
                loss_value = _zero_like_reference(logits)
                accuracy = logits.new_tensor(0.0)
                count = logits.new_tensor(0.0)
            else:
                loss_value = F.cross_entropy(selected_logits, encoded_targets)
                predictions = selected_logits.argmax(dim=-1)
                accuracy = (predictions == encoded_targets).float().mean()
                count = encoded_targets.new_tensor(float(encoded_targets.numel()), dtype=torch.float)

        weighted_loss = loss_weight * loss_value
        losses[loss_key] = loss_value
        metrics[acc_key] = accuracy.detach()
        metrics[f"{head_name}_count"] = count.detach()
        total_recon_loss = weighted_loss if total_recon_loss is None else total_recon_loss + weighted_loss

    if total_recon_loss is None:
        if recon_logits:
            reference = next(iter(recon_logits.values()))
            total_recon_loss = _zero_like_reference(reference)
        else:
            raise ValueError("No reconstruction heads are enabled; cannot compute reconstruction loss.")

    losses["recon_loss"] = total_recon_loss
    return losses, metrics


def compute_ranking_loss(
    real_outputs,
    corrupted_outputs,
    intra_track_weight: float = 1.0,
    inter_track_weight: float = 1.0,
    binary_weight: float = 1.0,
    margin: float = 0.0,
):
    score_real = real_outputs["graph_score"].view(-1)
    score_corrupted = corrupted_outputs["graph_score"].view(-1)
    reference = score_real

    intra_margin = score_real - score_corrupted
    intra_rank_loss = F.softplus(-(intra_margin - float(margin))).mean()
    intra_rank_acc = (intra_margin > float(margin)).float().mean()
    intra_mean_margin = intra_margin.mean()

    global_margin = score_real[:, None] - score_corrupted[None, :]
    global_rank_acc = (global_margin > float(margin)).float().mean()
    global_mean_margin = global_margin.mean()

    if score_real.numel() > 1:
        off_diagonal_mask = ~torch.eye(score_real.numel(), dtype=torch.bool, device=score_real.device)
        inter_margin = global_margin[off_diagonal_mask]
    else:
        inter_margin = global_margin.new_empty((0,))
    inter_rank_loss = _mean_or_zero(F.softplus(-(inter_margin - float(margin))), reference)
    inter_rank_acc = _mean_or_zero((inter_margin > float(margin)).float(), reference).detach()
    inter_mean_margin = _mean_or_zero(inter_margin, reference).detach()

    clean_targets = torch.ones_like(score_real)
    corrupted_targets = torch.zeros_like(score_corrupted)
    clean_binary_loss = F.binary_cross_entropy_with_logits(score_real, clean_targets)
    corrupted_binary_loss = F.binary_cross_entropy_with_logits(score_corrupted, corrupted_targets)
    graph_binary_loss = 0.5 * (clean_binary_loss + corrupted_binary_loss)
    clean_binary_acc = (torch.sigmoid(score_real) >= 0.5).float().mean()
    corrupted_binary_acc = (torch.sigmoid(score_corrupted) < 0.5).float().mean()
    graph_binary_acc = 0.5 * (clean_binary_acc + corrupted_binary_acc)

    weighted_terms = []
    active_weights = []
    if float(intra_track_weight) > 0.0:
        weighted_terms.append(float(intra_track_weight) * intra_rank_loss)
        active_weights.append(float(intra_track_weight))
    if float(inter_track_weight) > 0.0 and inter_margin.numel() > 0:
        weighted_terms.append(float(inter_track_weight) * inter_rank_loss)
        active_weights.append(float(inter_track_weight))
    if float(binary_weight) > 0.0:
        weighted_terms.append(float(binary_weight) * graph_binary_loss)
        active_weights.append(float(binary_weight))

    if weighted_terms:
        rank_loss = torch.stack(weighted_terms).sum() / float(sum(active_weights))
    else:
        rank_loss = _zero_like_reference(reference)

    metrics = {
        "rank_loss": rank_loss,
        "intra_rank_loss": intra_rank_loss,
        "inter_rank_loss": inter_rank_loss,
        "graph_binary_loss": graph_binary_loss,
        "rank_acc": global_rank_acc.detach(),
        "intra_rank_acc": intra_rank_acc.detach(),
        "inter_rank_acc": inter_rank_acc,
        "graph_binary_acc": graph_binary_acc.detach(),
        "graph_binary_clean_acc": clean_binary_acc.detach(),
        "graph_binary_corrupted_acc": corrupted_binary_acc.detach(),
        "mean_margin": global_mean_margin.detach(),
        "intra_mean_margin": intra_mean_margin.detach(),
        "inter_mean_margin": inter_mean_margin,
        "score_real_mean": score_real.mean().detach(),
        "score_corrupted_mean": score_corrupted.mean().detach(),
        "score_real_min": score_real.min().detach(),
        "score_corrupted_max": score_corrupted.max().detach(),
        "score_gap_minmax": (score_real.min() - score_corrupted.max()).detach(),
    }

    score_base_real = real_outputs.get("graph_score_base")
    score_base_corrupted = corrupted_outputs.get("graph_score_base")
    if isinstance(score_base_real, torch.Tensor) and isinstance(score_base_corrupted, torch.Tensor):
        base_real = score_base_real.view(-1)
        base_corrupted = score_base_corrupted.view(-1)
        if base_real.shape == score_real.shape and base_corrupted.shape == score_corrupted.shape:
            base_margin = base_real[:, None] - base_corrupted[None, :]
            metrics.update(
                {
                    "score_base_real_mean": base_real.mean().detach(),
                    "score_base_corrupted_mean": base_corrupted.mean().detach(),
                    "score_base_mean_margin": base_margin.mean().detach(),
                    "score_base_rank_acc": (base_margin > float(margin)).float().mean().detach(),
                }
            )

    local_summary_real = real_outputs.get("local_score_summaries")
    local_summary_corrupted = corrupted_outputs.get("local_score_summaries")
    if isinstance(local_summary_real, torch.Tensor) and isinstance(local_summary_corrupted, torch.Tensor):
        if local_summary_real.dim() == 2 and local_summary_corrupted.dim() == 2 and local_summary_real.size(-1) > 0:
            summary_real = local_summary_real.mean(dim=-1)
            summary_corrupted = local_summary_corrupted.mean(dim=-1)
            if summary_real.shape == score_real.shape and summary_corrupted.shape == score_corrupted.shape:
                summary_margin = summary_real[:, None] - summary_corrupted[None, :]
                metrics.update(
                    {
                        "score_local_summary_real_mean": summary_real.mean().detach(),
                        "score_local_summary_corrupted_mean": summary_corrupted.mean().detach(),
                        "score_local_summary_mean_margin": summary_margin.mean().detach(),
                        "score_local_summary_rank_acc": (summary_margin > float(margin)).float().mean().detach(),
                    }
                )

    return metrics


def _sample_clean_indices(
    graph_node_count: int,
    corrupted_indices: List[int],
    negatives_per_positive: int,
) -> List[int]:
    corrupted_set = set(corrupted_indices)
    clean_pool = [idx for idx in range(graph_node_count) if idx not in corrupted_set]
    if not clean_pool:
        return []
    count = min(len(clean_pool), max(1, len(corrupted_indices) * negatives_per_positive))
    permutation = torch.randperm(len(clean_pool))[:count].tolist()
    return [clean_pool[pos] for pos in permutation]


def _node_graph_ranges(batch, node_type: str):
    ptr = batch[node_type].ptr
    return [(int(ptr[i].item()), int(ptr[i + 1].item())) for i in range(ptr.numel() - 1)]


def compute_local_corruption_losses(
    corrupted_outputs,
    corrupted_batch,
    corruption_metadata: List[dict] | None,
    enabled_levels: Mapping[str, bool] | None = None,
    negatives_per_positive: int = 2,
):
    enabled_levels = enabled_levels or {"note": True, "chord": True, "onset": True}
    local_scores = corrupted_outputs.get("local_scores", {})
    graph_ranges = {
        "note": _node_graph_ranges(corrupted_batch, "note"),
        "chord": _node_graph_ranges(corrupted_batch, "chord"),
        "onset": _node_graph_ranges(corrupted_batch, "onset"),
    }

    losses = {}
    metrics = {}
    for level in ("note", "chord", "onset"):
        if not enabled_levels.get(level, True):
            continue
        level_logits_all = local_scores.get(level)
        if level_logits_all is None:
            continue
        loss_key = f"{level}_local_loss"
        acc_key = f"{level}_local_acc"
        if corruption_metadata is None:
            losses[loss_key] = _zero_like_reference(level_logits_all)
            continue

        key = f"{level}_corrupted_indices"
        sampled_indices = []
        sampled_targets = []
        for graph_index, metadata in enumerate(corruption_metadata):
            metadata = metadata or {}
            local_corrupted = [int(idx) for idx in (metadata.get(key) or [])]
            if not local_corrupted:
                continue
            start, end = graph_ranges[level][graph_index]
            graph_node_count = max(0, end - start)
            if graph_node_count <= 0:
                continue
            valid_corrupted = sorted({idx for idx in local_corrupted if 0 <= idx < graph_node_count})
            if not valid_corrupted:
                continue
            sampled_clean = _sample_clean_indices(
                graph_node_count=graph_node_count,
                corrupted_indices=valid_corrupted,
                negatives_per_positive=negatives_per_positive,
            )
            positive_indices = torch.tensor(
                [start + local_idx for local_idx in valid_corrupted],
                dtype=torch.long,
                device=level_logits_all.device,
            )
            sampled_indices.append(positive_indices)
            sampled_targets.append(level_logits_all.new_ones((positive_indices.numel(),)))
            if sampled_clean:
                negative_indices = torch.tensor(
                    [start + local_idx for local_idx in sampled_clean],
                    dtype=torch.long,
                    device=level_logits_all.device,
                )
                sampled_indices.append(negative_indices)
                sampled_targets.append(level_logits_all.new_zeros((negative_indices.numel(),)))

        if not sampled_indices:
            losses[loss_key] = _zero_like_reference(level_logits_all)
            continue

        level_indices = torch.cat(sampled_indices, dim=0)
        level_logits = level_logits_all.index_select(0, level_indices)
        level_targets = torch.cat(sampled_targets, dim=0)
        loss_value = F.binary_cross_entropy_with_logits(level_logits, level_targets)
        predictions = (torch.sigmoid(level_logits) >= 0.5).float()
        metrics[acc_key] = (predictions == level_targets).float().mean().detach()
        losses[loss_key] = loss_value
    return losses, metrics


def compute_teacher_ssl_losses(
    masked_outputs=None,
    real_outputs=None,
    corrupted_outputs=None,
    masked_batch=None,
    corrupted_batch=None,
    masked_labels: List[dict] | None = None,
    corruption_metadata: List[dict] | None = None,
    lambda_recon: float = 1.0,
    lambda_graph_rank: float = 0.5,
    lambda_note_local: float = 0.5,
    lambda_chord_local: float = 0.5,
    lambda_onset_local: float = 0.5,
    graph_rank_intra_weight: float = 1.0,
    graph_rank_inter_weight: float = 1.0,
    graph_binary_weight: float = 1.0,
    graph_rank_margin: float = 0.0,
    enable_recon: bool = True,
    enable_graph_rank: bool = True,
    enable_note_local: bool = True,
    enable_chord_local: bool = True,
    enable_onset_local: bool = True,
    recon_weights: Mapping[str, float] | None = None,
    enabled_heads: Mapping[str, bool] | None = None,
    local_negatives_per_positive: int = 2,
):
    if not any((enable_recon, enable_graph_rank, enable_note_local, enable_chord_local, enable_onset_local)):
        raise ValueError("At least one teacher training objective must be enabled.")

    reference_tensor = _find_reference_tensor(masked_outputs, real_outputs, corrupted_outputs)

    if enable_recon:
        if masked_outputs is None or masked_batch is None or masked_labels is None:
            raise ValueError("Reconstruction stage requires masked outputs, masked batch, and masked labels.")
        recon_losses, recon_metrics = compute_reconstruction_losses(
            masked_outputs=masked_outputs,
            masked_batch=masked_batch,
            masked_labels=masked_labels,
            recon_weights=recon_weights,
            enabled_heads=enabled_heads,
        )
    else:
        zero_recon = _zero_like_reference(reference_tensor)
        recon_losses = {"recon_loss": zero_recon}
        recon_metrics = {}

    if enable_graph_rank:
        if real_outputs is None or corrupted_outputs is None:
            raise ValueError("Graph-ranking stage requires both clean and corrupted outputs.")
        rank_bundle = compute_ranking_loss(
            real_outputs=real_outputs,
            corrupted_outputs=corrupted_outputs,
            intra_track_weight=graph_rank_intra_weight,
            inter_track_weight=graph_rank_inter_weight,
            binary_weight=graph_binary_weight,
            margin=graph_rank_margin,
        )
    else:
        zero_rank = _zero_like_reference(reference_tensor)
        zero_metric = zero_rank.detach()
        rank_bundle = {
            "rank_loss": zero_rank,
            "intra_rank_loss": zero_rank,
            "inter_rank_loss": zero_rank,
            "graph_binary_loss": zero_rank,
            "rank_acc": zero_metric,
            "intra_rank_acc": zero_metric,
            "inter_rank_acc": zero_metric,
            "graph_binary_acc": zero_metric,
            "graph_binary_clean_acc": zero_metric,
            "graph_binary_corrupted_acc": zero_metric,
            "mean_margin": zero_metric,
            "intra_mean_margin": zero_metric,
            "inter_mean_margin": zero_metric,
            "score_real_mean": zero_metric,
            "score_corrupted_mean": zero_metric,
            "score_real_min": zero_metric,
            "score_corrupted_max": zero_metric,
            "score_gap_minmax": zero_metric,
        }

    if any((enable_note_local, enable_chord_local, enable_onset_local)):
        if corrupted_outputs is None or corrupted_batch is None:
            raise ValueError("Local corruption stage requires corrupted outputs and corrupted batch.")
        local_losses, local_metrics = compute_local_corruption_losses(
            corrupted_outputs=corrupted_outputs,
            corrupted_batch=corrupted_batch,
            corruption_metadata=corruption_metadata,
            enabled_levels={
                "note": enable_note_local,
                "chord": enable_chord_local,
                "onset": enable_onset_local,
            },
            negatives_per_positive=local_negatives_per_positive,
        )
    else:
        local_losses = {}
        local_metrics = {}

    total_loss = lambda_recon * recon_losses["recon_loss"] if enable_recon else _zero_like_reference(reference_tensor)
    if enable_graph_rank:
        total_loss = total_loss + lambda_graph_rank * rank_bundle["rank_loss"]
    if enable_note_local and "note_local_loss" in local_losses:
        total_loss = total_loss + lambda_note_local * local_losses["note_local_loss"]
    if enable_chord_local and "chord_local_loss" in local_losses:
        total_loss = total_loss + lambda_chord_local * local_losses["chord_local_loss"]
    if enable_onset_local and "onset_local_loss" in local_losses:
        total_loss = total_loss + lambda_onset_local * local_losses["onset_local_loss"]

    loss_dict = {
        "loss": total_loss,
        **recon_losses,
        **local_losses,
        "rank_loss": rank_bundle["rank_loss"] if enable_graph_rank else _zero_like_reference(rank_bundle["rank_loss"]),
        "intra_rank_loss": rank_bundle["intra_rank_loss"] if enable_graph_rank else _zero_like_reference(rank_bundle["intra_rank_loss"]),
        "inter_rank_loss": rank_bundle["inter_rank_loss"] if enable_graph_rank else _zero_like_reference(rank_bundle["inter_rank_loss"]),
        "graph_binary_loss": rank_bundle["graph_binary_loss"] if enable_graph_rank else _zero_like_reference(rank_bundle["graph_binary_loss"]),
    }
    metric_dict = {
        **recon_metrics,
        **local_metrics,
        "rank_acc": rank_bundle["rank_acc"] if enable_graph_rank else rank_bundle["rank_acc"].new_tensor(0.0),
        "intra_rank_acc": rank_bundle["intra_rank_acc"] if enable_graph_rank else rank_bundle["intra_rank_acc"].new_tensor(0.0),
        "inter_rank_acc": rank_bundle["inter_rank_acc"] if enable_graph_rank else rank_bundle["inter_rank_acc"].new_tensor(0.0),
        "graph_binary_acc": rank_bundle["graph_binary_acc"] if enable_graph_rank else rank_bundle["graph_binary_acc"].new_tensor(0.0),
        "graph_binary_clean_acc": rank_bundle["graph_binary_clean_acc"] if enable_graph_rank else rank_bundle["graph_binary_clean_acc"].new_tensor(0.0),
        "graph_binary_corrupted_acc": rank_bundle["graph_binary_corrupted_acc"] if enable_graph_rank else rank_bundle["graph_binary_corrupted_acc"].new_tensor(0.0),
        "mean_margin": rank_bundle["mean_margin"] if enable_graph_rank else rank_bundle["mean_margin"].new_tensor(0.0),
        "intra_mean_margin": rank_bundle["intra_mean_margin"] if enable_graph_rank else rank_bundle["intra_mean_margin"].new_tensor(0.0),
        "inter_mean_margin": rank_bundle["inter_mean_margin"] if enable_graph_rank else rank_bundle["inter_mean_margin"].new_tensor(0.0),
        "score_real_mean": rank_bundle["score_real_mean"] if enable_graph_rank else rank_bundle["score_real_mean"].new_tensor(0.0),
        "score_corrupted_mean": rank_bundle["score_corrupted_mean"] if enable_graph_rank else rank_bundle["score_corrupted_mean"].new_tensor(0.0),
        "score_real_min": rank_bundle["score_real_min"] if enable_graph_rank else rank_bundle["score_real_min"].new_tensor(0.0),
        "score_corrupted_max": rank_bundle["score_corrupted_max"] if enable_graph_rank else rank_bundle["score_corrupted_max"].new_tensor(0.0),
        "score_gap_minmax": rank_bundle["score_gap_minmax"] if enable_graph_rank else rank_bundle["score_gap_minmax"].new_tensor(0.0),
    }
    for key, value in rank_bundle.items():
        if key in loss_dict or key in metric_dict:
            continue
        if key.startswith("score_base_") or key.startswith("score_local_summary_"):
            metric_dict[key] = value if enable_graph_rank else value.new_tensor(0.0)
    return loss_dict, metric_dict
