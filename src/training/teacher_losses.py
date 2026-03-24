from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

import torch
import torch.nn.functional as F

from src.models.teacher_heads import RECONSTRUCTION_SPECS


def _zero_like_reference(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _filter_and_encode_targets(
    selected_logits: torch.Tensor,
    target_values: torch.Tensor,
    valid_ids: Iterable[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_ids = list(valid_ids)
    id_to_index = {int(value): idx for idx, value in enumerate(valid_ids)}
    valid_positions = [idx for idx, value in enumerate(target_values.view(-1).tolist()) if int(value) in id_to_index]
    if not valid_positions:
        return selected_logits[:0], torch.empty((0,), dtype=torch.long, device=selected_logits.device)
    valid_index_tensor = torch.tensor(valid_positions, dtype=torch.long, device=selected_logits.device)
    filtered_logits = selected_logits.index_select(0, valid_index_tensor)
    filtered_targets = target_values.view(-1).index_select(0, valid_index_tensor).tolist()
    encoded_targets = torch.tensor([id_to_index[int(value)] for value in filtered_targets], dtype=torch.long, device=selected_logits.device)
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


def compute_ranking_loss(real_outputs, corrupted_outputs):
    score_real = real_outputs["graph_score"].view(-1)
    score_corrupted = corrupted_outputs["graph_score"].view(-1)
    margin = score_real - score_corrupted
    rank_loss = -F.logsigmoid(margin).mean()
    rank_acc = (margin > 0).float().mean()
    mean_margin = margin.mean()

    return {
        "rank_loss": rank_loss,
        "rank_acc": rank_acc.detach(),
        "mean_margin": mean_margin.detach(),
        "score_real_mean": score_real.mean().detach(),
        "score_corrupted_mean": score_corrupted.mean().detach(),
    }


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
        sampled_logits = []
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
            for local_idx in valid_corrupted:
                global_idx = start + local_idx
                sampled_logits.append(level_logits_all[global_idx])
                sampled_targets.append(1.0)
            for local_idx in sampled_clean:
                global_idx = start + local_idx
                sampled_logits.append(level_logits_all[global_idx])
                sampled_targets.append(0.0)

        if not sampled_logits:
            losses[loss_key] = _zero_like_reference(level_logits_all)
            continue

        level_logits = torch.stack(sampled_logits, dim=0)
        level_targets = torch.tensor(sampled_targets, dtype=torch.float, device=level_logits.device)
        loss_value = F.binary_cross_entropy_with_logits(level_logits, level_targets)
        predictions = (torch.sigmoid(level_logits) >= 0.5).float()
        metrics[acc_key] = (predictions == level_targets).float().mean().detach()
        losses[loss_key] = loss_value
    return losses, metrics


def compute_teacher_ssl_losses(
    masked_outputs,
    real_outputs,
    corrupted_outputs,
    masked_batch,
    corrupted_batch,
    masked_labels: List[dict],
    corruption_metadata: List[dict] | None = None,
    lambda_recon: float = 1.0,
    lambda_graph_rank: float = 0.5,
    lambda_note_local: float = 0.5,
    lambda_chord_local: float = 0.5,
    lambda_onset_local: float = 0.5,
    enable_graph_rank: bool = True,
    enable_note_local: bool = True,
    enable_chord_local: bool = True,
    enable_onset_local: bool = True,
    recon_weights: Mapping[str, float] | None = None,
    enabled_heads: Mapping[str, bool] | None = None,
    local_negatives_per_positive: int = 2,
):
    recon_losses, recon_metrics = compute_reconstruction_losses(
        masked_outputs=masked_outputs,
        masked_batch=masked_batch,
        masked_labels=masked_labels,
        recon_weights=recon_weights,
        enabled_heads=enabled_heads,
    )
    rank_bundle = compute_ranking_loss(real_outputs=real_outputs, corrupted_outputs=corrupted_outputs)
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

    total_loss = lambda_recon * recon_losses["recon_loss"]
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
    }
    metric_dict = {
        **recon_metrics,
        **local_metrics,
        "rank_acc": rank_bundle["rank_acc"] if enable_graph_rank else rank_bundle["rank_acc"].new_tensor(0.0),
        "mean_margin": rank_bundle["mean_margin"] if enable_graph_rank else rank_bundle["mean_margin"].new_tensor(0.0),
        "score_real_mean": rank_bundle["score_real_mean"] if enable_graph_rank else rank_bundle["score_real_mean"].new_tensor(0.0),
        "score_corrupted_mean": rank_bundle["score_corrupted_mean"] if enable_graph_rank else rank_bundle["score_corrupted_mean"].new_tensor(0.0),
    }
    return loss_dict, metric_dict
