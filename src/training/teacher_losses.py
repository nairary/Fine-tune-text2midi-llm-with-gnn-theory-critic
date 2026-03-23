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


def compute_reconstruction_losses(masked_outputs: Mapping[str, Dict[str, torch.Tensor]], masked_batch, masked_labels: List[dict]):
    recon_logits = masked_outputs["recon_logits"]
    losses = {}
    metrics = {}
    total_recon_loss = None

    for head_name, spec in RECONSTRUCTION_SPECS.items():
        logits = recon_logits[head_name]
        node_type = spec["node_type"]
        field_name = spec["field_name"]
        valid_ids = spec["valid_ids"]
        loss_weight = spec["loss_weight"]
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
        reference = next(iter(recon_logits.values()))
        total_recon_loss = _zero_like_reference(reference)

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


def compute_teacher_ssl_losses(
    masked_outputs,
    real_outputs,
    corrupted_outputs,
    masked_batch,
    masked_labels: List[dict],
    lambda_recon: float = 1.0,
    lambda_rank: float = 0.5,
):
    recon_losses, recon_metrics = compute_reconstruction_losses(
        masked_outputs=masked_outputs,
        masked_batch=masked_batch,
        masked_labels=masked_labels,
    )
    rank_bundle = compute_ranking_loss(real_outputs=real_outputs, corrupted_outputs=corrupted_outputs)

    total_loss = lambda_recon * recon_losses["recon_loss"] + lambda_rank * rank_bundle["rank_loss"]
    loss_dict = {
        "loss": total_loss,
        **recon_losses,
        "rank_loss": rank_bundle["rank_loss"],
    }
    metric_dict = {
        **recon_metrics,
        "rank_acc": rank_bundle["rank_acc"],
        "mean_margin": rank_bundle["mean_margin"],
        "score_real_mean": rank_bundle["score_real_mean"],
        "score_corrupted_mean": rank_bundle["score_corrupted_mean"],
    }
    return loss_dict, metric_dict
