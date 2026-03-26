from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

LOCAL_LEVELS = ("note", "chord", "onset")
SUPPORTED_CORRUPTION_MODES = (
    "note_sd_replacement",
    "chord_root_replacement",
    "chord_type_replacement",
    "swap_neighboring_chords",
    "onset_mismatch",
)


@dataclass
class MetricAccumulator:
    pos_count: int = 0
    neg_count: int = 0
    pos_logit_sum: float = 0.0
    neg_logit_sum: float = 0.0
    pos_prob_sum: float = 0.0
    neg_prob_sum: float = 0.0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, logits: torch.Tensor, probs: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor):
        if logits.numel() == 0:
            return
        logits_cpu = logits.detach().cpu().view(-1)
        probs_cpu = probs.detach().cpu().view(-1)
        targets_cpu = targets.detach().cpu().view(-1).long()
        preds_cpu = preds.detach().cpu().view(-1).long()

        pos_mask = targets_cpu == 1
        neg_mask = ~pos_mask
        pos_count = int(pos_mask.sum().item())
        neg_count = int(neg_mask.sum().item())

        self.pos_count += pos_count
        self.neg_count += neg_count
        if pos_count > 0:
            self.pos_logit_sum += float(logits_cpu[pos_mask].sum().item())
            self.pos_prob_sum += float(probs_cpu[pos_mask].sum().item())
        if neg_count > 0:
            self.neg_logit_sum += float(logits_cpu[neg_mask].sum().item())
            self.neg_prob_sum += float(probs_cpu[neg_mask].sum().item())

        self.tp += int(((preds_cpu == 1) & (targets_cpu == 1)).sum().item())
        self.tn += int(((preds_cpu == 0) & (targets_cpu == 0)).sum().item())
        self.fp += int(((preds_cpu == 1) & (targets_cpu == 0)).sum().item())
        self.fn += int(((preds_cpu == 0) & (targets_cpu == 1)).sum().item())

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        return float(numerator) / float(denominator) if denominator else 0.0

    def to_metrics(self) -> dict:
        precision = self._safe_div(self.tp, self.tp + self.fp)
        recall = self._safe_div(self.tp, self.tp + self.fn)
        f1 = self._safe_div(2.0 * precision * recall, precision + recall)
        tnr = self._safe_div(self.tn, self.tn + self.fp)

        return {
            "pos_count": self.pos_count,
            "neg_count": self.neg_count,
            "pos_logit_mean": self._safe_div(self.pos_logit_sum, self.pos_count),
            "neg_logit_mean": self._safe_div(self.neg_logit_sum, self.neg_count),
            "pos_prob_mean": self._safe_div(self.pos_prob_sum, self.pos_count),
            "neg_prob_mean": self._safe_div(self.neg_prob_sum, self.neg_count),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_acc": 0.5 * (recall + tnr),
        }


def _node_graph_ranges(batch, node_type: str) -> list[tuple[int, int]]:
    ptr = batch[node_type].ptr
    return [(int(ptr[i].item()), int(ptr[i + 1].item())) for i in range(ptr.numel() - 1)]


def _resolve_song_id(corrupted_batch, graph_index: int) -> str:
    metadata = getattr(corrupted_batch, "graph_metadata", None)
    if isinstance(metadata, list) and graph_index < len(metadata):
        item = metadata[graph_index]
        if isinstance(item, Mapping) and item.get("song_id") is not None:
            return str(item["song_id"])
    return f"graph_{graph_index}"


def build_level_binary_targets(corrupted_batch, corruption_metadata: List[dict] | None, level: str) -> list[dict]:
    ranges = _node_graph_ranges(corrupted_batch, level)
    key = f"{level}_corrupted_indices"
    bundle = []

    for graph_index, (start, end) in enumerate(ranges):
        node_count = max(0, end - start)
        targets = torch.zeros(node_count, dtype=torch.float)
        metadata = (corruption_metadata[graph_index] if corruption_metadata and graph_index < len(corruption_metadata) else {}) or {}
        positives = [int(idx) for idx in (metadata.get(key) or [])]
        valid_positives = sorted({idx for idx in positives if 0 <= idx < node_count})
        if valid_positives:
            targets[torch.tensor(valid_positives, dtype=torch.long)] = 1.0

        bundle.append(
            {
                "graph_index": graph_index,
                "song_id": _resolve_song_id(corrupted_batch, graph_index),
                "corruption_mode": str(metadata.get("mode", "unknown")),
                "start": start,
                "end": end,
                "targets": targets,
            }
        )
    return bundle


def _trim_examples(examples: list[dict], keep: int, reverse: bool) -> list[dict]:
    if len(examples) <= keep:
        return examples
    return sorted(examples, key=lambda item: item["prob"], reverse=reverse)[:keep]


def _empty_accumulators():
    level_accumulators = {level: MetricAccumulator() for level in LOCAL_LEVELS}
    mode_accumulators = {
        mode: {level: MetricAccumulator() for level in LOCAL_LEVELS}
        for mode in SUPPORTED_CORRUPTION_MODES
    }
    return level_accumulators, mode_accumulators


def _accumulate_local_scores(
    corrupted_outputs,
    corrupted_batch,
    corruption_metadata: List[dict] | None,
    threshold: float,
    hardest_k: int,
    level_accumulators: dict,
    mode_accumulators: dict,
    examples: dict,
):
    local_scores = corrupted_outputs.get("local_scores", {})

    for level in LOCAL_LEVELS:
        logits_all = local_scores.get(level)
        if logits_all is None:
            continue
        logits_all = logits_all.view(-1)
        for graph_target in build_level_binary_targets(corrupted_batch, corruption_metadata, level):
            start, end = graph_target["start"], graph_target["end"]
            if end <= start:
                continue
            logits = logits_all[start:end]
            targets = graph_target["targets"].to(logits.device)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            level_accumulators[level].update(logits, probs, targets, preds)
            mode = graph_target["corruption_mode"]
            if mode in mode_accumulators:
                mode_accumulators[mode][level].update(logits, probs, targets, preds)

            fp_indices = torch.nonzero((preds == 1.0) & (targets == 0.0), as_tuple=False).view(-1).tolist()
            fn_indices = torch.nonzero((preds == 0.0) & (targets == 1.0), as_tuple=False).view(-1).tolist()
            for local_idx in fp_indices:
                examples[level]["false_positives"].append(
                    {
                        "song_id": graph_target["song_id"],
                        "corruption_mode": mode,
                        "node_type": level,
                        "node_index": int(local_idx),
                        "logit": float(logits[local_idx].detach().cpu().item()),
                        "prob": float(probs[local_idx].detach().cpu().item()),
                        "target": 0,
                    }
                )
            for local_idx in fn_indices:
                examples[level]["false_negatives"].append(
                    {
                        "song_id": graph_target["song_id"],
                        "corruption_mode": mode,
                        "node_type": level,
                        "node_index": int(local_idx),
                        "logit": float(logits[local_idx].detach().cpu().item()),
                        "prob": float(probs[local_idx].detach().cpu().item()),
                        "target": 1,
                    }
                )

            examples[level]["false_positives"] = _trim_examples(examples[level]["false_positives"], hardest_k, reverse=True)
            examples[level]["false_negatives"] = _trim_examples(examples[level]["false_negatives"], hardest_k, reverse=False)


def collect_local_corruption_diagnostics(
    corrupted_outputs,
    corrupted_batch,
    corruption_metadata: List[dict] | None,
    threshold: float = 0.5,
    hardest_k: int = 8,
):
    level_accumulators, mode_accumulators = _empty_accumulators()
    examples = {level: {"false_positives": [], "false_negatives": []} for level in LOCAL_LEVELS}
    _accumulate_local_scores(
        corrupted_outputs=corrupted_outputs,
        corrupted_batch=corrupted_batch,
        corruption_metadata=corruption_metadata,
        threshold=threshold,
        hardest_k=hardest_k,
        level_accumulators=level_accumulators,
        mode_accumulators=mode_accumulators,
        examples=examples,
    )

    report = {level: level_accumulators[level].to_metrics() for level in LOCAL_LEVELS}
    report["by_corruption_mode"] = {
        mode: {level: mode_accumulators[mode][level].to_metrics() for level in LOCAL_LEVELS}
        for mode in SUPPORTED_CORRUPTION_MODES
    }
    return report, examples


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        "graph_corrupted": batch["graph_corrupted"].to(device),
        "corruption_metadata": batch.get("corruption_metadata"),
    }


def evaluate_teacher_local_corruption(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    threshold: float = 0.5,
    hardest_k: int = 8,
):
    model.eval()
    level_accumulators, mode_accumulators = _empty_accumulators()
    examples = {level: {"false_positives": [], "false_negatives": []} for level in LOCAL_LEVELS}

    with torch.no_grad():
        for step_index, batch in enumerate(loader, start=1):
            batch = _move_batch_to_device(batch, device)
            corrupted_outputs = model(batch["graph_corrupted"])
            _accumulate_local_scores(
                corrupted_outputs=corrupted_outputs,
                corrupted_batch=batch["graph_corrupted"],
                corruption_metadata=batch["corruption_metadata"],
                threshold=threshold,
                hardest_k=hardest_k,
                level_accumulators=level_accumulators,
                mode_accumulators=mode_accumulators,
                examples=examples,
            )
            if max_batches is not None and step_index >= max_batches:
                break

    report = {level: level_accumulators[level].to_metrics() for level in LOCAL_LEVELS}
    report["by_corruption_mode"] = {
        mode: {level: mode_accumulators[mode][level].to_metrics() for level in LOCAL_LEVELS}
        for mode in SUPPORTED_CORRUPTION_MODES
    }
    return report, examples


def save_local_diagnostic_reports(output_dir: Path, report: Mapping[str, object], examples: Mapping[str, object]):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "local_eval.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "local_eval_examples.json").write_text(json.dumps(examples, indent=2, sort_keys=True), encoding="utf-8")
