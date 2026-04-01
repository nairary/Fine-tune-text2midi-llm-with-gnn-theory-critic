from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch


def _node_tensor(graph, node_type: str) -> torch.Tensor:
    try:
        return graph[node_type].x
    except Exception:
        return torch.zeros((0, 0), dtype=torch.float32)


def _edge_index(graph, edge_type: tuple[str, str, str]) -> torch.Tensor:
    try:
        return graph[edge_type].edge_index
    except Exception:
        return torch.zeros((2, 0), dtype=torch.long)


def to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]
    return value


def graph_summary(graph) -> dict[str, dict[str, int]]:
    node_types = getattr(graph, "node_types", [])
    edge_types = getattr(graph, "edge_types", [])
    node_counts = {node_type: int(_node_tensor(graph, node_type).size(0)) for node_type in node_types}
    edge_counts = {"__".join(edge_type): int(_edge_index(graph, edge_type).size(1)) for edge_type in edge_types}
    return {"node_counts": node_counts, "edge_counts": edge_counts}


def serialize_graph_nodes(graph) -> dict[str, Any]:
    serialized = {}
    for node_type in getattr(graph, "node_types", []):
        rows = _node_tensor(graph, node_type).detach().cpu().tolist()
        serialized[node_type] = {
            "count": len(rows),
            "rows": [
                {
                    "node_index": row_idx,
                    "raw": row,
                }
                for row_idx, row in enumerate(rows)
            ],
        }
    return serialized


def serialize_graph_edges(graph) -> dict[str, Any]:
    serialized = {}
    for edge_type in getattr(graph, "edge_types", []):
        edge_name = "__".join(edge_type)
        edge_index = _edge_index(graph, edge_type).detach().cpu()
        edges = [[int(src), int(dst)] for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist())]
        serialized[edge_name] = {
            "edge_type": list(edge_type),
            "count": len(edges),
            "edges": edges,
        }
    return serialized


def serialize_graph(graph) -> dict[str, Any]:
    return {
        "nodes": serialize_graph_nodes(graph),
        "edges": serialize_graph_edges(graph),
    }


def _row_diffs(before: torch.Tensor, after: torch.Tensor, max_items: int = 20) -> dict[str, Any]:
    changed_rows = []
    row_count = min(before.size(0), after.size(0))
    for row_idx in range(row_count):
        before_row = before[row_idx]
        after_row = after[row_idx]
        if torch.equal(before_row, after_row):
            continue
        delta = after_row - before_row
        changed_rows.append(
            {
                "row_index": row_idx,
                "before": before_row.detach().cpu().tolist(),
                "after": after_row.detach().cpu().tolist(),
                "l1_diff": float(delta.abs().sum().item()),
                "l2_diff": float(delta.norm(p=2).item()),
            }
        )
    return {
        "changed_row_indices": [item["row_index"] for item in changed_rows[:max_items]],
        "changed_rows": changed_rows[:max_items],
        "changed_rows_count": len(changed_rows),
        "truncated": len(changed_rows) > max_items,
    }


def graph_diff(real_graph, corrupted_graph, max_items: int = 20) -> dict[str, Any]:
    node_diff = {}
    real_node_types = set(getattr(real_graph, "node_types", []))
    corrupted_node_types = set(getattr(corrupted_graph, "node_types", []))
    for node_type in sorted(real_node_types | corrupted_node_types):
        before = _node_tensor(real_graph, node_type)
        after = _node_tensor(corrupted_graph, node_type)
        count_before = int(before.size(0))
        count_after = int(after.size(0))
        row_payload = _row_diffs(before, after, max_items=max_items)
        node_diff[node_type] = {
            "count_before": count_before,
            "count_after": count_after,
            "added_rows_count": max(0, count_after - count_before),
            "removed_rows_count": max(0, count_before - count_after),
            **row_payload,
        }

    edge_diff = {}
    real_edge_types = set(getattr(real_graph, "edge_types", []))
    corrupted_edge_types = set(getattr(corrupted_graph, "edge_types", []))
    for edge_type in sorted(real_edge_types | corrupted_edge_types):
        edge_name = "__".join(edge_type)
        before_edges = set(map(tuple, _edge_index(real_graph, edge_type).t().detach().cpu().tolist()))
        after_edges = set(map(tuple, _edge_index(corrupted_graph, edge_type).t().detach().cpu().tolist()))
        added = [list(edge) for edge in sorted(after_edges - before_edges)]
        removed = [list(edge) for edge in sorted(before_edges - after_edges)]
        edge_diff[edge_name] = {
            "count_before": len(before_edges),
            "count_after": len(after_edges),
            "added_edges_count": len(added),
            "removed_edges_count": len(removed),
            "added_edges": added[:max_items],
            "removed_edges": removed[:max_items],
            "truncated": len(added) > max_items or len(removed) > max_items,
        }

    return {"nodes": node_diff, "edges": edge_diff}


def model_output_diff(model, device: torch.device, real_graph, corrupted_graph, top_k: int = 10) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        real_outputs = model(real_graph.to(device))
        corrupted_outputs = model(corrupted_graph.to(device))

    graph_score_real = float(real_outputs["graph_score"].detach().cpu().item())
    graph_score_corrupted = float(corrupted_outputs["graph_score"].detach().cpu().item())
    graph_embedding_real = real_outputs["graph_embedding"].detach().cpu()
    graph_embedding_corrupted = corrupted_outputs["graph_embedding"].detach().cpu()

    graph_payload = {
        "graph_score_real": graph_score_real,
        "graph_score_corrupted": graph_score_corrupted,
        "graph_score_delta": graph_score_corrupted - graph_score_real,
        "graph_embedding_delta_l2": float((graph_embedding_corrupted - graph_embedding_real).norm(p=2).item()),
    }
    if "graph_score_features" in real_outputs and "graph_score_features" in corrupted_outputs:
        graph_payload["graph_score_features_delta_l2"] = float(
            (corrupted_outputs["graph_score_features"] - real_outputs["graph_score_features"]).norm(p=2).item()
        )
        graph_payload["graph_score_features_real"] = real_outputs["graph_score_features"]
        graph_payload["graph_score_features_corrupted"] = corrupted_outputs["graph_score_features"]
    if "local_score_summaries" in real_outputs and "local_score_summaries" in corrupted_outputs:
        graph_payload["local_score_summaries_real"] = real_outputs["local_score_summaries"]
        graph_payload["local_score_summaries_corrupted"] = corrupted_outputs["local_score_summaries"]
        graph_payload["local_score_summaries_delta"] = (
            corrupted_outputs["local_score_summaries"] - real_outputs["local_score_summaries"]
        )

    node_payload = {}
    for node_type, real_embeddings in real_outputs["node_embeddings"].items():
        corrupted_embeddings = corrupted_outputs["node_embeddings"][node_type]
        deltas = (corrupted_embeddings - real_embeddings).norm(dim=-1).detach().cpu()
        k = min(top_k, deltas.numel())
        top_values, top_indices = torch.topk(deltas, k=k)
        node_payload[node_type] = {
            "top_changed_node_indices": top_indices.tolist(),
            "top_changed_node_delta_norms": top_values.tolist(),
        }

    local_payload = {}
    for node_type, real_local in real_outputs.get("local_scores", {}).items():
        corrupted_local = corrupted_outputs.get("local_scores", {}).get(node_type)
        if corrupted_local is None:
            continue
        delta = (corrupted_local - real_local).detach().cpu().abs()
        k = min(top_k, delta.numel())
        top_values, top_indices = torch.topk(delta, k=k)
        local_payload[node_type] = {
            "top_k_changed_nodes": top_indices.tolist(),
            "top_k_abs_delta": top_values.tolist(),
        }

    return {
        "graph": graph_payload,
        "nodes": node_payload,
        "local_scores": local_payload,
    }


def corruption_applied(metadata: Mapping[str, Any] | None) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    return bool(metadata.get("applied", False))


def write_json(path: Path, payload: Any):
    path.write_text(json.dumps(to_python(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def prioritize_diagnostic_examples(examples: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    applied = [example for example in examples if corruption_applied(example.get("corruption_metadata"))]
    fallback = [example for example in examples if not corruption_applied(example.get("corruption_metadata"))]
    return (applied + fallback)[:max_samples]
