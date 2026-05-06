from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


DEFAULT_TEACHER_OBJECTIVES = (
    "recon",
    "graph_rank",
    "note_local",
    "chord_local",
    "onset_local",
)


class DynamicLossWeighter(nn.Module):
    """Learn uncertainty-style weights for a set of scalar loss objectives."""

    def __init__(
        self,
        objective_names: Sequence[str] = DEFAULT_TEACHER_OBJECTIVES,
        *,
        init_log_var: float = 0.0,
        min_log_var: float = -5.0,
        max_log_var: float = 5.0,
        regularizer_weight: float = 1.0,
    ) -> None:
        super().__init__()
        deduped: list[str] = []
        seen: set[str] = set()
        for raw_name in objective_names:
            name = str(raw_name)
            if name in seen:
                continue
            deduped.append(name)
            seen.add(name)
        if not deduped:
            raise ValueError("DynamicLossWeighter requires at least one objective.")
        if float(min_log_var) > float(max_log_var):
            raise ValueError("min_log_var must be <= max_log_var.")

        self.objective_names = tuple(deduped)
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)
        self.regularizer_weight = float(regularizer_weight)
        self.log_vars = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(float(init_log_var), dtype=torch.float))
                for name in self.objective_names
            }
        )

    def forward(
        self,
        objective_losses: Mapping[str, torch.Tensor],
        base_weights: Mapping[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base_weights = base_weights or {}
        total_loss: torch.Tensor | None = None
        metrics: dict[str, torch.Tensor] = {}
        active_count = 0

        for name, loss_value in objective_losses.items():
            if name not in self.log_vars:
                raise KeyError(f"Unknown dynamic loss objective '{name}'.")
            base_weight = float(base_weights.get(name, 1.0))
            if base_weight <= 0.0:
                continue
            if loss_value.numel() != 1:
                raise ValueError(f"Dynamic loss objective '{name}' must be scalar.")

            log_var = torch.clamp(self.log_vars[name], min=self.min_log_var, max=self.max_log_var)
            effective_weight = torch.exp(-log_var) * base_weight
            component = effective_weight * loss_value + self.regularizer_weight * log_var
            total_loss = component if total_loss is None else total_loss + component
            active_count += 1

            metrics[f"dynamic_weight_{name}"] = effective_weight.detach()
            metrics[f"dynamic_log_var_{name}"] = log_var.detach()
            metrics[f"dynamic_loss_component_{name}"] = component.detach()

        if total_loss is None or active_count == 0:
            raise ValueError("No active positive-weight objectives were provided for dynamic loss weighting.")

        metrics["dynamic_active_objectives"] = next(iter(objective_losses.values())).new_tensor(float(active_count)).detach()
        return total_loss, metrics


def build_teacher_dynamic_loss_weighter(losses_cfg) -> DynamicLossWeighter | None:
    dynamic_cfg = losses_cfg.get("dynamic_weighting") if hasattr(losses_cfg, "get") else None
    if dynamic_cfg is None or not bool(dynamic_cfg.get("enabled", False)):
        return None

    method = str(dynamic_cfg.get("method", "uncertainty"))
    if method != "uncertainty":
        raise ValueError(f"Unsupported dynamic loss weighting method '{method}'.")

    objectives_cfg = dynamic_cfg.get("objectives", {})
    objective_names = [
        objective_name
        for objective_name in DEFAULT_TEACHER_OBJECTIVES
        if bool(objectives_cfg.get(objective_name, True))
    ]
    return DynamicLossWeighter(
        objective_names=objective_names,
        init_log_var=float(dynamic_cfg.get("init_log_var", 0.0)),
        min_log_var=float(dynamic_cfg.get("min_log_var", -5.0)),
        max_log_var=float(dynamic_cfg.get("max_log_var", 5.0)),
        regularizer_weight=float(dynamic_cfg.get("regularizer_weight", 1.0)),
    )
