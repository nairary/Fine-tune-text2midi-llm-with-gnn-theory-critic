from __future__ import annotations

import random
from typing import Sequence


class CorruptionModeBalancer:
    """Prioritize less-used corruption modes ahead of more frequent ones."""

    def __init__(self, modes: Sequence[str], mode_weights: dict[str, float] | None = None):
        deduped_modes: list[str] = []
        seen: set[str] = set()
        for raw_mode in modes:
            mode = str(raw_mode)
            if mode in seen:
                continue
            deduped_modes.append(mode)
            seen.add(mode)
        self._modes = tuple(deduped_modes)
        self._counts = {mode: 0 for mode in self._modes}
        self._weights = {}
        for mode in self._modes:
            raw_weight = 1.0 if mode_weights is None else float(mode_weights.get(mode, 1.0))
            self._weights[mode] = raw_weight if raw_weight > 0.0 else 1.0

    def ordered_modes(self, rng=None) -> list[str]:
        rng = rng or random
        decorated = []
        for index, mode in enumerate(self._modes):
            weight = self._weights.get(mode, 1.0)
            weighted_usage = (float(self._counts.get(mode, 0)) + float(rng.random())) / weight
            decorated.append((weighted_usage, index, mode))
        decorated.sort()
        return [mode for _, _, mode in decorated]

    def record_applied(self, mode: str) -> None:
        if mode in self._counts:
            self._counts[mode] += 1

    def usage_counts(self) -> dict[str, int]:
        return dict(self._counts)
