from __future__ import annotations

from collections import Counter

from src.dataloader.corruption_balancer import CorruptionModeBalancer
from src.observer.build_observer_pair_dataset import _build_pair_tasks, _seed_balancer_from_counts, _split_section_local_modes


def test_section_all_local_balanced_tasks_include_all_section_modes_and_local_slots():
    modes = [
        "adjacent_section_swap",
        "section_duplicate",
        "strongbeat_nonchord_note",
        "note_onset_shift",
    ]

    tasks = _build_pair_tasks(
        pair_mode_strategy="section_all_local_balanced",
        corruption_modes=modes,
        pairs_per_song=99,
        section_pairs_per_mode=1,
        local_pairs_per_song=2,
    )

    assert tasks == [
        ("section_forced", "adjacent_section_swap", 0),
        ("section_forced", "section_duplicate", 0),
        ("local_balanced", None, 0),
        ("local_balanced", None, 1),
    ]


def test_split_section_local_modes_preserves_order():
    section_modes, local_modes = _split_section_local_modes(
        [
            "strongbeat_nonchord_note",
            "adjacent_section_swap",
            "note_onset_shift",
            "section_duplicate",
        ]
    )

    assert section_modes == ["adjacent_section_swap", "section_duplicate"]
    assert local_modes == ["strongbeat_nonchord_note", "note_onset_shift"]


def test_seed_balancer_from_existing_counts_prioritizes_less_used_local_mode():
    balancer = CorruptionModeBalancer(["mode_a", "mode_b"])
    _seed_balancer_from_counts(balancer, Counter({"mode_a": 5, "mode_b": 0}))

    assert balancer.ordered_modes()[0] == "mode_b"
