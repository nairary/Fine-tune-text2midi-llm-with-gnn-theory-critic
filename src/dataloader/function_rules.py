"""Strict functional harmony rule tables for theory-aware corruptions."""

STRICT_ROOT_LABELS_RAW = {
    0: "I",
    1: "II",
    2: "III",
    3: "IV",
    4: "V",
    5: "VI",
    6: "VII",
    7: "bVII",
}

STRICT_FUNCTIONS_V1_SAFE = {
    "major": {
        "T_roots_raw": [0],
        "PD_roots_raw": [1, 3],
        "D_roots_raw": [4],
        "SKIP_roots_raw": [2, 5, 6, 7],
    },
    "minor": {
        "T_roots_raw": [0],
        "PD_roots_raw": [1, 3],
        "D_roots_raw": [4],
        "SKIP_roots_raw": [2, 5, 6, 7],
    },
}

STRICT_FUNCTIONS_V2_QUALITY_AWARE = {
    "major": {
        "T": [{"raw_root": 0, "quality": "any_tertian"}],
        "PD": [
            {"raw_root": 1, "quality": "diatonic_on_active_mode"},
            {"raw_root": 3, "quality": "diatonic_on_active_mode"},
        ],
        "D": [
            {"raw_root": 4, "quality": "diatonic_on_active_mode"},
            {"raw_root": 6, "quality": "leading_tone_diminished"},
        ],
        "SKIP": [2, 5, 7],
    },
    "minor": {
        "T": [{"raw_root": 0, "quality": "any_tertian"}],
        "PD": [
            {"raw_root": 1, "quality": "supertonic_diminished"},
            {"raw_root": 3, "quality": "diatonic_on_active_mode"},
        ],
        "D": [
            {"raw_root": 4, "quality": "major_or_dominant"},
            {"raw_root": 6, "quality": "leading_tone_diminished"},
        ],
        "SKIP": [2, 5, 7],
    },
}

STRICT_TRIPLET_PATTERNS_V1 = [
    ("T", "PD", "D"),
    ("PD", "D", "T"),
    ("T", "D", "T"),
    ("T", "PD", "T"),
]

STRICT_BAD_REPLACEMENTS_V1 = {
    "T_slot": ["PD", "D"],
    "PD_slot": ["T", "D"],
    "D_slot": ["T", "PD"],
}
