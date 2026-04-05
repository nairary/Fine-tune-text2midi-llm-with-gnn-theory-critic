"""Theory helper utilities for song-level corruption generation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .function_rules import STRICT_FUNCTIONS_V1_SAFE

ROOT = Path(__file__).resolve().parents[2]
VOCAB_DIR = ROOT / "metadata" / "vocabs"
SPEC_DIR = ROOT / "metadata" / "specs"

MODE_TO_PCSET = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "phrygian_dominant": [0, 1, 4, 5, 7, 8, 10],
}
MODE_SEQUENCE = tuple(MODE_TO_PCSET.keys())

SD_TOKEN_TO_CHROMATIC = {
    "1": 0,
    "b1": 11,
    "#1": 1,
    "2": 2,
    "b2": 1,
    "#2": 3,
    "3": 4,
    "b3": 3,
    "#3": 5,
    "4": 5,
    "b4": 4,
    "#4": 6,
    "5": 7,
    "b5": 6,
    "#5": 8,
    "6": 9,
    "b6": 8,
    "#6": 10,
    "7": 11,
    "b7": 10,
    "#7": 0,
    "bb1": 10,
}


def try_parse_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_float(value, default: float) -> float:
    parsed = try_parse_float(value)
    if parsed is None:
        return float(default)
    return parsed


@lru_cache(maxsize=1)
def _load_vocab_maps() -> dict:
    def _load_json(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    vocab_sd = _load_json(VOCAB_DIR / "vocab_melody_sd.json")
    vocab_scale = _load_json(VOCAB_DIR / "vocab_key_scale.json")
    vocab_borrowed_kind = _load_json(VOCAB_DIR / "vocab_borrowed_kind.json")
    vocab_borrowed_mode_name = _load_json(VOCAB_DIR / "vocab_borrowed_mode_name.json")
    spec_global = _load_json(SPEC_DIR / "spec_global.json")
    spec_chord_sets = _load_json(SPEC_DIR / "spec_chord_sets.json")

    root_allowed = spec_global["root"]["allowed_values"]
    type_allowed = spec_global["type"]["allowed_values"]
    inversion_allowed = spec_global["inversion"]["allowed_values"]
    applied_allowed = spec_global["applied"]["allowed_values"]

    root_id_to_raw = {idx + 1: raw for idx, raw in enumerate(root_allowed)}
    type_id_to_raw = {idx + 1: raw for idx, raw in enumerate(type_allowed)}
    inversion_id_to_raw = {idx + 1: raw for idx, raw in enumerate(inversion_allowed)}
    applied_id_to_raw = {idx + 1: raw for idx, raw in enumerate(applied_allowed)}

    return {
        "sd_token_to_id": vocab_sd,
        "sd_id_to_token": {v: k for k, v in vocab_sd.items()},
        "scale_name_to_id": vocab_scale,
        "scale_id_to_name": {v: k for k, v in vocab_scale.items()},
        "borrowed_kind_to_id": vocab_borrowed_kind,
        "borrowed_id_to_kind": {v: k for k, v in vocab_borrowed_kind.items()},
        "borrowed_mode_to_id": vocab_borrowed_mode_name,
        "borrowed_mode_id_to_name": {v: k for k, v in vocab_borrowed_mode_name.items()},
        "root_id_to_raw": root_id_to_raw,
        "type_id_to_raw": type_id_to_raw,
        "inversion_id_to_raw": inversion_id_to_raw,
        "applied_id_to_raw": applied_id_to_raw,
        "chord_add_allowed_values": spec_chord_sets["adds"]["allowed_values"],
    }


def build_theory_context() -> dict:
    maps = _load_vocab_maps()
    return {
        **maps,
        "mode_to_pcset": MODE_TO_PCSET,
        "mode_sequence": MODE_SEQUENCE,
        "sd_token_to_chromatic": SD_TOKEN_TO_CHROMATIC,
        "strict_functions_v1": STRICT_FUNCTIONS_V1_SAFE,
    }


def decode_sd_to_chromatic(sd_id: int, theory_ctx: dict) -> int | None:
    token = theory_ctx["sd_id_to_token"].get(int(sd_id))
    if token is None or token.startswith("<"):
        return None
    return theory_ctx["sd_token_to_chromatic"].get(token)


def ordered_mode_template(mode_name: str, theory_ctx: dict) -> list[int] | None:
    pcset = theory_ctx["mode_to_pcset"].get(mode_name)
    if not pcset or len(pcset) < 7:
        return None
    return list(pcset[:7])


def decode_root_raw(chord: dict, theory_ctx: dict) -> int | None:
    return theory_ctx["root_id_to_raw"].get(int(chord.get("root_id", 0)))


def decode_type_raw(chord: dict, theory_ctx: dict) -> int | None:
    return theory_ctx["type_id_to_raw"].get(int(chord.get("type_id", 0)))


def _resolve_root_anchor(root_raw: int, template: list[int]) -> tuple[int, int] | None:
    """
    Return (anchor_degree_idx, root_pc).

    Special-case support:
      - root_raw in 0..6 -> direct diatonic degree.
      - root_raw == 7 -> treat as bVII: flatten mode degree VII by semitone.
    """
    if 0 <= root_raw <= 6:
        return root_raw, template[root_raw] % 12
    if root_raw == 7:
        # v1 special-case (current relative-pitch representation):
        # root is fixed as chromatic b7; tertian stacking stays anchored on VII.
        return 6, 10
    return None


def chord_pitch_classes_tertian(song_obj: dict, chord: dict, theory_ctx: dict) -> set[int]:
    """Build chord pitch classes via tertian stacking in the active mode."""
    root_raw = decode_root_raw(chord, theory_ctx)
    type_raw = decode_type_raw(chord, theory_ctx)
    if root_raw is None or type_raw is None:
        return set()
    if type_raw not in {5, 7, 9, 11, 13}:
        return set()

    active_mode = select_active_mode_name(song_obj, chord, theory_ctx)
    template = ordered_mode_template(active_mode, theory_ctx)
    if template is None:
        return set()
    root_resolution = _resolve_root_anchor(root_raw, template)
    if root_resolution is None:
        return set()
    anchor_degree_idx, root_pc = root_resolution

    n_tones = (type_raw + 1) // 2  # 5->3, 7->4, ..., 13->7
    pcs = {root_pc}
    for tone_idx in range(n_tones):
        if tone_idx == 0:
            continue
        degree_idx = (anchor_degree_idx + 2 * tone_idx) % 7
        pcs.add(template[degree_idx] % 12)

    add_values = theory_ctx.get("chord_add_allowed_values", [])
    adds_vec = chord.get("adds_vec") or []
    for vec_idx, bit in enumerate(adds_vec):
        if not bit or vec_idx >= len(add_values):
            continue
        degree_value = int(add_values[vec_idx])
        add_degree_idx = (anchor_degree_idx + (degree_value - 1)) % 7
        pcs.add(template[add_degree_idx] % 12)

    return pcs


def chord_bass_and_top_pcs(song_obj: dict, chord: dict, theory_ctx: dict) -> tuple[int, int] | None:
    """Return approximate (bass_pc, top_voice_pc) from tertian construction."""
    root_raw = decode_root_raw(chord, theory_ctx)
    type_raw = decode_type_raw(chord, theory_ctx)
    if root_raw is None or type_raw is None:
        return None
    if type_raw not in {5, 7, 9, 11, 13}:
        return None

    active_mode = select_active_mode_name(song_obj, chord, theory_ctx)
    template = ordered_mode_template(active_mode, theory_ctx)
    if template is None:
        return None
    root_resolution = _resolve_root_anchor(root_raw, template)
    if root_resolution is None:
        return None
    anchor_degree_idx, root_pc = root_resolution

    n_tones = (type_raw + 1) // 2
    top_degree_idx = (anchor_degree_idx + 2 * (n_tones - 1)) % 7
    top_pc = template[top_degree_idx] % 12
    return root_pc % 12, top_pc


def select_active_mode_name(song_obj: dict, chord: dict | None, theory_ctx: dict) -> str:
    if chord:
        borrowed_kind = theory_ctx["borrowed_id_to_kind"].get(int(chord.get("borrowed_kind_id", 0)))
        borrowed_mode = theory_ctx["borrowed_mode_id_to_name"].get(int(chord.get("borrowed_mode_name_id", 0)))
        if borrowed_kind == "mode_name" and borrowed_mode in theory_ctx["mode_to_pcset"]:
            return borrowed_mode

    main_scale_id = int(song_obj.get("meta", {}).get("main_key_scale_id", 2))
    main_scale_name = theory_ctx["scale_id_to_name"].get(main_scale_id, "major")
    if main_scale_name not in theory_ctx["mode_to_pcset"]:
        return "major"
    return main_scale_name


def classify_function_from_root_raw(root_raw: int, mode_name: str, theory_ctx: dict) -> str | None:
    mode_key = "minor" if mode_name == "minor" else "major"
    mode_rules = theory_ctx["strict_functions_v1"][mode_key]
    if root_raw in mode_rules["T_roots_raw"]:
        return "T"
    if root_raw in mode_rules["PD_roots_raw"]:
        return "PD"
    if root_raw in mode_rules["D_roots_raw"]:
        return "D"
    return None


def is_strong_note_position(note: dict, song_obj: dict, include_midpoint: bool = True) -> bool:
    num_beats = safe_float(song_obj.get("meta", {}).get("main_num_beats", 4.0), 4.0)
    beat = try_parse_float(note.get("beat"))

    if "pos_in_bar" in note and note.get("pos_in_bar") is not None:
        pos = try_parse_float(note.get("pos_in_bar"))
        if pos is None:
            return False
    else:
        if beat is None:
            return False
        bar_idx = int((beat - 1.0) // num_beats)
        bar_start = 1.0 + bar_idx * num_beats
        pos = beat - bar_start

    if abs(pos) < 1e-6:
        return True
    if include_midpoint and abs(num_beats - 4.0) < 1e-6 and abs(pos - 2.0) < 1e-6:
        return True
    return False


def find_covering_chord_index(song_obj: dict, note: dict) -> int | None:
    beat = try_parse_float(note.get("beat"))
    if beat is None:
        return None
    chords = song_obj.get("chords", [])
    for idx, chord in enumerate(chords):
        c_start = try_parse_float(chord.get("beat"))
        c_duration = safe_float(chord.get("duration", 0.0), 0.0)
        if c_start is None:
            continue
        c_end = c_start + max(0.0, c_duration)
        if c_start <= beat < c_end:
            return idx
    return None
