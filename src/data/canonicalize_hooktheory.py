import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# =========================================================
# Constants
# =========================================================

VALID_SD = {
    "1", "2", "3", "4", "5", "6", "7",
    "b1", "b2", "b3", "b4", "b5", "b6", "b7",
    "#1", "#2", "#3", "#4", "#5", "#6", "#7",
    "bb1",
}

VALID_CHORD_TYPES = {5, 7, 9, 11, 13}
VALID_ROOTS = set(range(0, 8))          # 0..7
VALID_INVERSIONS = set(range(0, 8))     # оставим с запасом
VALID_APPLIED = set(range(0, 16))       # оставим с запасом

MODE_TO_PCSET = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],               # natural minor / aeolian
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "phrygian_dominant": [0, 1, 4, 5, 7, 8, 10],
}

PCSET_TO_MODE = {
    tuple(v): k for k, v in MODE_TO_PCSET.items()
}

TONIC_TO_PC = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

SECTION_LABEL_MAP = {
    "verse": "verse",
    "chorus": "chorus",
    "pre-chorus": "pre-chorus",
    "pre chorus": "pre-chorus",
    "prechorus": "pre-chorus",
    "bridge": "bridge",
    "intro": "intro",
    "outro": "outro",
    "instrumental": "instrumental",
    "solo": "solo",
}


# =========================================================
# Reporter
# =========================================================

class Reporter:
    def __init__(self, max_examples_per_key=20):
        self.max_examples_per_key = max_examples_per_key
        self.counts = Counter()
        self.examples = defaultdict(list)

    def warn(self, key, raw_value=None, song_id=None, note=None):
        self.counts[key] += 1
        if len(self.examples[key]) < self.max_examples_per_key:
            self.examples[key].append({
                "song_id": song_id,
                "raw_value": raw_value,
                "note": note,
            })

    def to_dict(self):
        return {
            "counts": dict(self.counts),
            "examples": dict(self.examples),
        }


# =========================================================
# Basic helpers
# =========================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_float(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def safe_int(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    try:
        if isinstance(x, float):
            if x.is_integer():
                return int(x)
            return None
        if isinstance(x, int):
            return x
        s = str(x).strip()
        if s == "":
            return None
        if re.fullmatch(r"[-+]?\d+", s):
            return int(s)
        if re.fullmatch(r"[-+]?\d+\.0+", s):
            return int(float(s))
        return None
    except Exception:
        return None


def safe_bool(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def maybe_round_int(x):
    if x is None:
        return None
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    return x


def canonical_split(split):
    if split is None:
        return None
    s = str(split).strip().lower()
    if s == "valid":
        return "val"
    if s in {"train", "val", "test"}:
        return s
    return s


def canonical_accidentals(s):
    if s is None:
        return None
    s = str(s).strip()
    s = s.replace("♭", "b").replace("♯", "#")
    s = s.replace("–", "-").replace("—", "-")
    return s


# =========================================================
# Tonic / scale
# =========================================================

def canonical_tonic_symbol(x, reporter=None, song_id=None):
    if x is None:
        return None

    s = canonical_accidentals(x)
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None

    # Нормализация регистра: буква + accidentals
    letter = s[0].upper()
    rest = s[1:].replace("♭", "b").replace("♯", "#")
    rest = rest.replace("B", "b") if rest not in {"b", "#", "bb", "##"} else rest
    # аккуратно — не ломаем "Bb", "C#", "E#"
    if rest in {"b", "#", "bb", "##", ""}:
        out = letter + rest
    else:
        out = letter + rest

    if out not in TONIC_TO_PC and reporter is not None:
        reporter.warn("invalid_tonic_symbol", raw_value=x, song_id=song_id)

    return out


def tonic_to_pc(tonic_symbol):
    if tonic_symbol is None:
        return None
    return TONIC_TO_PC.get(tonic_symbol)


def canonical_scale_label(x, reporter=None, song_id=None):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None

    s = s.replace("-", "_").replace(" ", "_")
    s_low = s.lower()

    alias = {
        "major": "major",
        "minor": "minor",
        "dorian": "dorian",
        "phrygian": "phrygian",
        "lydian": "lydian",
        "mixolydian": "mixolydian",
        "locrian": "locrian",
        "harmonicminor": "harmonic_minor",
        "harmonic_minor": "harmonic_minor",
        "phrygiandominant": "phrygian_dominant",
        "phrygian_dominant": "phrygian_dominant",
    }

    out = alias.get(s_low)
    if out is None:
        if reporter is not None:
            reporter.warn("unknown_scale_label", raw_value=x, song_id=song_id)
        return s_low
    return out


# =========================================================
# Melody
# =========================================================

def canonical_sd(x, reporter=None, song_id=None):
    if x is None:
        return "<UNK_SD>"

    s = canonical_accidentals(x)
    if s is None:
        return "<UNK_SD>"
    s = s.strip()

    if s == "":
        return "<EMPTY_SD>"

    # убираем пробелы внутри
    s = s.replace(" ", "")
    s = s.replace("♭", "b").replace("♯", "#")

    if s in VALID_SD:
        return s

    if reporter is not None:
        reporter.warn("unknown_sd", raw_value=x, song_id=song_id)
    return "<UNK_SD>"


# =========================================================
# List-like fields
# =========================================================

def normalize_int_list_field(values, field_name, reporter=None, song_id=None):
    if values is None:
        return []

    if not isinstance(values, list):
        values = [values]

    out = []
    for x in values:
        xi = safe_int(x)
        if xi is None:
            if reporter is not None:
                reporter.warn(f"invalid_{field_name}_item", raw_value=x, song_id=song_id)
            continue
        out.append(xi)

    out = sorted(set(out))
    return out


def canonical_alteration_token(x):
    if x is None:
        return None
    s = canonical_accidentals(x)
    if s is None:
        return None
    s = s.strip().replace(" ", "")
    if s == "":
        return None
    return s


def normalize_alterations(values, reporter=None, song_id=None):
    if values is None:
        return []

    if not isinstance(values, list):
        values = [values]

    out = []
    for x in values:
        token = canonical_alteration_token(x)
        if token is None:
            if reporter is not None:
                reporter.warn("invalid_alterations_item", raw_value=x, song_id=song_id)
            continue
        out.append(token)

    out = sorted(set(out))
    return out


# =========================================================
# Chord scalars
# =========================================================

def canonical_root(x, reporter=None, song_id=None):
    raw = safe_int(x)
    if raw is None:
        if x is not None and reporter is not None:
            reporter.warn("invalid_root", raw_value=x, song_id=song_id)
        return None
    if raw not in VALID_ROOTS:
        if reporter is not None:
            reporter.warn("out_of_domain_root", raw_value=x, song_id=song_id)
        return None
    return raw


def canonical_chord_type(x, reporter=None, song_id=None):
    raw = safe_int(x)
    if raw is None:
        if x is not None and reporter is not None:
            reporter.warn("invalid_chord_type", raw_value=x, song_id=song_id)
        return None
    if raw not in VALID_CHORD_TYPES:
        if reporter is not None:
            reporter.warn("out_of_domain_chord_type", raw_value=x, song_id=song_id)
        return None
    return raw


def canonical_inversion(x, reporter=None, song_id=None):
    raw = safe_int(x)
    if raw is None:
        if x is not None and reporter is not None:
            reporter.warn("invalid_inversion", raw_value=x, song_id=song_id)
        return None
    if raw not in VALID_INVERSIONS:
        if reporter is not None:
            reporter.warn("out_of_domain_inversion", raw_value=x, song_id=song_id)
        return None
    return raw


def canonical_applied(x, reporter=None, song_id=None):
    raw = safe_int(x)
    if raw is None:
        if x is not None and reporter is not None:
            reporter.warn("invalid_applied", raw_value=x, song_id=song_id)
        return None
    if raw not in VALID_APPLIED:
        if reporter is not None:
            reporter.warn("out_of_domain_applied", raw_value=x, song_id=song_id)
        return None
    return raw


# =========================================================
# Borrowed
# =========================================================

def parse_list_like_string(s):
    try:
        value = ast.literal_eval(s)
        if isinstance(value, list):
            return value
    except Exception:
        return None
    return None


def canonical_mode_name(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None

    s = s.replace("-", "_").replace(" ", "_")
    s_low = s.lower()

    alias = {
        "major": "major",
        "minor": "minor",
        "dorian": "dorian",
        "phrygian": "phrygian",
        "lydian": "lydian",
        "mixolydian": "mixolydian",
        "locrian": "locrian",
        "harmonicminor": "harmonic_minor",
        "harmonic_minor": "harmonic_minor",
        "phrygiandominant": "phrygian_dominant",
        "phrygian_dominant": "phrygian_dominant",
    }
    return alias.get(s_low, s_low)


def canonical_pcset(values, reporter=None, song_id=None):
    if values is None:
        return []

    if not isinstance(values, list):
        return []

    pcs = []
    for x in values:
        xi = safe_int(x)
        if xi is None:
            if reporter is not None:
                reporter.warn("invalid_borrowed_pcset_item", raw_value=x, song_id=song_id)
            continue
        pcs.append(xi % 12)

    return sorted(set(pcs))


def canonicalize_borrowed(x, reporter=None, song_id=None):
    """
    Возвращает объект:
    {
      "borrowed_kind": ...,
      "borrowed_mode_name": ...,
      "borrowed_pcset": [...]
    }
    """
    if x is None:
        return {
            "borrowed_kind": "none",
            "borrowed_mode_name": None,
            "borrowed_pcset": [],
        }

    # Пустая строка
    if isinstance(x, str) and x.strip() == "":
        return {
            "borrowed_kind": "none",
            "borrowed_mode_name": None,
            "borrowed_pcset": [],
        }

    # Строковый лад или строка-представление списка
    if isinstance(x, str):
        maybe_list = parse_list_like_string(x)
        if isinstance(maybe_list, list):
            pcset = canonical_pcset(maybe_list, reporter=reporter, song_id=song_id)
            mode_name = PCSET_TO_MODE.get(tuple(pcset))
            return {
                "borrowed_kind": "pcset",
                "borrowed_mode_name": mode_name,
                "borrowed_pcset": pcset,
            }

        mode_name = canonical_mode_name(x)
        if mode_name in MODE_TO_PCSET:
            return {
                "borrowed_kind": "mode_name",
                "borrowed_mode_name": mode_name,
                "borrowed_pcset": MODE_TO_PCSET[mode_name],
            }

        if reporter is not None:
            reporter.warn("unknown_borrowed_string", raw_value=x, song_id=song_id)
        return {
            "borrowed_kind": "unknown",
            "borrowed_mode_name": mode_name,
            "borrowed_pcset": [],
        }

    # Явный список
    if isinstance(x, list):
        pcset = canonical_pcset(x, reporter=reporter, song_id=song_id)
        mode_name = PCSET_TO_MODE.get(tuple(pcset))
        return {
            "borrowed_kind": "pcset",
            "borrowed_mode_name": mode_name,
            "borrowed_pcset": pcset,
        }

    if reporter is not None:
        reporter.warn("invalid_borrowed_type", raw_value=x, song_id=song_id)
    return {
        "borrowed_kind": "unknown",
        "borrowed_mode_name": None,
        "borrowed_pcset": [],
    }


# =========================================================
# Section labels
# =========================================================

def canonical_section_label(x, reporter=None, song_id=None):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s == "":
        return None

    s = s.replace("_", " ")
    s = " ".join(s.split())

    # Нормализуем частые варианты
    if s in SECTION_LABEL_MAP:
        return SECTION_LABEL_MAP[s]

    if reporter is not None:
        reporter.warn("unknown_section_label", raw_value=x, song_id=song_id)
    return s


def normalize_section_labels(labels, reporter=None, song_id=None):
    if labels is None:
        return [], []

    if isinstance(labels, str):
        labels = [labels]
    elif not isinstance(labels, list):
        labels = [labels]

    raw_labels = []
    norm_labels = []

    for x in labels:
        if x is None:
            continue
        raw_labels.append(x)
        norm = canonical_section_label(x, reporter=reporter, song_id=song_id)
        if norm is not None:
            norm_labels.append(norm)

    norm_labels = sorted(set(norm_labels))
    return raw_labels, norm_labels


# =========================================================
# Region normalization
# =========================================================

def dedup_sorted_dicts(regions, fields):
    """
    Удаляет точные последовательные дубли по указанным полям.
    regions должны быть уже отсортированы.
    """
    out = []
    prev_key = None
    for r in regions:
        key = tuple(r.get(f) for f in fields)
        if key == prev_key:
            continue
        out.append(r)
        prev_key = key
    return out


def normalize_keys(keys, reporter=None, song_id=None, keep_raw=False):
    out = []
    for obj in (keys or []):
        if not isinstance(obj, dict):
            if reporter is not None:
                reporter.warn("invalid_key_region", raw_value=obj, song_id=song_id)
            continue

        beat = safe_float(obj.get("beat"))
        tonic_raw = obj.get("tonic")
        scale_raw = obj.get("scale")

        tonic_symbol = canonical_tonic_symbol(tonic_raw, reporter=reporter, song_id=song_id)
        tonic_pc = tonic_to_pc(tonic_symbol)
        scale = canonical_scale_label(scale_raw, reporter=reporter, song_id=song_id)

        item = {
            "beat": maybe_round_int(beat),
            "tonic_symbol": tonic_symbol,
            "tonic_pc": tonic_pc,
            "scale": scale,
        }
        if keep_raw:
            item["tonic_symbol_raw"] = tonic_raw
            item["scale_raw"] = scale_raw

        out.append(item)

    out.sort(key=lambda x: (1e18 if x["beat"] is None else x["beat"]))
    out = dedup_sorted_dicts(out, ["beat", "tonic_symbol", "tonic_pc", "scale"])
    return out


def normalize_tempos(tempos, reporter=None, song_id=None):
    out = []
    for obj in (tempos or []):
        if not isinstance(obj, dict):
            if reporter is not None:
                reporter.warn("invalid_tempo_region", raw_value=obj, song_id=song_id)
            continue

        beat = safe_float(obj.get("beat"))
        bpm = safe_float(obj.get("bpm"))
        if bpm is None and obj.get("bpm") is not None and reporter is not None:
            reporter.warn("invalid_bpm", raw_value=obj.get("bpm"), song_id=song_id)

        out.append({
            "beat": maybe_round_int(beat),
            "bpm": maybe_round_int(bpm) if bpm is not None else None,
        })

    out.sort(key=lambda x: (1e18 if x["beat"] is None else x["beat"]))
    out = dedup_sorted_dicts(out, ["beat", "bpm"])
    return out


def normalize_meters(meters, reporter=None, song_id=None):
    out = []
    for obj in (meters or []):
        if not isinstance(obj, dict):
            if reporter is not None:
                reporter.warn("invalid_meter_region", raw_value=obj, song_id=song_id)
            continue

        beat = safe_float(obj.get("beat"))
        num_beats = safe_int(obj.get("num_beats"))
        beat_unit = safe_int(obj.get("beat_unit"))

        item = {
            "beat": maybe_round_int(beat),
            "num_beats": num_beats,
            "beat_unit": beat_unit,
            "meter_token": f"{num_beats}/{beat_unit}" if num_beats is not None and beat_unit is not None else None,
        }
        out.append(item)

    out.sort(key=lambda x: (1e18 if x["beat"] is None else x["beat"]))
    out = dedup_sorted_dicts(out, ["beat", "num_beats", "beat_unit"])
    return out


# =========================================================
# Main per-record normalization
# =========================================================

def normalize_song(song_id, song, reporter=None, keep_raw=False):
    meta = song.get("meta", {})
    melody = song.get("melody", [])
    chords = song.get("chords", [])
    sections = song.get("sections", [])

    # --- meta
    meta_out = {
        "split": canonical_split(meta.get("split")),
        "ori_uid": meta.get("ori_uid"),
        "end_beat": maybe_round_int(safe_float(meta.get("end_beat"))),
        "keys": normalize_keys(meta.get("keys"), reporter=reporter, song_id=song_id, keep_raw=keep_raw),
        "tempos": normalize_tempos(meta.get("tempos"), reporter=reporter, song_id=song_id),
        "meters": normalize_meters(meta.get("meters"), reporter=reporter, song_id=song_id),
    }

    # --- melody
    melody_out = []
    for note in (melody or []):
        if not isinstance(note, dict):
            if reporter is not None:
                reporter.warn("invalid_melody_note", raw_value=note, song_id=song_id)
            continue

        beat = maybe_round_int(safe_float(note.get("beat")))
        duration = maybe_round_int(safe_float(note.get("duration")))
        sd_raw = note.get("sd")
        sd = canonical_sd(sd_raw, reporter=reporter, song_id=song_id)
        octave = safe_int(note.get("octave"))
        if octave is None and note.get("octave") is not None and reporter is not None:
            reporter.warn("invalid_octave", raw_value=note.get("octave"), song_id=song_id)
        is_rest = safe_bool(note.get("is_rest"))

        item = {
            "beat": beat,
            "duration": duration,
            "sd": sd,
            "octave": octave,
            "is_rest": is_rest,
        }
        if keep_raw:
            item["sd_raw"] = sd_raw

        melody_out.append(item)

    # --- chords
    chords_out = []
    for chord in (chords or []):
        if not isinstance(chord, dict):
            if reporter is not None:
                reporter.warn("invalid_chord_event", raw_value=chord, song_id=song_id)
            continue

        beat = maybe_round_int(safe_float(chord.get("beat")))
        duration = maybe_round_int(safe_float(chord.get("duration")))
        root_raw = chord.get("root")
        type_raw = chord.get("type")
        inversion_raw = chord.get("inversion")
        applied_raw = chord.get("applied")
        borrowed_raw = chord.get("borrowed")
        alternate_raw = chord.get("alternate")

        root = canonical_root(root_raw, reporter=reporter, song_id=song_id)
        chord_type = canonical_chord_type(type_raw, reporter=reporter, song_id=song_id)
        inversion = canonical_inversion(inversion_raw, reporter=reporter, song_id=song_id)
        applied = canonical_applied(applied_raw, reporter=reporter, song_id=song_id)

        adds_raw = chord.get("adds")
        omits_raw = chord.get("omits")
        alterations_raw = chord.get("alterations")
        suspensions_raw = chord.get("suspensions")

        adds = normalize_int_list_field(adds_raw, "adds", reporter=reporter, song_id=song_id)
        omits = normalize_int_list_field(omits_raw, "omits", reporter=reporter, song_id=song_id)
        suspensions = normalize_int_list_field(suspensions_raw, "suspensions", reporter=reporter, song_id=song_id)
        alterations = normalize_alterations(alterations_raw, reporter=reporter, song_id=song_id)

        borrowed_info = canonicalize_borrowed(borrowed_raw, reporter=reporter, song_id=song_id)

        alternate = None
        if alternate_raw is not None:
            s = str(alternate_raw).strip()
            alternate = s if s != "" else None

        is_rest = safe_bool(chord.get("is_rest"))

        item = {
            "beat": beat,
            "duration": duration,
            "root": root,
            "type": chord_type,
            "inversion": inversion,
            "applied": applied,
            "adds": adds,
            "omits": omits,
            "alterations": alterations,
            "suspensions": suspensions,
            "borrowed_kind": borrowed_info["borrowed_kind"],
            "borrowed_mode_name": borrowed_info["borrowed_mode_name"],
            "borrowed_pcset": borrowed_info["borrowed_pcset"],
            "alternate": alternate,
            "is_rest": is_rest,
        }

        if keep_raw:
            item["root_raw"] = root_raw
            item["type_raw"] = type_raw
            item["inversion_raw"] = inversion_raw
            item["applied_raw"] = applied_raw
            item["adds_raw"] = adds_raw
            item["omits_raw"] = omits_raw
            item["alterations_raw"] = alterations_raw
            item["suspensions_raw"] = suspensions_raw
            item["borrowed_raw"] = borrowed_raw
            item["alternate_raw"] = alternate_raw

        chords_out.append(item)

    # --- sections
    sections_out = []
    for sec in (sections or []):
        if not isinstance(sec, dict):
            if reporter is not None:
                reporter.warn("invalid_section_segment", raw_value=sec, song_id=song_id)
            continue

        labels_raw, labels_norm = normalize_section_labels(
            sec.get("labels"),
            reporter=reporter,
            song_id=song_id
        )

        item = {
            "labels": labels_norm,
            "labels_tuple": labels_norm[:],  # удобно для downstream
            "duration_seconds": safe_float(sec.get("duration_seconds")),
            "segment_start_seconds": safe_float(sec.get("segment_start_seconds")),
            "segment_end_seconds": safe_float(sec.get("segment_end_seconds")),
        }
        if keep_raw:
            item["labels_raw"] = labels_raw

        sections_out.append(item)

    sections_out.sort(
        key=lambda x: (
            1e18 if x["segment_start_seconds"] is None else x["segment_start_seconds"],
            1e18 if x["segment_end_seconds"] is None else x["segment_end_seconds"],
        )
    )

    return {
        "song_id": song_id,
        "meta": meta_out,
        "melody": melody_out,
        "chords": chords_out,
        "sections": sections_out,
    }


# =========================================================
# Stats
# =========================================================

def compute_stats(canonical):
    stats = {
        "n_tracks": 0,
        "n_total_notes": 0,
        "n_total_chords": 0,
        "n_total_sections": 0,
        "n_tracks_with_sections": 0,
        "n_tracks_with_ori_uid": 0,
        "n_tracks_with_multiple_keys": 0,
        "n_tracks_with_multiple_tempos": 0,
        "n_tracks_with_multiple_meters": 0,
        "splits": Counter(),
        "sd_vocab": Counter(),
        "scale_vocab": Counter(),
        "section_label_vocab": Counter(),
        "borrowed_kind_vocab": Counter(),
    }

    for song in canonical.values():
        stats["n_tracks"] += 1

        split = song.get("meta", {}).get("split")
        stats["splits"][split] += 1

        if song.get("meta", {}).get("ori_uid") is not None:
            stats["n_tracks_with_ori_uid"] += 1

        keys = song.get("meta", {}).get("keys", [])
        tempos = song.get("meta", {}).get("tempos", [])
        meters = song.get("meta", {}).get("meters", [])

        if len(keys) > 1:
            stats["n_tracks_with_multiple_keys"] += 1
        if len(tempos) > 1:
            stats["n_tracks_with_multiple_tempos"] += 1
        if len(meters) > 1:
            stats["n_tracks_with_multiple_meters"] += 1

        for k in keys:
            stats["scale_vocab"][k.get("scale")] += 1

        melody = song.get("melody", [])
        chords = song.get("chords", [])
        sections = song.get("sections", [])

        stats["n_total_notes"] += len(melody)
        stats["n_total_chords"] += len(chords)
        stats["n_total_sections"] += len(sections)

        if len(sections) > 0:
            stats["n_tracks_with_sections"] += 1

        for n in melody:
            stats["sd_vocab"][n.get("sd")] += 1

        for c in chords:
            stats["borrowed_kind_vocab"][c.get("borrowed_kind")] += 1

        for s in sections:
            for label in s.get("labels", []):
                stats["section_label_vocab"][label] += 1

    # convert counters
    stats["splits"] = dict(stats["splits"])
    stats["sd_vocab"] = dict(stats["sd_vocab"])
    stats["scale_vocab"] = dict(stats["scale_vocab"])
    stats["section_label_vocab"] = dict(stats["section_label_vocab"])
    stats["borrowed_kind_vocab"] = dict(stats["borrowed_kind_vocab"])

    return stats


# =========================================================
# Main
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to hooktheory_processed.json")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for canonical outputs")
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw versions of normalized theory fields next to canonical ones"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = load_json(args.input)
    reporter = Reporter()

    canonical = {}
    for song_id, song in processed.items():
        canonical[song_id] = normalize_song(
            song_id=song_id,
            song=song,
            reporter=reporter,
            keep_raw=args.keep_raw,
        )

    stats = compute_stats(canonical)
    report = reporter.to_dict()

    dump_json(canonical, out_dir / "hooktheory_canonical.json")
    dump_json(stats, out_dir / "hooktheory_canonical.stats.json")
    dump_json(report, out_dir / "hooktheory_canonical.report.json")

    print("[INFO] done")
    print(f"[INFO] saved: {out_dir / 'hooktheory_canonical.json'}")
    print(f"[INFO] saved: {out_dir / 'hooktheory_canonical.stats.json'}")
    print(f"[INFO] saved: {out_dir / 'hooktheory_canonical.report.json'}")


if __name__ == "__main__":
    main()