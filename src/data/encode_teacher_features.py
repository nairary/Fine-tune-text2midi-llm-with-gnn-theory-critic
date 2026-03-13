import argparse
import json
from collections import Counter
from pathlib import Path


# =========================================================
# I/O
# =========================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# Metadata loading
# =========================================================

def load_metadata(metadata_dir):
    metadata_dir = Path(metadata_dir)

    vocabs_dir = metadata_dir / "vocabs"
    specs_dir = metadata_dir / "specs"

    vocabs = {
        "melody_sd": load_json(vocabs_dir / "vocab_melody_sd.json"),
        "key_scale": load_json(vocabs_dir / "vocab_key_scale.json"),
        "tonic_symbol": load_json(vocabs_dir / "vocab_tonic_symbol.json"),
        "borrowed_kind": load_json(vocabs_dir / "vocab_borrowed_kind.json"),
        "borrowed_mode_name": load_json(vocabs_dir / "vocab_borrowed_mode_name.json"),
        "section_label": load_json(vocabs_dir / "vocab_section_label.json"),
    }

    specs = {
        "global": load_json(specs_dir / "spec_global.json"),
        "chord_sets": load_json(specs_dir / "spec_chord_sets.json"),
        "field_specs": load_json(specs_dir / "field_specs.json"),
    }

    return vocabs, specs


# =========================================================
# Generic encoders
# =========================================================

def get_vocab_unk_id(vocab):
    for token in ["<UNK>", "<NONE>", "<PAD>"]:
        if token in vocab:
            return vocab[token]
    return 0


def encode_vocab(vocab, value):
    if value in vocab:
        return vocab[value]
    return get_vocab_unk_id(vocab)


def build_allowed_value_id_map(allowed_values, reserve_zero_for_unknown=True):
    """
    Builds {value: id}.
    If reserve_zero_for_unknown=True, ids start at 1.
    """
    start = 1 if reserve_zero_for_unknown else 0
    return {v: i + start for i, v in enumerate(allowed_values)}


def build_range_value_id_map(min_val, max_val, reserve_zero_for_unknown=True):
    values = list(range(min_val, max_val + 1))
    return build_allowed_value_id_map(values, reserve_zero_for_unknown=reserve_zero_for_unknown)


def encode_with_value_map(value_map, value, unknown_id=0):
    return value_map.get(value, unknown_id)


def make_multi_hot(values, allowed_values):
    allowed_index = {v: i for i, v in enumerate(allowed_values)}
    vec = [0] * len(allowed_values)
    if values is None:
        return vec

    for x in values:
        if x in allowed_index:
            vec[allowed_index[x]] = 1
    return vec


def make_fixed_range_multi_hot(values, size):
    vec = [0] * size
    if values is None:
        return vec
    for x in values:
        if isinstance(x, int) and 0 <= x < size:
            vec[x] = 1
    return vec


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x):
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def safe_bool_to_int(x):
    if x is True:
        return 1
    if x is False:
        return 0
    return -1


# =========================================================
# Build integer maps from specs
# =========================================================

def build_runtime_maps(specs):
    global_spec = specs["global"]

    runtime_maps = {
        "root_id_map": build_allowed_value_id_map(global_spec["root"]["allowed_values"]),
        "type_id_map": build_allowed_value_id_map(global_spec["type"]["allowed_values"]),
        "inversion_id_map": build_allowed_value_id_map(global_spec["inversion"]["allowed_values"]),
        "applied_id_map": build_allowed_value_id_map(global_spec["applied"]["allowed_values"]),
        "tonic_pc_id_map": build_allowed_value_id_map(global_spec["tonic_pc"]["allowed_values"]),
        "num_beats_id_map": build_allowed_value_id_map(global_spec["num_beats"]["allowed_values"]),
        "beat_unit_id_map": build_allowed_value_id_map(global_spec["beat_unit"]["allowed_values"]),
        "octave_id_map": build_range_value_id_map(
            global_spec["octave"]["min"],
            global_spec["octave"]["max"]
        ),
    }

    return runtime_maps


# =========================================================
# Per-field encoding
# =========================================================

def encode_key_region(region, vocabs, runtime_maps):
    tonic_symbol = region.get("tonic_symbol")
    tonic_pc = region.get("tonic_pc")
    scale = region.get("scale")

    return {
        "beat": safe_float(region.get("beat")),
        "tonic_symbol_id": encode_vocab(vocabs["tonic_symbol"], tonic_symbol),
        "tonic_pc_id": encode_with_value_map(runtime_maps["tonic_pc_id_map"], tonic_pc, unknown_id=0),
        "tonic_pc": tonic_pc,
        "scale_id": encode_vocab(vocabs["key_scale"], scale),
    }


def encode_tempo_region(region):
    return {
        "beat": safe_float(region.get("beat")),
        "bpm": safe_float(region.get("bpm")),
    }


def encode_meter_region(region, runtime_maps):
    num_beats = region.get("num_beats")
    beat_unit = region.get("beat_unit")

    return {
        "beat": safe_float(region.get("beat")),
        "num_beats_id": encode_with_value_map(runtime_maps["num_beats_id_map"], num_beats, unknown_id=0),
        "beat_unit_id": encode_with_value_map(runtime_maps["beat_unit_id_map"], beat_unit, unknown_id=0),
        "num_beats": num_beats,
        "beat_unit": beat_unit,
        "meter_token": region.get("meter_token"),
    }


def encode_melody_note(note, vocabs, runtime_maps):
    sd = note.get("sd")
    octave = note.get("octave")

    return {
        "beat": safe_float(note.get("beat")),
        "duration": safe_float(note.get("duration")),
        "sd_id": encode_vocab(vocabs["melody_sd"], sd),
        "octave_id": encode_with_value_map(runtime_maps["octave_id_map"], octave, unknown_id=0),
        "is_rest": safe_bool_to_int(note.get("is_rest")),
    }


def encode_chord(chord, vocabs, specs, runtime_maps):
    chord_set_spec = specs["chord_sets"]

    return {
        "beat": safe_float(chord.get("beat")),
        "duration": safe_float(chord.get("duration")),

        "root_id": encode_with_value_map(runtime_maps["root_id_map"], chord.get("root"), unknown_id=0),
        "type_id": encode_with_value_map(runtime_maps["type_id_map"], chord.get("type"), unknown_id=0),
        "inversion_id": encode_with_value_map(runtime_maps["inversion_id_map"], chord.get("inversion"), unknown_id=0),
        "applied_id": encode_with_value_map(runtime_maps["applied_id_map"], chord.get("applied"), unknown_id=0),

        "adds_vec": make_multi_hot(chord.get("adds"), chord_set_spec["adds"]["allowed_values"]),
        "omits_vec": make_multi_hot(chord.get("omits"), chord_set_spec["omits"]["allowed_values"]),
        "suspensions_vec": make_multi_hot(chord.get("suspensions"), chord_set_spec["suspensions"]["allowed_values"]),
        "alterations_vec": make_multi_hot(chord.get("alterations"), chord_set_spec["alterations"]["allowed_values"]),
        "borrowed_pcset_vec": make_fixed_range_multi_hot(
            chord.get("borrowed_pcset"),
            chord_set_spec["borrowed_pcset"]["size"]
        ),

        "borrowed_kind_id": encode_vocab(vocabs["borrowed_kind"], chord.get("borrowed_kind")),
        "borrowed_mode_name_id": encode_vocab(
            vocabs["borrowed_mode_name"],
            chord.get("borrowed_mode_name") if chord.get("borrowed_mode_name") is not None else "<NONE>"
        ),

        "is_rest": safe_bool_to_int(chord.get("is_rest")),
    }


def encode_section(section, vocabs):
    labels = section.get("labels") or []
    return {
        "label_ids": [encode_vocab(vocabs["section_label"], x) for x in labels],
        "duration_seconds": safe_float(section.get("duration_seconds")),
        "segment_start_seconds": safe_float(section.get("segment_start_seconds")),
        "segment_end_seconds": safe_float(section.get("segment_end_seconds")),
    }


# =========================================================
# Song-level encoding
# =========================================================

def get_first_or_none(items):
    if items and len(items) > 0:
        return items[0]
    return None


def encode_song(song_id, song, vocabs, specs, runtime_maps):
    meta = song.get("meta", {})
    melody = song.get("melody", [])
    chords = song.get("chords", [])
    sections = song.get("sections", [])

    key_regions = [encode_key_region(x, vocabs, runtime_maps) for x in (meta.get("keys") or [])]
    tempo_regions = [encode_tempo_region(x) for x in (meta.get("tempos") or [])]
    meter_regions = [encode_meter_region(x, runtime_maps) for x in (meta.get("meters") or [])]

    first_key = get_first_or_none(key_regions)
    first_tempo = get_first_or_none(tempo_regions)
    first_meter = get_first_or_none(meter_regions)

    encoded = {
        "song_id": song_id,
        "meta": {
            "split": meta.get("split"),
            "ori_uid": meta.get("ori_uid"),
            "end_beat": safe_float(meta.get("end_beat")),

            # region lists
            "key_regions": key_regions,
            "tempo_regions": tempo_regions,
            "meter_regions": meter_regions,

            # convenient summaries for dataloader / graph builder
            "main_key_tonic_symbol_id": first_key["tonic_symbol_id"] if first_key else 0,
            "main_key_tonic_pc_id": first_key["tonic_pc_id"] if first_key else 0,
            "main_key_tonic_pc": first_key["tonic_pc"] if first_key else None,
            "main_key_scale_id": first_key["scale_id"] if first_key else 0,

            "main_bpm": first_tempo["bpm"] if first_tempo else None,

            "main_num_beats_id": first_meter["num_beats_id"] if first_meter else 0,
            "main_beat_unit_id": first_meter["beat_unit_id"] if first_meter else 0,
            "main_num_beats": first_meter["num_beats"] if first_meter else None,
            "main_beat_unit": first_meter["beat_unit"] if first_meter else None,
        },

        "melody": [encode_melody_note(x, vocabs, runtime_maps) for x in melody],
        "chords": [encode_chord(x, vocabs, specs, runtime_maps) for x in chords],
        "sections": [encode_section(x, vocabs) for x in sections],
    }

    return encoded


# =========================================================
# Stats / reports
# =========================================================

def compute_stats(encoded):
    stats = {
        "n_tracks": 0,
        "n_total_notes": 0,
        "n_total_chords": 0,
        "n_total_sections": 0,
        "n_tracks_with_sections": 0,
        "splits": Counter(),
        "n_tracks_with_ori_uid": 0,
    }

    for song in encoded.values():
        stats["n_tracks"] += 1
        stats["splits"][song["meta"]["split"]] += 1

        stats["n_total_notes"] += len(song.get("melody", []))
        stats["n_total_chords"] += len(song.get("chords", []))
        stats["n_total_sections"] += len(song.get("sections", []))

        if len(song.get("sections", [])) > 0:
            stats["n_tracks_with_sections"] += 1
        if song["meta"].get("ori_uid") is not None:
            stats["n_tracks_with_ori_uid"] += 1

    stats["splits"] = dict(stats["splits"])
    return stats


# =========================================================
# Main
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to hooktheory_canonical.json")
    parser.add_argument("--metadata-dir", type=str, required=True, help="Path to metadata directory")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save final encoded JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical = load_json(args.input)
    vocabs, specs = load_metadata(args.metadata_dir)
    runtime_maps = build_runtime_maps(specs)

    encoded = {}
    for song_id, song in canonical.items():
        encoded[song_id] = encode_song(song_id, song, vocabs, specs, runtime_maps)

    stats = compute_stats(encoded)

    dump_json(encoded, out_dir / "teacher_encoded.json")
    dump_json(stats, out_dir / "teacher_encoded.stats.json")
    dump_json(runtime_maps, out_dir / "teacher_encoder_maps.json")

    print("[INFO] done")
    print(f"[INFO] saved: {out_dir / 'teacher_encoded.json'}")
    print(f"[INFO] saved: {out_dir / 'teacher_encoded.stats.json'}")
    print(f"[INFO] saved: {out_dir / 'teacher_encoder_maps.json'}")


if __name__ == "__main__":
    main()