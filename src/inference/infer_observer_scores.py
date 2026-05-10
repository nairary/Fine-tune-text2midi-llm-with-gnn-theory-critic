#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch_geometric.data import Batch

from src.dataloader.theory_helpers import build_theory_context
from src.observer.data_pipeline import build_observer_graph, build_observer_song_record
from src.observer.model import ObserverGNN
from src.observer.schema import OBSERVER_EDGE_TYPES, OBSERVER_NUM_FIELDS, build_observer_vocab_sizes


TONIC_TO_PC = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}

MODE_ALIASES = {
    "maj": "major",
    "major": "major",
    "ionian": "major",
    "min": "minor",
    "minor": "minor",
    "aeolian": "minor",
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


class ObserverInferenceError(ValueError):
    """Raised when observer batch scoring input or checkpoint is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a JSON array of MIDI candidates with a trained ObserverGNN. "
            "Each input row must contain midi_path, key, mode, bpm, "
            "meter_numenator, and meter_denumenator. Correct spellings "
            "meter_numerator/meter_denominator are accepted too."
        )
    )
    parser.add_argument("--input-json", type=Path, required=True, help="JSON array, or object with an 'items' array.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Observer checkpoint, e.g. training/best.pt.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output path. Defaults to stdout.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--chord-weights-yaml",
        type=Path,
        default=None,
        help="Override chord scorer weights. By default uses checkpoint config if present.",
    )
    parser.add_argument(
        "--chord-instrument-name",
        default=None,
        help="Override harmonic MIDI instrument. By default uses checkpoint config or 'chords'.",
    )
    parser.add_argument(
        "--use-fallback-44",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override fallback 4/4 behavior. By default uses checkpoint config or true.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Return rows with score=null and error=... instead of failing the whole batch.",
    )
    parser.add_argument("--sort", action="store_true", help="Sort output results by score descending.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser.parse_args()


def load_input_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        payload = payload["items"]
    if not isinstance(payload, list):
        raise ObserverInferenceError("Input JSON must be an array or an object with an 'items' array.")

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ObserverInferenceError(f"Input row index={idx} must be a JSON object.")
        rows.append(item)
    return rows


def _load_spec_global() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "metadata" / "specs" / "spec_global.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _config_mapping(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    config = checkpoint.get("config", {})
    return config if isinstance(config, Mapping) else {}


def _nested_mapping(root: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = root.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _config_get(root: Mapping[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = root
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[ObserverGNN, Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ObserverInferenceError(f"Checkpoint must be a mapping: {checkpoint_path}")
    if "model_state_dict" not in checkpoint:
        raise ObserverInferenceError(f"Checkpoint has no 'model_state_dict': {checkpoint_path}")

    cfg = _config_mapping(checkpoint)
    model_cfg = _nested_mapping(cfg, "observer_model")
    if not model_cfg:
        model_cfg = cfg

    spec_global = _load_spec_global()
    model = ObserverGNN(
        cat_vocab_sizes=build_observer_vocab_sizes(build_theory_context(), spec_global),
        num_feature_dims={node_type: len(OBSERVER_NUM_FIELDS[node_type]) for node_type in OBSERVER_NUM_FIELDS},
        edge_types=OBSERVER_EDGE_TYPES,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pooling_mode=str(model_cfg.get("pooling_mode", "mean")),
        pooling_output_dim=model_cfg.get("pooling_output_dim"),
        score_head_hidden_dim=model_cfg.get("score_head_hidden_dim"),
        use_bar_sequence_transformer=bool(model_cfg.get("use_bar_sequence_transformer", False)),
        bar_transformer_num_layers=int(model_cfg.get("bar_transformer_num_layers", 2)),
        bar_transformer_num_heads=int(model_cfg.get("bar_transformer_num_heads", 4)),
        bar_transformer_ff_dim=model_cfg.get("bar_transformer_ff_dim"),
        bar_transformer_dropout=model_cfg.get("bar_transformer_dropout"),
        bar_transformer_pooling=str(model_cfg.get("bar_transformer_pooling", "cls")),
        bar_transformer_combine=str(model_cfg.get("bar_transformer_combine", "concat")),
        score_head_activation=str(model_cfg.get("score_head_activation", "relu")),
        score_head_layer_norm=bool(model_cfg.get("score_head_layer_norm", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def _normalize_key_token(raw_key: Any) -> str:
    if raw_key is None:
        raise ObserverInferenceError("Missing required field 'key'.")
    token = str(raw_key).strip()
    if not token:
        raise ObserverInferenceError("Field 'key' must be non-empty.")
    token = token.replace("♭", "b").replace("♯", "#")
    # Accept accidental spelling in the first token even if callers pass "C major".
    token = token.split()[0]
    if len(token) == 1:
        return token.upper()
    return token[0].upper() + token[1:].replace("B", "b")


def parse_tonic_pc(raw_key: Any) -> int:
    if isinstance(raw_key, int):
        if 0 <= raw_key <= 11:
            return int(raw_key)
        raise ObserverInferenceError(f"Numeric key must be in [0..11], got {raw_key}.")
    token = _normalize_key_token(raw_key)
    pc = TONIC_TO_PC.get(token.upper())
    if pc is None:
        raise ObserverInferenceError(f"Unsupported key '{raw_key}'. Expected C, C#, Db, ..., B.")
    return int(pc)


def parse_mode_name(raw_mode: Any) -> str:
    if raw_mode is None:
        raise ObserverInferenceError("Missing required field 'mode'.")
    token = str(raw_mode).strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        raise ObserverInferenceError("Field 'mode' must be non-empty.")
    mode = MODE_ALIASES.get(token)
    if mode is None:
        raise ObserverInferenceError(
            f"Unsupported mode '{raw_mode}'. Expected one of: {', '.join(sorted(set(MODE_ALIASES.values())))}."
        )
    theory_ctx = build_theory_context()
    if mode not in theory_ctx["mode_to_pcset"]:
        raise ObserverInferenceError(f"Mode '{mode}' is not supported by theory context.")
    return mode


def _get_first_present(row: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise ObserverInferenceError(f"Missing required field '{names[0]}'.")


def _parse_positive_float(value: Any, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ObserverInferenceError(f"Field '{field_name}' must be numeric, got {value!r}.") from exc
    if out <= 0:
        raise ObserverInferenceError(f"Field '{field_name}' must be > 0, got {value!r}.")
    return out


def _parse_positive_int(value: Any, field_name: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ObserverInferenceError(f"Field '{field_name}' must be integer, got {value!r}.") from exc
    if out <= 0:
        raise ObserverInferenceError(f"Field '{field_name}' must be > 0, got {value!r}.")
    return out


def meter_denominator_to_beat_unit(numerator: int, denominator: int) -> int:
    # The observer schema follows teacher's beat_unit encoding: 1 for simple meters,
    # 3 for compound 6/8, 9/8, 12/8-style meters.
    if denominator == 8 and numerator in {6, 9, 12}:
        return 3
    return 1


def normalize_grpo_input_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    midi_path = row.get("midi_path")
    if not isinstance(midi_path, str) or not midi_path.strip():
        raise ObserverInferenceError("Missing required non-empty field 'midi_path'.")

    numerator = _parse_positive_int(
        _get_first_present(row, ("meter_numenator", "meter_numerator")),
        "meter_numenator",
    )
    denominator = _parse_positive_int(
        _get_first_present(row, ("meter_denumenator", "meter_denominator")),
        "meter_denumenator",
    )

    return {
        "song_id": str(row.get("song_id") or row.get("id") or f"candidate_{index}"),
        "sample_id": str(row.get("sample_id") or row.get("id") or f"candidate_{index}"),
        "midi_path": midi_path.strip(),
        "tonic_pc": parse_tonic_pc(row.get("key")),
        "mode_name": parse_mode_name(row.get("mode")),
        "bpm": _parse_positive_float(row.get("bpm"), "bpm"),
        "num_beats": numerator,
        "beat_unit": meter_denominator_to_beat_unit(numerator, denominator),
        "meter_numerator": numerator,
        "meter_denominator": denominator,
    }


def _resolve_optional_path(path_value: Any, checkpoint_path: Path) -> str | None:
    if path_value is None or str(path_value).strip() == "":
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path)
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return str(cwd_path)
    checkpoint_relative = checkpoint_path.parent / path
    if checkpoint_relative.exists():
        return str(checkpoint_relative)
    return str(path)


def resolve_runtime_options(
    cfg: Mapping[str, Any],
    checkpoint_path: Path,
    chord_weights_yaml: Path | None,
    chord_instrument_name: str | None,
    use_fallback_44: bool | None,
) -> dict[str, Any]:
    cfg_chord_weights = _config_get(cfg, ("observer_training", "chord_weights_yaml"))
    cfg_instrument = _config_get(cfg, ("observer_training", "chord_instrument_name"), "chords")
    cfg_fallback = _config_get(cfg, ("observer_training", "use_fallback_44"), True)

    return {
        "chord_weights_yaml": _resolve_optional_path(chord_weights_yaml or cfg_chord_weights, checkpoint_path),
        "chord_instrument_name": str(chord_instrument_name or cfg_instrument or "chords"),
        "use_fallback_44": bool(cfg_fallback if use_fallback_44 is None else use_fallback_44),
    }


def _score_graph_batch(model: ObserverGNN, graph_items: list[tuple[int, Any]], device: torch.device) -> list[tuple[int, float]]:
    if not graph_items:
        return []
    batch = Batch.from_data_list([graph for _, graph in graph_items]).to(device)
    with torch.no_grad():
        scores = model(batch).view(-1).detach().cpu().tolist()
    return [(index, float(score)) for (index, _), score in zip(graph_items, scores, strict=True)]


def assign_descending_ranks(results: list[dict[str, Any]]) -> None:
    valid = [row for row in results if row.get("score") is not None]
    valid.sort(key=lambda row: (-float(row["score"]), int(row["index"])))
    for rank, row in enumerate(valid, start=1):
        row["rank"] = rank
    for row in results:
        if row.get("score") is None:
            row["rank"] = None


def score_observer_rows(
    rows: list[Mapping[str, Any]],
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int = 8,
    chord_weights_yaml: Path | None = None,
    chord_instrument_name: str | None = None,
    use_fallback_44: bool | None = None,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    model, cfg = build_model_from_checkpoint(checkpoint_path, device)
    runtime = resolve_runtime_options(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        chord_weights_yaml=chord_weights_yaml,
        chord_instrument_name=chord_instrument_name,
        use_fallback_44=use_fallback_44,
    )

    results: list[dict[str, Any]] = []
    graph_batch: list[tuple[int, Any]] = []
    batch_n = max(1, int(batch_size))

    def flush_batch() -> None:
        nonlocal graph_batch
        for result_index, score in _score_graph_batch(model, graph_batch, device):
            results[result_index]["score"] = score
        graph_batch = []

    for index, row in enumerate(rows):
        result = {
            "index": index,
            "midi_path": row.get("midi_path"),
            "key": row.get("key"),
            "mode": row.get("mode"),
            "score": None,
            "rank": None,
            "error": None,
        }
        results.append(result)

        try:
            sample = normalize_grpo_input_row(row, index=index)
            record = build_observer_song_record(
                sample,
                chord_weights_yaml=runtime["chord_weights_yaml"],
                chord_instrument_name=runtime["chord_instrument_name"],
                use_fallback_44=runtime["use_fallback_44"],
            )
            graph = build_observer_graph(record)
            graph_batch.append((index, graph))
            if len(graph_batch) >= batch_n:
                flush_batch()
        except Exception as exc:  # noqa: BLE001
            if not continue_on_error:
                raise ObserverInferenceError(f"Failed to score input row index={index}: {exc}") from exc
            result["error"] = str(exc)

    flush_batch()
    assign_descending_ranks(results)
    scores = [row["score"] for row in sorted(results, key=lambda x: int(x["index"]))]
    return {
        "checkpoint": str(checkpoint_path),
        "count": len(results),
        "scores": scores,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    rows = load_input_rows(args.input_json)
    payload = score_observer_rows(
        rows=rows,
        checkpoint_path=args.checkpoint,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        chord_weights_yaml=args.chord_weights_yaml,
        chord_instrument_name=args.chord_instrument_name,
        use_fallback_44=args.use_fallback_44,
        continue_on_error=args.continue_on_error,
    )

    if args.sort:
        payload["results"] = sorted(
            payload["results"],
            key=lambda row: (
                row["score"] is None,
                -float(row["score"]) if row["score"] is not None else 0.0,
                int(row["index"]),
            ),
        )

    text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output_json is None:
        print(text)
        return
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
