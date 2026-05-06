from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.render_encoded_song_to_midi import load_octave_id_map, render_song_to_pretty_midi
from src.dataloader.corruption_balancer import CorruptionModeBalancer
from src.dataloader.song_corruptions import corrupt_song_obj
from src.dataloader.theory_helpers import build_theory_context
from src.observer.pipeline_paths import resolve_observer_pipeline_paths

LOGGER = logging.getLogger(__name__)


@dataclass
class BuildStats:
    total: int = 0
    built_pairs: int = 0
    skipped_rows: int = 0


class PairBuildError(ValueError):
    pass


SECTION_CORRUPTION_MODES = {
    "adjacent_section_swap",
    "non_adjacent_section_swap",
    "section_duplicate",
    "section_drop_keep_silence",
    "section_drop_and_close_gap",
    "section_entry_non_tonic_substitution",
    "section_exit_non_dominant_substitution",
}

PAIR_MODE_STRATEGIES = {"first_applicable", "all_modes", "section_all_local_balanced"}


def _base_cwd() -> Path:
    try:
        return Path(hydra.utils.get_original_cwd())
    except Exception:
        return Path(os.getcwd())


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        out = []
        for song_id, song_obj in payload.items():
            if isinstance(song_obj, dict):
                song_obj.setdefault("meta", {}).setdefault("song_id", song_id)
                out.append(song_obj)
        return out
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    raise PairBuildError("Encoded dataset must be dict or list")


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dedup_skip_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = json.dumps(
            {
                "index": row.get("index"),
                "split": row.get("split"),
                "source_song_id": row.get("source_song_id"),
                "pair_group_id": row.get("pair_group_id"),
                "sample_id": row.get("sample_id"),
                "reason_skipped": row.get("reason_skipped"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        dedup[key] = row
    return list(dedup.values())


def _resolve_mode_name(meta: dict[str, Any], scale_id_to_name: dict[int, str]) -> str | None:
    mode_name = meta.get("mode_name")
    if isinstance(mode_name, str) and mode_name:
        return mode_name
    try:
        return scale_id_to_name[int(meta.get("main_key_scale_id"))]
    except Exception:
        return None


def _resolve_tonic_pc(meta: dict[str, Any]) -> int | None:
    for key in ("main_key_tonic_pc", "tonic_pc"):
        raw = meta.get(key)
        if raw is None:
            continue
        try:
            return int(raw) % 12
        except Exception:
            continue
    return None


def _resolve_beat_origin(song_obj: dict[str, Any]) -> float:
    for root in (song_obj, song_obj.get("meta", {})):
        if not isinstance(root, dict):
            continue
        for key in ("beat_origin", "main_beat_origin"):
            raw = root.get(key)
            if raw is None:
                continue
            try:
                return float(raw)
            except Exception:
                continue
    return 1.0


def _resolve_meta(song_obj: dict[str, Any], scale_id_to_name: dict[int, str]) -> tuple[dict[str, Any] | None, str | None]:
    meta = song_obj.get("meta") if isinstance(song_obj.get("meta"), dict) else {}
    song_id = meta.get("song_id") or song_obj.get("song_id")
    if not isinstance(song_id, str) or not song_id:
        return None, "missing_song_id"
    split = meta.get("split")
    if not isinstance(split, str) or not split:
        return None, "missing_split"
    tonic_pc = _resolve_tonic_pc(meta)
    if tonic_pc is None:
        return None, "missing_tonic_pc"
    mode_name = _resolve_mode_name(meta, scale_id_to_name)
    if not mode_name:
        return None, "missing_mode_name"
    try:
        bpm = float(meta.get("main_bpm"))
        num_beats = int(meta.get("main_num_beats"))
        beat_unit = int(meta.get("main_beat_unit"))
    except Exception:
        return None, "missing_meter_or_bpm"
    return {
        "song_id": song_id,
        "split": split,
        "tonic_pc": tonic_pc,
        "mode_name": mode_name,
        "bpm": bpm,
        "num_beats": num_beats,
        "beat_unit": beat_unit,
        "beat_origin": _resolve_beat_origin(song_obj),
    }, None


def _infer_tonal_group(mode_name: str) -> str:
    return "major_family" if mode_name in {"major", "lydian", "mixolydian"} else "minor_family"


def _infer_corruption_group(corruption_name: str) -> str:
    strict_benign = {"transpose_with_tonic_shift", "merge_repeated_melody_notes", "split_long_melody_note"}
    near_benign = {"melody_octave_shift", "drop_tonic_seventh_on_strong_beat"}
    if corruption_name in strict_benign:
        return "strict_benign"
    if corruption_name in near_benign:
        return "near_benign"
    return "other"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _render_midi(song_obj: dict[str, Any], midi_path: Path, theory_ctx: dict[str, Any], octave_id_map: dict[int, int]) -> None:
    pm, _ = render_song_to_pretty_midi(song_obj=song_obj, theory_ctx=theory_ctx, octave_id_to_value=octave_id_map)
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(midi_path))


def _cleanup_artifacts(encoded_path: Path, midi_path: Path) -> None:
    encoded_path.unlink(missing_ok=True)
    midi_path.unlink(missing_ok=True)


def _stable_pair_group_id(
    song_id: str,
    pair_idx: int,
    corruption_modes: list[str],
    theory_cfg: dict[str, Any],
    *,
    forced_mode: str | None = None,
    task_kind: str | None = None,
) -> str:
    cfg_payload = {"m": sorted(corruption_modes), "t": theory_cfg, "forced_mode": forced_mode}
    if task_kind is not None:
        cfg_payload["task_kind"] = task_kind
    cfg_key = json.dumps(
        cfg_payload,
        sort_keys=True,
        ensure_ascii=False,
    )
    cfg_hash = hashlib.sha1(cfg_key.encode("utf-8")).hexdigest()[:10]
    if forced_mode is not None:
        pair_label = f"{forced_mode}-pair-{pair_idx}"
    elif task_kind:
        pair_label = f"{task_kind}-pair-{pair_idx}"
    else:
        pair_label = f"pair-{pair_idx}"
    return f"{song_id}::{pair_label}::{cfg_hash}"


def _split_section_local_modes(corruption_modes: list[str]) -> tuple[list[str], list[str]]:
    section_modes = [mode for mode in corruption_modes if mode in SECTION_CORRUPTION_MODES]
    local_modes = [mode for mode in corruption_modes if mode not in SECTION_CORRUPTION_MODES]
    return section_modes, local_modes


def _build_pair_tasks(
    *,
    pair_mode_strategy: str,
    corruption_modes: list[str],
    pairs_per_song: int,
    section_pairs_per_mode: int,
    local_pairs_per_song: int,
) -> list[tuple[str, str | None, int]]:
    if pair_mode_strategy == "all_modes":
        return [("forced", mode, pair_idx) for mode in corruption_modes for pair_idx in range(pairs_per_song)]
    if pair_mode_strategy == "first_applicable":
        return [("first_applicable", None, pair_idx) for pair_idx in range(pairs_per_song)]
    if pair_mode_strategy == "section_all_local_balanced":
        section_modes, local_modes = _split_section_local_modes(corruption_modes)
        tasks = [
            ("section_forced", mode, pair_idx)
            for mode in section_modes
            for pair_idx in range(section_pairs_per_mode)
        ]
        tasks.extend(("local_balanced", None, pair_idx) for pair_idx in range(local_pairs_per_song) if local_modes)
        return tasks
    raise PairBuildError(f"Unsupported pair_mode_strategy '{pair_mode_strategy}'")


def _seed_balancer_from_counts(balancer: CorruptionModeBalancer, counts: Counter[str]) -> None:
    for mode, count in counts.items():
        for _ in range(max(0, int(count))):
            balancer.record_applied(str(mode))


def _pair_is_complete(pair_row: dict[str, Any], manifest_by_sample_id: dict[str, dict[str, Any]]) -> bool:
    clean_id = str(pair_row.get("clean_sample_id", ""))
    corr_id = str(pair_row.get("corrupted_sample_id", ""))
    clean_row = manifest_by_sample_id.get(clean_id)
    corr_row = manifest_by_sample_id.get(corr_id)
    if clean_row is None or corr_row is None:
        return False
    for row in (clean_row, corr_row):
        if not Path(str(row.get("encoded_song_path", ""))).exists():
            return False
        if not Path(str(row.get("midi_path", ""))).exists():
            return False
    return True


def build_pairs(cfg: DictConfig) -> BuildStats:
    if str(cfg.dataloader.corruption_backend) != "song_theory":
        raise PairBuildError("observer pair builder supports only dataloader.corruption_backend='song_theory'")

    dataset_path = Path(cfg.data.json_path)
    if not dataset_path.is_absolute():
        dataset_path = _base_cwd() / dataset_path
    paths = resolve_observer_pipeline_paths(cfg)
    encoded_root = paths["encoded_root"]
    midi_root = paths["midi_root"]
    manifests_root = paths["manifests_root"]
    index_root = paths["pair_index_root"]
    skipped_log = paths["skipped_log_path"]

    overwrite = bool(cfg.observer_pipeline.get("overwrite", False))
    theory_cfg = OmegaConf.to_container(cfg.dataloader.theory_aware, resolve=True)
    deterministic = bool(theory_cfg.get("deterministic_per_sample", False))
    if not overwrite and not deterministic:
        raise PairBuildError("overwrite=false requires deterministic_per_sample=true")

    split_map = {k: str(v) for k, v in OmegaConf.to_container(cfg.data.split, resolve=True).items()}
    split_lookup = {v: k for k, v in split_map.items()}
    valid_split_names = set(split_map.keys())

    theory_ctx = build_theory_context()
    base_cwd = _base_cwd()
    octave_id_map = load_octave_id_map(base_cwd)
    vocab_scale = json.loads((base_cwd / "metadata" / "vocabs" / "vocab_key_scale.json").read_text(encoding="utf-8"))
    scale_id_to_name = {int(v): k for k, v in vocab_scale.items() if isinstance(v, int)}

    existing_skip_rows = [] if overwrite else _load_jsonl_rows(skipped_log)
    skip_rows: list[dict[str, Any]] = list(existing_skip_rows)
    global_skip_counter = Counter(str(x.get("reason_skipped", "unknown")) for x in existing_skip_rows)

    manifest_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    pair_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    for split_key in split_map:
        manifest_path = manifests_root / f"{split_key}.jsonl"
        pairs_path = index_root / f"{split_key}_pairs.jsonl"
        existing_manifest = [] if overwrite else _load_jsonl_rows(manifest_path)
        existing_pairs = [] if overwrite else _load_jsonl_rows(pairs_path)
        manifest_by_split[split_key] = {str(x["sample_id"]): x for x in existing_manifest}
        pair_by_split[split_key] = {str(x["pair_group_id"]): x for x in existing_pairs}

    # cleanup stale split files/dirs not present in current split config
    if manifests_root.exists():
        for stale_file in manifests_root.glob("*.jsonl"):
            if stale_file.stem not in valid_split_names:
                stale_file.unlink(missing_ok=True)
    if index_root.exists():
        for stale_file in index_root.glob("*_pairs.jsonl"):
            split_name = stale_file.name.replace("_pairs.jsonl", "")
            if split_name not in valid_split_names:
                stale_file.unlink(missing_ok=True)
    for data_root in (encoded_root, midi_root):
        if data_root.exists():
            for child in data_root.iterdir():
                if child.is_dir() and child.name not in valid_split_names:
                    for nested in child.glob("*"):
                        if nested.is_file():
                            nested.unlink(missing_ok=True)
                    child.rmdir()

    dataset = _load_dataset(dataset_path)
    stats = BuildStats(total=len(dataset))
    pairs_per_song = max(1, int(cfg.dataloader.get("pairs_per_song", 1)))
    corruption_modes = list(cfg.dataloader.corruption_modes)
    pair_mode_strategy = str(cfg.dataloader.get("pair_mode_strategy", "first_applicable"))
    if pair_mode_strategy not in PAIR_MODE_STRATEGIES:
        raise PairBuildError(f"dataloader.pair_mode_strategy must be one of {sorted(PAIR_MODE_STRATEGIES)}")
    section_pairs_per_mode = max(1, int(cfg.dataloader.get("section_pairs_per_mode", 1)))
    local_pairs_per_song = int(cfg.dataloader.get("local_pairs_per_song", pairs_per_song))
    if local_pairs_per_song < 0:
        raise PairBuildError("dataloader.local_pairs_per_song must be >= 0")
    section_modes, local_modes = _split_section_local_modes(corruption_modes)
    max_pairs_per_split_per_mode_raw = cfg.dataloader.get("max_pairs_per_split_per_mode")
    max_pairs_per_split_per_mode = (
        int(max_pairs_per_split_per_mode_raw) if max_pairs_per_split_per_mode_raw is not None else None
    )
    max_pairs_per_mode_by_split = dict(cfg.dataloader.get("max_pairs_per_mode_by_split", {}))
    balance_mode_usage = (
        pair_mode_strategy != "section_all_local_balanced"
        and bool(theory_cfg.get("balance_mode_usage", False))
        and not deterministic
        and len(corruption_modes) > 1
    )
    mode_balancer = CorruptionModeBalancer(corruption_modes) if balance_mode_usage else None

    # validate meta exactly once per row
    valid_rows: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for idx, song_obj in enumerate(dataset):
        meta, reason = _resolve_meta(song_obj, scale_id_to_name)
        if meta is None:
            stats.skipped_rows += 1
            skip_rows.append({"index": idx, "reason_skipped": reason})
            global_skip_counter[str(reason)] += 1
            continue
        split_key = split_lookup.get(meta["split"])
        if split_key is None:
            continue
        valid_rows.append((song_obj, meta, idx))

    for split_key in split_map:
        split_skip_counter: Counter[str] = Counter()
        manifest_by_sample_id = manifest_by_split[split_key]
        pair_by_group_id = pair_by_split[split_key]
        split_mode_counter: Counter[str] = Counter(str(row.get("corruption_name", "")) for row in pair_by_group_id.values())
        split_local_balancer = CorruptionModeBalancer(local_modes) if pair_mode_strategy == "section_all_local_balanced" and local_modes else None
        if split_local_balancer is not None:
            _seed_balancer_from_counts(split_local_balancer, Counter({mode: split_mode_counter.get(mode, 0) for mode in local_modes}))
        split_max_pairs_per_mode = (
            int(max_pairs_per_mode_by_split[split_key])
            if split_key in max_pairs_per_mode_by_split and max_pairs_per_mode_by_split[split_key] is not None
            else max_pairs_per_split_per_mode
        )

        for song_obj, meta, idx in valid_rows:
            if split_lookup.get(meta["split"]) != split_key:
                continue

            pair_tasks = _build_pair_tasks(
                pair_mode_strategy=pair_mode_strategy,
                corruption_modes=corruption_modes,
                pairs_per_song=pairs_per_song,
                section_pairs_per_mode=section_pairs_per_mode,
                local_pairs_per_song=local_pairs_per_song,
            )

            for task_kind, forced_mode, pair_idx in pair_tasks:
                if (
                    forced_mode is not None
                    and split_max_pairs_per_mode is not None
                    and split_mode_counter[forced_mode] >= split_max_pairs_per_mode
                ):
                    continue

                pair_group_id = _stable_pair_group_id(
                    meta["song_id"],
                    pair_idx,
                    corruption_modes,
                    theory_cfg,
                    forced_mode=forced_mode,
                    task_kind=task_kind if pair_mode_strategy == "section_all_local_balanced" else None,
                )
                existing_pair = pair_by_group_id.get(pair_group_id)
                if existing_pair is not None and not overwrite:
                    if _pair_is_complete(existing_pair, manifest_by_sample_id):
                        continue
                    # stale/incomplete -> force rebuild
                    old_clean = manifest_by_sample_id.pop(existing_pair.get("clean_sample_id", ""), None)
                    old_corr = manifest_by_sample_id.pop(existing_pair.get("corrupted_sample_id", ""), None)
                    for old_row in (old_clean, old_corr):
                        if old_row is None:
                            continue
                        _cleanup_artifacts(Path(str(old_row.get("encoded_song_path", ""))), Path(str(old_row.get("midi_path", ""))))
                    pair_by_group_id.pop(pair_group_id, None)

                rng = random
                if deterministic:
                    stable_song_seed = int(hashlib.sha1(meta["song_id"].encode("utf-8")).hexdigest()[:8], 16)
                    seed = int(theory_cfg.get("deterministic_seed", 0)) + stable_song_seed + pair_idx
                    rng = random.Random(seed)

                if task_kind == "local_balanced":
                    requested_modes = split_local_balancer.ordered_modes(rng) if split_local_balancer is not None else list(local_modes)
                    shuffle_modes = False
                else:
                    requested_modes = [forced_mode] if forced_mode is not None else corruption_modes
                    shuffle_modes = forced_mode is None
                if mode_balancer is not None and forced_mode is None:
                    requested_modes = mode_balancer.ordered_modes(rng)
                    shuffle_modes = False
                corrupted_song, corr_meta = corrupt_song_obj(
                    song_obj,
                    corruption_modes=requested_modes,
                    corruption_cfg=theory_cfg,
                    theory_ctx=theory_ctx,
                    rng=rng,
                    shuffle_modes=shuffle_modes,
                )
                corr_meta = corr_meta or {}
                corr_name = str(corr_meta.get("corruption_name", "identity"))
                corr_applied = bool(corr_meta.get("applied", False)) and corr_name != "identity"
                if not corr_applied:
                    stats.skipped_rows += 1
                    reason_skip = f"corruption_not_applied:{corr_meta.get('reason_skipped', 'unknown')}"
                    split_skip_counter[reason_skip] += 1
                    skip_rows.append({"source_song_id": meta["song_id"], "pair_group_id": pair_group_id, "split": split_key, "reason_skipped": reason_skip})
                    global_skip_counter[reason_skip] += 1
                    continue
                if task_kind == "local_balanced" and split_local_balancer is not None and corr_name in local_modes:
                    split_local_balancer.record_applied(corr_name)
                if mode_balancer is not None and forced_mode is None:
                    mode_balancer.record_applied(corr_name)

                candidate_rows: list[dict[str, Any]] = []
                clean_written = False
                corr_written = False
                had_root_failure = False

                for is_corrupted, payload, local_meta in [
                    (False, song_obj, {"corruption_name": "identity", "corruption_params": {}, "topology_changed": False, "note_corrupted_indices": [], "chord_corrupted_indices": [], "onset_corrupted_indices": []}),
                    (True, corrupted_song, corr_meta),
                ]:
                    suffix = "corrupted" if is_corrupted else "clean"
                    sample_id = f"{pair_group_id}::{suffix}"
                    encoded_path = encoded_root / split_key / f"{sample_id}.json"
                    midi_path = midi_root / split_key / f"{sample_id}.mid"
                    try:
                        _write_json(encoded_path, payload)
                        _render_midi(payload, midi_path, theory_ctx, octave_id_map)
                    except Exception as exc:  # noqa: BLE001
                        _cleanup_artifacts(encoded_path, midi_path)
                        stats.skipped_rows += 1
                        had_root_failure = True
                        reason_skip = f"render_failed:{type(exc).__name__}"
                        split_skip_counter[reason_skip] += 1
                        skip_rows.append({"sample_id": sample_id, "pair_group_id": pair_group_id, "split": split_key, "reason_skipped": f"render_failed:{exc}"})
                        global_skip_counter[reason_skip] += 1
                        if not bool(cfg.observer_pipeline.get("skip_render_failures", True)):
                            raise
                        continue

                    row = {
                        "sample_id": sample_id,
                        "song_id": sample_id,
                        "source_song_id": meta["song_id"],
                        "pair_group_id": pair_group_id,
                        "split": split_key,
                        "is_corrupted": is_corrupted,
                        "corruption_name": str(local_meta.get("corruption_name", "identity")),
                        "midi_path": str(midi_path),
                        "encoded_song_path": str(encoded_path),
                        "tonic_pc": int(meta["tonic_pc"]),
                        "mode_name": meta["mode_name"],
                        "bpm": float(meta["bpm"]),
                        "num_beats": int(meta["num_beats"]),
                        "beat_unit": int(meta["beat_unit"]),
                        "beat_origin": float(meta["beat_origin"]),
                        "tonal_group": _infer_tonal_group(meta["mode_name"]),
                        "corruption_group": _infer_corruption_group(str(local_meta.get("corruption_name", "identity"))),
                        "corruption_params": local_meta.get("corruption_params", {}),
                        "topology_changed": bool(local_meta.get("topology_changed", False)),
                        "note_corrupted_indices": local_meta.get("note_corrupted_indices", []),
                        "chord_corrupted_indices": local_meta.get("chord_corrupted_indices", []),
                        "onset_corrupted_indices": local_meta.get("onset_corrupted_indices", []),
                        "attempted_corruption_modes": local_meta.get("attempted_corruption_modes", []),
                        "skipped_corruption_attempts": local_meta.get("skipped_corruption_attempts", []),
                    }
                    candidate_rows.append(row)
                    clean_written = clean_written or not is_corrupted
                    corr_written = corr_written or is_corrupted

                if clean_written and corr_written and len(candidate_rows) == 2:
                    for row in candidate_rows:
                        manifest_by_sample_id[row["sample_id"]] = row
                    pair_by_group_id[pair_group_id] = {
                        "pair_group_id": pair_group_id,
                        "split": split_key,
                        "source_song_id": meta["song_id"],
                        "clean_sample_id": f"{pair_group_id}::clean",
                        "corrupted_sample_id": f"{pair_group_id}::corrupted",
                        "corruption_name": corr_name,
                        "tonal_group": _infer_tonal_group(meta["mode_name"]),
                        "corruption_group": _infer_corruption_group(corr_name),
                        "topology_changed": bool(corr_meta.get("topology_changed", False)),
                        "attempted_corruption_modes": corr_meta.get("attempted_corruption_modes", []),
                        "skipped_corruption_attempts": corr_meta.get("skipped_corruption_attempts", []),
                        "is_valid_pair_for_rank": True,
                    }
                    stats.built_pairs += 1
                    if forced_mode is not None:
                        split_mode_counter[forced_mode] += 1
                    else:
                        split_mode_counter[corr_name] += 1
                else:
                    for row in candidate_rows:
                        _cleanup_artifacts(Path(row["encoded_song_path"]), Path(row["midi_path"]))
                    if not had_root_failure:
                        stats.skipped_rows += 1
                        reason_skip = "pair_incomplete"
                        split_skip_counter[reason_skip] += 1
                        skip_rows.append({"pair_group_id": pair_group_id, "split": split_key, "source_song_id": meta["song_id"], "reason_skipped": reason_skip})
                        global_skip_counter[reason_skip] += 1

        manifest_rows_sorted = sorted(manifest_by_sample_id.values(), key=lambda x: str(x["sample_id"]))
        pair_rows_sorted = sorted(pair_by_group_id.values(), key=lambda x: str(x["pair_group_id"]))
        _write_jsonl(manifests_root / f"{split_key}.jsonl", manifest_rows_sorted)
        _write_jsonl(index_root / f"{split_key}_pairs.jsonl", pair_rows_sorted)
        LOGGER.info("Pair build split=%s built=%d skip_top=%s", split_key, len(pair_by_group_id), dict(split_skip_counter.most_common(5)))
        LOGGER.info(
            "Pair build split=%s corruption_counts=%s",
            split_key,
            dict(Counter(str(row.get("corruption_name", "")) for row in pair_rows_sorted).most_common()),
        )

    _write_jsonl(skipped_log, _dedup_skip_rows(skip_rows))
    LOGGER.info("Pair build totals: total=%d built_pairs=%d skipped=%d", stats.total, stats.built_pairs, stats.skipped_rows)
    LOGGER.info("Pair build global skip reasons: %s", dict(global_skip_counter.most_common(10)))
    return stats


@hydra.main(version_base=None, config_path="../../configs", config_name="observer_distill")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    build_pairs(cfg)


if __name__ == "__main__":
    main()
