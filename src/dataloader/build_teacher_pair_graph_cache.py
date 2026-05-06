from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.dataloader.utils_graph import build_graph_from_encoded

LOGGER = logging.getLogger(__name__)


@dataclass
class CacheStats:
    split: str
    total: int = 0
    built: int = 0
    reused: int = 0
    skipped: int = 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def _graph_filename(sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.pt"


def _load_encoded_song(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected encoded song object at {path}")
    return payload


def _build_split_cache(
    *,
    split: str,
    manifest_path: Path,
    graph_dir: Path,
    index_path: Path,
    base_dir: Path,
    overwrite: bool,
    skip_failures: bool,
) -> CacheStats:
    stats = CacheStats(split=split)
    manifest_rows = _read_jsonl(manifest_path)
    stats.total = len(manifest_rows)

    index_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            stats.skipped += 1
            LOGGER.warning("split=%s skipped row without sample_id", split)
            continue

        encoded_path = _resolve_path(row["encoded_song_path"], base_dir)
        graph_path = graph_dir / split / _graph_filename(sample_id)
        if graph_path.exists() and not overwrite:
            stats.reused += 1
        else:
            try:
                song_obj = _load_encoded_song(encoded_path)
                graph = build_graph_from_encoded(song_obj)
                graph.sample_id = sample_id
                graph.source_song_id = str(row.get("source_song_id", ""))
                graph.pair_group_id = str(row.get("pair_group_id", ""))
                graph.is_corrupted = bool(row.get("is_corrupted", False))
                graph.corruption_name = str(row.get("corruption_name", "identity"))
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(graph, graph_path)
                stats.built += 1
            except Exception as exc:  # noqa: BLE001
                stats.skipped += 1
                LOGGER.warning("split=%s sample_id=%s skipped teacher graph build: %s", split, sample_id, exc)
                if not skip_failures:
                    raise
                continue

        out_row = dict(row)
        out_row["graph_path"] = str(graph_path)
        index_rows.append(out_row)

    _write_jsonl(index_path, sorted(index_rows, key=lambda item: str(item["sample_id"])))
    return stats


def build_teacher_pair_graph_cache(
    *,
    pair_corpus_root: Path,
    manifest_dir: str = "pairs/manifests",
    graph_cache_dir: str = "teacher_graphs",
    splits: list[str] | None = None,
    base_dir: Path | None = None,
    overwrite: bool = False,
    skip_failures: bool = True,
) -> list[CacheStats]:
    base_dir = base_dir or Path.cwd()
    manifest_root = pair_corpus_root / manifest_dir
    graph_root = pair_corpus_root / graph_cache_dir
    graph_dir = graph_root / "graphs"
    index_dir = graph_root / "index"

    if splits is None:
        splits = sorted(path.stem for path in manifest_root.glob("*.jsonl"))
    if not splits:
        raise FileNotFoundError(f"No split manifests found under {manifest_root}")

    stats: list[CacheStats] = []
    for split in splits:
        manifest_path = manifest_root / f"{split}.jsonl"
        if not manifest_path.exists():
            LOGGER.warning("split=%s skipped: missing manifest %s", split, manifest_path)
            continue
        split_stats = _build_split_cache(
            split=split,
            manifest_path=manifest_path,
            graph_dir=graph_dir,
            index_path=index_dir / f"{split}.jsonl",
            base_dir=base_dir,
            overwrite=overwrite,
            skip_failures=skip_failures,
        )
        stats.append(split_stats)
        LOGGER.info(
            "Teacher graph cache split=%s total=%d built=%d reused=%d skipped=%d",
            split_stats.split,
            split_stats.total,
            split_stats.built,
            split_stats.reused,
            split_stats.skipped,
        )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached TeacherGNN HeteroData graphs from a clean/corrupted pair corpus.")
    parser.add_argument("--pair-corpus-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", default="pairs/manifests")
    parser.add_argument("--graph-cache-dir", default="teacher_graphs")
    parser.add_argument("--split", action="append", dest="splits", help="Split to build. Can be passed multiple times.")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Base dir for relative encoded_song_path values.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true", help="Raise on first graph build failure.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    build_teacher_pair_graph_cache(
        pair_corpus_root=args.pair_corpus_root,
        manifest_dir=args.manifest_dir,
        graph_cache_dir=args.graph_cache_dir,
        splits=list(args.splits) if args.splits else None,
        base_dir=args.base_dir,
        overwrite=bool(args.overwrite),
        skip_failures=not bool(args.fail_fast),
    )


if __name__ == "__main__":
    main()
