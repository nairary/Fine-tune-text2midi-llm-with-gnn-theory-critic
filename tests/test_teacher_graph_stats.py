import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader.utils_graph import build_graph_from_encoded


EDGE_TYPES_TO_REPORT = [
    ("song", "contains_bar", "bar"),
    ("bar", "next_bar", "bar"),
    ("bar", "contains_onset", "onset"),
    ("onset", "next_onset", "onset"),
    ("onset", "starts_note", "note"),
    ("onset", "starts_chord", "chord"),
    ("note", "next_note", "note"),
    ("chord", "next_chord", "chord"),
    ("chord", "covers_note", "note"),
]


def iter_songs(json_path: str, limit: int = 5):
    with open(Path(json_path), "r", encoding="utf-8") as f:
        raw = json.load(f)
    songs = list(raw.values()) if isinstance(raw, dict) else list(raw)
    return songs[:limit]


def main():
    songs = iter_songs("data/HTCanon/encoded_full/teacher_encoded.json", limit=5)
    print(f"n_songs_checked={len(songs)}")

    total_covers_note_edges = 0
    total_notes = 0
    total_chords = 0

    for idx, song_obj in enumerate(songs):
        graph = build_graph_from_encoded(song_obj)
        print(f"\n=== graph[{idx}] song_id={song_obj.get('song_id')} ===")
        for node_type in ["song", "bar", "onset", "note", "chord"]:
            print(f"nodes[{node_type}]={graph[node_type].x.size(0)}")
        for edge_type in EDGE_TYPES_TO_REPORT:
            print(f"edges[{edge_type}]={graph[edge_type].edge_index.size(1)}")

        total_covers_note_edges += graph[("chord", "covers_note", "note")].edge_index.size(1)
        total_notes += graph["note"].x.size(0)
        total_chords += graph["chord"].x.size(0)

    mean_covers_per_note = total_covers_note_edges / max(total_notes, 1)
    mean_covers_per_chord = total_covers_note_edges / max(total_chords, 1)
    print("\n=== aggregate ===")
    print(f"total_covers_note_edges={total_covers_note_edges}")
    print(f"mean_covers_per_note={mean_covers_per_note:.4f}")
    print(f"mean_covers_per_chord={mean_covers_per_chord:.4f}")


if __name__ == "__main__":
    main()
