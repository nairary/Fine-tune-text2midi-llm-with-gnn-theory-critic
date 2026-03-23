from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader

from src.dataloader.graph_layouts import NODE_DIMS
from src.dataloader.hooktheory_dataset import HookTheoryDataset, collate_fn


def main():
    dataset = HookTheoryDataset("data/HTCanon/encoded_full/teacher_encoded.json")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    batch = next(iter(loader))
    graph = batch["graph_real"]

    print(graph)
    print("node types:", graph.node_types)
    print("edge types:", graph.edge_types)

    for node_type in ["song", "bar", "onset", "note", "chord"]:
        print(f"{node_type}.x shape = {tuple(graph[node_type].x.shape)} | expected dim={NODE_DIMS[node_type]}")

    for edge_type in graph.edge_types:
        print(f"{edge_type}: {tuple(graph[edge_type].edge_index.shape)}")

    print("masked label keys:", batch["masked_labels"][0].keys())
    print("mask note fields:", batch["masked_labels"][0]["note"]["field_names"])
    print("mask chord fields:", batch["masked_labels"][0]["chord"]["field_names"])


if __name__ == "__main__":
    main()
