from torch.utils.data import DataLoader

from src.dataloader.graph_layouts import NODE_DIMS
from src.dataloader.hooktheory_dataset import HookTheoryDataset, collate_fn


dataset = HookTheoryDataset("data/HTCanon/encoded_full/teacher_encoded.json")
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

batch = next(iter(loader))
graph = batch["graph_real"]

print(graph)
for node_type in ["song", "bar", "onset", "note", "chord"]:
    print(f"{node_type}.x shape = {tuple(graph[node_type].x.shape)} | expected dim={NODE_DIMS[node_type]}")
for edge_type in graph.edge_types:
    print(f"{edge_type}: {tuple(graph[edge_type].edge_index.shape)}")
print("covers_note edges:", graph[("chord", "covers_note", "note")].edge_index.shape)
print(batch["masked_labels"][0].keys())
