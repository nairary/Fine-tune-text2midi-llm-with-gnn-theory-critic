from torch.utils.data import DataLoader
from src.dataloader.hooktheory_dataset import HookTheoryDataset, collate_fn

dataset = HookTheoryDataset("data/HTCanon/HK_processed/encoded_full/teacher_encoded.json")
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

for batch in loader:
    print(batch["graph_real"])
    print(batch["graph_masked"])
    print(batch["graph_corrupted"])
    print(batch["graph_real"]["note"].x.shape)
    print(batch["graph_real"]["chord"].x.shape)
    print(batch["graph_real"]["note", "next_note", "note"].edge_index.shape)
    print(batch["graph_real"]["chord", "next_chord", "chord"].edge_index.shape)
    print(batch["masked_labels"][0].keys())
    break