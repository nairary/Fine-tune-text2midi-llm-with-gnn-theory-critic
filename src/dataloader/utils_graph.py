# src/dataloader/utils_graph.py
import torch
from torch_geometric.data import HeteroData
import copy
import random

def build_graph_from_encoded(song_obj):
    """
    Превращает teacher_encoded song в HeteroData graph
    """
    g = HeteroData()

    # === note nodes ===
    notes = song_obj.get('melody', [])
    if notes:
        g['note'].x = torch.tensor([[n['sd_id'], n['octave_id'], n['is_rest'], n['beat'], n['duration']] for n in notes], dtype=torch.float)

    # === chord nodes ===
    chords = song_obj.get('chords', [])
    if chords:
        g['chord'].x = torch.tensor(
            [
                [
                    c['root_id'],
                    c['type_id'],
                    c['inversion_id'],
                    c['applied_id'],
                    c['borrowed_kind_id'],
                    c['borrowed_mode_name_id'],
                ]
                for c in chords
            ],
            dtype=torch.float,
        )

    # TODO: добавить multi-hot vectors (adds_vec, omits_vec, suspensions_vec, alterations_vec, borrowed_pcset_vec)
    # Их можно конкатенировать к chord.x

    # === edges ===
    # Простейшие связи note->note, chord->chord, chord->note if onset alignment, и т.д.
    # Для начала можно просто последовательные:
    if notes:
        src = torch.arange(len(notes)-1)
        dst = torch.arange(1, len(notes))
        g['note', 'next_note', 'note'].edge_index = torch.stack([src, dst], dim=0)

    if chords:
        src = torch.arange(len(chords)-1)
        dst = torch.arange(1, len(chords))
        g['chord', 'next_chord', 'chord'].edge_index = torch.stack([src, dst], dim=0)

    return g

def mask_graph(graph: HeteroData, mask_prob=0.15):
    """
    Маскируем 15% узлов note/chord для SSL
    """
    masked_labels = {}
    g_masked = copy.deepcopy(graph)

    for ntype in ['note', 'chord']:
        if ntype in g_masked.node_types:
            x = g_masked[ntype].x.clone()
            num_nodes = x.size(0)
            num_mask = max(1, int(num_nodes * mask_prob))
            mask_idx = random.sample(range(num_nodes), num_mask)
            masked_labels[ntype] = x[mask_idx].clone()
            x[mask_idx] = 0  # mask value
            g_masked[ntype].x = x

    return g_masked, masked_labels

def corrupt_graph(graph: HeteroData):
    """
    Коррупция графа для real-vs-corrupted discrimination
    """
    g_corrupted = copy.deepcopy(graph)

    # примеры corruption:
    if 'note' in g_corrupted.node_types:
        x = g_corrupted['note'].x
        # случайно поменять sd_id
        if x.size(0) > 1:
            idx = random.randint(0, x.size(0)-1)
            x[idx,0] = random.randint(1, 7)  # допустимые sd
            g_corrupted['note'].x = x

    if 'chord' in g_corrupted.node_types:
        x = g_corrupted['chord'].x
        if x.size(0) > 1:
            idx = random.randint(0, x.size(0)-1)
            x[idx,0] = random.randint(0, 7)  # root_id
            g_corrupted['chord'].x = x

    return g_corrupted

def extract_masked_labels(real_graph, masked_graph):
    """
    возвращает target для masked nodes
    """
    masked_labels = {}
    for ntype in ['note', 'chord']:
        if ntype in masked_graph.node_types:
            diff = masked_graph[ntype].x != real_graph[ntype].x
            idx = diff.sum(dim=1) > 0
            masked_labels[ntype] = real_graph[ntype].x[idx]
    return masked_labels