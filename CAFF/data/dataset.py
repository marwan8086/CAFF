import json
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from data.bfs_extractor import build_query_dataset


class CAFFDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        kg_path: str,
        encoder_name: str,
        max_relation_length: int = 128,
        max_query_length: int = 512,
        L: int = 3,
        Kr: int = 20,
        device: str = 'cpu',
    ):
        self.device = device
        self.L = L
        self.Kr = Kr
        self.max_query_length = max_query_length
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if not self.data or not self._is_preprocessed(self.data[0]):
            self.data = build_query_dataset(self.data, kg_path, L, Kr)

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)

        self.relation_texts, self.relation_to_id = self._load_relation_vocab(kg_path)
        self.num_relations = len(self.relation_texts)
        self.rel_embeddings = None

    def _load_relation_vocab(self, kg_path: str) -> Tuple[List[str], Dict[str, int]]:
        rel_set = set()
        with open(kg_path, 'r', encoding='utf-8') as f:
            for line in f:
                _, relation, _ = line.strip().split('\t')
                rel_set.add(relation)
        relation_texts = sorted(rel_set)
        relation_to_id = {rel: idx for idx, rel in enumerate(relation_texts)}
        return relation_texts, relation_to_id

    def _is_preprocessed(self, item: Dict) -> bool:
        if 'hop_candidates' not in item or 'hop_labels' not in item:
            return False
        if not isinstance(item['hop_candidates'], list) or not isinstance(item['hop_labels'], list):
            return False
        for hop in item['hop_candidates']:
            if hop and not all(isinstance(triple, (list, tuple)) and len(triple) == 3 for triple in hop):
                return False
        for hop in item['hop_labels']:
            if hop and not all(isinstance(label, (int, float)) for label in hop):
                return False
        return True

    def set_relation_embeddings(self, rel_embeddings: torch.Tensor):
        self.rel_embeddings = rel_embeddings.to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        item = self.data[index]
        query = item['query']
        candidate_ids = []
        label_vectors = []
        candidate_masks = []

        for hop in range(self.L):
            if hop < len(item['hop_candidates']):
                triples = item['hop_candidates'][hop]
                labels = item['hop_labels'][hop]
            else:
                triples = []
                labels = []

            rel_ids = [self.relation_to_id[triple[1]] for triple in triples]
            candidate_ids.append(torch.tensor(rel_ids, dtype=torch.long))
            label_vectors.append(torch.tensor(labels, dtype=torch.float))
            candidate_masks.append(torch.ones(len(rel_ids), dtype=torch.bool))

        # Quality filter: skip items with no candidates at all
        if sum(len(h) for h in candidate_ids) == 0:
            return None

        return {
            'query': query,
            'candidate_rel_ids': candidate_ids,
            'labels': label_vectors,
            'cand_masks': candidate_masks,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    # Filter out None items (quality filter)
    batch = [b for b in batch if b is not None]
    if not batch:
        # Return empty batch if all were filtered
        return {
            'queries': [],
            'candidate_rel_ids': [],
            'labels': [],
            'masks': [],
        }

    batch_size = len(batch)
    L = len(batch[0]['candidate_rel_ids'])
    max_lengths = [max(item['candidate_rel_ids'][hop].size(0) for item in batch) for hop in range(L)]

    padded_rel_ids = []
    padded_labels = []
    padded_masks = []

    for hop in range(L):
        hop_rel_ids = []
        hop_labels = []
        hop_masks = []
        for item in batch:
            rel_ids = item['candidate_rel_ids'][hop]
            labels = item['labels'][hop]
            mask = item['cand_masks'][hop]
            pad_length = max_lengths[hop] - rel_ids.size(0)
            if pad_length > 0:
                rel_ids = torch.cat([rel_ids, torch.full((pad_length,), -1, dtype=torch.long)])
                labels = torch.cat([labels, torch.zeros(pad_length, dtype=torch.float)])
                mask = torch.cat([mask, torch.zeros(pad_length, dtype=torch.bool)])
            hop_rel_ids.append(rel_ids)
            hop_labels.append(labels)
            hop_masks.append(mask)

        padded_rel_ids.append(torch.stack(hop_rel_ids))
        padded_labels.append(torch.stack(hop_labels))
        padded_masks.append(torch.stack(hop_masks))

    return {
        'queries': [item['query'] for item in batch],
        'candidate_rel_ids': padded_rel_ids,
        'labels': padded_labels,
        'masks': padded_masks,
    }
