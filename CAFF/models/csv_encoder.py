import torch
import torch.nn as nn
from typing import List


class CSVEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, retained_relation_ids: List[torch.Tensor], rel_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = len(retained_relation_ids)
        device = rel_embeddings.device
        d = rel_embeddings.size(1)
        z = torch.zeros(batch_size, d, device=device)
        for i, rel_ids in enumerate(retained_relation_ids):
            if rel_ids.numel() == 0:
                continue
            selected = rel_embeddings[rel_ids]
            z[i] = selected.mean(dim=0)
        return z
