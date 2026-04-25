from typing import List
import torch
from transformers import AutoModel, AutoTokenizer


class BioLinkBERTEncoder:
    def __init__(self, model_name: str, device: str = 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, texts: List[str], max_length: int = 128, normalize: bool = True) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / lengths
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled
