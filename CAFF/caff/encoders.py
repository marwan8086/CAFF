"""
caff/encoders.py
================
Frozen BioLinkBERT-Large encoder (paper §8.4).

Paper §8.4: "The relation-scoring encoder is BioLinkBERT-Large
(d=768, 340M parameters) [...] frozen throughout CAFF training."

This module provides:
  • A frozen wrapper around the HuggingFace model
  • Mean-pool + L2 normalization per paper §8.4
  • A relation-embedding cache (Smart Engineering S1):
      since relations are encoded once and never change,
      we precompute all |R| relation embeddings at startup
      and never re-encode during training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class FrozenBioEncoder(nn.Module):
    """Frozen BioLinkBERT-Large with mean-pool + L2 norm output.

    Per paper §8.4, this encoder is NEVER updated during training.
    All parameters have requires_grad=False, asserted at construction.

    Output: ℓ2-normalized 768-d embedding (q or e_r in the paper).
    """

    def __init__(
        self,
        model_name: str = "michiyasunaga/BioLinkBERT-large",
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        logger.info(f"Loading frozen encoder: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Freeze every parameter — paper invariant
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(device)
        self.device = device

        # Sanity check: paper §8.4 says "no trainable encoder params"
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        assert n_trainable == 0, (
            f"Encoder has {n_trainable} trainable params — paper §8.4 "
            f"requires fully frozen encoder."
        )

        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Encoder loaded: {n_total / 1e6:.1f}M params (all frozen)")

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Encode a batch of strings to ℓ2-normalized embeddings.

        Parameters
        ----------
        texts : list of str
            Query strings or relation surface forms.
        batch_size : int
            Internal batching for memory.

        Returns
        -------
        Tensor of shape (len(texts), d) with ||row||_2 == 1.
        """
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**tok)
            # Mean-pool over tokens, masked by attention
            mask = tok["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed / counts
            normalized = F.normalize(mean_pooled, p=2, dim=-1)
            all_embeds.append(normalized.cpu())
        return torch.cat(all_embeds, dim=0)


class RelationEmbeddingCache:
    """Smart Engineering S1: cache all |R| relation embeddings on GPU.

    Encoding is deterministic and frozen, so we encode once at
    startup and reuse across all queries, all hops, all epochs.
    Saves O(|E|) re-encoding cost (millions of triples) per epoch.
    """

    def __init__(
        self,
        encoder: FrozenBioEncoder,
        relations: list[str],
        cache_path: Path | None = None,
    ) -> None:
        self.relations = relations
        self.relation_to_idx = {r: i for i, r in enumerate(relations)}

        if cache_path is not None and cache_path.exists():
            logger.info(f"Loading relation embedding cache from {cache_path}")
            self.embeddings = torch.load(cache_path)
            assert self.embeddings.shape[0] == len(relations), (
                f"Cache size mismatch: {self.embeddings.shape[0]} vs "
                f"{len(relations)} relations"
            )
        else:
            logger.info(f"Encoding {len(relations)} relation strings...")
            self.embeddings = encoder.encode(relations)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.embeddings, cache_path)
                logger.info(f"Saved relation embedding cache to {cache_path}")

        # Move to encoder device for fast lookup during training
        self.embeddings = self.embeddings.to(encoder.device)

    def __len__(self) -> int:
        return len(self.relations)

    def get(self, relation: str) -> torch.Tensor:
        """Look up a single relation's frozen embedding."""
        return self.embeddings[self.relation_to_idx[relation]]

    def get_batch(self, relations: list[str]) -> torch.Tensor:
        """Look up a batch of relations. Returns (B, d) tensor."""
        idx = torch.tensor(
            [self.relation_to_idx[r] for r in relations],
            device=self.embeddings.device,
        )
        return self.embeddings[idx]