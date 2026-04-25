import torch
import torch.nn as nn
from typing import List, Optional
from .encoder_utils import BioLinkBERTEncoder
from .csv_encoder import CSVEncoder
from .dbm import DBM


class CAFFModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        d_model: int,
        rho: int,
        L: int,
        num_relations: int,
        device: str = 'cpu',
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.d_model = self.encoder.config.hidden_size  # Enforce actual d from encoder
        self.rho = rho
        self.L = L  # Depth-stratified inference L=3
        self.num_relations = num_relations
        self.device = device

        # Contextual Summary Vector (CSV) - equation (13)
        self.csv_encoder = ContextualSummaryVector(self.d_model, rho)

        # Dynamic Bilinear Modulation (DBM) - equation (15)
        self.dbm = DynamicBilinearModulation(self.d_model, rho, L, num_relations)

        # Relation embeddings - frozen BioLinkBERT (enforce d=768)
        self.rel_embeddings = None

        self.register_buffer('rel_embeddings', torch.zeros(num_relations, d_model))

    def set_relation_embeddings(self, relation_texts: List[str]):
        embeddings = self.encoder.encode(relation_texts, max_length=128)
        self.rel_embeddings = embeddings.detach().clone()

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        return self.encoder.encode(queries, max_length=512)

    def score_candidates(
        self,
        q: torch.Tensor,
        rel_ids: torch.Tensor,
        z_prev: torch.Tensor,
        ell: int,
    ) -> torch.Tensor:
        batch_size, max_cands = rel_ids.shape
        pad_mask = rel_ids == -1
        rel_ids_safe = rel_ids.clone()
        rel_ids_safe[pad_mask] = 0
        e_r = self.rel_embeddings[rel_ids_safe]
        W_ctx = self.W0[ell] + self.dbms[ell](z_prev)
        W_ctx = W_ctx.unsqueeze(0).expand(batch_size, -1, -1)
        qW = torch.bmm(q.unsqueeze(1), W_ctx)
        logits_bilin = torch.bmm(qW, e_r.transpose(1, 2)).squeeze(1)
        q_expanded = q.unsqueeze(1).expand(-1, max_cands, -1)
        interaction = (q_expanded * e_r).matmul(self.v)
        logits = logits_bilin + interaction + self.beta[ell]
        logits = logits.masked_fill(pad_mask, -1e9)
        return logits

    def forward(
        self,
        queries: List[str],
        candidate_rel_ids: List[torch.Tensor],
        retained_relation_ids: Optional[List[List[torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        batch_size = len(queries)
        q = self.encode_queries(queries)
        z_prev = torch.zeros(batch_size, self.d_model, device=self.device)
        logits_by_hop = []

        for ell in range(self.L):
            logits = self.score_candidates(q, candidate_rel_ids[ell], z_prev, ell)
            logits_by_hop.append(logits)
            if retained_relation_ids is not None and ell < len(retained_relation_ids):
                z_prev = self.csv_encoder(retained_relation_ids[ell], self.rel_embeddings)
            else:
                z_prev = torch.zeros(batch_size, self.d_model, device=self.device)

        return logits_by_hop

    def score_relation_with_context(
        self,
        q: torch.Tensor,
        rel_id: int,
        z_prev: torch.Tensor,
        ell: int,
    ) -> torch.Tensor:
        e_r = self.rel_embeddings[rel_id].unsqueeze(0)
        W_ctx = self.W0.unsqueeze(0) + (self.A[ell] @ self.B[ell].T).unsqueeze(0) + self.dbms[ell](z_prev)
        qW = torch.bmm(q.unsqueeze(1), W_ctx)
        logits_bilin = torch.bmm(qW, e_r.unsqueeze(-1)).squeeze(1).squeeze(1)
        interaction = (q * e_r).matmul(self.v)
        return logits_bilin + interaction + self.beta[ell]
