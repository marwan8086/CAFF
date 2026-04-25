import torch
from typing import List, Tuple


def hc3_loss(
    model,
    queries: List[str],
    candidate_rel_ids: List[torch.Tensor],
    labels_by_hop: List[torch.Tensor],
    masks_by_hop: List[torch.Tensor],
    retained_relation_ids: List[List[torch.Tensor]],
    gamma_C: float = 0.25,
    max_pairs: int = 256,
) -> torch.Tensor:
    batch_size = len(queries)
    device = candidate_rel_ids[0].device
    q_embed = model.encode_queries(queries)
    z_prev_by_hop = []
    z_prev_by_hop.append(torch.zeros(batch_size, model.d_model, device=device))
    for ell in range(1, model.L):
        z_prev = model.csv_encoder(retained_relation_ids[ell - 1], model.rel_embeddings)
        z_prev_by_hop.append(z_prev)

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for ell in range(model.L):
        relation_groups = {}
        rel_ids = candidate_rel_ids[ell]
        labels = labels_by_hop[ell]
        masks = masks_by_hop[ell]
        for i in range(batch_size):
            for pos in range(rel_ids.size(1)):
                if not masks[i, pos].item():
                    continue
                rid = rel_ids[i, pos].item()
                if rid < 0:
                    continue
                relation_groups.setdefault(rid, []).append((i, labels[i, pos].item()))

        for rid, items in relation_groups.items():
            positives = [i for i, lab in items if lab == 1.0]
            negatives = [i for i, lab in items if lab == 0.0]
            if not positives or not negatives:
                continue
            pair_count = min(len(positives) * len(negatives), max_pairs)
            for k in range(pair_count):
                pos_idx = positives[k % len(positives)]
                neg_idx = negatives[k % len(negatives)]
                s_pos = model.score_relation_with_context(
                    q_embed[pos_idx : pos_idx + 1], rid, z_prev_by_hop[ell][pos_idx : pos_idx + 1], ell
                )
                s_neg = model.score_relation_with_context(
                    q_embed[neg_idx : neg_idx + 1], rid, z_prev_by_hop[ell][neg_idx : neg_idx + 1], ell
                )
                total_loss += torch.relu(s_neg - s_pos + gamma_C).squeeze(0)
                count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / count
