import torch
from typing import List


def depth_contrastive_loss(
    candidate_rel_ids: List[torch.Tensor],
    logits_by_hop: List[torch.Tensor],
    labels_by_hop: List[torch.Tensor],
    masks_by_hop: List[torch.Tensor],
    gamma_D: float = 0.2,
) -> torch.Tensor:
    batch_size = logits_by_hop[0].shape[0]
    total_loss = torch.tensor(0.0, device=logits_by_hop[0].device)
    count = 0

    L = len(logits_by_hop)
    for i in range(batch_size):
        relation_to_hop_positions = {}
        for ell in range(L):
            rel_ids = candidate_rel_ids[ell][i]
            labels = labels_by_hop[ell][i]
            mask = masks_by_hop[ell][i]
            for pos in range(rel_ids.size(0)):
                if not mask[pos].item():
                    continue
                rid = rel_ids[pos].item()
                if rid < 0:
                    continue
                relation_to_hop_positions.setdefault(rid, {}).setdefault(ell, []).append(pos)

        for rid, hop_positions in relation_to_hop_positions.items():
            for ell_pos, pos_indices in hop_positions.items():
                positive_scores = []
                for pos in pos_indices:
                    if labels_by_hop[ell_pos][i, pos] == 1.0:
                        positive_scores.append(logits_by_hop[ell_pos][i, pos])
                if not positive_scores:
                    continue
                best_pos = torch.stack(positive_scores).max()

                for ell_neg, neg_positions in hop_positions.items():
                    if ell_neg == ell_pos:
                        continue
                    negative_scores = []
                    for pos in neg_positions:
                        if labels_by_hop[ell_neg][i, pos] == 0.0:
                            negative_scores.append(logits_by_hop[ell_neg][i, pos])
                    if not negative_scores:
                        continue
                    best_neg = torch.stack(negative_scores).max()
                    margin_loss = torch.relu(best_neg - best_pos + gamma_D)
                    total_loss += margin_loss
                    count += 1

    if count == 0:
        return torch.tensor(0.0, device=logits_by_hop[0].device)
    return total_loss / count
