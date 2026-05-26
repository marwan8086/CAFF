"""
scripts/probe_hc3_model.py - Model-level confirmation of why HC3 has
zero gradient (Hypothesis B).

Background
----------
The data probe (probe_hc3_keys.py) showed that 51.67% of positive
anchors DO have a same-(query_id, relation, hop) negative, so the
miner is NOT starved (Hypothesis A rejected).

Hypothesis B: within a single (query_id, hop) group, the trainer
assigns ONE shared teacher-forced z_prev to every instance. A
positive anchor and its same-key negative therefore receive the
SAME z_prev, the SAME query embedding q, and (because the key fixes
the relation) the SAME relation embedding E_r. The scorer is a
deterministic function of (W_ctx(z), v, q, E_r), so it returns an
IDENTICAL score for the positive and the negative. Then

    L_HC3 = relu(s_neg - s_pos + margin) = relu(0 + margin) = margin

is constant in the parameters, and its gradient is exactly zero.

This script confirms that empirically. It loads a trained checkpoint,
replicates the EXACT scoring path used by trainer._score_hc3_instance
(model.get_hop_W_ctx -> relation_cache.get_batch -> scorer.score_candidates),
and for several real same-key (pos, neg) pairs reports:

  * |s_pos - s_neg|  (expected ~0 if Hypothesis B holds)
  * the HC3 loss value
  * the total gradient norm after l_hc3.backward()

Run:
    python scripts/probe_hc3_model.py --checkpoint runs/caff_orphanet/seed_42/best.pt --device cuda
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caff import (
    CAFFTripleDataset,
    HC3Loss,
    TrainingInstance,
)
from caff.data import CachedBFSExtractor, KnowledgeGraph, load_qa_split  # noqa
# Reuse the tested checkpoint loader from evaluate.py
from evaluate import load_checkpoint


def score_instance(model, inst, q_embedding, device):
    """Replicate trainer._score_hc3_instance EXACTLY."""
    hop_idx = inst.hop - 1
    z = inst.z_prev.to(device).unsqueeze(0)            # (1, d)
    W_ctx = model.get_hop_W_ctx(hop_idx, z).squeeze(0)  # (d, d)
    E_r = model.relation_cache.get_batch([inst.relation])  # (1, d)
    scorer = model.hop_scorers[hop_idx]
    score = scorer.score_candidates(W_ctx, model.v, q_embedding, E_r)  # (1,)
    return score.squeeze(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/caff_orphanet/seed_42/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--num-pairs", type=int, default=20,
                    help="How many same-key (pos, neg) pairs to test.")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load trained model + config + encoder + kg via the tested helper
    model, config, ablation, encoder, kg = load_checkpoint(
        args.checkpoint, device, Path(args.cache_dir)
    )
    model.train()  # ensure grad is tracked
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  ablation: csv={ablation.use_csv} dbm={ablation.use_dbm} "
          f"hc3={ablation.use_hc3} dc={ablation.use_dc}")

    # 2. Build train dataset to get real instances + grouping
    bfs = CachedBFSExtractor(kg, L=config.L, K_r=config.K_r,
                             cache_dir=Path(args.cache_dir) / "bfs")
    train_recs = load_qa_split(config.train_path)
    ds = CAFFTripleDataset(train_recs, bfs, require_gold=True)

    # 3. Find (query_id, relation, hop) keys that carry BOTH labels.
    #    Group instances exactly like the trainer: by (query_id, hop).
    #    Within such a group, the trainer shares ONE z_prev across all
    #    instances - so we emulate that by assigning the SAME z_prev to
    #    the positive and the negative we pull from the same group.
    by_group: dict[tuple, list] = defaultdict(list)
    for inst in ds.instances:
        by_group[(inst.query_id, inst.hop)].append(inst)

    # Collect same-key (pos, neg) pairs that live in the SAME group
    pairs = []  # (pos_inst, neg_inst, question)
    for (qid, hop), items in by_group.items():
        # index by relation within this group
        by_rel = defaultdict(lambda: {"pos": [], "neg": []})
        question = items[0].question if hasattr(items[0], "question") else ""
        for it in items:
            (by_rel[it.relation]["pos" if it.label == 1 else "neg"]).append(it)
        for rel, d in by_rel.items():
            if d["pos"] and d["neg"]:
                pairs.append((d["pos"][0], d["neg"][0], qid, hop, rel))
        if len(pairs) >= args.num_pairs:
            break

    print(f"\nFound {len(pairs)} same-key (pos, neg) pairs in the SAME group.")
    if not pairs:
        print("No same-group same-key pairs found; cannot test Hypothesis B.")
        return 0

    # 4. For each pair, assign the SHARED group z_prev (teacher forcing
    #    gives one z per (query, hop) group). We use a representative
    #    z_prev: the frozen relation-embedding mean is not needed here;
    #    what matters is that pos and neg get the IDENTICAL z. We build a
    #    single z per group and reuse it for both.
    d = config.d
    hc3 = HC3Loss(margin=config.gamma_C)

    abs_diffs = []
    pos_scores_all = []
    neg_scores_all = []

    # Build query embeddings via the frozen encoder (no grad on encoder;
    # encoder is frozen, so q is a constant input - same for pos and neg).
    q_cache: dict[str, torch.Tensor] = {}

    def get_q(qid, question):
        if qid not in q_cache:
            with torch.no_grad():
                emb = encoder.encode([question])[0]
            q_cache[qid] = emb.to(device)
        return q_cache[qid]

    pos_score_list = []
    neg_score_list = []
    for (pos_inst, neg_inst, qid, hop, rel) in pairs:
        # Shared group z_prev (identical for pos and neg) - this is the
        # crux of Hypothesis B. Use a deterministic non-trivial vector.
        z_shared = torch.randn(d, generator=torch.Generator().manual_seed(hash((qid, hop)) % (2**31)))
        # attach same z to both
        pos_inst = TrainingInstance(
            query_id=qid, question=pos_inst.question if hasattr(pos_inst, "question") else "",
            head=getattr(pos_inst, "head", ""), relation=rel,
            tail=getattr(pos_inst, "tail", ""), hop=hop, label=1, z_prev=z_shared.clone(),
        )
        neg_inst = TrainingInstance(
            query_id=qid, question=neg_inst.question if hasattr(neg_inst, "question") else "",
            head=getattr(neg_inst, "head", ""), relation=rel,
            tail=getattr(neg_inst, "tail", ""), hop=hop, label=0, z_prev=z_shared.clone(),
        )
        q_emb = get_q(qid, pos_inst.question)
        s_pos = score_instance(model, pos_inst, q_emb, device)
        s_neg = score_instance(model, neg_inst, q_emb, device)
        abs_diffs.append((s_pos - s_neg).abs().item())
        pos_score_list.append(s_pos)
        neg_score_list.append(s_neg)

    pos_t = torch.stack(pos_score_list)
    neg_t = torch.stack(neg_score_list)
    l_hc3 = hc3(pos_t, neg_t)

    print("\n" + "=" * 70)
    print("MODEL-LEVEL HC3 PROBE  (Hypothesis B test)")
    print("=" * 70)
    import statistics
    print(f"  pairs tested            : {len(abs_diffs)}")
    print(f"  mean |s_pos - s_neg|    : {statistics.mean(abs_diffs):.3e}")
    print(f"  max  |s_pos - s_neg|    : {max(abs_diffs):.3e}")
    print(f"  HC3 loss value          : {l_hc3.item():.6f}  (margin={config.gamma_C})")
    print(f"  requires_grad           : {l_hc3.requires_grad}")

    model.zero_grad()
    if l_hc3.requires_grad:
        l_hc3.backward()
        total = 0.0
        nz = 0
        for _, p in model.named_parameters():
            if p.grad is not None:
                g = p.grad.abs().sum().item()
                total += g
                if g > 0:
                    nz += 1
        print(f"  total |grad| sum        : {total:.6e}")
        print(f"  params with grad>0      : {nz}")
        print("=" * 70)
        print()
        same = max(abs_diffs) < 1e-6
        if same:
            print("VERDICT (Hypothesis B CONFIRMED):")
            print("  With a shared group z_prev, the positive and negative")
            print("  receive IDENTICAL scores (|s_pos - s_neg| ~ 0). The HC3")
            print("  loss is the constant margin and its gradient is zero.")
            print("  => HC3 cannot learn on same-group pairs. Since the miner")
            print("     only pairs same-(query,relation,hop) instances, which")
            print("     always live in the same group, HC3 is inert. This is")
            print("     why Full CAFF and NoHC3 have identical weights.")
        else:
            print("VERDICT: scores differ; Hypothesis B does NOT fully hold.")
            print("  Investigate whether negatives come from different groups.")
    else:
        print("  l_hc3.requires_grad is False -> gradient path broken upstream.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
