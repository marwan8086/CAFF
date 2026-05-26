"""
scripts/probe_hc3_fix.py - Test the proposed cross-query contrastive
fix WITHOUT modifying caff/miners.py or retraining.

Section 22 established that the existing HC3 loss is inert: positive
and negative share the same (query_id, hop) group, hence the same
teacher-forced z_prev, hence identical scores and zero gradient.

This probe simulates the proposed fix (drawing the negative from a
DIFFERENT query that shares the same (relation, hop), so its z_prev
differs) and measures whether that restores a non-zero contrast and
gradient. It tests BOTH scoring paths:
  * score_candidates (post-sigmoid, what HC3 currently uses)
  * score_logits     (pre-sigmoid, the proposed improvement)

IMPORTANT (honesty note): this cross-query variant changes BOTH the
query embedding q AND the context z between positive and negative, so
it is NOT the paper's HC3 ("same triple, different context"). It is a
cross-query contrastive variant. We measure it because the empirical
effect on F1 is worth recording regardless of the name.

It only reads the model and data; it changes nothing on disk.

Run:
    python scripts/probe_hc3_fix.py --checkpoint runs/caff_orphanet/seed_42/best.pt --device cuda
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caff.data import CachedBFSExtractor, load_qa_split, CAFFTripleDataset  # noqa
from caff.trainer import teacher_forced_z_prev
from evaluate import load_checkpoint


def build_per_query_by_hop(instances):
    """query_id -> {hop -> [instances]} so teacher_forced_z_prev can run."""
    out = defaultdict(lambda: defaultdict(list))
    for inst in instances:
        out[inst.query_id][inst.hop].append(inst)
    return out


def score(model, hop_idx, z, q_emb, relation, use_logits):
    z = z.unsqueeze(0)  # (1, d)
    W_ctx = model.get_hop_W_ctx(hop_idx, z).squeeze(0)
    E_r = model.relation_cache.get_batch([relation])
    scorer = model.hop_scorers[hop_idx]
    if use_logits:
        s = scorer.score_logits(W_ctx, model.v, q_emb, E_r)
    else:
        s = scorer.score_candidates(W_ctx, model.v, q_emb, E_r)
    return s.squeeze(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/caff_orphanet/seed_42/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--num-pairs", type=int, default=30)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, config, ablation, encoder, kg = load_checkpoint(
        args.checkpoint, device, Path(args.cache_dir))
    model.train()

    bfs = CachedBFSExtractor(kg, L=config.L, K_r=config.K_r,
                             cache_dir=Path(args.cache_dir) / "bfs")
    recs = load_qa_split(config.train_path)
    ds = CAFFTripleDataset(recs, bfs, require_gold=True)
    per_query = build_per_query_by_hop(ds.instances)

    # Query embedding cache (frozen encoder)
    q_cache = {}

    def get_q(qid, question):
        if qid not in q_cache:
            with torch.no_grad():
                q_cache[qid] = encoder.encode([question])[0].to(device)
        return q_cache[qid]

    # Group positives and negatives by (relation, hop) ACROSS queries,
    # restricted to hop >= 2 (where teacher-forced z is non-zero).
    pos_by_relhop = defaultdict(list)
    neg_by_relhop = defaultdict(list)
    for inst in ds.instances:
        if inst.hop < 2:
            continue
        key = (inst.relation, inst.hop)
        (pos_by_relhop if inst.label == 1 else neg_by_relhop)[key].append(inst)

    # Build cross-query (pos, neg) pairs: same (relation, hop), DIFFERENT query
    pairs = []
    for key, positives in pos_by_relhop.items():
        negs = neg_by_relhop.get(key, [])
        if not negs:
            continue
        for p in positives:
            cross = [n for n in negs if n.query_id != p.query_id]
            if cross:
                pairs.append((p, cross[0]))
                break  # one pair per key is enough for the probe
        if len(pairs) >= args.num_pairs:
            break

    print(f"Cross-query (pos, neg) pairs found (hop>=2): {len(pairs)}")
    if not pairs:
        print("No cross-query same-(relation,hop) pairs at hop>=2.")
        print("The fix cannot create contrast on this data either.")
        return 0

    for use_logits in (False, True):
        mode = "score_logits (pre-sigmoid)" if use_logits else "score_candidates (sigmoid)"
        diffs = []
        pos_list, neg_list = [], []
        for p, n in pairs:
            z_pos = teacher_forced_z_prev(model.csv, per_query[p.query_id],
                                          target_hop=p.hop, d=config.d, device=device)
            z_neg = teacher_forced_z_prev(model.csv, per_query[n.query_id],
                                          target_hop=n.hop, d=config.d, device=device)
            q_pos = get_q(p.query_id, p.question)
            q_neg = get_q(n.query_id, n.question)
            s_pos = score(model, p.hop - 1, z_pos, q_pos, p.relation, use_logits)
            s_neg = score(model, n.hop - 1, z_neg, q_neg, n.relation, use_logits)
            diffs.append((s_pos - s_neg).abs().item())
            pos_list.append(s_pos)
            neg_list.append(s_neg)

        pos_t = torch.stack(pos_list)
        neg_t = torch.stack(neg_list)
        margin = config.gamma_C
        l_hc3 = torch.relu(neg_t - pos_t + margin).mean()

        # also measure z difference to confirm contexts really differ
        z_diffs = []
        for p, n in pairs:
            z_pos = teacher_forced_z_prev(model.csv, per_query[p.query_id],
                                          target_hop=p.hop, d=config.d, device=device)
            z_neg = teacher_forced_z_prev(model.csv, per_query[n.query_id],
                                          target_hop=n.hop, d=config.d, device=device)
            z_diffs.append((z_pos - z_neg).abs().max().item())

        print("\n" + "=" * 70)
        print(f"MODE: {mode}")
        print("=" * 70)
        print(f"  pairs                   : {len(diffs)}")
        print(f"  mean |z_pos - z_neg|max : {statistics.mean(z_diffs):.4f}")
        print(f"  mean |s_pos - s_neg|    : {statistics.mean(diffs):.4e}")
        print(f"  max  |s_pos - s_neg|    : {max(diffs):.4e}")
        print(f"  contrastive loss value  : {l_hc3.item():.6f}  (margin={margin})")

        model.zero_grad()
        if l_hc3.requires_grad:
            l_hc3.backward()
            total = sum(p.grad.abs().sum().item()
                        for _, p in model.named_parameters() if p.grad is not None)
            nz = sum(1 for _, p in model.named_parameters()
                     if p.grad is not None and p.grad.abs().sum().item() > 0)
            print(f"  total |grad| sum        : {total:.6e}")
            print(f"  params with grad>0      : {nz}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("If |s_pos - s_neg| > 0 and total |grad| > 0 (especially in the")
    print("score_logits mode), the cross-query contrastive variant produces")
    print("a real training signal, unlike the inert same-group HC3. If the")
    print("sigmoid mode is ~0 but logits mode is > 0, saturation was masking")
    print("the signal and the fix must operate on logits.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
