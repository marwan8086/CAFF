"""
scripts/probe_hc3_keys.py - Data-only probe (no model, no training).

Goal: determine why HC3 produces zero gradient (Full CAFF weights ==
NoHC3 weights, diff = 0.0).

The HC3 miner (caff/miners.py get_negatives_for) builds, for a
positive anchor, a key = (query_id, relation, hop) and looks for
OTHER buffered instances with the SAME key but label = 0. Those
become the negative contexts.

For HC3 to produce a useful gradient, two things must hold:
  (1) There must exist (query_id, relation, hop) keys that carry
      BOTH a label=1 and a label=0 instance. Otherwise the miner
      returns no negatives and the loss is identically zero.
  (2) Even if such keys exist, the positive and negative instances
      must end up with DIFFERENT upstream context z, or the scorer
      gives them identical scores and relu(s_neg - s_pos + margin)
      is a constant -> zero gradient.

This probe only checks (1) here, using the dataset exactly as the
trainer builds it. It does NOT need the model. It reports how many
keys carry both labels, which directly tells us whether HC3 has any
material to learn from.

Run:
    python scripts/probe_hc3_keys.py --config configs/caff_orphanet.yaml
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caff import (
    CAFFConfig,
    CAFFTripleDataset,
    CachedBFSExtractor,
    KnowledgeGraph,
    load_qa_split,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/caff_orphanet.yaml")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = CAFFConfig(**raw.get("config", {}))

    kg = KnowledgeGraph.from_tsv(cfg.kg_path, min_relation_freq=cfg.min_relation_freq)
    bfs = CachedBFSExtractor(kg, L=cfg.L, K_r=cfg.K_r, cache_dir=Path(args.cache_dir) / "bfs")

    split_path = {"train": cfg.train_path, "dev": cfg.dev_path, "test": cfg.test_path}[args.split]
    recs = load_qa_split(split_path)
    ds = CAFFTripleDataset(recs, bfs, require_gold=True)

    print(f"Loaded {len(ds):,} triple instances from {split_path}")

    # Build the same key the HC3 miner uses: (query_id, relation, hop)
    key_labels: dict[tuple, set] = defaultdict(set)
    key_pos_count: dict[tuple, int] = defaultdict(int)
    key_neg_count: dict[tuple, int] = defaultdict(int)

    for inst in ds.instances:
        key = (inst.query_id, inst.relation, inst.hop)
        key_labels[key].add(inst.label)
        if inst.label == 1:
            key_pos_count[key] += 1
        else:
            key_neg_count[key] += 1

    total_keys = len(key_labels)
    keys_both = [k for k, labs in key_labels.items() if labs == {0, 1}]
    keys_only_pos = [k for k, labs in key_labels.items() if labs == {1}]
    keys_only_neg = [k for k, labs in key_labels.items() if labs == {0}]

    n_pos_total = sum(key_pos_count.values())

    # How many positive anchors actually have a same-key negative available?
    pos_anchors_with_neg = 0
    pos_anchors_total = 0
    for inst in ds.instances:
        if inst.label != 1:
            continue
        pos_anchors_total += 1
        key = (inst.query_id, inst.relation, inst.hop)
        if key_neg_count.get(key, 0) > 0:
            pos_anchors_with_neg += 1

    print()
    print("=" * 70)
    print("HC3 KEY ANALYSIS  key = (query_id, relation, hop)")
    print("=" * 70)
    print(f"  distinct keys                       : {total_keys:,}")
    print(f"  keys with BOTH label 0 and 1        : {len(keys_both):,}")
    print(f"  keys with only positives            : {len(keys_only_pos):,}")
    print(f"  keys with only negatives            : {len(keys_only_neg):,}")
    print()
    print(f"  positive anchors total              : {pos_anchors_total:,}")
    print(f"  positive anchors with same-key neg  : {pos_anchors_with_neg:,}")
    if pos_anchors_total:
        pct = 100.0 * pos_anchors_with_neg / pos_anchors_total
        print(f"  => fraction of anchors minable      : {pct:.2f}%")
    print("=" * 70)
    print()

    if pos_anchors_with_neg == 0:
        print("VERDICT (Hypothesis A confirmed):")
        print("  NO positive anchor has a same-(query,relation,hop) negative.")
        print("  The HC3 miner can never assemble a triplet, so the HC3 loss")
        print("  is identically zero and contributes no gradient. This fully")
        print("  explains why Full CAFF and NoHC3 produce identical weights.")
        print()
        print("  This is a DATA/DESIGN issue: the gold-labeling scheme assigns")
        print("  one label per (query, relation, hop) almost everywhere, so the")
        print("  'same triple under a different context' that HC3 needs does")
        print("  not exist in this KG-derived dataset.")
    else:
        print("VERDICT (Hypothesis A rejected):")
        print(f"  {pos_anchors_with_neg:,} anchors DO have same-key negatives.")
        print("  HC3 has material to mine. The zero-gradient effect must then")
        print("  come from identical upstream context z within a group")
        print("  (Hypothesis B) - the positive and negative share the same")
        print("  z_prev, so the scorer returns equal scores and relu() is")
        print("  constant. That requires a model-level probe to confirm.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
