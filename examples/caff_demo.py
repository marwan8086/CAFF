"""
⚠️  WARNING — DEMONSTRATION FILE ONLY  ⚠️
This file uses np.random.randn(...) to generate FAKE embeddings for
quick UI/visualization demos. It is NOT representative of CAFF's
actual training or evaluation pipeline.
For the real CAFF pipeline, see:
  - train.py     (training entry point)
  - evaluate.py  (evaluation entry point)
  - caff/        (core implementation)
DO NOT cite results from this file. Numbers shown are illustrative only.
"""#!/usr/bin/env python
"""
Simple CAFF demonstration with synthetic data
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List

# Simple data structures
@dataclass
class Triple:
    h: str
    r: str
    t: str

@dataclass
class QARecord:
    query_id: str
    question: str
    seeds: List[str]
    gold_answer: str = None
    answer_label: str = None

class KnowledgeGraph:
    def __init__(self, triples, min_relation_freq=0):
        if min_relation_freq > 0:
            rel_counts = defaultdict(int)
            for t in triples:
                rel_counts[t.r] += 1
            kept_relations = {r for r, c in rel_counts.items() if c >= min_relation_freq}
            triples = [t for t in triples if t.r in kept_relations]

        self.triples = triples
        self.entities = sorted({t.h for t in triples} | {t.t for t in triples})
        self.relations = sorted({t.r for t in triples})

        self.adj = defaultdict(list)
        for tr in triples:
            self.adj[tr.h].append((tr.r, tr.t))

    @classmethod
    def from_tsv(cls, path, min_relation_freq=0):
        triples = []
        with open(path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            h_idx = header.index('head')
            r_idx = header.index('relation')
            t_idx = header.index('tail')
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) > max(h_idx, r_idx, t_idx):
                    triples.append(Triple(parts[h_idx], parts[r_idx], parts[t_idx]))
        return cls(triples, min_relation_freq)

def load_qa_split(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    records = []
    for r in raw:
        records.append(QARecord(
            query_id=r["query_id"],
            question=r["question"],
            seeds=r.get("seeds", []),
            gold_answer=r.get("gold_answer"),
            answer_label=r.get("answer_label"),
        ))
    return records

def bfs_candidate_sets(kg, seeds, L=3):
    """Simple BFS to get candidate triples up to depth L"""
    candidates = []
    visited = set()
    current = set(seeds)

    for hop in range(L):
        next_level = set()
        hop_candidates = []

        for entity in current:
            if entity in visited:
                continue
            visited.add(entity)

            for r, t in kg.adj.get(entity, []):
                hop_candidates.append(Triple(entity, r, t))
                next_level.add(t)

        candidates.append(hop_candidates)
        current = next_level

    return candidates

def main():
    print("Loading synthetic biomedical KG...")
    kg = KnowledgeGraph.from_tsv('data/processed/merged_kg.tsv', min_relation_freq=0)
    print(f"KG loaded: {len(kg.entities)} entities, {len(kg.triples)} triples, {len(kg.relations)} relations")

    print("\nLoading QA data...")
    qa_records = load_qa_split('data/processed/train.json')
    print(f"Loaded {len(qa_records)} QA records")

    print("\nDemonstrating BFS candidate extraction...")
    for record in qa_records[:2]:  # Show first 2
        print(f"\nQuery: {record.question}")
        print(f"Seeds: {record.seeds}")
        candidates = bfs_candidate_sets(kg, record.seeds, L=2)
        for hop, triples in enumerate(candidates):
            print(f"  Hop {hop+1}: {len(triples)} candidates")
            for triple in triples[:3]:  # Show first 3 per hop
                print(f"    {triple.h} -> {triple.r} -> {triple.t}")

    print("\nCAFF demonstration complete!")
    print("This shows the basic pipeline: KG loading, QA data, and BFS candidate extraction.")
    print("The full CAFF model would learn to score these candidates based on context.")

if __name__ == "__main__":
    main()
