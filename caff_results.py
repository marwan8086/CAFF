#!/usr/bin/env python
"""
CAFF Preliminary Results Demo
Shows the core CAFF pipeline with synthetic data
"""

import sys
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict

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

class SimpleCAFFScorer:
    """Simplified CAFF-like scorer for demonstration"""

    def __init__(self, kg):
        self.kg = kg
        # Simple embeddings (random for demo)
        self.entity_emb = {e: np.random.randn(10) for e in kg.entities}
        self.relation_emb = {r: np.random.randn(10) for r in kg.relations}

        # Simple bilinear scoring matrix
        self.W = np.random.randn(10, 10)

    def score_triple(self, triple, context_triples=None):
        """Score a triple with optional context"""
        h_emb = self.entity_emb[triple.h]
        r_emb = self.relation_emb[triple.r]
        t_emb = self.entity_emb[triple.t]

        # Basic bilinear score
        score = np.dot(np.dot(h_emb, self.W), r_emb) + np.dot(r_emb, t_emb)

        # Add context bonus (simplified CSV-like effect)
        if context_triples:
            context_bonus = len(context_triples) * 0.1
            score += context_bonus

        return score

    def rank_candidates(self, candidates, context=None):
        """Rank candidates by score"""
        scored = []
        for triple in candidates:
            score = self.score_triple(triple, context)
            scored.append((triple, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

def evaluate_predictions(predictions, gold_answers):
    """Simple evaluation metrics"""
    correct = 0
    total = len(predictions)

    for pred, gold in zip(predictions, gold_answers):
        if pred == gold:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}

def main():
    print("=== CAFF Preliminary Results Demo ===\n")

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    print("1. Loading synthetic biomedical KG...")
    kg = KnowledgeGraph.from_tsv('data/processed/merged_kg.tsv', min_relation_freq=0)
    print(f"   KG: {len(kg.entities)} entities, {len(kg.triples)} triples, {len(kg.relations)} relations")

    print("\n2. Loading QA evaluation data...")
    test_records = load_qa_split('data/processed/test.json')
    print(f"   Test set: {len(test_records)} questions")

    print("\n3. Initializing CAFF-like scorer...")
    scorer = SimpleCAFFScorer(kg)
    print("   Scorer initialized with random embeddings")

    print("\n4. Running evaluation...")

    predictions = []
    gold_answers = []

    for record in test_records:
        print(f"\n   Processing: {record.question}")

        # Get candidates via BFS
        candidates = bfs_candidate_sets(kg, record.seeds, L=2)
        all_candidates = [t for hop in candidates for t in hop]

        print(f"   Found {len(all_candidates)} candidate triples")

        # Score and rank candidates
        ranked = scorer.rank_candidates(all_candidates)

        # For demo, predict the top-ranked tail entity
        if ranked:
            top_triple = ranked[0][0]
            prediction = top_triple.t
            print(f"   Top prediction: {prediction} (score: {ranked[0][1]:.3f})")
        else:
            prediction = None
            print("   No candidates found")

        predictions.append(prediction)
        gold_answers.append(record.gold_answer)

        # Show top 3 candidates
        print("   Top candidates:")
        for i, (triple, score) in enumerate(ranked[:3]):
            print(f"     {i+1}. {triple.h} -> {triple.r} -> {triple.t} (score: {score:.3f})")

    print("\n5. Evaluation Results:")
    metrics = evaluate_predictions(predictions, gold_answers)
    print(f"   Accuracy: {metrics['accuracy']:.3f} ({metrics['correct']}/{metrics['total']})")

    print("\n6. Analysis:")
    print("   This demonstrates the CAFF pipeline:")
    print("   - BFS extracts multi-hop candidate triples")
    print("   - Context-aware scoring ranks candidates")
    print("   - Top-ranked entities are predicted answers")
    print("   - In a real CAFF model, the scorer would be trained to optimize this ranking")

    print("\n=== Demo Complete ===")
    print("For full CAFF implementation, the model would learn:")
    print("- CSV: Context-aware embeddings from previous hop selections")
    print("- DBM: Dynamic bilinear modulation based on context")
    print("- HC3: Contrastive loss for better ranking")
    print("- Training on gold relevance annotations from shortest paths")

if __name__ == "__main__":
    main()