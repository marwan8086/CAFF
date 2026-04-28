#!/usr/bin/env python
"""
create_expanded_synthetic_data.py — Create expanded synthetic biomedical KG and QA data.

Creates synthetic data at scale approaching paper results:
- KG: ~10K entities, ~50K triples, ~20 relations
- QA: ~1K queries with BFS annotations
"""

import json
import random
from pathlib import Path

import networkx as nx
import pandas as pd


def create_synthetic_kg(n_entities=10000, n_relations=20, avg_degree=5):
    """Create synthetic biomedical KG with realistic structure."""
    entities = [f"Entity{i}" for i in range(n_entities)]

    # Biomedical relation types
    relations = [
        "associated_with",
        "causes",
        "treats",
        "interacts_with",
        "expressed_in",
        "mutated_in",
        "inhibits",
        "activates",
        "binds_to",
        "regulates",
        "belongs_to_pathway",
        "has_phenotype",
        "located_in",
        "derived_from",
        "similar_to",
        "co_occurs_with",
        "correlated_with",
        "precedes",
        "follows",
        "part_of"
    ][:n_relations]

    # Create graph with preferential attachment (scale-free)
    G = nx.DiGraph()
    G.add_nodes_from(entities)

    triples = []
    cui_counter = 1000000  # Start CUIs from C1000000

    # Add edges with realistic connectivity
    for i in range(len(entities)):
        for j in range(avg_degree):
            target = random.choice(entities)
            if target != entities[i]:  # No self-loops
                relation = random.choice(relations)
                head_cui = f"C{cui_counter}"
                tail_cui = f"C{cui_counter + 1}"
                cui_counter += 2

                triples.append({
                    "head": entities[i],
                    "relation": relation,
                    "tail": target,
                    "head_cui": head_cui,
                    "tail_cui": tail_cui,
                    "source": "synthetic"
                })

    # Deduplicate
    seen = set()
    unique_triples = []
    for t in triples:
        key = (t["head_cui"], t["relation"], t["tail_cui"])
        if key not in seen:
            seen.add(key)
            unique_triples.append(t)

    print(f"Created KG: {len(entities)} entities, {len(unique_triples)} triples, {len(relations)} relations")
    return unique_triples


def create_synthetic_qa(kg_triples, n_queries=1000, max_hops=3):
    """Create synthetic QA data with gold answers reachable within max_hops."""
    # Build undirected graph for reachability (since biomedical KGs are often treated as undirected)
    G = nx.Graph()
    entity_to_cui = {}
    cui_to_entity = {}

    for t in kg_triples:
        G.add_edge(t["head_cui"], t["tail_cui"], relation=t["relation"])
        entity_to_cui[t["head"]] = t["head_cui"]
        entity_to_cui[t["tail"]] = t["tail_cui"]
        cui_to_entity[t["head_cui"]] = t["head"]
        cui_to_entity[t["tail_cui"]] = t["tail"]

    queries = []
    query_id = 1

    while len(queries) < n_queries:
        # Pick random connected seed and gold answer
        edges = list(G.edges())
        if not edges:
            break

        # Pick a random edge and use its endpoints
        edge = random.choice(edges)
        seed_cui = edge[0]
        gold_cui = edge[1]

        # Verify they are connected within max_hops
        try:
            path_length = nx.shortest_path_length(G, seed_cui, gold_cui)
            if path_length > max_hops:
                continue
        except nx.NetworkXNoPath:
            continue

        seed_entity = cui_to_entity[seed_cui]
        gold_entity = cui_to_entity[gold_cui]

        # Create question
        question = f"What is associated with {seed_entity}?"

        queries.append({
            "query_id": f"synth_{query_id}",
            "question": question,
            "seeds": [seed_cui],
            "gold_answer": gold_cui,
            "answer_label": "yes"
        })

        query_id += 1

    print(f"Created {len(queries)} QA queries")
    return queries


def main():
    random.seed(42)

    # Create expanded synthetic KG
    kg_triples = create_synthetic_kg(n_entities=10000, n_relations=20, avg_degree=5)

    # Save KG
    df = pd.DataFrame(kg_triples)
    df.to_csv("caff_expanded_kg.tsv", sep="\t", index=False)

    # Create QA data
    qa_data = create_synthetic_qa(kg_triples, n_queries=1000)

    # Split into train/dev/test
    random.shuffle(qa_data)
    n_train = int(0.7 * len(qa_data))
    n_dev = int(0.15 * len(qa_data))

    train_data = qa_data[:n_train]
    dev_data = qa_data[n_train:n_train + n_dev]
    test_data = qa_data[n_train + n_dev:]

    # Save QA splits
    for split_name, split_data in [("train", train_data), ("dev", dev_data), ("test", test_data)]:
        with open(f"caff_expanded_{split_name}.json", "w") as f:
            json.dump(split_data, f, indent=2)

    print("Expanded synthetic data created:")
    print(f"  KG: caff_expanded_kg.tsv")
    print(f"  Train: caff_expanded_train.json ({len(train_data)} queries)")
    print(f"  Dev: caff_expanded_dev.json ({len(dev_data)} queries)")
    print(f"  Test: caff_expanded_test.json ({len(test_data)} queries)")


if __name__ == "__main__":
    main()