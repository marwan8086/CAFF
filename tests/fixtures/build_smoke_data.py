"""
build_smoke_data.py
Builds a small synthetic biomedical KG + QA splits where:
  - KG entities ARE CUIs (no Entity-name vs CUI mismatch)
  - QA seeds and gold answers point to real CUIs in the KG
  - Every QA record has a verifiable path from seed to gold within L hops
"""
import json
import random
from collections import defaultdict
import networkx as nx
random.seed(42)
# ─── KG generation ────────────────────────────────────────────────
N_ENTITIES = 5000
N_RELATIONS = 20
AVG_DEGREE = 8  # higher density for directed reachability
relations = [
    "associated_with", "causes", "treats", "interacts_with",
    "expressed_in", "mutated_in", "inhibits", "activates",
    "binds_to", "regulates", "belongs_to_pathway", "has_phenotype",
    "located_in", "derived_from", "similar_to", "co_occurs_with",
    "correlated_with", "precedes", "follows", "part_of",
][:N_RELATIONS]
# Generate CUI-named entities
entities = [f"C{1000000 + i}" for i in range(N_ENTITIES)]
# Build a connected scale-free-ish graph
triples = []
seen = set()
for i, src in enumerate(entities):
    # Each entity gets AVG_DEGREE outgoing edges
    for _ in range(AVG_DEGREE):
        tgt = random.choice(entities)
        if tgt == src:
            continue
        rel = random.choice(relations)
        key = (src, rel, tgt)
        if key in seen:
            continue
        seen.add(key)
        triples.append((src, rel, tgt))
print(f"KG: {N_ENTITIES} entities, {len(triples)} triples, {N_RELATIONS} relations")
# Write KG TSV
with open("smoke_kg.tsv", "w", encoding="utf-8") as f:
    f.write("head\trelation\ttail\n")
    for h, r, t in triples:
        f.write(f"{h}\t{r}\t{t}\n")
# ─── Build undirected graph for path finding ───────────────────────
G = nx.DiGraph()  # directed: matches trainer kg.adj BFS
for h, r, t in triples:
    G.add_edge(h, t)
# ─── QA generation: pick (seed, gold) pairs reachable within 1-3 hops
N_QUERIES = 1000
queries = []
attempts = 0
max_attempts = N_QUERIES * 100  # directed paths are sparser
while len(queries) < N_QUERIES and attempts < max_attempts:
    attempts += 1
    seed = random.choice(entities)
    gold = random.choice(entities)
    if seed == gold:
        continue
    try:
        path_len = nx.shortest_path_length(G, source=seed, target=gold)
    except nx.NetworkXNoPath:
        continue
    if path_len < 1 or path_len > 3:
        continue
    queries.append({
        "query_id": f"smoke_{len(queries):04d}",
        "question": f"What is associated with {seed}?",
        "seeds": [seed],
        "gold_answer": gold,
        "answer_label": "yes",
    })
print(f"QA: {len(queries)} records (after {attempts} attempts)")
# ─── Split ─────────────────────────────────────────────────────────
random.shuffle(queries)
n_train = int(0.7 * len(queries))
n_dev = int(0.15 * len(queries))
train = queries[:n_train]
dev = queries[n_train:n_train + n_dev]
test = queries[n_train + n_dev:]
for name, split in [("train", train), ("dev", dev), ("test", test)]:
    with open(f"smoke_{name}.json", "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(f"  smoke_{name}.json: {len(split)} records")
print()
print("Sanity check on train[0]:")
sample = train[0]
print(f"  qid={sample['query_id']}")
print(f"  seeds={sample['seeds']} (in KG entities: {sample['seeds'][0] in {h for h,r,t in triples} | {t for h,r,t in triples}})")
print(f"  gold={sample['gold_answer']}")
print(f"  shortest path length: {nx.shortest_path_length(G, sample['seeds'][0], sample['gold_answer'])}")




