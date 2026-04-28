import json
from caff.data import KnowledgeGraph
# Load data
data = json.load(open("caff_expanded_train.json", encoding="utf-8"))
print(f"Total records: {len(data)}")
print()
print("First 3 records:")
for r in data[:3]:
    print(f"  qid={r['query_id']}  seeds={r['seeds']}  gold={r['gold_answer']}")
print()
# Load KG
kg = KnowledgeGraph.from_tsv("caff_expanded_kg.tsv", min_relation_freq=50)
print()
print(f"KG entities sample (first 5): {kg.entities[:5]}")
print(f"KG entities sample (last 5): {kg.entities[-5:]}")
print()
# Check overlap
sample = data[0]
print(f"Sample query: {sample['query_id']}")
print(f"  Seeds: {sample['seeds']}")
print(f"  Seed found in KG: {[s in kg.entity_to_idx for s in sample['seeds']]}")
print(f"  Gold: {sample['gold_answer']}")
print(f"  Gold found in KG: {sample['gold_answer'] in kg.entity_to_idx}")
print()
# Count overlap across all records
n_seeds_found = 0
n_gold_found = 0
n_both_found = 0
for r in data:
    seeds_ok = all(s in kg.entity_to_idx for s in r["seeds"])
    gold_ok = r["gold_answer"] in kg.entity_to_idx
    if seeds_ok:
        n_seeds_found += 1
    if gold_ok:
        n_gold_found += 1
    if seeds_ok and gold_ok:
        n_both_found += 1
print(f"Records with all seeds in KG:   {n_seeds_found}/{len(data)}")
print(f"Records with gold in KG:         {n_gold_found}/{len(data)}")
print(f"Records with BOTH in KG:         {n_both_found}/{len(data)}")
