#!/usr/bin/env python
"""
Simple test script for CAFF components
"""

import sys
sys.path.insert(0, '.')

from caff.data import KnowledgeGraph

# Test KG loading
print("Loading KG...")
kg = KnowledgeGraph.from_tsv('data/processed/merged_kg.tsv', min_relation_freq=0)
print(f"KG loaded: {len(kg.entities)} entities, {len(kg.triples)} triples, {len(kg.relations)} relations")

print("KG test successful!")