"""
tests/test_data.py
==================
Verify data pipeline:

  • FreqCap (Eq. 13): no (h,r) pair has > K_r candidates
  • BFS depth respects L
  • Gold annotation flags shortest-path edges
"""

from __future__ import annotations

import pytest

from caff.data import (
    KnowledgeGraph,
    Triple,
    annotate_gold_relevance,
    apply_frequency_cap,
    bfs_candidate_sets,
)


@pytest.fixture
def small_kg():
    """A small toy KG for unit tests."""
    triples = [
        # 1 head 'A' has 30 (A, r, *) — needs FreqCap
        *[Triple("A", "r", f"t{i}") for i in range(30)],
        # 2-hop chain: A -> B -> C
        Triple("A", "r2", "B"),
        Triple("B", "r3", "C"),
        # Distractor
        Triple("X", "r4", "Y"),
    ]
    # min_relation_freq=0 to avoid removing toy relations
    return KnowledgeGraph(triples, min_relation_freq=0)


def test_freqcap_enforces_K_r(small_kg):
    """Eq. 13: no (h, r) exceeds K_r=20."""
    candidates = [Triple("A", "r", f"t{i}") for i in range(30)]
    capped = apply_frequency_cap(candidates, small_kg, K_r=20)
    by_pair: dict = {}
    for tr in capped:
        by_pair.setdefault((tr.h, tr.r), []).append(tr)
    for (h, r), group in by_pair.items():
        assert len(group) <= 20, f"({h},{r}) has {len(group)} > 20"


def test_bfs_depth_l(small_kg):
    """BFS up to L hops, no further."""
    L = 2
    sets = bfs_candidate_sets(small_kg, ["A"], L=L)
    assert len(sets) == L
    # Hop 1 includes (A, r, t*) and (A, r2, B)
    assert all(t.h == "A" for t in sets[0])
    # Hop 2 expands B (since A→B in hop 1)
    hop2_heads = {t.h for t in sets[1]}
    assert "B" in hop2_heads


def test_gold_annotation_marks_shortest_path(small_kg):
    """A triple on the shortest path A→B→C should be labeled 1."""
    pos = annotate_gold_relevance(
        small_kg, seeds=["A"], gold_answer="C", L=3,
    )
    assert ("A", "r2", "B") in pos
    assert ("B", "r3", "C") in pos
    # Distractor must NOT be on the path
    assert ("X", "r4", "Y") not in pos


def test_gold_annotation_unreachable():
    """If gold is unreachable, annotation returns empty set."""
    kg = KnowledgeGraph(
        [Triple("A", "r", "B"), Triple("X", "r", "Y")],
        min_relation_freq=0,
    )
    pos = annotate_gold_relevance(kg, seeds=["A"], gold_answer="Y", L=3)
    assert pos == set()