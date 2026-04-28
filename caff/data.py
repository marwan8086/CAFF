"""
caff/data.py
============
Data pipeline for CAFF: KG loading, BFS extraction with frequency
capping, gold-relevance annotation, and PyTorch Datasets.

Implements:
  • Eq. 1   — BFS candidate sets C_ℓ
  • Eq. 13  — Frequency cap K_r=20 per (head, relation)
  • Paper §8.1 — Gold annotation via shortest-path reachability
  • Paper §8.4 — Train/dev/test splits

Smart Engineering:
  • S2 — Pre-computed BFS subgraphs cached to disk (deterministic
         given G and S(Q), so we never recompute across seeds).

Data formats expected
---------------------
merged_kg.tsv :
    Tab-separated, columns: head, relation, tail, [head_type], [tail_type]
    Header row required.

train.json / dev.json / test.json :
    JSON list of objects:
      {
        "query_id": str,
        "question": str,           # natural-language query
        "seeds": [str, ...],       # entities linked from the question
        "gold_answer": str | null, # answer entity (for shortest-path
                                   # annotation) — used at training only
        "answer_label": str | null # for end-to-end QA (yes/no/maybe)
      }
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import networkx as nx
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Knowledge Graph container
# ─────────────────────────────────────────────────────────────────


@dataclass
class Triple:
    """A directed typed triple (h, r, t) per Assumption 1."""
    h: str
    r: str
    t: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.h, self.r, self.t)


class KnowledgeGraph:
    """Biomedical KG (paper §8.1).

    After construction:
      |V|  ≈ 148,423  (entities)
      |E|  ≈ 2,318,941 (triples)
      |R|  = 42       (after singleton removal: relations with <50 triples dropped)

    Stores both:
      • A list of Triple objects (canonical)
      • An adjacency dict   adj[h] -> list[(r, t)]   for fast BFS
      • A reverse adjacency rev[t] -> list[(r, h)]   for tail-degree lookups
      • A NetworkX MultiDiGraph for shortest-path queries
    """

    def __init__(
        self,
        triples: list[Triple],
        min_relation_freq: int = 50,
    ) -> None:
        # Singleton-relation removal (paper §8.1: "Singleton relations
        # (<50 triples) are removed, retaining |R|=42").
        if min_relation_freq > 0:
            rel_counts: dict[str, int] = defaultdict(int)
            for t in triples:
                rel_counts[t.r] += 1
            kept_relations = {r for r, c in rel_counts.items() if c >= min_relation_freq}
            n_before = len(triples)
            triples = [t for t in triples if t.r in kept_relations]
            logger.info(
                f"Filtered {n_before - len(triples)} triples from "
                f"{len(rel_counts) - len(kept_relations)} singleton relations "
                f"(min_freq={min_relation_freq})"
            )

        self.triples: list[Triple] = triples
        self.entities: list[str] = sorted({t.h for t in triples} | {t.t for t in triples})
        self.relations: list[str] = sorted({t.r for t in triples})
        self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}
        self.relation_to_idx = {r: i for i, r in enumerate(self.relations)}

        # Forward adjacency: h -> [(r, t), ...]
        self.adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        # Reverse adjacency: t -> [(r, h), ...]  (for tail-degree lookups)
        self.rev: dict[str, list[tuple[str, str]]] = defaultdict(list)
        # Tail-degree cache: t -> deg_in(t)
        self._tail_degree: dict[str, int] = defaultdict(int)

        for tr in triples:
            self.adj[tr.h].append((tr.r, tr.t))
            self.rev[tr.t].append((tr.r, tr.h))
            self._tail_degree[tr.t] += 1

        # Fail-fast: an empty KG after filtering is ALWAYS a configuration bug.
        if len(triples) == 0:
            raise ValueError(
                f"KnowledgeGraph is empty after filtering "
                f"(min_relation_freq={min_relation_freq}). "
                f"Either the source TSV is empty, or every relation has "
                f"fewer than min_relation_freq triples. "
                f"Inspect the KG file before retrying."
            )

        logger.info(
            f"KG loaded: |V|={len(self.entities):,}  "
            f"|E|={len(triples):,}  |R|={len(self.relations)}"
        )

    def tail_degree(self, entity: str) -> int:
        """In-degree of `entity` = number of triples with t=entity.

        Used by FreqCap (Eq. 13) to rank candidates by tail degree.
        """
        return self._tail_degree.get(entity, 0)

    def neighbors(self, head: str) -> list[tuple[str, str]]:
        """Return [(r, t), ...] outgoing from `head`."""
        return self.adj.get(head, [])

    def to_networkx(self) -> nx.MultiDiGraph:
        """Build a NetworkX graph for shortest-path annotation.

        Built lazily and cached — the graph is large (~2.3M edges)
        and only needed for gold annotation, not for training.
        """
        if not hasattr(self, "_nx_graph"):
            G = nx.MultiDiGraph()
            G.add_nodes_from(self.entities)
            for tr in self.triples:
                G.add_edge(tr.h, tr.t, key=tr.r, relation=tr.r)
            self._nx_graph = G
            logger.info(f"NetworkX graph built: {G.number_of_edges():,} edges")
        return self._nx_graph

    @classmethod
    def from_tsv(
        cls,
        path: str | Path,
        min_relation_freq: int = 50,
    ) -> "KnowledgeGraph":
        """Load merged KG from a tab-separated file."""
        path = Path(path)
        triples: list[Triple] = []
        with path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            assert "head" in header and "relation" in header and "tail" in header, (
                f"merged_kg.tsv must have columns 'head', 'relation', 'tail'; "
                f"got {header}"
            )
            h_idx = header.index("head")
            r_idx = header.index("relation")
            t_idx = header.index("tail")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(h_idx, r_idx, t_idx):
                    continue
                triples.append(Triple(parts[h_idx], parts[r_idx], parts[t_idx]))
        return cls(triples, min_relation_freq=min_relation_freq)


# ─────────────────────────────────────────────────────────────────
# BFS candidate extraction (Eq. 1) and Frequency Cap (Eq. 13)
# ─────────────────────────────────────────────────────────────────


def bfs_candidate_sets(
    kg: KnowledgeGraph,
    seeds: list[str],
    L: int,
) -> list[list[Triple]]:
    """Eq. 1 — extract BFS candidate sets up to depth L.

        C_ℓ = { (h, r, t) ∈ E : d_G(S(Q), h) = ℓ - 1 }

    where d_G is the (undirected) shortest-path distance from any
    seed in S(Q). We expand BFS forward only (h → t) since the
    paper's reasoning chains follow directed semantics.

    Parameters
    ----------
    kg : KnowledgeGraph
    seeds : list of entity names — S(Q)
    L : max hop depth (paper default: 3)

    Returns
    -------
    candidate_sets : list of length L
        candidate_sets[ℓ] = C_{ℓ+1}, the candidates at hop ℓ+1
        (Python is 0-indexed; paper uses 1-indexed ℓ).
    """
    visited: set[str] = set(seeds)
    frontier: set[str] = {s for s in seeds if s in kg.adj}
    candidate_sets: list[list[Triple]] = []

    for hop in range(L):
        next_frontier: set[str] = set()
        hop_candidates: list[Triple] = []
        for h in frontier:
            for r, t in kg.neighbors(h):
                hop_candidates.append(Triple(h, r, t))
                if t not in visited:
                    next_frontier.add(t)
        candidate_sets.append(hop_candidates)
        visited |= next_frontier
        frontier = next_frontier

    return candidate_sets


def apply_frequency_cap(
    candidates: list[Triple],
    kg: KnowledgeGraph,
    K_r: int = 20,
) -> list[Triple]:
    """Eq. 13 — Frequency cap: per (head, relation) keep top-K_r tails by deg(t).

        C^{(h,r)}_ℓ = top-K_r({(h, r, t) ∈ E}, by deg(t))

    This prevents hub-entity relation embeddings from saturating
    the CSV (paper §6.1).

    Parameters
    ----------
    candidates : list of Triple
        BFS-extracted candidates from one hop.
    kg : KnowledgeGraph
        For tail-degree lookups.
    K_r : int
        Cap (paper default: 20).

    Returns
    -------
    Filtered list of Triple, with at most K_r per (h, r) pair.

    Failure mode F-data-1 (silent over-cap):
    We assert in the caller that no (h, r) pair exceeds K_r.
    """
    # Group by (h, r)
    grouped: dict[tuple[str, str], list[Triple]] = defaultdict(list)
    for tr in candidates:
        grouped[(tr.h, tr.r)].append(tr)

    capped: list[Triple] = []
    for (h, r), group in grouped.items():
        if len(group) <= K_r:
            capped.extend(group)
        else:
            # Sort by tail degree (descending), keep top-K_r
            sorted_group = sorted(
                group, key=lambda tr: kg.tail_degree(tr.t), reverse=True
            )
            capped.extend(sorted_group[:K_r])
    return capped


# ─────────────────────────────────────────────────────────────────
# Gold-relevance annotation (paper §8.1)
# ─────────────────────────────────────────────────────────────────


def annotate_gold_relevance(
    kg: KnowledgeGraph,
    seeds: list[str],
    gold_answer: str,
    L: int,
) -> set[tuple[str, str, str]]:
    """Paper §8.1: triples on any shortest path from seed to gold
    answer entity receive y=1; all others y=0.

    Returns
    -------
    Set of (h, r, t) tuples with y=1.

    Implementation
    --------------
    For each seed s, find ALL shortest paths from s to gold_answer
    in the underlying graph (ignoring relation labels for path
    finding — paper uses unlabeled shortest path). Every edge on
    any shortest path becomes a positive triple.
    """
    G = kg.to_networkx()
    if gold_answer not in G:
        return set()

    positive_edges: set[tuple[str, str, str]] = set()
    for s in seeds:
        if s not in G:
            continue
        try:
            length = nx.shortest_path_length(G, source=s, target=gold_answer)
            if length > L:
                continue
            for path in nx.all_shortest_paths(G, source=s, target=gold_answer):
                for u, v in zip(path[:-1], path[1:]):
                    # All parallel edges (relations) between u and v
                    # are considered on the shortest path.
                    for key in G[u][v]:
                        positive_edges.add((u, key, v))
        except nx.NetworkXNoPath:
            continue
        except nx.NodeNotFound:
            continue

    return positive_edges


# ─────────────────────────────────────────────────────────────────
# Cached BFS extractor (Smart Engineering S2)
# ─────────────────────────────────────────────────────────────────


class CachedBFSExtractor:
    """Pre-computes and caches BFS subgraphs per query_id.

    BFS is deterministic given (KG, seeds, L), so caching across
    seeds and epochs is safe and saves substantial wall-clock.

    Cache layout:
      cache_dir/
        bfs_<query_id>.pkl  →  dict {
          "candidate_sets":      list[list[Triple]],
          "candidate_sets_cap":  list[list[Triple]],  # post-FreqCap
          "gold_positives":      set[tuple[str,str,str]] | None,
        }
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        L: int = 3,
        K_r: int = 20,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.kg = kg
        self.L = L
        self.K_r = K_r
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, query_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        # Sanitize query_id for filesystem
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in query_id)
        return self.cache_dir / f"bfs_{safe}.pkl"

    def extract(
        self,
        query_id: str,
        seeds: list[str],
        gold_answer: str | None = None,
    ) -> dict:
        """Extract (or load from cache) BFS data for a single query."""
        cache_path = self._cache_path(query_id)
        if cache_path is not None and cache_path.exists():
            with cache_path.open("rb") as f:
                return pickle.load(f)

        candidate_sets = bfs_candidate_sets(self.kg, seeds, self.L)
        candidate_sets_cap = [
            apply_frequency_cap(C, self.kg, self.K_r) for C in candidate_sets
        ]

        # Sanity check (failure mode F-data-1)
        for hop_idx, C in enumerate(candidate_sets_cap):
            counts: dict[tuple[str, str], int] = defaultdict(int)
            for tr in C:
                counts[(tr.h, tr.r)] += 1
            for (h, r), c in counts.items():
                assert c <= self.K_r, (
                    f"FreqCap violation at hop {hop_idx}: "
                    f"({h}, {r}) has {c} candidates > K_r={self.K_r}"
                )

        gold_positives: set[tuple[str, str, str]] | None = None
        if gold_answer is not None:
            gold_positives = annotate_gold_relevance(
                self.kg, seeds, gold_answer, self.L
            )

        data = {
            "candidate_sets": candidate_sets,
            "candidate_sets_cap": candidate_sets_cap,
            "gold_positives": gold_positives,
        }
        if cache_path is not None:
            with cache_path.open("wb") as f:
                pickle.dump(data, f)
        return data


# ─────────────────────────────────────────────────────────────────
# QA query records
# ─────────────────────────────────────────────────────────────────


@dataclass
class QARecord:
    """A single QA instance from train/dev/test split."""
    query_id: str
    question: str
    seeds: list[str]
    gold_answer: str | None
    answer_label: str | None  # for end-to-end QA evaluation


def load_qa_split(path: str | Path) -> list[QARecord]:
    """Load a train/dev/test JSON file into QARecord objects."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    for r in raw:
        records.append(
            QARecord(
                query_id=r["query_id"],
                question=r["question"],
                seeds=r.get("seeds", []),
                gold_answer=r.get("gold_answer"),
                answer_label=r.get("answer_label"),
            )
        )
    logger.info(f"Loaded {len(records)} QA records from {path}")
    return records


# ─────────────────────────────────────────────────────────────────
# Triple-level dataset (per training instance = one BFS triple)
# ─────────────────────────────────────────────────────────────────


@dataclass
class TripleInstance:
    """One training example: a single (Q, h, r, t) at a specific hop ℓ.

    The label y comes from gold-relevance annotation. The upstream
    context z_{ℓ-1} is computed lazily during training (we cannot
    cache it because it depends on the model's current decisions
    at hops 1..ℓ-1).
    """
    query_id: str
    question: str
    head: str
    relation: str
    tail: str
    hop: int            # ℓ ∈ {1, ..., L}
    label: int          # y ∈ {0, 1}


class CAFFTripleDataset(Dataset):
    """PyTorch Dataset of triple-level training instances.

    Each item is one TripleInstance. The trainer iterates over
    these in mini-batches, but actual scoring happens at the
    query+hop level (because W^ctx is per-query-per-hop, not
    per-triple). The DataLoader thus uses a custom collate_fn
    that groups by (query_id, hop) — provided in caff/trainer.py.
    """

    def __init__(
        self,
        qa_records: list[QARecord],
        bfs_extractor: CachedBFSExtractor,
        require_gold: bool = True,
    ) -> None:
        self.qa_records = qa_records
        self.bfs_extractor = bfs_extractor
        self.instances: list[TripleInstance] = []
        self._build(require_gold=require_gold)

    def _build(self, require_gold: bool) -> None:
        """Materialize per-triple training instances from QA records."""
        skipped_no_gold = 0
        for rec in self.qa_records:
            if require_gold and rec.gold_answer is None:
                skipped_no_gold += 1
                continue
            bfs = self.bfs_extractor.extract(
                rec.query_id, rec.seeds, rec.gold_answer
            )
            gold_set = bfs["gold_positives"] or set()
            for hop_idx, C in enumerate(bfs["candidate_sets_cap"]):
                hop = hop_idx + 1
                for tr in C:
                    label = 1 if tr.as_tuple() in gold_set else 0
                    self.instances.append(
                        TripleInstance(
                            query_id=rec.query_id,
                            question=rec.question,
                            head=tr.h,
                            relation=tr.r,
                            tail=tr.t,
                            hop=hop,
                            label=label,
                        )
                    )
        logger.info(
            f"Built {len(self.instances):,} triple instances "
            f"(skipped {skipped_no_gold} records lacking gold answers)"
        )
        if len(self.instances) > 0:
            n_pos = sum(1 for x in self.instances if x.label == 1)
            logger.info(
                f"  Class balance: {n_pos / len(self.instances) * 100:.2f}% positive"
            )

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> TripleInstance:
        return self.instances[idx]

    def iter_by_query_hop(
        self,
    ) -> Iterator[tuple[str, int, str, list[TripleInstance]]]:
        """Yield (query_id, hop, question, [instances]) groups.

        Used by the trainer to batch all candidates of one
        (query, hop) together, since they share W^ctx (S3).
        """
        groups: dict[tuple[str, int], list[TripleInstance]] = defaultdict(list)
        for inst in self.instances:
            groups[(inst.query_id, inst.hop)].append(inst)
        for (qid, hop), items in groups.items():
            yield qid, hop, items[0].question, items

    def class_imbalance_pos_weight(self) -> float:
        """Compute pos_weight = N_neg / N_pos for BCEWithLogitsLoss."""
        n_pos = sum(1 for x in self.instances if x.label == 1)
        n_neg = len(self.instances) - n_pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos