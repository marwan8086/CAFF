import networkx as nx
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set


def build_graph(kg_path: str) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    with open(kg_path, 'r', encoding='utf-8') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            G.add_edge(head, tail, relation=relation)
    return G


def compute_tail_degree(G: nx.MultiDiGraph) -> Dict[str, int]:
    return dict(G.degree())


def bfs_candidate_sets(
    G: nx.MultiDiGraph,
    seeds: List[str],
    L: int,
    Kr: int = 20,
) -> List[List[Tuple[str, str, str]]]:
    candidates_by_hop = []
    current_frontier = list(seeds)
    visited = set(seeds)

    tail_degree = compute_tail_degree(G)
    for _ in range(L):
        next_frontier: List[str] = []
        hop_candidates: List[Tuple[str, str, str]] = []
        for node in current_frontier:
            if node not in G:
                continue
            for nbr in G.successors(node):
                edges = G.get_edge_data(node, nbr)
                for _, data in edges.items():
                    hop_candidates.append((node, data['relation'], nbr))
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.append(nbr)

        hop_candidates = apply_frequency_cap(hop_candidates, tail_degree, Kr)
        candidates_by_hop.append(hop_candidates)
        current_frontier = next_frontier

    return candidates_by_hop


def apply_frequency_cap(
    triples: List[Tuple[str, str, str]],
    tail_degree: Dict[str, int],
    Kr: int,
) -> List[Tuple[str, str, str]]:
    grouped = defaultdict(list)
    for h, r, t in triples:
        grouped[(h, r)].append((t, tail_degree.get(t, 0)))

    capped: List[Tuple[str, str, str]] = []
    for (h, r), items in grouped.items():
        items.sort(key=lambda x: x[1], reverse=True)
        for t, _ in items[:Kr]:
            capped.append((h, r, t))
    return capped


def multi_source_shortest_distances(
    G: nx.DiGraph,
    sources: List[str],
    max_depth: Optional[int] = None,
) -> Dict[str, int]:
    distances = {node: float('inf') for node in G.nodes}
    queue = deque()
    for src in sources:
        if src in G:
            distances[src] = 0
            queue.append(src)

    while queue:
        node = queue.popleft()
        depth = distances[node]
        if max_depth is not None and depth >= max_depth:
            continue
        for nbr in G.successors(node):
            if distances[nbr] > depth + 1:
                distances[nbr] = depth + 1
                queue.append(nbr)
    return distances


def reverse_shortest_distances(
    G: nx.MultiDiGraph,
    targets: List[str],
    max_depth: Optional[int] = None,
) -> Dict[str, int]:
    reverse_G = G.reverse(copy=False)
    return multi_source_shortest_distances(reverse_G, targets, max_depth)


def label_candidates_by_shortest_path(
    G: nx.MultiDiGraph,
    seeds: List[str],
    answer_entities: List[str],
    candidates_by_hop: List[List[Tuple[str, str, str]]],
) -> List[List[int]]:
    if len(answer_entities) == 0:
        return [[0] * len(cands) for cands in candidates_by_hop]

    dist_from_seed = multi_source_shortest_distances(G, seeds)
    dist_to_answer = reverse_shortest_distances(G, answer_entities)
    min_answer_distance = min(dist_from_seed.get(ans, float('inf')) for ans in answer_entities)

    labels_by_hop: List[List[int]] = []
    for hop_triples in candidates_by_hop:
        hop_labels = []
        for h, _, t in hop_triples:
            d_h = dist_from_seed.get(h, float('inf'))
            d_t = dist_to_answer.get(t, float('inf'))
            if d_h == float('inf') or d_t == float('inf'):
                hop_labels.append(0)
            elif d_h + 1 + d_t == min_answer_distance:
                hop_labels.append(1)
            else:
                hop_labels.append(0)
        labels_by_hop.append(hop_labels)
    return labels_by_hop


def build_query_dataset(
    query_records: List[Dict],
    kg_path: str,
    L: int = 3,
    Kr: int = 20,
) -> List[Dict]:
    G = build_graph(kg_path)
    tail_degree = compute_tail_degree(G)
    dataset: List[Dict] = []

    for record in query_records:
        query_text = record['query']
        seeds = record['seed_entities']
        answers = record['answer_entities']
        if not any(seed in G for seed in seeds):
            cands_by_hop = [[] for _ in range(L)]
            labels_by_hop = [[] for _ in range(L)]
        else:
            cands_by_hop = bfs_candidate_sets(G, seeds, L=L, Kr=Kr)
            labels_by_hop = label_candidates_by_shortest_path(G, seeds, answers, cands_by_hop)

        dataset.append({
            'query': query_text,
            'seed_entities': seeds,
            'answer_entities': answers,
            'hop_candidates': cands_by_hop,
            'hop_labels': labels_by_hop,
        })

    return dataset
