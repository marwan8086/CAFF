import os
import pandas as pd
from typing import List, Tuple, Optional


def load_tsv_triples(path: str) -> List[Tuple[str, str, str]]:
    df = pd.read_csv(path, sep='\t', header=None, names=['head', 'relation', 'tail'], dtype=str)
    df = df.dropna()
    return list(df.itertuples(index=False, name=None))


def filter_relation_frequency(triples: List[Tuple[str, str, str]], min_count: int = 50):
    df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
    rel_counts = df['relation'].value_counts()
    keep_rels = rel_counts[rel_counts >= min_count].index
    df = df[df['relation'].isin(keep_rels)]
    return list(df.itertuples(index=False, name=None))


def build_merged_kg(
    orphanet_path: Optional[str] = None,
    disgenet_path: Optional[str] = None,
    omim_path: Optional[str] = None,
    output_path: str = './data/merged_kg.tsv',
    min_relation_count: int = 50,
):
    triples = []
    if orphanet_path and os.path.exists(orphanet_path):
        triples.extend(load_tsv_triples(orphanet_path))
    if disgenet_path and os.path.exists(disgenet_path):
        triples.extend(load_tsv_triples(disgenet_path))
    if omim_path and os.path.exists(omim_path):
        triples.extend(load_tsv_triples(omim_path))

    triples = list(set(triples))
    triples = filter_relation_frequency(triples, min_relation_count)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
    df.to_csv(output_path, sep='\t', index=False, header=False)
    return output_path
