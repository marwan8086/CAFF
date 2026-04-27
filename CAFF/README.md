<h1 align="center">
  CAFF: Context-Aware Feedback Filtering<br>
  for Multi-Hop Biomedical Knowledge Graph Evidence Selection
</h1>

<p align="center">
  <a href="#"><img alt="Paper" src="https://img.shields.io/badge/Paper-IEEE%20TKDE%20(under%20review)-1f6feb?style=flat-square"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-2ea44f?style=flat-square"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
</p>

<p align="center">
  <b>Marwan Dhifallah</b><sup>*</sup> &middot; <b>Yu Liu</b><br>
  <i>Dalian University of Technology, Dalian, China</i>
</p>

---

> Reference implementation accompanying the paper
> **"CAFF: Context-Aware Feedback Filtering for Multi-Hop Biomedical Knowledge Graph Evidence Selection"**
> (under review, IEEE Transactions on Knowledge and Data Engineering, 2026).

## TL;DR

Existing triple filters for multi-hop KG-RAG score each candidate from `(Query, relation, BFS_depth)` alone — they are **blind** to which triples were retained at the previous hop. We prove this blindness incurs an **irreducible Bayes error floor** ε* > 0 (Theorem 1, via the Data Processing Inequality). **CAFF** closes this gap with a three-piece, filtering-layer-only feedback loop:

- **CSV** — a parameter-free, permutation-invariant summary of the previously retained set.
- **DBM** — a low-rank, sigmoid-gated perturbation of the bilinear scoring matrix, *generated dynamically* from the CSV.
- **HC3** — a contrastive loss that provably maximizes a variational lower bound on the conditional mutual information `I(Y; S | z)`.

## Repository Status

This is **reference implementation code** — every component is type-annotated, paper-referenced (every function cites its equation in the paper), and unit-tested. Running it end-to-end requires:

1. The merged KG (`merged_kg.tsv`), built from Orphanet + DisGeNET + OMIM via `scripts/build_kg.py` (you must obtain the source data; see [Data](#data)).
2. A GPU. The paper trained on 1× A100-80GB; this repo includes a hardware-aware batch-sizing policy that adapts via gradient accumulation to preserve the paper's effective batch=256 on smaller GPUs (T4, L4, A100-40GB).

> **Reproducibility status.** The numerical results reported in the paper (PubMedQA 79.6%, JSD 1.84 bits, etc.) were measured on a specific hardware configuration. Running this code on different hardware will produce different — but qualitatively similar — numbers. Final checkpoints, training logs, and exact metric breakdowns will accompany the paper upon publication.

## Repository Layout


caff/
├── caff/
│   ├── config.py              # CAFFConfig, AblationFlags
│   ├── encoders.py            # Frozen BioLinkBERT-Large + relation cache
│   ├── csv.py                 # Eq. 14   — Contextual Summary Vector
│   ├── dbm.py                 # Eqs. 16-17 — Dynamic Bilinear Modulation
│   ├── scorer.py              # Eqs. 18-19 — Context-aware scoring
│   ├── model.py               # Full CAFFModel + Algorithm 1
│   ├── losses.py              # Eqs. 21-23 — BCE + DC + HC3
│   ├── miners.py              # §6.4      — HC3 rolling buffer
│   ├── data.py                # Eqs. 1, 13 — KG, BFS, FreqCap
│   ├── trainer.py             # AdamW, warmup-cosine, grad-accum
│   └── evaluator.py           # P/R/F1, MAP, NDCG, hop-strat, PSR, JSD
├── scripts/
│   ├── build_kg.py            # Merge Orphanet/DisGeNET/OMIM on UMLS CUIs
│   └── annotate_triples.py    # Shortest-path gold-relevance labeling
├── configs/
│   ├── caff_full.yaml         # Paper §8.4 defaults
│   ├── caff_no_hc3.yaml       # Ablation: HC3 disabled
│   └── depthbilinear.yaml     # Baseline: depth-stratified, no context
├── tests/                     # Unit tests for every paper component
├── train.py                   # Entry point: training
├── evaluate.py                # Entry point: evaluation + bootstrap
├── context_swap_diagnostic.py # Entry point: paper Appendix C
├── requirements.txt
├── LICENSE
└── README.md

## Installation

```bash
git clone https://github.com/marwan8086/caff.git
cd caff
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the matching PyTorch build (CUDA 11.8 example):
```bash
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

## Data

CAFF operates on a merged biomedical KG joined on UMLS Concept Unique Identifiers (CUIs).

| Source     | Used for                       | Access                                |
|------------|--------------------------------|---------------------------------------|
| Orphanet   | Rare-disease ontology          | <https://www.orphadata.com>           |
| DisGeNET   | Gene-disease associations      | <https://www.disgenet.org>            |
| OMIM       | Mendelian inheritance          | <https://www.omim.org>                |
| UMLS       | CUI normalization              | <https://uts.nlm.nih.gov>             |
| PubMedQA   | QA benchmark (1,000 questions) | <https://pubmedqa.github.io>          |
| BioASQ 7b  | QA benchmark (1,141 instances) | <http://bioasq.org>                   |

This repository **does not redistribute** the source data. After obtaining each, build the merged KG:

```bash
python scripts/build_kg.py \
    --orphanet  data/raw/orphanet/ \
    --disgenet  data/raw/disgenet/all_gene_disease_associations.tsv \
    --omim      data/raw/omim/ \
    --umls      data/raw/umls/MRCONSO.RRF \
    --out       data/processed/merged_kg.tsv \
    --min-relation-freq 50
```

Then annotate gold relevance:
```bash
python scripts/annotate_triples.py \
    --kg          data/processed/merged_kg.tsv \
    --qa-input    data/raw/pubmedqa_with_seeds.json \
    --qa-output   data/processed/train.json \
    --cache-dir   cache/bfs/
```

## Training

```bash
python train.py --config configs/caff_full.yaml --seed 42
```

To reproduce the paper's three-seed protocol:
```bash
for s in 42 1337 2024; do
    python train.py --config configs/caff_full.yaml --seed $s
done
```

## Evaluation

```bash
python evaluate.py \
    --checkpoint runs/caff_full/seed_42/best.pt \
    --report-bootstrap-vs runs/depthbilinear/seed_42/best.pt \
    --output-json results/caff_full_seed_42.json
```

## Context-Swap Diagnostic (Paper Appendix C)

The single most important experiment for verifying CBE elimination:

```bash
python context_swap_diagnostic.py --checkpoint runs/caff_full/seed_42/best.pt
```

Expected (paper Table 10):
- Any context-agnostic baseline → **0.00 bits**
- CAFF-NoHC3 → **1.41 bits**
- CAFF (Full) → **1.84 bits**

## Tests

Every paper component is unit-tested:
```bash
pytest tests/ -v
```

Tests verify:
- CSV: empty-set returns zero, permutation invariance, injectivity (Lemma 2)
- DBM: rank ≤ ρ (Proposition 4), graceful degradation at z=0
- Scorer: Eq. 18 decomposition, S3 caching correctness
- HC3 miner: buffer capacity, k=8 negatives, refresh schedule
- Losses: hinge correctness, combined-loss decomposition (Eq. 22)
- Evaluator: P/R/F1, JSD = 0 for identical contexts (CBE signature)

## Key Hyperparameters (paper §8.4)

| Symbol     | Meaning                       | Default |
|------------|-------------------------------|--------:|
| `d`        | Embedding dim (BioLinkBERT-L) | 768     |
| `L`        | Max BFS hop depth             | 3       |
| `ρ`        | DBM rank                      | 16      |
| `θ`        | Retention threshold           | 0.50    |
| `K_r`      | FreqCap per (head, relation)  | 20      |
| `γ_C`      | HC3 margin                    | 0.25    |
| `γ_D`      | DC margin                     | 0.20    |
| `λ_C`      | HC3 weight                    | 0.35    |
| `λ_D`      | DC weight                     | 0.40    |

## Citation

```bibtex
@article{dhifallah2026caff,
  title   = {{CAFF}: Context-Aware Feedback Filtering for Multi-Hop
             Biomedical Knowledge Graph Evidence Selection},
  author  = {Dhifallah, Marwan and Liu, Yu},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE). The merged KG derived from Orphanet, DisGeNET, and OMIM is **not redistributed**; users must obtain the source data under each provider's terms.

## Contact

| Role                 | Name              | Email                      |
|----------------------|-------------------|----------------------------|
| Corresponding author | Marwan Dhifallah  | marwan@mail.dlut.edu.cn    |
| Supervisor           | Prof. Yu Liu      | yuliu@dlut.edu.cn          |