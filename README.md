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

Existing triple filters for multi-hop KG-RAG score each candidate from `(Query, relation, BFS_depth)` alone -- they are **blind** to which triples were retained at the previous hop. We prove this blindness incurs an **irreducible Bayes error floor** (epsilon\* > 0; Theorem 1, via the Data Processing Inequality). **CAFF** closes this gap with a three-piece, filtering-layer-only feedback loop:

- **CSV** -- a parameter-free, permutation-invariant summary of the previously retained set.
- **DBM** -- a low-rank, sigmoid-gated perturbation of the bilinear scoring matrix, *generated dynamically* from the CSV.
- **HC3** -- a contrastive loss that provably maximizes a variational lower bound on the conditional mutual information `I(Y; S | z)`.

## Repository Status

This is **reference implementation code** -- every component is type-annotated, paper-referenced (every function cites its equation in the paper), and unit-tested.

The pipeline runs end-to-end (KG loading -> BFS -> encoding -> CSV -> DBM -> scoring -> HC3 loss -> backward -> evaluation -> checkpointing). A built-in smoke test on a small synthetic graph validates this in roughly 60 seconds on CPU; see [Quick Smoke Test](#quick-smoke-test) below.

Producing the paper's reported metrics on PubMedQA / BioASQ requires:

1. The merged biomedical KG (`merged_kg.tsv`), built from Orphanet + DisGeNET + OMIM via `scripts/build_kg.py` (you must obtain the source data; see [Data](#data)).
2. A GPU. The paper trained on 1x A100-80GB; this repo includes a hardware-aware batch-sizing policy that adapts via gradient accumulation to preserve the paper's effective batch=256 on smaller GPUs (T4, L4, A100-40GB).

> **Reproducibility status (live document).**
> The numerical results reported in the paper -- PubMedQA F1, JSD bits, and the Context-Swap Diagnostic numbers in Appendix C -- were measured on the authors' hardware on the source biomedical data. This repository is being prepared for full public reproducibility; final checkpoints, training logs, and exact metric breakdowns will accompany the paper upon publication. Until then, anyone can verify the pipeline mechanics via the smoke test below.

## Quick Smoke Test

The fastest way to confirm the implementation runs end-to-end on your machine. Uses a synthetic 5,000-entity KG and `bert-base-uncased` (no GPU required).

```bash
# 1. Build synthetic data (one-off, ~5 seconds)
python tests/fixtures/build_smoke_data.py

# 2. Run a 2-epoch training pass (~3 minutes on CPU)
python train.py --config configs/caff_smoke.yaml --seed 42
```

Expected output (abridged):
```
KG loaded: |V|=5,000  |E|=24,995  |R|=20
Encoder loaded: 109.5M params (all frozen)
Built 20,982 triple instances (Class balance: 0.21% positive)
Trainer initialized: steps/epoch=43  warmup_steps=43
[Epoch   1/2] loss=0.0433  dev_f1=0.0000  dev_map=0.0136  lr=3.00e-04
[Epoch   2/2] loss=0.0426  dev_f1=0.0057  dev_map=0.0081  lr=1.15e-05
Training complete.
```

The smoke test does **not** measure paper-quality metrics (synthetic data has no semantic structure for the encoder to exploit). It validates that loss decreases, gradients propagate, the LR scheduler advances, dev evaluation runs, and checkpoints are written -- i.e. the pipeline is functional.

## Repository Layout

```
caff/
|-- caff/                          # Core implementation (paper sections 6-7)
|   |-- __init__.py
|   |-- config.py                  # CAFFConfig dataclass + validation
|   |-- encoders.py                # Frozen BioLinkBERT-Large
|   |-- csv.py                     # Eq. 14 -- Contextual Summary Vector
|   |-- dbm.py                     # Eqs. 16-17 -- Dynamic Bilinear Modulation
|   |-- scorer.py                  # Eqs. 18-19 -- Context-aware scorer
|   |-- model.py                   # Full CAFFModel
|   |-- losses.py                  # BCE + DC + HC3 (Eqs. 21-23)
|   |-- miners.py                  # HC3 triplet miner with rolling buffer
|   |-- data.py                    # KG loader, BFS, FreqCap (Eq. 13)
|   |-- trainer.py                 # Training engine
|   |-- evaluator.py               # P/R/F1/MAP/NDCG/PSR/JSD
|   `-- utils/
|       |-- seeding.py
|       `-- logging.py
|-- tests/
|   |-- fixtures/                  # Synthetic data + smoke-test builder
|   |   |-- build_smoke_data.py
|   |   `-- (generated *.json / *.tsv)
|   |-- test_csv.py
|   |-- test_dbm.py
|   |-- test_scorer.py
|   |-- test_losses.py
|   |-- test_miners.py
|   |-- test_data.py
|   `-- test_evaluator.py
|-- scripts/
|   |-- build_kg.py                # Merge raw biomedical sources -> KG TSV
|   |-- annotate_triples.py        # Gold-relevance annotation via BFS
|   `-- extract_bfs.py             # Stand-alone BFS dumper
|-- configs/
|   |-- caff_full.yaml             # Paper main config
|   |-- caff_no_hc3.yaml           # Ablation: lambda_C = 0
|   |-- caff_smoke.yaml            # Smoke test config (CPU-friendly)
|   `-- depthbilinear.yaml         # Baseline: no CSV, no DBM, no HC3
|-- examples/                      # Demo files (DO NOT cite)
|   |-- caff_demo.py               # Visualization with random embeddings
|   `-- caff_results.py            # Mock results display
|-- train.py                       # Main entrypoint
|-- evaluate.py
|-- context_swap_diagnostic.py     # Paper Appendix C
|-- requirements.txt               # Core dependencies
|-- requirements-optional.txt      # spacy/openai/wandb (optional features)
`-- README.md
```

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

Optional features (entity NER, OpenAI backbone, W&B tracking):
```bash
pip install -r requirements-optional.txt
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

The diagnostic measures the Jensen-Shannon Divergence between filtering distributions when the upstream context is permuted. A context-aware model produces a non-zero value; a context-blind baseline produces exactly 0 bits by construction. Paper Table 10 reports the exact reference numbers.

## Tests

Every paper component is unit-tested:
```bash
pytest tests/ -v
```

Tests verify:
- CSV: empty-set returns zero, permutation invariance, injectivity (Lemma 2)
- DBM: rank <= rho (Proposition 4), graceful degradation at z=0
- Scorer: Eq. 18 decomposition, S3 caching correctness
- HC3 miner: buffer capacity, k=8 negatives, refresh schedule
- Losses: hinge correctness, combined-loss decomposition (Eq. 22)
- Evaluator: P/R/F1, JSD = 0 for identical contexts (CBE signature)

## Key Hyperparameters (paper Section 8.4)

| Symbol     | Meaning                       | Default |
|------------|-------------------------------|--------:|
| `d`        | Embedding dim (BioLinkBERT-L) | 768     |
| `L`        | Max BFS hop depth             | 3       |
| `rho`      | DBM rank                      | 16      |
| `theta`    | Retention threshold           | 0.50    |
| `K_r`      | FreqCap per (head, relation)  | 20      |
| `gamma_C`  | HC3 margin                    | 0.25    |
| `gamma_D`  | DC margin                     | 0.20    |
| `lambda_C` | HC3 weight                    | 0.35    |
| `lambda_D` | DC weight                     | 0.40    |

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

MIT -- see [LICENSE](LICENSE). The merged KG derived from Orphanet, DisGeNET, and OMIM is **not redistributed**; users must obtain the source data under each provider's terms.

## Contact

| Role                 | Name              | Email                      |
|----------------------|-------------------|----------------------------|
| Corresponding author | Marwan Dhifallah  | marwan@mail.dlut.edu.cn    |
| Supervisor           | Prof. Yu Liu      | yuliu@dlut.edu.cn          |