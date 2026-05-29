<!-- ============================================================ -->
<!--  CAFF - Context-Aware Feedback Filtering                     -->
<!--  Official repository README                                  -->
<!-- ============================================================ -->

<h1 align="center">
  CAFF: Context-Aware Feedback Filtering for Multi-Hop Biomedical Knowledge Graph Evidence Selection
</h1>

<p align="center">
  <a href="#"><img alt="Paper" src="https://img.shields.io/badge/Paper-IEEE%20TKDE%20(under%20review)-1f6feb?style=flat-square"></a>
  <a href="#license"><img alt="License" src="https://img.shields.io/badge/License-MIT-2ea44f?style=flat-square"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-11.8-76B900?style=flat-square&logo=nvidia&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Research%20Code-orange?style=flat-square">
</p>

<p align="center">
  <b>Marwan Dhifallah</b><sup>*</sup> &nbsp;.&nbsp; <b>Yu Liu</b><br>
  <i>Dalian University of Technology, Dalian, China</i><br>
  <code>marwan@mail.dlut.edu.cn</code> &nbsp;.&nbsp; <code>yuliu@dlut.edu.cn</code>
</p>

<p align="center">
  <i>Official PyTorch implementation of the paper</i><br>
  <b>"CAFF: Context-Aware Feedback Filtering for Multi-Hop Biomedical Knowledge Graph Evidence Selection"</b><br>
  <i>(under review, IEEE Transactions on Knowledge and Data Engineering, 2026).</i>
</p>

> **Status note (May 2026).** This README reflects the **as-measured** state
> of the codebase. All numbers reported below are reproduced from
> experiments documented in `PAPER_DISCREPANCIES.md` (Sections 1-26). The
> paper's original headline (PubMedQA 79.6, BioASQ 74.3) was framed on a
> larger four-source KG (Orphanet+DisGeNET+OMIM+UMLS, |V|=148K) with an
> A100-80GB. This repository ships a consumer-hardware reproduction path
> on the smaller Orphanet+HPO+OMIM KG (|V|=38K), with all results measured
> on a held-out Orphanet QA test set.

---

## Table of Contents

1. [TL;DR](#tldr)
2. [The Context Blindness Error (CBE)](#the-context-blindness-error-cbe)
3. [Key Contributions](#key-contributions)
4. [Method Overview](#method-overview)
5. [Theoretical Background](#theoretical-background)
6. [Repository Structure](#repository-structure)
7. [Installation](#installation)
8. [Data Preparation](#data-preparation)
9. [Training](#training)
10. [Evaluation](#evaluation)
11. [Main Results (Measured)](#main-results-measured)
12. [Ablation Study (Measured)](#ablation-study-measured)
13. [Configurations](#configurations)
14. [Hyperparameters](#hyperparameters)
15. [Reproducibility](#reproducibility)
16. [Hardware Requirements](#hardware-requirements)
17. [Limitations and Future Work](#limitations-and-future-work)
18. [Citation](#citation)
19. [License](#license)
20. [Acknowledgements](#acknowledgements)
21. [Contact](#contact)

---

## TL;DR

> Existing triple filters for multi-hop KG-RAG score each candidate from
> `(Query, relation, BFS_depth)` alone; they are **blind** to which triples
> were retained at the previous hop. The paper proves this blindness incurs
> an **irreducible** Bayes error floor `eps* > 0` (Theorem 1, via the Data
> Processing Inequality). **CAFF** addresses this with a three-piece,
> filtering-layer-only feedback loop:
>
> - **CSV** -- a parameter-free, permutation-invariant summary of the previously retained set.
> - **DBM** -- a low-rank, sigmoid-gated perturbation of the bilinear scoring matrix, generated dynamically from the CSV.
> - **HC3** -- a contrastive loss that targets a variational lower bound on the conditional mutual information `I(Y; S | z)`.
>
> **Measured headline on Orphanet QA (this repository, held-out test set, 3 seeds, paired bootstrap):**
>
> | Configuration                | Test F1 (per-hop) | Significance vs. previous default |
> |------------------------------|------------------:|-----------------------------------|
> | Previous default (Full CAFF, all components) | 0.5524 +/- 0.0016 | -- |
> | **Current default (No-DC: CSV + DBM, DC disabled)** | **0.5764 +/- 0.0022** | **delta_AP = +0.025, p < 0.01** |
>
> The Day 14-15 ablation showed that **CSV and DBM are the essential
> components**: removing either costs about 0.046 F1. The
> DepthContrastive (DC) loss as specified in the paper (lambda_D=0.40)
> measurably **hurts** held-out F1 on this KG; the HC3 loss is **inert**
> (zero gradient on this data). Full breakdown in
> [Ablation Study](#ablation-study-measured) and
> `PAPER_DISCREPANCIES.md` Sections 22-26.

---

## The Context Blindness Error (CBE)

Consider the clinical query:

> *"What drug targets the pathway of the causal gene of Fanconi anemia complementation group D1?"*

The same hop-2 triple `<BRCA2, participates_in, HR-repair>` is:

- **Relevant** when the hop-1 retained set is `{<Fanconi anemia D1, caused_by, BRCA2>}`.
- **Irrelevant** when the hop-1 retained set is `{<Fanconi anemia D1, caused_by, BRIP1>}`.

A filter that sees only `(Query, relation, hop)` cannot distinguish these two cases. The paper formalizes this as the **Context Blindness Error** and proves a lower bound on its expected loss (Theorem 1). CAFF closes the gap by feeding a summary of the previous hop's retained set into the current hop's scorer.

---

## Key Contributions

The paper proposes four contributions: CSV, DBM, HC3, and a depth-contrastive auxiliary loss (DC). After full leave-one-out ablation in this repository (3 seeds, paired bootstrap), the empirical picture is:

| Component | Role (as designed) | Effect on held-out F1 (measured) | Status |
|-----------|--------------------|---------------------------------:|--------|
| **CSV** | Context vector from previous retained set | Removal costs **-0.047** F1 | Essential |
| **DBM** | Low-rank dynamic gating of the bilinear scorer | Removal costs **-0.046** F1 | Essential |
| **DC** | Depth-contrastive auxiliary loss (lambda_D=0.40) | Removal **gains +0.024** F1 (p < 0.01) | **Harmful at paper config** |
| **HC3** | CMI-bound contrastive loss | Removal changes nothing | **Inert** (zero gradient) |

CSV + DBM are validated as the real architectural contribution. DC and HC3 are documented as **negative results** of this measurement campaign; see `PAPER_DISCREPANCIES.md` Sections 22-26 for the full evidence trail.

---

## Method Overview

CAFF is a four-stage pipeline applied during multi-hop evidence retrieval.

### Stage 1 -- BFS Candidate Stratification

A BFS from the query seed entities collects all triples reachable within `L=3` hops, stratified by hop depth. A per-relation frequency cap (`K_r=20`, configurable) prevents any one relation from saturating a candidate set.

### Stage 2 -- Contextual Summary Vector (CSV)

For each hop `ell > 1`, the retained set from hop `ell-1` is summarized by a parameter-free, permutation-invariant pool over the relation embeddings of its triples:

```
z_{ell-1} = pool({ E[r] : (h, r, t) in S_{ell-1} })
```

Default pool is `mean`. At `ell=1`, `z_0 = 0`.

### Stage 3 -- Dynamic Bilinear Modulation (DBM)

The base hop-conditioned bilinear scorer

```
s_base(Q, r, ell) = Q^T W_ell E[r]
```

is augmented with a low-rank, context-dependent perturbation generated from `z_{ell-1}`:

```
Delta_ell(z) = sigmoid(U z) * (A z) (B z)^T,  rank rho << d
s_CAFF(Q, r, ell, z) = Q^T (W_ell + Delta_ell(z)) E[r]
```

This is the only context-aware path in the architecture; `Delta_ell` adds about 0.8 M parameters at `rho=16, d=1024`.

### Stage 4 -- Auxiliary losses (HC3 and DC)

The paper adds two auxiliary losses on top of the BCE filter loss:

- **HC3** (margin-based contrastive loss targeting a CMI lower bound on `I(Y; S | z)`)
- **DC** (depth-contrastive hinge: `max(0, s_wrong_hop - s_correct_hop + gamma_D)`)

**Measured behavior on this codebase:** HC3 produces zero gradient
under the paper's setting because keys collide between positives and
negatives at the teacher-forced training step (`PAPER_DISCREPANCIES.md`
Section 22). DC is gradient-active but its sign of effect is **negative**
on held-out F1 (Sections 25-26). The codebase exposes both losses via
ablation flags (`use_hc3`, `use_dc`); the current default disables DC.

---

## Theoretical Background

The paper develops two results that survive empirically:

1. **CBE lower bound (Theorem 1).** Any filter that sees `(Q, r, ell)` only has Bayes error at least `eps*`, where `eps*` is bounded below by `I(Y; S | Q, r, ell)`. CSV+DBM provide a sufficient channel through which `S_{ell-1}` can influence the score function.

2. **DPI under context conditioning.** Conditioning on a sufficient context vector `z` of the retained set is at least as good (in DPI terms) as conditioning on the raw set, justifying the CSV pool plus DBM modulation rather than per-element attention.

The CMI bound that motivates HC3 is preserved in the paper as theoretical background. The measured behavior of HC3 on this codebase is reported under [Key Contributions](#key-contributions) and Section 22 of the discrepancies log.

---

## Repository Structure

```
CAFF/
|-- caff/                            # Core package (importable)
|   |-- __init__.py                  # Public API surface
|   |-- config.py                    # CAFFConfig + AblationFlags dataclasses
|   |-- csv.py                       # Contextual Summary Vector (Stage 2)
|   |-- data.py                      # KG loader, BFS extractor, datasets
|   |-- dbm.py                       # Dynamic Bilinear Modulation (Stage 3)
|   |-- encoders.py                  # Frozen encoder + relation cache
|   |-- evaluator.py                 # Metrics, MAP / NDCG, threshold tuning
|   |-- losses.py                    # BCE + DC + HC3 loss objects
|   |-- miners.py                    # DCMiner + HC3Miner + buffers
|   |-- model.py                     # CAFFModel (CSV + DBM + scoring head)
|   |-- scorer.py                    # DepthBilinear + HopScorer
|   |-- trainer.py                   # CAFFTrainer + CheckpointManager
|   `-- utils/                       # seeding, logging
|
|-- scripts/                         # Reproduction pipeline
|   |-- convert_orphanet_xml_to_tsv.py   # Orphanet XML -> TSV
|   |-- convert_hpo_to_tsv.py            # HPO obo -> TSV
|   |-- convert_mondo_to_tsv.py          # MONDO obo -> TSV (experimental)
|   |-- build_kg.py                      # Base KG from Orphanet TSV
|   |-- merge_hpo_into_kg.py             # KG v2 = + HPO/OMIM annotations
|   |-- merge_mondo_into_kg.py           # KG v3 = + MONDO (experimental)
|   |-- build_orphanet_qa.py             # Sample QA records from KG
|   |-- annotate_triples.py              # Shortest-path gold annotation
|   |-- extract_bfs.py                   # Precompute BFS candidates
|   |-- threshold_sweep.py               # Find optimal global theta
|   |-- per_hop_threshold_sweep.py       # Per-hop theta tuning
|   |-- probe_hc3_keys.py                # HC3 diagnostic (Section 22)
|   |-- probe_hc3_model.py               # HC3 diagnostic (Section 22)
|   `-- apply_hc3_fix_FINAL.py           # HC3 cross-query variant (Section 23)
|
|-- configs/                         # YAML training configs (see Configurations below)
|   |-- no_dc.yaml                   # CURRENT DEFAULT (headline, F1=0.5764)
|   |-- caff_orphanet.yaml           # Previous default (Full CAFF, F1=0.5524)
|   |-- caff_full.yaml               # DEPRECATED: paper-spec, never trained, d=768 buggy
|   |-- caff_no_hc3.yaml             # Ablation (HC3 off)
|   |-- no_csv.yaml                  # Ablation (CSV off)
|   |-- no_dbm.yaml                  # Ablation (DBM off)
|   |-- no_freqcap.yaml              # Ablation (FreqCap off)
|   |-- depthbilinear.yaml           # Baseline (all components off)
|   `-- caff_smoke.yaml              # Smoke test (tiny synthetic KG, bert-base, 2 epochs)
|
|-- tests/                           # Unit tests (run by CI)
|   |-- fixtures/                    # Tiny synthetic KG
|   `-- test_*.py                    # Per-module tests
|
|-- .github/workflows/
|   `-- tests.yml                    # CI: lint + pytest on every push
|
|-- data/                            # gitignored (raw + processed)
|-- runs/                            # gitignored (checkpoints, logs)
|-- cache/                           # gitignored (BFS + relation cache)
|-- examples/                        # short usage snippets
|
|-- train.py                         # Training entry point
|-- evaluate.py                      # Standalone evaluation script
|
|-- README.md                        # This file
|-- CONTRIBUTING.md                  # Contribution guidelines
|-- PAPER_DISCREPANCIES.md           # 26-section running experiment log
|-- LICENSE                          # MIT
|-- requirements.txt                 # Core dependencies
|-- requirements-optional.txt        # Optional dependencies
|-- .gitattributes
`-- .gitignore
```

`PAPER_DISCREPANCIES.md` is the source of truth for every empirical
decision in the repo. New experiments append a new numbered section
there before changing the headline numbers.

---

## Installation

### Prerequisites

- **Python** >= 3.10
- **CUDA** 11.8 (a single consumer 8 GB GPU is sufficient for this codebase; the paper used A100-80GB)
- **Git LFS** (optional, only if downloading released checkpoints)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/caff.git
cd caff

# 2. Create a clean virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install PyTorch (matched to your CUDA toolkit)
pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. (Optional) Download the BioLinkBERT-Large encoder
python -c "from transformers import AutoModel; AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-large')"
```

### Core Dependencies

```
torch>=2.0
transformers>=4.30
networkx>=3.0
numpy, scipy, scikit-learn, pandas
tqdm, pyyaml
```

`scispacy` and `openai` are listed in `requirements-optional.txt`; they are not required to run the measured pipeline in this repository.

---

## Data Preparation

This repository ships a reproducible pipeline using **public, non-credentialed** biomedical sources. The paper's larger KG (with DisGeNET and UMLS) is referenced for completeness, but those sources require institutional access and are **not used** in this repository's measured results.

### Sources used in this repository

| Source | Used for | Access |
|--------|----------|--------|
| **Orphanet** (2024) | Rare-disease ontology, gene-disease links | <https://www.orphadata.com> |
| **HPO** | Phenotype-to-disease relations | <https://hpo.jax.org> |
| **OMIM** | Mendelian inheritance, gene-phenotype (annotations only) | <https://www.omim.org> |

### Build the merged KG (v2: Orphanet + HPO + OMIM annotations)

```bash
# 1. Convert raw ontologies to TSV
python scripts/convert_orphanet_xml_to_tsv.py --in data/raw/orphanet/ --out data/processed/orphanet.tsv
python scripts/convert_hpo_to_tsv.py          --in data/raw/hpo/hp.obo  --out data/processed/hpo.tsv

# 2. Build base KG and merge in HPO/OMIM
python scripts/build_kg.py            --orphanet data/processed/orphanet.tsv --out data/processed/merged_kg.tsv
python scripts/merge_hpo_into_kg.py   --in data/processed/merged_kg.tsv --hpo data/processed/hpo.tsv --out data/processed/merged_kg_v2.tsv

# 3. Sample QA records from the KG
python scripts/build_orphanet_qa.py --kg data/processed/merged_kg_v2.tsv --n 20000 --out data/processed/

# 4. Pre-compute BFS candidates and gold annotations
python scripts/extract_bfs.py       --kg data/processed/merged_kg_v2.tsv --L 3 --K_r 20
python scripts/annotate_triples.py  --kg data/processed/merged_kg_v2.tsv --qa data/processed/
```

After construction, the KG used in all measured results here is:

| Property                | Value (this repository) | Paper's larger KG |
|-------------------------|-------------------------|-------------------|
| Entities `|V|`          | **38,456**              | 148,423           |
| Triples `|E|`           | **291,335**             | 2,318,941         |
| Relation types `|R|`    | **11** (after `min_relation_freq=50`) | 42 |
| Max BFS depth `L`       | 3                       | 3                 |
| QA records (train/dev/test) | 14,000 / 3,000 / 3,000 | varies            |

> **Note on PubMedQA and BioASQ.** The paper headline uses PubMedQA and
> BioASQ 7b for end-to-end QA evaluation. This repository does **not**
> include those benchmarks; all measured numbers below are on the
> Orphanet-derived QA test set. PubMedQA/BioASQ evaluation is listed
> under [Limitations](#limitations-and-future-work).

---

## Training

The current headline configuration is `configs/no_dc.yaml`. To reproduce it on three seeds:

```bash
for s in 42 1337 2024; do
    python train.py --config configs/no_dc.yaml --seed $s
done
```

Each seed takes about 40 minutes on an 8 GB consumer GPU (RTX 4060). The `train.py` script auto-detects CUDA and applies sensible hardware overrides (`micro_batch_size=4, grad_accum_steps=64, mixed_precision=fp16`, effective batch 256).

### Reproduce the ablation suite

```bash
# Full leave-one-out ablation (3 seeds each)
for cfg in caff_orphanet no_dc no_csv no_dbm no_freqcap caff_no_hc3 depthbilinear; do
    for s in 42 1337 2024; do
        python train.py --config configs/$cfg.yaml --seed $s
    done
done

# Per-hop threshold tuning on each variant
for cfg in caff_orphanet no_dc no_csv no_dbm no_freqcap depthbilinear; do
    for s in 42 1337 2024; do
        python scripts/per_hop_threshold_sweep.py \
            --config configs/$cfg.yaml \
            --checkpoint runs/$cfg/seed_$s/best.pt
    done
done
```

### Key training hyperparameters (no_dc.yaml, the current default)

| Hyperparameter | Default | Notes |
|----------------|--------:|-------|
| Optimizer | AdamW | weight decay 1e-2 |
| Base learning rate | 3e-4 | cosine decay to 1e-5, 1-epoch warmup |
| Effective batch size | 256 | micro=4, accum=64 on 8 GB GPU |
| Epochs | 10 | early stopping on dev F1, patience 5 |
| Gradient clip | 1.0 | |
| Random seeds | {42, 1337, 2024} | three runs reported |
| Encoder | BioLinkBERT-large (340 M, frozen) | output dim 1024 |

---

## Evaluation

### Filtering-layer metrics (per-hop test F1, MAP, NDCG@10)

```bash
python scripts/per_hop_threshold_sweep.py \
    --config configs/no_dc.yaml \
    --checkpoint runs/no_dc/seed_42/best.pt
```

This script tunes per-hop thresholds on the dev set, then reports F1 / precision / recall on the held-out test set under three regimes: global theta=0.50, global theta=0.80, and per-hop tuned thresholds.

### Paired-bootstrap significance test

```bash
python evaluate.py \
    --checkpoint runs/no_dc/seed_42/best.pt \
    --report-bootstrap-vs runs/caff_orphanet/seed_42/best.pt \
    --mode autoregressive \
    --output-json results/bench_no_dc_vs_full_seed_42.json
```

Outputs the test-set metrics (F1, MAP, NDCG@10, per-hop precision) at `theta=0.80` in autoregressive (no gold leakage) inference mode, plus a paired bootstrap on per-query AP versus the baseline checkpoint (10,000 resamples).

The end-to-end QA evaluation with an LLM backbone (paper Section 9.2)
is **not implemented** in this repository; see
[Limitations](#limitations-and-future-work).

---

## Main Results (Measured)

All numbers below are measured on the held-out Orphanet QA test set
(3,000 queries, 102,317 candidate triples, never used in training or
threshold tuning). Three seeds, deterministic.

### Headline (No-DC, the current default)

| Metric                     | Mean +/- std       | Mode |
|----------------------------|---------------------|------|
| **Test F1 (per-hop)**      | **0.5764 +/- 0.0022** | teacher-forced |
| Test F1 (autoregressive)   | 0.5477 +/- 0.0006   | autoregressive (no gold leakage) |
| Test MAP                   | 0.6741 +/- 0.0003   | autoregressive |
| Test NDCG@10               | 0.7090 +/- 0.0003   | autoregressive |
| Hop-1 precision            | 0.8234 +/- 0.0047   | autoregressive |
| Hop-2 precision            | 0.4378 +/- 0.0006   | autoregressive |
| Hop-3 precision            | 0.2426 +/- 0.0026   | autoregressive |

### Comparison to previous default

| Configuration | Test F1 (per-hop) | Test F1 (autoregressive) | delta_AP vs No-DC | p-value |
|---------------|------------------:|--------------------------:|-------------------|--------:|
| **No-DC (current default)** | **0.5764 +/- 0.0022** | **0.5477 +/- 0.0006** | -- | -- |
| Full CAFF (previous default) | 0.5524 +/- 0.0016 | not separately rerun | -0.0250 +/- 0.0039 | < 0.01 |

Paired bootstrap on per-query AP with 10,000 resamples, computed per
seed; all three 95% confidence intervals exclude zero, and p < 0.01 on
every seed. Full benchmark in `PAPER_DISCREPANCIES.md` Section 26.

---

## Ablation Study (Measured)

Leave-one-out over every component the paper proposes, plus the
strongest depth-stratified baseline, on the held-out Orphanet test set.
Three seeds per variant; per-hop test F1 with thresholds tuned on dev.

| Variant | dev_f1 | global theta=0.80 | per-hop test F1 | vs Full CAFF |
|---------|-------:|------------------:|----------------:|-------------:|
| **No-DC (current default)** | **0.5662** | **0.5787** | **0.5764 +/- 0.0022** | **+0.0240** |
| Full CAFF (previous default) | 0.5099 | 0.5315 | 0.5524 +/- 0.0016 | -- |
| No-HC3 (HC3 off) | 0.5099 | 0.5315 | 0.5524 (identical to Full) | 0.0000 |
| No-FreqCap (frequency cap off) | 0.5099 | 0.5315 | 0.5524 (identical to Full) | 0.0000 |
| No-DBM (DBM off) | 0.4535 | 0.4878 | 0.5063 +/- 0.0046 | -0.0461 |
| No-CSV (CSV off) | 0.4540 | 0.4821 | 0.5054 +/- 0.0027 | -0.0470 |
| DepthBilinear (all off) | 0.5135 | 0.5250 | 0.4966 +/- 0.0121 | -0.0558 |

Take-aways:

1. **CSV and DBM are the real architectural contribution.** They form a coupled pair (CSV produces `z`, DBM consumes it); removing either breaks the context-aware path and costs about 0.046 F1.
2. **DC hurts at the paper configuration (lambda_D=0.40).** No-DC outperforms Full on every metric (dev, global theta, per-hop, autoregressive, paired bootstrap). Section 25-26 of `PAPER_DISCREPANCIES.md` records the full evidence including the code-level verification that No-DC differs from Full only in DC (loss-only, identical forward pass).
3. **HC3 is inert.** The training-time loss produces zero gradient under the paper's setting because keys collide between teacher-forced positives and negatives; an attempted cross-query fix raised the gradient norm but did not change held-out F1 (Sections 22-23).
4. **FreqCap is inert here.** The Orphanet+HPO+OMIM KG has only 11 relations after `min_relation_freq=50`, so the per-relation cap has nothing to act on.

---

## Configurations

This repository ships nine YAML configs under `configs/`. The first
four cover the headline plus the previous default; the next four are
ablations; the last is for CI smoke-testing.

| Config file | Purpose | Trained in this repo? | Per-hop test F1 (3 seeds) |
|-------------|---------|:---------------------:|--------------------------:|
| `no_dc.yaml` | **Current headline** (CSV + DBM, DC off) | Yes | **0.5764 +/- 0.0022** |
| `caff_orphanet.yaml` | Previous headline (Full CAFF) | Yes | 0.5524 +/- 0.0016 |
| `caff_no_hc3.yaml` | Ablation (HC3 off) | Yes | = Full (HC3 inert) |
| `no_freqcap.yaml` | Ablation (FreqCap off) | Yes | = Full (FreqCap inert) |
| `no_csv.yaml` | Ablation (CSV off, lower bound) | Yes | 0.5054 +/- 0.0027 |
| `no_dbm.yaml` | Ablation (DBM off, lower bound) | Yes | 0.5063 +/- 0.0046 |
| `depthbilinear.yaml` | Baseline (all components off) | Yes | 0.4966 +/- 0.0121 |
| `caff_smoke.yaml` | CI smoke test (tiny KG, 2 epochs, bert-base) | Yes (CI) | n/a |
| `caff_full.yaml` | **DEPRECATED**: paper-spec config, never trained in this repo (`d=768` does not match BioLinkBERT-Large output dim 1024). Kept for reference. | **No** | n/a |

To re-run any of these, see [Training](#training).

---

## Hyperparameters

The current headline (`no_dc.yaml`) uses:

| Symbol | Meaning | Value |
|--------|---------|------:|
| `d` | Embedding dimension (BioLinkBERT-Large output) | 1024 |
| `L` | Maximum BFS hop depth | 3 |
| `rho` | DBM rank | 16 |
| `theta` | Retention threshold (global default) | 0.80 |
| `K_r` | Frequency cap per relation per head | 20 |
| `gamma_C` | HC3 margin | 0.25 |
| `gamma_D` | Depth-contrastive margin | 0.20 |
| `lambda_C` | HC3 loss weight | 0.35 (loss inert at this value) |
| `lambda_D` | DC loss weight | **0.0** (current default; paper specifies 0.40) |
| `min_relation_freq` | Drop singleton relations at KG load | 50 |

The values for `gamma_C`, `lambda_C`, `gamma_D` are kept identical to the paper so the inertness of HC3 and the harmfulness of DC can be reproduced under the paper's own setting. A lambda_D sweep (whether smaller positive values rescue DC) is listed under future work.

---

## Reproducibility

- All reported results are **mean across three seeds** `{42, 1337, 2024}`.
- Training is deterministic (`config.deterministic = true`); the same seed produces bit-identical results across runs on the same hardware.
- Standard deviations are reported in every table above.
- Hardware: NVIDIA Studio Driver 596.36, RTX 4060 Laptop 8 GB, PyTorch 2.2.1 + CUDA 11.8, Windows 11.
- Checkpoints under `runs/<config_name>/seed_<seed>/best.pt`.
- Per-seed benchmark JSON outputs under `results/bench_no_dc_vs_full_seed_<seed>.json`.

---

## Hardware Requirements

| Stage | Minimum | This repo's reference | Paper's reference |
|-------|---------|------------------------|-------------------|
| KG build + BFS | 8 GB RAM, any CPU | i9-13900H, 32 GB | -- |
| Training (per seed) | 8 GB consumer GPU | RTX 4060 Laptop, ~40 min | A100-80GB, ~80 min |
| Evaluation | 8 GB consumer GPU | RTX 4060, ~2 min | -- |
| End-to-end QA with LLM | not implemented in this repo | -- | A100 + GPT-3.5-turbo API |

CPU-only training is supported via `train.py`'s automatic hardware
override (`micro_batch_size=8, grad_accum_steps=32`), but a CPU-only
run with the BioLinkBERT-Large encoder is impractical (the encoder
takes ~9 GB of CPU RAM and inference is roughly 8x slower than on the
8 GB GPU).

---

## Limitations and Future Work

This repository is honest about what is **not** measured here, so a
reader can decide whether the gaps matter for their use case.

1. **No PubMedQA or BioASQ evaluation.** The paper's headline accuracy numbers (PubMedQA 79.6, BioASQ 74.3) are end-to-end QA results on those benchmarks. This repository does not include those datasets or their QA pipelines. Adapting CAFF to PubMedQA requires (a) a UMLS-linked KG that covers the PubMedQA entity space, and (b) an LLM backbone to consume the filtered triples; neither is shipped here. BioASQ evaluation is a closer target if a compatible KG can be assembled.

2. **No end-to-end QA with an LLM backbone.** The paper uses GPT-3.5-turbo to produce final answers from the filtered triple set. This repository measures the **filtering layer only** (F1, MAP, NDCG@10, per-hop precision). End-to-end QA accuracy is not measured.

3. **No context-swap diagnostic in code.** The paper's Appendix C diagnostic (controlled context-swap, JSD in bits) is described in the paper but is not implemented in this codebase. The architecture exposes the necessary hooks (the CSV vector `z` is a tensor that can be edited at evaluation time); adding the diagnostic is straightforward future work.

4. **No path-survival-rate (PSR) metric.** The paper reports PSR on PubMedQA. This metric is not computed in the evaluator; it requires joining filter decisions back to the original BFS paths.

5. **Smaller KG than the paper.** This repository uses Orphanet + HPO + OMIM (|V|=38K, |R|=11). The paper uses Orphanet + DisGeNET + OMIM + UMLS (|V|=148K, |R|=42). DisGeNET and UMLS require institutional access. The MONDO ontology was tested as a partial substitute (Section 11 of `PAPER_DISCREPANCIES.md`) and reverted because it injected borderline-confident candidates that hurt F1.

6. **DC at lambda_D=0.40 is harmful here.** Whether a smaller lambda_D (e.g., 0.05 to 0.20) could rescue DC is open. The current headline simply disables DC; a lambda_D sweep is listed as the next ablation.

7. **HC3 is inert here.** The cross-query variant of the negative miner (Section 23) raised the loss gradient but did not change held-out F1. A larger, more diverse KG might re-activate the loss; on this KG it does not contribute.

8. **Relation-type aggregation only.** The CSV pools relation embeddings; head/tail entity types in the retained set are discarded. A typed CSV variant is plausible future work.

9. **Fixed hop depth `L=3`.** Deeper paths are mechanically supported but would require longer HC3/DC chains and larger triplet-mining buffers.

10. **Empty retained sets.** When `S_{ell-1}` is empty, CSV outputs `z=0` and DBM reduces to `Delta=0`, so CAFF degrades gracefully to DepthBilinear. A soft-retention variant is a natural extension.

---

## Citation

If you use this codebase or the methodology, please cite:

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

The empirical results in this README (No-DC headline, leave-one-out
ablation, paired bootstrap, HC3 inertness, DC harmfulness) are
documented in `PAPER_DISCREPANCIES.md` Sections 22-26.

---

## License

This project is released under the **MIT License**; see [`LICENSE`](LICENSE) for the full text.

> The merged KG **derived from** Orphanet, HPO, and OMIM is **not redistributed**; users must obtain the source data directly under each provider's terms.

---

## Acknowledgements

This research was conducted at the **School of Software Engineering, Dalian University of Technology (DUT)**, with support from the **CSC Type-B Scholarship**. We thank the maintainers of **Orphanet**, **HPO**, **OMIM**, and **BioLinkBERT** for making their resources publicly available.

---

## Contact

| Role | Name | Email |
|------|------|-------|
| Corresponding author | **Marwan Dhifallah** (M.Sc. student, DUT) | <marwan@mail.dlut.edu.cn> |
| Supervisor | **Prof. Yu Liu** (Associate Professor, DUT) | <yuliu@dlut.edu.cn> |

For bugs and feature requests, please open an [issue](../../issues). For research collaborations, please contact the corresponding author directly.
