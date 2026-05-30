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

---

## Table of Contents

1. [TL;DR](#tldr)
2. [The Context Blindness Error](#the-context-blindness-error)
3. [Approach](#approach)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Data](#data)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Results](#results)
10. [Ablation Study](#ablation-study)
11. [Configurations](#configurations)
12. [Hyperparameters](#hyperparameters)
13. [Reproducibility](#reproducibility)
14. [Hardware](#hardware)
15. [Scope and Future Work](#scope-and-future-work)
16. [Citation](#citation)
17. [License](#license)
18. [Acknowledgements](#acknowledgements)
19. [Contact](#contact)

---

## TL;DR

CAFF is a triple filter for multi-hop biomedical knowledge graph
retrieval-augmented generation. It addresses the **Context Blindness
Error (CBE)**: filters that score each candidate triple from
`(Query, relation, BFS_depth)` alone cannot distinguish whether the
same triple is relevant or irrelevant under different upstream
retained sets. CAFF fixes this with two coupled components:

- **CSV** -- a parameter-free, permutation-invariant summary of the previous hop's retained set.
- **DBM** -- a low-rank, sigmoid-gated dynamic perturbation of the bilinear scoring matrix, generated from the CSV.

**Headline results** (Orphanet biomedical KG, 3,000 held-out test queries, 3 random seeds, paired bootstrap):

| Metric                        | Mean +/- std       |
|-------------------------------|--------------------|
| Test F1 (per-hop thresholds)  | **0.5764 +/- 0.0022** |
| Test F1 (autoregressive)      | 0.5477 +/- 0.0006  |
| Test MAP                      | 0.6741 +/- 0.0003  |
| Test NDCG@10                  | 0.7090 +/- 0.0003  |

A complete leave-one-out ablation establishes CSV and DBM as the
essential components; two additional losses that were considered
during development (a depth-contrastive loss and an HC3-style
contrastive loss) were measured and removed because they did not
improve held-out F1 on this knowledge graph. See
[Ablation Study](#ablation-study).

---

## The Context Blindness Error

Consider the clinical query:

> *"What drug targets the pathway of the causal gene of Fanconi anemia complementation group D1?"*

The same hop-2 triple `<BRCA2, participates_in, HR-repair>` is:

- **Relevant** when the hop-1 retained set is `{<Fanconi anemia D1, caused_by, BRCA2>}`.
- **Irrelevant** when the hop-1 retained set is `{<Fanconi anemia D1, caused_by, BRIP1>}`.

A filter that sees only `(Query, relation, hop)` cannot tell these two
situations apart and therefore must score the triple identically in
both. We call this the **Context Blindness Error** (CBE). By the Data
Processing Inequality, any filter that ignores the previous hop's
retained set has expected loss bounded below by
`I(Y; S_{ell-1} | Q, r, ell)`, which is strictly positive whenever the
retained set carries information about the gold label.

The architectural fix is to feed a summary of `S_{ell-1}` into the
scoring function for hop `ell`. CAFF does this with CSV (the summary)
and DBM (a gating mechanism on the bilinear scorer).

---

## Approach

CAFF operates in four stages during multi-hop evidence retrieval.

### Stage 1 -- BFS candidate stratification

A BFS from the query's seed entities collects all triples reachable
within `L=3` hops, stratified by hop depth. A per-relation frequency
cap (`K_r=20`) prevents any one relation from dominating a candidate
set on highly connected entities.

### Stage 2 -- Contextual Summary Vector (CSV)

For each hop `ell > 1`, the retained set from the previous hop is
summarized by a parameter-free, permutation-invariant pool over the
relation embeddings of its triples:

```
z_{ell-1} = pool({ E[r] : (h, r, t) in S_{ell-1} })
```

The default pool is `mean`. At `ell=1` the retained set is empty by
convention, so `z_0 = 0` and the scorer reduces to its base form.

### Stage 3 -- Dynamic Bilinear Modulation (DBM)

The base hop-conditioned bilinear scorer

```
s_base(Q, r, ell) = Q^T W_ell E[r]
```

is augmented with a low-rank, context-dependent perturbation generated
from `z_{ell-1}`:

```
Delta_ell(z) = sigmoid(U z) * (A z) (B z)^T,   with rank rho << d
s_CAFF(Q, r, ell, z) = Q^T (W_ell + Delta_ell(z)) E[r]
```

This is the only context-aware path in the architecture; `Delta_ell`
adds about 0.8 M parameters at `rho=16, d=1024`. The sigmoid gate lets
DBM smoothly fall back to the base scorer when the context vector is
uninformative.

### Stage 4 -- Training objective

The training loss is BCE on the per-triple retain/drop label, optionally
augmented with two auxiliary losses (a depth-contrastive hinge and an
HC3 contrastive loss). Both auxiliaries are exposed as ablation flags.
On the held-out Orphanet QA test set, neither auxiliary improves F1 at
the configurations we measured; the default training therefore uses
BCE alone. See [Ablation Study](#ablation-study) for the evidence.

---

## Repository Structure

```
CAFF/
|-- caff/                            # Core package (importable)
|   |-- __init__.py                  # Public API surface
|   |-- config.py                    # CAFFConfig + AblationFlags dataclasses
|   |-- csv.py                       # Contextual Summary Vector
|   |-- data.py                      # KG loader, BFS extractor, datasets
|   |-- dbm.py                       # Dynamic Bilinear Modulation
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
|   |-- convert_orphanet_xml_to_tsv.py
|   |-- convert_hpo_to_tsv.py
|   |-- build_kg.py
|   |-- merge_hpo_into_kg.py
|   |-- build_orphanet_qa.py
|   |-- annotate_triples.py
|   |-- extract_bfs.py
|   |-- threshold_sweep.py
|   `-- per_hop_threshold_sweep.py
|
|-- configs/                         # YAML training configs (see Configurations below)
|   |-- no_dc.yaml                   # Default training configuration
|   |-- caff_orphanet.yaml           # Alternative with depth-contrastive loss
|   |-- caff_no_hc3.yaml             # Ablation (HC3 loss off)
|   |-- no_csv.yaml                  # Ablation (CSV off)
|   |-- no_dbm.yaml                  # Ablation (DBM off)
|   |-- no_freqcap.yaml              # Ablation (frequency cap off)
|   |-- depthbilinear.yaml           # Baseline (all CAFF components off)
|   `-- caff_smoke.yaml              # CI smoke test (tiny synthetic KG)
|
|-- tests/                           # Unit tests (run by CI)
|-- .github/workflows/tests.yml      # CI: lint + pytest on every push
|
|-- data/                            # gitignored (raw + processed)
|-- runs/                            # gitignored (checkpoints, logs)
|-- cache/                           # gitignored (BFS + relation cache)
|-- results/                         # benchmark JSON outputs
|
|-- train.py                         # Training entry point
|-- evaluate.py                      # Standalone evaluation script
|
|-- README.md                        # This file
|-- PAPER_DISCREPANCIES.md           # Detailed experimental log (26 sections)
|-- LICENSE                          # MIT
|-- requirements.txt
`-- .gitignore
```

`PAPER_DISCREPANCIES.md` is the running experimental log; every
empirical claim in this README is backed by a numbered section there.

---

## Installation

### Prerequisites

- Python >= 3.10
- CUDA 11.8 (an 8 GB consumer GPU is sufficient)
- Git

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/caff.git
cd caff

# 2. Create a clean virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install PyTorch
pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Pre-download the BioLinkBERT-Large encoder (optional, also auto-downloads)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-large')"
```

### Core dependencies

```
torch>=2.0
transformers>=4.30
networkx>=3.0
numpy, scipy, scikit-learn, pandas
tqdm, pyyaml
```

---

## Data

CAFF operates on a merged biomedical knowledge graph built from three
public sources: **Orphanet** (rare-disease ontology, gene-disease links),
**HPO** (phenotype ontology), and **OMIM** annotations (Mendelian
inheritance, gene-phenotype). All three are publicly available and
non-credentialed.

### Build the KG

```bash
# 1. Convert raw ontologies to TSV
python scripts/convert_orphanet_xml_to_tsv.py --in data/raw/orphanet/ --out data/processed/orphanet.tsv
python scripts/convert_hpo_to_tsv.py          --in data/raw/hpo/hp.obo  --out data/processed/hpo.tsv

# 2. Build base KG and merge in HPO/OMIM
python scripts/build_kg.py          --orphanet data/processed/orphanet.tsv --out data/processed/merged_kg.tsv
python scripts/merge_hpo_into_kg.py --in data/processed/merged_kg.tsv --hpo data/processed/hpo.tsv --out data/processed/merged_kg_v2.tsv

# 3. Sample QA records from the KG
python scripts/build_orphanet_qa.py --kg data/processed/merged_kg_v2.tsv --n 20000 --out data/processed/

# 4. Pre-compute BFS candidates and gold annotations
python scripts/extract_bfs.py       --kg data/processed/merged_kg_v2.tsv --L 3 --K_r 20
python scripts/annotate_triples.py  --kg data/processed/merged_kg_v2.tsv --qa data/processed/
```

### KG statistics

| Property                        | Value     |
|---------------------------------|-----------|
| Entities `|V|`                  | 38,456    |
| Triples `|E|`                   | 291,335   |
| Relation types `|R|`            | 11 (after `min_relation_freq=50`) |
| Maximum BFS hop depth `L`       | 3         |
| QA records (train/dev/test)     | 14,000 / 3,000 / 3,000 |
| Triple instances (train)        | 473,471 (6.24% positive) |
| Triple instances (test)         | 102,317 (6.27% positive) |

Triples on any shortest path from a seed entity to the gold answer
entity receive label `y=1`; all others `y=0`. The test set is
held out completely from training and threshold tuning.

---

## Training

The default training configuration is `configs/no_dc.yaml`. To
reproduce the headline numbers on three seeds:

```bash
for s in 42 1337 2024; do
    python train.py --config configs/no_dc.yaml --seed $s
done
```

Each seed takes about 40 minutes on an 8 GB consumer GPU (RTX 4060).
`train.py` auto-detects CUDA and applies hardware-appropriate overrides
(`micro_batch_size=4, grad_accum_steps=64, mixed_precision=fp16`,
effective batch 256). Training is deterministic: the same seed produces
bit-identical results across runs on the same hardware.

### Reproduce the full ablation suite

```bash
# Train each variant on 3 seeds
for cfg in no_dc caff_orphanet no_csv no_dbm no_freqcap caff_no_hc3 depthbilinear; do
    for s in 42 1337 2024; do
        python train.py --config configs/$cfg.yaml --seed $s
    done
done

# Per-hop threshold tuning on each variant
for cfg in no_dc caff_orphanet no_csv no_dbm no_freqcap depthbilinear; do
    for s in 42 1337 2024; do
        python scripts/per_hop_threshold_sweep.py \
            --config configs/$cfg.yaml \
            --checkpoint runs/$cfg/seed_$s/best.pt
    done
done
```

### Key hyperparameters (`no_dc.yaml`)

| Hyperparameter        | Value  |
|-----------------------|-------:|
| Optimizer             | AdamW (weight decay 1e-2) |
| Base learning rate    | 3e-4 (cosine to 1e-5, 1-epoch warmup) |
| Effective batch size  | 256   |
| Epochs                | 10 (early stopping on dev F1, patience 5) |
| Gradient clip         | 1.0   |
| Encoder               | BioLinkBERT-large (340 M, frozen, d=1024) |
| Random seeds          | {42, 1337, 2024} |

---

## Evaluation

### Per-hop threshold sweep (the headline metric)

```bash
python scripts/per_hop_threshold_sweep.py \
    --config configs/no_dc.yaml \
    --checkpoint runs/no_dc/seed_42/best.pt
```

Tunes per-hop retention thresholds on the dev set, then reports
precision, recall, and F1 on the held-out test set under three regimes:
global theta=0.50, global theta=0.80, and per-hop tuned thresholds.

### Paired bootstrap significance test

```bash
python evaluate.py \
    --checkpoint runs/no_dc/seed_42/best.pt \
    --report-bootstrap-vs runs/caff_orphanet/seed_42/best.pt \
    --mode autoregressive \
    --output-json results/bench_no_dc_vs_full_seed_42.json
```

Reports test metrics (F1, MAP, NDCG@10, per-hop precision) at
theta=0.80 in autoregressive inference mode (no leakage of gold
relations from prior hops), plus a paired bootstrap on per-query AP
versus the baseline checkpoint (10,000 resamples).

---

## Results

All numbers below are measured on the held-out Orphanet QA test set
(3,000 queries, 102,317 candidate triples), 3 seeds, deterministic.
The full per-seed outputs are in `results/`.

### Default configuration (no_dc.yaml)

| Metric                       | Mean +/- std       | Inference mode |
|------------------------------|--------------------|----------------|
| Test F1 (per-hop)            | **0.5764 +/- 0.0022** | teacher-forced |
| Test F1 (autoregressive)     | 0.5477 +/- 0.0006   | autoregressive |
| Test MAP                     | 0.6741 +/- 0.0003   | autoregressive |
| Test NDCG@10                 | 0.7090 +/- 0.0003   | autoregressive |
| Hop-1 precision              | 0.8234 +/- 0.0047   | autoregressive |
| Hop-2 precision              | 0.4378 +/- 0.0006   | autoregressive |
| Hop-3 precision              | 0.2426 +/- 0.0026   | autoregressive |

Per-hop is the headline metric: thresholds are tuned per hop on the
dev set, then applied unchanged on test. Autoregressive is reported
separately because it does not leak gold relations from prior hops
during inference; the F1 gap of about 0.029 between the two modes is
the cost of realistic deployment.

### Statistical significance versus alternative configurations

Paired bootstrap on per-query AP, 10,000 resamples, computed per seed:

| Comparison                       | delta_AP        | 95% CI                | p-value |
|----------------------------------|----------------:|-----------------------|--------:|
| no_dc vs caff_orphanet, seed 42  | +0.0295         | [+0.0251, +0.0340]    | 0.0000  |
| no_dc vs caff_orphanet, seed 1337| +0.0227         | [+0.0188, +0.0267]    | 0.0000  |
| no_dc vs caff_orphanet, seed 2024| +0.0227         | [+0.0190, +0.0267]    | 0.0000  |
| **mean**                         | **+0.0250**     | (each CI excludes 0)  | < 0.01  |

The default configuration outperforms the alternative
(`caff_orphanet.yaml`, which adds the depth-contrastive auxiliary loss)
significantly on every seed.

---

### Generalization to novel seed entities

The Orphanet test set was not constructed to share seeds with training.
Of the 2,876 distinct seed entities in the test set, 1,840 (64.2
percent of distinct seeds, and 64.2 percent of test queries) do not
appear anywhere in training. Stratifying F1 by this seen / unseen
distinction (3 seeds, theta=0.80, autoregressive):

| group       | n queries | F1 (mean +/- std)     |
|-------------|----------:|----------------------:|
| seen seed   |     1,074 | 0.5642 +/- 0.0006     |
| unseen seed |     1,926 | **0.5384 +/- 0.0013** |
| **gap**     |        -- | **+0.0259 +/- 0.0018** (4.6% relative) |

Recall is nearly identical between the two groups; only precision drops
on novel seeds. The 4.6 percent gap is concentrated at hop 2 (12.0
percent relative there); hops 1 and 3 show no measurable dependence on
whether the seed was seen during training. Full analysis in
`PAPER_DISCREPANCIES.md` Section 30. CAFF generalizes to novel seed
entities within the same KG schema.

---

### Per-relation breakdown

The Orphanet test set has 11 relation types, but the positives are
concentrated in two: `is_a` (73.8 percent) and `has_phenotype` (24.1
percent). Splitting the headline F1 by relation reveals that CAFF
performs very differently on the two:

| relation                  | n_total | n_pos | precision | recall | F1 (mean +/- std)     |
|---------------------------|--------:|------:|----------:|-------:|----------------------:|
| `is_a`                    | 75,469  | 5,214 | 0.534     | 0.687  | **0.6014 +/- 0.0013** |
| `has_phenotype`           | 24,692  | 1,146 | 0.387     | 0.033  | 0.0604 +/- 0.0097     |
| 9 other (rare) relations  |  ~2,156 |    56 | varies    | varies | ~0 (data sparse)      |
| **overall**               | 102,317 | 6,416 | 0.528     | 0.568  | **0.5477 +/- 0.0006** |

(Autoregressive mode, theta=0.80, 3 seeds.) The 0.5477 overall F1 is
essentially the `is_a` F1 averaged with a near-zero `has_phenotype`
contribution. The model handles taxonomy edges very well; it learns
`has_phenotype` (recall recovers from 0.033 at theta=0.80 to 0.685 at
theta=0.50 on the same checkpoint) but its confidence rankings on
phenotype attachments are weaker. Per-relation thresholds do not raise
the aggregate F1: `has_phenotype` caps at F1 = 0.20 even at its peak
threshold (theta=0.65), and `is_a` already dominates the average. Full
analysis in `PAPER_DISCREPANCIES.md` Section 27.

---

## Ablation Study

Leave-one-out over every component, plus a depth-stratified baseline,
on the held-out test set. Three seeds per variant; per-hop test F1
with thresholds tuned on dev.

| Variant                           | Test F1 (per-hop) | delta vs Default |
|-----------------------------------|------------------:|-----------------:|
| **Default (no_dc.yaml)**          | **0.5764 +/- 0.0022** | --           |
| no_dc + HC3 (`caff_no_hc3` off)   | 0.5524 +/- 0.0016 | -0.0240          |
| no_dc + DC (`caff_orphanet.yaml`) | 0.5524 +/- 0.0016 | -0.0240          |
| no_dc - FreqCap                   | 0.5524 (identical to caff_orphanet, frequency cap inert on this KG) | -0.0240 |
| no_dc - DBM                       | 0.5063 +/- 0.0046 | -0.0701          |
| no_dc - CSV                       | 0.5054 +/- 0.0027 | -0.0710          |
| DepthBilinear (no CSV, no DBM)    | 0.4966 +/- 0.0121 | -0.0798          |

Take-aways:

1. **CSV and DBM are the essential architectural components.** Removing either drops test F1 by about 0.07 points; they form a coupled pair (CSV produces `z`, DBM consumes it), so removing one effectively breaks the context-aware path.
2. **The depth-contrastive auxiliary loss hurts at lambda_D=0.40.** Adding it back (i.e., switching from `no_dc` to `caff_orphanet`) costs 0.024 F1 (paired bootstrap p < 0.01 across three seeds). A smaller positive lambda_D is left to future work; the default disables DC.
3. **HC3 is inert.** The HC3 loss as implemented produces zero gradient at the configurations tested (positives and negatives collide at the teacher-forced training step); turning it on changes neither the gradients nor held-out F1. An attempted cross-query variant raised the loss gradient norm but did not change test F1. Detailed diagnostics are in `PAPER_DISCREPANCIES.md` Sections 22-23.
4. **The per-relation frequency cap is inert here.** The KG has only 11 relations after `min_relation_freq=50` at load time, so the cap has nothing to act on.

The full evidence trail, including code-level verification that
`no_dc.yaml` differs from `caff_orphanet.yaml` only in the DC loss
weight, is in `PAPER_DISCREPANCIES.md` Sections 22-26.

---

## Configurations

| Config file              | Purpose                              | Trained?  | Test F1 (per-hop)      |
|--------------------------|--------------------------------------|:---------:|-----------------------:|
| `no_dc.yaml`             | **Default training configuration**   | Yes       | **0.5764 +/- 0.0022**  |
| `caff_orphanet.yaml`     | Alternative with DC loss on          | Yes       | 0.5524 +/- 0.0016      |
| `caff_no_hc3.yaml`       | Ablation (HC3 off, DC on)            | Yes       | 0.5524 (HC3 inert)     |
| `no_csv.yaml`            | Ablation (CSV off)                   | Yes       | 0.5054 +/- 0.0027      |
| `no_dbm.yaml`            | Ablation (DBM off)                   | Yes       | 0.5063 +/- 0.0046      |
| `no_freqcap.yaml`        | Ablation (freq cap off)              | Yes       | 0.5524 (cap inert)     |
| `depthbilinear.yaml`     | Baseline (all CAFF components off)   | Yes       | 0.4966 +/- 0.0121      |
| `caff_smoke.yaml`        | CI smoke test (tiny synthetic KG)    | Yes (CI)  | n/a                    |
| `caff_full.yaml`         | Legacy paper-spec config, kept for reference; not runnable as-is (`d=768` does not match BioLinkBERT-Large's output of 1024) | No | n/a |

All trained variants have checkpoints under `runs/<config_name>/seed_<seed>/`.

---

## Hyperparameters

The default configuration (`no_dc.yaml`) uses:

| Symbol               | Meaning                                          | Value |
|----------------------|--------------------------------------------------|------:|
| `d`                  | Embedding dimension (BioLinkBERT-Large output)   | 1024  |
| `L`                  | Maximum BFS hop depth                            | 3     |
| `rho`                | DBM rank                                          | 16    |
| `theta`              | Retention threshold (global default)             | 0.80  |
| `K_r`                | Frequency cap per relation per head              | 20    |
| `lambda_C`           | HC3 loss weight                                   | 0.35 (inert) |
| `lambda_D`           | Depth-contrastive loss weight                    | **0.0** (default; 0.40 disabled) |
| `min_relation_freq`  | Drop singleton relations at KG load              | 50    |
| `gamma_C`            | HC3 margin                                       | 0.25  |
| `gamma_D`            | Depth-contrastive margin                         | 0.20  |

A theta sensitivity analysis and a lambda_D sweep are listed under
future work.

---

## Reproducibility

- All results are **mean across three seeds** `{42, 1337, 2024}`.
- Training is deterministic (`config.deterministic = true`).
- Standard deviations are reported in every results table.
- Per-seed benchmark JSON outputs are committed under `results/`.
- Checkpoints under `runs/<config_name>/seed_<seed>/best.pt`.
- A full mirror of trained checkpoints, runs, and cache is maintained on Hugging Face at <https://huggingface.co/MrDhifallah/CAFF>.

---

## Hardware

| Stage                  | Reference setup                | Time per seed |
|------------------------|--------------------------------|--------------:|
| KG build + BFS         | i9-13900H, 32 GB RAM           | ~5 min one-time |
| Training               | NVIDIA RTX 4060 Laptop, 8 GB   | ~40 min       |
| Per-hop threshold sweep| RTX 4060                       | ~10 min       |
| Paired bootstrap eval  | RTX 4060                       | ~3 min        |

Pretty much any modern 8 GB consumer GPU suffices. CPU-only training
is technically supported (via `train.py`'s automatic hardware
override), but is impractical because the BioLinkBERT-Large encoder
consumes about 9 GB of CPU RAM and runs roughly 8x slower than on the
GPU.

---

## Scope and Future Work

This release evaluates CAFF on a single, well-characterized biomedical
benchmark. The following are sensible next steps; none of them are
implemented in this release.

1. **Lambda_D sweep.** The default disables the depth-contrastive auxiliary because it hurts at lambda_D=0.40. Whether a smaller positive value (0.05 to 0.20) helps is open.
2. **K-fold cross-validation.** The current results use a fixed 14K/3K/3K split. A 5-fold cross-validation would tighten the variance estimates.
3. **Theta sensitivity analysis.** The headline uses theta=0.80 from a dev sweep; reporting F1 across theta in [0.5, 0.9] would document the operating-point behavior more thoroughly.
4. **Typed CSV for semantic relations.** The per-relation breakdown above (and Section 27 of `PAPER_DISCREPANCIES.md`) shows that CAFF reaches F1 = 0.60 on `is_a` but caps at F1 = 0.20 on `has_phenotype`, even when the threshold is tuned per relation. The mean-pool CSV compresses ontological chains cleanly but discards information that matters for many-to-many semantic relations. A typed CSV that keeps head and tail entity types from the retained set is the natural next step.
5. **External rare-disease benchmark.** Datasets such as RareBench would test CAFF's transfer behavior on data not sampled from the training KG.
6. **End-to-end question answering with an LLM backbone.** This release measures the filtering layer only (F1, MAP, NDCG, per-hop precision). Connecting CAFF's filtered output to an LLM and measuring downstream QA accuracy is a separate engineering task.
7. **Larger KGs and broader biomedical domains.** Datasets like DisGeNET or UMLS require institutional access and are not used here. Validating CAFF on a broader KG is future work.
8. **A typed CSV.** The CSV currently pools relation embeddings; head/tail entity types in the retained set are discarded. A typed variant could carry additional signal.

---

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

The empirical results in this README are documented in
`PAPER_DISCREPANCIES.md` (Sections 22-26 for the ablation, paired
bootstrap, and configuration analysis).

---

## License

This project is released under the **MIT License**; see [`LICENSE`](LICENSE) for the full text.

The merged KG derived from Orphanet, HPO, and OMIM is **not redistributed**; users must obtain the source data directly under each provider's terms.

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
