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
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.13-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-11.8-76B900?style=flat-square&logo=nvidia&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Research%20Code-orange?style=flat-square">
</p>

<p align="center">
  <b>Marwan Dhifallah</b><sup>*</sup> &nbsp;·&nbsp; <b>Yu Liu</b><br>
  <i>Dalian University of Technology, Dalian, China</i><br>
  <code>marwan@mail.dlut.edu.cn</code> &nbsp;·&nbsp; <code>yuliu@dlut.edu.cn</code>
</p>

<p align="center">
  <i>Official PyTorch implementation of the paper</i><br>
  <b>"CAFF: Context-Aware Feedback Filtering for Multi-Hop Biomedical Knowledge Graph Evidence Selection"</b><br>
  <i>(under review, IEEE Transactions on Knowledge and Data Engineering, 2026).</i>
</p>

---

## Table of Contents

1. [TL;DR](#tldr)
2. [The Context Blindness Error (CBE)](#the-context-blindness-error-cbe)
3. [Key Contributions](#key-contributions)
4. [Method Overview](#method-overview)
   - [Stage 1  BFS Candidate Stratification](#stage-1-bfs-candidate-stratification)
   - [Stage 2  Contextual Summary Vector (CSV)](#stage-2-contextual-summary-vector-csv)
   - [Stage 3  Dynamic Bilinear Modulation (DBM)](#stage-3-dynamic-bilinear-modulation-dbm)
   - [Stage 4  Hop-Conditioned Context Contrast (HC3) Loss](#stage-4-hop-conditioned-context-contrast-hc3-loss)
5. [Theoretical Guarantees](#theoretical-guarantees)
6. [Repository Structure](#repository-structure)
7. [Installation](#installation)
8. [Data Preparation](#data-preparation)
9. [Training](#training)
10. [Evaluation](#evaluation)
11. [Main Results](#main-results)
12. [Ablation Study](#ablation-study)
13. [Hyperparameters](#hyperparameters)
14. [Reproducibility](#reproducibility)
15. [Hardware Requirements](#hardware-requirements)
16. [Limitations](#limitations)
17. [Citation](#citation)
18. [License](#license)
19. [Acknowledgements](#acknowledgements)
20. [Contact](#contact)

---

## TL;DR

> Existing triple filters for multi-hop KG-RAG score each candidate from `(Query, relation, BFS_depth)` alone , they are **blind** to which triples were retained at the previous hop. We prove this blindness incurs an **irreducible** Bayes error floor `ε* > 0` (Theorem 1, via the Data Processing Inequality). **CAFF** closes this gap with a three-piece, filtering-layer-only feedback loop:
>
> - **CSV**  a parameter-free, permutation-invariant summary of the previously retained set.
> - **DBM**  a low-rank, sigmoid-gated perturbation of the bilinear scoring matrix, *generated dynamically* from the CSV.
> - **HC3**  a contrastive loss that provably maximizes a variational lower bound on the conditional mutual information `I(Y; S | z)`.
>
> CAFF lifts PubMedQA accuracy from **76.9 → 79.6** (+2.7 pts) and BioASQ 7b macro-F1 from **71.1 → 74.3** (+3.2 pts) over the strongest depth-stratified baseline, with gains concentrated at the deepest hops (**+6.9** pts at hop 2, **+9.3** pts at hop 3) exactly where CBE is most severe.

---

## The Context Blindness Error (CBE)

Consider the clinical query:

> *"What drug targets the pathway of the causal gene of Fanconi anemia complementation group D1?"*

The same hop-2 triple `⟨BRCA2, participates_in, HR-repair⟩` is:
- **Diagnostically essential** when hop 1 retained `⟨FANCD1, causal_mutation, BRCA2⟩`,
- **Irrelevant noise** when hop 1 retained only `⟨FANCD1, has_phenotype, bone-marrow-failure⟩`.

A depth-stratified filter sees only `(Q, participates_in, ℓ=2)` and assigns **the same score in both cases**. It cannot distinguish the two evidential trajectories  it commits the **Context Blindness Error**.

Formally, for a context-agnostic filter `f ∈ F_agn` and any threshold `τ`:

```
P( 𝟙[f(X) ≥ τ] ≠ Y )  ≥  R*  +  ε*,    where  ε* > 0
```

with the closed-form lower bound

```
ε*  ≥  ½ · 𝔼_X [ Var_{Z|X} ( P(Y=1 | X, Z) ) ].
```

This floor is **architectural**, not representational  no parameter scaling of `f(Q, r, ℓ)` can recover information about `S_{ℓ-1}` that was never given to it as input.

---

## Key Contributions

| # | Contribution | Paper Section |
|---|---|---|
| **C1** | Formal definition of the Context Blindness Error (CBE) and a proof that it induces an irreducible Bayes error floor `ε* > 0`, with a closed-form variance lower bound. | §4 |
| **C2** | **Contextual Summary Vector (CSV)**  a parameter-free, permutation-invariant encoder of the previously retained set, with a formal injectivity guarantee under linearly independent relation embeddings. | §5.2 |
| **C3** | **Dynamic Bilinear Modulation (DBM)**  a low-rank, sigmoid-gated, *dynamically generated* perturbation of the scoring matrix, with **zero per-candidate overhead** after one-time precomputation. | §5.3 |
| **C4** | **HC3 loss**  an InfoNCE-derived contrastive objective formally equivalent to maximizing a variational lower bound on the conditional mutual information `I(Y; S \| z)`. | §5.4 |
| **C5** | State-of-the-art results on PubMedQA and BioASQ 7b, with hop-stratified ablations and a diagnostic context-swap experiment showing **1.84 bits** of context separation vs. **0.00 bits** for every context-agnostic baseline. | §7–§8 |

---

## Method Overview

```
┌─────────┐   ┌────────────┐   ┌─────────────────────────────────────────┐   ┌─────────────┐
│ Query Q │──▶│  Entity    │──▶│  BFS subgraph extraction (depth L=3)    │──▶│ Candidate   │
└─────────┘   │  Linker    │   └─────────────────────────────────────────┘   │ sets {C_ℓ}  │
              └────────────┘                                                  └──────┬──────┘
                                                                                     │
                            ┌──────────────────  CAFF filtering layer  ──────────────┘
                            ▼
                    ┌───────────────┐    z_{ℓ-1}    ┌────────────────┐
                    │  CSV          │──────────────▶│  DBM gate      │
                    │  (Eq. 14)     │               │  Δ_ℓ (Eq. 17)  │
                    └───────────────┘               └────────┬───────┘
                                                             ▼
                                                   W^ctx_ℓ = W₀ + A_ℓB_ℓᵀ + Δ_ℓ
                                                             │
                                                             ▼
                                                   s_ℓ = σ(qᵀ W^ctx_ℓ e_r + ...)
                                                             │
                                                             ▼ (threshold θ)
                                                          S_ℓ ────┐
                                                                   │  feedback
                                                                   └──▶ z_ℓ = CSV(S_ℓ)
                                                                               │
                                                                               ▼
                                                                       (next hop ℓ+1)
```

### Stage 1  BFS Candidate Stratification

For each head entity at depth `ℓ-1`, retain at most `K_r = 20` triples per relation type, ranked by descending tail degree. This **frequency cap** prevents hub-entity relation embeddings from saturating the CSV.

### Stage 2  Contextual Summary Vector (CSV)

Parameter-free, permutation-invariant, formally faithful:

```
              1
z_{ℓ-1}  =  ──── · Σ_{(h,r,t) ∈ S_{ℓ-1}}  e_r        (mean of frozen relation embeddings)
            |S_{ℓ-1}|
```

with `z_{ℓ-1} = 0` when `S_{ℓ-1} = ∅` (so CAFF reduces *gracefully* to the depth-stratified baseline at hop 1).

**Lemma (CSV faithfulness).** If the relation-embedding matrix `E` has full row rank, the map `Z ↦ EᵀZ` is **injective** on the simplex  distinct retained-context distributions produce distinct CSVs.

### Stage 3  Dynamic Bilinear Modulation (DBM)

A low-rank, sigmoid-gated, **runtime-generated** increment to the scoring matrix:

```
g_ℓ  =  σ( P_ℓ z_{ℓ-1} + b_ℓ )   ∈ (0,1)^ρ           (context gate, ρ = 16)

Δ_ℓ  =  U_ℓ · diag(g_ℓ) · V_ℓᵀ                       (rank-ρ context perturbation)

W^ctx_ℓ  =  W₀          +   A_ℓ B_ℓᵀ        +   Δ_ℓ(g_ℓ)
            ─────           ─────────              ─────────
            shared          depth-specific         context-specific
            base            (PCE correction)       (CBE correction)

s_ℓ  =  σ( qᵀ W^ctx_ℓ e_r  +  vᵀ(q ⊙ e_r)  +  β_ℓ )
```

**Cost.** `W^ctx_ℓ` is precomputed **once per hop** (not per candidate). Per-hop overhead with `d=768`, `ρ=16`, `N_ℓ ≤ 500`:
- gate: `O(ρd)`,
- DBM assembly: `O(d²ρ)`,
- candidate scoring: `O(N_ℓ d²)`,
- total CAFF surcharge ≈ **9.4 × 10⁶ FLOPs / hop**, **zero per candidate**.

> **DBM vs. LoRA / Adapters.** LoRA learns a *fixed* low-rank increment during fine-tuning. DBM **generates** its rank-ρ increment *dynamically at inference time* from the CSV  context-specific modulation without a separate parameter set per context.

### Stage 4  Hop-Conditioned Context Contrast (HC3) Loss

For each anchor `(Q, r, ℓ)`, mine a positive context `z^(a)` (where the triple was labeled 1) and up to 8 negative contexts `z^(b)` (where it was labeled 0):

```
L_HC3  =  (1 / |T_HC3|) · Σ  max( 0,  s(Q, r, ℓ, z^(b))  −  s(Q, r, ℓ, z^(a))  +  γ_C )
```

with margin `γ_C = 0.25`. **Proposition.** Minimizing `L_HC3` maximizes a variational lower bound on `I(Y; S | z)`.

The full objective:

```
L  =  L_BCE  +  λ_D · L_DC  +  λ_C · L_HC3      (λ_D = 0.40,  λ_C = 0.35)
```

---

## Theoretical Guarantees

| # | Statement | Where |
|---|---|---|
| **Theorem 1** | Under positive CMI `I(Y; Z \| X) > 0`, every context-agnostic filter incurs an irreducible Bayes error floor `ε* > 0` with `ε* ≥ ½ · 𝔼_X[Var_{Z\|X}(π(X,Z))]`. | §4.2 |
| **Lemma 1** | CBE forces label indistinguishability: there exist `(X, z₁, y=1)` and `(X, z₂, y=0)` receiving identical agnostic scores. | §3.3 |
| **Lemma 2** | The CSV is injective on the simplex of retained-context distributions whenever the relation-embedding matrix has full row rank. | §5.2 |
| **Proposition 1** | CSV noise stability: `𝔼[‖z̃ − z‖²] = σ²d / \|S_{ℓ-1}\|`  context summaries are **most stable in the regime where they carry the most information**. | §5.2 |
| **Proposition 2** | DBM rank sufficiency: with `ρ ≥ rank(Δ*)`, DBM can exactly represent any context-induced perturbation (Eckart–Young). | §6 |
| **Proposition 3** | Minimizing `L_HC3` maximizes a variational lower bound on `I(Y; S \| z)`. | §5.4 |
| **Proposition 4** | `ε*` is monotone non-decreasing in `I(Y; Z \| X)` (via Pinsker). | §4.2 |
| **Theorem 2** | Strict TPR benefit: there exist contexts at which the optimal context-aware filter strictly outperforms the optimal context-agnostic one. | §6 |

---



---

## Installation

### Prerequisites

- **Python** ≥ 3.10
- **CUDA** 11.8 (a single NVIDIA A100-80GB or equivalent is recommended)
- **Git LFS** (for downloading model checkpoints, when released)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/caff.git
cd caff

# 2. Create a clean virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install PyTorch (matched to your CUDA toolkit)
pip install torch==1.13.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. (Optional) Download the BioLinkBERT-Large encoder
python -c "from transformers import AutoModel; AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-large')"
```

### Core Dependencies

```
torch==1.13.1
transformers>=4.30
scispacy>=0.5
networkx>=3.0
numpy, scipy, scikit-learn, pandas
tqdm, pyyaml, wandb (optional)
openai>=1.0          # only for end-to-end QA with GPT-3.5-turbo
```

---

## Data Preparation

CAFF operates on a **merged biomedical KG** built from three primary sources, joined on shared **UMLS Concept Unique Identifiers (CUIs)**.

| Source | Version | Used for | Access |
|---|---|---|---|
| **Orphanet** | 2024 release | Rare-disease ontology, gene–disease links | <https://www.orphadata.com> |
| **DisGeNET** | v7.0 (2020) | Gene–disease associations | <https://www.disgenet.org> |
| **OMIM** | 2023 update | Mendelian inheritance, gene–phenotype | <https://www.omim.org> |
| **PubMedQA** | — | QA benchmark (1,000 questions) | <https://pubmedqa.github.io/> |
| **BioASQ 7b** | 2019 release | QA benchmark (1,141 yes/no + factoid) | <http://bioasq.org> |

> **Licensing note.** Orphanet, DisGeNET, OMIM, and BioASQ have their own access terms. We **do not redistribute** the raw data; users must obtain it directly from each source. The build script reproduces the merged KG deterministically.

### Build the merged KG

```bash
python scripts/build_kg.py \
    --orphanet  data/raw/orphanet/ \
    --disgenet  data/raw/disgenet/all_gene_disease_associations.tsv \
    --omim      data/raw/omim/ \
    --umls      data/raw/umls/MRCONSO.RRF \
    --out       data/processed/kg.parquet \
    --min-relation-freq 50
```

After construction, the merged KG contains:

| Property | Value |
|---|---|
| Entities `\|V\|` | **148,423** |
| Triples `\|E\|` | **2,318,941** |
| Relation types `\|R\|` (after singleton removal) | **42** |
| Max BFS depth `L` | 3 |

### Annotate gold relevance

Triples on any shortest path from a seed entity to the gold answer entity receive `y = 1`; all others `y = 0`. This yields **≈ 3.7 M** labeled training instances across hop depths 1–3.

```bash
python scripts/annotate_triples.py \
    --kg          data/processed/kg.parquet \
    --pubmedqa    data/benchmarks/pubmedqa.json \
    --linker      scispacy + UMLS \
    --out         data/processed/gold_pubmedqa.jsonl
```

---

## Training

### Quick start — full CAFF on PubMedQA

```bash
python train.py --config configs/caff_full.yaml
```

### Reproduce the full benchmark suite

```bash
# 1. Strongest baseline (depth-stratified bilinear, no context)
python train.py --config configs/depthbilinear.yaml -seed 42

# 2. CAFF without HC3 loss (CSV + DBM only — ablates the CMI bound)
python train.py --config configs/caff_no_hc3.yaml -seed 42

# 3. Full CAFF
python train.py --config configs/caff_full.yaml -seed 42

# 4. Repeat across the three reported seeds
for s in 42 1337 2024; do
    python train.py --config configs/caff_full.yaml --seed $s
done
```

### Key training hyperparameters

| Hyperparameter | Default | Notes |
|---|---:|---|
| Optimizer | AdamW | weight decay `1e-2` |
| Base learning rate | `3e-4` | cosine decay to `1e-5`, 2-epoch linear warmup |
| Batch size | 256 | |
| Epochs | 30 | early stopping on dev, patience 5 |
| Gradient clip | `‖∇‖₂ ≤ 1.0` | |
| Random seeds | `{42, 1337, 2024}` | three runs reported |

---

## Evaluation

### Filtering-layer metrics

```bash
python evaluate.py \
    --checkpoint  runs/caff_full/seed_42/best.pt \
    --benchmark   pubmedqa \
    --metrics     precision recall f1 map ndcg@10 \
    --hop-stratified
```

### End-to-end question answering (GPT-3.5-turbo backbone)

```bash
export OPENAI_API_KEY=<your-key>

python evaluate.py \
    --checkpoint  runs/caff_full/seed_42/best.pt \
    --benchmark   pubmedqa \
    --llm-backbone gpt-3.5-turbo \
    --temperature 0 --top-p 1 \
    --metrics     accuracy
```

Triples are serialized as one per line:

```
[head] -- [relation] --> [tail]
```

### Diagnostic context-swap experiment (Appendix C)

This experiment **directly measures CBE**: it presents the same `(Q, r, ℓ=2)` under two semantically opposing upstream contexts and reports the Jensen–Shannon divergence between the resulting score distributions.

```bash
python context_swap_diagnostic.py \
    --checkpoint  runs/caff_full/seed_42/best.pt \
    --report-bits
```

> Every context-agnostic baseline yields **JSD = 0.00 bits** (the empirical signature of CBE). CAFF achieves **JSD = 1.84 bits**  concrete proof that the architectural fix is doing what the theory predicts.

---

## Main Results

### End-to-end QA (mean over 3 seeds)

| Method | PubMedQA Acc. | PubMedQA MAP | BioASQ 7b Macro-F1 | BioASQ MAP |
|---|---:|---:|---:|---:|
| BM25 | 68.2 | 0.571 | 61.3 | 0.519 |
| DPR-Bio | 72.4 | 0.618 | 65.7 | 0.562 |
| BioRAG | 74.1 | 0.641 | 68.9 | 0.591 |
| SubgraphRAG | 75.6 | 0.658 | 69.8 | 0.605 |
| DepthBilinear (B5, immediate predecessor) | 76.9 | 0.672 | 71.1 | 0.621 |
| CAFF - NoHC3 | _78.2_ | _0.689_ | _72.8_ | _0.638_ |
| **CAFF (Full)** | **79.6** | **0.703** | **74.3** | **0.652** |
| **Δ (Full vs. B5)** | **+2.7** | **+0.031** | **+3.2** | **+0.031** |

All gains over DepthBilinear are statistically significant at `p < 0.01` (paired bootstrap, 10,000 resamples).

### Hop-stratified triple precision (PubMedQA test)

| Method | Hop 1 | Hop 2 | Hop 3 | Avg |
|---|---:|---:|---:|---:|
| DepthBilinear | 61.7 | 54.3 | 48.8 | 54.9 |
| **CAFF (Full)** | **62.4** | **61.2** | **58.1** | **60.6** |
| **Δ_ℓ** | **+0.7** | **+6.9** | **+9.3** | **+5.7** |

> The gain at hop 1 is **near zero** by design , with no prior retained set, `z₀ = 0` and CAFF reduces exactly to DepthBilinear. Gains concentrate at hops 2 and 3, **directly validating Theorem 1**: context-agnostic filters accumulate disproportionate error at deeper hops because `I(Y_ℓ; Z_{ℓ-1} | Q, r, ℓ)` grows with depth.

### Path-survival rate (PSR)

A filter that maximizes edge-level F1 independently per hop can still drive multi-hop **path survival** to zero. CAFF's context conditioning correlates retention decisions *along the same path*:

| Method | F1 | PSR | End-to-end Acc. |
|---|---:|---:|---:|
| DepthBilinear | 66.6 | 63.3 | 76.9 |
| CAFF - NoHC3 | 68.8 | 71.2 | 78.2 |
| **CAFF (Full)** | **70.5** | **75.7** | **79.6** |

> The **+12.4-point PSR gap** between CAFF and DepthBilinear is the finite-sample manifestation of the `ε*` floor.

### Context-separation diagnostic (controlled context-swap)

| Method | s^(A) | s^(B) | **JSD (bits)** |
|---|---:|---:|---:|
| BM25 | 0.617 | 0.617 | 0.00 |
| DPR-Bio | 0.638 | 0.638 | 0.00 |
| BioRAG | 0.621 | 0.621 | 0.00 |
| SubgraphRAG | 0.634 | 0.634 | 0.00 |
| DepthBilinear | 0.620 | 0.620 | 0.00 |
| CAFF - NoHC3 | 0.741 | 0.301 | **1.41** |
| **CAFF (Full)** | **0.792** | **0.238** | **1.84** |

> Every context-agnostic baseline yields exactly **0.00 bits** of context separation — direct empirical confirmation of CBE.

---

## Ablation Study

| Variant | Acc. | F1 | ΔAcc. |
|---|---:|---:|---:|
| **CAFF (Full)** | **79.6** | **70.5** | — |
| − CSV (`z_{ℓ-1} ≡ 0`) | 76.9 | 66.6 | −2.7 |
| − DBM (`Δ_ℓ ≡ 0`) | 77.8 | 67.9 | −1.8 |
| − `L_HC3` | 78.2 | 68.8 | −1.4 |
| − `L_DC` | 78.9 | 69.7 | −0.7 |
| − Frequency cap | 78.4 | 69.1 | −1.2 |
| Mean-CSV → max-pool | 79.1 | 69.9 | −0.5 |
| sigmoid gate → ReLU | 79.3 | 70.1 | −0.3 |
| `ρ = 8` (half rank) | 79.0 | 69.7 | −0.6 |
| `ρ = 32` (double rank) | 79.5 | 70.4 | −0.1 |

**Take-aways.**
1. Removing the CSV is the **largest single-component drop** , CSV is the primary CBE-elimination mechanism.
2. HC3 contributes an independent **+1.4 pts** by maximizing the CMI bound.
3. `ρ = 16` is near-optimal; `ρ = 32` yields only a marginal `−0.1` improvement.

---

## Hyperparameters

| Symbol | Meaning | Default |
|---|---|---:|
| `d` | Embedding dimension (BioLinkBERT-Large) | 768 |
| `L` | Maximum BFS hop depth | 3 |
| `ρ` | DBM rank | 16 |
| `θ` | Retention threshold | 0.50 |
| `K_r` | Frequency cap per relation per head | 20 |
| `γ_C` | HC3 margin | 0.25 |
| `γ_D` | Depth-contrastive margin | 0.20 |
| `λ_C` | HC3 loss weight | 0.35 |
| `λ_D` | DC loss weight | 0.40 |

CAFF maintains accuracy within **±1.5 pts** of its best configuration over the joint robustness box

```
γ_C ∈ [0.15, 0.35],   γ_D ∈ [0.10, 0.30],   lr ∈ [2e-4, 5e-4],
```

a wider basin than reported for SubgraphRAG or BioRAG.

---

## Reproducibility

- All reported results are **mean across three seeds** `{42, 1337, 2024}`.
- Standard deviations exceeding `0.3` points are noted in the paper text.
- Statistical significance is assessed via **paired bootstrap resampling** (`B = 10,000`).
- The frozen relation encoder (`BioLinkBERT-Large`, 340 M params) is **never updated**, confining all learnable capacity to **< 12 M parameters** (≈ 3.5% of the backbone).
- Code, preprocessing scripts, trained checkpoints, and the merged-KG construction pipeline will be released **upon publication** under the MIT License.

---

## Hardware Requirements

| Resource | Specification |
|---|---|
| GPU | 1 × NVIDIA A100-80GB SXM4 (recommended) |
| GPU memory | ≥ 40 GB for default batch size (256) |
| System RAM | ≥ 64 GB (KG fits in memory) |
| Disk | ≈ 25 GB (raw + processed data + checkpoints) |
| Framework | PyTorch 1.13 · CUDA 11.8 |

| Stage | Wall-clock (single A100) |
|---|---:|
| KG construction (one-time) | ≈ 25 min |
| Triple annotation (one-time) | ≈ 40 min |
| Training (single seed) | ≈ 4.5 h |
| Training (3 seeds, full pipeline) | ≈ 13.5 h |

---

## Limitations

1. **Relation-type aggregation only.** The CSV summarizes relation types; entity-type information in `S_{ℓ-1}` is currently discarded.
2. **Fixed hop depth.** Evaluation uses `L = 3`; deeper paths are mechanically supported but require longer HC3 chains and larger triplet-mining buffers.
3. **LLM backbone.** End-to-end QA results use GPT-3.5-turbo. Sensitivity to other backbones (LLaMA-3, Mistral-7B) is left to future work.
4. **KG completeness.** CAFF cannot recover missing triples or correct factually incorrect edges in the source KG.
5. **Annotation method.** Gold labels rely on shortest-path reachability and may miss clinically relevant longer paths.
6. **Empty retained sets.** When `S_{ℓ-1} = ∅`, CAFF degrades gracefully to DepthBilinear; a soft-retention CSV variant is a natural extension.

---

## Citation

If you use CAFF, the merged KG construction, or the context-swap diagnostic in your work, please cite:

```bibtex
@article{dhifallah2025caff,
  title   = {{CAFF}: Context-Aware Feedback Filtering for Multi-Hop
             Biomedical Knowledge Graph Evidence Selection},
  author  = {Dhifallah, Marwan and Liu, Yu},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

This project is released under the **MIT License**  see [`LICENSE`](LICENSE) for the full text.

> The merged KG **derived from** Orphanet, DisGeNET, and OMIM is **not redistributed**; users must obtain the source data directly under each provider's terms.

---

## Acknowledgements

This research was conducted at the **School of Computer Science and Technology, Dalian University of Technology (DUT)**, with support from the **CSC Type-B Scholarship**. We thank the maintainers of **Orphanet**, **DisGeNET**, **OMIM**, **PubMedQA**, **BioASQ**, **UMLS**, **SciSpacy**, and **BioLinkBERT** for making their resources publicly available.

---

## Contact

| Role | Name | Email |
|---|---|---|
| Corresponding author | **Marwan Dhifallah** (M.Sc. student, DUT) | <marwan@mail.dlut.edu.cn> |
| Supervisor | **Prof. Yu Liu** (Associate Professor, DUT) | <yuliu@dlut.edu.cn> |

For bugs and feature requests, please open an [issue](../../issues). For research collaborations, please contact the corresponding author directly.

---

<p align="center">
  <i>If CAFF helps your research, a ⭐ on this repository is appreciated.</i>
</p>
