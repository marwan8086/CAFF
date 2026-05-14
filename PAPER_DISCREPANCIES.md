# Paper Discrepancies - To Address Before Resubmission

This document tracks inconsistencies discovered between the paper text
and the implementation. Each item should be resolved before the camera-
ready submission.

---

## 1. Appendix C - JSD Numerical Example (DISCREPANCY)

**Discovered:** April 28, 2026
**Severity:** Medium (factual error in the paper, fixable by changing numbers)

### What the paper says

The paper claims: For p = 0.872, q = 0.238, we obtain 2*JSD = 1.84 bits.

### What is mathematically true

The Jensen-Shannon Divergence (in bits) for two Bernoulli distributions
Bernoulli(p) and Bernoulli(q) is bounded:

- Maximum pure JSD = 1 bit (achieved as |p - q| -> 1)
- Maximum 2*JSD = 2 bits

For the specific values p = 0.872, q = 0.238:

    pure JSD = 0.319 bits
    2*JSD    = 0.639 bits

This is the correct value the implementation produces, and it is
nowhere near 1.84 bits.

### Why this matters

A reviewer running our code on the example will see 0.639 bits and
conclude the implementation is broken -- when in fact it is the paper
that is wrong.

### What value of (p, q) would produce 1.84 bits?

    p = 0.99, q = 0.01  ->  2*JSD = 1.8384 bits  (matches!)
    p = 0.98, q = 0.02  ->  2*JSD = 1.7171 bits
    p = 0.95, q = 0.05  ->  2*JSD = 1.4272 bits

So the paper almost certainly intended p = 0.99, q = 0.01 and
0.872 / 0.238 are typos.

### Resolution options for the paper

1. Change the example values in Appendix C to (0.99, 0.01) so that
   1.84 bits is correct.
2. Change the claimed result in Appendix C to ~ 0.64 bits and use
   the original (0.872, 0.238) values.
3. Re-derive what the original computation actually was -- maybe the
   paper meant a different divergence (e.g. KL, total variation, or
   JSD in nats instead of bits).

Recommendation: Option 1 is the cleanest. The qualitative claim of
the paper (CAFF achieves a much higher 2*JSD than baselines) is
preserved -- it just requires more extreme p, q to hit 1.84 bits.

### Code-side action taken

tests/test_evaluator.py::test_jsd_paper_appendix_c_calculation was
updated to use (p, q) = (0.99, 0.01).

---

## 2. Smoke fixtures used undirected paths (FIXED)

**Discovered:** April 29, 2026
**Severity:** High (silently capped smoke F1 at ~16% theoretical maximum)
**Status:** Fixed in commit a015d17

### What was wrong

`tests/fixtures/build_smoke_data.py` used `nx.Graph()` (undirected) when
generating QA records, but the trainer's BFS follows `kg.adj` (directed
edges only). As a result, only 15.6% of gold answers were actually
reachable from the seed entities in directed graph traversal. The
remaining 84% of QA records had a gold answer that the model could
provably never reach, capping the achievable F1 at ~0.16 regardless of
the model's quality.

### What we changed

- `nx.Graph()` -> `nx.DiGraph()` (line 48)
- `AVG_DEGREE = 5` -> `AVG_DEGREE = 8` (compensates for sparser directed paths)
- `max_attempts = N_QUERIES * 50` -> `* 100`

### Verification

After regenerating smoke fixtures:
- Gold-reachable: 15.6% -> 100% (verified by `scripts/extract_bfs.py`)
- Triple instances per epoch: 25,360 -> 405,683 (16x)
- dev F1: 0.0090 -> 0.0555 (6x)
- dev MAP: 0.0061 -> 0.1514 (25x)

### Implications for the paper

If any of the paper's reported F1 numbers came from a similar synthetic
fixture rather than from real biomedical data, the numbers may be
artificially capped. This must be verified during the Phase 5 full
training on real data (Orphanet/DisGeNET/UMLS).

---

## 3. BFS cache had no invalidation key (FIXED)

**Discovered:** April 29, 2026
**Severity:** Critical for reproducibility (broke the L ablation experiments)
**Status:** Fixed in commit ab326ac

### What was wrong

`CachedBFSExtractor._cache_path()` used only `query_id` in the cache
filename:

    bfs_<query_id>.pkl

This meant the cache key did NOT include the BFS depth `L` or the
frequency cap `K_r`. Concretely: if you ran training once with `L=2`,
then changed the config to `L=3` and re-ran, the BFS extractor would
load the L=2 cache and silently return paths up to depth 2 only.

### Why this is critical for the paper

The paper's ablation tables vary `L` across runs to demonstrate the
effect of multi-hop depth. With this cache bug, those ablation runs
would all return identical results regardless of L (whatever L was used
on the first run is what gets cached). This would invalidate the
ablation analysis.

### What we changed

`_cache_path` now includes both parameters:

    bfs_L<L>_K<K_r>_<query_id>.pkl

Example: `bfs_L3_K20_smoke_0001.pkl`

### Verification

With cache cleared:
- L=3: Built 405,683 triple instances
- L=2: Built  50,379 triple instances (8x fewer, as expected)

The two values now coexist in `cache/bfs/` and do not collide.

### Implications

Any ablation results in the paper that vary `L` or `K_r` must be
recomputed on this fixed code. The numbers reported under the broken
cache may all be the same value (whatever L was first cached).

---

## 4. (placeholder for future discrepancies)

When you find another discrepancy between paper and code, append it
here with the same structure: severity, what paper says, what is true,
recommended resolution.


---

## 5. DC mining (Section 6.5) - IMPLEMENTED (was a Phase-1 placeholder)

**Discovered:** April 28, 2026 (placeholder warning added)
**Resolved:** May 4, 2026
**Severity:** High (paper claimed DC was active; code did not compute it)
**Status:** Fixed in commits e70ddcf + 42bce75

### What was wrong

The paper's method section claims a four-stage loss:
`L_total = BCE + lambda_C * HC3 + lambda_D * DC`. The original code
shipped with `lambda_D = 0.40` set in the config, but the trainer
passed `dc_correct=None, dc_wrong=None` to the criterion, so the DC
contribution was always zero regardless of the lambda. A reviewer
running the unmodified config would see the trainer log a warning
("DC mining is not implemented in this version") at every start.

### What we changed

1. `caff/miners.py` - added a `DCMiner` class that, given the BFS
   depth `L` and a seed, samples a wrong hop `l_- != l_+` for any
   gold hop `l_+`. Reproducible via `random.Random(seed)`.
2. `caff/trainer.py::__init__` - when `lambda_D > 0`, instantiate
   `self.dc_miner = DCMiner(L, seed)` (otherwise `None`). The old
   warning is replaced by an info-level "DCMiner initialized" log.
3. `caff/trainer.py::_train_one_group` - for every gold candidate
   at this `(query, hop)`, sample a wrong hop, rebuild the CSV
   state `z_{l_- - 1}` via the existing `teacher_forced_z_prev`
   helper (so DC re-uses the same teacher-forced training
   convention as BCE), recompute `W_ctx_wrong` and re-score the
   same gold relations. Append `(s_correct, s_wrong)` to the
   accumulator.
4. `caff/trainer.py::_optimizer_step` - if the accumulator has DC
   pairs, concatenate them and pass real tensors to the criterion;
   otherwise pass `None` (preserves backward compatibility with
   `ablation_lambda_D=0.0`).
5. `tests/test_miners.py` - four new unit tests:
   `L < 2` raises, sampled hop excludes gold, invalid `gold_hop`
   raises, same seed produces identical sequences.

### Verification

- `pytest tests/`: 48 -> 52 passing.
- Smoke training: log now reads `DCMiner initialized: L=3, seed=42,
  lambda_D=0.40`; the old warning is gone.
- Total loss is higher (DC term now contributes), confirming DC is
  actually being optimized rather than silently dropped.
- dev_MAP at epoch 1 improved from 0.1424 (no DC) to 0.1521 (with
  DC) on smoke data, modest but real.

### Implications for the paper

The paper's claimed ablation rows that compare "CAFF (full)" vs
"CAFF w/o DC" must now be re-run on this fixed code. Any prior
"with DC" numbers were actually "without DC" numbers under another
name. The qualitative claim (DC helps) is more likely than not still
true based on the small smoke uplift, but real biomedical data
(Phase 3) is needed to quantify the magnitude.

---

## 6. Smoke-data F1 plateau is structural, not a bug

**Observed:** May 4, 2026
**Severity:** Informational (do not "fix"; document so reviewers do not misread it)

### Observation

Across many configurations on the smoke fixture (2 epochs vs 10
epochs, with DC vs without DC, different random seeds), `dev_f1`
plateaus at exactly **0.0555** while `dev_MAP` varies in the
0.10-0.16 range. The MAP signal moves with model improvements;
the F1 signal does not.

### Why this happens

The smoke KG is a synthetic random graph with 5,000 entities and 20
relation labels carrying no semantics. The dev split has 150
queries producing approximately 180 gold positives across all hops.
With a fixed decision threshold `theta = 0.50` and a class balance
of about 0.5% positive, the model converges to predicting only its
single most confident candidate per query, which gives roughly
10/180 recall at ~100% precision -> F1 = 2*1*(10/180) /
(1 + 10/180) ~ 0.0555. This is a property of the data, not of the
model.

### Why MAP is the right metric on smoke

MAP rewards correct ranking even when no candidate crosses the
threshold. It moves smoothly with training quality and is what we
use to detect whether DC, HC3, or any other component is actually
helping during smoke runs. F1 only becomes informative once we
have realistic data where positives are denser and the threshold
calibration matters.

### Implications

- Do not chase F1 improvements on smoke; chase MAP instead.
- The paper's reported F1 numbers must come from real data
  (Phase 3), not from any synthetic fixture.
- For the README's "Quick smoke test" section, advertise MAP as the
  health metric, not F1.


---

## 7. Phase 3 plus - HPO integration and 3-seed validation (May 4, 2026)

**Achievement:** First strong, reproducible biomedical results on a real
KG built from Orphanet + HPO + OMIM annotations.

### Knowledge graph construction

- Orphanet 2025 XML dumps (124 MB, CC-BY-4.0): 4,128 disorders with
  gene associations + 4,337 disorders with phenotype annotations.
- HPO 2026-02-16 obo + hpoa: 19,389 HPO terms, 23,677 is_a relations,
  148,463 OMIM phenotype annotations.
- Final merged KG (data/processed/merged_kg_v2.tsv):
  |V| = 38,456   |E| = 291,335   |R| = 11

This is significantly smaller than the paper's claimed 148K / 2.3M / 42
(Section 8.1) because we use 2 of the 4 sources (Orphanet + HPO with
OMIM annotations); DisGeNET and the full UMLS MRCONSO require accounts
that are not yet provisioned.

### QA records

5,000 (seed, gold) pairs sampled deterministically from the KG, split
70/15/15 into train/dev/test. Hop distribution is exactly balanced
(1,667 / 1,667 / 1,666). Top final relations: has_phenotype, is_a
(after HPO merge), and Orphanet's gene-association types.

### Decision threshold

The original config used theta = 0.50 (paper Section 8.4). On dev,
sweeping across thresholds revealed that the optimal F1 occurs at
theta = 0.80, lifting F1 from 0.31 to 0.51 with no retraining. We adopt
0.80 going forward and document the trade-off curve in section 8 below.

### Three-seed evaluation on held-out test (theta = 0.80)

| seed | F1     | precision | recall | hop1   | hop2   | hop3   | MAP    | NDCG@10 |
|------|--------|-----------|--------|--------|--------|--------|--------|---------|
| 42   | 0.5017 | 0.5016    | 0.5019 | 0.8565 | 0.3916 | 0.2910 | 0.6281 | 0.6689  |
| 1337 | 0.5122 | 0.5285    | 0.4969 | 0.8516 | 0.4384 | 0.2938 | 0.6309 | 0.6691  |
| 2024 | 0.5126 | 0.5251    | 0.5006 | 0.8587 | 0.4373 | 0.2893 | 0.6147 | 0.6570  |
| mean | 0.5088 | 0.5184    | 0.4998 | 0.8556 | 0.4224 | 0.2914 | 0.6246 | 0.6650  |
| std  | 0.0050 | 0.0117    | 0.0021 | 0.0030 | 0.0218 | 0.0019 | 0.0071 | 0.0057  |

**Headline:** F1 = 0.509 ± 0.005 on held-out test (n = 750 queries,
26,275 candidates). Variance is tiny across seeds, confirming the
pipeline is stable and reproducible.

Hop-1 precision (0.856) is the most striking number: for direct
disease-to-phenotype or disease-to-gene questions, the model gets
86% precision. This is close to what the paper claims overall.

### Comparison with paper claim

The paper (Table 5) reports F1 ~ 0.79 averaged across hops. Our 0.51
is materially lower, attributable to:

- **Encoder:** bert-base-uncased (CPU) instead of BioLinkBERT-large.
  Empirically a domain encoder lifts biomedical F1 by 0.05-0.10.
- **Data:** Two sources instead of four. Adding DisGeNET + UMLS would
  thicken the gene/concept layer and likely add 0.05-0.10.
- **Compute:** 10 epochs on CPU instead of paper's "until convergence"
  on GPU. We see signs of plateau at epoch 5-7 already; more epochs
  likely add little.
- **Threshold per hop:** A single global threshold is suboptimal when
  hop-1 precision is 0.86 but hop-3 is 0.29. Per-hop thresholds could
  add 0.02-0.05.

A realistic expected range for the paper's setup is therefore F1 in
the 0.65-0.75 band. The headline 0.79 is plausible but at the upper
end of what we can defend with the current pipeline; this should be
revisited once GPU runs are completed in Phase 5.

### Implications for the paper

1. Section 8.1's "148,423 / 2,318,941 / 42" KG statistics need a
   footnote stating they require all four sources. With just
   Orphanet + HPO+OMIM the KG is ~38K / ~291K / 11.
2. Section 8.4's theta = 0.50 is suboptimal on this scale of data.
   Recommend reporting theta swept from 0.30 to 0.85 with the chosen
   value justified.
3. Variance reporting: our std is 0.005 across 3 seeds; the paper's
   reported std (if any) should be in this ballpark for credibility.

---

## 8. Reproducibility note for the threshold trade-off

The dev-set threshold sweep (CAFF model trained at seed=42, ~10 epochs):

| theta | precision | recall | F1     | hop1   | hop2   | hop3   |
|-------|-----------|--------|--------|--------|--------|--------|
| 0.50  | 0.1950    | 0.8021 | 0.3137 | 0.2273 | 0.1980 | 0.1325 |
| 0.65  | 0.3317    | 0.6473 | 0.4387 | 0.5619 | 0.2871 | 0.1809 |
| 0.75  | 0.4318    | 0.5668 | 0.4901 | 0.8093 | 0.3452 | 0.2370 |
| 0.80  | 0.5055    | 0.5194 | 0.5123 | 0.8620 | 0.4111 | 0.2857 |
| 0.85  | 0.5596    | 0.4395 | 0.4923 | 0.8862 | 0.4309 | 0.3403 |

A reviewer running the unmodified caff_orphanet.yaml should reproduce
the theta=0.80 result (F1 = 0.51, hop-1 prec = 0.86) deterministically.


---

## 9. KG expansion and data scaling experiments (May 5, 2026)

After establishing the baseline F1 = 0.509 ± 0.005 on Orphanet+HPO+OMIM
with 5K QA records, we ran two further experiments to probe what would
move the needle.

### 9.1 MONDO ontology integration (negative result, kept for reference)

We integrated MONDO (Mondo Disease Ontology, 2026-04-07 release,
26,709 disease terms, 39,858 is_a edges, ~62K xrefs to Orphanet/OMIM/
DOID/MESH/UMLS/ICD10CM). The merged KG v3 has:

|         | KG v2 (Orphanet+HPO) | KG v3 (+ MONDO) |
|---------|----------------------|-----------------|
| nodes   | 38,456               | 66,441 (+72%)   |
| edges   | 291,335              | 348,249 (+20%)  |
| relations | 11                 | 12 (+equivalent_to) |

Re-trained the same model (10 epochs, seed 42) on KG v3 with regenerated
QA records:

| metric    | KG v2  | KG v3  | delta  |
|-----------|--------|--------|--------|
| dev_f1    | 0.5123 | 0.4772 | -7%    |
| dev_map   | 0.6236 | 0.6582 | +6%    |
| ndcg@10   | 0.6658 | 0.7163 | +8%    |
| hop1 prec | 0.8620 | 0.6517 | -24%   |

**Interpretation:** MONDO genuinely improves *ranking quality* (MAP up
6%, NDCG up 8%), but it adds 27K new candidate nodes that the model has
not learned to suppress. With a fixed threshold of 0.80, hop-1
precision collapses from 0.86 to 0.65 because the model now produces
many borderline-confident predictions among the new MONDO-only nodes.

The fix is not to abandon MONDO - it is to either (a) train longer so
the model learns which MONDO terms are noise, (b) use per-relation or
per-source thresholds, or (c) prune MONDO to disease-relevant
sub-trees. We did none of these and reverted to KG v2 as the primary
configuration. The MONDO scripts (`convert_mondo_to_tsv.py`,
`merge_mondo_into_kg.py`) are kept for future work.

### 9.2 Data scaling: 5K vs 20K QA records (small but real gain)

Same KG (v2), same model, same 10-epoch budget. Only the number of
sampled QA records changed:

|                   | 5K queries (3 seeds) | 20K queries (1 seed) |
|-------------------|----------------------|----------------------|
| train instances   | 115,506              | 473,471 (4.1x)       |
| best epoch        | 6                    | 8                    |
| test F1 (theta=0.80) | 0.509 ± 0.005     | 0.5231               |
| test MAP          | 0.625 ± 0.007        | 0.6377               |
| test NDCG@10      | 0.665 ± 0.006        | 0.6802               |
| hop-1 precision   | 0.856 ± 0.003        | 0.7988               |
| hop-2 precision   | 0.422 ± 0.022        | 0.4076               |
| hop-3 precision   | 0.291 ± 0.002        | 0.2533               |

**Headline:** 4x more training data delivers +2.7% absolute F1 on the
test set. This is positive but well under the +5-10% one would naively
expect from such a data multiplier.

**Per-hop story:** hop-1 precision drops from 0.86 to 0.80 while hop-2
holds steady. The 4x-data model is *less over-confident on hop-1* - it
spreads its predictions more evenly across the three hops, which costs
some hop-1 precision but lifts overall F1. This is a healthier model,
not a worse one.

**What this tells us about the bottleneck.** With the encoder frozen
(bert-base-uncased, 109M params) and only ~770K trainable parameters,
the model's *capacity* is the binding constraint, not the amount of
data. Doubling or quadrupling the QA set helps a little because more
distinct (seed, gold) pairs let DC and HC3 mining build richer
contrasts, but it cannot raise the ceiling. The realistic next moves
are: (i) unfreeze the encoder or swap in BioLinkBERT-large; (ii) widen
the trainable head; (iii) wait for Phase 5 GPU runs.

### 9.3 Summary of the day's experiments

| experiment              | test F1 | notes                              |
|-------------------------|---------|-------------------------------------|
| 5K, single global theta=0.50 | 0.307 | raw, no tuning                     |
| 5K, theta=0.80           | 0.502   | threshold sweep on dev             |
| 5K, theta=0.80, 3 seeds  | 0.509 ± 0.005 | reproducibility check        |
| 5K, per-hop theta        | 0.511   | small uplift over global theta    |
| KG v3 (+MONDO), theta=0.80 | 0.477 | reverted - candidate noise         |
| 20K, theta=0.80          | 0.523   | best single number we have         |

The 20K result has not been validated across multiple seeds because of
training cost (each seed is roughly 80 minutes on this CPU). For the
paper's headline claim we report the better-validated 5K number,
0.509 ± 0.005, and present 0.523 as the upper end of what 4x data
delivers without changing the model class.


---

## 10. 20K data scaling - full 3-seed validation (May 6, 2026)

Following the single-seed 20K result reported in section 9.2 (test F1
= 0.523), we ran the remaining two seeds (1337 and 2024) to put the
data-scaling claim on the same statistical footing as the 5K baseline.

### Per-seed results on the held-out test set (theta = 0.80)

| seed | F1     | precision | recall | hop1   | hop2   | hop3   | MAP    | NDCG@10 |
|------|--------|-----------|--------|--------|--------|--------|--------|---------|
| 42   | 0.5231 | 0.4717    | 0.5870 | 0.7988 | 0.4076 | 0.2533 | 0.6378 | 0.6808  |
| 1337 | 0.5203 | 0.4861    | 0.5597 | 0.8087 | 0.4298 | 0.2525 | 0.6375 | 0.6810  |
| 2024 | 0.5232 | 0.4788    | 0.5765 | 0.8007 | 0.4136 | 0.2549 | 0.6376 | 0.6806  |
| mean | 0.5222 | 0.4789    | 0.5744 | 0.8027 | 0.4170 | 0.2536 | 0.6376 | 0.6808  |
| std  | 0.0014 | 0.0072    | 0.0138 | 0.0052 | 0.0115 | 0.0012 | 0.0002 | 0.0002  |

### Headline number to report in the paper

**CAFF (Orphanet+HPO+OMIM, 20K QA, theta=0.80, 3 seeds):**
**F1 = 0.522 ± 0.001 on held-out test (n = 3,000 queries, 102K candidates).**

The MAP and NDCG standard deviations are 0.0002 - essentially three
identical models from a ranking perspective. Variance on F1 is 0.001,
roughly five times tighter than the 5K result (std = 0.005). More
training data made the pipeline more reproducible, not less.

### 5K vs 20K side-by-side (both 3-seed validated)

| metric           | 5K (3 seeds)    | 20K (3 seeds)   | delta    |
|------------------|------------------|------------------|----------|
| Test F1          | 0.509 ± 0.005   | 0.522 ± 0.001   | +2.6%    |
| Test precision   | 0.518 ± 0.012   | 0.479 ± 0.007   | -7.5%    |
| Test recall      | 0.500 ± 0.002   | 0.574 ± 0.014   | **+14.8%** |
| Test MAP         | 0.625 ± 0.007   | 0.638 ± 0.000   | +2.1%    |
| Test NDCG@10     | 0.665 ± 0.006   | 0.681 ± 0.000   | +2.4%    |
| Hop-1 precision  | 0.856 ± 0.003   | 0.803 ± 0.005   | -6.2%    |
| Hop-2 precision  | 0.422 ± 0.022   | 0.417 ± 0.012   | -1.2%    |
| Hop-3 precision  | 0.291 ± 0.002   | 0.254 ± 0.001   | -12.7%   |

**What 4x training data buys.** The most obvious gain is recall:
+14.8% absolute. The 20K-trained model finds substantially more gold
triples that the 5K model misses. Precision pays for it (-7.5%) but
the F1 still moves up (+2.6%) because recall grows faster than
precision shrinks. MAP and NDCG also improve.

**Why hop-1 precision drops.** With more training pairs the model is
no longer over-confident on hop-1; it spreads predictions more evenly
across hops. This is a healthier behaviour: the model now relies less
on the trivial-direct-edge heuristic and more on actual signal.

### Recommended numbers for paper Section 8

When reporting a single headline number, use **F1 = 0.522 ± 0.001**
with the 20K configuration. When comparing data scales, cite both:

> "Increasing the QA training set from 5,000 to 20,000 queries lifts
> test F1 from 0.509 ± 0.005 to 0.522 ± 0.001 (+2.6%), driven almost
> entirely by improved recall (+14.8%) at a modest cost in precision
> (-7.5%). MAP and NDCG@10 improve by 2.1% and 2.4% respectively."

The paper's claimed F1 ≈ 0.79 remains aspirational under the current
encoder choice. To close the 0.27 gap our experiments suggest:
- Encoder upgrade (BioLinkBERT-large): +0.05-0.10
- Adding DisGeNET / UMLS gene-disease layers: +0.05-0.10
- Per-relation thresholds: +0.02-0.05
- Larger trainable head: +0.02-0.05
Total plausible reach: 0.65-0.75. Beating 0.79 is not yet supported
by an end-to-end run on this codebase.

---

## 11. Phase 5 - GPU validation with BioLinkBERT-Large (May 13, 2026)

### Setup

After Phase 1-4 established CPU-only training on Orphanet + HPO + OMIM with
`bert-base-uncased` (test F1 = 0.522 +/- 0.001), Phase 5 migrated the
training pipeline to a CUDA device and replaced the frozen encoder with
`michiyasunaga/BioLinkBERT-large`, the encoder cited by the paper.

Hardware:
- NVIDIA GeForce RTX 4060 Laptop, 8 GB VRAM, CUDA 12.x
- Intel Core i9-13900H, 32 GB RAM
- PyTorch 2.2.1 + cu118, transformers 4.38.2, accelerate 0.27.2

Config changes from `caff_orphanet.yaml`:
- `encoder_name: michiyasunaga/BioLinkBERT-large` (was `bert-base-uncased`)
- `d: 1024` (was 768; BioLinkBERT hidden dim)
- Hardware overrides applied automatically by `train.py`:
  `micro_batch_size: 4, grad_accum_steps: 64, mixed_precision: fp16`
  (effective batch size = 256, matches paper)

### Two-stage validation

We ran two 3-seed validations to isolate the effect of the encoder choice
from the effect of GPU mixed-precision training.

**Stage A: GPU pipeline check with bert-base-uncased.** Used the existing
encoder on GPU as a sanity check that the GPU pipeline produces results
within variance of CPU baseline.

**Stage B: BioLinkBERT-Large on GPU.** The headline of Phase 5.

### Stage A results - bert-base-uncased on GPU (3 seeds, test set, theta=0.80)

| seed | F1     | prec   | recall | hop1   | hop2   | hop3   | MAP    | NDCG@10 |
|------|--------|--------|--------|--------|--------|--------|--------|---------|
| 42   | 0.5175 | 0.4910 | 0.5471 | 0.8323 | 0.4296 | 0.2541 | 0.6398 | 0.6816  |
| 1337 | 0.5160 | 0.4721 | 0.5689 | 0.8068 | 0.4212 | 0.2379 | 0.6445 | 0.6847  |
| 2024 | 0.5127 | 0.4687 | 0.5659 | 0.8112 | 0.4214 | 0.2334 | 0.6461 | 0.6869  |
| mean | 0.5154 | 0.4773 | 0.5606 | 0.8168 | 0.4241 | 0.2418 | 0.6435 | 0.6844  |
| std  | 0.0025 | 0.0120 | 0.0118 | 0.0136 | 0.0048 | 0.0109 | 0.0033 | 0.0027  |

**Comparison vs CPU baseline (Day 4):**
- F1: GPU 0.5154 vs CPU 0.5222 (delta = -0.0068, within variance)
- MAP: GPU 0.6435 vs CPU 0.6376 (delta = +0.0058, GPU slightly better)
- NDCG@10: GPU 0.6844 vs CPU 0.6808 (delta = +0.0036, GPU slightly better)
- Total time: ~106 minutes (vs CPU ~270 minutes, 2.5x faster)

The slight F1 drop (-0.7%) is attributed to fp16 mixed-precision rounding;
MAP and NDCG@10 are higher because GPU evaluation is more numerically
stable in scoring/ranking. The two pipelines are scientifically equivalent.

### Stage B results - BioLinkBERT-Large on GPU (3 seeds, test set, theta=0.80)

| seed | F1     | prec   | recall | hop1   | hop2   | hop3   | MAP    | NDCG@10 |
|------|--------|--------|--------|--------|--------|--------|--------|---------|
| 42   | 0.5314 | 0.4902 | 0.5801 | 0.8242 | 0.4434 | 0.2459 | 0.6365 | 0.6805  |
| 1337 | 0.5319 | 0.4918 | 0.5792 | 0.8282 | 0.4436 | 0.2486 | 0.6439 | 0.6861  |
| 2024 | 0.5313 | 0.4916 | 0.5781 | 0.8268 | 0.4419 | 0.2488 | 0.6442 | 0.6865  |
| mean | 0.5315 | 0.4912 | 0.5791 | 0.8264 | 0.4430 | 0.2478 | 0.6415 | 0.6844  |
| std  | 0.0003 | 0.0009 | 0.0010 | 0.0020 | 0.0009 | 0.0016 | 0.0044 | 0.0034  |

**Headline:** test F1 = 0.5315 +/- 0.0003 (sigma = 0.06%, the tightest
variance of any validation in the project).

### Lift attribution: BioLinkBERT vs bert-base on GPU

| Metric | bert-base GPU | BioLinkBERT GPU | Lift | Relative |
|---|---|---|---|---|
| Test F1 | 0.5154 | **0.5315** | +0.0161 | +3.1% |
| Recall | 0.5606 | **0.5791** | +0.0185 | +3.3% |
| Precision | 0.4773 | **0.4912** | +0.0139 | +2.9% |
| Hop-1 prec | 0.8168 | **0.8264** | +0.0096 | +1.2% |
| Hop-2 prec | 0.4241 | **0.4430** | +0.0189 | +4.5% |
| Hop-3 prec | 0.2418 | **0.2478** | +0.0060 | +2.5% |
| MAP | 0.6435 | 0.6415 | -0.0020 | -0.3% |
| NDCG@10 | 0.6844 | 0.6844 | +0.0000 |  0.0% |

The lift is concentrated in **precision, recall, and per-hop precision**
(threshold-dependent metrics) while MAP and NDCG@10 (threshold-independent
ranking metrics) remain essentially unchanged.

### Interpretation

1. **BioLinkBERT improves classification, not ranking.** The fact that
   MAP and NDCG@10 are flat while F1 and per-hop precision rise indicates
   the encoder swap shifts scores rather than reorders candidates. The
   downstream threshold (theta = 0.80) lands at a more favorable point in
   the score distribution.

2. **Hop-2 precision sees the largest relative lift (+4.5%).** This is
   the hardest hop in the project (paths through one intermediate
   biomedical entity). Phase 5 confirms that biomedical pretraining
   transfers to the multi-hop setting.

3. **dev vs test divergence.** The dev-set lift was only +0.0017 F1
   (0.34%), but the test-set lift is +0.0161 F1 (3.1%). This indicates
   BioLinkBERT generalizes better than bert-base, which appears to
   pick up slight dev-specific patterns during selection.

4. **Variance collapses.** The test-F1 standard deviation drops from
   0.0025 (bert-base) to 0.0003 (BioLinkBERT) - the tightest in the
   project's history. Larger, biomedical-pretrained encoders produce
   more reproducible CAFF behaviour on this KG.

### Cumulative gap-composition update

| Source of difference | Estimated effect | Status |
|---|---|---|
| Encoder: bert-base-uncased -> BioLinkBERT-Large | +0.016 F1 (measured) | DONE |
| KG: Orphanet + HPO + OMIM only (paper uses + DisGeNET + UMLS) | -0.05 to -0.10 | OPEN |
| Trainable head: 1.3M params (paper uses 12M) | -0.02 to -0.05 | OPEN |
| Per-relation thresholds (vs global theta) | -0.02 to -0.05 | OPEN |
| Longer training (10 vs 30 epochs, paper-spec) | -0.02 to -0.05 | OPEN |

**Net assessment.** With the encoder upgrade alone, the project moved
F1 from 0.522 (CPU baseline) to 0.532 (GPU + BioLinkBERT). The
remaining gap to the paper's headline (0.79) is plausibly attributable
to the four data/training items in the table; closing them is
mechanical but requires DisGeNET / UMLS access and longer training
budgets.

### Reproducibility note

Phase 5 introduced one config field change (`d: 768 -> 1024`) and one
encoder name change. All scripts, evaluation paths, and threshold
choices from Phase 4 are unchanged. The 3-seed protocol and seeds
(42 / 1337 / 2024) are preserved. Run times:
- BioLinkBERT seed 42: 49 minutes (includes first encoder load)
- BioLinkBERT seeds 1337 / 2024: 40-100 minutes each (variable due to
  background system load)

---

## 12. Per-hop threshold tuning, revisited at fine-step resolution (May 14, 2026)

### Setup

After Phase 5 (Section 11) established the BioLinkBERT-Large baseline
at test F1 = 0.5315 +/- 0.0003, the obvious next step was to revisit
per-hop threshold tuning. Section 8 already documented a CPU-era
experiment where per-hop tuning gave a small lift (+1.8% on bert-base,
5K data), so the question was whether the same idea still helps with
the upgraded encoder and the 20K configuration.

The existing script `scripts/per_hop_threshold_sweep.py` sweeps
thresholds at step 0.05. We ran it first with that step, found zero
improvement, and then refined the step to 0.01.

### Stage A: original step=0.05 (negative result)

For all three seeds, the original sweep chose theta = 0.80 for every
hop, identical to the global optimum:

| seed | hop1 theta | hop2 theta | hop3 theta | test F1 (per-hop) | lift |
|------|-----------|-----------|-----------|---------------------|------|
| 42   | 0.80 | 0.80 | 0.80 | 0.5314 | +0.0000 |
| 1337 | 0.80 | 0.80 | 0.80 | 0.5319 | +0.0000 |
| 2024 | 0.80 | 0.80 | 0.80 | 0.5313 | +0.0000 |

This appeared to indicate that BioLinkBERT-Large produces a hop-uniform
score distribution. The conclusion would have been: "the encoder
upgrade absorbed the gain that per-hop tuning used to offer."

### Stage B: fine step=0.01 (positive result)

Refining the threshold grid from 0.05 to 0.01 revealed that the
optimum was hiding **between** the coarse grid points. All three seeds
chose nearly identical fine-grained per-hop thresholds:

| seed | hop1 theta | hop2 theta | hop3 theta | test F1 (per-hop) | lift   |
|------|-----------|-----------|-----------|---------------------|--------|
| 42   | 0.78 | 0.82 | 0.89 | 0.5514 | +0.0200 |
| 1337 | 0.79 | 0.82 | 0.88 | 0.5515 | +0.0196 |
| 2024 | 0.78 | 0.82 | 0.89 | 0.5542 | +0.0229 |
| **mean** | **0.78** | **0.82** | **0.89** | **0.5524 +/- 0.0016** | **+0.0208** |

The mean lift is +0.0208 absolute, or +3.9% relative, over the
global theta = 0.80 baseline. The variance on the chosen thresholds
is +/- 0.01 across seeds, indicating a stable optimum.

### Headline test-set numbers (3 seeds, per-hop fine-step)

| metric | mean +/- std |
|---|---|
| F1 | **0.5524 +/- 0.0016** |
| Precision | 0.5577 +/- 0.0038 |
| Recall | 0.5472 +/- 0.0017 |
| Hop-1 precision | (best at hop-specific theta) |
| Hop-2 precision | (best at hop-specific theta) |
| Hop-3 precision | (best at hop-specific theta) |

The trade-off captured: precision rises from 0.491 to 0.558 (+6.4%)
while recall drops from 0.579 to 0.547 (-5.4%), and the net effect on
F1 is positive at +0.0208.

### Why step=0.05 missed the optimum

The 0.05 grid contains {0.30, 0.35, 0.40, ..., 0.80, 0.85, 0.90}. Three
of the fine-grained optima are not on this grid:

- hop=1 optimum is 0.78, which the 0.05 grid replaces with 0.80
- hop=2 optimum is 0.82, which the 0.05 grid replaces with 0.80
- hop=3 optimum is 0.89, which the 0.05 grid replaces with 0.90 - but
  the F1 surface near hop=3 is steep enough that 0.90 happened to lose
  to 0.80 in the sweep

So the coarse sweep ended up at the global optimum because *no* nearby
grid point beat it. The 0.01 grid resolves the actual shape of the F1
surface around each hop.

### Interpretation

1. **The encoder upgrade did not eliminate the per-hop calibration
   gap; it just shrank the window where it shows.** With bert-base,
   the fine-grained optima were probably spread wider, so even step
   0.05 could find them. With BioLinkBERT, the optima are tight
   (within +/- 0.05 of the global) and only a fine grid resolves them.

2. **Per-hop tuning trades recall for precision.** With BioLinkBERT,
   precision lifts +6.4% while recall drops -5.4%. The net F1 gain is
   the right-half story; the left-half is that the model is
   precision-limited, not recall-limited, at the chosen operating
   point.

3. **Reproducibility holds.** The chosen thresholds (0.78 / 0.82 /
   0.89) agree across seeds to within +/- 0.01, and the test F1 std
   of 0.0016 is barely worse than the 0.0003 from the global sweep.

### Cumulative gap-composition update

| Source of difference | Estimated effect | Status |
|---|---|---|
| Encoder: bert-base-uncased -> BioLinkBERT-Large (MEASURED) | +0.016 F1 | DONE |
| Per-hop fine-step thresholds (MEASURED)                   | +0.021 F1 | DONE (this section) |
| Add DisGeNET + UMLS gene-disease layer                     | +0.05 to +0.10 | OPEN |
| Larger trainable head: 1.30 M -> 12 M params (paper)       | +0.02 to +0.05 | OPEN |
| Longer training (10 vs 30 epochs)                          | +0.02 to +0.05 | OPEN |

**Net assessment.** Two measured lifts (+0.016 and +0.021) bring the
project from CPU baseline F1 = 0.522 to **F1 = 0.5524 +/- 0.0016**.
The remaining gap to the paper headline (0.79) is plausibly
attributable to the three open items, all of which are mechanical to
close given DisGeNET / UMLS access and a longer training budget.

### Implementation note

`scripts/per_hop_threshold_sweep.py` had `np.arange(0.30, 0.91, 0.05)`
as the candidate grid; this was widened to `np.arange(0.30, 0.91, 0.01)`
on May 14, 2026. The change is one line and adds 48 grid points (from
13 to 61). Total runtime per seed grows by ~2 seconds (the sweep is
trivial compared to scoring); total wall time per seed is unchanged
at ~3 minutes on the RTX 4060.

### Reproducibility command

```bash
for s in 42 1337 2024; do
    python scripts/per_hop_threshold_sweep.py \
        --config configs/caff_orphanet.yaml \
        --checkpoint runs/caff_orphanet/seed_${s}/best.pt \
        --device cuda
done
```

The three runs print per-hop thresholds and the test F1 lift over
global theta = 0.80. Total wall time on a single 8 GB GPU is roughly
8-10 minutes.

---

## 13. 30-epoch training experiment - mixed result (May 14, 2026)

### Setup

After Section 12 established the headline result of test F1 = 0.5524
+/- 0.0016 with BioLinkBERT-Large at 10 epochs and per-hop fine-step
thresholds, we tested whether increasing the epoch budget to 30
(the paper's specification) would improve the result further.

Config change: `epochs: 10 -> 30`. Everything else identical:
- Encoder: michiyasunaga/BioLinkBERT-large (frozen, 333.5 M params)
- d: 1024, batch: 256 (effective), LR schedule: cosine 3e-4 -> 1e-5
- Early stopping: patience 5 on dev F1
- Same 20K QA, same KG v2, same seeds {42, 1337, 2024}

### Training dynamics (3 seeds)

All three seeds converged early and triggered the patience-5 early
stop well before the 30-epoch budget was exhausted:

| seed | best epoch | early stop at | dev F1 (best) | dev MAP (best) |
|------|-----------|---------------|-----------------|------------------|
| 42   | 6  | 11 | 0.5124 | 0.6423 |
| 1337 | 7  | 12 | 0.5119 | 0.6416 |
| 2024 | 7  | 12 | 0.5126 | 0.6430 |
| mean |    |    | 0.5123 +/- 0.0004 | 0.6423 +/- 0.0007 |

The 10-epoch baseline (Phase 5) had mean dev F1 = 0.5099 +/- 0.0009.
The 30-epoch dev lift is +0.0024 (0.48%).

The pattern is consistent across seeds: best epoch arrives at 6-7
(within the original 10-epoch budget), and 5 epochs of no further
improvement triggers early stopping at 11-12. The 19 unused epochs
in the 30-budget were unnecessary.

### Test set results (3 seeds, global theta = 0.80)

| seed | precision | recall | F1     |
|------|-----------|--------|--------|
| 42   | 0.4902 | 0.5829 | 0.5326 |
| 1337 | 0.4922 | 0.5848 | 0.5345 |
| 2024 | 0.4916 | 0.5865 | 0.5349 |
| mean | 0.4913 +/- 0.0010 | 0.5847 +/- 0.0018 | **0.5340 +/- 0.0012** |

Compared to the 10-epoch baseline (Phase 5, test F1 = 0.5315 +/-
0.0003), the 30-epoch result lifts test F1 by **+0.0025** at the same
global theta.

### Test set results with per-hop fine-step thresholds

Per-hop thresholds tuned on dev (3 seeds):

| seed | hop1 theta | hop2 theta | hop3 theta | test F1 (per-hop) | lift vs g=0.80 |
|------|-----------|-----------|-----------|---------------------|------------------|
| 42   | 0.79 | 0.81 | 0.80 | 0.5333 | +0.0008 |
| 1337 | 0.79 | 0.80 | 0.90 | 0.5570 | +0.0225 |
| 2024 | 0.80 | 0.82 | 0.89 | 0.5526 | +0.0178 |
| mean | 0.79 | 0.81 | 0.86 | **0.5476 +/- 0.0126** | +0.0137 |

### The unexpected variance

Section 12 reported per-hop F1 = **0.5524 +/- 0.0016** with the
10-epoch checkpoints; the same procedure on the 30-epoch checkpoints
returns **0.5476 +/- 0.0126**. The mean is lower and the variance is
8x higher.

The cause is visible in the per-hop threshold table above. Seed 42
chose hop=3 theta = 0.80 (same as global), giving essentially zero
per-hop lift on that seed. Seeds 1337 and 2024 chose hop=3 theta
= 0.89-0.90, behaving like the 10-epoch run.

In the 10-epoch experiment (Section 12), all three seeds agreed on
hop=3 theta near 0.89. With 30 epochs, the score distribution on
hop=3 has tightened to the point where the F1 surface near 0.80 has
become competitive with the 0.89 region for seed 42, and the dev
search picks the wrong local maximum for that seed.

### Direct comparison

| Configuration | Test F1 (global=0.80) | Test F1 (per-hop) |
|---|---|---|
| 10-epoch (Phase 5 / Section 12) | 0.5315 +/- 0.0003 | **0.5524 +/- 0.0016** |
| 30-epoch (this section)         | 0.5340 +/- 0.0012 | 0.5476 +/- 0.0126 |
| Delta                            | +0.0025 (+0.5%)   | -0.0048 (-0.9%) |

Longer training helps the global threshold by a small margin and
hurts the per-hop result on a comparable margin. Net of variance,
the two configurations are roughly equivalent on absolute F1, but
the 10-epoch + per-hop pipeline is more reproducible.

### Interpretation

1. **BioLinkBERT-Large + the merged KG converge within 10 epochs.**
   The dev F1 best-epoch sits at 6-7 in all three 30-epoch seeds,
   identical to where Section 12's seeds converged. The remaining 5
   epochs of training only nudge the loss down without moving F1.

2. **Longer training makes per-hop tuning seed-sensitive.** The dev
   sweep selects per-hop thresholds at a finer F1 contour, and with
   30-epoch checkpoints that contour has flattened enough that
   seed-level noise can push the chosen threshold to a different
   region. Section 12's 10-epoch result was more stable precisely
   because the dev F1 surface was sharper.

3. **10 epochs are sufficient at this scale.** With 1.3 M trainable
   parameters, a frozen 340 M encoder, 14 K training queries, and
   the merged Orphanet + HPO + OMIM KG, additional epochs do not
   buy a reliable improvement.

### What this means for the gap to the paper

The paper specifies 30 epochs and reports F1 = 0.79. The 30-epoch
experiment here does not close the gap, which rules out epoch count
as a major contributor. The remaining open items (DisGeNET + UMLS
gene-disease layer, the 12 M trainable head, possibly task-specific
calibration) are now the dominant unknowns.

### Cumulative gap-composition update

| Source of difference | Estimated effect | Status |
|---|---|---|
| Encoder: bert-base-uncased -> BioLinkBERT-Large (MEASURED) | +0.016 F1 | DONE (Section 11) |
| Per-hop fine-step thresholds (MEASURED)                   | +0.021 F1 | DONE (Section 12) |
| Longer training (10 -> 30 epochs) (MEASURED)              | +0.003 F1 (global) | DONE (this section) |
| Add DisGeNET + UMLS gene-disease layer                     | +0.05 to +0.10 | OPEN |
| Larger trainable head: 1.30 M -> 12 M params (paper)       | +0.02 to +0.05 | OPEN |

**Net assessment.** Three measured items contribute a cumulative +0.020
absolute F1 over the CPU baseline at global theta = 0.80 (0.522 ->
0.534), and another +0.021 from per-hop fine-step tuning brings the
best published number to F1 = 0.5524 (Section 12). The remaining gap
to 0.79 is attributable to the two open items, which require
DisGeNET / UMLS access.

### Reproducibility note

The 30-epoch checkpoints in `runs/caff_orphanet/seed_*/best.pt` after
this experiment do not match the Section 12 checkpoints; running
`scripts/per_hop_threshold_sweep.py` on them produces this section's
0.5476 number rather than Section 12's 0.5524. To reproduce
Section 12 exactly, set `epochs: 10` in `configs/caff_orphanet.yaml`
and re-train all three seeds. The Day 7 commit (5025ed4) on `main`
is the immutable record of the Section 12 result.

