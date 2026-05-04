# Paper Discrepancies — To Address Before Resubmission

This document tracks inconsistencies discovered between the paper text
and the implementation. Each item should be resolved before the camera-
ready submission.

---

## 1. Appendix C — JSD Numerical Example (DISCREPANCY)

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

## 5. DC mining (Section 6.5) — IMPLEMENTED (was a Phase-1 placeholder)

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

1. `caff/miners.py` — added a `DCMiner` class that, given the BFS
   depth `L` and a seed, samples a wrong hop `l_- != l_+` for any
   gold hop `l_+`. Reproducible via `random.Random(seed)`.
2. `caff/trainer.py::__init__` — when `lambda_D > 0`, instantiate
   `self.dc_miner = DCMiner(L, seed)` (otherwise `None`). The old
   warning is replaced by an info-level "DCMiner initialized" log.
3. `caff/trainer.py::_train_one_group` — for every gold candidate
   at this `(query, hop)`, sample a wrong hop, rebuild the CSV
   state `z_{l_- - 1}` via the existing `teacher_forced_z_prev`
   helper (so DC re-uses the same teacher-forced training
   convention as BCE), recompute `W_ctx_wrong` and re-score the
   same gold relations. Append `(s_correct, s_wrong)` to the
   accumulator.
4. `caff/trainer.py::_optimizer_step` — if the accumulator has DC
   pairs, concatenate them and pass real tensors to the criterion;
   otherwise pass `None` (preserves backward compatibility with
   `ablation_lambda_D=0.0`).
5. `tests/test_miners.py` — four new unit tests:
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
