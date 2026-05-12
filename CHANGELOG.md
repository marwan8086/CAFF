# Changelog

All notable changes to CAFF are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For a running list of paper-vs-implementation discrepancies and experiment
notes, see [PAPER_DISCREPANCIES.md](PAPER_DISCREPANCIES.md).

---

## [Unreleased]

### Planned (not yet implemented)

- **Phase 5**: replace `bert-base-uncased` with `BioLinkBERT-Large` on
  a CUDA device. Expected F1 lift: +0.05 to +0.10.
- DisGeNET gene-disease layer integration (requires registration).
- Open Targets Parquet pipeline (requires `pyarrow`).
- Per-relation threshold tuning to handle heterogeneous relation
  signal strength (motivated by the KG v3 MONDO experiment).
- Longer training (30 epochs) on KG v3 to test whether the MONDO
  candidate noise can be learned away.

---

## [0.6.0] - 2026-05-12

### Documentation transparency pass

This release brings the repository's documentation in line with the
empirically measured behaviour of the as-shipped pipeline, so that a
reviewer cloning the repo today understands exactly what the code can
deliver versus what the paper claims.

### Added

- README: new `Implementation Reality Check` section (133 lines)
  documenting the CPU configuration, measured 3-seed numbers
  (F1 = 0.522 +/- 0.001), the gap-composition analysis, the
  5K-vs-20K data-scaling experiment, the MONDO negative result, and
  the full reproduction workflow.
- README: new `Repository Structure` section (96 lines) with an
  accurate tree of `caff/`, `scripts/`, `configs/`, `tests/`, and
  root-level files. Fixes a broken TOC anchor that pointed to a
  non-existent section.
- README: cross-reference notes on `Main Results`, `Data Preparation`,
  and `Training` directing readers to `Implementation Reality Check`
  for the as-shipped CPU workflow.
- CONTRIBUTING.md: new `Recent milestones` section listing the
  resolved Phase 2 / Phase 3 work, plus a refreshed
  `Known gaps (good first issues)` listing Phase 5+ work.

### Changed

- CONTRIBUTING.md: expected test count updated from 48 to 52 to
  reflect the four `DCMiner` tests added in v0.3.0.
- PAPER_DISCREPANCIES.md: replaced 11 remaining em-dashes with
  hyphens for ASCII consistency, completing a cleanup started in
  the upstream web edit (commit 14b269d).

### Fixed

- `scripts/build_kg.py` and `scripts/extract_bfs.py`: stripped a
  leading UTF-8 BOM (U+FEFF) that broke `ast.parse()` and some
  Linux tooling. Files remain valid UTF-8 and behave identically.

---

## [0.5.0] - 2026-05-07

### Headline result: F1 = 0.522 +/- 0.001 on held-out test (20K, 3 seeds)

### Added

- 3-seed validation (seeds 42 / 1337 / 2024) on a 20,000-record QA
  set sampled from the Orphanet + HPO + OMIM knowledge graph.
- PAPER_DISCREPANCIES.md section 10 with the full per-seed table,
  the 5K-vs-20K comparison, and the gap-composition analysis.

### Changed

- Recommended training data scale: 5,000 -> 20,000 QA records
  (lifts test F1 from 0.509 +/- 0.005 to 0.522 +/- 0.001, with
  recall improving by +14.8% absolute and F1 variance shrinking 5x).

### Notes

- Each 20K seed takes roughly 80 minutes of single-thread CPU time.
- Headline number reported with the same theta = 0.80 chosen on the
  5K dev sweep; revisiting the threshold on 20K data does not change
  the optimum.

---

## [0.4.0] - 2026-05-04 to 2026-05-05

### Real-data validation and threshold tuning

### Added

- `scripts/threshold_sweep.py`: dev-set sweep over a list of theta
  values, reporting precision, recall, F1, MAP, NDCG@10, and per-hop
  precision at each threshold.
- `scripts/per_hop_threshold_sweep.py`: independent theta selection
  per hop. Lifts test F1 from 0.5017 to 0.5107 (+1.8%) on the 5K
  configuration.
- `scripts/convert_mondo_to_tsv.py` and
  `scripts/merge_mondo_into_kg.py`: experimental KG v3 with the
  MONDO disease ontology (26,709 terms, 39,858 is_a edges, 17,056
  equivalent_to xrefs).
- 3-seed validation on 5K QA records: F1 = 0.509 +/- 0.005.
- PAPER_DISCREPANCIES.md sections 7, 8, 9 documenting the HPO+OMIM
  integration, the threshold trade-off, and the MONDO experiment.

### Changed

- Default retention threshold: `theta = 0.50` (paper) -> `theta = 0.80`
  (chosen from the dev sweep on the real Orphanet+HPO+OMIM KG).

### Negative result (kept on record)

- KG v3 (with MONDO) improves MAP (+5%) and NDCG (+8%) but reduces
  F1 by 7% at the chosen threshold because it injects many
  borderline-confident candidates that the model has not learned to
  suppress. The merger scripts are kept for future work that adds
  per-source thresholds or longer training.

---

## [0.3.0] - 2026-05-04

### Phase 2 (DC mining) and Phase 3 (real biomedical KG) complete

### Added

- `caff/miners.py::DCMiner`: depth-contrastive mining that, given a
  gold (head, relation, tail, hop) tuple, samples a wrong-hop
  counterpart from the BFS-stratified candidates. Wired into
  `caff/trainer.py` behind the `lambda_D > 0` switch.
- Four new unit tests in `tests/test_miners.py` covering the
  DCMiner: invalid-L raises, sample-wrong-hop excludes gold,
  invalid-gold-hop raises, reproducible-with-seed (total test count
  rises from 48 to 52).
- `scripts/convert_orphanet_xml_to_tsv.py`: parses Orphanet's
  `en_product1.xml`, `en_product4.xml`, `en_product6.xml` into TSV.
- `scripts/build_kg.py` (full rewrite): produces a 19,049-entity /
  124,253-triple KG from the Orphanet TSV exports.
- `scripts/convert_hpo_to_tsv.py` + `scripts/merge_hpo_into_kg.py`:
  add HPO ontology + OMIM annotations to yield
  `merged_kg_v2.tsv` (38,456 nodes, 291,335 edges, 11 relations).
- `scripts/build_orphanet_qa.py`: deterministic QA sampler that
  produces train/dev/test JSON splits with controlled hop
  distribution.
- `configs/caff_orphanet.yaml`: CPU-friendly config for the real-data
  pipeline.

### Documented

- PAPER_DISCREPANCIES.md sections 5 and 6 (DC mining implementation
  notes; smoke-F1 plateau analysis).

---

## [0.2.0] - 2026-04-29

### CI + correctness fixes

### Added

- `.github/workflows/tests.yml`: GitHub Actions CI that lints and
  runs the full pytest suite on every push to main.
- CONTRIBUTING.md (initial version): dev-environment setup, test
  expectations, coding style, pull-request workflow.

### Fixed

- **Smoke data reachability** (commit a6f0944): the synthetic
  fixtures used `nx.Graph` (undirected), but the production code
  treats the KG as a directed graph. Reachability was 15.6%;
  switching the fixtures to `nx.DiGraph` brought it to 100% and
  unlocked the full L = 3 training (~16x more triples, ~6x F1).
- **BFS cache invalidation** (commit ab326ac): cache filenames did
  not include `L` and `K_r`, so changing either silently reused a
  stale cache. Cache keys now incorporate both.
- **CI escape bugs** (commits 7af6c00, 46aa885): single-quote
  escaping in `hashFiles()` and import-test print statements.

### Documented

- PAPER_DISCREPANCIES.md sections 2, 3 (smoke fixtures fix; BFS
  cache invalidation).

---

## [0.1.0] - 2026-04-28

### Phase 1 - core pipeline works end-to-end

### Added

- Full implementation of CAFF's Stage 1 to Stage 4 from the paper:
  `caff/data.py` (BFS candidate stratification), `caff/csv.py`
  (Contextual Summary Vector), `caff/dbm.py` (Dynamic Bilinear
  Modulation), `caff/losses.py` (BCE + HC3 + DC), `caff/miners.py`
  (HC3 buffer + miner), `caff/scorer.py` (DepthBilinear + HopScorer),
  `caff/model.py` (`CAFFModel`), `caff/trainer.py` (`CAFFTrainer`).
- Test suite of 48 unit tests in `tests/` covering each component.
- Top-level entry points: `train.py`, `evaluate.py`,
  `context_swap_diagnostic.py`.
- Paper-aligned configs: `caff_full.yaml`, `caff_no_hc3.yaml`,
  `depthbilinear.yaml`.
- `PAPER_DISCREPANCIES.md` (initial) tracking the JSD numerical
  example discrepancy (section 1).

### Fixed

- Eight critical correctness issues in the initial code drop
  (paper Section 6.3 implementation, BFS expansion direction,
  loss normalization, miner buffer overflow, etc.). See commit
  787234f for the consolidated fix.
- Test suite repair: rewrote tests that relied on internal symbols
  that no longer existed (commit 422f9d9).

### Removed

- Old artifacts from the pre-clean-room scaffold (`CAFF/`
  directory, `caff_expanded_*.json`, `caff_demo.py`,
  `test_components.py`, `caff_results.py`, etc.).

---

## [0.0.x] - 2026-04-25 to 2026-04-27

### Pre-implementation

Initial repository setup. README iteration, author / affiliation
edits, first code upload, badge polish. No functional code yet.

---

[Unreleased]: https://github.com/marwan8086/CAFF/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/marwan8086/CAFF/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/marwan8086/CAFF/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/marwan8086/CAFF/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/marwan8086/CAFF/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/marwan8086/CAFF/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/marwan8086/CAFF/releases/tag/v0.1.0
