# Contributing to CAFF

Thank you for your interest in contributing to CAFF. This document explains
how the project is structured and how to make changes that match the
existing style.

---

## Project structure

```
caff/                 Core implementation (paper Sections 6-7)
configs/              YAML training configurations
data/                 Real-data placeholder (KGs, QA splits)
examples/             Demonstration files (do NOT cite their numbers)
scripts/              Standalone utilities (KG building, BFS, annotation)
tests/                Unit tests + smoke fixtures
runs/                 Training artifacts (auto-generated)
```

Top-level entry points: `train.py`, `evaluate.py`, `context_swap_diagnostic.py`.

---

## Setting up a development environment

```bash
git clone https://github.com/marwan8086/caff.git
cd caff
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# or:  .venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
```

For the smoke test you also need to build the synthetic fixtures once:

```bash
python tests/fixtures/build_smoke_data.py
```

This produces `tests/fixtures/smoke_kg.tsv` and three QA JSON splits.

---

## Running the test suite

The full suite must pass before any commit:

```bash
python -m pytest tests/ -v
```

Expected: 52 tests passing in roughly 5-10 seconds (depending on machine).

If you add new functionality, add a corresponding unit test in `tests/`. We
prefer pytest's plain-function style over class-based tests; see existing
test files for examples.

---

## Smoke training (end-to-end check)

Before submitting changes that touch the training loop, run the CPU-friendly
smoke training to verify the pipeline still trains end-to-end:

```bash
python train.py --config configs/caff_smoke.yaml --seed 42
```

Expected output (abridged):

```
KG loaded: |V|=5,000  |E|=24,995  |R|=20
Built 20,982 triple instances
[Epoch 1/2] loss=0.04xx  dev_f1=0.0xxx  ...
[Epoch 2/2] loss=0.04xx  dev_f1=0.0xxx  ...
Training complete.
```

Loss should decrease across epochs. F1 numbers will be small because the
fixtures are random synthetic data; this is expected.

---

## Coding style

- Python 3.10+ syntax (we use `X | Y` unions, `list[...]`, etc.)
- Type-annotate function signatures
- Docstrings reference the paper's equation numbers when applicable, e.g.
  `Eq. 14`, `paper Section 8.4`
- Use ASCII characters in source files. Em-dashes, smart quotes, and other
  Unicode punctuation should be avoided in code, configs, and the README so
  that Windows cp1252 environments do not break logging
- Source files use LF line endings (enforced by `.gitattributes`)
- Configuration values must round-trip: every field added to `CAFFConfig`
  must be readable from YAML and validated in `__post_init__`

---

## Validating configurations

`CAFFConfig.__post_init__` raises `ValueError` for invalid combinations.
Prefer `raise ValueError(...)` over `assert`, because `python -O` strips
asserts in production. Example:

```python
if self.rho >= self.d:
    raise ValueError(f"rho ({self.rho}) must be < d ({self.d})")
```

---

## Reporting bugs

If you find a discrepancy between the paper and the code, add it to
`PAPER_DISCREPANCIES.md` rather than silently editing the code or the
paper. Include:

1. What the paper says
2. What the code produces
3. Which one is correct (with reasoning)
4. Recommended resolution

---

## Pull requests

1. Create a topic branch off `main`
2. Make focused commits with clear messages (one logical change per commit)
3. Run `pytest tests/` and the smoke training before pushing
4. Open a pull request describing the change and its motivation
5. Reference paper sections / equations when relevant

---

## Recent milestones

The following used to be open issues and have since been resolved:

- **Phase 2 - DC mining** (May 4, 2026): Implemented in `caff/miners.py` as
  `DCMiner`, wired into `caff/trainer.py::__init__` and exercised by four
  new unit tests in `tests/test_miners.py`. See PAPER_DISCREPANCIES.md
  section 5 for the design notes.
- **Phase 3 - Real biomedical KG** (May 4, 2026): `scripts/build_kg.py`
  now loads Orphanet TSV exports produced by `convert_orphanet_xml_to_tsv.py`,
  and `merge_hpo_into_kg.py` adds HPO + OMIM annotations to yield a
  38,456-node / 291,335-edge KG (called `merged_kg_v2.tsv`).
- **3-seed validation** (May 6, 2026): The full pipeline has been run with
  seeds 42 / 1337 / 2024 on 20K QA records, producing
  F1 = 0.522 +/- 0.001 on a held-out test set. See PAPER_DISCREPANCIES.md
  section 10 and the README's Implementation Reality Check.

---

## Known gaps (good first issues)

These items would close the remaining gap between the as-shipped
F1 = 0.522 and the paper's headline F1 = 0.79:

- **Phase 5 - GPU + BioLinkBERT-Large**: replace `bert-base-uncased` with
  `michiyasunaga/BioLinkBERT-large` (340 M frozen params). Expected lift:
  +0.05 to +0.10 F1. Requires a CUDA device with at least 16 GB VRAM.
- **DisGeNET integration**: add a gene-disease association layer to the
  KG. The current public DisGeNET tier requires registration and a
  manual license agreement; an issue is open to track API access.
- **Open Targets parquet pipeline**: Open Targets ships association data
  in Parquet rather than TSV. A small adapter using `pyarrow` would let
  us reuse the merge logic in `merge_hpo_into_kg.py`.
- **Per-relation thresholds**: `per_hop_threshold_sweep.py` tunes one
  theta per hop; a per-relation variant would help when the KG contains
  many high-volume relations of different signal strength (e.g. after
  the MONDO experiment in PAPER_DISCREPANCIES.md section 9).
- **MONDO follow-up**: the experimental KG v3 built by
  `merge_mondo_into_kg.py` improves MAP and NDCG but hurts F1 because
  the model has not learned to suppress the new candidates. Longer
  training or candidate filtering should recover the lost precision.

If you want to tackle any of these, please open an issue first so we can
coordinate.

---

## Contact

For questions about the algorithm or paper, contact the corresponding author:

- Marwan Dhifallah  (marwan@mail.dlut.edu.cn)
- Yu Liu (supervisor)  (yuliu@dlut.edu.cn)

For implementation-only questions, please open a GitHub issue.
