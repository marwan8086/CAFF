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

Expected: 48 tests passing in roughly 3-10 seconds (depending on machine).

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

## Known gaps (good first issues)

These items are documented in the code with `TODO(Phase-2)` or
`TODO(Phase-3)` markers:

- **DC mining** (Phase 2): `DepthContrastiveLoss` is fully implemented but
  the miner that produces (correct_hop, wrong_hop) score pairs is not. See
  `caff/trainer.py::_optimizer_step` for details.
- **Real biomedical data pipeline** (Phase 3): `scripts/build_kg.py` exists
  but has not been validated end-to-end against Orphanet / DisGeNET / OMIM
  releases. The smoke fixtures in `tests/fixtures/` are synthetic.

If you want to tackle either of these, please open an issue first so we can
coordinate.

---

## Contact

For questions about the algorithm or paper, contact the corresponding author:

- Marwan Dhifallah  (marwan@mail.dlut.edu.cn)
- Yu Liu (supervisor)  (yuliu@dlut.edu.cn)

For implementation-only questions, please open a GitHub issue.
