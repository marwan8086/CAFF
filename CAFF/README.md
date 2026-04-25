# CAFF: Context-Aware Feedback Filtering

This repository implements the CAFF architecture described in "CAFF: Context-Aware Feedback Filtering for Multi-Hop Biomedical Knowledge Graph Evidence Selection".

## Overview

CAFF enhances multi-hop biomedical knowledge graph filtering by conditioning each hop's triple scoring on the specific set of triples retained at the previous hop. The implementation contains:

- Contextual Summary Vector (CSV)
- Dynamic Bilinear Modulation (DBM)
- Hop-Conditioned Context Contrast (HC3)
- Depth-Contrastive (DC) loss
- BFS candidate generation with relation frequency cap

## Project structure

- `config.yaml`: configuration for training and evaluation
- `main.py`: entry point for training
- `data/`: KG building, BFS candidate generation, dataset preparation
- `models/`: CAFF model, encoder utilities, CSV, DBM
- `losses/`: BCE, DC, and HC3 losses
- `training/`: training loop and evaluation metrics
- `utils/`: generic utilities

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config config.yaml
```

## Notes

This implementation assumes that the merged medical KG and query JSON files are prepared in `data/`.
The data pipeline includes utilities to build the KG and to generate jump-labeled training examples.
