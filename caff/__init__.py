"""
CAFF — Context-Aware Feedback Filtering for Multi-Hop Biomedical
Knowledge Graph Evidence Selection.

Reference implementation accompanying the IEEE TKDE submission by
Marwan Dhifallah and Yu Liu (Dalian University of Technology).
"""

from __future__ import annotations

from .config import AblationFlags, CAFFConfig
from .csv import CSV
from .data import (
    CachedBFSExtractor,
    CAFFTripleDataset,
    KnowledgeGraph,
    QARecord,
    Triple,
    TripleInstance,
    apply_frequency_cap,
    bfs_candidate_sets,
    annotate_gold_relevance,
    load_qa_split,
)
from .dbm import ContextGate, DBM, DBMBlock
from .encoders import FrozenBioEncoder, RelationEmbeddingCache
from .evaluator import CAFFEvaluator, EvaluationReport, context_swap_diagnostic, paired_bootstrap
from .losses import BCEFilterLoss, CAFFCombinedLoss, DepthContrastiveLoss, HC3Loss
from .miners import HC3Buffer, HC3Miner, TrainingInstance, TripleKey
from .model import CAFFModel, HopOutput
from .scorer import DepthCorrection, HopScorer
from .trainer import CAFFTrainer, EpochMetrics, TrainingHistory

__version__ = "1.0.0"
__all__ = [
    "AblationFlags", "CAFFConfig",
    "CSV",
    "CachedBFSExtractor", "CAFFTripleDataset", "KnowledgeGraph",
    "QARecord", "Triple", "TripleInstance",
    "apply_frequency_cap", "bfs_candidate_sets",
    "annotate_gold_relevance", "load_qa_split",
    "ContextGate", "DBM", "DBMBlock",
    "FrozenBioEncoder", "RelationEmbeddingCache",
    "CAFFEvaluator", "EvaluationReport",
    "context_swap_diagnostic", "paired_bootstrap",
    "BCEFilterLoss", "CAFFCombinedLoss",
    "DepthContrastiveLoss", "HC3Loss",
    "HC3Buffer", "HC3Miner", "TrainingInstance", "TripleKey",
    "CAFFModel", "HopOutput",
    "DepthCorrection", "HopScorer",
    "CAFFTrainer", "EpochMetrics", "TrainingHistory",
]