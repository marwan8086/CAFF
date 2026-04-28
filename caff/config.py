"""
caff/config.py
==============
Centralized configuration for CAFF, mirroring §8.4 of the paper.

All hyperparameters are paper-faithful defaults. The dataclass is
frozen and hashable for reproducibility manifests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class CAFFConfig:
    """Paper-faithful configuration for CAFF.

    All defaults match §8.4 (Implementation Details) of the paper.
    Override only via explicit constructor arguments — never mutate.

    References
    ----------
    Paper §8.4: "ρ=16, L=3, θ=0.50, γ_D=0.20, γ_C=0.25,
                 λ_D=0.40, λ_C=0.35, K_r=20.
                 Optimizer: AdamW, base lr=3e-4, weight decay 1e-2,
                 batch size 256, 30 epochs, 2-epoch linear warmup,
                 cosine decay to 1e-5. Gradients clipped at ||∇||_2=1.0.
                 Early stopping on dev set with patience 5."
    """

    # ─── Architecture (paper §6) ────────────────────────────────
    d: int = 768                     # BioLinkBERT-Large hidden size
    rho: int = 16                    # DBM rank (paper §6.3)
    L: int = 3                       # Max BFS hop depth
    theta: float = 0.50              # Retention threshold (Eq. 2)
    K_r: int = 20                    # FreqCap per (head, relation), Eq. 13

    # ─── Loss weights and margins (paper §6.4–6.5) ──────────────
    gamma_C: float = 0.25            # HC3 margin (Eq. 21)
    gamma_D: float = 0.20            # Depth-contrastive margin (Eq. 23)
    lambda_C: float = 0.35           # HC3 weight (Eq. 22)
    lambda_D: float = 0.40           # DC weight (Eq. 22)

    # ─── HC3 mining (paper §6.4) ────────────────────────────────
    hc3_buffer_size: int = 1000      # Rolling buffer of recent instances
    hc3_negatives_per_anchor: int = 8
    hc3_buffer_refresh_steps: int = 500

    # ─── Optimization (paper §8.4) ──────────────────────────────
    lr: float = 3e-4                 # Base learning rate
    lr_min: float = 1e-5             # Cosine decay target
    weight_decay: float = 1e-2
    batch_size: int = 256            # Effective batch (paper §8.4)
    epochs: int = 30
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    early_stop_patience: int = 5
    early_stop_metric: Literal["f1", "map", "loss"] = "f1"

    # ─── Encoder (paper §8.4) ───────────────────────────────────
    encoder_name: str = "michiyasunaga/BioLinkBERT-large"
    encoder_frozen: bool = True      # ALWAYS True per paper §8.4

    # ─── Data ───────────────────────────────────────────────────
    kg_path: str = "data/processed/merged_kg.tsv"
    train_path: str = "data/processed/train.json"
    dev_path: str = "data/processed/dev.json"
    test_path: str = "data/processed/test.json"
    min_relation_freq: int = 50      # Singleton-relation removal threshold (paper §8.1)

    # ─── Reproducibility (paper §8.4) ───────────────────────────
    seed: int = 42                   # Paper seeds: {42, 1337, 2024}
    deterministic: bool = True

    # ─── Hardware adaptation (NOT in paper — Colab compat) ──────
    # The paper trained on 1× A100-80GB with batch=256.
    # On smaller GPUs we use gradient accumulation to preserve the
    # effective batch size while reducing memory pressure.
    micro_batch_size: int = 256      # Auto-adjusted at runtime
    grad_accum_steps: int = 1        # Auto-adjusted at runtime
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"

    # ─── Validation ─────────────────────────────────────────────
    def __post_init__(self) -> None:
        # All paper invariants must hold exactly. We use raise instead of
        # assert because Python's -O flag strips assert statements, which
        # would silently disable validation in optimized runs.
        if self.rho >= self.d:
            raise ValueError(f"rho ({self.rho}) must be < d ({self.d})")
        if self.L < 1:
            raise ValueError(f"L must be >= 1, got {self.L}")
        if not (0 < self.theta < 1):
            raise ValueError(f"theta must be in (0,1), got {self.theta}")
        if self.K_r <= 0:
            raise ValueError(f"K_r must be > 0, got {self.K_r}")
        if self.gamma_C <= 0:
            raise ValueError(f"gamma_C must be > 0, got {self.gamma_C}")
        if self.gamma_D <= 0:
            raise ValueError(f"gamma_D must be > 0, got {self.gamma_D}")
        if self.lambda_C < 0 or self.lambda_D < 0:
            raise ValueError(
                f"lambda_C ({self.lambda_C}) and lambda_D ({self.lambda_D}) must be >= 0"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.micro_batch_size * self.grad_accum_steps != self.batch_size:
            raise ValueError(
                f"micro_batch_size ({self.micro_batch_size}) * "
                f"grad_accum_steps ({self.grad_accum_steps}) "
                f"must equal batch_size ({self.batch_size})"
            )
        if self.early_stop_metric not in {"f1", "map", "loss"}:
            raise ValueError(
                f"early_stop_metric must be one of 'f1', 'map', 'loss'; "
                f"got {self.early_stop_metric!r}"
            )
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"mixed_precision must be one of 'no', 'fp16', 'bf16'; "
                f"got {self.mixed_precision!r}"
            )

    def hash(self) -> str:
        """Content-addressable hash for reproducibility manifests."""
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CAFFConfig":
        return cls(**d)


# Variant configs for ablations (paper §10.1)
@dataclass(frozen=True)
class AblationFlags:
    """Toggle individual paper components for ablation study."""

    use_csv: bool = True             # If False: z_{ℓ-1} ≡ 0
    use_dbm: bool = True             # If False: Δ_ℓ ≡ 0
    use_hc3: bool = True             # If False: λ_C = 0
    use_dc: bool = True              # If False: λ_D = 0
    use_freqcap: bool = True         # If False: no Eq. 13
    csv_pool: Literal["mean", "max"] = "mean"  # paper §10.1
    gate_activation: Literal["sigmoid", "relu"] = "sigmoid"

    def __post_init__(self) -> None:
        if self.csv_pool not in {"mean", "max"}:
            raise ValueError(
                f"csv_pool must be 'mean' or 'max', got {self.csv_pool!r}"
            )
        if self.gate_activation not in {"sigmoid", "relu"}:
            raise ValueError(
                f"gate_activation must be 'sigmoid' or 'relu', "
                f"got {self.gate_activation!r}"
            )
