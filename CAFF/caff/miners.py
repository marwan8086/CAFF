"""
caff/miners.py
==============
HC3 triplet mining — paper §6.4.

Quoting paper §6.4:
  "For each anchor (Q, r, ℓ), positive contexts z^(a) are drawn
   from queries where the triple was labeled 1, negative contexts
   z^(b) from queries where it was labeled 0. We collect up to 8
   negatives per positive anchor from a rolling buffer of the
   1,000 most recent training instances, updated every 500
   gradient steps."

This module implements that rolling buffer and the negative
mining policy. Failure mode F4 (mining in-batch instead of from
the rolling buffer) is what would break the JSD reproducibility
claim in paper §10.4.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TripleKey:
    """Canonical key for a (Q, r, ℓ) anchor.

    Two instances with the same (query_id, relation, hop) form a
    candidate HC3 anchor — but only if they have OPPOSITE labels.
    """
    query_id: str
    relation: str
    hop: int


@dataclass
class TrainingInstance:
    """A single (Q, h, r, t, ℓ, y) training instance, plus its
    upstream context z_{ℓ-1}.

    Stored in the HC3 rolling buffer for negative mining.
    """
    query_id: str
    head: str
    relation: str
    tail: str
    hop: int                         # ℓ ∈ {1, ..., L}
    label: int                       # y ∈ {0, 1}
    z_prev: torch.Tensor             # z_{ℓ-1}, shape (d,)


class HC3Buffer:
    """Rolling buffer of recent training instances for HC3 mining.

    Per paper §6.4:
      • capacity = 1000 most recent instances
      • refreshed every 500 gradient steps (steps tracked externally)
      • indexed by (query_id, relation, hop) for fast lookup of
        same-anchor instances with opposite labels

    Implementation: a deque + a secondary index. When the deque
    is full, oldest instances are evicted, and the index is
    updated lazily.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._buffer: deque[TrainingInstance] = deque(maxlen=capacity)
        # Secondary index: anchor key → list of buffer positions
        # We rebuild this index lazily via _rebuild_index().
        self._index: dict[TripleKey, list[int]] = defaultdict(list)
        self._index_dirty = True

    def add(self, instance: TrainingInstance) -> None:
        """Add a new training instance. Oldest is evicted if full."""
        self._buffer.append(instance)
        self._index_dirty = True

    def add_batch(self, instances: list[TrainingInstance]) -> None:
        for inst in instances:
            self.add(inst)

    def _rebuild_index(self) -> None:
        """Rebuild the (Q, r, ℓ) → positions index from the deque."""
        self._index = defaultdict(list)
        for pos, inst in enumerate(self._buffer):
            key = TripleKey(inst.query_id, inst.relation, inst.hop)
            self._index[key].append(pos)
        self._index_dirty = False

    def __len__(self) -> int:
        return len(self._buffer)

    def get_negatives_for(
        self,
        anchor: TrainingInstance,
        k: int = 8,
        rng: random.Random | None = None,
    ) -> list[TrainingInstance]:
        """Mine up to k negative-context instances for a positive anchor.

        Parameters
        ----------
        anchor : TrainingInstance with anchor.label == 1
            The positive anchor (Q, r, ℓ, y=1, z^(a)).
        k : int
            Maximum number of negatives. Paper §6.4: 8.
        rng : random.Random
            For reproducible sampling.

        Returns
        -------
        List of up to k instances with the SAME (Q, r, ℓ) but
        label = 0. These provide the negative contexts z^(b).

        Notes
        -----
        Per paper §6.4, "negative contexts are drawn from queries
        where it [the same triple] was labeled 0". This means we
        look for buffered instances whose (Q, r, ℓ) matches the
        anchor's but whose label disagrees.
        """
        if anchor.label != 1:
            raise ValueError("HC3 anchors must be positive (label=1)")
        if rng is None:
            rng = random.Random()

        if self._index_dirty:
            self._rebuild_index()

        key = TripleKey(anchor.query_id, anchor.relation, anchor.hop)
        candidate_positions = self._index.get(key, [])
        # Filter to those with label = 0
        negatives = [
            self._buffer[pos]
            for pos in candidate_positions
            if self._buffer[pos].label == 0
        ]
        if len(negatives) <= k:
            return negatives
        return rng.sample(negatives, k)


class HC3Miner:
    """High-level HC3 triplet mining orchestrator.

    Wraps the buffer with the refresh schedule from paper §6.4
    ("updated every 500 gradient steps"). External training loop
    calls `step()` once per gradient step.
    """

    def __init__(
        self,
        buffer_capacity: int = 1000,
        negatives_per_anchor: int = 8,
        refresh_every: int = 500,
        seed: int = 42,
    ) -> None:
        self.buffer = HC3Buffer(capacity=buffer_capacity)
        self.negatives_per_anchor = negatives_per_anchor
        self.refresh_every = refresh_every
        self._step_count = 0
        self._rng = random.Random(seed)

    def step(self) -> None:
        """Increment the gradient-step counter."""
        self._step_count += 1

    def is_refresh_step(self) -> bool:
        """True every `refresh_every` steps. External callers should
        regenerate fresh HC3 triplets at these checkpoints."""
        return self._step_count > 0 and self._step_count % self.refresh_every == 0

    def mine_triplets(
        self,
        positive_anchors: list[TrainingInstance],
    ) -> list[tuple[TrainingInstance, list[TrainingInstance]]]:
        """For a batch of positive anchors, mine k negatives each.

        Parameters
        ----------
        positive_anchors : list of TrainingInstance
            All must have label == 1.

        Returns
        -------
        List of (anchor, negatives) pairs. Anchors with no
        available negatives are silently skipped (loss ignores them).
        """
        triplets = []
        for anchor in positive_anchors:
            negs = self.buffer.get_negatives_for(
                anchor,
                k=self.negatives_per_anchor,
                rng=self._rng,
            )
            if len(negs) > 0:
                triplets.append((anchor, negs))
        return triplets