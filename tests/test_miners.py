"""
tests/test_miners.py
====================
Verify HC3 mining (paper Â§6.4):

  â€¢ Buffer capacity = 1000                       (paper Â§6.4)
  â€¢ Refresh policy = every 500 steps             (paper Â§6.4)
  â€¢ Negatives drawn from same (Q,r,â„“) but opposite label
  â€¢ k = 8 negatives per anchor                   (paper Â§6.4)

This guards Failure Mode F4 (mining in-batch instead of from buffer).
"""

from __future__ import annotations

import random

import torch

from caff.miners import HC3Buffer, HC3Miner, TrainingInstance


def _make_inst(qid: str, rel: str, hop: int, label: int) -> TrainingInstance:
    return TrainingInstance(
        query_id=qid, question="dummy", head="h", relation=rel, tail="t",
        hop=hop, label=label, z_prev=torch.zeros(8),
    )


def test_buffer_capacity():
    buf = HC3Buffer(capacity=5)
    for i in range(10):
        buf.add(_make_inst(f"q{i}", "r1", 2, i % 2))
    assert len(buf) == 5
    # Oldest (q0..q4) should have been evicted
    qids_in_buffer = {inst.query_id for inst in buf._buffer}
    assert "q0" not in qids_in_buffer
    assert "q9" in qids_in_buffer


def test_negative_mining_same_anchor_different_label():
    buf = HC3Buffer(capacity=100)
    # Same (Q, r, â„“) but opposite labels
    pos = _make_inst("q1", "r1", 2, label=1)
    neg1 = _make_inst("q1", "r1", 2, label=0)
    neg2 = _make_inst("q1", "r1", 2, label=0)
    # Different anchor, should NOT be selected
    other = _make_inst("q2", "r2", 2, label=0)

    buf.add_batch([pos, neg1, neg2, other])
    rng = random.Random(42)
    negs = buf.get_negatives_for(pos, k=8, rng=rng)
    assert len(negs) == 2
    assert all(n.label == 0 for n in negs)
    assert all(n.relation == "r1" and n.hop == 2 for n in negs)


def test_negative_mining_respects_k():
    buf = HC3Buffer(capacity=100)
    pos = _make_inst("q1", "r1", 2, label=1)
    buf.add(pos)
    # Add 20 negatives â€” miner must return only 8
    for i in range(20):
        buf.add(_make_inst("q1", "r1", 2, label=0))

    rng = random.Random(42)
    negs = buf.get_negatives_for(pos, k=8, rng=rng)
    assert len(negs) == 8


def test_miner_refresh_schedule():
    """Paper Â§6.4: refresh every 500 gradient steps."""
    miner = HC3Miner(buffer_capacity=10, refresh_every=500)
    for _ in range(499):
        miner.step()
    assert not miner.is_refresh_step()
    miner.step()              # step 500
    assert miner.is_refresh_step()


def test_miner_paper_invariants():
    """k = 8 and capacity = 1000 are paper Â§6.4 defaults."""
    miner = HC3Miner()
    assert miner.negatives_per_anchor == 8
    assert miner.buffer.capacity == 1000
    assert miner.refresh_every == 500




# ─── DCMiner tests (Phase 2) ────────────────────────────────────


def test_dcminer_invalid_L_raises():
    """L < 2 must raise — no wrong hop exists for L=1."""
    import pytest
    from caff.miners import DCMiner
    with pytest.raises(ValueError, match="L >= 2"):
        DCMiner(L=1)
    with pytest.raises(ValueError, match="L >= 2"):
        DCMiner(L=0)


def test_dcminer_sample_wrong_hop_excludes_gold():
    """Sampled wrong hop is always != gold_hop and within [1, L]."""
    from caff.miners import DCMiner
    miner = DCMiner(L=3, seed=42)
    for _ in range(50):
        for gold in (1, 2, 3):
            w = miner.sample_wrong_hop(gold)
            assert 1 <= w <= 3
            assert w != gold


def test_dcminer_invalid_gold_hop_raises():
    """gold_hop outside [1, L] must raise."""
    import pytest
    from caff.miners import DCMiner
    miner = DCMiner(L=3, seed=42)
    with pytest.raises(ValueError, match="gold_hop must be in"):
        miner.sample_wrong_hop(0)
    with pytest.raises(ValueError, match="gold_hop must be in"):
        miner.sample_wrong_hop(4)


def test_dcminer_reproducible_with_seed():
    """Two miners with same seed produce identical sequences."""
    from caff.miners import DCMiner
    a = DCMiner(L=3, seed=123)
    b = DCMiner(L=3, seed=123)
    seq_a = [a.sample_wrong_hop(2) for _ in range(20)]
    seq_b = [b.sample_wrong_hop(2) for _ in range(20)]
    assert seq_a == seq_b
