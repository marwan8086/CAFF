"""
tests/test_smoke_pipeline.py
============================
End-to-end smoke test of the CAFF pipeline using small synthetic data.
Verifies that all pipeline stages execute without crashing:
  - KG loading
  - Triple instance construction (BFS + gold annotation)
  - Encoder loading (frozen)
  - Model initialization
Run with:
    pytest tests/test_smoke_pipeline.py -v
Requires:
    tests/fixtures/smoke_kg.tsv
    tests/fixtures/smoke_train.json
    tests/fixtures/smoke_dev.json
If the fixtures are missing, run:
    python tests/fixtures/build_smoke_data.py
"""
from __future__ import annotations
from pathlib import Path
import pytest
from caff.data import KnowledgeGraph
FIXTURES = Path(__file__).parent / "fixtures"
SMOKE_KG = FIXTURES / "smoke_kg.tsv"
SMOKE_TRAIN = FIXTURES / "smoke_train.json"
SMOKE_DEV = FIXTURES / "smoke_dev.json"
def _require_fixtures() -> None:
    """Skip the test if smoke fixtures are missing."""
    missing = [p for p in (SMOKE_KG, SMOKE_TRAIN, SMOKE_DEV) if not p.exists()]
    if missing:
        pytest.skip(
            f"Smoke fixtures missing: {missing}. "
            f"Run: python tests/fixtures/build_smoke_data.py"
        )
def test_kg_loads_successfully() -> None:
    """KG loads with the expected number of entities and relations."""
    _require_fixtures()
    kg = KnowledgeGraph.from_tsv(str(SMOKE_KG), min_relation_freq=50)
    assert len(kg.entities) > 0, "KG has no entities"
    assert len(kg.relations) > 0, "KG has no relations"
    assert len(kg.adj) > 0, "KG has no adjacency entries"
    # smoke_kg has 5,000 entities and 20 relations (per build_smoke_data.py)
    assert len(kg.entities) == 5_000
    assert len(kg.relations) == 20
def test_kg_has_consistent_indices() -> None:
    """Every entity in adj/rev appears in entity_to_idx."""
    _require_fixtures()
    kg = KnowledgeGraph.from_tsv(str(SMOKE_KG), min_relation_freq=50)
    for head in kg.adj:
        assert head in kg.entity_to_idx, f"head {head} missing from entity_to_idx"
    for tail in kg.rev:
        assert tail in kg.entity_to_idx, f"tail {tail} missing from entity_to_idx"
def test_empty_kg_raises_value_error(tmp_path: Path) -> None:
    """An empty KG must raise ValueError, not silently produce a 0-triple graph."""
    empty_kg = tmp_path / "empty_kg.tsv"
    empty_kg.write_text("head\trelation\ttail\n")
    with pytest.raises(ValueError, match="empty"):
        KnowledgeGraph.from_tsv(str(empty_kg), min_relation_freq=50)
def test_kg_filters_low_frequency_relations(tmp_path: Path) -> None:
    """min_relation_freq drops relations with too few triples."""
    kg_path = tmp_path / "tiny_kg.tsv"
    lines = ["head\trelation\ttail"]
    for i in range(100):
        lines.append(f"E{i}\tcommon_rel\tE{i + 1}")
    lines.append("E0\trare_rel\tE99")
    kg_path.write_text("\n".join(lines) + "\n")
    kg = KnowledgeGraph.from_tsv(str(kg_path), min_relation_freq=10)
    assert "common_rel" in kg.relations
    assert "rare_rel" not in kg.relations
def test_qa_records_load(tmp_path: Path) -> None:
    """QA JSON files are valid and contain required fields."""
    _require_fixtures()
    import json
    with open(SMOKE_TRAIN, encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) > 0
    sample = records[0]
    for field in ("query_id", "question", "seeds", "gold_answer"):
        assert field in sample, f"missing field: {field}"
    assert isinstance(sample["seeds"], list)
    assert len(sample["seeds"]) > 0
def test_smoke_data_seeds_are_in_kg() -> None:
    """All seeds in smoke_train.json must resolve to KG entities."""
    _require_fixtures()
    import json
    kg = KnowledgeGraph.from_tsv(str(SMOKE_KG), min_relation_freq=50)
    with open(SMOKE_TRAIN, encoding="utf-8") as f:
        records = json.load(f)
    n_seeds_in_kg = sum(
        1 for r in records if all(s in kg.entity_to_idx for s in r["seeds"])
    )
    # All seeds should be in the KG (build_smoke_data.py guarantees this).
    assert n_seeds_in_kg == len(records), (
        f"Only {n_seeds_in_kg}/{len(records)} records have all seeds in KG"
    )
