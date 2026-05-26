"""
scripts/apply_hc3_fix.py - Apply the HC3 cross-query contrastive fix
automatically and safely.

It edits only pure-ASCII code lines (robust to the non-ASCII glyphs in
the surrounding docstrings), backs up the originals first, and verifies
the result parses.

Changes:
  1. caff/miners.py  _rebuild_index:
       key = TripleKey(inst.query_id, inst.relation, inst.hop)
     -> key = TripleKey("", inst.relation, inst.hop)
  2. caff/miners.py  get_negatives_for:
       key = TripleKey(anchor.query_id, anchor.relation, anchor.hop)
     -> key = TripleKey("", anchor.relation, anchor.hop)
     and the negative-filter list comprehension gains:
       and self._buffer[pos].query_id != anchor.query_id
       and self._buffer[pos].hop >= 2
  3. caff/trainer.py  _score_hc3_instance:
       scorer.score_candidates(  ->  scorer.score_logits(

Run:
    python scripts/apply_hc3_fix.py            # apply
    python scripts/apply_hc3_fix.py --revert   # restore from backup
"""
from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

MINERS = Path("caff/miners.py")
TRAINER = Path("caff/trainer.py")
MINERS_BAK = Path("caff/miners_backup_orig.py")
TRAINER_BAK = Path("caff/trainer_backup_orig.py")


def revert():
    ok = True
    for bak, target in [(MINERS_BAK, MINERS), (TRAINER_BAK, TRAINER)]:
        if bak.exists():
            shutil.copy(bak, target)
            print(f"Restored {target} from {bak}")
        else:
            print(f"WARNING: backup {bak} not found")
            ok = False
    return 0 if ok else 1


def apply():
    miners = MINERS.read_text(encoding="utf-8")
    trainer = TRAINER.read_text(encoding="utf-8")

    # Backups (only once - do not overwrite an existing original backup)
    if not MINERS_BAK.exists():
        MINERS_BAK.write_text(miners, encoding="utf-8")
        print(f"Backed up {MINERS} -> {MINERS_BAK}")
    if not TRAINER_BAK.exists():
        TRAINER_BAK.write_text(trainer, encoding="utf-8")
        print(f"Backed up {TRAINER} -> {TRAINER_BAK}")

    changes = 0

    # --- Edit 1: _rebuild_index key ---
    old1 = "            key = TripleKey(inst.query_id, inst.relation, inst.hop)"
    new1 = '            key = TripleKey("", inst.relation, inst.hop)'
    if old1 in miners:
        miners = miners.replace(old1, new1)
        changes += 1
        print("Edit 1 applied: _rebuild_index uses (relation, hop) key")
    elif new1 in miners:
        print("Edit 1 already applied")
    else:
        print("ERROR: Edit 1 anchor not found")
        return 1

    # --- Edit 2: get_negatives_for key + filter ---
    old2 = (
        "        key = TripleKey(anchor.query_id, anchor.relation, anchor.hop)\n"
        "        candidate_positions = self._index.get(key, [])\n"
        "        # Filter to those with label = 0\n"
        "        negatives = [\n"
        "            self._buffer[pos]\n"
        "            for pos in candidate_positions\n"
        "            if self._buffer[pos].label == 0\n"
        "        ]"
    )
    new2 = (
        '        key = TripleKey("", anchor.relation, anchor.hop)\n'
        "        candidate_positions = self._index.get(key, [])\n"
        "        # Cross-query fix: keep label=0 negatives from a DIFFERENT\n"
        "        # query (so z_prev differs) at hop >= 2 (z is zero at hop 1).\n"
        "        negatives = [\n"
        "            self._buffer[pos]\n"
        "            for pos in candidate_positions\n"
        "            if self._buffer[pos].label == 0\n"
        "            and self._buffer[pos].query_id != anchor.query_id\n"
        "            and self._buffer[pos].hop >= 2\n"
        "        ]"
    )
    if old2 in miners:
        miners = miners.replace(old2, new2)
        changes += 1
        print("Edit 2 applied: get_negatives_for is cross-query, hop>=2")
    elif 'key = TripleKey("", anchor.relation, anchor.hop)' in miners:
        print("Edit 2 already applied")
    else:
        print("ERROR: Edit 2 anchor not found (get_negatives_for body)")
        return 1

    # --- Edit 3: trainer _score_hc3_instance -> score_logits ---
    old3 = (
        "        score = scorer.score_candidates(\n"
        "            W_ctx, self.model.v, q_embedding, E_r,\n"
        "        )  # (1,)"
    )
    new3 = (
        "        score = scorer.score_logits(\n"
        "            W_ctx, self.model.v, q_embedding, E_r,\n"
        "        )  # (1,) pre-sigmoid: avoids saturation, stronger HC3 gradient"
    )
    if old3 in trainer:
        trainer = trainer.replace(old3, new3)
        changes += 1
        print("Edit 3 applied: _score_hc3_instance uses score_logits")
    elif "score = scorer.score_logits(" in trainer:
        print("Edit 3 already applied")
    else:
        print("ERROR: Edit 3 anchor not found (_score_hc3_instance)")
        return 1

    # Verify syntax before writing. Strip a leading BOM (U+FEFF) for the
    # parse check only; the file content itself is written unchanged.
    try:
        ast.parse(miners.lstrip("\ufeff"))
        ast.parse(trainer.lstrip("\ufeff"))
    except SyntaxError as e:
        print(f"ERROR: resulting file has a syntax error: {e}")
        return 1

    # Note: we do NOT guard against em-dashes here. The original source
    # files legitimately contain em-dashes in their docstrings, and our
    # three edits are pure ASCII, so they cannot introduce new ones.

    MINERS.write_text(miners, encoding="utf-8")
    TRAINER.write_text(trainer, encoding="utf-8")
    print(f"\nDone. {changes} edits written. Syntax verified.")
    print("Revert any time with:  python scripts/apply_hc3_fix.py --revert")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    return revert() if args.revert else apply()


if __name__ == "__main__":
    sys.exit(main())
