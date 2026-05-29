#!/usr/bin/env python
"""
theta_sensitivity.py -- Global threshold sensitivity analysis.

For a trained checkpoint, runs `evaluate.py` at a range of global
threshold values theta and reports how F1, precision, recall, and
per-hop precision change. The purpose is to localize the operating
point on the precision-recall trade-off and verify that the headline
theta (0.80 in the current default) is a sensible choice.

Usage
-----
    python scripts/theta_sensitivity.py \
        --checkpoint runs/no_dc/seed_42/best.pt \
        --thetas 0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90 \
        --mode autoregressive \
        --output-dir results/theta_sens

Output
------
- One JSON per theta (from evaluate.py) under <output-dir>/
- A summary JSON aggregating all thetas at <output-dir>/summary.json
- A pretty table printed to stdout
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Theta sensitivity sweep over a checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    p.add_argument("--thetas", required=True,
                   help="Comma-separated theta values, e.g. 0.50,0.60,0.70,0.80")
    p.add_argument("--mode", default="autoregressive",
                   choices=["teacher_forced", "autoregressive"])
    p.add_argument("--output-dir", required=True,
                   help="Directory to write per-theta JSON outputs and summary.")
    p.add_argument("--evaluate-script", default="evaluate.py",
                   help="Path to evaluate.py (default: ./evaluate.py)")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter (default: current).")
    return p.parse_args()


def run_one_theta(
    checkpoint: str,
    theta: float,
    mode: str,
    output_dir: Path,
    evaluate_script: str,
    python_bin: str,
) -> dict:
    """Run evaluate.py at a single theta and return the parsed metrics."""
    out_json = output_dir / f"theta_{theta:.2f}.json"
    cmd = [
        python_bin, evaluate_script,
        "--checkpoint", checkpoint,
        "--mode", mode,
        "--threshold", str(theta),
        "--output-json", str(out_json),
    ]
    logger.info(f"  theta={theta:.2f}  running evaluate.py ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  evaluate.py FAILED at theta={theta:.2f}")
        logger.error(f"  stderr: {result.stderr[-500:]}")
        return {"theta": theta, "error": result.stderr[-500:]}

    if not out_json.exists():
        logger.error(f"  expected output {out_json} not found")
        return {"theta": theta, "error": "no output JSON"}

    with out_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload.get("metrics", {})
    return {
        "theta": theta,
        "f1": metrics.get("f1"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "map": metrics.get("map"),
        "ndcg@10": metrics.get("ndcg@10"),
        "hop1_prec": metrics.get("hop1_prec"),
        "hop2_prec": metrics.get("hop2_prec"),
        "hop3_prec": metrics.get("hop3_prec"),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    thetas = sorted(float(x) for x in args.thetas.split(","))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info(f"Theta sensitivity sweep")
    logger.info(f"  checkpoint: {args.checkpoint}")
    logger.info(f"  thetas:     {thetas}")
    logger.info(f"  mode:       {args.mode}")
    logger.info(f"  output:     {output_dir}")
    logger.info("=" * 72)

    rows = []
    for theta in thetas:
        row = run_one_theta(
            args.checkpoint, theta, args.mode,
            output_dir, args.evaluate_script, args.python,
        )
        rows.append(row)

    # Save summary
    summary = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "thetas": thetas,
        "rows": rows,
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_path}")

    # Print pretty table
    print()
    print("=" * 96)
    print(f"Theta sensitivity ({args.mode} mode)  --  checkpoint: {args.checkpoint}")
    print("=" * 96)
    print(f"{'theta':>6} | {'F1':>7} | {'prec':>7} | {'recall':>7} | {'MAP':>7} | "
          f"{'NDCG@10':>7} | {'hop1':>7} | {'hop2':>7} | {'hop3':>7}")
    print("-" * 96)
    for row in rows:
        if "error" in row:
            print(f"{row['theta']:>6.2f} | ERROR: {row['error'][:80]}")
            continue
        print(f"{row['theta']:>6.2f} | "
              f"{row['f1']:>7.4f} | {row['precision']:>7.4f} | "
              f"{row['recall']:>7.4f} | {row['map']:>7.4f} | "
              f"{row['ndcg@10']:>7.4f} | {row['hop1_prec']:>7.4f} | "
              f"{row['hop2_prec']:>7.4f} | {row['hop3_prec']:>7.4f}")
    print("=" * 96)

    # Find peak F1
    valid_rows = [r for r in rows if "f1" in r and r["f1"] is not None]
    if valid_rows:
        best = max(valid_rows, key=lambda r: r["f1"])
        print(f"\nPeak F1 = {best['f1']:.4f} at theta = {best['theta']:.2f}")


if __name__ == "__main__":
    main()
