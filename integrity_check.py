"""
integrity_check.py -- Verify that today's work is intact and consistent.

Run from the CAFF root:
    python integrity_check.py

Checks that the critical files exist, have expected line counts, and
that the trained checkpoints from today's experiments are present.
"""
import hashlib
from pathlib import Path


def md5_short(path: Path) -> str:
    h = hashlib.md5(path.read_bytes()).hexdigest()
    return h[:12]


def check_file(label: str, path: str, expected_lines: int | None = None,
               must_contain: list[str] | None = None) -> bool:
    p = Path(path)
    if not p.exists():
        print(f"  [MISS] {label}: {path}")
        return False
    text = p.read_text(encoding="utf-8")
    lines = text.count("\n") + (0 if text.endswith("\n") else 1)
    h = md5_short(p)
    line_ok = (expected_lines is None) or (lines == expected_lines)
    contain_ok = True
    missing = []
    if must_contain:
        for s in must_contain:
            if s not in text:
                contain_ok = False
                missing.append(s)
    status = "OK  " if (line_ok and contain_ok) else "WARN"
    print(f"  [{status}] {label}: {lines} lines, md5={h}, path={path}")
    if not line_ok:
        print(f"         expected {expected_lines} lines, got {lines}")
    if missing:
        print(f"         missing: {missing}")
    return line_ok and contain_ok


def check_checkpoint(label: str, path: str) -> bool:
    p = Path(path)
    if not p.exists():
        print(f"  [MISS] {label}: {path}")
        return False
    size_mb = p.stat().st_size / (1024 * 1024)
    print(f"  [OK  ] {label}: {size_mb:.1f} MB at {path}")
    return True


def main() -> None:
    print("=" * 70)
    print("       Integrity check for today's work (May 29, 2026)")
    print("=" * 70)

    print()
    print("Critical files:")
    print("-" * 50)
    all_ok = True
    all_ok &= check_file(
        "README.md (v2, reframed)", "README.md",
        expected_lines=589,
        must_contain=["0.5764", "CAFF is a triple filter", "Scope and Future Work"],
    )
    all_ok &= check_file(
        "caff_full.yaml (DEPRECATED)", "configs/caff_full.yaml",
        must_contain=["DEPRECATED", "d: 768"],
    )
    all_ok &= check_file(
        "no_dc.yaml (current default)", "configs/no_dc.yaml",
        must_contain=["use_dc: false"],
    )
    all_ok &= check_file(
        "theta_sensitivity.py", "scripts/theta_sensitivity.py",
        must_contain=["def run_one_theta"],
    )
    all_ok &= check_file(
        "per_relation_f1.py", "scripts/per_relation_f1.py",
        must_contain=["sys.path.insert", "from caff import"],
    )

    print()
    print("Checkpoints (no_dc, 3 seeds):")
    print("-" * 50)
    for seed in [42, 1337, 2024]:
        all_ok &= check_checkpoint(
            f"no_dc seed {seed}", f"runs/no_dc/seed_{seed}/best.pt"
        )

    print()
    print("Results from today:")
    print("-" * 50)
    for seed in [42, 1337, 2024]:
        check_file(
            f"per_relation seed {seed}",
            f"results/per_relation_seed{seed}.json",
            must_contain=['"is_a"', '"has_phenotype"'],
        )
    for seed in [42, 1337, 2024]:
        check_file(
            f"theta_sens summary seed {seed}",
            f"results/theta_sens_seed{seed}/summary.json",
            must_contain=['"thetas"', '"f1"'],
        )

    print()
    print("=" * 70)
    if all_ok:
        print("  ALL CRITICAL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED -- review above")
    print("=" * 70)


if __name__ == "__main__":
    main()
