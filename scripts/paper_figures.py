"""Regenerate every figure and table fragment the manuscript uses (issue #54).

House rule: ``paper/figures/`` comes from committed scripts and is never
hand-edited. The convergence figures, the two seed-basis figures and every
table fragment come from the results cache in ``paper/data/`` (written by
the stiff drivers with ``--data-dir paper/data``) and take seconds. The
stills and the spectra need simulation state the cache does not hold, so
``--all`` also runs the drivers in print style against the reference and
operator caches under ``outputs/`` (minutes) and copies their PDFs in.
Print figures pin ``SOURCE_DATE_EPOCH``, so a regenerated PDF is
byte-identical to the committed one; ``--check`` regenerates the cached
set into a temporary directory and fails on any difference, the gate the
assembly pass (#61) can script.

    uv run python scripts/paper_figures.py            # from paper/data, ~10 s
    uv run python scripts/paper_figures.py --all      # + stills and spectra, ~50 min
    uv run python scripts/paper_figures.py --check    # byte identity with paper/figures
"""

import argparse
import filecmp
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rbf_hyperbolic_interfaces.plotting import use_print_style
from rbf_hyperbolic_interfaces.results_cache import ResultsCache
from rbf_hyperbolic_interfaces.stiff_figures import (
    comparators_1d,
    convergence_1d,
    convergence_2d,
    seeds_1d,
    seeds_2d,
)
from rbf_hyperbolic_interfaces.stiff_tables import (
    table_1d,
    table_1d_comparators,
    table_2d,
    table_spectra,
    table_truncation,
    write_fragment,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data"
FIGURES = ROOT / "paper" / "figures"
OUTPUTS = ROOT / "outputs"

# Figure -> (cache file, drawer). The drawer takes (cache, out, print_mode=True).
CONVERGENCE = {
    "wave1d_stiff_convergence.pdf": ("wave1d_stiff.json", convergence_1d),
    "wave1d_stiff_comparators.pdf": ("wave1d_stiff.json", comparators_1d),
    "wave2d_stiff_convergence.pdf": ("wave2d_stiff.json", convergence_2d),
    "wave2d_stiff_oblique_convergence.pdf": ("wave2d_stiff_d12.json", convergence_2d),
    "wave2d_stiff_convergence_curved.pdf": ("wave2d_stiff_a0.02.json", convergence_2d),
}

# Fragment -> (cache file, writer). The writer takes (cache, source).
TABLES = {
    "tab_1d_knee.tex": ("wave1d_stiff.json", table_1d),
    "tab_1d_comparators.tex": ("wave1d_stiff.json", table_1d_comparators),
    "tab_2d_flat_v.tex": ("wave2d_stiff.json", lambda c, s: table_2d(c, "v", s)),
    "tab_2d_flat_h.tex": ("wave2d_stiff.json", lambda c, s: table_2d(c, "h", s)),
    "tab_2d_flat_cmp_v.tex": (
        "wave2d_stiff_cmp.json",
        lambda c, s: table_2d(c, "v", s),
    ),
    "tab_2d_curved_cmp_v.tex": (
        "wave2d_stiff_cmp_a0.02.json",
        lambda c, s: table_2d(c, "v", s),
    ),
    "tab_2d_flat_u.tex": (
        "wave2d_stiff.json",
        lambda c, s: table_2d(c, "max_u", s, rates=False),
    ),
    "tab_2d_oblique_v.tex": ("wave2d_stiff_d12.json", lambda c, s: table_2d(c, "v", s)),
    "tab_2d_oblique_u.tex": ("wave2d_stiff_d12.json", lambda c, s: table_2d(c, "u", s)),
    "tab_2d_oblique_h.tex": ("wave2d_stiff_d12.json", lambda c, s: table_2d(c, "h", s)),
    "tab_2d_curved_v.tex": (
        "wave2d_stiff_a0.02.json",
        lambda c, s: table_2d(c, "v", s),
    ),
    "tab_2d_curved_u.tex": (
        "wave2d_stiff_a0.02.json",
        lambda c, s: table_2d(c, "u", s),
    ),
    "tab_2d_curved_h.tex": (
        "wave2d_stiff_a0.02.json",
        lambda c, s: table_2d(c, "h", s),
    ),
    "tab_2d_curved_truncation.tex": ("wave2d_stiff_a0.02.json", table_truncation),
    "tab_2d_curved_trim_v.tex": (
        "wave2d_stiff_a0.02_r0.001.json",
        lambda c, s: table_2d(c, "v", s),
    ),
    "tab_spectra_n2500.tex": ("wave2d_stiff_eigenvalues_n2500.json", table_spectra),
    "tab_spectra_n2500_curved.tex": (
        "wave2d_stiff_eigenvalues_n2500_a0.02.json",
        table_spectra,
    ),
    "tab_spectra_n900_variants.tex": (
        "wave2d_stiff_eigenvalues_n900_variants.json",
        table_spectra,
    ),
}

# --all: driver invocations (print style, PDF) and the outputs/ file each
# leaves behind -> its name under paper/figures/.
PRINT = ["--style", "print", "--format", "pdf"]
DRIVER_RUNS = [
    (
        ["scripts/wave1d_stiff.py"],
        {"wave1d_stiff_snapshot.pdf": "wave1d_stiff_snapshot.pdf"},
    ),
    (
        ["scripts/wave2d_stiff.py", "--snapshot-only", "--snapshot-times", "1.0"],
        {"wave2d_stiff_snapshot.pdf": "wave2d_stiff_snapshot.pdf"},
    ),
    (
        [
            "scripts/wave2d_stiff.py",
            "--snapshot-only",
            "--snapshot-times",
            "1.0",
            "--direction",
            "1",
            "2",
            "--widths",  # the driver refuses a jump at oblique incidence
            "0.0025",
            "0.01",
        ],
        {"wave2d_stiff_snapshot_d12.pdf": "wave2d_stiff_oblique_snapshot.pdf"},
    ),
    (
        [
            "scripts/wave2d_stiff.py",
            "--snapshot-only",
            "--snapshot-times",
            "1.0",
            "--amplitude",
            "0.02",
            "--snapshot-width",
            "0.005",
        ],
        {"wave2d_stiff_snapshot_a0.02.pdf": "wave2d_stiff_snapshot_curved.pdf"},
    ),
    (
        ["scripts/wave2d_stiff_eigenvalues.py", "--n", "2500"],
        {"wave2d_stiff_eigenvalues_n2500.pdf": "wave2d_stiff_eigenvalues_n2500.pdf"},
    ),
    (
        ["scripts/wave2d_stiff_eigenvalues.py", "--n", "2500", "--amplitude", "0.02"],
        {
            "wave2d_stiff_eigenvalues_n2500_a0.02.pdf": (
                "wave2d_stiff_eigenvalues_n2500_curved.pdf"
            )
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--all",
        action="store_true",
        help="also the stills and spectra, through the drivers (needs outputs/)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate the cached set elsewhere and compare bytes with paper/figures",
    )
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    return parser.parse_args()


def from_cache(out_dir: Path) -> list[Path]:
    """Everything the cache alone can produce; missing caches are reported
    and skipped so the script works while a run is still pending."""
    written = []
    for fig_name, (name, draw) in CONVERGENCE.items():
        path = DATA / name
        if not path.exists():
            print(f"  skip {fig_name}: no {path.relative_to(ROOT)}")
            continue
        written.append(
            draw(ResultsCache.read(path), out_dir / fig_name, print_mode=True)
        )
    written.append(seeds_1d(out_dir / "wave1d_stiff_seeds.pdf", print_mode=True))
    written.append(seeds_2d(out_dir / "wave2d_stiff_seeds.pdf", print_mode=True))
    for frag, (name, writer) in TABLES.items():
        path = DATA / name
        if not path.exists():
            print(f"  skip {frag}: no {path.relative_to(ROOT)}")
            continue
        source = str(path.relative_to(ROOT))
        written.append(
            write_fragment(out_dir / frag, writer(ResultsCache.read(path), source))
        )
    return written


def through_drivers(out_dir: Path) -> list[Path]:
    """The stills and spectra: run each driver in print style, copy its PDFs."""
    written = []
    for cmd, copies in DRIVER_RUNS:
        t0 = time.perf_counter()
        print(f"  {' '.join(cmd)}")
        subprocess.run([sys.executable, *cmd, *PRINT], cwd=ROOT, check=True)
        for src, dst in copies.items():
            shutil.copyfile(OUTPUTS / src, out_dir / dst)
            written.append(out_dir / dst)
        print(f"    ({time.perf_counter() - t0:.0f}s)")
    return written


def check(out_dir: Path) -> int:
    """Byte identity of the cached set against ``out_dir``; 1 on any difference."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh = from_cache(Path(tmp))
        bad = []
        for path in fresh:
            committed = out_dir / path.name
            if not committed.exists():
                bad.append(f"missing {committed.relative_to(ROOT)}")
            elif not filecmp.cmp(path, committed, shallow=False):
                bad.append(f"differs {committed.relative_to(ROOT)}")
    for line in bad:
        print("  " + line)
    print(f"{len(fresh) - len(bad)} of {len(fresh)} identical")
    return 1 if bad else 0


def main() -> None:
    args = parse_args()
    use_print_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.check:
        raise SystemExit(check(args.out_dir))
    t0 = time.perf_counter()
    written = from_cache(args.out_dir)
    print(f"from paper/data: {len(written)} files in {time.perf_counter() - t0:.0f}s")
    if args.all:
        t0 = time.perf_counter()
        written += through_drivers(args.out_dir)
        print(f"through the drivers: {time.perf_counter() - t0:.0f}s")
    for path in written:
        print("  " + str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))


if __name__ == "__main__":
    main()
