"""Compare two results caches, record by record (issue #6).

The cold run of #6 rebuilds `paper/data` from an empty `outputs/` in a scratch
directory; this script says whether that cache is the committed one. Records
are matched in order, every non-float leaf (kind, n, mode, field, ...) must
agree exactly, and each float leaf gives |a - b| / max(|a|, |b|). A cold run
reproduces the committed cache exactly where the arithmetic is serial, and to
round-off (~1e-10) where a threaded reference solve reorders a sum.

    uv run python scripts/compare_caches.py paper/data <scratch dir>
    uv run python scripts/compare_caches.py paper/data <dir> --tol 1e-12

Prints the worst record of each file and every record above --tol; exits 1 on
a difference above --tol, a non-float mismatch, or a file on one side only.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Provenance of the run, not results: a rerun differs here by design.
SKIP_META = {"date", "git_sha"}
# Arguments that name where a run wrote, or how many processes it used.
SKIP_ARGS = {"data_dir", "out_dir", "workers", "out", "style", "format"}
# The fields that identify a record, for the report's labels.
LABEL_KEYS = (
    "kind",
    "n",
    "delta",
    "mode",
    "field",
    "t",
    "group",
    "width_label",
    "variant",
)


def leaves(obj, prefix: str = ""):
    """Every scalar in a nested record, as (dotted path, value)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from leaves(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from leaves(value, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def label(record: dict) -> str:
    return " ".join(f"{k}={record[k]}" for k in LABEL_KEYS if k in record)


def compare_record(a: dict, b: dict) -> tuple[float, float, str, list[str]]:
    """(worst relative difference, its absolute difference, its leaf, mismatches)."""
    la, lb = dict(leaves(a)), dict(leaves(b))
    bad = sorted(set(la) ^ set(lb))
    worst = (0.0, 0.0, "")
    for key in sorted(set(la) & set(lb)):
        x, y = la[key], lb[key]
        if not isinstance(x, float) and not isinstance(y, float):
            if x != y:
                bad.append(key)
            continue
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            bad.append(key)  # a float on one side, a string or None on the other
            continue
        if math.isnan(x) and math.isnan(y):
            continue
        diff = abs(x - y)
        scale = max(abs(x), abs(y))
        rel = diff / scale if scale else 0.0
        if rel > worst[0]:
            worst = (rel, diff, key)
    return (*worst, bad)


def compare_file(ref: Path, new: Path, tol: float) -> tuple[list[str], list[str]]:
    """(report lines, findings) for one pair of cache files."""
    a, b = json.loads(ref.read_text()), json.loads(new.read_text())
    findings = []
    meta = [
        k
        for k in sorted(set(a) | set(b))
        if k not in SKIP_META | {"records", "args"} and a.get(k) != b.get(k)
    ]
    args = [
        k
        for k in sorted(set(a["args"]) | set(b["args"]))
        if k not in SKIP_ARGS and a["args"].get(k) != b["args"].get(k)
    ]
    if meta:
        findings.append(f"{ref.name}: metadata differs: {meta}")
    if args:
        findings.append(f"{ref.name}: arguments differ: {args}")
    if len(a["records"]) != len(b["records"]):
        findings.append(
            f"{ref.name}: {len(a['records'])} records vs {len(b['records'])}"
        )
    rows = []
    for i, (x, y) in enumerate(zip(a["records"], b["records"], strict=False)):
        rel, diff, leaf, bad = compare_record(x, y)
        rows.append((rel, diff, leaf, bad, i, label(x)))
        if bad:
            findings.append(f"{ref.name} #{i} {label(x)}: fields differ: {bad[:6]}")
        elif rel > tol:
            findings.append(
                f"{ref.name} #{i} {label(x)}: rel {rel:.2e} abs {diff:.2e} .{leaf}"
            )
    worst = max(rows, key=lambda row: row[0]) if rows else None
    lines = [
        f"{ref.name}: {len(a['records'])} records, worst rel "
        f"{worst[0]:.2e} (abs {worst[1]:.2e}) at #{worst[4]} {worst[5]} .{worst[2]}"
        if worst
        else f"{ref.name}: no records"
    ]
    lines += [f"    {f}" for f in findings if f.startswith(f"{ref.name} #")]
    return lines, findings


def compare_dirs(ref: Path, new: Path, tol: float) -> tuple[list[str], list[str]]:
    """(report lines, findings) for two directories of cache files."""
    lines, findings = [], []
    ref_files = {p.name for p in ref.glob("*.json")}
    new_files = {p.name for p in new.glob("*.json")}
    for name in sorted(ref_files ^ new_files):
        side = "reference" if name in ref_files else "candidate"
        findings.append(f"{name}: only in the {side} cache")
        lines.append(f"{name}: ONLY IN THE {side.upper()} CACHE")
    for name in sorted(ref_files & new_files):
        file_lines, file_findings = compare_file(ref / name, new / name, tol)
        lines += file_lines
        findings += file_findings
    return lines, findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("ref", type=Path, help="the reference cache (paper/data)")
    parser.add_argument("new", type=Path, help="the cache to check")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-9,
        help="relative difference allowed per leaf (default 1e-9, round-off)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lines, findings = compare_dirs(args.ref, args.new, args.tol)
    print("\n".join(lines))
    print(f"{len(findings)} findings above rel {args.tol:g}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
