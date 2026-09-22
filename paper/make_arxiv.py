#!/usr/bin/env python3
"""Assemble and verify a self-contained arXiv upload for the manuscript (#62).

arXiv builds from the uploaded source tree and does **not** run BibTeX, so the
upload carries the built ``main.bbl``. From ``paper/`` this script stages

* ``main.tex`` and ``references.bib`` with their whole-line comments removed:
  the ``% TRACE`` and ``% NOTATION`` blocks and the dated ``VERIFIED`` notes
  are load-bearing in the repository, where the ledger and the README point
  at them, and are not part of the paper (arXiv republishes source verbatim);
* ``main.bbl``;
* every figure an ``\\includegraphics`` names and every ``figures/tab_*.tex``
  fragment an ``\\input`` names, in the same ``figures/`` layout;

into a clean ``arxiv/`` directory and writes ``arxiv.tar.gz``.

Nothing about the transform is trusted. Before staging, the repository's own
checks run (``scripts/paper_numbers.py``: every cache-backed number the text
quotes; ``scripts/paper_figures.py --check``: byte identity of the figures and
table fragments with what the cache regenerates). After staging, draft
markers are refused, the staged tree is rebuilt in a scratch copy and its
text compared with the committed ``main.pdf`` (comment stripping is exactly
the transform that fails silently: a ``%`` ending a line eats the newline,
``\\%`` is a literal percent), and the accented bibliography labels
(``[Pó22]``, ``[Mü73]``, UTF-8 in the ``.bbl`` since #61) are checked to
render. Any failure stops the packaging.

    tectonic --keep-intermediates main.tex   # produces main.bbl
    uv run python make_arxiv.py

Options: ``--keep-comments`` stages the sources verbatim; ``--no-verify``
skips every gate (not recommended).

Adapted from the same author's weil-positivity-lab packaging (the comment
stripping and the rebuild gate) and bolza-bending / dirichlet-bridge (the
staging).
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent  # paper/
ROOT = HERE.parent
OUT = HERE / "arxiv"
FIG_DIR = HERE / "figures"

# Environments in which '%' is content rather than a comment. None are used;
# the stripper refuses rather than guess if one ever appears.
VERBATIM_ENVS = ("verbatim", "Verbatim", "lstlisting", "alltt", "minted")
# Draft markers that must not ship (the scaffold's \stub macro, #52).
DRAFT_MARKERS = ("\\stub", "\\todo", "[Stub,")
COMMENT_LINE = re.compile(r"^\s*%")
INCLUDEGRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
INPUT = re.compile(r"\\input\{([^}]+)\}")
BIBLABEL = re.compile(r"\\bibitem\[([^\]]*)\]")
DATE = re.compile(r"^\\date\{([^}]*)\}", re.MULTILINE)
GRAPHIC_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")
# Code points pdfTeX's utf8 input encoding maps without extra packages
# (Latin-1 Supplement, Latin Extended-A, general punctuation). arXiv builds
# with pdfTeX; anything else (Greek letters in text, math symbols) would fail
# there while tectonic's XeTeX, which this script verifies with, accepts it.
PDFTEX_SAFE = [(0x00A0, 0x017F), (0x2010, 0x2027)]
# The repository's own gates, run from the root before anything is staged.
REPO_CHECKS = (
    ["scripts/paper_numbers.py", "--quiet"],
    ["scripts/paper_figures.py", "--check"],
)


def strip_whole_line_comments(tex: str) -> tuple[str, int]:
    """Drop lines whose first non-space character is ``%``.

    Inline comments stay: removing the tail of a line can change whether its
    newline is suppressed, which is a spacing change in TeX. Whole comment
    lines cannot alter tokenization, since TeX already discards them together
    with the newline that follows.
    """
    for env in VERBATIM_ENVS:
        if f"\\begin{{{env}}}" in tex:
            raise SystemExit(
                f"refusing to strip comments: the source contains a {env!r}"
                " environment, where '%' is content. Re-run with"
                " --keep-comments, or teach this script to skip verbatim spans."
            )
    lines = tex.splitlines(keepends=True)
    kept = [ln for ln in lines if not COMMENT_LINE.match(ln)]
    return "".join(kept), len(lines) - len(kept)


def referenced_figures(tex: str, fig_dir: Path) -> list[Path]:
    """Resolve every ``\\includegraphics{...}`` to a file in ``fig_dir``.

    The manuscript writes figure names without an extension and lets the
    graphics package supply one, so a bare lookup finds nothing. Names that
    carry a suffix are taken as they are; an unresolvable name is returned
    bare so the caller reports it as missing.
    """
    resolved: dict[str, Path] = {}
    for raw in INCLUDEGRAPHICS.findall(tex):
        name = Path(raw).name
        if Path(name).suffix:
            resolved[name] = fig_dir / name
            continue
        for ext in GRAPHIC_EXTS:
            candidate = fig_dir / f"{name}{ext}"
            if candidate.exists():
                resolved[name] = candidate
                break
        else:
            resolved[name] = fig_dir / name
    return [resolved[k] for k in sorted(resolved)]


def referenced_inputs(tex: str) -> list[str]:
    """Paths of every ``\\input{...}``, relative to ``paper/``, ``.tex`` added."""
    paths = set()
    for raw in INPUT.findall(tex):
        paths.add(raw if Path(raw).suffix else raw + ".tex")
    return sorted(paths)


def non_ascii_labels(bbl: str) -> list[str]:
    """Bibliography labels with a non-ASCII character, as the PDF prints them."""
    return [lab for lab in BIBLABEL.findall(bbl) if not lab.isascii()]


def unsafe_for_pdftex(texts: dict[str, str]) -> list[str]:
    """Non-ASCII characters outside the ranges pdfTeX reads without help."""
    bad = []
    for name, text in texts.items():
        for ch in sorted({c for c in text if not c.isascii()}):
            if not any(lo <= ord(ch) <= hi for lo, hi in PDFTEX_SAFE):
                bad.append(f"{name}: {ch!r} (U+{ord(ch):04X})")
    return bad


def date_of(tex: str) -> str | None:
    m = DATE.search(tex)
    return m.group(1) if m else None


def stage_text(src: Path, dst: Path, strip: bool) -> int:
    """Copy a TeX source, stripped of whole-line comments unless told not to."""
    text = src.read_text(encoding="utf-8")
    removed = 0
    if strip:
        text, removed = strip_whole_line_comments(text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return removed


def pdf_text(pdf: Path) -> str:
    if not shutil.which("pdftotext"):
        raise SystemExit(
            "pdftotext not found, so the staged source cannot be verified"
            " against main.pdf. Install poppler (`brew install poppler`) or"
            " re-run with --no-verify and check the build by hand."
        )
    proc = subprocess.run(
        ["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True
    )
    return proc.stdout


def run_repo_checks() -> None:
    for cmd in REPO_CHECKS:
        proc = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        if proc.returncode != 0:
            raise SystemExit(f"{' '.join(cmd)} failed; refusing to package.")
        print(f"  ok: {' '.join(cmd)}", flush=True)


def verify_rebuild(canonical_pdf: Path, labels: list[str]) -> None:
    """Build the staged tree in a scratch copy; fail unless its text matches.

    A copy, so ``arxiv/`` never holds build artifacts and the tarball cannot
    carry a stale ``main.pdf``.
    """
    if not shutil.which("tectonic"):
        raise SystemExit(
            "tectonic not found, so the staged source cannot be rebuilt."
            " Install it, or re-run with --no-verify and check by hand."
        )
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "build"
        shutil.copytree(OUT, scratch)
        proc = subprocess.run(
            ["tectonic", "main.tex"], cwd=scratch, capture_output=True, text=True
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit("the staged source does not build; refusing to package.")
        built = scratch / "main.pdf"
        if not built.exists():
            raise SystemExit("the staged build produced no main.pdf.")
        got, want = pdf_text(built), pdf_text(canonical_pdf)
        if got != want:
            diff = difflib.unified_diff(
                want.splitlines(), got.splitlines(), "main.pdf", "staged", lineterm=""
            )
            sys.stderr.write("\n".join(list(diff)[:40]) + "\n")
            raise SystemExit(
                "the staged source builds, but its text differs from the"
                f" committed {canonical_pdf.name} (first differences above)."
                " The packaging transform changed the document; refusing to"
                " package. Re-run with --keep-comments to isolate whether"
                " comment stripping is the cause."
            )
        missing = [lab for lab in labels if f"[{lab}]" not in got]
        if missing:
            raise SystemExit(
                "accented bibliography labels do not render in the staged"
                " build: " + ", ".join(missing)
            )
    print("  verified: the staged source rebuilds to the text of main.pdf")
    if labels:
        print(f"  verified: accented labels render ({', '.join(labels)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--keep-comments",
        action="store_true",
        help="stage the sources verbatim instead of dropping whole-line comments",
    )
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the repository checks and the rebuild gate (not recommended)",
    )
    args = ap.parse_args()
    strip = not args.keep_comments

    main_tex = HERE / "main.tex"
    if not main_tex.exists():
        raise SystemExit("main.tex not found; run this from the paper/ directory.")
    bbl = HERE / "main.bbl"
    if not bbl.exists():
        raise SystemExit(
            "main.bbl not found: arXiv does not run BibTeX, so an upload"
            " without it builds with an empty bibliography.\nBuild with"
            " `tectonic --keep-intermediates main.tex` (plain `tectonic"
            " main.tex` discards the .bbl), then re-run."
        )
    # references.bib is inert on arXiv (AutoTeX runs no BibTeX) but the
    # rebuild gate needs it: tectonic re-runs BibTeX, and without the .bib the
    # bibliography comes back empty and the gate fails.
    bib = HERE / "references.bib"
    if not bib.exists():
        raise SystemExit("references.bib not found; the rebuild gate needs it.")
    canonical_pdf = HERE / "main.pdf"
    if not args.no_verify and not canonical_pdf.exists():
        raise SystemExit("main.pdf not found; build it first, or pass --no-verify.")

    if not args.no_verify:
        print("repository checks:", flush=True)
        run_repo_checks()

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    tex = main_tex.read_text(encoding="utf-8")
    removed = stage_text(main_tex, OUT / "main.tex", strip)
    staged_tex = (OUT / "main.tex").read_text(encoding="utf-8")
    for marker in DRAFT_MARKERS:
        if marker in staged_tex:
            raise SystemExit(f"draft marker {marker!r} in main.tex; not shipping it.")
    bib_removed = stage_text(bib, OUT / "references.bib", strip)
    shutil.copy2(bbl, OUT / "main.bbl")

    staged_texts = {"main.tex": staged_tex}
    missing = []
    for rel in referenced_inputs(tex):
        src = HERE / rel
        if src.exists():
            stage_text(src, OUT / rel, strip)
            staged_texts[rel] = (OUT / rel).read_text(encoding="utf-8")
        else:
            missing.append(rel)
    for src in referenced_figures(tex, FIG_DIR):
        if src.exists():
            dst = OUT / src.relative_to(HERE)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            missing.append(src.name)
    if missing:
        raise SystemExit(
            "missing figures or fragments (regenerate with"
            " scripts/paper_figures.py): " + ", ".join(missing)
        )
    unsafe = unsafe_for_pdftex(staged_texts)
    if unsafe:
        raise SystemExit(
            "characters arXiv's pdfTeX would not read:\n  " + "\n  ".join(unsafe)
        )

    n_fig = sum(1 for p in (OUT / "figures").glob("*.pdf"))
    n_tab = sum(1 for p in (OUT / "figures").glob("*.tex"))
    print(
        f"staged {OUT.name}/: main.tex, main.bbl, references.bib,"
        f" {n_fig} figures, {n_tab} table fragments"
    )
    if strip:
        print(
            f"  stripped {removed} whole-line comments from main.tex,"
            f" {bib_removed} from references.bib"
        )
    print(f"  \\date is {date_of(staged_tex)!r}; set it to the submission date")

    labels = non_ascii_labels(bbl.read_text(encoding="utf-8"))
    if args.no_verify:
        print("  WARNING: every gate skipped (--no-verify)")
    else:
        verify_rebuild(canonical_pdf, labels)

    tar_path = HERE / "arxiv.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(OUT)))
    size = tar_path.stat().st_size / 1e6
    print(f"wrote {tar_path.name} ({size:.1f} MB).")
    print("Upload arxiv.tar.gz, or the contents of arxiv/ directly.")


if __name__ == "__main__":
    main()
