#!/usr/bin/env python3
"""Assemble a self-contained arXiv upload for the manuscript (epic #51, #62).

arXiv builds from a flat source tree and does **not** run BibTeX — the upload
must carry the built ``main.bbl``. This script copies ``main.tex``,
``references.bib`` and ``main.bbl`` into a clean ``arxiv/`` directory, copies
every figure referenced by ``main.tex`` from ``figures/`` into
``arxiv/figures/``, and writes an ``arxiv.tar.gz`` ready to upload.

Run from the ``paper/`` directory:  ``uv run python make_arxiv.py``

(Copied from the same author's bolza-bending paper tooling, itself adapted
from dirichlet-bridge. #62 decides whether to add weil-positivity-lab's
comment stripping and rebuild-and-compare gate.)
"""

import re
import shutil
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent  # paper/
FIG_SRC = HERE / "figures"
OUT = HERE / "arxiv"
OUT_FIG = OUT / "figures"


def referenced_figures(tex: str) -> list[str]:
    """Filenames in every \\includegraphics{...} of the source."""
    names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    # strip any directory prefix; graphicspath resolves the basename
    return sorted({Path(n).name for n in names})


def main() -> None:
    main_tex = HERE / "main.tex"
    if not main_tex.exists():
        raise SystemExit("main.tex not found; run from the paper/ directory.")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT_FIG.mkdir(parents=True)

    tex = main_tex.read_text(encoding="utf-8")
    shutil.copy2(main_tex, OUT / "main.tex")

    bib = HERE / "references.bib"
    if bib.exists():
        shutil.copy2(bib, OUT / "references.bib")
    else:
        print("WARNING: references.bib not found — bibliography will be empty.")

    # arXiv's AutoTeX does not run BibTeX, so an upload without main.bbl builds
    # with an empty bibliography. Hard-fail rather than package a broken tarball.
    bbl = HERE / "main.bbl"
    if not bbl.exists():
        raise SystemExit(
            "main.bbl not found — arXiv does not run BibTeX, so the upload needs"
            " it.\nBuild with `tectonic --keep-intermediates main.tex` (plain"
            " `tectonic main.tex` discards the .bbl) or `latexmk -pdf main.tex`,"
            " then rerun."
        )
    shutil.copy2(bbl, OUT / "main.bbl")

    missing = []
    for name in referenced_figures(tex):
        src = FIG_SRC / name
        if src.exists():
            shutil.copy2(src, OUT_FIG / name)
        else:
            missing.append(name)
    if missing:
        raise SystemExit(
            "Missing figures (regenerate with the figure scripts; see the"
            " figures issue / README): " + ", ".join(missing)
        )

    # Build the tarball.
    tar_path = HERE / "arxiv.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(OUT)))

    n_fig = len(list(OUT_FIG.glob("*")))
    print(f"Wrote {OUT}/ ({n_fig} figures) and {tar_path.name}.")
    print("Upload arxiv.tar.gz to arXiv, or upload the contents of arxiv/ directly.")


if __name__ == "__main__":
    main()
