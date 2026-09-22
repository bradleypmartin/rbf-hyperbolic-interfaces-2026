"""The packaging transforms of paper/make_arxiv.py (#62).

The script is not a package; it is loaded from its path. The rebuild gate
itself (tectonic, pdftotext) is exercised by running the script, not here.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def _load():
    spec = importlib.util.spec_from_file_location("make_arxiv", PAPER / "make_arxiv.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_arxiv = _load()


def test_strip_drops_whole_line_comments_only():
    tex = "% TRACE: notes\n  % indented\ntext % inline stays\n\\%literal\nmore\n"
    out, removed = make_arxiv.strip_whole_line_comments(tex)
    assert removed == 2
    assert out == "text % inline stays\n\\%literal\nmore\n"


def test_strip_refuses_verbatim():
    with pytest.raises(SystemExit, match="verbatim"):
        make_arxiv.strip_whole_line_comments("\\begin{verbatim}\n%\n\\end{verbatim}\n")


def test_referenced_figures_resolves_extensionless_names(tmp_path):
    (tmp_path / "a.pdf").touch()
    (tmp_path / "b.png").touch()
    tex = (
        "\\includegraphics[width=\\textwidth]{a}\n"
        "\\includegraphics{b}\n\\includegraphics{figures/a}\n\\includegraphics{c}\n"
    )
    got = make_arxiv.referenced_figures(tex, tmp_path)
    assert [p.name for p in got] == ["a.pdf", "b.png", "c"]


def test_referenced_inputs_adds_tex_and_dedups():
    tex = "\\input{figures/tab_a.tex}\n\\input{figures/tab_b}\n"
    tex += "\\input{figures/tab_a.tex}"
    assert make_arxiv.referenced_inputs(tex) == [
        "figures/tab_a.tex",
        "figures/tab_b.tex",
    ]


def test_non_ascii_labels():
    bbl = "\\bibitem[AS55]{a}\n\\bibitem[Mü73]{b}\n\\bibitem[Pó22]{c}\n"
    assert make_arxiv.non_ascii_labels(bbl) == ["Mü73", "Pó22"]


def test_unsafe_for_pdftex_flags_greek_but_not_accents():
    texts = {"ok": "Pólya – Mühlbach", "bad": "width δ"}
    got = make_arxiv.unsafe_for_pdftex(texts)
    assert len(got) == 1 and got[0].startswith("bad: 'δ'")


def test_date_of():
    assert make_arxiv.date_of("\\title{x}\n\\date{September 20, 2026}\n") == (
        "September 20, 2026"
    )


def test_manuscript_references_resolve():
    """Every figure and fragment main.tex names is in paper/figures/."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    figures = make_arxiv.referenced_figures(tex, PAPER / "figures")
    assert figures and all(p.exists() for p in figures), figures
    inputs = make_arxiv.referenced_inputs(tex)
    assert inputs and all((PAPER / rel).exists() for rel in inputs), inputs
    stripped, _ = make_arxiv.strip_whole_line_comments(tex)
    assert not make_arxiv.unsafe_for_pdftex({"main.tex": stripped})
