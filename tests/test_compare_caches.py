"""The cache comparison of scripts/compare_caches.py (#6).

The script is not a package; it is loaded from its path.
"""

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CACHE = {
    "schema": "pdes-demo/results-cache@1",
    "script": "scripts/wave1d_stiff.py",
    "args": {"widths": [0.0, 0.01], "data_dir": "paper/data"},
    "date": "2026-09-20",
    "git_sha": "abc1234",
    "records": [
        {"kind": "error", "n": 100, "mode": "naive", "error": 0.25, "rate": None},
        {"kind": "error", "n": 200, "mode": "naive", "error": 0.0625, "rate": 2.0},
    ],
}


def _load():
    path = ROOT / "scripts" / "compare_caches.py"
    spec = importlib.util.spec_from_file_location("compare_caches", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_caches = _load()


def _write(dir_path: Path, cache: dict, name: str = "wave1d_stiff.json") -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / name).write_text(json.dumps(cache))
    return dir_path


def test_identical_caches_have_no_findings(tmp_path):
    ref = _write(tmp_path / "ref", CACHE)
    new = _write(tmp_path / "new", CACHE)
    lines, findings = compare_caches.compare_dirs(ref, new, 1e-9)
    assert findings == []
    assert "worst rel 0.00e+00" in lines[0]


def test_provenance_of_a_rerun_is_not_a_finding(tmp_path):
    rerun = copy.deepcopy(CACHE)
    rerun["date"] = "2026-09-22"
    rerun["git_sha"] = "def5678"
    rerun["args"]["data_dir"] = "/scratch/data"
    ref = _write(tmp_path / "ref", CACHE)
    new = _write(tmp_path / "new", rerun)
    assert compare_caches.compare_dirs(ref, new, 1e-9)[1] == []


def test_difference_below_tolerance_passes_and_above_fails(tmp_path):
    drifted = copy.deepcopy(CACHE)
    drifted["records"][0]["error"] *= 1 + 1e-10
    ref = _write(tmp_path / "ref", CACHE)
    new = _write(tmp_path / "new", drifted)
    assert compare_caches.compare_dirs(ref, new, 1e-9)[1] == []
    findings = compare_caches.compare_dirs(ref, new, 1e-12)[1]
    assert len(findings) == 1
    assert "#0 kind=error n=100 mode=naive" in findings[0]
    assert ".error" in findings[0]


def test_changed_argument_is_a_finding(tmp_path):
    other = copy.deepcopy(CACHE)
    other["args"]["widths"] = [0.0, 0.02]
    ref = _write(tmp_path / "ref", CACHE)
    new = _write(tmp_path / "new", other)
    findings = compare_caches.compare_dirs(ref, new, 1e-9)[1]
    assert any("arguments differ" in f and "widths" in f for f in findings)


def test_non_float_mismatch_is_a_finding(tmp_path):
    relabelled = copy.deepcopy(CACHE)
    relabelled["records"][0]["mode"] = "aware"
    ref = _write(tmp_path / "ref", CACHE)
    new = _write(tmp_path / "new", relabelled)
    findings = compare_caches.compare_dirs(ref, new, 1e-9)[1]
    assert any("fields differ" in f and "mode" in f for f in findings)


def test_missing_and_extra_files_are_findings(tmp_path):
    ref = _write(tmp_path / "ref", CACHE)
    _write(ref, CACHE, "wave2d_stiff.json")
    new = _write(tmp_path / "new", CACHE)
    _write(new, CACHE, "wave2d_stiff_d12.json")
    findings = compare_caches.compare_dirs(ref, new, 1e-9)[1]
    assert any("wave2d_stiff.json: only in the reference" in f for f in findings)
    assert any("wave2d_stiff_d12.json: only in the candidate" in f for f in findings)


def test_record_counts_that_differ_are_a_finding(tmp_path):
    short = copy.deepcopy(CACHE)
    short["records"] = short["records"][:1]
    ref = _write(tmp_path / "ref", CACHE)
    new = _write(tmp_path / "new", short)
    findings = compare_caches.compare_dirs(ref, new, 1e-9)[1]
    assert any("2 records vs 1" in f for f in findings)
