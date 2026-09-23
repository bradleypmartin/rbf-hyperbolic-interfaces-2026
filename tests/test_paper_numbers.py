"""The --data-dir flag of scripts/paper_numbers.py (#6).

The script is not a package; it is loaded from its path. The cold run of #6
points it at a scratch cache, so the flag must read that directory and nothing
else: a perturbed copy of paper/data has to fail.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data"


def _load():
    path = ROOT / "scripts" / "paper_numbers.py"
    spec = importlib.util.spec_from_file_location("paper_numbers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper_numbers = _load()


def test_committed_cache_passes():
    assert paper_numbers.main(["--quiet"]) == 0


def test_copy_of_cache_passes(tmp_path):
    shutil.copytree(DATA, tmp_path, dirs_exist_ok=True)
    assert paper_numbers.main(["--quiet", "--data-dir", str(tmp_path)]) == 0


def test_perturbed_copy_fails(tmp_path, capsys):
    shutil.copytree(DATA, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "wave1d_stiff.json"
    cache = json.loads(path.read_text())
    for rec in cache["records"]:
        if rec.get("kind") == "error" and rec.get("mode") == "aware":
            rec["error"] *= 3
    path.write_text(json.dumps(cache))
    assert paper_numbers.main(["--quiet", "--data-dir", str(tmp_path)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_missing_file_names_it(tmp_path):
    shutil.copytree(DATA, tmp_path, dirs_exist_ok=True)
    (tmp_path / "wave2d_stiff_cmp.json").unlink()
    with pytest.raises(SystemExit, match="wave2d_stiff_cmp.json is missing"):
        paper_numbers.main(["--data-dir", str(tmp_path)])


def test_empty_dir_refused(tmp_path):
    with pytest.raises(SystemExit, match="no \\*.json"):
        paper_numbers.main(["--data-dir", str(tmp_path)])
