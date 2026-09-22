"""The results cache of #54: schema, round trip, selection. No sweeps run here."""

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from rbf_hyperbolic_interfaces.results_cache import (
    KINDS,
    SCHEMA,
    ResultsCache,
    json_ready,
)


def _cache() -> ResultsCache:
    args = Namespace(
        widths=[0.0, 0.0025],
        ns=[100, 200, 400],
        out_dir=Path("outputs"),
        direction=(0, 1),
        seed=np.int64(0),
    )
    cache = ResultsCache.new("scripts/wave1d_stiff.py", args)
    ns = [100, 200, 400]
    h = [2 / n for n in ns]
    naive = dict(h=h, delta=0.0, mode="naive", field="f")
    cache.add_errors(ns, [7.1e-2, 3.2e-2, 1.6e-2], [1.15, 1.0], **naive)
    aware = dict(h=h, delta=0.0, mode="aware", field="f")
    cache.add_errors(ns, [3.9e-3, 2.5e-4, 1.6e-5], [3.96, 3.97], **aware)
    cache.add(
        "spectrum",
        n=900,
        h=1 / 30,
        delta=0.0,
        width_label="jump",
        variant="naive",
        rows=0,
        dt=0.01,
        gamma=1e-9,
        max_re=1e-3,
        max_re_hyper=5.9e-2,
        min_re_hyper=-170.0,
        rk4_max=1.0004,
    )
    return cache


def test_args_are_json_native_and_provenance_is_recorded() -> None:
    cache = _cache()
    assert cache.schema == SCHEMA
    assert cache.args["out_dir"] == "outputs"
    assert cache.args["direction"] == [0, 1]
    assert cache.args["seed"] == 0 and type(cache.args["seed"]) is int
    assert len(cache.date) == 10
    json.dumps(cache.to_dict())  # nothing left that json cannot take


def test_records_carry_every_key_of_their_kind() -> None:
    cache = _cache()
    for rec in cache.records:
        assert set(rec) == {"kind", *KINDS[rec["kind"]]}
    spectrum = cache.select("spectrum")[0]
    assert spectrum["energy_ratio"] is None  # not measured: present, None
    with pytest.raises(ValueError, match="unknown record kind"):
        cache.add("energy", n=1)
    with pytest.raises(ValueError, match="unknown keys"):
        cache.add("error", n=1, colour="blue")


def test_add_errors_aligns_rates_with_the_previous_resolution() -> None:
    cache = _cache()
    naive = cache.select("error", mode="naive")
    assert [r["n"] for r in naive] == [100, 200, 400]
    assert naive[0]["rate"] is None
    assert naive[1]["rate"] == pytest.approx(1.15)
    assert naive[2]["h"] == pytest.approx(0.005)
    with pytest.raises(ValueError, match="one fewer"):
        cache.add_errors([1, 2], [1.0, 2.0], [1.0, 2.0], h=[1.0, 0.5], mode="x")


def test_round_trip_through_json(tmp_path: Path) -> None:
    cache = _cache()
    a, b = tmp_path / "a" / "c.json", tmp_path / "b.json"
    cache.write(a, b)
    assert a.read_text() == b.read_text()
    back = ResultsCache.read(a)
    assert back.to_dict() == cache.to_dict()
    assert back.select("error", mode="aware", n=400)[0]["error"] == pytest.approx(
        1.6e-5
    )
    assert back.values("mode", "error") == ["naive", "aware"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other", "script": "x"}))
    with pytest.raises(ValueError, match="schema"):
        ResultsCache.read(bad)


def test_json_ready_handles_nested_numpy() -> None:
    out = json_ready({"a": (np.float64(1.5), [np.int32(2)]), "p": Path("x/y")})
    assert out == {"a": [1.5, [2]], "p": "x/y"}
    assert type(out["a"][0]) is float
