"""Check every cache-backed number the manuscript quotes (issue demo#61).

The assembly pass re-checked each number in ``paper/main.tex`` against the
notes and the results cache of demo#54. This script is the scripted half of that
check: for every ratio, rate, factor and error the text quotes from
``paper/data/*.json`` it recomputes the value from the cache's unrounded
records and fails if it no longer rounds to the quoted figure (half a unit in
the quoted figure's last digit, capped at 6% for a round headline figure
such as 200; ``<=`` and ``>=`` for bounds). Numbers the
text quotes from the notes only (row counts, timings, scratch runs) are not
here; their ``% TRACE`` comments say so.

    uv run python scripts/paper_numbers.py          # prints one line per check
    uv run python scripts/paper_numbers.py --quiet  # failures only
    uv run python scripts/paper_numbers.py --data-dir <dir>  # another cache (#6)

Add a line to CHECKS whenever the manuscript quotes a new cache-backed
number, with the section it appears in.
"""

# ruff: noqa: E501  (the CHECKS table quotes the manuscript's own figures)
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data"
NS = [2500, 4900, 10000, 19600]
N1 = [100, 200, 400, 800, 1600]

# Filled by load(); the helpers below read it, so the CHECKS table stays as quoted.
CACHE = {}


def load(data_dir):
    """Replace CACHE with the ``*.json`` files of ``data_dir``, keyed by stem."""
    paths = sorted(Path(data_dir).glob("*.json"))
    if not paths:
        raise SystemExit(f"no *.json in {data_dir}")
    CACHE.clear()
    CACHE.update({p.stem: json.loads(p.read_text()) for p in paths})


def records(name, **filters):
    if name not in CACHE:
        raise SystemExit(f"{name}.json is missing from the data dir")
    out = [
        r
        for r in CACHE[name]["records"]
        if all(r.get(k) == v for k, v in filters.items())
    ]
    return sorted(out, key=lambda r: (r.get("n") or 0, r.get("t") or 0))


def errors(name, delta, mode, field):
    """``{n: error}`` for one (delta, mode, field) sweep."""
    return {
        r["n"]: r["error"]
        for r in records(name, kind="error", delta=delta, mode=mode, field=field)
    }


def rates(name, delta, mode, field):
    """The cache's rates between successive n, coarsest pair first."""
    rr = records(name, kind="error", delta=delta, mode=mode, field=field)
    return [r["rate"] for r in rr if r["rate"] is not None]


def truncation(name, delta, mode, group):
    return {
        r["n"]: r["error"]
        for r in records(name, kind="truncation", delta=delta, mode=mode, group=group)
    }


def snapshot(name, t, mode, field):
    (r,) = records(name, kind="snapshot", t=t, mode=mode, field=field)
    return r["error"]


def spectrum(name, width, variant, key):
    (r,) = records(name, kind="spectrum", width_label=width, variant=variant)
    return r[key]


def ratio(a, b, ns=NS):
    return [a[n] / b[n] for n in ns]


def rate_in_h(e, ns=NS):
    """Order in h = n^{-1/2} between successive n (the 2-D drivers' rule)."""
    return [
        2 * math.log(e[a] / e[b]) / math.log(b / a)
        for a, b in zip(ns, ns[1:], strict=False)
    ]


def rounds_to(value, quoted):
    """True when ``value`` rounds to the figure ``quoted`` (a string)."""
    q = float(quoted)
    mant, _, exp = quoted.lower().partition("e")
    exponent = int(exp) if exp else 0
    if "." in mant:
        unit = 10.0 ** (exponent - len(mant.split(".")[1]))
    else:  # trailing zeros of an integer figure are not significant: 200, 730
        digits = mant.lstrip("-")
        unit = 10.0 ** (exponent + len(digits) - len(digits.rstrip("0")))
        if digits != digits.rstrip("0"):  # a "200" must not pass a 25% drift
            unit = min(unit, 0.12 * abs(q))
    return abs(value - q) <= 0.5 * unit + 1e-9 * abs(q)


def check(section, what, value, quoted, kind="="):
    q = float(quoted)
    if kind == "=":
        ok = rounds_to(value, quoted)
    elif kind == "<=":
        ok = value <= q * (1 + 1e-9)
    elif kind == ">=":
        ok = value >= q * (1 - 1e-9)
    else:
        raise ValueError(kind)
    return (ok, section, what, value, quoted, kind)


def each(section, what, values, quoted, kind="="):
    """One check per element of ``values`` against the matching ``quoted``."""
    assert len(values) == len(quoted), (section, what)
    return [
        check(section, f"{what} [{i}]", v, q, kind)
        for i, (v, q) in enumerate(zip(values, quoted, strict=True))
    ]


def build_checks():  # noqa: PLR0915 (a table, not logic)
    C = []
    # ------------------------------------------------------------ 1-D, §1 / §4
    f = "wave1d_stiff"
    naive = {d: errors(f, d, "naive", "f") for d in (0.0, 0.0025, 0.01, 0.04)}
    seeds = {d: errors(f, d, "aware", "f") for d in (0.0, 0.0025, 0.01, 0.04)}
    r025 = rates(f, 0.0025, "naive", "f")
    C += each(
        "§1, §4", "FD4 rates while h > delta, delta = 0.0025", r025[:2], ["1.3", "2.2"]
    )
    C.append(
        check(
            "§1, §4",
            "FD4 drop 400 -> 800 nodes, delta = 0.0025",
            naive[0.0025][400] / naive[0.0025][800],
            "200",
        )
    )
    C.append(
        check(
            "§4",
            "FD4 drop 400 -> 1600 nodes, delta = 0.0025",
            naive[0.0025][400] / naive[0.0025][1600],
            "850",
        )
    )
    C.append(
        check(
            "§4",
            "FD4 average rate over those two doublings",
            math.log(naive[0.0025][400] / naive[0.0025][1600]) / math.log(4),
            "4.9",
        )
    )
    C += each("§4", "FD4 rates past the knee, delta = 0.0025", r025[2:], ["7.7", "2.0"])
    C += each(
        "§4",
        "FD4 rates after the knee, delta = 0.01",
        rates(f, 0.01, "naive", "f")[1:],
        ["4.0", "5.0", "4.0"],
    )
    C += each(
        "§4",
        "interface-aware errors at the jump, n = 100 and 1600",
        [seeds[0.0][100], seeds[0.0][1600]],
        ["3.9e-3", "6.2e-8"],
    )
    C.append(
        check(
            "§1, §4",
            "seeds within 0.3% of the jump case, delta = 0.0025 (max % over n)",
            100 * max(abs(x - 1) for x in ratio(seeds[0.0025], seeds[0.0], N1)),
            "0.3",
            "<=",
        )
    )
    C.append(
        check(
            "§4",
            "seeds within 2% of the jump case, delta = 0.01",
            100 * max(abs(x - 1) for x in ratio(seeds[0.01], seeds[0.0], N1)),
            "2",
            "<=",
        )
    )
    C.append(
        check(
            "§4, §6.4",
            "FD4 and seeds within 1% at delta = 0.04",
            100 * max(abs(x - 1) for x in ratio(naive[0.04], seeds[0.04], N1)),
            "1",
            "<=",
        )
    )
    q = ratio(naive[0.0025], seeds[0.0025], N1)
    C += each(
        "§4",
        "FD4 / seeds at h = 8 delta, 2 delta, delta / 2",
        [q[0], q[2], q[4]],
        ["18", "400", "120"],
    )
    C += each(
        "§4",
        "1-D still: FD4 sawtooth and seed error, fractions of the pulse",
        [
            snapshot(f, 1.0, "naive", "max_abs_f"),
            snapshot(f, 1.0, "aware", "max_abs_f"),
        ],
        ["0.04", "0.003"],
    )
    # the standing alternative, 1-D (§1.1, §4, §7.3)
    T = {
        (d, m): errors(f, d, m, "f")
        for d in (0.0, 0.0025, 0.01, 0.04)
        for m in ("cell", "cell2", "bandlimit", "widen1", "widen2")
    }
    j = ratio(naive[0.0], T[0.0, "cell2"], N1)
    C += each(
        "§4",
        "sampled / two-cell mean at the jump, n = 100 and 1600",
        [j[0], j[-1]],
        ["5.5", "87"],
    )
    j = ratio(T[0.0, "cell2"], seeds[0.0], N1)
    C += each(
        "§1.1, §4, §7.3",
        "two-cell mean / interface-aware at the jump, n = 100 and 1600",
        [j[0], j[-1]],
        ["3.3", "730"],
    )
    C.append(
        check(
            "§4",
            "band-limited / sampled at the jump, n = 1600",
            T[0.0, "bandlimit"][1600] / naive[0.0][1600],
            "0.8",
        )
    )
    C += each(
        "§4",
        "widened h, 2h / sampled at the jump, n = 1600",
        [
            T[0.0, "widen1"][1600] / naive[0.0][1600],
            T[0.0, "widen2"][1600] / naive[0.0][1600],
        ],
        ["1.4", "2.7"],
    )
    c1 = (
        ratio(naive[0.0025], T[0.0025, "cell"], N1)[:3]
        + ratio(naive[0.0025], T[0.0025, "bandlimit"], N1)[:3]
    )
    C += each(
        "§4",
        "one-cell mean and band-limited gain while h > delta, delta = 0.0025 (min, max)",
        [min(c1), max(c1)],
        ["1.25", "1.6"],
    )
    c2 = ratio(naive[0.0025], T[0.0025, "cell2"], N1)
    C += each(
        "§4",
        "two-cell mean gain while h >= 2 delta (min, max), loss from h = delta (min, max)",
        [min(c2[:3]), max(c2[:3]), 1 / max(c2[3:]), 1 / min(c2[3:])],
        ["5.7", "11", "4.7", "4.9"],
    )
    s2 = ratio(T[0.0025, "cell2"], seeds[0.0025], N1)
    C += each(
        "§4",
        "two-cell mean / seeds at h = 8, 2, 1, 1/2 delta",
        [s2[0], s2[2], s2[3], s2[4]],
        ["3.2", "37", "146", "580"],
    )
    w1 = [1 / x for x in ratio(naive[0.0025], T[0.0025, "widen1"], N1)[:3]]
    w2 = [1 / x for x in ratio(naive[0.0025], T[0.0025, "widen2"], N1)]
    C += each(
        "§4",
        "widened h / sampled while it acts (min, max); widened 2h / sampled at h >= 2 delta (min, max) and at h = delta",
        [min(w1), max(w1), min(w2[:3]), max(w2[:3]), w2[3]],
        ["1.03", "1.7", "2.2", "5.1", "350"],
    )
    res = [
        T[d, m][1600] / naive[d][1600]
        for d in (0.01, 0.04)
        for m in ("cell", "cell2", "bandlimit")
    ]
    C += each(
        "§4",
        "treatments / sampled on a resolved edge at n = 1600 (min, max)",
        [min(res), max(res)],
        ["47", "530"],
    )
    # ------------------------------------------------------------ 2-D flat, §6.2–6.4
    f = "wave2d_stiff"
    fl_v, fl_u = errors(f, None, "floor", "v"), errors(f, None, "floor", "max_u")
    nv = {d: errors(f, d, "naive", "v") for d in (0.0, 0.0025, 0.01, 0.04)}
    sv = {d: errors(f, d, "aware", "v") for d in (0.0, 0.0025, 0.01, 0.04)}
    nu = {d: errors(f, d, "naive", "max_u") for d in (0.0, 0.0025, 0.01, 0.04)}
    su = {d: errors(f, d, "aware", "max_u") for d in (0.0, 0.0025, 0.01, 0.04)}
    nh = {d: errors(f, d, "naive", "h") for d in (0.0, 0.0025, 0.01, 0.04)}
    sh = {d: errors(f, d, "aware", "h") for d in (0.0, 0.0025, 0.01, 0.04)}
    C.append(
        check(
            "§6.2, §6.4",
            "naive within 7% of the floor at delta = 0.04 (max % over n)",
            100 * max(abs(x - 1) for x in ratio(nv[0.04], fl_v)),
            "7",
            "<=",
        )
    )
    C += each(
        "§6.2",
        "naive rates in v, delta = 0.0025",
        rates(f, 0.0025, "naive", "v"),
        ["1.9", "1.6", "2.2"],
    )
    C.append(
        check(
            "§6.2",
            "naive / floor in v at 19,600 nodes, delta = 0.0025",
            nv[0.0025][19600] / fl_v[19600],
            "10",
        )
    )
    q = ratio(nv[0.0025], nv[0.0])
    C += each(
        "§6.2",
        "naive delta = 0.0025 / naive jump in v (min, max)",
        [min(q), max(q)],
        ["0.6", "0.76"],
    )
    C += each(
        "§6.2",
        "naive excess over the floor in %, delta = 0.01",
        [100 * (x - 1) for x in ratio(nv[0.01], fl_v)],
        ["19", "17", "18", "7"],
    )
    ex = [math.sqrt(nv[0.01][n] ** 2 - fl_v[n] ** 2) for n in NS]
    C += each(
        "§6.2",
        "the edge's own contribution in quadrature, delta = 0.01, coarsest and finest",
        [ex[0], ex[-1]],
        ["3.3e-2", "5.1e-4"],
    )
    C += each(
        "§6.2",
        "spurious-u floor, coarsest and finest",
        [fl_u[2500], fl_u[19600]],
        ["8.9e-4", "3.0e-5"],
    )
    C += each(
        "§6.2",
        "naive u / floor u at the jump",
        ratio(nu[0.0], fl_u),
        ["11", "24", "46", "194"],
    )
    C += each(
        "§6.2, §6.4",
        "naive u / floor u, delta = 0.0025",
        ratio(nu[0.0025], fl_u),
        ["11", "24", "39", "116"],
    )
    q = ratio(nu[0.04], fl_u)
    C += each(
        "§6.2",
        "naive u / floor u, delta = 0.04 (min, max)",
        [min(q), max(q)],
        ["0.8", "1.3"],
    )
    C += each(
        "§6.2",
        "naive u / floor u, delta = 0.01",
        ratio(nu[0.01], fl_u),
        ["3", "3", "6", "3"],
    )
    C += each(
        "§6.2",
        "u drop 10,000 -> 19,600: naive at delta = 0.01, the floor",
        [nu[0.01][10000] / nu[0.01][19600], fl_u[10000] / fl_u[19600]],
        ["9", "4"],
    )
    C += each(
        "§6.4",
        "seed rates in v and sigma_yy, delta = 0.0025",
        rates(f, 0.0025, "aware", "v") + rates(f, 0.0025, "aware", "h"),
        ["4.1", "3.7", "3.5", "3.6", "3.2", "3.5"],
    )
    q = rates(f, 0.0025, "naive", "v")
    C += each(
        "§6.4",
        "naive rates in v, delta = 0.0025 (min, max)",
        [min(q), max(q)],
        ["1.6", "2.2"],
    )
    C += each(
        "§1, §6.4",
        "naive / seeds in v, delta = 0.0025",
        ratio(nv[0.0025], sv[0.0025]),
        ["2.9", "6.2", "13", "20"],
    )
    C += each(
        "§1, §6.4",
        "naive / seeds in spurious u, delta = 0.0025",
        ratio(nu[0.0025], su[0.0025]),
        ["5.5", "16", "25", "67"],
    )
    q = ratio(su[0.0025], fl_u)
    C += each(
        "§6.4",
        "seeds' u / floor u, delta = 0.0025 (min, max)",
        [min(q), max(q)],
        ["1.5", "1.9"],
    )
    C += each(
        "§6.4",
        "seeds and jump-aware in v at 2500 and 19,600 nodes",
        [sv[0.0025][2500], sv[0.0][2500], sv[0.0025][19600], sv[0.0][19600]],
        ["3.0e-2", "5.4e-2", "6.4e-4", "1.23e-3"],
    )
    C += each(
        "§6.4",
        "naive / seeds in v, delta = 0.01 (the crossover)",
        ratio(nv[0.01], sv[0.01]),
        ["2.1", "2.0", "1.5", "0.88"],
    )
    q = ratio(nu[0.01], su[0.01])
    C += each(
        "§6.4",
        "u at the crossover: seeds / naive at 2500, then naive / seeds",
        [1 / q[0], q[1], q[2], q[3]],
        ["1.3", "1.4", "1.8", "0.75"],
    )
    C += each(
        "§6.4",
        "seed and naive rates in v across delta = 0.01, first and last",
        [
            rates(f, 0.01, "aware", "v")[0],
            rates(f, 0.01, "aware", "v")[-1],
            rates(f, 0.01, "naive", "v")[0],
            rates(f, 0.01, "naive", "v")[-1],
        ],
        ["3.1", "2.7", "3.1", "4.2"],
    )
    qv, qu = ratio(sv[0.04], nv[0.04]), ratio(su[0.04], nu[0.04])
    C += each(
        "§1, §6.4",
        "seeds / naive on the resolved edge, v (min, max) and u (min, max)",
        [min(qv), max(qv), min(qu), max(qu)],
        ["1.4", "2.0", "3.7", "5.7"],
    )
    rs, rn = rates(f, 0.04, "aware", "v"), rates(f, 0.04, "naive", "v")
    C += each(
        "§6.4",
        "rates on the resolved edge, seeds (min, max) and naive (min, max)",
        [min(rs), max(rs), min(rn), max(rn)],
        ["2.9", "3.6", "2.9", "3.9"],
    )
    C += each(
        "§6.4",
        "seeds in sigma_yy at 2500 and 19,600 nodes, naive / seeds at 19,600, delta = 0.0025",
        [sh[0.0025][2500], sh[0.0025][19600], nh[0.0025][19600] / sh[0.0025][19600]],
        ["2.3e-2", "6.7e-4", "18"],
    )
    C += each(
        "§6.4",
        "flat still at t = 1: naive v, u; seeds v, u",
        [
            snapshot(f, 1.0, "naive", "v"),
            snapshot(f, 1.0, "naive", "max_u"),
            snapshot(f, 1.0, "aware", "v"),
            snapshot(f, 1.0, "aware", "max_u"),
        ],
        ["0.027", "4.7e-3", "0.0021", "1.9e-4"],
    )
    C += each(
        "§6.4",
        "flat still at t = 0.25: naive v, u; seeds v, u",
        [
            snapshot(f, 0.25, "naive", "v"),
            snapshot(f, 0.25, "naive", "max_u"),
            snapshot(f, 0.25, "aware", "v"),
            snapshot(f, 0.25, "aware", "max_u"),
        ],
        ["0.015", "3.5e-3", "0.0007", "1.9e-4"],
    )
    # ------------------------------------------------------------ comparators, 2-D (§1.1, §6.4, §7.3)
    fc, fk = "wave2d_stiff_cmp", "wave2d_stiff_cmp_a0.02"
    cnv = {d: errors(fc, d, "naive", "v") for d in (0.0025, 0.01)}
    knv = {d: errors(fk, d, "naive", "v") for d in (0.0025, 0.005, 0.01)}
    ksv = {
        d: errors("wave2d_stiff_a0.02", d, "aware", "v") for d in (0.0025, 0.005, 0.01)
    }
    cell = {("f", d): errors(fc, d, "cell", "v") for d in (0.0025, 0.01)} | {
        ("k", d): errors(fk, d, "cell", "v") for d in (0.0025, 0.005, 0.01)
    }
    lvl = [
        cnv[0.0025][2500] / cell["f", 0.0025][2500],
        knv[0.0025][2500] / cell["k", 0.0025][2500],
    ]
    C.append(
        check(
            "§6.4",
            "one-cell mean level with sampling at 2500 nodes, delta = 0.0025 (max % off, both geometries)",
            100 * max(abs(x - 1) for x in lvl),
            "5",
            "<=",
        )
    )
    fine = [cnv[0.0025][n] / cell["f", 0.0025][n] for n in NS[2:]] + [
        knv[0.0025][n] / cell["k", 0.0025][n] for n in NS[2:]
    ]
    C += each(
        "§6.4",
        "sampled / one-cell mean at 10,000 and 19,600, delta = 0.0025 (min, max)",
        [min(fine), max(fine)],
        ["2.1", "3.6"],
    )
    rr = rates(fc, 0.0025, "cell", "v")[1:] + rates(fk, 0.0025, "cell", "v")[1:]
    C += each(
        "§6.4",
        "one-cell mean rates over the last two doublings (min, max)",
        [min(rr), max(rr)],
        ["3.4", "3.6"],
    )
    cs = ratio(cell["f", 0.0025], sv[0.0025]) + ratio(cell["k", 0.0025], ksv[0.0025])
    C += each(
        "§1.1, §6.4, §7.3",
        "one-cell mean / seeds, delta = 0.0025, both geometries (min, max)",
        [min(cs), max(cs)],
        ["3.1", "5.9"],
    )
    bl = {("f", d): errors(fc, d, "bandlimit", "v") for d in (0.0025, 0.01)} | {
        ("k", d): errors(fk, d, "bandlimit", "v") for d in (0.0025, 0.005, 0.01)
    }
    coarse = [cnv[0.0025][n] / bl["f", 0.0025][n] for n in NS[:2]] + [
        knv[0.0025][n] / bl["k", 0.0025][n] for n in NS[:2]
    ]
    C.append(
        check(
            "§6.4",
            "band-limited level with or above sampling on the coarse sets (max sampled / treated)",
            max(coarse),
            "1.05",
            "<=",
        )
    )
    fine = [cnv[0.0025][n] / bl["f", 0.0025][n] for n in NS[2:]] + [
        knv[0.0025][n] / bl["k", 0.0025][n] for n in NS[2:]
    ]
    C += each(
        "§6.4",
        "sampled / band-limited on the fine sets, delta = 0.0025 (min, max)",
        [min(fine), max(fine)],
        ["1.5", "1.7"],
    )
    c2 = ratio(errors(fc, 0.0025, "cell2", "v"), cnv[0.0025]) + ratio(
        errors(fk, 0.0025, "cell2", "v"), knv[0.0025]
    )
    C += each(
        "§6.4",
        "two-cell mean / sampled, delta = 0.0025, both geometries (min, max)",
        [min(c2), max(c2)],
        ["1.05", "1.4"],
    )
    q = ratio(cnv[0.01], cell["f", 0.01])
    C += each(
        "§6.4",
        "flat delta = 0.01: sampled / one-cell mean while h >= 1.4 delta (max), one-cell mean / sampled at h = 0.7 delta",
        [max(q[:2]), 1 / q[3]],
        ["1.1", "2.4"],
    )
    q = ratio(bl["f", 0.01], cnv[0.01])
    C += each(
        "§6.4",
        "flat delta = 0.01: band-limited / sampled (min, max)",
        [min(q), max(q)],
        ["1.2", "1.8"],
    )
    q = ratio(errors(fc, 0.01, "cell2", "v"), cnv[0.01])
    C += each(
        "§6.4",
        "flat delta = 0.01: two-cell mean / sampled (min, max)",
        [min(q), max(q)],
        ["1.8", "10"],
    )
    q = ratio(knv[0.005], cell["k", 0.005])
    C.append(
        check(
            "§6.4",
            "curved delta = 0.005: one-cell mean below sampling at h = 2 delta",
            q[2],
            "1",
            ">=",
        )
    )
    C.append(
        check(
            "§6.4",
            "curved delta = 0.005: one-cell mean above sampling at h = 1.4 delta",
            q[3],
            "1",
            "<=",
        )
    )
    wid = []
    for name, base in ((fc, cnv), (fk, knv)):
        for d in base:
            for m in ("widen1", "widen2"):
                w = errors(name, d, m, "v")
                wid += [w[n] / base[d][n] for n in NS if w[n] / base[d][n] > 1.001]
    C += each(
        "§6.4",
        "widened edge / sampled wherever it acts, both geometries (min, max)",
        [min(wid), max(wid)],
        ["5", "120"],
    )
    q = ratio(errors(fc, 0.0025, "naive", "max_u"), errors(fc, 0.0025, "cell", "max_u"))
    C += each(
        "§6.4",
        "spurious u: sampled / one-cell mean, delta = 0.0025 (min, max)",
        [min(q), max(q)],
        ["1.3", "1.8"],
    )
    # ------------------------------------------------------------ oblique, §6.5
    f = "wave2d_stiff_d12"
    ov = {
        (d, m): errors(f, d, m, "v")
        for d in (0.0025, 0.01)
        for m in ("naive", "aware", "ablate", "sfloor")
    }
    ou = {
        (d, m): errors(f, d, m, "u")
        for d in (0.0025, 0.01)
        for m in ("naive", "aware", "ablate", "sfloor")
    }
    oh = {
        (d, m): errors(f, d, m, "h")
        for d in (0.0025, 0.01)
        for m in ("naive", "aware", "ablate")
    }
    ofl = errors(f, None, "floor", "v")
    C += each(
        "§1, §6.5",
        "naive / seeds in v, delta = 0.0025",
        ratio(ov[0.0025, "naive"], ov[0.0025, "aware"]),
        ["1.4", "1.7", "1.9", "1.7"],
    )
    C += each(
        "§6.5",
        "naive / seeds in u, delta = 0.0025",
        ratio(ou[0.0025, "naive"], ou[0.0025, "aware"]),
        ["1.2", "1.3", "1.5", "1.6"],
    )
    qv, qu = (
        ratio(ov[0.01, "naive"], ov[0.01, "aware"]),
        ratio(ou[0.01, "naive"], ou[0.01, "aware"]),
    )
    C += each(
        "§1, §6.5",
        "naive / seeds, delta = 0.01: v (min, max), u (min, max), v at h = 0.71 delta",
        [min(qv), max(qv), min(qu), max(qu), qv[3]],
        ["1.4", "1.7", "1.1", "1.7", "1.6"],
    )
    C += each(
        "§6.5",
        "ablation / seeds in v, delta = 0.0025",
        ratio(ov[0.0025, "ablate"], ov[0.0025, "aware"]),
        ["1.2", "1.6", "2.2", "2.3"],
    )
    C += each(
        "§6.5",
        "ablation / seeds in u, delta = 0.0025",
        ratio(ou[0.0025, "ablate"], ou[0.0025, "aware"]),
        ["1.3", "1.7", "2.4", "2.7"],
    )
    C.append(
        check(
            "§6.5",
            "ablation worse than naive in u at every n (min ratio)",
            min(ratio(ou[0.0025, "ablate"], ou[0.0025, "naive"])),
            "1",
            ">=",
        )
    )
    q = ratio(ov[0.0025, "ablate"], ov[0.0025, "naive"])
    C.append(
        check(
            "§6.5",
            "ablation better than naive in v on the coarse sets (max ratio)",
            max(q[:2]),
            "1",
            "<=",
        )
    )
    C.append(
        check(
            "§6.5",
            "ablation worse than naive in v from 10,000 nodes on (min ratio)",
            min(q[2:]),
            "1",
            ">=",
        )
    )
    C += each(
        "§6.5",
        "ablation and naive at 19,600 nodes: v, v; u, u",
        [
            ov[0.0025, "ablate"][19600],
            ov[0.0025, "naive"][19600],
            ou[0.0025, "ablate"][19600],
            ou[0.0025, "naive"][19600],
        ],
        ["1.91e-2", "1.40e-2", "4.3e-2", "2.6e-2"],
    )
    dev = [
        abs(x - 1)
        for x in ratio(ov[0.01, "ablate"], ov[0.01, "aware"])
        + ratio(ou[0.01, "ablate"], ou[0.01, "aware"])
    ]
    C.append(
        check(
            "§6.5",
            "ablation and seeds within 4% at delta = 0.01 (max % off, v and u)",
            100 * max(dev),
            "4.5",
            "<=",
        )
    )
    C += each(
        "§6.5",
        "seed and naive rates in v, delta = 0.0025",
        rates(f, 0.0025, "aware", "v") + rates(f, 0.0025, "naive", "v"),
        ["2.5", "2.3", "1.9", "2.0", "2.0", "2.2"],
    )
    C.append(
        check(
            "§6.5",
            "seeds / train floor in v at 19,600 nodes, delta = 0.0025",
            ov[0.0025, "aware"][19600] / ofl[19600],
            "7",
        )
    )
    C += each(
        "§6.5",
        "sigma_yy at 19,600 nodes, seeds and naive; seed rates in sigma_yy (min, max)",
        [
            oh[0.0025, "aware"][19600],
            oh[0.0025, "naive"][19600],
            min(rates(f, 0.0025, "aware", "h")),
            max(rates(f, 0.0025, "aware", "h")),
        ],
        ["1.62e-2", "1.58e-2", "1.3", "1.5"],
    )
    C += each(
        "§6.5",
        "oblique still at t = 1: naive v, u; seeds v, u",
        [
            snapshot(f, 1.0, "naive", "v"),
            snapshot(f, 1.0, "naive", "u"),
            snapshot(f, 1.0, "aware", "v"),
            snapshot(f, 1.0, "aware", "u"),
        ],
        ["0.030", "0.055", "0.016", "0.036"],
    )
    C += each(
        "§6.5",
        "oblique still at t = 0.25: naive v, u; seeds v, u",
        [
            snapshot(f, 0.25, "naive", "v"),
            snapshot(f, 0.25, "naive", "u"),
            snapshot(f, 0.25, "aware", "v"),
            snapshot(f, 0.25, "aware", "u"),
        ],
        ["0.012", "0.020", "0.009", "0.015"],
    )
    # ------------------------------------------------------------ curved, §6.6, §7.1
    f = "wave2d_stiff_a0.02"
    D = (0.0, 0.0025, 0.005, 0.01)
    cv = {(d, m): errors(f, d, m, "v") for d in D for m in ("naive", "aware")}
    cu = {(d, m): errors(f, d, m, "u") for d in D for m in ("naive", "aware")}
    ch = {(d, m): errors(f, d, m, "h") for d in D for m in ("naive", "aware")}
    sf = {d: errors(f, d, "sfloor", "v") for d in D[1:]}
    cfl = errors(f, None, "floor", "v")
    q = ratio(cv[0.0, "aware"], cfl)
    C += each(
        "§6.6",
        "curved jump stencils / floor in v (min, max); their rates; naive / jump-aware at 19,600",
        [min(q), max(q)]
        + rates(f, 0.0, "aware", "v")
        + [cv[0.0, "naive"][19600] / cv[0.0, "aware"][19600]],
        ["0.93", "1.07", "3.3", "3.6", "3.6", "14"],
    )
    C += each(
        "§6.6",
        "seeds / seed floor in v, delta = 0.005",
        ratio(cv[0.005, "aware"], sf[0.005]),
        ["0.79", "0.96", "1.2", "1.06"],
    )
    C += each(
        "§1, §6.6",
        "naive / seeds in v, delta = 0.005; naive rates",
        ratio(cv[0.005, "naive"], cv[0.005, "aware"]) + rates(f, 0.005, "naive", "v"),
        ["3.1", "7.0", "3.1", "2.2", "2.8", "4.5", "3.5"],
    )
    C += each(
        "§6.6",
        "naive / seeds in sigma_yy, delta = 0.005",
        ratio(ch[0.005, "naive"], ch[0.005, "aware"]),
        ["3.8", "7.7", "5.1", "2.9"],
    )
    flat = errors("wave2d_stiff", 0.01, "aware", "v")
    C += each(
        "§6.6",
        "seeds in v at delta = 0.01 from 4900 nodes on, curved then flat",
        [cv[0.01, "aware"][n] for n in NS[1:]] + [flat[n] for n in NS[1:]],
        ["1.04e-2", "3.9e-3", "1.62e-3", "1.05e-2", "3.9e-3", "1.57e-3"],
    )
    flatn = errors("wave2d_stiff", 0.01, "naive", "v")
    C.append(
        check(
            "§6.6",
            "curved naive within 17% of the flat naive at delta = 0.01 (max % off)",
            100 * max(abs(x - 1) for x in ratio(cv[0.01, "naive"], flatn)),
            "17",
        )
    )
    q = ratio(cv[0.01, "naive"], cv[0.01, "aware"])
    C += each(
        "§6.6",
        "delta = 0.01: seeds win at h = delta by, naive wins at h = 0.71 delta by",
        [q[2], 1 / q[3]],
        ["1.35", "1.25"],
    )
    q = ratio(cfl, sf[0.005])
    C += each(
        "§6.6",
        "naive floor / seed floor at 4900, 10,000, 19,600 nodes, delta = 0.005",
        q[1:],
        ["3.8", "2.9", "1.6"],
    )
    C += each(
        "§6.6",
        "seed floor and naive floor at those n",
        [
            sf[0.005][4900],
            cfl[4900],
            sf[0.005][10000],
            cfl[10000],
            sf[0.005][19600],
            cfl[19600],
        ],
        ["4.7e-3", "1.81e-2", "1.68e-3", "4.8e-3", "8.3e-4", "1.30e-3"],
    )
    fine = [rates(f, d, "sfloor", "v")[-1] for d in (0.005, 0.01)]
    C += each(
        "§6.6, §7.1",
        "seed floor's fine-end rates, delta = 0.005 and 0.01 (min, max); the naive floor's last two rates (min, max)",
        [
            min(fine),
            max(fine),
            min(rates(f, None, "floor", "v")[1:]),
            max(rates(f, None, "floor", "v")[1:]),
        ],
        ["2.1", "2.6", "3.7", "3.9"],
    )
    fine = [
        rates("wave2d_stiff", 0.01, "aware", "v")[-1],
        rates("wave2d_stiff_d12", 0.01, "aware", "v")[-1],
        rates(f, 0.005, "aware", "v")[-1],
        rates(f, 0.01, "aware", "v")[-1],
    ]
    C += each(
        "§6.6, §7.1",
        "seeds' fine-end rates where the seeded region floors them (min, max)",
        [min(fine), max(fine)],
        ["2.4", "2.7"],
    )
    C += each(
        "§6.6",
        "u at the jump: naive / jump-aware at 19,600; the two errors; jump-aware rates; naive rates",
        [
            cu[0.0, "naive"][19600] / cu[0.0, "aware"][19600],
            cu[0.0, "aware"][19600],
            cu[0.0, "naive"][19600],
        ]
        + rates(f, 0.0, "aware", "u")
        + rates(f, 0.0, "naive", "u"),
        ["4.2", "6.6e-3", "2.8e-2", "2.5", "3.1", "3.4", "2.0", "2.2", "1.6"],
    )
    C += each(
        "§6.6",
        "u at delta = 0.005: naive / seeds at 19,600; the two errors; seed rates",
        [
            cu[0.005, "naive"][19600] / cu[0.005, "aware"][19600],
            cu[0.005, "aware"][19600],
            cu[0.005, "naive"][19600],
        ]
        + rates(f, 0.005, "aware", "u"),
        ["2.3", "2.6e-3", "6.0e-3", "3.4", "3.5", "3.7"],
    )
    C += each(
        "§6.6",
        "u at delta = 0.01 and 19,600 nodes: seeds, naive",
        [cu[0.01, "aware"][19600], cu[0.01, "naive"][19600]],
        ["3.9e-3", "4.4e-3"],
    )
    C += each(
        "§6.6",
        "seeds' v rates at the fine end, delta = 0.005 (min, max)",
        [
            min(rates(f, 0.005, "aware", "v")[1:]),
            max(rates(f, 0.005, "aware", "v")[1:]),
        ],
        ["2.3", "2.5"],
    )
    tr = {
        (d, m, g): truncation(f, d, m, g)
        for d in D[1:]
        for m in ("naive", "aware")
        for g in ("edge", "bulk")
    }
    q = ratio(tr[0.005, "aware", "edge"], tr[0.005, "aware", "bulk"]) + ratio(
        tr[0.01, "aware", "edge"], tr[0.01, "aware", "bulk"]
    )
    C += each(
        "§6.6",
        "seed rows / bulk rows in the truncation probe, delta = 0.005 and 0.01 (min, max)",
        [min(q), max(q)],
        ["2.0", "3.6"],
    )
    rs = rate_in_h(tr[0.005, "aware", "edge"]) + rate_in_h(tr[0.01, "aware", "edge"])
    rb = rate_in_h(tr[0.005, "aware", "bulk"]) + rate_in_h(tr[0.01, "aware", "bulk"])
    C += each(
        "§6.6",
        "truncation rates: seed rows (min, max), bulk rows (min, max)",
        [min(rs), max(rs), min(rb), max(rb)],
        ["3.3", "3.9", "3.1", "3.7"],
    )
    q = ratio(tr[0.005, "naive", "edge"], tr[0.005, "aware", "edge"])
    rn = rate_in_h(tr[0.005, "naive", "edge"])
    C += each(
        "§6.6",
        "naive edge rows / seed rows, delta = 0.005 (min, max); naive edge-row rates (min, max)",
        [min(q), max(q), min(rn), max(rn)],
        ["4", "24", "1.5", "2.4"],
    )
    C += each(
        "§1, §6.6",
        "delta = 0.0025: naive / seeds in v; in sigma_yy and u at 19,600",
        ratio(cv[0.0025, "naive"], cv[0.0025, "aware"])
        + [
            ch[0.0025, "naive"][19600] / ch[0.0025, "aware"][19600],
            cu[0.0025, "naive"][19600] / cu[0.0025, "aware"][19600],
        ],
        ["3.2", "5.7", "8.2", "12.5", "14", "4.8"],
    )
    C += each(
        "§6.6",
        "delta = 0.0025: seed rates, naive rates in v",
        rates(f, 0.0025, "aware", "v") + rates(f, 0.0025, "naive", "v"),
        ["3.8", "2.9", "3.3", "2.1", "1.9", "2.1"],
    )
    C += each(
        "§6.6",
        "delta = 0.0025 at 19,600: seeds in v, the naive floor",
        [cv[0.0025, "aware"][19600], cfl[19600]],
        ["9.5e-4", "1.30e-3"],
    )
    C += each(
        "§6.6, §7.1",
        "curved / flat seeds in v, delta = 0.0025",
        ratio(cv[0.0025, "aware"], errors("wave2d_stiff", 0.0025, "aware", "v")),
        ["0.95", "1.04", "1.4", "1.5"],
    )
    C += each(
        "§6.6",
        "seeds / seed floor in v, delta = 0.0025",
        ratio(cv[0.0025, "aware"], sf[0.0025]),
        ["1.05", "1.2", "1.5", "1.3"],
    )
    q = ratio(cv[0.005, "aware"], sf[0.005])
    C += each(
        "§6.6",
        "seeds / seed floor in v, delta = 0.005 (min, max)",
        [min(q), max(q)],
        ["0.8", "1.2"],
    )
    q1, q2 = (
        ratio(tr[0.0025, "aware", "edge"], tr[0.0025, "aware", "bulk"]),
        ratio(tr[0.005, "aware", "edge"], tr[0.005, "aware", "bulk"]),
    )
    C += each(
        "§6.6",
        "truncation ratio seed rows / bulk, first and last n: delta = 0.0025, then 0.005",
        [q1[0], q1[-1], q2[0], q2[-1]],
        ["4.3", "4.8", "3.6", "2.8"],
    )
    C.append(
        check(
            "§6.6",
            "geometric contribution at 19,600 nodes, delta = 0.0025 (excess over the seed floor in quadrature)",
            math.sqrt(cv[0.0025, "aware"][19600] ** 2 - sf[0.0025][19600] ** 2),
            "6.3e-4",
        )
    )
    ft = "wave2d_stiff_a0.02_r0.001"
    tv, tu = errors(ft, 0.005, "aware", "v"), errors(ft, 0.005, "aware", "u")
    C += each(
        "§6.6, §7.1",
        "trimmed seeds' rates in v",
        rates(ft, 0.005, "aware", "v"),
        ["3.4", "3.2", "3.6"],
    )
    C += each(
        "§6.6",
        "trimmed and untrimmed seeds in v at 19,600",
        [tv[19600], cv[0.005, "aware"][19600]],
        ["9.4e-4", "8.8e-4"],
    )
    qv, qu = ratio(tv, cv[0.005, "aware"]), ratio(tu, cu[0.005, "aware"])
    C += each(
        "§6.6",
        "trimmed / untrimmed on the coarser sets: v (min, max), u (max)",
        [min(qv[:3]), max(qv[:3]), max(qu)],
        ["1.2", "2.2", "1.5"],
    )
    fs, fsc = "wave2d_stiff_eigenvalues_n2500", "wave2d_stiff_eigenvalues_n2500_a0.02"
    W = ("h/8", "h/2", "2h")
    C += each(
        "§6.6",
        "curved spectra: seeds' largest real part at h/8, h/2, 2h; the flat ones",
        [spectrum(fsc, w, "seeds", "max_re_hyper") for w in W]
        + [spectrum(fs, w, "seeds", "max_re_hyper") for w in W],
        ["6.0e-2", "8.3e-2", "1.05e-1", "6.3e-2", "7.8e-2", "9.6e-2"],
    )
    rk = [spectrum(fsc, w, "seeds", "rk4_max") for w in W]
    C += each(
        "§6.6",
        "curved spectra: RK4 amplification (min, max); energy ratios; jump-aware real part and amplification",
        [min(rk), max(rk)]
        + [spectrum(fsc, w, "seeds", "energy_ratio") for w in W]
        + [
            spectrum(fsc, "jump", "aware", "max_re_hyper"),
            spectrum(fsc, "jump", "aware", "rk4_max"),
        ],
        ["1.00025", "1.00043", "0.988", "0.999", "1.016", "4.7e-2", "1.0002"],
    )
    C.append(
        check(
            "§6.6",
            "curved vs flat rightmost eigenvalues about 10% apart at most (max % off)",
            100
            * max(
                abs(
                    spectrum(fsc, w, "seeds", "max_re_hyper")
                    / spectrum(fs, w, "seeds", "max_re_hyper")
                    - 1
                )
                for w in W
            ),
            "10",
        )
    )
    C += each(
        "§6.6",
        "curved still at t = 1: naive v, u; seeds v, u",
        [
            snapshot(f, 1.0, "naive", "v"),
            snapshot(f, 1.0, "naive", "u"),
            snapshot(f, 1.0, "aware", "v"),
            snapshot(f, 1.0, "aware", "u"),
        ],
        ["0.0063", "0.020", "0.0020", "0.0093"],
    )
    C += each(
        "§6.6",
        "curved still at t = 0.25: naive v, u; seeds v, u",
        [
            snapshot(f, 0.25, "naive", "v"),
            snapshot(f, 0.25, "naive", "u"),
            snapshot(f, 0.25, "aware", "v"),
            snapshot(f, 0.25, "aware", "u"),
        ],
        ["0.0037", "0.016", "0.0004", "0.005"],
    )
    # ------------------------------------------------------------ stability, §6.3
    fv = "wave2d_stiff_eigenvalues_n900_variants"
    C += each(
        "§6.3",
        "19-node hyperviscosity rows: largest real part at h/8, h/4, h/2, h; RK4 amplification at h/2, h",
        [spectrum(fv, w, "hyper19", "max_re_hyper") for w in ("h/8", "h/4", "h/2", "h")]
        + [spectrum(fv, w, "hyper19", "rk4_max") for w in ("h/2", "h")],
        ["6.2e-2", "8.7e-2", "1.05", "2.65", "1.003", "1.018"],
    )
    C.append(
        check(
            "§6.3",
            "plain 19-node degree-three stencils everywhere: smallest largest-real-part over the widths",
            min(
                spectrum(fv, w, "naive19", "max_re_hyper")
                for w in ("h/8", "h/4", "h/2", "h", "1.9h")
            ),
            "2.4",
        )
    )
    C += each(
        "§5.3, §6.3",
        "extreme Delta^3 eigenvalue: all-seed 19-node rows at h/2, the naive operator",
        [
            spectrum(fv, "h/2", "hyper19", "min_re_hyper"),
            spectrum(fv, "h/2", "naive", "min_re_hyper"),
        ],
        ["-82", "-151"],
    )
    C += each(
        "§6.3",
        "30-node seed rows: largest real part at h/2, h, 1.9h, h/8, h/4; RK4 amplification at h/8",
        [
            spectrum(fv, w, "seeds30", "max_re_hyper")
            for w in ("h/2", "h", "1.9h", "h/8", "h/4")
        ]
        + [spectrum(fv, "h/8", "seeds30", "rk4_max")],
        ["0.10", "0.05", "0.06", "30", "28", "1.23"],
    )
    ch_ = [
        spectrum(fv, w, "seeds", "max_re_hyper")
        for w in ("h/8", "h/4", "h/2", "h", "1.9h")
    ]
    C += each(
        "§6.3",
        "the chosen operator on 900 nodes: largest real part (min, max), RK4 amplification (max), extreme Delta^3 eigenvalue at h/8, the naive amplification",
        [
            min(ch_),
            max(ch_),
            max(
                spectrum(fv, w, "seeds", "rk4_max")
                for w in ("h/8", "h/4", "h/2", "h", "1.9h")
            ),
            spectrum(fv, "h/8", "seeds", "min_re_hyper"),
            spectrum(fv, "h/8", "naive", "rk4_max"),
        ],
        ["4.1e-2", "6.6e-2", "1.0005", "-176", "1.0004"],
    )
    rk = [spectrum(fs, w, "seeds", "rk4_max") for w in W]
    C += each(
        "§6.3",
        "2500 nodes: seeds' RK4 amplification (min, max), jump-aware, naive; energy at h/8, h/2, 2h and the floor; rightmost real part at 2h",
        [
            min(rk),
            max(rk),
            spectrum(fs, "jump", "aware", "rk4_max"),
            spectrum(fs, "jump", "naive", "rk4_max"),
        ]
        + [spectrum(fs, w, "seeds", "energy_ratio") for w in W]
        + [
            spectrum(fs, "uniform", "floor", "energy_ratio"),
            spectrum(fs, "2h", "seeds", "max_re_hyper"),
        ],
        [
            "1.0003",
            "1.0004",
            "1.0002",
            "1.00001",
            "0.995",
            "1.003",
            "1.012",
            "0.925",
            "0.1",
        ],
    )
    return C


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quiet", action="store_true", help="print failures only")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DATA,
        help="results cache to check (default paper/data)",
    )
    args = ap.parse_args(argv)
    load(args.data_dir)
    checks = build_checks()
    failed = 0
    for ok, section, what, value, quoted, kind in checks:
        if ok and args.quiet:
            continue
        flag = "ok  " if ok else "FAIL"
        print(f"{flag} {section:<16} {what}: {value:.4g} {kind} {quoted}")
        failed += not ok
    print(f"{len(checks) - failed} of {len(checks)} checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
