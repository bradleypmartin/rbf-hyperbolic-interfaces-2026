"""Results cache for the Part 3 drivers (issue demo#54).

The stiff drivers (``scripts/wave1d_stiff.py``, ``scripts/wave2d_stiff.py``,
``scripts/wave2d_stiff_eigenvalues.py``) cache their *inputs* under
``outputs/`` (spectral references, seed operators) but until demo#54 printed
their *results* to the terminal only, so the convergence tables lived in
``docs/stiff-features.md`` alone. This module gives every driver one JSON
file of records that the manuscript's figures and tables regenerate from
without the sweeps (``scripts/paper_figures.py``), and that the assembly
pass (demo#61) can script its number check against.

Schema (``SCHEMA`` below), one file per driver invocation::

    {
      "schema":  "pdes-demo/stiff-results/1",
      "script":  "scripts/wave2d_stiff.py",
      "args":    {...},                 # the parsed argparse namespace
      "date":    "2026-09-20",
      "git_sha": "ccc31ee",             # HEAD when the run started, if known
      "records": [ {...}, ... ]
    }

Every record carries ``kind`` and the keys of its kind:

``error``
    ``n, h, delta, mode, field, error, rate``. One per (resolution, edge
    width, scheme, field). ``rate`` is the convergence order between this
    ``n`` and the previous one in the sweep (``None`` at the first ``n``),
    in ``h`` (the 2-D drivers' ``2 log(e_prev / e) / log(n / n_prev)``).
    ``mode`` is the scheme: ``naive``, ``aware`` (the seeds; the
    interface-aware stencils at ``delta = 0``), ``ablate``, ``floor`` (the
    naive scheme in the uniform medium) or ``sfloor`` (the seed operator
    through a 1e-6 contrast). ``field`` is ``f`` in 1-D; ``v``, ``h``,
    ``u`` (a relative error) or ``max_u`` (the largest spurious ``|u|``
    where the true ``u`` is zero) in 2-D.
``truncation``
    ``n, h, delta, mode, group, error``: the operator's relative truncation
    error on the reference state, per row group (``edge`` or ``bulk``).
``snapshot``
    ``n, delta, t, mode, field, error``: the still's numbers.
``spectrum``
    ``n, h, delta, width_label, variant, rows, dt, gamma, max_re,
    max_re_hyper, min_re_hyper, rk4_max`` and, from a ``--run``,
    ``energy_ratio, max_u, err_v`` (``None`` when not measured).

Records are plain dicts so the schema can grow by adding keys; readers
select on the keys they know (:meth:`ResultsCache.select`).

The ``schema`` string names the file format, not an import path, so it keeps
the ``pdes-demo`` prefix of the talk repository this package was split out
of (as ``pdes_demo``; now ``rbf_hyperbolic_interfaces``). :meth:`ResultsCache.read`
rejects any other string, and changing it would rewrite every committed JSON
in ``paper/data/`` for no change in content.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "pdes-demo/stiff-results/1"

KINDS = {
    "error": ("n", "h", "delta", "mode", "field", "error", "rate"),
    "truncation": ("n", "h", "delta", "mode", "group", "error"),
    "snapshot": ("n", "delta", "t", "mode", "field", "error"),
    "spectrum": (
        "n",
        "h",
        "delta",
        "width_label",
        "variant",
        "rows",
        "dt",
        "gamma",
        "max_re",
        "max_re_hyper",
        "min_re_hyper",
        "rk4_max",
        "energy_ratio",
        "max_u",
        "err_v",
    ),
}


def git_sha(cwd: Path | None = None) -> str | None:
    """Short SHA of HEAD, or ``None`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def json_ready(value: Any) -> Any:
    """``value`` with Paths, tuples, sets and numpy scalars made JSON-native."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if hasattr(value, "item") and callable(value.item):  # numpy scalar
        return value.item()
    if hasattr(value, "__dict__") and not isinstance(value, type):  # Namespace
        return json_ready(vars(value))
    return value


@dataclass
class ResultsCache:
    """One driver run's records plus the provenance the schema asks for."""

    script: str
    args: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    date: str = field(default_factory=lambda: _dt.date.today().isoformat())
    git_sha: str | None = field(default_factory=git_sha)
    schema: str = SCHEMA

    @classmethod
    def new(cls, script: str, args: Any) -> ResultsCache:
        return cls(script=script, args=json_ready(args))

    def add(self, kind: str, **record: Any) -> dict[str, Any]:
        """Append one record of ``kind``; missing keys of the kind are ``None``."""
        if kind not in KINDS:
            raise ValueError(f"unknown record kind {kind!r}; known: {sorted(KINDS)}")
        unknown = set(record) - set(KINDS[kind])
        if unknown:
            raise ValueError(f"{kind} record has unknown keys {sorted(unknown)}")
        full = {"kind": kind}
        for key in KINDS[kind]:
            full[key] = json_ready(record.get(key))
        self.records.append(full)
        return full

    def add_errors(
        self,
        ns: list[int],
        errors: list[float],
        rates: list[float],
        *,
        h: list[float],
        **common: Any,
    ) -> None:
        """``error`` records for one (delta, mode, field) sweep over ``ns``.

        ``rates`` has one entry fewer than ``ns`` (between consecutive
        resolutions); the first record's rate is ``None``.
        """
        if len(errors) != len(ns) or len(h) != len(ns) or len(rates) != len(ns) - 1:
            raise ValueError("ns, errors and h must align; rates has one fewer")
        for i, n in enumerate(ns):
            self.add(
                "error",
                n=int(n),
                h=float(h[i]),
                error=float(errors[i]),
                rate=None if i == 0 else float(rates[i - 1]),
                **common,
            )

    def select(self, kind: str | None = None, **filters: Any) -> list[dict[str, Any]]:
        """Records matching every filter (equality; ``None`` matches ``None``),
        sorted by ``n`` where present."""
        out = []
        for rec in self.records:
            if kind is not None and rec["kind"] != kind:
                continue
            if all(rec.get(k) == v for k, v in filters.items()):
                out.append(rec)
        return sorted(out, key=lambda r: (r.get("n") or 0, r.get("t") or 0))

    def values(self, key: str, kind: str | None = None, **filters: Any) -> list[Any]:
        """Distinct values of ``key`` among the matching records, in order."""
        seen: dict[Any, None] = {}
        for rec in self.select(kind, **filters):
            seen.setdefault(rec.get(key))
        return list(seen)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: d[k] for k in ("schema", "script", "args", "date", "git_sha")} | {
            "records": d["records"]
        }

    def write(self, *paths: Path) -> None:
        """Write the same JSON to every path (``outputs/`` and ``paper/data/``)."""
        text = json.dumps(self.to_dict(), indent=1, sort_keys=False) + "\n"
        for path in paths:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    @classmethod
    def read(cls, path: Path) -> ResultsCache:
        data = json.loads(Path(path).read_text())
        if data.get("schema") != SCHEMA:
            raise ValueError(f"{path}: schema {data.get('schema')!r} != {SCHEMA!r}")
        return cls(
            script=data["script"],
            args=data.get("args", {}),
            records=data.get("records", []),
            date=data.get("date", ""),
            git_sha=data.get("git_sha"),
            schema=data["schema"],
        )
