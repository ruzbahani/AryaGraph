# =============================================================================
#
#      _                     ____                 _
#     / \   _ __ _   _  __ _ / ___|_ __ __ _ _ __ | |__
#    / _ \ | '__| | | |/ _` | |  _| '__/ _` | '_ \| '_ \
#   / ___ \| |  | |_| | (_| | |_| | | | (_| | |_) | | | |
#  /_/   \_\_|   \__, |\__,_|\____|_|  \__,_| .__/|_| |_|
#               |___/                     |_|
#
#          Graph & DAG visualization, analysis and simulation.
#
# -----------------------------------------------------------------------------
#  Copyright (c) 2026 Ali Mohammadi Ruzbahani
#  SPDX-License-Identifier: MIT
#
#  https://ruzbahani.com/aryagraph
# =============================================================================

"""Ensembles: many independent runs of a stochastic simulation, summarised.

>>> ens = aryagraph.sim.run_ensemble(aryagraph.sim.sir, runs=200, g=g, beta=0.3, gamma=0.1,
...                              method="gillespie", seed=7)
>>> ens.mean["I"], ens.quantiles["I"][95]      # mean curve and upper band
>>> ens.final_sizes["R"].mean()                # expected final epidemic size
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import SimulationResult, _seed_record


@dataclass(repr=False)
class EnsembleResult:
    """Counts of each state across runs, on a common time grid.

    Attributes
    ----------
    times:
        ``(P,)`` common grid, ``linspace(0, longest run, n_points)``.
    samples:
        ``(runs, P, S)`` count of nodes in each state for every run.
    mean:
        ``{state: (P,) array}`` holding the mean count of nodes in each state
        over the runs, at every grid time.
    std:
        ``{state: (P,) array}`` holding the standard deviation of that count
        over the runs; ``std`` uses ``ddof=1`` (0 for a single run).
    quantiles:
        ``{state: {q: (P,) array}}`` for each requested percentile *q*.
    final_sizes:
        ``{state: (runs,) int array}`` holding each run's count at its last frame.
    seeds:
        The integer seed of every run: ``fn(..., seed=seeds[i])`` replays run *i*.
    results:
        The individual results when ``keep=True``, else ``None``.
    """

    times: np.ndarray
    states: list[str]
    roles: dict[str, str]
    samples: np.ndarray
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    quantiles: dict[str, dict[float, np.ndarray]]
    final_sizes: dict[str, np.ndarray]
    runs: int
    n_nodes: int
    model: str
    params: dict
    seed: int | None
    seeds: list[int]
    results: list[SimulationResult] | None = None

    def fractions(self) -> dict[str, np.ndarray]:
        """Mean share of nodes in each state over the grid."""
        n = self.n_nodes
        return {s: (m / n if n else np.zeros_like(m)) for s, m in self.mean.items()}

    def band(self, state: str, lo: float = 5, hi: float = 95) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper quantile curves of *state* (both must have been requested)."""
        q = self.quantiles[state]
        return q[lo], q[hi]

    def plot(self, **kwargs: Any):
        """Mean curves with quantile bands; see :func:`aryagraph.charts.sim.plot_ensemble`."""
        from ..charts.sim import plot_ensemble

        return plot_ensemble(self, **kwargs)

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{s}={v.mean():.4g}±{(v.std(ddof=1) if v.size > 1 else 0.0):.3g}" for s, v in self.final_sizes.items()
        )
        return f"<EnsembleResult {self.model}: {self.runs} runs, {self.n_nodes} nodes; final {parts}>"


def run_ensemble(
    fn: Callable[..., SimulationResult],
    runs: int = 100,
    *,
    seed: int | None = None,
    n_points: int = 101,
    quantiles: Sequence[float] = (5, 25, 50, 75, 95),
    interpolation: str = "step",
    keep: bool = False,
    **kwargs: Any,
) -> EnsembleResult:
    """Run ``fn(**kwargs, seed=...)`` *runs* times and summarise the state counts.

    Parameters
    ----------
    fn:
        Any function returning a categorical :class:`SimulationResult` and taking
        a ``seed`` keyword, such as :func:`aryagraph.sim.sir`,
        ``SIR(0.3, 0.1).simulate`` or :func:`aryagraph.sim.independent_cascade`.
    runs:
        Number of independent runs.
    seed:
        Master seed. Child seeds come from ``numpy.random.SeedSequence(seed).spawn``
        (as 64-bit integers, so every run can be replayed on its own), which
        makes the runs statistically independent and the ensemble reproducible.
    n_points:
        Size of the common time grid, which spans the longest run.
    interpolation:
        ``"step"`` (default) holds each count until the next frame, which is
        exact for jump processes such as Gillespie runs or discrete steps;
        ``"linear"`` interpolates between frames. After its last frame a run
        keeps its final counts (absorbing states persist).
    keep:
        Also return every individual result (memory grows with *runs*).
    **kwargs:
        Passed to *fn* (``g=...`` for the graph).
    """
    runs = int(runs)
    if runs < 1:
        raise ValueError("runs must be at least 1")
    n_points = int(n_points)
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if interpolation not in ("step", "linear"):
        raise ValueError(f"interpolation must be 'step' or 'linear', got {interpolation!r}")
    children = np.random.SeedSequence(seed).spawn(runs)
    seeds = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in children]

    kept: list[SimulationResult] = []
    series: list[tuple[np.ndarray, np.ndarray]] = []  # (times, (T, S) counts) per run
    states: list[str] = []
    first: SimulationResult | None = None
    for s in seeds:
        res = fn(**kwargs, seed=s)
        if not isinstance(res, SimulationResult) or not res.categorical:
            raise TypeError("run_ensemble needs a function returning a categorical SimulationResult")
        if first is None:
            first, states = res, list(res.states)
        elif res.states != states:
            raise ValueError("runs disagree on the list of states")
        counts = np.stack([np.count_nonzero(res.values == c, axis=1) for c in range(len(states))], axis=1)
        series.append((res.times, counts))
        if keep:
            kept.append(res)
    assert first is not None

    t_end = max(float(times[-1]) for times, _ in series)
    grid = np.linspace(0.0, t_end, n_points) if t_end > 0 else np.zeros(1)
    samples = np.empty((runs, grid.size, len(states)))
    finals = np.empty((runs, len(states)), dtype=np.int64)
    for k, (times, counts) in enumerate(series):
        finals[k] = counts[-1]
        if interpolation == "step":
            samples[k] = counts[np.searchsorted(times, grid, side="right") - 1]
        else:
            for c in range(len(states)):
                samples[k, :, c] = np.interp(grid, times, counts[:, c])

    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1) if runs > 1 else np.zeros_like(mean)
    qs = [float(q) for q in quantiles]
    qv = np.percentile(samples, qs, axis=0) if qs else np.empty((0,) + mean.shape)
    keys = [int(q) if q == int(q) else q for q in qs]
    params = dict(first.params)
    params.pop("seed", None)
    return EnsembleResult(
        times=grid,
        states=states,
        roles=dict(first.roles),
        samples=samples,
        mean={s: mean[:, c] for c, s in enumerate(states)},
        std={s: std[:, c] for c, s in enumerate(states)},
        quantiles={s: {q: qv[j, :, c] for j, q in enumerate(keys)} for c, s in enumerate(states)},
        final_sizes={s: finals[:, c] for c, s in enumerate(states)},
        runs=runs,
        n_nodes=first.N,
        model=first.model,
        params=params,
        seed=_seed_record(seed),
        seeds=seeds,
        results=kept if keep else None,
    )


__all__ = ["run_ensemble", "EnsembleResult"]
