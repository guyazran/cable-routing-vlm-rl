"""Prespecified statistics for the horizon screen (protocol Sections 11-12).

Everything here operates on per-episode records and is independent of the
simulator, so it can be unit-tested before any episode exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Single proportions
# ---------------------------------------------------------------------------

def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Paired five-vs-ten comparison
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedTable:
    """2x2 table of paired outcomes; index 1 = success. n10: five succeeds, ten fails."""

    n11: int
    n10: int
    n01: int
    n00: int

    @property
    def n(self) -> int:
        return self.n11 + self.n10 + self.n01 + self.n00

    @property
    def decrease(self) -> float:
        """SR(5) - SR(10) estimated from the pairs."""
        return (self.n10 - self.n01) / self.n if self.n else float("nan")


def paired_table(five: Sequence[bool], ten: Sequence[bool]) -> PairedTable:
    a = np.asarray(five, dtype=bool)
    b = np.asarray(ten, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("five and ten outcomes must be paired by scene")
    return PairedTable(
        n11=int(np.sum(a & b)),
        n10=int(np.sum(a & ~b)),
        n01=int(np.sum(~a & b)),
        n00=int(np.sum(~a & ~b)),
    )


def exact_mcnemar(table: PairedTable) -> float:
    """Two-sided exact McNemar p-value (binomial test on discordant pairs)."""
    m = table.n10 + table.n01
    if m == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(table.n10, table.n01), m, 0.5)))


def discordance_difference_ci(table: PairedTable, alpha: float = 0.05) -> tuple[float, float]:
    """Conservative CI for p10 - p01 using two (1 - alpha/2) Clopper-Pearson intervals.

    Bonferroni coverage is at least 1 - alpha (protocol Section 12).
    """
    lo10, hi10 = clopper_pearson(table.n10, table.n, alpha / 2)
    lo01, hi01 = clopper_pearson(table.n01, table.n, alpha / 2)
    return (lo10 - hi01, hi10 - lo01)


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci: tuple[float, float]
    degenerate: bool  # all resamples identical, e.g. no outcome variation


def interaction_bootstrap(
    vlm5: Sequence[bool],
    vlm10: Sequence[bool],
    ref5: Sequence[bool],
    ref10: Sequence[bool],
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Paired scene bootstrap for I = Delta_VLM - Delta_REF.

    All four outcomes of a scene are resampled together.
    """
    x = np.stack([np.asarray(v, dtype=float) for v in (vlm5, vlm10, ref5, ref10)], axis=1)
    n = x.shape[0]

    def interaction(rows: np.ndarray) -> np.ndarray:
        m = rows.mean(axis=-2)
        return (m[..., 0] - m[..., 1]) - (m[..., 2] - m[..., 3])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = interaction(x[idx])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        estimate=float(interaction(x)),
        ci=(float(lo), float(hi)),
        degenerate=bool(np.all(boot == boot[0])),
    )


# ---------------------------------------------------------------------------
# Stage-wise progress
# ---------------------------------------------------------------------------

def progress_survival(first_attained: Sequence[int], horizon: int) -> np.ndarray:
    """S(k) for k = 0..horizon.

    ``first_attained[i]`` is the longest simultaneously valid ordered prefix that
    run i ever reached before termination (not its final prefix).
    """
    a = np.asarray(first_attained)
    return np.array([1.0] + [float(np.mean(a >= k)) for k in range(1, horizon + 1)])


def conditional_stage_failure(survival: np.ndarray) -> list[float | None]:
    """h(k) = 1 - S(k)/S(k-1) for k = 1..N; None where the at-risk set is empty."""
    return [
        None if survival[k - 1] == 0 else float(1 - survival[k] / survival[k - 1])
        for k in range(1, len(survival))
    ]


def prefix_budget_attainment(
    attain_invocation: Sequence[Sequence[int | None]], per_clip_budget: int, horizon: int
) -> np.ndarray:
    """Fraction of runs reaching prefix k within b*k physical skill invocations.

    ``attain_invocation[i][k-1]`` is the 1-based invocation count at which run i
    first held a valid prefix of length k, or None if never.
    """
    out = []
    for k in range(1, horizon + 1):
        hits = [
            run[k - 1] is not None and run[k - 1] <= per_clip_budget * k
            for run in attain_invocation
        ]
        out.append(float(np.mean(hits)) if hits else float("nan"))
    return np.array(out)


# ---------------------------------------------------------------------------
# Design calculation (Section 12 power table)
# ---------------------------------------------------------------------------

def mcnemar_power(n: int, discordance: float, effect: float = 0.20, alpha: float = 0.05) -> float:
    """Power of the two-sided exact McNemar test for a true decrease ``effect``.

    M ~ Binomial(n, d) discordant pairs, n10 | M ~ Binomial(M, (d + effect) / (2d)).
    """
    q = (discordance + effect) / (2 * discordance)
    power = 0.0
    for m in range(n + 1):
        pm = stats.binom.pmf(m, n, discordance)
        if m == 0 or pm == 0:
            continue
        reject = 0.0
        for k in range(m + 1):
            p = min(1.0, 2 * stats.binom.cdf(min(k, m - k), m, 0.5))
            if p <= alpha:
                reject += stats.binom.pmf(k, m, q)
        power += pm * reject
    return power
