# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Monitoring the canonical coordinates (experimental).

The detector is a chi-square on the class-time rates, the conditional rates that a
change of prior does not move, against a reference taken as the average of the
estimates over a warm-up. A label shift does not change them in expectation, so an
alarm points to a change of the concept; the converse does not hold, since a
concept change that leaves the rates alone is not seen.

**This module is experimental and its limits are part of its contract.**

* A per-step chi-square level does not control the false alarm per run: the
  statistic is evaluated at every step on an autocorrelated estimator. The
  threshold must be calibrated by simulation on a stationary stream with the same
  ``alpha``, length and priors (:func:`calibrate_threshold`); the default
  threshold is not a 5% run-level test.
* The variances are binomial (multinomial for more than two classes) and assume
  independent arrivals. On streams in native order the dependence multiplies the
  variance of a rate by ``kappa(alpha)``
  (:func:`aggmetrics.stream.sizing.estimate_kappa`), and the detector does not
  correct for it. In the paper this explains part of the false alarms on real
  streams, and the rest is real drift.
* The reference variance is approximated by ``R_ref (1 - R_ref) / n_ref``, with
  ``n_ref`` the number of warm-up arrivals of the class; the paper's experiments
  use the exact variance of the average of the estimates, which is smaller.
* The power on the recall coordinate is limited by the number of positive arrivals
  after a change, so it is low at a low prior.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .state import ClassTimeState

__all__ = ["RateMonitor", "calibrate_threshold"]


class RateMonitor:
    """Chi-square monitor of the class-time rates against a warm-up reference.

    Parameters
    ----------
    n_classes
        Number of classes.
    alpha
        Fading factor per arrival of each class.
    warmup
        Number of instances, from the first, used to build the reference. The
        reference is the average of the rate estimates over the steps of the warm-up
        after the estimate became defined; the monitor reports ``nan`` until it is
        frozen and while some class has no arrival yet.

    Attributes
    ----------
    statistic
        The chi-square at the last step (``nan`` while undefined).
    max_statistic
        The maximum of the statistic since the reference was frozen: the quantity
        whose null distribution :func:`calibrate_threshold` calibrates.
    """

    def __init__(self, n_classes: int = 2, *, alpha: float = 0.99, warmup: int = 1000):
        if warmup < 1:
            raise ValueError("warmup must be positive")
        self.m = int(n_classes)
        self.state = ClassTimeState(self.m, alpha=alpha)
        self.alpha = alpha
        self.warmup = int(warmup)
        self.t = 0
        self._sum = np.zeros((self.m, self.m))
        self._cnt = np.zeros(self.m)
        self._ref = None
        self._ref_n = None
        self.statistic = float("nan")
        self.max_statistic = float("-inf")

    @property
    def frozen(self) -> bool:
        return self._ref is not None

    def update(self, y_true: int, y_pred: int) -> float:
        """Feed one instance; return the statistic (``nan`` until defined)."""
        self.state.update(y_true, y_pred)
        self.t += 1
        R = self.state.rates()
        seen = ~np.isnan(R[:, 0])
        if self.t <= self.warmup:
            self._sum[seen] += R[seen]
            self._cnt[seen] += 1
            if self.t == self.warmup and seen.all():
                self._ref = self._sum / self._cnt[:, None]
                self._ref_n = self.state.arrivals.astype(float)
            return float("nan")
        if self._ref is None:
            return float("nan")
        self.statistic = self._chi2(R)
        if np.isfinite(self.statistic):
            self.max_statistic = max(self.max_statistic, self.statistic)
        return self.statistic

    def _chi2(self, R: np.ndarray) -> float:
        n = self.state.n_ess()
        if self.m == 2:
            tot = 0.0
            for j in (0, 1):
                r, rr = R[j, j], self._ref[j, j]            # specificity, recall
                var = r * (1 - r) / n[j] + rr * (1 - rr) / self._ref_n[j]
                if var <= 0:
                    return float("nan")
                tot += (r - rr) ** 2 / var
            return float(tot)
        stats = []
        for j in range(self.m):
            off = [k for k in range(self.m) if k != j]
            r, rr = R[j], self._ref[j]
            cov = (np.diag(r) - np.outer(r, r))[np.ix_(off, off)] / n[j] \
                + (np.diag(rr) - np.outer(rr, rr))[np.ix_(off, off)] / self._ref_n[j]
            d = (r - rr)[off]
            cov = cov + 1e-12 * np.eye(len(off))
            stats.append(float(d @ np.linalg.solve(cov, d)))
        return float(max(stats))


def calibrate_threshold(run_null: Callable[[int], float], *, n_runs: int = 200,
                        target: float = 0.05, seed: int = 0) -> float:
    """Threshold with run-level false alarm probability ``target`` on a null stream.

    ``run_null(seed)`` simulates one stationary stream (same ``alpha``, length,
    class priors and rates as the stream to monitor), feeds it to a fresh
    :class:`RateMonitor` and returns its ``max_statistic``. The threshold is the
    ``1 - target`` empirical quantile of the maxima over ``n_runs`` runs, so that
    the probability that a null run ever exceeds it is ``target``. Use calibration
    seeds disjoint from the ones you evaluate on.
    """
    if n_runs < 20:
        raise ValueError("n_runs must be at least 20")
    if not 0.0 < target < 1.0:
        raise ValueError("target must be in (0, 1)")
    maxima = np.array([run_null(seed + i) for i in range(n_runs)], float)
    maxima = maxima[np.isfinite(maxima)]
    if len(maxima) < n_runs // 2:
        raise ValueError("most null runs produced no statistic: lengthen the stream "
                         "or shorten the warm-up")
    return float(np.quantile(maxima, 1.0 - target, method="higher"))
