# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Sizing a temporal operator before deployment.

How often a class is undefined in a window, how long the queue of unlabelled
instances can grow, how much a late label is worth, and how many independent
observations a faded estimate is worth, as functions of the window ``W``, the fading
factor ``alpha``, the class probability ``p`` and the latency. Each function states
the result it computes and the assumptions it needs; none touches a stream.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

import numpy as np

__all__ = [
    "memory_horizon", "effective_sample_size", "half_life", "fading_variance",
    "eviction_age", "queue_bound", "eviction_error_bound",
    "undefined_window_probability", "undefined_fading_bounds", "required_window",
    "kappa_ar1", "variance_inflation", "estimate_kappa",
]


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def _check_prob(p: float, name: str = "p") -> None:
    if not 0.0 < p <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")


def memory_horizon(alpha: float) -> float:
    """``n_eff = 1 / (1 - alpha)``: the mean age of the fading weights, plus one."""
    _check_alpha(alpha)
    return 1.0 / (1.0 - alpha)


def half_life(alpha: float) -> float:
    """Steps after which a weight halves, ``ln 2 / ln(1/alpha)`` (about
    ``0.693 * memory_horizon(alpha)``). A sliding window forgets after exactly
    ``W`` steps instead."""
    _check_alpha(alpha)
    return math.log(2.0) / math.log(1.0 / alpha)


def effective_sample_size(alpha: float, kappa: float = 1.0) -> float:
    """``(1 + alpha) / (1 - alpha) / kappa``: the number of independent
    observations a fading estimate is worth. ``kappa`` is the variance inflation
    factor of dependent entries (:func:`variance_inflation`); 1 for independent
    entries."""
    _check_alpha(alpha)
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    return (1.0 + alpha) / (1.0 - alpha) / kappa


def fading_variance(alpha: float, v: float, t: Optional[int] = None,
                    kappa: float = 1.0) -> float:
    """Variance of the faded mean of entries with variance ``v``.

    Independent entries and ``t`` steps: ``v (1-alpha)(1-alpha^{2t}) / (1+alpha)``.
    Without ``t``, the limit ``v (1-alpha) / (1+alpha)`` times ``kappa``, the
    asymptotic variance of stationary dependent entries. The finite-``t`` value is
    given for independent entries only.
    """
    _check_alpha(alpha)
    if t is None:
        return v * (1.0 - alpha) / (1.0 + alpha) * kappa
    if kappa != 1.0:
        raise ValueError("a finite-t variance is available for independent "
                         "entries only (kappa = 1)")
    if t < 1:
        raise ValueError("t must be at least 1")
    return v * (1.0 - alpha) * (1.0 - alpha ** (2 * t)) / (1.0 + alpha)


def eviction_age(alpha: float, eps: float, strict: bool = True) -> int:
    """``s*``: the first age at which the normalised weight ``(1-alpha) alpha^s``
    of an instance falls below ``eps`` (``strict``, the rule of the evaluator's
    eviction) or does not exceed it (``strict=False``, the rule that defines the
    support bounds). The two differ only when the weight equals ``eps``."""
    _check_alpha(alpha)
    if eps <= 0:
        raise ValueError("eps must be positive")

    def hit(s: int) -> bool:
        w = (1.0 - alpha) * alpha ** s
        return w < eps if strict else w <= eps

    # start one step below the real root: the weight decreases, so the first hit
    # above that start is the first hit
    root = math.log(eps / (1.0 - alpha)) / math.log(alpha) if eps < 1.0 - alpha else 0
    s = max(0, math.floor(root) - 1)
    while not hit(s):
        s += 1
    return s


def queue_bound(alpha: float, eps: float) -> int:
    """Largest number of instances the eviction queue can hold, ``s*``, at any time,
    independent of ``t`` and of how many instances are never labelled. At most
    ``n_eff ln(1 / (n_eff eps)) + 2`` when ``n_eff * eps < 1``."""
    return eviction_age(alpha, eps, strict=True)


def eviction_error_bound(alpha: float, eps: float) -> float:
    """``alpha ** s*``: the share of the normalised state held by the entries that
    can no longer be corrected, an exact geometric tail, below ``eps * n_eff``."""
    return float(alpha ** eviction_age(alpha, eps, strict=True))


DelaySpec = Union[int, Callable[[int], float], None]


def _delay_cdf(delay: DelaySpec) -> Callable[[int], float]:
    if delay is None:
        return lambda s: 1.0
    if callable(delay):
        return delay
    if int(delay) != delay or delay < 0:
        raise ValueError("a constant delay must be a non-negative integer")
    d = int(delay)
    return lambda s: 1.0 if s >= d else 0.0


def undefined_window_probability(p: float, window: int, *, delay: DelaySpec = None,
                                 audit: float = 1.0) -> float:
    """Probability that a class has no verified instance in a sliding window.

    Class ``j`` arrives i.i.d. with probability ``p``, each instance is audited
    with probability ``audit`` (``b``) and its label reaches the evaluator after a
    delay with distribution function ``F``. Then the probability that the class is
    undefined (the macro measures that divide by its support) is

        prod_{s=0}^{W-1} (1 - p b F(s)),

    where ``s`` is the age of the instance. Without delay, ``F = 1`` and with
    ``b = 1`` this is ``(1 - p)^W``. For a constant delay ``D < W`` it is
    ``(1 - p b)^(W - D)``, and it is 1 if ``D >= W``.

    ``delay`` is ``None`` (immediate), a non-negative integer (constant delay), or
    a callable ``F(s)`` (the distribution function of the delay at age ``s``).
    """
    _check_prob(p)
    if window < 1 or int(window) != window:
        raise ValueError("window must be a positive integer")
    if not 0.0 <= audit <= 1.0:
        raise ValueError("audit must be in [0, 1]")
    F = _delay_cdf(delay)
    return float(np.prod([1.0 - p * audit * F(s) for s in range(int(window))]))


def undefined_fading_bounds(p: float, alpha: float, eps: float):
    """Bounds on the probability that a class has no support under fading.

    With tolerance ``eps`` on the normalised mass and ``t >= s*``,

        (1-p)^{s_eps} <= P(u_j = 0) <= (1-p)^{s*},

    where ``s* = eviction_age(alpha, eps, strict=False)`` is the first age at
    which the weight of one arrival does not exceed ``eps``, and
    ``s_eps = ceil(ln(1/eps) / ln(1/alpha))`` is the number of silent steps after
    which the whole mass of the class is at most ``eps``. Returns ``(lower,
    upper)``. Needs independent arrivals and no latency.
    """
    _check_prob(p)
    _check_alpha(alpha)
    s_star = eviction_age(alpha, eps, strict=False)
    s_eps = math.ceil(math.log(1.0 / eps) / math.log(1.0 / alpha))
    return float((1.0 - p) ** s_eps), float((1.0 - p) ** s_star)


def required_window(p: float, delta: float, *, delay: int = 0,
                    audit: float = 1.0) -> int:
    """Smallest window for which a class is undefined with probability at most
    ``delta``: ``D + ln(1/delta) / (-ln(1 - p b))``, rounded up. The window needed
    grows by the constant delay ``D``. Valid for a constant delay."""
    _check_prob(p)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not 0.0 < audit <= 1.0:
        raise ValueError("audit must be in (0, 1]")
    if int(delay) != delay or delay < 0:
        raise ValueError("delay must be a non-negative integer")
    q = p * audit
    if q >= 1.0:
        return int(delay) + 1
    return int(delay) + math.ceil(math.log(1.0 / delta) / -math.log(1.0 - q))


def kappa_ar1(alpha: float, rho: float) -> float:
    """``(1 + alpha rho) / (1 - alpha rho)``: the variance inflation factor of AR(1)
    entries, whose autocorrelation is ``rho^d`` at lag ``d``."""
    _check_alpha(alpha)
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be in (-1, 1)")
    return (1.0 + alpha * rho) / (1.0 - alpha * rho)


def variance_inflation(alpha: float, acf: Sequence[float]) -> float:
    """``kappa(alpha) = 1 + 2 sum_{d>=1} alpha^d rho_d`` from autocorrelations.

    ``acf[d]`` is the autocorrelation at lag ``d`` (``acf[0]`` is the lag-0 value,
    1, and is ignored). The asymptotic variance of the faded mean of stationary
    entries with variance ``v`` is ``v (1-alpha)/(1+alpha) kappa``, so the effective
    sample size is divided by ``kappa``. The sum is truncated at the lags given; it
    converges geometrically because ``|alpha^d rho_d| <= alpha^d``.
    """
    _check_alpha(alpha)
    rho = np.asarray(acf, float)
    if rho.ndim != 1 or len(rho) < 2:
        raise ValueError("acf must hold the autocorrelations at lags 0, 1, ...")
    d = np.arange(1, len(rho))
    return float(1.0 + 2.0 * np.sum(alpha ** d * rho[1:]))


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()
    v = x.var()
    if v <= 0:
        return np.zeros(nlags)
    f = np.fft.rfft(x, 2 * len(x))
    a = np.fft.irfft(f * np.conj(f))[:nlags] / len(x)
    return a / v


def estimate_kappa(x, alpha: float, *, max_lag: Optional[int] = None,
                   detrend: Optional[int] = None) -> float:
    """Estimate ``kappa(alpha)`` from a sequence of entries (for instance the 0/1
    indicator that a prediction is correct, or the correctness of the arrivals of
    one class).

    ``max_lag`` defaults to ``3 / (1 - alpha)`` capped at a third of the sequence.
    ``detrend`` is the length of a moving average removed first. The factor
    measures any variation of the mean that remains in the sequence, so on a stream
    with real drift it includes that drift. The estimate is floored at 1.
    """
    _check_alpha(alpha)
    s = np.asarray(x, float).ravel()
    if detrend:
        k = np.ones(detrend) / detrend
        s = (s - np.convolve(s, k, mode="same"))[detrend:-detrend]
    if len(s) < 8:
        raise ValueError("the sequence is too short")
    L = max_lag if max_lag is not None else int(3 / (1 - alpha))
    L = int(min(L, max(2, len(s) // 3)))
    rho = _acf(s, L)
    return max(1.0, 1.0 + 2.0 * float(np.sum(alpha ** np.arange(1, L) * rho[1:L])))
