#!/usr/bin/env python3
# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Worked examples for the aggmetrics paper.

Each block below appears in the paper's "Illustrative examples" section, and the
printed output is what the paper reports. Run with:

    python examples/quickstart.py
"""
from __future__ import annotations

import numpy as np

from aggmetrics import (IndicatorMatrix, diagnose, get_measure, list_measures,
                         marginal_defect, micro_macro_gap)

SEP = "=" * 74


# ---------------------------------------------------------------------------
def example_1_one_measure_three_operators():
    print(SEP); print("1. One binary measure, three aggregation operators"); print(SEP)
    rng = np.random.default_rng(0)
    n, m = 400, 6
    Y = (rng.random((n, m)) < 0.3).astype(int)
    P = Y.copy()
    flip = rng.random((n, m)) < 0.2
    P[flip] = 1 - P[flip]

    im = IndicatorMatrix(Y, P, task="multilabel")
    print(f"  {im}\n")
    print(f"  {'measure':20s} {'micro':>9s} {'macro':>9s} {'exemplar':>9s}")
    for name in ("precision", "recall", "f1", "jaccard", "mcc",
                 "balanced_accuracy", "informedness"):
        vals = [im.compute(name, a) for a in ("micro", "macro", "exemplar")]
        print(f"  {name:20s} " + " ".join(f"{v:9.4f}" for v in vals))
    print("\n  Note that macro-MCC and micro-informedness have no standard named")
    print("  counterpart in the literature; they require no extra code here.")


# ---------------------------------------------------------------------------
def example_2_decomposability():
    print("\n" + SEP); print("2. Which measures admit a weighted-averaging identity")
    print(SEP)
    dec = list_measures(decomposable=True)
    non = list_measures(decomposable=False)
    print(f"  decomposable ({len(dec)}):     {', '.join(dec)}")
    print(f"  not decomposable ({len(non)}): {', '.join(non)}")

    # the counterexample from the companion paper
    CM = np.array([[110, 10, 10], [20, 40, 5], [25, 5, 50]])
    y, yhat = [], []
    for j in range(3):
        for k in range(3):
            y += [j] * CM[j, k]; yhat += [k] * CM[j, k]
    im = IndicatorMatrix(np.array(y), np.array(yhat), task="multiclass")

    print(f"\n  On a 3-class problem with n = {im.n}:")
    print(f"  {'measure':20s} {'per-class range':>22s} {'micro':>9s}  in range?")
    for name in ("recall", "specificity", "balanced_accuracy", "gmean", "mcc"):
        parts = im.parts(name, "macro")
        micro = im.compute(name, "micro")
        lo, hi = np.nanmin(parts), np.nanmax(parts)
        inside = lo - 1e-12 <= micro <= hi + 1e-12
        print(f"  {name:20s} {f'[{lo:.4f}, {hi:.4f}]':>22s} {micro:9.4f}"
              f"  {'yes' if inside else 'NO'}")
    print("\n  A value outside the per-class range cannot be any weighted average")
    print("  of them, so no weighting interpretation of micro-averaging exists.")

    g = micro_macro_gap(im, "recall")
    print(f"\n  For recall the gap is exactly a normalised covariance: "
          f"gap = {g['gap']:.6f}, predicted = {g['predicted_gap']:.6f}")
    g = micro_macro_gap(im, "balanced_accuracy")
    print(f"  For balanced accuracy the identity does not apply: exact={g['exact']}")


# ---------------------------------------------------------------------------
def example_3_diagnostics():
    print("\n" + SEP); print("3. Diagnostics on a ten-class problem"); print(SEP)
    m = 10
    y = np.arange(500) % m
    yhat = (y + 1) % m                      # every prediction wrong
    im = IndicatorMatrix(y, yhat, task="multiclass")
    print(f"  Every one of the {im.n} predictions is wrong. Reported values:")
    for name in ("label_accuracy", "specificity", "balanced_accuracy", "mcc",
                 "kappa", "f1", "gmean"):
        print(f"    {name:20s} {im.compute(name, 'micro'):8.4f}")
    print(f"    {'accuracy':20s} {im.accuracy():8.4f}   (standard, = TP/n)")

    print("\n  Diagnostics:")
    for f in diagnose(im, measures=["label_accuracy", "mcc", "recall", "f1"]):
        print(f"    [{f.severity:7s}] {f.code}")
        print(f"              {f.message}")


# ---------------------------------------------------------------------------
def example_4_thresholds():
    print("\n" + SEP); print("4. Which measures standard training actually targets")
    print(SEP)
    print(f"  {'measure':20s} {'Bayes threshold':>16s} {'ERM-consistent':>15s}")
    for name in ("label_accuracy", "error_rate", "recall", "specificity",
                 "precision", "f1", "mcc"):
        meas = get_measure(name)
        t = meas.bayes_threshold()
        c = meas.erm_consistent()
        ts = "n/a (no closed form)" if t is None else f"{t:.4f}"
        cs = "n/a" if c is None else str(c)
        print(f"  {name:20s} {ts:>16s} {cs:>15s}")
    print("\n  Recall is maximised at threshold 0 (predict everything positive)")
    print("  and specificity at 1 (predict everything negative). Neither is the")
    print("  rule that minimising 0/1 loss produces.")


# ---------------------------------------------------------------------------
def example_5_soft_labels():
    print("\n" + SEP); print("5. Soft ground truth and the choice of t-norm"); print(SEP)
    rng = np.random.default_rng(2)
    Y = rng.random((200, 4))            # annotator proportions
    P = np.clip(Y + rng.normal(0, 0.2, Y.shape), 0, 1)

    print(f"  {'t-norm':14s} {'marginal defect':>16s} {'micro-F1':>10s} "
          f"{'exemplar-F1':>12s}")
    for t in ("godel", "product", "lukasiewicz"):
        im = IndicatorMatrix(Y, P, task="multilabel", tnorm=t)
        print(f"  {t:14s} {marginal_defect(t):16.4f} "
              f"{im.compute('f1', 'micro'):10.4f} "
              f"{im.compute('f1', 'exemplar'):12.4f}")
    print("\n  Only the product t-norm preserves the marginals, so only under it do")
    print("  the four fuzzy cells behave like a contingency table.")


# ---------------------------------------------------------------------------
def example_6_cost_and_ordinal():
    print("\n" + SEP); print("6. Ordinal losses as cost matrices"); print(SEP)
    m, n = 5, 300
    rng = np.random.default_rng(4)
    y = rng.integers(0, m, n)
    yhat = np.clip(y + rng.integers(-2, 3, n), 0, m - 1)
    im = IndicatorMatrix(y, yhat, task="multiclass")
    J, K = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")

    mae = im.cost(np.abs(J - K)) / n
    mse = im.cost((J - K) ** 2.0) / n
    print(f"  cost with C[j,k] = |j-k|   / n = {mae:.6f}"
          f"   (direct MAE = {np.abs(y - yhat).mean():.6f})")
    print(f"  cost with C[j,k] = (j-k)^2 / n = {mse:.6f}"
          f"   (direct MSE = {((y - yhat) ** 2).mean():.6f})")
    print("\n  MAE and MSE are not separate evaluation paradigms but instances of")
    print("  the cost-sensitive construction with particular cost matrices.")




# ---------------------------------------------------------------------------
def example_7_generative():
    """The generative claim as an operation: supply a formula, get the properties."""
    print("\n" + SEP); print("7. Registering a new measure: properties are derived")
    print(SEP)
    from aggmetrics import infer_properties, register_measure, verify_inference

    print("  (a) Inference reproduces every analytic annotation in the registry:")
    rep = verify_inference()
    bad = [k for k, v in rep.items() if not v["agrees"]]
    errs = [v["coeff_max_error"] for v in rep.values() if "coeff_max_error" in v]
    print(f"      {len(rep) - len(bad)}/{len(rep)} measures agree; "
          f"recovered coefficients accurate to {max(errs):.1e}")

    print("\n  (b) A measure the registry does not contain: prevalence threshold.")

    def prevalence_threshold(tp, fp, fn, tn):
        import math
        r = tp / (tp + fn) if (tp + fn) else float("nan")
        s = tn / (tn + fp) if (tn + fp) else float("nan")
        if math.isnan(r) or math.isnan(s):
            return float("nan")
        d = r + s - 1.0
        return (math.sqrt(r * (1 - s)) + s - 1) / d if abs(d) > 1e-12 else float("nan")

    print(infer_properties(prevalence_threshold).summary()
          .replace("\n", "\n      ").rjust(6))
    m = register_measure("prevalence_threshold", prevalence_threshold,
                         overwrite=True)
    rng = np.random.default_rng(1)
    Y = (rng.random((200, 5)) < 0.4).astype(int)
    P = Y.copy(); flip = rng.random(Y.shape) < 0.2; P[flip] = 1 - P[flip]
    im = IndicatorMatrix(Y, P, task="multilabel")
    print("\n      now available under every operator:")
    for a in ("micro", "macro", "exemplar"):
        print(f"        {a + '-prevalence_threshold':32s} {im.compute(m, a):.4f}")

    print("\n  (c) A measure of the admissible form inherits the identities:")

    def tp_minus_fp_rate(tp, fp, fn, tn):
        d = tp + fp + fn + tn
        return (tp - fp) / d if d else float("nan")

    m2 = register_measure("tp_minus_fp_rate", tp_minus_fp_rate, overwrite=True)
    print(f"      decomposable      : {m2.decomposable}")
    print(f"      Bayes threshold   : {m2.bayes_threshold():.4f}   "
          f"(derived from the recovered coefficients)")
    print(f"      ERM-consistent    : {m2.erm_consistent()}")
    print(f"      floor at m = 10   : {m2.attainable_floor(10):.4f}")
    print("\n      None of these were supplied; all four were computed from the"
          "\n      formula. This is what the generative claim amounts to in code.")


if __name__ == "__main__":
    example_1_one_measure_three_operators()
    example_2_decomposability()
    example_3_diagnostics()
    example_4_thresholds()
    example_5_soft_labels()
    example_6_cost_and_ordinal()
    example_7_generative()
    print("\n" + SEP)
