# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""
Quickstart recipes for aggmetrics.
Demonstrates standard classification, diagnostics, custom metrics, and sklearn integration.
"""
import numpy as np
from aggmetrics import (
    IndicatorMatrix,
    diagnose,
    get_measure,
    infer_properties,
    list_measures,
    micro_macro_gap,
    pareto_dominance,
    register_measure,
)
from aggmetrics.aggregation import fold_aggregate
from aggmetrics.diagnostics import check_redundancy
from aggmetrics.sklearn_compat import make_scorer


def example_multiclass():
    print("=== 1. Multiclass Evaluation ===")
    y_true = np.array([0, 1, 2, 2, 1, 0, 0, 1, 2, 0])
    y_pred = np.array([0, 2, 2, 1, 1, 0, 0, 1, 2, 1])

    im = IndicatorMatrix(y_true, y_pred, task="multiclass")

    print(f"Accuracy:          {im.accuracy():.4f}")
    print(f"Macro F1:          {im.compute('f1', 'macro'):.4f}")
    print(f"Micro F1:          {im.compute('f1', 'micro'):.4f}")
    print(f"Macro MCC:         {im.compute('mcc', 'macro'):.4f}")
    print(f"Balanced Accuracy: {im.compute('balanced_accuracy', 'macro'):.4f}")


def example_diagnostics():
    print("\n=== 2. Diagnostic Checks ===")
    # Intentionally bad predictions on 5 classes
    y_true = np.array([0, 1, 2, 3, 4] * 10)
    y_pred = np.array([(y + 1) % 5 for y in y_true])

    im = IndicatorMatrix(y_true, y_pred, task="multiclass")

    findings = diagnose(im)
    for f in findings:
        print(f"[{f.severity.upper()}] {f.code}: {f.message}")


def example_orthogonality_and_redundancy():
    print("\n=== 3. Orthogonality & Redundancy Analysis ===")
    # Binary classification setup
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])

    im = IndicatorMatrix(y_true, y_pred, task="binary")
    dof = im.degrees_of_freedom
    coords = im.canonical_coordinates()

    print(f"Degrees of Freedom (d): {dof}")
    print(f"Canonical Orthogonal Basis (Recall, Specificity): {coords}")

    # Check redundancy in a bloated metric suite
    bloated_suite = ["accuracy", "balanced_accuracy", "f1", "recall", "specificity", "precision"]
    f = check_redundancy(im, bloated_suite)
    if f:
        print(f"[{f.severity.upper()}] {f.code}: {f.message}")


def example_pareto_dominance():
    print("\n=== 4. Universal Pareto Dominance ===")
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p_a    = np.array([0, 0, 1, 1, 1, 0, 1, 0])  # perfect
    p_b    = np.array([0, 1, 1, 0, 1, 0, 1, 0])  # 2 errors
    p_c    = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # predicts all positive

    im_a = IndicatorMatrix(y_true, p_a, task="binary")
    im_b = IndicatorMatrix(y_true, p_b, task="binary")
    im_c = IndicatorMatrix(y_true, p_c, task="binary")

    print("Dominance (A vs B):", pareto_dominance(im_a, im_b))  # A_dominates_B
    print("Dominance (B vs C):", pareto_dominance(im_b, im_c))  # incomparable


def example_micro_macro_gap():
    print("\n=== 5. Micro-Macro Gap as Normalized Covariance ===")
    # Multilabel setup with 5 classes
    rng = np.random.default_rng(42)
    Y = (rng.random((100, 5)) < 0.3).astype(int)
    P = Y.copy()
    flip = rng.random((100, 5)) < 0.2
    P[flip] = 1 - P[flip]

    im = IndicatorMatrix(Y, P, task="multilabel")
    gap = micro_macro_gap(im, "f1")
    print(f"Actual Gap (Micro - Macro): {gap['gap']:.6f}")
    print(f"Predicted Gap via Cov(w,v)/wbar: {gap['predicted_gap']:.6f}")
    print(f"Exact Identity Holds: {gap['exact']}")


def example_ordinal_and_costs():
    print("\n=== 6. Ordinal Evaluation & Cost Matrices ===")
    y_true = np.array([0, 1, 2, 3, 4, 2, 1, 3, 0, 4])
    y_pred = np.array([0, 2, 2, 2, 4, 3, 1, 4, 0, 3])

    im = IndicatorMatrix(y_true, y_pred, task="multiclass")
    m, n = im.m, im.n

    J, K = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    C_mae = np.abs(J - K)
    mae = im.cost(C_mae) / n
    print(f"MAE from Cost Matrix: {mae:.4f}")

    im_ord = IndicatorMatrix(y_true, y_pred, task="ordinal")
    print(f"Ordinal Cumulative Macro-Recall: {im_ord.compute('recall', 'macro'):.4f}")


def example_cross_validation_aggregation():
    print("\n=== 7. Cross-Validation Aggregation (Sigma_k Pooling) ===")
    rng = np.random.default_rng(123)
    folds = []
    for _ in range(5):
        y = rng.integers(0, 3, 50)
        p = (y + rng.choice([0, 1], size=50, p=[0.7, 0.3])) % 3
        folds.append(IndicatorMatrix(y, p, task="multiclass"))

    mcc_pooled = fold_aggregate(folds, "mcc", how="macro", pooling="pooled")
    mcc_averaged = fold_aggregate(folds, "mcc", how="macro", pooling="averaged")
    print(f"MCC (Pooled Inter-Fold Sigma_k): {mcc_pooled:.4f}")
    print(f"MCC (Fold-Averaged):             {mcc_averaged:.4f}")


def example_custom_measure():
    print("\n=== 8. Custom Measure & Property Inference ===")

    def custom_harmonic(tp, fp, fn, tn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    measure = register_measure("custom_harmonic", custom_harmonic)
    print(f"Registered measure: {measure.name}")
    print(f"Decomposable:       {measure.decomposable}")
    print(f"Skew-invariant:     {measure.skew_invariant}")


def example_sklearn_scorer():
    print("\n=== 9. Scikit-learn Compatible Scorer ===")
    scorer = make_scorer("mcc", aggregation="macro")
    print(f"Generated Scorer: {scorer}")


if __name__ == "__main__":
    example_multiclass()
    example_diagnostics()
    example_orthogonality_and_redundancy()
    example_pareto_dominance()
    example_micro_macro_gap()
    example_ordinal_and_costs()
    example_cross_validation_aggregation()
    example_custom_measure()
    example_sklearn_scorer()
