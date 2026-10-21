# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""What a coding agent produces when it follows the ``aggmetrics-guide`` skill.

The request (see ``agent_session_example.md``): *"I trained two classifiers on an
imbalanced 10-class problem and reported 0.89 accuracy for the logistic regression.
Which one should I deploy, and how should I report it?"*

The script follows the skill's recipes in order: build indicator matrices (recipe 1),
run the diagnostics (recipe 2), compare the models for universal dominance (recipe 7),
and pool the cross-validation folds (recipe 10). Run it with::

    pip install "aggmetrics[examples]"
    python agent_session_example.py
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from aggmetrics import IndicatorMatrix, diagnose, fold_aggregate, pareto_dominance

warnings.filterwarnings("ignore")

# -- the data and the two models the user already trained ------------------------------
X, y = load_digits(return_X_y=True)
keep = {0: 170, 1: 120, 2: 85, 3: 60, 4: 42, 5: 30, 6: 20, 7: 14, 8: 9, 9: 5}
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.flatnonzero(y == c), k, replace=False)
                      for c, k in keep.items()])
X, y = X[idx] / 16.0, y[idx]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

models = {
    "logistic regression": LogisticRegression(C=0.1, max_iter=3000),
    "random forest": RandomForestClassifier(n_estimators=200, random_state=0),
}
ims = {name: IndicatorMatrix(yte, m.fit(Xtr, ytr).predict(Xte), task="multiclass",
                             n_classes=10)
       for name, m in models.items()}

# -- recipe 1: the user's number, and what the skill adds ------------------------------
print("1. Headline numbers (held-out half, 10 classes, 5 to 85 examples per class)")
for name, im in ims.items():
    print(f"   {name:20s} accuracy={im.accuracy():.3f}  "
          f"macro-F1={im.compute('f1', 'macro'):.3f}  "
          f"macro-recall={im.compute('recall', 'macro'):.3f}")

# -- recipe 2: diagnostics -------------------------------------------------------------
print("\n2. Diagnostics")
for name, im in ims.items():
    kinds = {}
    for f in diagnose(im, ["precision", "recall", "f1", "mcc"]):
        kinds.setdefault(f.code, []).append(f)
    print(f"   {name}: {sorted(kinds)}")
    for f in kinds.get("undefined_entries", []):
        if f.detail["measure"] == "precision":
            d = f.detail
            print(f"     macro-precision: {d['undefined']} of {d['total']} classes never predicted; "
                  f"{d['exclude']:.3f} with 'exclude' vs {d['zero']:.3f} with 'zero'")

# -- recipe 7: universal dominance -----------------------------------------------------
print("\n3. Does one model beat the other on every monotone measure?")
verdict = pareto_dominance(ims["logistic regression"], ims["random forest"])
print(f"   pareto_dominance(logistic regression, random forest) = {verdict!r}")
if verdict == "incomparable":
    print("   -> the ranking depends on which measure is chosen; state the measure.")

# -- recipe 10: pooled cross-validation ------------------------------------------------
print("\n4. Five-fold cross-validation on all examples (pooled vs averaged macro-F1)")
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, m in models.items():
    folds = [IndicatorMatrix(y[te], m.fit(X[tr], y[tr]).predict(X[te]), task="multiclass",
                             n_classes=10) for tr, te in cv.split(X, y)]
    pooled = fold_aggregate(folds, "f1", "macro", pooling="pooled")
    averaged = fold_aggregate(folds, "f1", "macro", pooling="averaged")
    per_fold = [f.compute("f1", "macro") for f in folds]
    print(f"   {name:20s} pooled={pooled:.3f}  averaged={averaged:.3f}  "
          f"sd across folds={np.std(per_fold, ddof=1):.3f}")
