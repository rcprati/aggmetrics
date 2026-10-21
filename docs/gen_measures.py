# Copyright (c) 2026, Ronaldo C. Prati
# SPDX-License-Identifier: BSD-3-Clause
"""Generate the table of built-in measures from the registry."""
from pathlib import Path

from aggmetrics import get_measure, list_measures


def _bayes(m):
    t = m.bayes_threshold()
    return "-" if t is None else f"{t:.3g}"


def write(path: Path) -> None:
    rows = [
        "| Measure | Aliases | Better | Decomposable | Skew-invariant "
        "| Depends on tn at tp=0 | Bayes threshold | Value if all wrong (m=10) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in list_measures():
        m = get_measure(name)
        yes = lambda b: "yes" if b else "no"
        rows.append(
            f"| `{m.name}` | {', '.join(f'`{a}`' for a in m.aliases) or '-'} "
            f"| {'higher' if m.greater_is_better else 'lower'} | {yes(m.decomposable)} "
            f"| {yes(m.skew_invariant)} | {yes(m.tn_dependent_at_zero)} "
            f"| {_bayes(m)} | {m.attainable_floor(10):.4g} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")
