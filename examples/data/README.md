# Data used by the examples

## `yeast.npz`

The multilabel *yeast* benchmark of Elisseeff and Weston: 2417 genes, 103 features
(micro-array expression data and phylogenetic profiles) and 14 labels (functional classes).
Used by `notebooks/03_multilabel_ranking.ipynb`.

| Array | Shape | Type | Contents |
|---|---|---|---|
| `X` | (2417, 103) | `float64` | features |
| `Y` | (2417, 14) | `uint8` | label indicator matrix (1 if the gene has the function) |

**Sources.** The same dataset is distributed by two repositories:

- OpenML, dataset 40597, version 4 (<https://www.openml.org/d/40597>), where it is listed with
  the licence "Public". The arrays here are the data as OpenML serves them.
- The Mulan multilabel dataset repository (<https://mulan.sourceforge.net/datasets-mlc.html>),
  as `yeast.rar`. Its page states no terms of use for the datasets. The statistics it lists
  (2417 instances, 103 numeric features, 14 labels, label cardinality 4.237, density 0.303,
  198 distinct label combinations) are those of this file, which was checked.

The only conversion is that the label columns, stored as the strings `TRUE` and `FALSE` in the
OpenML version, became 1 and 0:

```python
import numpy as np
from sklearn.datasets import fetch_openml

data = fetch_openml("yeast", version=4, as_frame=False, parser="auto")
np.savez_compressed("yeast.npz", X=data.data,
                    Y=(data.target.astype(str) == "TRUE").astype(np.uint8))
```

**Citation.** A. Elisseeff and J. Weston. A kernel method for multi-labelled classification.
In T. G. Dietterich, S. Becker and Z. Ghahramani (eds.), *Advances in Neural Information
Processing Systems 14*, 2001.

This file is a copy for convenience, so that the notebook runs offline and in continuous
integration. It is not covered by this repository's BSD-3-Clause licence. If you hold rights
to the data and object to the copy, please open an issue and it will be removed.
