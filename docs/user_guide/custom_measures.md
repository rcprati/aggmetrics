# Your own measures

`register_measure(name, fn)` accepts any function of $(tp, fp, fn, tn)$ and infers its
structural properties from the formula: decomposability (by solving a homogeneous linear
system for the coefficients and validating it on held-out count vectors), skew-invariance,
dependence on $tn$ at $tp = 0$, and direction. The measure is then usable under every
operator.

```python
from aggmetrics import register_measure, infer_properties

props = infer_properties(my_formula)    # properties only, nothing registered
print(props.summary())
measure = register_measure("my_measure", my_formula)
```

**Checking the inference.** The properties of the 21 built-in measures are also known
analytically, so `verify_inference()` compares the two. They agree on all 21, with recovered
coefficients within $6\times 10^{-15}$, about 28 times the double-precision machine epsilon.

:::{note}
**Sketch.** To write: what inference cannot determine; formulas with undefined regions;
passing explicit properties to override the inference.
:::
