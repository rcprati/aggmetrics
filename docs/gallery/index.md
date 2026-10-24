# Pitfall gallery

Each notebook starts from a mistake that is easy to make, shows what the usual tools report,
and then what `aggmetrics` reports. They run in a few seconds, need only NumPy,
scikit-learn, pandas and matplotlib, and are executed in continuous integration.

| Notebook | What it shows |
|---|---|
| {doc}`01_floor_trap` | A classifier wrong on every example scores 0.998 on label accuracy with 1000 classes |
| {doc}`02_micro_macro_imbalance` | Micro-F1 equals accuracy; micro-MCC outside the per-class range; the hidden choice for 0/0 |
| {doc}`03_multilabel_ranking` | Three models on *yeast* ranked under 18 measure-operator combinations |
| {doc}`04_ordinal_and_cost` | Nine in ten errors are one grade off; a cost matrix changes which prediction is best |
| {doc}`05_model_selection_and_ci` | The scorer picks the model, pooled and averaged folds differ, and a test that fails on a flawed evaluation |
| {doc}`06_long_tail_images` | The same CIFAR-10-LT model scores 0.92 on a long-tailed set and 0.66 on a balanced one; what changes with the class prior |
| {doc}`07_thresholds_and_degenerate_optima` | Recall, specificity and precision are maximised by trivial classifiers; a cost-weighted measure has a closed-form threshold |
| {doc}`08_soft_labels` | Image classifiers against the distribution of 50 human annotators (CIFAR-10H): does the soft reference change the verdict, where do the models err, and a simulation of when it reverses the verdict |
| {doc}`09_segmentation` | Cell nuclei (BBBC039): pixel accuracy gives 0.82 to a predictor that outputs nothing; pooled and per-image IoU, and the convention for images with no object |
| {doc}`10_custom_measures` | Register your own formula: what is inferred (decomposable, skew-invariant), what follows on data, and a piecewise measure on which the inference is wrong |
| {doc}`11_torch_and_keras` | Evaluating framework models: logits, pooling batches instead of averaging a measure over them, accumulating counts, and a Keras callback that logs a measure per epoch |
| {doc}`12_choosing_measures` | Remark 2 on data: which measures are redundant (rank of the Jacobian), the best pair for a binary problem, and why 63 aggregated measures span only 11 of the 90 directions of a ten-class confusion matrix |

```{toctree}
:hidden:

01_floor_trap
02_micro_macro_imbalance
03_multilabel_ranking
04_ordinal_and_cost
05_model_selection_and_ci
06_long_tail_images
07_thresholds_and_degenerate_optima
08_soft_labels
09_segmentation
10_custom_measures
11_torch_and_keras
12_choosing_measures
```
