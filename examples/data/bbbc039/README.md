# `bbbc039/subset.npz`

A derived subset of the image set **BBBC039v1** (U2OS cell nuclei, Hoechst stain, fluorescence
microscopy), from the Broad Bioimage Benchmark Collection. Used by
`notebooks/09_segmentation.ipynb`.

**Source and licence.** <https://bbbc.broadinstitute.org/BBBC039>. The images and annotations are
released under CC0 1.0 (public domain dedication). The attribution below is a courtesy, not a
requirement.

> Caicedo, J. C., Roth, J., Goodman, A., Becker, T., Karhohs, K. W., Broisin, M., Molnar, C.,
> McQuin, C., Singh, S., Theis, F. J. and Carpenter, A. E. (2019). Evaluation of deep learning
> strategies for nucleus segmentation in fluorescence images. *Cytometry Part A*, 95(9), 952-965.
>
> Ljosa, V., Sokolnicki, K. L. and Carpenter, A. E. (2012). Annotated high-throughput microscopy
> image sets for validation. *Nature Methods*, 9, 637.

| Array | Shape | Type | Contents |
|---|---|---|---|
| `images` | (12, 384, 512) | `uint8` | Hoechst images, rescaled to 8 bits |
| `labels` | (12, 384, 512) | `uint8` | 0 for background, 1 to 3 for the nuclei (a 3-colour labelling: touching nuclei get different values) |
| `names` | (12,) | `str` | file stem of each field in BBBC039 |
| `meta` | scalar | `str` (JSON) | source, crop and rescaling constants |

**Processing.** From the 50 fields of the official `test` split (`metadata/test.txt`), the fields
were sorted by foreground share and every fourth one was kept, starting at the second (12 fields).
Each field was cropped to the centre window `[68:452, 92:604]` (384 x 512 pixels) and rescaled to
8 bits with one linear map, `(x - 132) / (1141 - 132)` clipped to [0, 1] and multiplied by 255,
where 132 and 1141 are the 0.5th and 99.5th percentiles of the 12 crops. The label masks are the
red channel of the annotation PNGs, cropped the same way. Nothing else was changed.
