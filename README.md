# TP 2 — Standardized PCA

## Contents

- `pca_solution.py`: complete implementation of standardized PCA from scratch and with scikit-learn. It prints the requested dataset summaries, standardized data, correlation matrix, eigensystem, coordinates, explained variance, and aligned comparison. It also creates `pca_comparison.png` when run.
- `results.md`: key numerical results and interpretation.

## Run

Install Python dependencies and run from this folder:

```bash
python -m pip install numpy pandas scikit-learn matplotlib
python pca_solution.py
```

## Method notes

The standardization uses population standard deviation (`ddof=0`), matching `StandardScaler`. The sample covariance matrix used internally by `PCA` has eigenvalues larger by the common factor `n/(n-1)` than the correlation matrix eigenspectrum here; explained variance ratios and component directions therefore agree. Component signs may flip independently, so coordinate comparisons must allow sign alignment.
