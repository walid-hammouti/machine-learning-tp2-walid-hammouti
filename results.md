# Numerical results and interpretation

The data matrix has **6 rows and 4 variables**. The means are `[12.6667, 12.8333, 13.6667, 13.0000]`, and the population standard deviations used for standardization are `[3.0368, 2.4095, 2.7487, 2.7689]` in the order Mathematics, Statistics, Programming, Algorithms.

## PCA from scratch

Correlation matrix (rounded):

| | Mathematics | Statistics | Programming | Algorithms |
|---|---:|---:|---:|---:|
| Mathematics | 1.0000 | 0.9263 | 0.9451 | 0.9911 |
| Statistics | 0.9263 | 1.0000 | 0.7717 | 0.8744 |
| Programming | 0.9451 | 0.7717 | 1.0000 | 0.9635 |
| Algorithms | 0.9911 | 0.8744 | 0.9635 | 1.0000 |

Descending eigenvalues are approximately `[3.739727, 0.240228, 0.019117, 0.000927]`. Their explained variance ratios are `[93.4932%, 6.0057%, 0.4779%, 0.0232%]`; cumulative values are `[93.4932%, 99.4989%, 99.9768%, 100.0000%]`.

| Student | PC1 | PC2 |
|---|---:|---:|
| S1 | 0.416636 | -0.674008 |
| S2 | -1.416464 | -0.237266 |
| S3 | 2.950619 | -0.227866 |
| S4 | -2.881443 | -0.135694 |
| S5 | 1.652021 | 0.482733 |
| S6 | -0.721369 | 0.792100 |

## Comparison and interpretation

The first two components retain **99.4989%** of the total variance. The four grades are strongly positively correlated, so PC1 captures the common overall achievement pattern. PC2 accounts for a much smaller contrast among subject profiles. Both implementations produce matching directions and projections after allowing for sign flips. Eigenvalues reported by scikit-learn are covariance eigenvalues of the standardized observations and equal the correlation-matrix eigenvalues above multiplied by `n/(n-1) = 1.2` (approximately `[4.487673, 0.288274]` for the first two). The explained variance ratios are unchanged by that shared scale factor. A principal component eigenvector can be multiplied by `-1` with no mathematical change, which flips that axis's plotted coordinates but preserves pairwise geometry and variance.
