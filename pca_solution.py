"""TP 2: standardized PCA from scratch and with scikit-learn.
Install dependencies with: python -m pip install numpy pandas scikit-learn matplotlib
Run: python pca_solution.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA = pd.DataFrame(
    [[12, 11, 14, 13], [15, 14, 16, 15], [8, 9, 10, 9],
     [17, 16, 18, 17], [10, 12, 11, 10], [14, 15, 13, 14]],
    index=["S1", "S2", "S3", "S4", "S5", "S6"],
    columns=["Mathematics", "Statistics", "Programming", "Algorithms"],
)

print("Dataset shape:", DATA.shape)
print("\nDescriptive statistics:\n", DATA.describe().round(4))

# Standardize with population standard deviation (ddof=0), as StandardScaler does.
means = DATA.mean()
stds = DATA.std(ddof=0)
Z = (DATA - means) / stds
correlation = Z.corr()
print("\nStandardized data:\n", Z.round(4))
print("\nCorrelation matrix:\n", correlation.round(4))

# eigh is designed for real symmetric matrices. Sort descending and keep eigenvectors
# as columns, so Z @ components produces student coordinates.
eigenvalues, eigenvectors = np.linalg.eigh(correlation.to_numpy())
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
components = eigenvectors[:, :2]
scores_scratch = Z.to_numpy() @ components
ratio_scratch = eigenvalues / eigenvalues.sum()
cumulative_scratch = np.cumsum(ratio_scratch)

print("\nEigenvalues (descending):\n", np.round(eigenvalues, 8))
print("\nEigenvectors (columns, descending):\n", np.round(eigenvectors, 8))
print("\nScratch PC1/PC2 coordinates:\n", pd.DataFrame(scores_scratch, index=DATA.index, columns=["PC1", "PC2"]).round(6))
print("\nScratch explained variance ratios:\n", np.round(ratio_scratch, 8))
print("Cumulative explained variance:\n", np.round(cumulative_scratch, 8))

# sklearn standardizes the original variables, then PCA centers the scaled values.
Z_scaled = StandardScaler().fit_transform(DATA)
pca = PCA(n_components=2)
scores_sklearn = pca.fit_transform(Z_scaled)
ratio_sklearn = pca.explained_variance_ratio_
print("\nScikit-learn coordinates:\n", pd.DataFrame(scores_sklearn, index=DATA.index, columns=["PC1", "PC2"]).round(6))
print("\nScikit-learn eigenvalues:\n", np.round(pca.explained_variance_, 8))
print("Scikit-learn explained variance ratios:\n", np.round(ratio_sklearn, 8))
print("Cumulative explained variance:\n", np.round(np.cumsum(ratio_sklearn), 8))

# Eigenvectors can flip sign without changing a principal component. Align each sklearn
# axis with the scratch result before comparing coordinates.
scores_aligned = scores_sklearn.copy()
for j in range(2):
    if np.dot(scores_aligned[:, j], scores_scratch[:, j]) < 0:
        scores_aligned[:, j] *= -1
print("\nMaximum absolute coordinate difference after sign alignment:",
      np.max(np.abs(scores_aligned - scores_scratch)))
print("Variance retained by first two PCs: {:.2f}%".format(100 * cumulative_scratch[1]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
for ax, scores, title in zip(axes, [scores_scratch, scores_sklearn],
                             ["PCA from scratch", "PCA with scikit-learn"]):
    ax.scatter(scores[:, 0], scores[:, 1], s=65, color="#2878B5")
    for label, (x, y) in zip(DATA.index, scores):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")
    ax.axhline(0, color="0.75", linewidth=0.8)
    ax.axvline(0, color="0.75", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(alpha=0.2)
fig.suptitle("Students projected onto the first two principal components")
fig.savefig("pca_comparison.png", dpi=180)
print("\nSaved plot to pca_comparison.png")
