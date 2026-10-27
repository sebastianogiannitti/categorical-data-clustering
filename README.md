# Clustering Nominal Data: K-Means vs. Hierarchical Clustering

A reproducible **unsupervised machine learning** study comparing K-Means and agglomerative clustering on nominal categorical features, implemented with **Python, NumPy, pandas, scikit-learn, and Matplotlib**.

Developed from a Machine Learning and Data Mining university exercise, then reorganized and re-evaluated as a standalone portfolio project.

> **Dataset transparency:** The original examination dataset is no longer available. The included file is a **synthetic replacement**, *not* a reconstruction of the original observations. It has 1,000 samples, three categorical features (`F0`, `F1`, `F2`), and letters `a`–`j` as possible values. Approximate one-variable category frequencies and the initial preview match the earlier saved notebook, but relationships between variables cannot be reconstructed from those statistics. **All benchmark numbers and figures below refer only to the synthetic replacement.**

## Results at a glance

The primary selection criterion of this particular exercise is **balanced cluster sizes**. The Balanced Partition Index (BPI) is defined as `std(cluster sizes) / mean(cluster sizes)`; **smaller is more balanced**.

| Method | Selected setup | Silhouette | BPI | Inertia |
| --- | --- | ---: | ---: | ---: |
| K-Means | 10 clusters, `random_state=42`, `n_init=20` | **0.1945** | **0.0045** | 1778.7893 |
| Agglomerative | 10 clusters, `average` linkage | 0.0452 | 0.3515 | N/A |

*The method/parameters are selected by minimum BPI with silhouette as a tie-breaker, rather than assumed beforehand. For a controlled comparison, agglomerative linkage is selected while holding the cluster count fixed at the K-Means choice.*

![K-Means selection metrics](figures/kmeans_metrics.png)

![Sizes of selected clusters](figures/cluster_sizes.png)

**Interpretation:** K-Means produces the most balanced partitions *under this selection rule on this synthetic dataset*. This is **not evidence of ground-truth classes or general clustering quality**. The silhouette values are modest, cluster IDs are arbitrary, and the `k=10` result lies at the upper edge of the tested range. The data are synthetic and have no defined cluster labels.

## Approach

1. Explore categorical value distributions and missing values.
2. One-hot encode three nominal attributes to avoid imposing an arbitrary order on letters.
3. Evaluate K-Means for `k=2,...,10` using inertia, silhouette score and BPI; select by minimum BPI, using silhouette to break ties.
4. Refit K-Means with the same random seed as the search.
5. Compare `ward`, `average`, `single` and `complete` agglomerative linkages **at the same `k`**, then fit the selected linkage.
6. Compare final cluster-size histograms and discuss what the metrics do and do not show.

K-Means and Euclidean distances on one-hot encoded categories are useful **educational baselines**, not necessarily ideal for nominal data. A possible extension is K-Modes or another categorical-distance-based clustering method.

## Run the notebook

```bash
git clone https://github.com/sebastianogiannitti/categorical-data-clustering.git
cd categorical-data-clustering
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
jupyter notebook clustering_analysis.ipynb
```

Choose **Kernel → Restart & Run All**. The notebook loads `data/synthetic_categorical_data.tsv`, which is intentionally **tab-separated**, and writes charts to `figures/` and numeric comparison tables to `results/`. All displayed experiments have been run successfully against the included dataset with a fixed K-Means seed.

## Repository layout

```text
.
├── README.md
├── clustering_analysis.ipynb        # Full analysis with saved outputs
├── data/
│   └── synthetic_categorical_data.tsv
├── figures/
│   ├── cluster_sizes.png
│   └── kmeans_metrics.png
├── results/
│   ├── final_comparison.csv
│   ├── hierarchical_linkages.csv
│   └── kmeans_search.csv
├── requirements.txt
└── .gitignore
```

## Reproducibility and caveats

- The data are **synthetic**; they do not reproduce the unavailable examination dataset's joint distribution or numerical results.
- We compare models using a stated **balance-first objective**, not a claim that this objective is suitable for every clustering task.
- Inertia normally decreases with increasing `k`; selecting solely on inertia or solely on BPI is not a universal best-practice rule.
- Running under materially different scikit-learn versions can produce slightly different metrics, especially for tied or nearly tied solutions.

**Author:** Sebastiano Giannitti | [GitHub profile](https://github.com/sebastianogiannitti)