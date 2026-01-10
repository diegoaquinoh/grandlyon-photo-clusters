# Clustering Decisions & Rationale

## Session 2 - Clustering Experimentation Summary

**Date:** 2026-01-10

---

## Algorithm Comparison Results

| Algorithm | Silhouette | Largest Cluster % | Clusters | Verdict |
|-----------|------------|-------------------|----------|---------|
| **K-Means (k=100)** | 0.535 ✅ | 9.0% ✅ | 100 | **SELECTED** |
| Hierarchical (k=100) | 0.510 | 9.5% | 100 | Good alternative |
| DBSCAN (eps=0.005) | 0.404 | 83.8% ⚠️ | 59 | Not suitable |

---

## Final Choices

### Clustering Algorithm: **K-Means**
- **k = 100 clusters**
- **Scaled coordinates** (StandardScaler)
- **Rationale:** 
  - Highest silhouette score (0.535) = best cluster separation
  - Balanced cluster sizes (largest = 9.0%, not dominant)
  - Fast and scalable

### Rejected Alternatives

#### DBSCAN
- **Tested:** eps=[0.002, 0.003, 0.004, 0.005, 0.006], min_samples=10
- **Best eps:** 0.005 (silhouette=0.35)
- **Why rejected:** 83.8% of points in one mega-cluster → not useful for geographic analysis

#### Hierarchical (Agglomerative)
- **Tested:** linkages=[ward, complete, average], k=[20-120]
- **Best config:** ward linkage, k=100
- **Why not selected:** Slightly lower silhouette (0.510) and O(n²) memory → crashes on full dataset

---

## Parameter Tuning Notes

### K-Means
- Tested k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
- User override: **k=100** (forced)
- Elbow method suggested lower k, but k=100 provides better granularity

### DBSCAN
- eps too small (0.002) → too many clusters, low silhouette
- eps too large (0.006) → over-merging, silhouette drops
- Optimal window: 0.003-0.005

### Hierarchical
- `average` linkage: artificially high silhouette due to one dominant cluster (50-70%)
- `ward` linkage: most balanced cluster distribution
- Sample-based evaluation (20k points) due to memory constraints

---

## Files Generated

- `reports/kmeans_parameter_sweep.png` - K-Means elbow & silhouette plots
- `reports/dbscan_parameter_sweep.png` - DBSCAN parameter sweep
- `reports/hierarchical_parameter_sweep.png` - Hierarchical linkage comparison
- `reports/clustering_comparison.csv` - Final algorithm comparison
- `app/cluster_map.html` - Interactive cluster visualization (pending)
