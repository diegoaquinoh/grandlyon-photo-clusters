# Clustering Decisions & Rationale

## Session 2 - Updated 2026-02-01

---

## Algorithm Comparison Results

| Algorithm            | Silhouette    | Largest Cluster % | Clusters | Noise % | Verdict          |
| -------------------- | ------------- | ----------------- | -------- | ------- | ---------------- |
| **HDBSCAN (mcs=30)** | **0.7674** ✅ | 1.2% ✅           | 924      | 44.8%   | **SELECTED**     |
| K-Means (k=120)      | 0.5414        | ~9%               | 120      | 0%      | Good alternative |
| Hierarchical (k=100) | 0.510         | 9.5%              | 100      | 0%      | Memory limited   |
| DBSCAN (eps=0.005)   | 0.404         | 83.8% ⚠️          | 59       | 0.3%    | Not suitable     |

---

## Final Choice: **HDBSCAN**

### Configuration

- **min_cluster_size = 30**
- **min_samples = 30** (same as min_cluster_size)
- **Unscaled coordinates** (euclidean on lat/lon)

### Rationale

- **42% higher silhouette** (0.7674 vs 0.5414) compared to K-Means
- **No mega-cluster problem** - largest cluster only 1.2% of data
- **Automatic noise detection** - 44.8% of isolated photos filtered as noise
- **No need to predefine k** - clusters discovered automatically

### Trade-offs

- Higher noise percentage (44.8%) - but these are truly isolated photos
- More clusters (924) - provides finer granularity for tourist hotspots

---

## Parameter Tuning: HDBSCAN

| min_cluster_size | Clusters | Noise %   | Silhouette    |
| ---------------- | -------- | --------- | ------------- |
| 10               | 3,051    | 33.4%     | 0.7163        |
| 15               | 1,999    | 38.7%     | 0.7577        |
| 20               | 1,451    | 41.5%     | 0.7665        |
| **30**           | **924**  | **44.8%** | **0.7674** ✅ |
| 50               | 517      | 47.2%     | 0.7512        |
| 75               | 316      | 48.3%     | 0.7171        |
| 100              | 237      | 48.1%     | 0.7162        |

---

## Files Generated

- `reports/hdbscan_parameter_sweep.png` - HDBSCAN parameter sweep
- `reports/best_clustering_params.json` - Selected configuration
- `scripts/hdbscan_experiment.py` - Experiment script
