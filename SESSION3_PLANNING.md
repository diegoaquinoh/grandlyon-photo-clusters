# Session 3 — Professional Planning

**Date:** February 2, 2026  
**Duration:** 4 hours  
**Presentation:** February 2-6, 2026 (10 min demo + 5 min Q&A)

**Goal:** Finalize clustering recommendation with evidence, implement association rules for auto-naming, analyze temporal patterns, and deliver a bulletproof demo.

---

## 📊 Current Status (End of Session 2)

| Component | Status | Notes |
|-----------|--------|-------|
| Data Cleaning | ✅ Complete | 167,578 photos (40% retention) |
| K-Means Clustering | ✅ Complete | Silhouette: 0.535, 100 clusters |
| Hierarchical Clustering | ✅ Complete | Silhouette: 0.510 |
| DBSCAN | ⚠️ Not suitable | 83.8% in one cluster |
| HDBSCAN | ✅ Explored | 924 clusters, promising |
| TF-IDF Text Mining | ✅ Complete | All clusters described |
| Cluster Map | ✅ Complete | `app/cluster_map.html` |
| Association Rules | ❌ Not started | Session 3 priority |
| Temporal Analysis | ❌ Not started | Session 3 priority |

---

## ⏰ Time-Boxed Schedule (4 hours)

| Time | Duration | Task | Priority |
|------|----------|------|----------|
| 0:00 - 0:30 | 30 min | **Task 1:** Clustering finalization & comparison doc | 🔴 Critical |
| 0:30 - 1:30 | 60 min | **Task 2:** Association rules implementation | 🔴 Critical |
| 1:30 - 2:00 | 30 min | **Task 3:** Auto-naming integration in app | 🔴 Critical |
| 2:00 - 2:45 | 45 min | **Task 4:** Temporal/event analysis | 🔴 Critical |
| 2:45 - 3:15 | 30 min | **Task 5:** Map & demo polish | 🟡 Important |
| 3:15 - 4:00 | 45 min | **Task 6:** Presentation prep & dry-run | 🔴 Critical |

---

## Task 1: Clustering Optimization & Recommendation (30 min)

**Objective:** Produce a clear, justified recommendation for Grand Lyon.

### 1.1 Final Comparison Document
- [ ] Create `reports/clustering_recommendation.md` with:
  - Algorithm comparison table (K-Means vs Hierarchical vs HDBSCAN)
  - Visual evidence (silhouette plots, cluster size distributions)
  - Scalability analysis (runtime on 167k points)
  - Map interpretability assessment

### 1.2 Recommended Algorithm: **K-Means (k=100)** or **HDBSCAN**
- **K-Means advantages:**
  - Highest silhouette (0.535)
  - Balanced cluster sizes (max 9%)
  - Fast, reproducible, easy to explain
- **HDBSCAN advantages:**
  - No need to pre-specify k
  - Better at finding variable-density clusters
  - More granular (924 clusters)

### 1.3 Decision Criteria for Grand Lyon
- [ ] Document sensitivity to parameters
- [ ] Explain trade-off: granularity vs interpretability
- [ ] Provide clear "If you want X, use Y" guidance

**Deliverable:** `reports/clustering_recommendation.md`

---

## Task 2: Association Rules Implementation (60 min)

**Objective:** Implement second text mining method for richer cluster descriptions.

### 2.1 Implementation Strategy
```python
# Target file: src/text_mining.py

# Method: Apriori / FP-Growth for frequent itemsets
# Libraries: mlxtend.frequent_patterns

# Steps:
# 1. Create binary matrix: (cluster, term) → 0/1
# 2. Run Apriori with min_support = 0.1 (adjust per cluster size)
# 3. Generate association rules with min_confidence = 0.5
# 4. Extract top rules per cluster
```

### 2.2 Key Implementation Tasks
- [ ] Add `compute_association_rules()` function to `src/text_mining.py`
- [ ] Parameters: `min_support`, `min_confidence`, `max_itemset_size=3`
- [ ] Output: JSON with top 5 rules per cluster

### 2.3 Validation
- [ ] Manually verify rules for 5 known clusters:
  - Fourvière (basilique, notre dame, panoramic)
  - Vieux Lyon (traboule, renaissance, cathedral)
  - Fête des Lumières (illuminations, december, place bellecour)
  - Demeure du Chaos (chaos, ddc, abodeofchaos)
  - Parc de la Tête d'Or (zoo, botanical, lake)

**Deliverable:** `reports/association_rules_summary.md`

---

## Task 3: Auto-Naming & App Integration (30 min)

**Objective:** Display cluster names on the map with meaningful labels.

### 3.1 Naming Strategy
```
Priority order for cluster name:
1. Best association rule (if confidence > 0.7)
2. Top 2-3 TF-IDF terms (excluding generic words)
3. Fallback: "Cluster {id} ({size} photos)"
```

### 3.2 Implementation Tasks
- [ ] Add `generate_cluster_name()` function to `src/text_mining.py`
- [ ] Update `src/map_visualization.py` to display names in popups
- [ ] Add cluster summary panel (side panel or expandable markers)

### 3.3 Map Enhancements
- [ ] Popup content: Cluster name, size, top 5 terms, photo date range
- [ ] Legend with top 10 clusters by size
- [ ] Optional: Filter by cluster name search

**Deliverable:** Updated `app/cluster_map.html` with named clusters

---

## Task 4: Temporal/Event Analysis (45 min)

**Objective:** Identify one-time events vs permanent tourist spots.

### 4.1 Temporal Aggregation
- [ ] Create `src/temporal_analysis.py` with:
  - `aggregate_by_period(df, period='month')` → counts per cluster
  - `detect_peaks(series, threshold=2.0)` → z-score peak detection
  - `classify_cluster_type()` → one-time / recurring / permanent

### 4.2 Analysis Tasks
- [ ] Compute monthly photo counts per cluster (2010-2025)
- [ ] Identify clusters with strong temporal patterns:
  - **December spikes** → Fête des Lumières candidates
  - **Summer peaks** → Tourism/festivals
  - **Single date spikes** → One-time events

### 4.3 Visualization
- [ ] Generate time series plots for top 10 clusters
- [ ] Create heatmap: cluster × month (normalized)
- [ ] Annotate with known Lyon events

### 4.4 Expected Findings
| Pattern Type | Example Clusters | Characteristics |
|--------------|------------------|-----------------|
| **Permanent POI** | Fourvière, Vieux Lyon | Consistent year-round activity |
| **Recurring Event** | Place Bellecour (December) | Annual spike (Fête des Lumières) |
| **One-time Event** | Sports venues, concerts | Single date/week spike |

**Deliverable:** `reports/temporal_analysis.md` + visualizations

---

## Task 5: Map & Demo Polish (30 min)

**Objective:** Ensure flawless demo experience.

### 5.1 Map Improvements
- [ ] Add temporal filter (date range slider)
- [ ] Add cluster type legend (permanent/recurring/event)
- [ ] Improve performance (cluster centroids if too many points)
- [ ] Test on different browsers

### 5.2 Pipeline Freeze
- [ ] Create `scripts/run_full_pipeline.py` (one-command reproduction)
- [ ] Verify all outputs regenerate correctly
- [ ] Create backup of final outputs

### 5.3 Backup Screenshots
- [ ] Screenshot: Full map with clusters
- [ ] Screenshot: Example cluster popup with name
- [ ] Screenshot: Temporal analysis chart
- [ ] Screenshot: Cluster comparison table

**Deliverable:** Demo-ready application + backup assets

---

## Task 6: Presentation Preparation (45 min)

**Objective:** 10-minute demo with clear narrative.

### 6.1 Presentation Structure (10 min total)

| Section | Time | Content |
|---------|------|---------|
| **Context** | 1 min | Grand Lyon challenge, Flickr dataset |
| **Data Pipeline** | 1.5 min | 420k → 167k photos, key cleaning decisions |
| **Clustering Comparison** | 2 min | 3 algorithms, metrics, why K-Means/HDBSCAN |
| **Live Demo: Map** | 2.5 min | Navigate clusters, show popups with names |
| **Text Mining** | 1.5 min | TF-IDF + Association Rules, auto-naming |
| **Temporal Insights** | 1 min | Permanent vs events, Fête des Lumières example |
| **Conclusion** | 0.5 min | Recommendations for Grand Lyon |

### 6.2 Key Messages for Q&A
- **Why this cleaning?** → 60% duplicates, coherent GPS/dates
- **Why K-Means over DBSCAN?** → DBSCAN creates one mega-cluster
- **How are clusters named?** → TF-IDF terms + association rules
- **Scalability?** → K-Means handles millions, pipeline runs in <5 min
- **What would we improve?** → Deep learning embeddings, more data sources

### 6.3 Dry-Run Checklist
- [ ] Practice demo path (2 runs minimum)
- [ ] Test all interactive elements
- [ ] Prepare for common failures (offline backup)
- [ ] Time each section

**Deliverable:** Presentation slides (optional) + demo script

---

## 📋 Session 3 Deliverables Checklist

| Deliverable | File/Location | Status |
|-------------|---------------|--------|
| Clustering recommendation | `reports/clustering_recommendation.md` | ⬜ |
| Association rules code | `src/text_mining.py` | ⬜ |
| Association rules report | `reports/association_rules_summary.md` | ⬜ |
| Auto-naming function | `src/text_mining.py` | ⬜ |
| Updated cluster map | `app/cluster_map.html` | ⬜ |
| Temporal analysis code | `src/temporal_analysis.py` | ⬜ |
| Temporal analysis report | `reports/temporal_analysis.md` | ⬜ |
| Full pipeline script | `scripts/run_full_pipeline.py` | ⬜ |
| Backup screenshots | `reports/screenshots/` | ⬜ |
| Demo script | `reports/demo_script.md` | ⬜ |

---

## 🎯 Success Criteria (Milestone 3 - 10 points)

Per project requirements:
1. ✅ **Clustering optimization complete** with documented comparison
2. ✅ **2 text mining algorithms** (TF-IDF + Association Rules)
3. ✅ **Cluster names appear on application**
4. ✅ **Temporal exploration** with results visualized
5. ✅ **Demo-ready** with no bugs during presentation

---

## ⚠️ Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Association rules too slow | Medium | Use FP-Growth, limit itemset size |
| No clear temporal patterns | Low | Focus on known events (December) |
| Demo crashes | Medium | Local backup + screenshots |
| Time overrun | High | Strict time-boxing, skip polish if needed |

**Fallback priorities if running out of time:**
1. Skip map polish (Task 5) → Use current map
2. Simplify temporal analysis → Just monthly histograms
3. Use basic naming → TF-IDF top 3 terms only

---

## 📚 Reference: Session 3 Milestones from Project Spec

> **Milestones for the 3rd session (10/20 points):**
> - Complete the optimization of the clustering algorithms, compare and discuss the results obtained and decide which algorithm you recommend;
> - Finalize the implementation of 2 text mining algorithms to automatically name clusters, make it appear on the application;
> - Explore the data through the scope of time, show the results of your exploration;
> - Prepare the final demo to avoid any bug during the demonstration.
