# Grand Lyon Photo Clusters — Project Planning

## Session 1 (4h) — Data understanding + first end-to-end baseline ✅

**Goal:** Get a working pipeline from raw data → cleaned sample → map → first clustering run (even if rough).

### Tasks

1. **Project setup & workflow**
   * Create repo structure (data/, src/, notebooks/, app/, reports/), environment (requirements.txt / conda).

2. **Data ingestion & profiling**
   * Load dataset efficiently (chunking if needed); document schema: *(photo_id, user_id, lat, lon, tags, description/title, dates)*.
   * Quick diagnostics: missing values, duplicates, GPS ranges/outliers, date parsing issues, language mix in text.

3. **Data cleaning v1 (most impactful issues first)**
   * Remove duplicates (by photo_id and/or exact duplicate rows).
   * Filter invalid coordinates (nulls, out-of-range, obvious outliers).
   * Parse dates to a consistent timezone-aware format; flag impossible dates.

4. **Visualization app: working map**
   * Build a minimal map view (e.g., Folium) showing:
     * A **sample** of points (or density/heatmap if too many points).
     * Basic controls (zoom, layer toggle) and a quick data tooltip.

5. **First clustering baseline**
   * Pick **one** clustering algorithm (fast baseline): DBSCAN or K-Means.
   * Run on a **subset** first (to validate pipeline), store cluster labels.
   * Produce a simple cluster summary (number of clusters, noise/outliers count, cluster sizes).

### Deliverables by end of session

* Cleaning notebook/script (v1), with a short "issues found" list.
* Map app running and showing points.
* A first clustering run with basic stats saved (even if not yet meaningful).

---

## Session 2 (4h) — Cleaning completion + 3 clustering methods + first text mining ✅

**Goal:** Robust clustering comparison + clusters visible on the map + first automatic cluster descriptors.

### Tasks

1. **Finalize cleaning pipeline**
   * Confirm coherency checks (GPS + dates).
   * Decide final filtering rules (keep a log of what you drop and why).
   * Performance improvements: caching cleaned data (Parquet), reproducible scripts.

2. **Clustering experimentation (3 algorithms)**
   * Implement and run **three** approaches (recommended set):
     * **K-Means**
     * **Hierarchical/Agglomerative**
     * **DBSCAN**
   * Parameter search strategy (coarse → refine), focusing on:
     * Cluster size distribution (avoid huge "everything" cluster)
     * Outlier/noise rate (especially for DBSCAN)
     * Simple quality metrics (e.g., silhouette where applicable) + qualitative map inspection

3. **Cluster visualization on the map**
   * Display clusters with colors + legend.
   * Clicking a cluster (or point) should show:
     * Cluster ID, size, sample photos metadata (at least tags/title/date)
   * Add basic filters (min cluster size, toggle noise points).

4. **Text pattern mining v1 (to describe clusters)**
   * Preprocess text (tags + title/description):
     * stopwords (EN/FR), lowercasing, tokenization, removing "too frequent but useless" words
   * Implement **one** descriptor method:
     * TF or TF-IDF top terms per cluster (top 5–15)

### Deliverables by end of session

* Cleaned dataset export + documented cleaning rules.
* 3 clustering runs + initial parameter tuning notes.
* Map showing clusters.
* First "cluster description" output (top words per cluster).

---

## Session 3 (4h) — Final selection + 2 text-mining methods + temporal/event analysis + demo readiness

> 📋 **Detailed planning available in:** [`SESSION3_PLANNING.md`](SESSION3_PLANNING.md)

**Date:** February 2, 2026  
**Presentation:** February 2-6, 2026 (10 min demo + 5 min Q&A)

**Goal:** Finalize clustering recommendation with evidence, implement association rules for auto-naming, analyze temporal patterns, and deliver a bulletproof demo.

---

### ⏰ Time-Boxed Schedule (4 hours)

| Time | Duration | Task | Priority |
|------|----------|------|----------|
| 0:00 - 0:30 | 30 min | Task 1: Clustering finalization & comparison doc | 🔴 Critical |
| 0:30 - 1:30 | 60 min | Task 2: Association rules implementation | 🔴 Critical |
| 1:30 - 2:00 | 30 min | Task 3: Auto-naming integration in app | 🔴 Critical |
| 2:00 - 2:45 | 45 min | Task 4: Temporal/event analysis | 🔴 Critical |
| 2:45 - 3:15 | 30 min | Task 5: Map & demo polish | 🟡 Important |
| 3:15 - 4:00 | 45 min | Task 6: Presentation prep & dry-run | 🔴 Critical |

---

### Tasks Summary

1. **Clustering optimization + recommendation**
   * Final parameter tuning on the strongest candidates
   * Compare algorithms clearly (quality, sensitivity, scalability)
   * Choose and justify the **recommended** algorithm for Grand Lyon use
   * **Deliverable:** `reports/clustering_recommendation.md`

2. **Text mining v2 (2 methods total) + auto-naming**
   * Keep TF-IDF and add **Association rules** (frequent itemsets)
   * Define naming rule with fallback logic
   * Validate names with quick sanity checks
   * **Deliverable:** `reports/association_rules_summary.md`

3. **Integrate into the application**
   * Show cluster names on the map (labels in popup/side panel)
   * Provide a "cluster summary panel" (size, top terms, time distribution)
   * **Deliverable:** Updated `app/cluster_map.html`

4. **Temporal / event exploration**
   * Build time views: counts by day/week/month per cluster
   * Detect peaks → identify one-time vs recurring patterns
   * Produce 2–3 compelling examples (festival vs permanent spot)
   * **Deliverable:** `reports/temporal_analysis.md`

5. **Final demo & presentation preparation**
   * Freeze the pipeline: one command/notebook to reproduce results
   * Prepare demo script (10 min) + backup screenshots
   * Create a crisp methodology narrative
   * **Deliverable:** Demo-ready project

---

### 📋 Session 3 Deliverables Checklist

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
| Demo script | `reports/demo_script.md` | ⬜ |

---

### 🎯 Success Criteria (Milestone 3 - 10 points)

Per project requirements:
1. ✅ **Clustering optimization complete** with documented comparison
2. ✅ **2 text mining algorithms** (TF-IDF + Association Rules)
3. ✅ **Cluster names appear on application**
4. ✅ **Temporal exploration** with results visualized
5. ✅ **Demo-ready** with no bugs during presentation

---

### ⚠️ Risk Mitigation

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
