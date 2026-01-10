
## Session 1 (4h) — Data understanding + first end-to-end baseline

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

* Cleaning notebook/script (v1), with a short “issues found” list.
* Map app running and showing points.
* A first clustering run with basic stats saved (even if not yet meaningful).

---

## Session 2 (4h) — Cleaning completion + 3 clustering methods + first text mining

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

     * Cluster size distribution (avoid huge “everything” cluster)
     * Outlier/noise rate (especially for DBSCAN)
     * Simple quality metrics (e.g., silhouette where applicable) + qualitative map inspection

3. **Cluster visualization on the map**

   * Display clusters with colors + legend.
   * Clicking a cluster (or point) should show:

     * Cluster ID, size, sample photos metadata (at least tags/title/date)
   * Add basic filters (min cluster size, toggle noise points).

4. **Text pattern mining v1 (to describe clusters)**

   * Preprocess text (tags + title/description):

     * stopwords (EN/FR), lowercasing, tokenization, removing “too frequent but useless” words
   * Implement **one** descriptor method:

     * TF or TF-IDF top terms per cluster (top 5–15)

### Deliverables by end of session

* Cleaned dataset export + documented cleaning rules.
* 3 clustering runs + initial parameter tuning notes.
* Map showing clusters.
* First “cluster description” output (top words per cluster).

---

## Session 3 (4h) — Final selection + 2 text-mining methods + temporal/event analysis + demo readiness

**Goal:** Decide and justify the best clustering method, auto-name clusters, analyze time patterns (events), and make the demo bulletproof.

### Tasks

1. **Clustering optimization + recommendation**

   * Final parameter tuning on the strongest candidates.
   * Compare algorithms clearly:

     * Result quality (map interpretability, coherent areas of interest)
     * Sensitivity to parameters
     * Runtime/complexity + scalability considerations
   * Choose and justify the **recommended** algorithm for Grand Lyon use.

2. **Text mining v2 (2 methods total) + auto-naming**

   * Keep TF-IDF (or improve it) and add a second method, e.g.:

     * **Association rules** (frequent itemsets of tags/terms) to extract meaningful combinations
   * Define a naming rule:

     * Example: “Top 3 TF-IDF terms” OR “Best association rule terms” + fallback logic
   * Validate names with quick sanity checks (manually inspect a few clusters).

3. **Integrate into the application**

   * Show cluster names on the map (labels in popup/side panel).
   * Provide a “cluster summary panel” (size, top terms, top term pairs, time distribution).

4. **Temporal / event exploration**

   * For each cluster (area), build time views:

     * counts by day/week/month
     * detect peaks (simple peak detection) → identify one-time vs recurring patterns
   * Produce 2–3 compelling examples (e.g., a known festival cluster vs permanent tourist spot).

5. **Final demo & presentation preparation**

   * Freeze the pipeline: one command/notebook to reproduce results.
   * Prepare demo script (10 min) + backup screenshots.
   * Create a crisp methodology narrative:

     * data → cleaning → clustering comparison → chosen method → text labeling → time/event insights → usefulness for Grand Lyon

### Deliverables by end of session

* Final recommended clustering approach with evidence.
* Two text mining methods implemented + cluster names in the app.
* Temporal/event analysis results + visuals.
* Demo-ready project (no surprises, reproducible, stable).
