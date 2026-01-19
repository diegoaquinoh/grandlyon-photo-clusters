# Grand Lyon Photo Clusters — Progress Report

**Date:** January 11, 2026  
**Project:** IF4 Data Mining — INSA Lyon 2025-2026

---

## Executive Summary

This project aims to automatically discover and characterize Points of Interest (POIs) and events from 400,000+ geolocated Flickr photos in the Lyon metropolitan area. After two sessions of work, we have completed **data cleaning**, **clustering experimentation**, **map visualization**, and **initial text mining**.

---

## Session 1 Deliverables ✅

| Task | Status | Notes |
|------|--------|-------|
| Project setup & structure | ✅ Complete | `data/`, `src/`, `notebooks/`, `app/`, `reports/` |
| Data ingestion & profiling | ✅ Complete | 420,240 photos loaded |
| Data cleaning v1 | ✅ Complete | Pipeline with logging |
| Working map visualization | ✅ Complete | `app/map.html` |
| First clustering baseline | ✅ Complete | DBSCAN baseline saved |

---

## Session 2 Deliverables ✅

| Task | Status | Notes |
|------|--------|-------|
| Finalize cleaning pipeline | ✅ Complete | 167,578 photos retained |
| 3 Clustering algorithms | ✅ Complete | DBSCAN, K-Means, Hierarchical |
| Parameter tuning | ✅ Complete | Sweeps with visualizations |
| Cluster map visualization | ✅ Complete | `app/cluster_map.html` |
| Text pattern mining (TF-IDF) | ✅ Complete | 100 clusters described |

---

## Data Cleaning Results

**Source:** `reports/cleaning_log.json`

| Step | Rows Before | Rows After | Removed | % |
|------|-------------|------------|---------|---|
| GPS validation | 420,240 | 420,240 | 0 | 0% |
| Date validation | 420,240 | 419,826 | 414 | 0.1% |
| Date coherency | 419,826 | 418,898 | 928 | 0.22% |
| Deduplication | 418,898 | 167,578 | 251,320 | **60%** |
| Lyon bbox filter | 167,578 | 167,578 | 0 | 0% |

**Final dataset:** 167,578 photos (39.88% retention rate)

> **Key finding:** 60% of original data were duplicates (same photo ID).

---

## Clustering Comparison

**Source:** `reports/clustering_comparison.csv`

| Algorithm | Parameters | Silhouette | Largest Cluster % | Recommendation |
|-----------|------------|------------|-------------------|----------------|
| **K-Means** | k=100, scaled | **0.535** | 9.0% | ✅ **SELECTED** |
| Hierarchical | k=100, ward | 0.510 | 9.5% | Good alternative |
| DBSCAN | eps=0.005, min_samples=10 | 0.404 | 83.8% | ⚠️ Not suitable |

### Why K-Means?

1. **Highest silhouette score** (0.535) — best cluster separation
2. **Balanced cluster sizes** — largest cluster only 9% of data
3. **Fast and scalable** — handles 167k points efficiently

### Why not DBSCAN?

DBSCAN creates one mega-cluster containing 83.8% of points — not useful for identifying distinct areas of interest.

---

## Text Mining Results

**Source:** `reports/text_mining_summary.md`

**Method:** TF-IDF with unigrams and bigrams  
**Clusters analyzed:** 100

### Top 10 Largest Clusters

| Cluster | Size | Top Descriptive Terms |
|---------|------|-----------------------|
| 5 | 15,157 | chaos, ddc, demeureduchaos, abodeofchaos |
| 13 | 8,401 | saint jean, vieuxlyon, cathedral |
| 21 | 8,303 | terreaux, beaux arts, bartholdi |
| 36 | 8,257 | bellecour, place bellecour |
| 49 | 7,996 | basilique, fourviere, dame |
| 98 | 6,547 | vieuxlyon, miniature, traboule |
| 6 | 5,671 | pasted paper, croixrousse, streetart |
| 91 | 5,349 | jacobins, place jacobins |
| 61 | 5,250 | opera, saxotaz, croixrousse |
| 92 | 5,045 | cordeliers, bourse, celebrity |

### Identified Lyon Landmarks

The clusters successfully identify major Lyon POIs:
- **Place Bellecour** (Cluster 36)
- **Basilique de Fourvière** (Cluster 49)
- **Vieux Lyon / Saint Jean** (Clusters 13, 98)
- **Place des Terreaux** (Cluster 21)
- **Musée des Confluences** (Cluster 47)
- **Demeure du Chaos** (Cluster 5)

---

## Generated Files

### Source Code (`src/`)

| File | Description |
|------|-------------|
| `data_loader.py` | Data ingestion and cleaning pipeline |
| `clustering.py` | DBSCAN, K-Means, Hierarchical implementations |
| `text_mining.py` | TF-IDF analysis and preprocessing |
| `map_visualization.py` | Folium map generation |

### Reports (`reports/`)

| File | Description |
|------|-------------|
| `cleaning_log.json` | Detailed cleaning statistics |
| `clustering_comparison.csv` | Algorithm comparison table |
| `clustering_decisions.md` | Rationale for algorithm selection |
| `text_mining_summary.md` | TF-IDF results for all clusters |
| `cluster_descriptors.json` | Machine-readable cluster terms |
| `*_parameter_sweep.png` | Parameter tuning visualizations |

### Application (`app/`)

| File | Size | Description |
|------|------|-------------|
| `map.html` | 6.7 MB | Photo density heatmap |
| `cluster_map.html` | 32.9 MB | Interactive cluster visualization |

---

## Session 3 — Remaining Work

| Task | Priority | Estimated Effort |
|------|----------|------------------|
| Final clustering justification | High | 30 min |
| Association Rules (2nd text mining method) | High | 1 hour |
| Auto-naming clusters in app UI | Medium | 30 min |
| Temporal/event analysis | High | 1.5 hours |
| Demo preparation & freeze | Critical | 30 min |

### Key Session 3 Objectives

1. **Compare and justify** the recommended clustering algorithm with evidence
2. **Implement Association Rules** to find meaningful term combinations
3. **Analyze temporal patterns** to distinguish:
   - Permanent landmarks (constant activity)
   - One-time events (single peak)
   - Recurring events (e.g., Fête des Lumières)
4. **Prepare demo** with reproducible pipeline and backup screenshots

---

## Notes for Final Presentation

- **Main data issue discovered:** 60% duplicates
- **Best algorithm:** K-Means (k=100) with silhouette = 0.535
- **DBSCAN limitation:** Creates mega-cluster (83.8%) — unsuitable for geographic POI discovery
- **Validation:** TF-IDF terms match known Lyon landmarks

---

*Report generated: January 11, 2026*
