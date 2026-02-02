# Grand Lyon Photo Clusters

Automatic discovery and characterization of Points of Interest (POIs) and events from 400,000+ geolocated Flickr photos in the Lyon metropolitan area.

## 🎯 Key Results

| Metric                                | Value   |
| ------------------------------------- | ------- |
| Photos analyzed                       | 140,040 |
| Clusters discovered                   | 924     |
| Permanent landmarks                   | 265     |
| Recurring events (🎄 Dec / ☀️ Summer) | 439     |
| One-time events                       | 220     |

## 🗺️ Interactive Map

Explore the clusters with the enhanced interactive map:

```bash
open app/cluster_map_v2.html
```

**Features:**

- 📅 **Month slider** - See which clusters are active in each month
- 🏷️ **Type filters** - Filter by Landmark, December, Summer, One-time
- 🔍 **Search** - Find clusters by name or keyword
- 📊 **Peak month info** - Popups show temporal patterns

## Project Structure

```
├── data/              # Raw and processed datasets
├── src/               # Python modules
│   ├── data_loader.py   # Data cleaning & filtering
│   ├── clustering.py    # HDBSCAN clustering
│   ├── text_mining.py   # TF-IDF & association rules
│   ├── temporal_analysis.py  # Temporal classification
│   └── map_visualization.py  # Folium map generation
├── scripts/           # Pipeline scripts
│   ├── run_full_pipeline.py      # Complete Session 3 pipeline
│   └── create_enhanced_map_v2.py # Enhanced map with month slider
├── notebooks/         # Jupyter notebooks for experimentation
├── app/               # Interactive map outputs
└── reports/           # Generated reports & visualizations
```

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run full pipeline (default: HDBSCAN with min_cluster_size=120)
python scripts/run_full_pipeline.py

# Choose a different algorithm
python scripts/run_full_pipeline.py --algorithm hdbscan --min-cluster-size 120
python scripts/run_full_pipeline.py --algorithm dbscan --eps 0.005 --min-samples 10
python scripts/run_full_pipeline.py --algorithm kmeans --n-clusters 50
python scripts/run_full_pipeline.py --algorithm hierarchical --n-clusters 50

# Regenerate map only
python scripts/create_enhanced_map_v2.py
```

### Algorithm Options

| Algorithm      | Default Params                 | Description                                   |
| -------------- | ------------------------------ | --------------------------------------------- |
| `hdbscan`      | `--min-cluster-size 120`       | Hierarchical density clustering (recommended) |
| `dbscan`       | `--eps 0.005 --min-samples 10` | Density-based clustering                      |
| `kmeans`       | `-k 50`                        | K-Means (requires specifying k)               |
| `hierarchical` | `-k 50`                        | Agglomerative clustering                      |

## Pipeline Stages

| Stage                | Script/Module                       | Output                             |
| -------------------- | ----------------------------------- | ---------------------------------- |
| 1. Data Cleaning     | `src/data_loader.py`                | `data/flickr_cleaned.parquet`      |
| 2. Clustering        | `src/clustering.py` (HDBSCAN)       | `data/flickr_clustered.csv`        |
| 3. Text Mining       | `src/text_mining.py`                | `reports/cluster_descriptors.json` |
| 4. Temporal Analysis | `src/temporal_analysis.py`          | `reports/temporal_analysis.md`     |
| 5. Map Generation    | `scripts/create_enhanced_map_v2.py` | `app/cluster_map_v2.html`          |

## Cluster Types

| Type               | Icon | Description                                                  |
| ------------------ | ---- | ------------------------------------------------------------ |
| Permanent Landmark | 🏛️   | Stable activity year-round (e.g., Fourvière, Parc Tête d'Or) |
| December Event     | 🎄   | Fête des Lumières pattern (>25% December activity)           |
| Summer Hotspot     | ☀️   | Tourism peak in July-August                                  |
| Seasonal           | 📅   | Other recurring patterns                                     |
| One-time Event     | ⚡   | Single event spike                                           |

## Data Format

Each photo record contains:

```
⟨photo_id, user_id, latitude, longitude, tags, description, dates⟩
```

Access photos at: `https://www.flickr.com/photos/<user_id>/<photo_id>`

## Team

- Diego Aquino

---

_IF4 Data Mining Project – INSA Lyon 2025-2026_
