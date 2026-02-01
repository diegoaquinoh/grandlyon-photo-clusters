# Grand Lyon Photo Clusters

Automatic discovery and characterization of Points of Interest (POIs) and events from 400,000+ geolocated Flickr photos in the Lyon metropolitan area.

## Project Overview

| Phase                 | Description                                                                            |
| --------------------- | -------------------------------------------------------------------------------------- |
| **Data Processing**   | Ingest, clean, and filter raw photo data (geolocation, timestamps, text)               |
| **Clustering**        | Apply spatial clustering (K-Means, Hierarchical, DBSCAN) to identify areas of interest |
| **Text Mining**       | Generate cluster labels using TF-IDF and Association Rules on tags/titles              |
| **Temporal Analysis** | Distinguish permanent landmarks from one-time/recurring events                         |
| **Visualization**     | Interactive Folium map to explore discovered clusters                                  |

## Project Structure

```
├── data/           # Raw and processed datasets (not tracked in git)
├── src/            # Python source modules
├── notebooks/      # Jupyter notebooks for analysis
├── app/            # Streamlit web application
└── reports/        # Generated reports and visualizations
```

## Setup

**Option 1 – pip:**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Option 2 – Conda:**

```bash
conda env create -f environment.yml
conda activate grandlyon-photo-clusters
```

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
