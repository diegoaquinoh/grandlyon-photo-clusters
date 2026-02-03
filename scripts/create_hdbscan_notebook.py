#!/usr/bin/env python3
"""
Create HDBSCAN Experimentation Notebook

Creates a Jupyter notebook for comprehensive HDBSCAN parameter tuning.
"""

import json
from pathlib import Path

def create_notebook():
    """Create the HDBSCAN experimentation notebook."""
    
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Session 3: HDBSCAN Experimentation\n",
            "\n",
            "**Objective**: Comprehensive parameter tuning for HDBSCAN (Hierarchical DBSCAN).\n",
            "\n",
            "HDBSCAN advantages over DBSCAN:\n",
            "- Does not require eps parameter selection\n",
            "- Finds clusters of varying densities\n",
            "- More robust noise handling\n",
            "- Produces more stable clusterings\n",
            "\n",
            "Key parameters to tune:\n",
            "- **min_cluster_size**: Minimum points to form a cluster (most important)\n",
            "- **min_samples**: Core point neighborhood size (defaults to min_cluster_size)\n",
            "- **cluster_selection_epsilon**: Distance threshold\n",
            "- **cluster_selection_method**: 'eom' (Excess of Mass) or 'leaf'"
        ]
    })
    
    # Setup cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.insert(0, '..')\n",
            "import os\n",
            "\n",
            "# Set notebook directory as working directory\n",
            "notebook_dir = '/Users/diegoaquino/IF4/DataMining/grandlyon-photo-clusters/notebooks'\n",
            "os.chdir(notebook_dir)\n",
            "print(f\"Working directory: {os.getcwd()}\")\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from datetime import datetime\n",
            "import hdbscan\n",
            "from sklearn.metrics import silhouette_score\n",
            "\n",
            "from src.data_loader import load_cleaned_data\n",
            "from src.clustering import (\n",
            "    prepare_coordinates, \n",
            "    get_cluster_stats\n",
            ")\n",
            "\n",
            "plt.style.use('seaborn-v0_8-whitegrid')\n",
            "sns.set_palette('husl')\n",
            "\n",
            "print(\"Libraries loaded!\")"
        ]
    })
    
    # Data loading section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Load Data"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load cleaned data\n",
            "df = load_cleaned_data()\n",
            "print(f\"Loaded {len(df):,} photos\")\n",
            "\n",
            "# Prepare coordinates (unscaled for HDBSCAN - uses euclidean on lat/lon)\n",
            "coords = prepare_coordinates(df, scale=False)\n",
            "\n",
            "print(f\"Coordinate range: lat ({coords[:, 0].min():.4f}, {coords[:, 0].max():.4f})\")\n",
            "print(f\"                  lon ({coords[:, 1].min():.4f}, {coords[:, 1].max():.4f})\")"
        ]
    })
    
    # HDBSCAN parameter sweep section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. HDBSCAN Parameter Sweep: min_cluster_size\n",
            "\n",
            "The **min_cluster_size** is the most important parameter in HDBSCAN.\n",
            "It sets the minimum number of points to form a cluster.\n",
            "\n",
            "Let's sweep across a range of values and measure:\n",
            "- Number of clusters found\n",
            "- Noise percentage\n",
            "- Largest cluster size percentage\n",
            "- Silhouette score (on a sample for efficiency)"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Define parameter values to test\n",
            "min_cluster_sizes = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]\n",
            "\n",
            "# Pre-sample for fast silhouette calculation\n",
            "np.random.seed(42)\n",
            "sample_size = 10000\n",
            "sample_idx = np.random.choice(len(coords), min(sample_size, len(coords)), replace=False)\n",
            "\n",
            "def run_hdbscan_test(mcs, coords, sample_idx):\n",
            "    \"\"\"Run HDBSCAN and compute metrics.\"\"\"\n",
            "    clusterer = hdbscan.HDBSCAN(\n",
            "        min_cluster_size=mcs,\n",
            "        min_samples=None,  # Defaults to min_cluster_size\n",
            "        cluster_selection_epsilon=0.0,\n",
            "        cluster_selection_method='eom',\n",
            "        metric='euclidean'\n",
            "    )\n",
            "    labels = clusterer.fit_predict(coords)\n",
            "    stats = get_cluster_stats(labels)\n",
            "    \n",
            "    # Silhouette on sample (only for clustered points)\n",
            "    mask = labels[sample_idx] != -1\n",
            "    if mask.sum() > 50 and stats['n_clusters'] > 1:\n",
            "        sil = silhouette_score(coords[sample_idx][mask], labels[sample_idx][mask])\n",
            "    else:\n",
            "        sil = None\n",
            "    \n",
            "    return {\n",
            "        'min_cluster_size': mcs,\n",
            "        'n_clusters': stats['n_clusters'],\n",
            "        'noise_pct': round(stats['noise_percentage'], 1),\n",
            "        'largest_pct': round(stats['largest_cluster'] / len(labels) * 100, 1),\n",
            "        'median_size': stats['median_cluster_size'],\n",
            "        'silhouette': round(sil, 4) if sil else None\n",
            "    }\n",
            "\n",
            "print(f\"Testing {len(min_cluster_sizes)} min_cluster_size values...\")\n",
            "results = []\n",
            "for i, mcs in enumerate(min_cluster_sizes):\n",
            "    print(f\"  [{i+1}/{len(min_cluster_sizes)}] min_cluster_size={mcs}\", end=\" \")\n",
            "    result = run_hdbscan_test(mcs, coords, sample_idx)\n",
            "    results.append(result)\n",
            "    print(f\"→ clusters={result['n_clusters']}, noise={result['noise_pct']}%, sil={result['silhouette']}\")\n",
            "\n",
            "hdbscan_df = pd.DataFrame(results)\n",
            "hdbscan_df"
        ]
    })
    
    # Visualization cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Visualize Parameter Sweep Results"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualize HDBSCAN parameter sweep\n",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 12))\n",
            "\n",
            "# Number of clusters\n",
            "axes[0, 0].plot(hdbscan_df['min_cluster_size'], hdbscan_df['n_clusters'], 'o-', linewidth=2, markersize=8, color='#3498db')\n",
            "axes[0, 0].set_xlabel('min_cluster_size')\n",
            "axes[0, 0].set_ylabel('Number of Clusters')\n",
            "axes[0, 0].set_title('HDBSCAN: Clusters vs min_cluster_size')\n",
            "axes[0, 0].grid(True, alpha=0.3)\n",
            "\n",
            "# Noise percentage\n",
            "axes[0, 1].plot(hdbscan_df['min_cluster_size'], hdbscan_df['noise_pct'], 'o-', linewidth=2, markersize=8, color='#e74c3c')\n",
            "axes[0, 1].set_xlabel('min_cluster_size')\n",
            "axes[0, 1].set_ylabel('Noise %')\n",
            "axes[0, 1].set_title('HDBSCAN: Noise vs min_cluster_size')\n",
            "axes[0, 1].grid(True, alpha=0.3)\n",
            "\n",
            "# Largest cluster percentage\n",
            "axes[1, 0].plot(hdbscan_df['min_cluster_size'], hdbscan_df['largest_pct'], 'o-', linewidth=2, markersize=8, color='#9b59b6')\n",
            "axes[1, 0].set_xlabel('min_cluster_size')\n",
            "axes[1, 0].set_ylabel('Largest Cluster %')\n",
            "axes[1, 0].set_title('HDBSCAN: Largest Cluster vs min_cluster_size')\n",
            "axes[1, 0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')\n",
            "axes[1, 0].grid(True, alpha=0.3)\n",
            "\n",
            "# Silhouette score\n",
            "valid_sil = hdbscan_df.dropna(subset=['silhouette'])\n",
            "axes[1, 1].plot(valid_sil['min_cluster_size'], valid_sil['silhouette'], 'o-', linewidth=2, markersize=8, color='#2ecc71')\n",
            "axes[1, 1].set_xlabel('min_cluster_size')\n",
            "axes[1, 1].set_ylabel('Silhouette Score')\n",
            "axes[1, 1].set_title('HDBSCAN: Silhouette vs min_cluster_size')\n",
            "axes[1, 1].grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../reports/hdbscan_parameter_sweep.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "# Find best min_cluster_size\n",
            "best_hdbscan = hdbscan_df.dropna(subset=['silhouette']).loc[hdbscan_df['silhouette'].dropna().idxmax()]\n",
            "print(f\"\\n✅ Best HDBSCAN parameter (by silhouette):\")\n",
            "print(f\"   min_cluster_size = {int(best_hdbscan['min_cluster_size'])}, silhouette = {best_hdbscan['silhouette']:.4f}\")\n",
            "print(f\"   Resulting in {int(best_hdbscan['n_clusters'])} clusters with {best_hdbscan['noise_pct']:.1f}% noise\")"
        ]
    })
    
    # min_samples sweep section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Secondary Parameter Sweep: min_samples\n",
            "\n",
            "With the best min_cluster_size fixed, let's explore **min_samples**.\n",
            "This controls how conservative the algorithm is in forming clusters."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Use the best min_cluster_size from previous sweep\n",
            "best_mcs = int(best_hdbscan['min_cluster_size'])\n",
            "min_samples_values = [3, 5, 7, 10, 15, best_mcs, best_mcs + 10]\n",
            "\n",
            "def run_hdbscan_ms_test(mcs, ms, coords, sample_idx):\n",
            "    \"\"\"Run HDBSCAN with specific min_samples.\"\"\"\n",
            "    clusterer = hdbscan.HDBSCAN(\n",
            "        min_cluster_size=mcs,\n",
            "        min_samples=ms,\n",
            "        cluster_selection_epsilon=0.0,\n",
            "        cluster_selection_method='eom',\n",
            "        metric='euclidean'\n",
            "    )\n",
            "    labels = clusterer.fit_predict(coords)\n",
            "    stats = get_cluster_stats(labels)\n",
            "    \n",
            "    mask = labels[sample_idx] != -1\n",
            "    if mask.sum() > 50 and stats['n_clusters'] > 1:\n",
            "        sil = silhouette_score(coords[sample_idx][mask], labels[sample_idx][mask])\n",
            "    else:\n",
            "        sil = None\n",
            "    \n",
            "    return {\n",
            "        'min_cluster_size': mcs,\n",
            "        'min_samples': ms,\n",
            "        'n_clusters': stats['n_clusters'],\n",
            "        'noise_pct': round(stats['noise_percentage'], 1),\n",
            "        'largest_pct': round(stats['largest_cluster'] / len(labels) * 100, 1),\n",
            "        'silhouette': round(sil, 4) if sil else None\n",
            "    }\n",
            "\n",
            "print(f\"Testing {len(min_samples_values)} min_samples values (with min_cluster_size={best_mcs})...\")\n",
            "ms_results = []\n",
            "for i, ms in enumerate(min_samples_values):\n",
            "    print(f\"  [{i+1}/{len(min_samples_values)}] min_samples={ms}\", end=\" \")\n",
            "    result = run_hdbscan_ms_test(best_mcs, ms, coords, sample_idx)\n",
            "    ms_results.append(result)\n",
            "    print(f\"→ clusters={result['n_clusters']}, noise={result['noise_pct']}%, sil={result['silhouette']}\")\n",
            "\n",
            "ms_df = pd.DataFrame(ms_results)\n",
            "ms_df"
        ]
    })
    
    # min_samples visualization
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualize min_samples sweep\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
            "\n",
            "# Number of clusters\n",
            "axes[0].bar(range(len(ms_df)), ms_df['n_clusters'], color='#3498db')\n",
            "axes[0].set_xticks(range(len(ms_df)))\n",
            "axes[0].set_xticklabels([str(x) for x in ms_df['min_samples']])\n",
            "axes[0].set_xlabel('min_samples')\n",
            "axes[0].set_ylabel('Number of Clusters')\n",
            "axes[0].set_title(f'Clusters vs min_samples (mcs={best_mcs})')\n",
            "\n",
            "# Noise percentage\n",
            "axes[1].bar(range(len(ms_df)), ms_df['noise_pct'], color='#e74c3c')\n",
            "axes[1].set_xticks(range(len(ms_df)))\n",
            "axes[1].set_xticklabels([str(x) for x in ms_df['min_samples']])\n",
            "axes[1].set_xlabel('min_samples')\n",
            "axes[1].set_ylabel('Noise %')\n",
            "axes[1].set_title(f'Noise vs min_samples (mcs={best_mcs})')\n",
            "\n",
            "# Silhouette\n",
            "valid_ms = ms_df.dropna(subset=['silhouette'])\n",
            "axes[2].bar(range(len(valid_ms)), valid_ms['silhouette'], color='#2ecc71')\n",
            "axes[2].set_xticks(range(len(valid_ms)))\n",
            "axes[2].set_xticklabels([str(x) for x in valid_ms['min_samples']])\n",
            "axes[2].set_xlabel('min_samples')\n",
            "axes[2].set_ylabel('Silhouette Score')\n",
            "axes[2].set_title(f'Silhouette vs min_samples (mcs={best_mcs})')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../reports/hdbscan_min_samples_sweep.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ]
    })
    
    # Results table
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 5. Complete Results Table"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display full results table\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"HDBSCAN PARAMETER SWEEP RESULTS\")\n",
            "print(\"=\"*70)\n",
            "print(\"\\nmin_cluster_size sweep:\")\n",
            "print(hdbscan_df.to_string(index=False))\n",
            "print(\"\\nmin_samples sweep (with best min_cluster_size):\")\n",
            "print(ms_df.to_string(index=False))"
        ]
    })
    
    # Save results section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 6. Save Results"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Save parameter sweep results\n",
            "hdbscan_df.to_csv('../reports/hdbscan_mcs_tuning.csv', index=False)\n",
            "ms_df.to_csv('../reports/hdbscan_ms_tuning.csv', index=False)\n",
            "\n",
            "# Find overall best configuration\n",
            "best_ms_row = ms_df.dropna(subset=['silhouette']).loc[ms_df['silhouette'].dropna().idxmax()]\n",
            "\n",
            "# Save best parameters summary\n",
            "best_params = {\n",
            "    'timestamp': datetime.now().isoformat(),\n",
            "    'hdbscan': {\n",
            "        'min_cluster_size': int(best_ms_row['min_cluster_size']),\n",
            "        'min_samples': int(best_ms_row['min_samples']),\n",
            "        'silhouette': float(best_ms_row['silhouette']),\n",
            "        'n_clusters': int(best_ms_row['n_clusters']),\n",
            "        'noise_pct': float(best_ms_row['noise_pct'])\n",
            "    }\n",
            "}\n",
            "\n",
            "import json\n",
            "with open('../reports/best_hdbscan_params.json', 'w') as f:\n",
            "    json.dump(best_params, f, indent=2)\n",
            "\n",
            "print(\"✅ All results saved to reports/\")\n",
            "print(f\"   - hdbscan_mcs_tuning.csv\")\n",
            "print(f\"   - hdbscan_ms_tuning.csv\")\n",
            "print(f\"   - hdbscan_parameter_sweep.png\")\n",
            "print(f\"   - hdbscan_min_samples_sweep.png\")\n",
            "print(f\"   - best_hdbscan_params.json\")"
        ]
    })
    
    # Summary section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 7. Summary & Recommendations"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=\" * 70)\n",
            "print(\"HDBSCAN EXPERIMENTATION SUMMARY\")\n",
            "print(\"=\" * 70)\n",
            "print()\n",
            "print(\"Best parameters found:\")\n",
            "print()\n",
            "print(f\"  min_cluster_size: {int(best_ms_row['min_cluster_size'])}\")\n",
            "print(f\"  min_samples:      {int(best_ms_row['min_samples'])}\")\n",
            "print(f\"  Silhouette:       {best_ms_row['silhouette']:.4f}\")\n",
            "print(f\"  Clusters:         {int(best_ms_row['n_clusters'])}\")\n",
            "print(f\"  Noise:            {best_ms_row['noise_pct']:.1f}%\")\n",
            "print()\n",
            "print(\"=\" * 70)\n",
            "print(\"Key observations:\")\n",
            "print(\"-\" * 70)\n",
            "print(\"1. HDBSCAN automatically detects clusters of varying density\")\n",
            "print(\"2. No need to specify 'eps' like in DBSCAN\")\n",
            "print(\"3. min_cluster_size is the most influential parameter\")\n",
            "print(\"4. Higher min_cluster_size = fewer, larger clusters + more noise\")\n",
            "print(\"5. min_samples controls sensitivity to noise points\")\n",
            "print()\n",
            "print(\"Recommendation: Use these parameters in the main pipeline!\")\n",
            "print(\"=\" * 70)"
        ]
    })
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write notebook
    output_path = Path("../notebooks/03_hdbscan_experimentation.ipynb")
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=4)
    
    print(f"✅ Created notebook: {output_path}")
    return output_path

if __name__ == "__main__":
    create_notebook()
