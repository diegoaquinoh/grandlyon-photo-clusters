"""
HDBSCAN Clustering Experimentation
==================================

This script adds HDBSCAN as a clustering option with parameter tuning,
and compares it with the best K-Means configuration from Session 2.

Run this script to experiment with HDBSCAN parameters and see how it
compares to K-Means for the photo clustering task.
"""

import sys
sys.path.insert(0, '..')
import os

# Set notebook directory as working directory
notebook_dir = '/Users/diegoaquino/IF4/DataMining/grandlyon-photo-clusters/notebooks'
os.chdir(notebook_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

from src.data_loader import load_cleaned_data
from src.clustering import (
    prepare_coordinates, 
    run_hdbscan, find_optimal_hdbscan,
    run_kmeans,
    get_cluster_stats, calculate_quality_metrics
)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

print("=" * 70)
print("HDBSCAN CLUSTERING EXPERIMENTATION")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = load_cleaned_data()
print(f"\nLoaded {len(df):,} photos")

# Prepare coordinates
coords = prepare_coordinates(df, scale=False)
coords_scaled = prepare_coordinates(df, scale=True)

print(f"Coordinate range: lat ({coords[:, 0].min():.4f}, {coords[:, 0].max():.4f})")
print(f"                  lon ({coords[:, 1].min():.4f}, {coords[:, 1].max():.4f})")

# =============================================================================
# 2. HDBSCAN PARAMETER TUNING
# =============================================================================
print("\n" + "=" * 70)
print("HDBSCAN PARAMETER TUNING")
print("=" * 70)
print("""
Key parameters:
- **min_cluster_size**: Minimum points required to form a cluster (most important)
- **min_samples**: Number of samples in neighborhood for core points
  (smaller = more points clustered, larger = more noise)
""")

# Test min_cluster_size values
min_cluster_sizes = [10, 15, 20, 30, 50, 75, 100]
min_samples_values = [None]  # None means same as min_cluster_size

print(f"\nTesting min_cluster_sizes: {min_cluster_sizes}")
print(f"min_samples: same as min_cluster_size\n")

hdbscan_results = find_optimal_hdbscan(
    coords, 
    min_cluster_sizes=min_cluster_sizes,
    min_samples_values=min_samples_values
)

hdbscan_df = pd.DataFrame(hdbscan_results)
print("\n" + "-" * 70)
print("HDBSCAN Results:")
print(hdbscan_df.to_string(index=False))

# =============================================================================
# 3. VISUALIZE HDBSCAN RESULTS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Number of clusters
axes[0, 0].plot(hdbscan_df['min_cluster_size'], hdbscan_df['n_clusters'], 'o-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('min_cluster_size')
axes[0, 0].set_ylabel('Number of Clusters')
axes[0, 0].set_title('HDBSCAN: Clusters vs min_cluster_size')

# Noise percentage
axes[0, 1].plot(hdbscan_df['min_cluster_size'], hdbscan_df['noise_pct'], 'o-', linewidth=2, markersize=8, color='orange')
axes[0, 1].set_xlabel('min_cluster_size')
axes[0, 1].set_ylabel('Noise %')
axes[0, 1].set_title('HDBSCAN: Noise vs min_cluster_size')

# Largest cluster percentage
axes[1, 0].plot(hdbscan_df['min_cluster_size'], hdbscan_df['largest_pct'], 'o-', linewidth=2, markersize=8, color='red')
axes[1, 0].set_xlabel('min_cluster_size')
axes[1, 0].set_ylabel('Largest Cluster %')
axes[1, 0].set_title('HDBSCAN: Largest Cluster vs min_cluster_size')
axes[1, 0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')

# Silhouette score
valid_sil = hdbscan_df[hdbscan_df['silhouette'].notna()]
axes[1, 1].plot(valid_sil['min_cluster_size'], valid_sil['silhouette'], 'o-', linewidth=2, markersize=8, color='green')
axes[1, 1].set_xlabel('min_cluster_size')
axes[1, 1].set_ylabel('Silhouette Score')
axes[1, 1].set_title('HDBSCAN: Silhouette vs min_cluster_size')

plt.tight_layout()
plt.savefig('../reports/hdbscan_parameter_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# Find best HDBSCAN config
valid_results = hdbscan_df[hdbscan_df['silhouette'].notna()]
if len(valid_results) > 0:
    best_hdbscan = valid_results.loc[valid_results['silhouette'].idxmax()]
    print(f"\n✅ Best HDBSCAN parameters (by silhouette):")
    print(f"   min_cluster_size = {int(best_hdbscan['min_cluster_size'])}")
    print(f"   silhouette = {best_hdbscan['silhouette']:.4f}")
    print(f"   n_clusters = {int(best_hdbscan['n_clusters'])}")
    print(f"   noise = {best_hdbscan['noise_pct']:.1f}%")

# =============================================================================
# 4. COMPARISON: HDBSCAN vs K-MEANS
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON: HDBSCAN vs K-MEANS")
print("=" * 70)

# Best K-Means from previous experiments (k=120 had best silhouette based on notebook)
best_k = 120
print(f"\nRunning K-Means with best k={best_k} from previous experiments...")

kmeans_labels = run_kmeans(coords_scaled, n_clusters=best_k)
kmeans_stats = get_cluster_stats(kmeans_labels)
kmeans_metrics = calculate_quality_metrics(coords_scaled, kmeans_labels)

# Run HDBSCAN with best parameters
best_mcs = int(best_hdbscan['min_cluster_size'])
print(f"Running HDBSCAN with best min_cluster_size={best_mcs}...")

hdbscan_labels = run_hdbscan(coords, min_cluster_size=best_mcs)
hdbscan_stats = get_cluster_stats(hdbscan_labels)
hdbscan_metrics = calculate_quality_metrics(coords, hdbscan_labels)

# Create comparison table
comparison_data = {
    'Algorithm': ['K-Means', 'HDBSCAN'],
    'Parameters': [f'k={best_k}', f'min_cluster_size={best_mcs}'],
    'N Clusters': [kmeans_stats['n_clusters'], hdbscan_stats['n_clusters']],
    'Noise %': [0.0, round(hdbscan_stats['noise_percentage'], 2)],
    'Largest Cluster': [kmeans_stats['largest_cluster'], hdbscan_stats['largest_cluster']],
    'Largest %': [
        round(kmeans_stats['largest_cluster'] / len(df) * 100, 1),
        round(hdbscan_stats['largest_cluster'] / len(df) * 100, 1)
    ],
    'Median Size': [kmeans_stats['median_cluster_size'], hdbscan_stats['median_cluster_size']],
    'Silhouette': [
        round(kmeans_metrics['silhouette'], 4) if kmeans_metrics.get('silhouette') else None,
        round(hdbscan_metrics['silhouette'], 4) if hdbscan_metrics.get('silhouette') else None
    ],
    'Davies-Bouldin': [
        round(kmeans_metrics['davies_bouldin'], 4) if kmeans_metrics.get('davies_bouldin') else None,
        round(hdbscan_metrics['davies_bouldin'], 4) if hdbscan_metrics.get('davies_bouldin') else None
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + "-" * 70)
print("COMPARISON RESULTS:")
print("-" * 70)
print(comparison_df.to_string(index=False))

# =============================================================================
# 5. CONCLUSION
# =============================================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

kmeans_sil = kmeans_metrics.get('silhouette', 0) or 0
hdbscan_sil = hdbscan_metrics.get('silhouette', 0) or 0

if hdbscan_sil > kmeans_sil:
    winner = "HDBSCAN"
    diff = hdbscan_sil - kmeans_sil
    print(f"\n🏆 HDBSCAN outperforms K-Means!")
    print(f"   HDBSCAN silhouette: {hdbscan_sil:.4f}")
    print(f"   K-Means silhouette: {kmeans_sil:.4f}")
    print(f"   Improvement: +{diff:.4f} ({diff/kmeans_sil*100:.1f}% improvement)")
    
    print("\n📊 HDBSCAN Advantages:")
    print("   - No need to specify number of clusters upfront")
    print("   - Automatically identifies noise/outlier points")
    print("   - Can find clusters of varying densities")
    print("   - More robust to parameter selection")
    
    if hdbscan_stats['noise_percentage'] > 0:
        print(f"\n   Note: HDBSCAN identified {hdbscan_stats['noise_percentage']:.1f}% of points as noise")
        print("   These are likely isolated photos not part of any tourist hotspot")
else:
    winner = "K-Means"
    print(f"\n🏆 K-Means performs better for this dataset!")
    print(f"   K-Means silhouette: {kmeans_sil:.4f}")
    print(f"   HDBSCAN silhouette: {hdbscan_sil:.4f}")

print(f"\n💡 Recommendation: Use {winner} for the final clustering pipeline")

# Save comparison results
comparison_df.to_csv('../reports/hdbscan_vs_kmeans_comparison.csv', index=False)
print(f"\n📁 Saved comparison to: ../reports/hdbscan_vs_kmeans_comparison.csv")

# =============================================================================
# 6. VISUALIZE CLUSTER DISTRIBUTIONS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means cluster size distribution
kmeans_sizes = sorted([int((kmeans_labels == l).sum()) for l in range(best_k)], reverse=True)
axes[0].bar(range(len(kmeans_sizes)), kmeans_sizes, color='steelblue', alpha=0.7)
axes[0].set_xlabel('Cluster Rank')
axes[0].set_ylabel('Cluster Size')
axes[0].set_title(f'K-Means (k={best_k}) Cluster Size Distribution')
axes[0].set_xlim(-1, 50)  # Show top 50

# HDBSCAN cluster size distribution
unique_labels = set(hdbscan_labels) - {-1}
hdbscan_sizes = sorted([int((hdbscan_labels == l).sum()) for l in unique_labels], reverse=True)
axes[1].bar(range(len(hdbscan_sizes)), hdbscan_sizes, color='seagreen', alpha=0.7)
axes[1].set_xlabel('Cluster Rank')
axes[1].set_ylabel('Cluster Size')
axes[1].set_title(f'HDBSCAN (min_cluster_size={best_mcs}) Cluster Size Distribution')
axes[1].set_xlim(-1, 50)  # Show top 50

plt.tight_layout()
plt.savefig('../reports/clustering_comparison_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ HDBSCAN experimentation complete!")
