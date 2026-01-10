"""
Clustering module for the Grand Lyon Photo Clusters project.
Provides utilities for spatial clustering of photo locations.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path
from typing import Tuple, Optional
import json
from datetime import datetime

from .data_loader import load_cleaned_data, PROJECT_ROOT

# Output paths
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"


def prepare_coordinates(df: pd.DataFrame) -> np.ndarray:
    """
    Extract and prepare coordinates for clustering.
    
    Args:
        df: DataFrame with 'lat' and 'long' columns
    
    Returns:
        NumPy array of shape (n_samples, 2) with [lat, lon] coordinates
    """
    coords = df[['lat', 'long']].values
    return coords


def run_dbscan(
    coords: np.ndarray,
    eps: float = 0.005,
    min_samples: int = 10,
    scale_coords: bool = False
) -> np.ndarray:
    """
    Run DBSCAN clustering on coordinates.
    
    DBSCAN is well-suited for geographic clustering because:
    - It can find clusters of arbitrary shape
    - It identifies noise/outliers automatically
    - No need to specify number of clusters upfront
    
    Args:
        coords: Array of [lat, lon] coordinates
        eps: Maximum distance between points in a cluster (in degrees)
             0.005 degrees ≈ ~500m at Lyon's latitude
        min_samples: Minimum points to form a cluster
        scale_coords: Whether to standardize coordinates first
    
    Returns:
        Array of cluster labels (-1 = noise)
    """
    if scale_coords:
        scaler = StandardScaler()
        coords = scaler.fit_transform(coords)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(coords)
    
    return labels


def get_cluster_stats(labels: np.ndarray) -> dict:
    """
    Calculate summary statistics for clustering results.
    
    Args:
        labels: Array of cluster labels
    
    Returns:
        Dictionary of statistics
    """
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = (labels == -1).sum()
    
    # Cluster size distribution (excluding noise)
    cluster_sizes = []
    for label in unique_labels:
        if label != -1:
            cluster_sizes.append(int((labels == label).sum()))
    
    cluster_sizes = sorted(cluster_sizes, reverse=True)
    
    return {
        'n_clusters': n_clusters,
        'n_noise': int(n_noise),
        'noise_percentage': float(n_noise / len(labels) * 100),
        'total_points': len(labels),
        'clustered_points': len(labels) - int(n_noise),
        'largest_cluster': int(cluster_sizes[0]) if cluster_sizes else 0,
        'smallest_cluster': int(cluster_sizes[-1]) if cluster_sizes else 0,
        'median_cluster_size': int(np.median(cluster_sizes)) if cluster_sizes else 0,
        'cluster_sizes_top10': [int(x) for x in cluster_sizes[:10]]
    }


def run_baseline_clustering(
    df: Optional[pd.DataFrame] = None,
    eps: float = 0.005,
    min_samples: int = 10,
    save_results: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Run the baseline DBSCAN clustering on photo locations.
    
    Args:
        df: DataFrame with photo data (loads cleaned data if None)
        eps: DBSCAN epsilon parameter (distance in degrees)
        min_samples: DBSCAN min_samples parameter
        save_results: Whether to save results to files
    
    Returns:
        Tuple of (DataFrame with cluster labels, stats dictionary)
    """
    # Load data if not provided
    if df is None:
        df = load_cleaned_data()
    
    print(f"Running DBSCAN clustering on {len(df):,} photos...")
    print(f"  Parameters: eps={eps}, min_samples={min_samples}")
    
    # Prepare coordinates and run clustering
    coords = prepare_coordinates(df)
    labels = run_dbscan(coords, eps=eps, min_samples=min_samples)
    
    # Add labels to dataframe
    df = df.copy()
    df['cluster'] = labels
    
    # Calculate statistics
    stats = get_cluster_stats(labels)
    stats['parameters'] = {'eps': eps, 'min_samples': min_samples}
    stats['timestamp'] = datetime.now().isoformat()
    
    print(f"\nResults:")
    print(f"  Clusters found: {stats['n_clusters']}")
    print(f"  Noise points: {stats['n_noise']:,} ({stats['noise_percentage']:.1f}%)")
    print(f"  Largest cluster: {stats['largest_cluster']:,} points")
    print(f"  Median cluster size: {stats['median_cluster_size']} points")
    
    # Save results
    if save_results:
        # Save clustered data
        clustered_path = DATA_DIR / "flickr_clustered.csv"
        df.to_csv(clustered_path, index=False)
        print(f"\n  Saved clustered data to: {clustered_path}")
        
        # Save stats
        stats_path = REPORTS_DIR / "clustering_baseline_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved stats to: {stats_path}")
    
    return df, stats


def main():
    """Run baseline clustering and print results."""
    print("=" * 50)
    print("Running Baseline DBSCAN Clustering")
    print("=" * 50)
    
    df, stats = run_baseline_clustering()
    
    print("\n" + "=" * 50)
    print("✅ Baseline clustering complete!")
    print("=" * 50)
    
    return df, stats


def calculate_quality_metrics(coords: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calculate clustering quality metrics.
    
    Args:
        coords: Array of coordinates
        labels: Cluster labels
    
    Returns:
        Dictionary of quality metrics
    """
    # Only calculate on clustered points (exclude noise)
    mask = labels != -1
    n_clustered = mask.sum()
    n_clusters = len(set(labels[mask]))
    
    if n_clustered < 2 or n_clusters < 2:
        return {
            'silhouette': None,
            'davies_bouldin': None,
            'calinski_harabasz': None,
            'note': 'Not enough clusters or points for metrics'
        }
    
    return {
        'silhouette': float(silhouette_score(coords[mask], labels[mask])),
        'davies_bouldin': float(davies_bouldin_score(coords[mask], labels[mask])),
        'calinski_harabasz': float(calinski_harabasz_score(coords[mask], labels[mask]))
    }


def run_parameter_sweep(
    df: Optional[pd.DataFrame] = None,
    eps_values: list = None,
    min_samples: int = 10,
    save_results: bool = True
) -> pd.DataFrame:
    """
    Run DBSCAN with multiple eps values and compare results.
    
    Args:
        df: DataFrame with photo data
        eps_values: List of eps values to try
        min_samples: Fixed min_samples value
        save_results: Whether to save results
    
    Returns:
        DataFrame with results for each eps value
    """
    if eps_values is None:
        eps_values = [0.001, 0.002, 0.003, 0.004, 0.005]
    
    if df is None:
        df = load_cleaned_data()
    
    coords = prepare_coordinates(df)
    results = []
    
    print("=" * 60)
    print("DBSCAN Parameter Sweep")
    print("=" * 60)
    print(f"Testing eps values: {eps_values}")
    print(f"Fixed min_samples: {min_samples}")
    print(f"Total points: {len(df):,}")
    print("-" * 60)
    
    for eps in eps_values:
        print(f"\nRunning eps={eps}...")
        
        # Run clustering
        labels = run_dbscan(coords, eps=eps, min_samples=min_samples)
        
        # Get basic stats
        stats = get_cluster_stats(labels)
        
        # Get quality metrics
        metrics = calculate_quality_metrics(coords, labels)
        
        # Combine results
        result = {
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': stats['n_clusters'],
            'noise_pct': round(stats['noise_percentage'], 2),
            'largest_cluster_pct': round(stats['largest_cluster'] / len(labels) * 100, 1),
            'median_size': stats['median_cluster_size'],
            'silhouette': round(metrics['silhouette'], 4) if metrics.get('silhouette') else None,
            'davies_bouldin': round(metrics['davies_bouldin'], 4) if metrics.get('davies_bouldin') else None,
        }
        results.append(result)
        
        print(f"  Clusters: {result['n_clusters']}, Noise: {result['noise_pct']}%, "
              f"Largest: {result['largest_cluster_pct']}%, Silhouette: {result['silhouette']}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Find best eps by silhouette score
    valid_results = results_df[results_df['silhouette'].notna()]
    if len(valid_results) > 0:
        best_idx = valid_results['silhouette'].idxmax()
        best_eps = results_df.loc[best_idx, 'eps']
        print(f"\n✅ Best eps by silhouette score: {best_eps}")
    
    # Save results
    if save_results:
        sweep_path = REPORTS_DIR / "clustering_parameter_sweep.csv"
        results_df.to_csv(sweep_path, index=False)
        print(f"\nSaved sweep results to: {sweep_path}")
    
    return results_df


if __name__ == "__main__":
    main()
