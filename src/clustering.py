"""
Clustering module for the Grand Lyon Photo Clusters project.
Provides utilities for spatial clustering of photo locations.

Session 2: Enhanced with K-Means, Hierarchical, and improved DBSCAN.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import json
from datetime import datetime

from .data_loader import load_cleaned_data, PROJECT_ROOT

# Output paths
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# DENSITY-BASED OUTLIER DETECTION
# =============================================================================

def filter_low_density_points(
    coords: np.ndarray,
    min_neighbors: int = 5,
    radius: float = 0.003
) -> np.ndarray:
    """
    Density-based outlier detection.
    
    Points with fewer than `min_neighbors` within `radius` are considered
    outliers (low local density). This is conceptually similar to how
    DBSCAN identifies noise points.
    
    Args:
        coords: Array of [lat, lon] coordinates (NOT scaled)
        min_neighbors: Minimum neighbors required to be considered dense
        radius: Search radius in degrees (~0.003 ≈ 300m at Lyon latitude)
    
    Returns:
        Boolean mask (True = keep, False = outlier)
    
    Example:
        >>> mask = filter_low_density_points(coords, min_neighbors=5, radius=0.003)
        >>> df_clean = df[mask]
        >>> print(f"Removed {(~mask).sum()} outliers")
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Build neighbor index
    nn = NearestNeighbors(radius=radius, algorithm='ball_tree', metric='euclidean')
    nn.fit(coords)
    
    # Find neighbors within radius for each point
    neighbors = nn.radius_neighbors(coords, return_distance=False)
    
    # Count neighbors (excluding self)
    neighbor_counts = np.array([len(n) - 1 for n in neighbors])
    
    # Points with enough neighbors are "dense" (not outliers)
    mask = neighbor_counts >= min_neighbors
    
    return mask


def filter_outliers_and_report(
    df: pd.DataFrame,
    min_neighbors: int = 5,
    radius: float = 0.003,
    verbose: bool = True,
    update_log: bool = True
) -> pd.DataFrame:
    """
    Apply density-based outlier filtering with a summary report.
    
    Args:
        df: DataFrame with 'lat' and 'long' columns
        min_neighbors: Minimum neighbors required (default: 5)
        radius: Search radius in degrees (default: 0.003 ≈ 300m)
        verbose: Print summary
        update_log: Whether to append to cleaning_log.json
    
    Returns:
        Filtered DataFrame with outliers removed
    """
    coords = df[['lat', 'long']].values
    initial_count = len(df)
    
    mask = filter_low_density_points(coords, min_neighbors=min_neighbors, radius=radius)
    
    n_outliers = (~mask).sum()
    pct_outliers = n_outliers / initial_count * 100
    final_count = mask.sum()
    
    if verbose:
        print(f"\n{'=' * 50}")
        print("DENSITY-BASED OUTLIER FILTERING")
        print(f"{'=' * 50}")
        print(f"Parameters: min_neighbors={min_neighbors}, radius={radius}")
        print(f"Total points: {initial_count:,}")
        print(f"Outliers (low density): {n_outliers:,} ({pct_outliers:.1f}%)")
        print(f"Retained: {final_count:,} ({final_count/initial_count*100:.1f}%)")
    
    # Append to cleaning log
    if update_log:
        log_path = REPORTS_DIR / "cleaning_log.json"
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                
                # Add new step (convert numpy int64 to Python int for JSON)
                new_step = {
                    "step": "Density-based outlier filtering",
                    "rows_before": int(initial_count),
                    "rows_after": int(final_count),
                    "rows_removed": int(n_outliers),
                    "removal_percentage": round(pct_outliers, 2),
                    "reason": f"Removed isolated points with < {min_neighbors} neighbors within {radius}° radius (low local density)"
                }
                log_data["steps"].append(new_step)
                
                # Update final count
                log_data["final_count"] = int(final_count)
                log_data["total_removed"] = int(log_data["initial_count"] - final_count)
                log_data["retention_rate"] = round(final_count / log_data["initial_count"] * 100, 2)
                
                with open(log_path, 'w') as f:
                    json.dump(log_data, f, indent=2)
                
                if verbose:
                    print(f"\n  ✅ Updated cleaning log: {log_path}")
            except Exception as e:
                if verbose:
                    print(f"\n  ⚠️ Could not update log: {e}")
    
    return df[mask].copy()


def prepare_coordinates(df: pd.DataFrame, scale: bool = False) -> np.ndarray:
    """
    Extract and prepare coordinates for clustering.
    
    Args:
        df: DataFrame with 'lat' and 'long' columns
        scale: Whether to standardize coordinates (recommended for K-Means)
    
    Returns:
        NumPy array of shape (n_samples, 2) with [lat, lon] coordinates
    """
    coords = df[['lat', 'long']].values
    if scale:
        scaler = StandardScaler()
        coords = scaler.fit_transform(coords)
    return coords


# =============================================================================
# DBSCAN CLUSTERING
# =============================================================================

def run_dbscan(
    coords: np.ndarray,
    eps: float = 0.005,
    min_samples: int = 10
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
    
    Returns:
        Array of cluster labels (-1 = noise)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm='ball_tree')
    labels = dbscan.fit_predict(coords)
    return labels


# =============================================================================
# K-MEANS CLUSTERING
# =============================================================================

def run_kmeans(
    coords: np.ndarray,
    n_clusters: int = 50,
    random_state: int = 42,
    max_iter: int = 300
) -> np.ndarray:
    """
    Run K-Means clustering on coordinates.
    
    K-Means creates spherical clusters of similar size. Works best when:
    - You have an idea of how many clusters to expect
    - Clusters are roughly similar in size
    - Data is scaled
    
    Args:
        coords: Array of [lat, lon] coordinates (recommend scaling first)
        n_clusters: Number of clusters to create
        random_state: Random seed for reproducibility
        max_iter: Maximum iterations
    
    Returns:
        Array of cluster labels (0 to n_clusters-1)
    """
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=random_state, 
        max_iter=max_iter,
        n_init=10
    )
    labels = kmeans.fit_predict(coords)
    return labels


def find_optimal_k(
    coords: np.ndarray,
    k_range: range = range(10, 101, 10),
    random_state: int = 42,
    sample_size: int = 10000
) -> Dict:
    """
    Find optimal number of clusters for K-Means using elbow method and silhouette.
    
    Args:
        coords: Array of coordinates
        k_range: Range of k values to test
        random_state: Random seed
        sample_size: Number of points to sample for silhouette (speeds up calculation)
    
    Returns:
        Dictionary with results for each k value
    """
    results = []
    n_points = len(coords)
    
    # Create sample indices for silhouette calculation (much faster)
    np.random.seed(random_state)
    if n_points > sample_size:
        sample_idx = np.random.choice(n_points, sample_size, replace=False)
    else:
        sample_idx = np.arange(n_points)
    
    for k in k_range:
        print(f"  Testing k={k}...", end=" ")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=3)  # Reduced n_init
        labels = kmeans.fit_predict(coords)
        
        inertia = kmeans.inertia_
        
        # Calculate silhouette on sample only (100x faster)
        if k > 1:
            silhouette = silhouette_score(coords[sample_idx], labels[sample_idx])
        else:
            silhouette = 0
        
        results.append({
            'k': k,
            'inertia': float(inertia),
            'silhouette': float(silhouette)
        })
        print(f"silhouette={silhouette:.4f}")
    
    return results


# =============================================================================
# HIERARCHICAL CLUSTERING
# =============================================================================

def run_hierarchical(
    coords: np.ndarray,
    n_clusters: int = 50,
    linkage: str = 'ward'
) -> np.ndarray:
    """
    Run Agglomerative Hierarchical clustering on coordinates.
    
    Hierarchical clustering builds a tree of clusters. Ward linkage
    minimizes variance within clusters, good for compact clusters.
    
    Args:
        coords: Array of [lat, lon] coordinates
        n_clusters: Number of clusters to create
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
    
    Returns:
        Array of cluster labels (0 to n_clusters-1)
    """
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    labels = hierarchical.fit_predict(coords)
    return labels


# =============================================================================
# HDBSCAN CLUSTERING
# =============================================================================

def run_hdbscan(
    coords: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: int = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = 'eom'
) -> np.ndarray:
    """
    Run HDBSCAN clustering on coordinates.
    
    HDBSCAN (Hierarchical DBSCAN) is an improvement over DBSCAN that:
    - Does not require eps parameter selection
    - Finds clusters of varying densities
    - Handles noise points automatically
    - Produces more stable clusterings
    
    Args:
        coords: Array of [lat, lon] coordinates
        min_cluster_size: Minimum number of points to form a cluster (most important param)
        min_samples: Number of samples in neighborhood for core points (defaults to min_cluster_size)
        cluster_selection_epsilon: Distance threshold for cluster selection (0 = no threshold)
        cluster_selection_method: 'eom' (Excess of Mass) or 'leaf'
    
    Returns:
        Array of cluster labels (-1 = noise)
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(coords)
    return labels


def find_optimal_hdbscan(
    coords: np.ndarray,
    min_cluster_sizes: List[int] = None,
    min_samples_values: List[int] = None,
    sample_size: int = 10000
) -> List[Dict]:
    """
    Grid search over HDBSCAN parameters to find optimal configuration.
    
    Args:
        coords: Array of coordinates
        min_cluster_sizes: List of min_cluster_size values to test
        min_samples_values: List of min_samples values to test (None means same as min_cluster_size)
        sample_size: Number of points to sample for silhouette calculation
    
    Returns:
        List of dictionaries with results for each parameter combination
    """
    if min_cluster_sizes is None:
        min_cluster_sizes = [10, 15, 20, 30, 50, 75, 100]
    if min_samples_values is None:
        min_samples_values = [None, 5, 10, 15]  # None means use min_cluster_size
    
    results = []
    n_points = len(coords)
    
    # Create sample indices for silhouette calculation
    np.random.seed(42)
    if n_points > sample_size:
        sample_idx = np.random.choice(n_points, sample_size, replace=False)
    else:
        sample_idx = np.arange(n_points)
    
    total_combos = len(min_cluster_sizes) * len(min_samples_values)
    combo_num = 0
    
    for mcs in min_cluster_sizes:
        for ms in min_samples_values:
            combo_num += 1
            actual_ms = ms if ms is not None else mcs
            print(f"  [{combo_num}/{total_combos}] min_cluster_size={mcs}, min_samples={actual_ms}", end=" ")
            
            labels = run_hdbscan(coords, min_cluster_size=mcs, min_samples=ms)
            stats = get_cluster_stats(labels)
            
            # Calculate silhouette on sample (only for clustered points)
            mask = labels[sample_idx] != -1
            if mask.sum() > 50 and stats['n_clusters'] > 1:
                sil = silhouette_score(coords[sample_idx][mask], labels[sample_idx][mask])
            else:
                sil = None
            
            result = {
                'min_cluster_size': mcs,
                'min_samples': actual_ms,
                'n_clusters': stats['n_clusters'],
                'noise_pct': round(stats['noise_percentage'], 1),
                'largest_pct': round(stats['largest_cluster'] / len(labels) * 100, 1),
                'median_size': stats['median_cluster_size'],
                'silhouette': round(sil, 4) if sil else None
            }
            results.append(result)
            print(f"→ clusters={result['n_clusters']}, noise={result['noise_pct']}%, sil={result['silhouette']}")
    
    return results

# =============================================================================
# CLUSTER STATISTICS & METRICS
# =============================================================================

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
        'mean_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0,
        'std_cluster_size': float(np.std(cluster_sizes)) if cluster_sizes else 0,
        'cluster_sizes_top10': [int(x) for x in cluster_sizes[:10]]
    }


def calculate_quality_metrics(
    coords: np.ndarray, 
    labels: np.ndarray,
    sample_size: int = 10000
) -> dict:
    """
    Calculate clustering quality metrics.
    
    Args:
        coords: Array of coordinates
        labels: Cluster labels
        sample_size: Number of points to sample for metrics (speeds up calculation)
    
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
    
    # Sample for faster calculation
    coords_masked = coords[mask]
    labels_masked = labels[mask]
    
    if len(coords_masked) > sample_size:
        sample_idx = np.random.choice(len(coords_masked), sample_size, replace=False)
        coords_sample = coords_masked[sample_idx]
        labels_sample = labels_masked[sample_idx]
    else:
        coords_sample = coords_masked
        labels_sample = labels_masked
    
    return {
        'silhouette': float(silhouette_score(coords_sample, labels_sample)),
        'davies_bouldin': float(davies_bouldin_score(coords_sample, labels_sample)),
        'calinski_harabasz': float(calinski_harabasz_score(coords_sample, labels_sample))
    }


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_algorithms(
    df: Optional[pd.DataFrame] = None,
    dbscan_params: Dict = None,
    kmeans_k: int = 50,
    hierarchical_k: int = 50,
    scale_for_kmeans: bool = True,
    hier_sample_size: int = 20000
) -> pd.DataFrame:
    """
    Run all three clustering algorithms and compare results.
    
    Args:
        df: DataFrame with photo data
        dbscan_params: Dict with 'eps' and 'min_samples' for DBSCAN
        kmeans_k: Number of clusters for K-Means
        hierarchical_k: Number of clusters for Hierarchical
        scale_for_kmeans: Whether to scale coordinates for K-Means
        hier_sample_size: Sample size for hierarchical clustering (memory-safe)
    
    Returns:
        DataFrame with comparison results
    """
    if dbscan_params is None:
        dbscan_params = {'eps': 0.003, 'min_samples': 10}
    
    if df is None:
        df = load_cleaned_data()
    
    # Get coordinates
    coords = prepare_coordinates(df, scale=False)
    coords_scaled = prepare_coordinates(df, scale=True)
    
    results = []
    
    print("=" * 70)
    print("CLUSTERING ALGORITHM COMPARISON (Optimized)")
    print("=" * 70)
    print(f"Data points: {len(df):,}")
    print()
    
    # DBSCAN
    print(f"[1/3] Running DBSCAN (eps={dbscan_params['eps']}, min_samples={dbscan_params['min_samples']})...")
    dbscan_labels = run_dbscan(coords, **dbscan_params)
    dbscan_stats = get_cluster_stats(dbscan_labels)
    dbscan_metrics = calculate_quality_metrics(coords, dbscan_labels)
    results.append({
        'algorithm': 'DBSCAN',
        'parameters': f"eps={dbscan_params['eps']}, min_samples={dbscan_params['min_samples']}",
        'n_clusters': dbscan_stats['n_clusters'],
        'noise_pct': round(dbscan_stats['noise_percentage'], 2),
        'largest_cluster': dbscan_stats['largest_cluster'],
        'largest_pct': round(dbscan_stats['largest_cluster'] / len(df) * 100, 1),
        'median_size': dbscan_stats['median_cluster_size'],
        'silhouette': round(dbscan_metrics['silhouette'], 4) if dbscan_metrics.get('silhouette') else None,
        'davies_bouldin': round(dbscan_metrics['davies_bouldin'], 4) if dbscan_metrics.get('davies_bouldin') else None,
    })
    print(f"      → {dbscan_stats['n_clusters']} clusters, {dbscan_stats['noise_percentage']:.1f}% noise")
    
    # K-Means
    print(f"[2/3] Running K-Means (k={kmeans_k})...")
    kmeans_coords = coords_scaled if scale_for_kmeans else coords
    kmeans_labels = run_kmeans(kmeans_coords, n_clusters=kmeans_k)
    kmeans_stats = get_cluster_stats(kmeans_labels)
    kmeans_metrics = calculate_quality_metrics(kmeans_coords, kmeans_labels)
    results.append({
        'algorithm': 'K-Means',
        'parameters': f"k={kmeans_k}, scaled={scale_for_kmeans}",
        'n_clusters': kmeans_stats['n_clusters'],
        'noise_pct': 0.0,
        'largest_cluster': kmeans_stats['largest_cluster'],
        'largest_pct': round(kmeans_stats['largest_cluster'] / len(df) * 100, 1),
        'median_size': kmeans_stats['median_cluster_size'],
        'silhouette': round(kmeans_metrics['silhouette'], 4) if kmeans_metrics.get('silhouette') else None,
        'davies_bouldin': round(kmeans_metrics['davies_bouldin'], 4) if kmeans_metrics.get('davies_bouldin') else None,
    })
    print(f"      → {kmeans_stats['n_clusters']} clusters")
    
    # Hierarchical (on sample for memory safety)
    sample_size = min(hier_sample_size, len(coords))
    use_sample = len(coords) > hier_sample_size
    
    if use_sample:
        print(f"[3/3] Running Hierarchical (k={hierarchical_k}, linkage=ward) on {sample_size:,} sample...")
        sample_idx = np.random.choice(len(coords), sample_size, replace=False)
        coords_hier = coords[sample_idx]
    else:
        print(f"[3/3] Running Hierarchical (k={hierarchical_k}, linkage=ward)...")
        coords_hier = coords
    
    hier_model = AgglomerativeClustering(n_clusters=hierarchical_k, linkage='ward')
    hier_labels = hier_model.fit_predict(coords_hier)
    hier_stats = get_cluster_stats(hier_labels)
    hier_metrics = calculate_quality_metrics(coords_hier, hier_labels)
    
    results.append({
        'algorithm': 'Hierarchical',
        'parameters': f"k={hierarchical_k}, linkage=ward" + (f", sampled={sample_size}" if use_sample else ""),
        'n_clusters': hier_stats['n_clusters'],
        'noise_pct': 0.0,
        'largest_cluster': hier_stats['largest_cluster'],
        'largest_pct': round(hier_stats['largest_cluster'] / len(coords_hier) * 100, 1),
        'median_size': hier_stats['median_cluster_size'],
        'silhouette': round(hier_metrics['silhouette'], 4) if hier_metrics.get('silhouette') else None,
        'davies_bouldin': round(hier_metrics['davies_bouldin'], 4) if hier_metrics.get('davies_bouldin') else None,
    })
    print(f"      → {hier_stats['n_clusters']} clusters")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    
    return comparison_df


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


def run_dbscan_grid_search(
    df: Optional[pd.DataFrame] = None,
    eps_values: list = None,
    min_samples_values: list = None,
    save_results: bool = True,
    sample_size: int = None
) -> pd.DataFrame:
    """
    Full grid search over DBSCAN parameters (eps × min_samples).
    
    This addresses the issue that only eps was being tuned while 
    min_samples was fixed at 10.
    
    Args:
        df: DataFrame with photo data
        eps_values: List of eps values to try (default: [0.002, 0.003, 0.004, 0.005, 0.006])
        min_samples_values: List of min_samples to try (default: [5, 10, 15, 20, 30])
        save_results: Whether to save results to CSV
        sample_size: Optional sample size for faster testing
    
    Returns:
        DataFrame with results for each parameter combination
    """
    if eps_values is None:
        eps_values = [0.002, 0.003, 0.004, 0.005, 0.006]
    if min_samples_values is None:
        min_samples_values = [5, 10, 15, 20, 30]
    
    if df is None:
        df = load_cleaned_data()
    
    # Optional sampling for faster experimentation
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size:,} points for faster grid search...")
        df = df.sample(n=sample_size, random_state=42)
    
    coords = prepare_coordinates(df)
    results = []
    
    total_combinations = len(eps_values) * len(min_samples_values)
    
    print("=" * 70)
    print("DBSCAN GRID SEARCH (eps × min_samples)")
    print("=" * 70)
    print(f"eps values: {eps_values}")
    print(f"min_samples values: {min_samples_values}")
    print(f"Total combinations: {total_combinations}")
    print(f"Data points: {len(df):,}")
    print("-" * 70)
    
    combo_num = 0
    for eps in eps_values:
        for min_s in min_samples_values:
            combo_num += 1
            print(f"\n[{combo_num}/{total_combinations}] eps={eps}, min_samples={min_s}...", end=" ")
            
            import time
            start = time.time()
            
            # Run DBSCAN
            labels = run_dbscan(coords, eps=eps, min_samples=min_s)
            
            dbscan_time = time.time() - start
            
            # Get statistics
            stats = get_cluster_stats(labels)
            
            # Get quality metrics (skip if too few clusters)
            metrics = calculate_quality_metrics(coords, labels)
            
            total_time = time.time() - start
            
            result = {
                'eps': eps,
                'min_samples': min_s,
                'n_clusters': stats['n_clusters'],
                'noise_pct': round(stats['noise_percentage'], 2),
                'largest_cluster_pct': round(stats['largest_cluster'] / len(labels) * 100, 2),
                'median_size': stats['median_cluster_size'],
                'mean_size': round(stats['mean_cluster_size'], 1),
                'silhouette': round(metrics['silhouette'], 4) if metrics.get('silhouette') else None,
                'davies_bouldin': round(metrics['davies_bouldin'], 4) if metrics.get('davies_bouldin') else None,
                'time_sec': round(total_time, 1)
            }
            results.append(result)
            
            print(f"clusters={result['n_clusters']}, noise={result['noise_pct']}%, sil={result['silhouette']} ({total_time:.1f}s)")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Find best configuration by silhouette (excluding mega-cluster cases)
    valid_results = results_df[
        (results_df['silhouette'].notna()) & 
        (results_df['largest_cluster_pct'] < 50)  # Reject if one cluster has >50%
    ]
    
    if len(valid_results) > 0:
        best_idx = valid_results['silhouette'].idxmax()
        best = results_df.loc[best_idx]
        print(f"\n✅ BEST CONFIGURATION (by silhouette, excluding mega-clusters):")
        print(f"   eps={best['eps']}, min_samples={best['min_samples']}")
        print(f"   Clusters: {best['n_clusters']}, Silhouette: {best['silhouette']}")
        print(f"   Noise: {best['noise_pct']}%, Largest cluster: {best['largest_cluster_pct']}%")
        
        # Save best params
        best_params = {
            'algorithm': 'DBSCAN',
            'eps': float(best['eps']),
            'min_samples': int(best['min_samples']),
            'silhouette': float(best['silhouette']) if best['silhouette'] else None,
            'n_clusters': int(best['n_clusters']),
            'timestamp': datetime.now().isoformat()
        }
        best_path = REPORTS_DIR / "best_dbscan_params.json"
        with open(best_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\n   Saved best params to: {best_path}")
    else:
        print("\n⚠️  No valid configuration found (all have mega-clusters or failed metrics)")
    
    # Save full results
    if save_results:
        grid_path = REPORTS_DIR / "dbscan_grid_search.csv"
        results_df.to_csv(grid_path, index=False)
        print(f"\n📊 Saved grid search results to: {grid_path}")
    
    return results_df

def main():
    """Run clustering comparison and print results."""
    print("=" * 70)
    print("GRAND LYON PHOTO CLUSTERS - CLUSTERING MODULE")
    print("=" * 70)
    
    # Run comparison of all three algorithms
    comparison_df = compare_algorithms(
        dbscan_params={'eps': 0.003, 'min_samples': 10},
        kmeans_k=50,
        hierarchical_k=50
    )
    
    # Save comparison
    comparison_path = REPORTS_DIR / "clustering_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✅ Saved comparison to: {comparison_path}")
    
    return comparison_df


if __name__ == "__main__":
    main()
