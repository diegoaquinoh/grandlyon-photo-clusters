#!/usr/bin/env python3
"""
Master pipeline script for Grand Lyon Photo Clusters.

Runs the complete pipeline from raw data to final cluster map:
1. Data Cleaning (load_and_clean_data)
2. Clustering (HDBSCAN with min_cluster_size=30)
3. Text Mining (TF-IDF descriptors)
4. Map Generation (cluster_map.html)

Usage:
    python scripts/run_pipeline.py [OPTIONS]

Options:
    --skip-cleaning     Skip cleaning if cleaned data already exists
    --skip-clustering   Skip clustering if clustered data already exists
    --bbox TYPE         Bbox size: 'large', 'metro', or 'center' (default: large)
    --min-cluster-size N   HDBSCAN min_cluster_size (default: 30)
    --sample N          Sample size for testing (default: all data)
    --dry-run           Show what would be done without executing
"""

import sys
import argparse
import time
from pathlib import Path

# Add project root to path so src modules can import each other
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_and_clean_data, load_cleaned_data, 
    CLEANED_DATA_PATH, DATA_DIR, REPORTS_DIR
)
from src.clustering import prepare_coordinates, run_hdbscan, get_cluster_stats, filter_outliers_and_report
from src.text_mining import run_text_mining
from src.map_visualization import create_cluster_map

# Output paths
CLUSTERED_DATA_PATH = DATA_DIR / "flickr_clustered.csv"
CLUSTER_MAP_PATH = PROJECT_ROOT / "app" / "cluster_map.html"


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step_num: int, total: int, description: str):
    """Print a step indicator."""
    print(f"\n[{step_num}/{total}] {description}")
    print("-" * 50)


def run_pipeline(
    skip_cleaning: bool = False,
    skip_clustering: bool = False,
    bbox_type: str = "large",
    filter_outliers: bool = False,
    min_cluster_size: int = 30,
    sample_size: int = None,
    dry_run: bool = False
):
    """
    Run the complete Grand Lyon Photo Clusters pipeline.
    
    Args:
        skip_cleaning: Skip if cleaned data exists
        skip_clustering: Skip if clustered data exists
        bbox_type: Bbox size - 'large', 'metro', or 'center'
        min_cluster_size: HDBSCAN min_cluster_size parameter
        sample_size: Optional sample size for testing
        dry_run: Just show what would be done
    """
    start_time = time.time()
    
    print_header("GRAND LYON PHOTO CLUSTERS - FULL PIPELINE")
    print(f"Configuration:")
    print(f"  - Bbox: {bbox_type}")
    print(f"  - Filter outliers: {filter_outliers}")
    print(f"  - HDBSCAN min_cluster_size: {min_cluster_size}")
    print(f"  - Sample size: {sample_size or 'all data'}")
    print(f"  - Skip cleaning: {skip_cleaning}")
    print(f"  - Skip clustering: {skip_clustering}")
    print(f"  - Dry run: {dry_run}")
    
    if dry_run:
        print("\n[DRY RUN] Would execute:")
        print("  1. Load and clean raw data → flickr_cleaned.parquet")
        print("  2. Run HDBSCAN clustering → flickr_clustered.csv")
        print("  3. Generate TF-IDF descriptors → cluster_descriptors.json")
        print("  4. Create cluster map → cluster_map.html")
        return
    
    # =========================================================================
    # STAGE 1-2: DATA CLEANING
    # =========================================================================
    print_step(1, 4, "DATA CLEANING")
    
    if skip_cleaning and CLEANED_DATA_PATH.exists():
        print(f"Skipping cleaning - loading from cache: {CLEANED_DATA_PATH}")
        import pandas as pd
        df = pd.read_parquet(CLEANED_DATA_PATH)
        print(f"Loaded {len(df):,} rows from cache")
    else:
        df = load_and_clean_data(
            filter_bbox=True,
            bbox_type=bbox_type,
            save_cache=True,
            save_log=True,
            verbose=True
        )
    
    # Optional: sample for testing
    if sample_size and len(df) > sample_size:
        print(f"\nSampling {sample_size:,} rows for testing...")
        df = df.sample(n=sample_size, random_state=42)
    
    # Optional: density-based outlier filtering (stricter: need 10 neighbors in 200m)
    if filter_outliers:
        print("\nApplying density-based outlier filter...")
        df = filter_outliers_and_report(df, min_neighbors=10, radius=0.002)
    
    print(f"\n✅ Cleaning complete: {len(df):,} photos ready")
    
    # =========================================================================
    # STAGE 3: CLUSTERING
    # =========================================================================
    print_step(2, 4, "CLUSTERING (HDBSCAN)")
    
    if skip_clustering and CLUSTERED_DATA_PATH.exists():
        print(f"Skipping clustering - loading from: {CLUSTERED_DATA_PATH}")
        import pandas as pd
        df = pd.read_csv(CLUSTERED_DATA_PATH)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        print(f"Loaded {len(df):,} rows with cluster labels")
    else:
        print(f"Running HDBSCAN with min_cluster_size={min_cluster_size}...")
        
        # Prepare coordinates (not scaled for HDBSCAN - uses euclidean distance on lat/lon)
        coords = prepare_coordinates(df, scale=False)
        print(f"  Prepared {len(coords):,} coordinate pairs")
        
        # Run HDBSCAN
        labels = run_hdbscan(coords, min_cluster_size=min_cluster_size)
        df['cluster'] = labels
        
        # Print stats
        stats = get_cluster_stats(labels)
        print(f"\n  Clusters created: {stats['n_clusters']}")
        print(f"  Noise points: {stats['n_noise']:,} ({stats['noise_percentage']:.1f}%)")
        print(f"  Largest cluster: {stats['largest_cluster']:,} points")
        print(f"  Median cluster size: {stats['median_cluster_size']}")
        
        # Save
        df.to_csv(CLUSTERED_DATA_PATH, index=False)
        print(f"\n  Saved to: {CLUSTERED_DATA_PATH}")
    
    print(f"\n✅ Clustering complete: {df['cluster'].nunique()} clusters")
    
    # =========================================================================
    # STAGE 4: TEXT MINING
    # =========================================================================
    print_step(3, 4, "TEXT MINING (TF-IDF)")
    
    descriptors = run_text_mining(df=df, top_n=10, save_results=True)
    
    print(f"\n✅ Text mining complete: {len(descriptors)} clusters described")
    
    # =========================================================================
    # STAGE 5: MAP GENERATION
    # =========================================================================
    print_step(4, 4, "MAP GENERATION")
    
    create_cluster_map(
        df=df,
        min_cluster_size=100,  # Only show clusters with ≥100 points
        show_noise=False,
        sample_per_cluster=50,  # Reduced for HDBSCAN's 924 clusters
        include_heatmap=False,
        use_polygons=True,  # Use polygon areas instead of individual points
        output_path=CLUSTER_MAP_PATH
    )
    
    print(f"\n✅ Map generated: {CLUSTER_MAP_PATH}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    
    print_header("PIPELINE COMPLETE")
    print(f"\nTotal time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\nOutputs:")
    print(f"  - Cleaned data:  {CLEANED_DATA_PATH}")
    print(f"  - Clustered data: {CLUSTERED_DATA_PATH}")
    print(f"  - Descriptors:   {REPORTS_DIR / 'cluster_descriptors.json'}")
    print(f"  - Summary:       {REPORTS_DIR / 'text_mining_summary.md'}")
    print(f"  - Cluster map:   {CLUSTER_MAP_PATH}")
    print(f"\n🎉 Open {CLUSTER_MAP_PATH} in your browser to view the results!")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete Grand Lyon Photo Clusters pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip cleaning if cached data exists"
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip clustering if clustered data exists"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=30,
        help="HDBSCAN min_cluster_size parameter (default: 30)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for testing (default: all data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default="large",
        choices=["large", "metro", "center"],
        help="Bbox size: 'large' (1400km²), 'metro' (150km²), 'center' (20km²)"
    )
    parser.add_argument(
        "--filter-outliers",
        action="store_true",
        help="Apply density-based outlier filtering"
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            skip_cleaning=args.skip_cleaning,
            skip_clustering=args.skip_clustering,
            bbox_type=args.bbox,
            filter_outliers=args.filter_outliers,
            min_cluster_size=args.min_cluster_size,
            sample_size=args.sample,
            dry_run=args.dry_run
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
