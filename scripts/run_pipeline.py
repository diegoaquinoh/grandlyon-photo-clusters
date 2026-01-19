#!/usr/bin/env python3
"""
Master pipeline script for Grand Lyon Photo Clusters.

Runs the complete pipeline from raw data to final cluster map:
1. Data Cleaning (load_and_clean_data)
2. Clustering (K-Means with k=100)
3. Text Mining (TF-IDF descriptors)
4. Map Generation (cluster_map.html)

Usage:
    python scripts/run_pipeline.py [OPTIONS]

Options:
    --skip-cleaning     Skip cleaning if cleaned data already exists
    --skip-clustering   Skip clustering if clustered data already exists
    --k N               Number of clusters (default: 100)
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
from src.clustering import prepare_coordinates, run_kmeans, get_cluster_stats
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
    n_clusters: int = 100,
    sample_size: int = None,
    dry_run: bool = False
):
    """
    Run the complete Grand Lyon Photo Clusters pipeline.
    
    Args:
        skip_cleaning: Skip if cleaned data exists
        skip_clustering: Skip if clustered data exists
        n_clusters: Number of clusters for K-Means
        sample_size: Optional sample size for testing
        dry_run: Just show what would be done
    """
    start_time = time.time()
    
    print_header("GRAND LYON PHOTO CLUSTERS - FULL PIPELINE")
    print(f"Configuration:")
    print(f"  - Clusters: {n_clusters}")
    print(f"  - Sample size: {sample_size or 'all data'}")
    print(f"  - Skip cleaning: {skip_cleaning}")
    print(f"  - Skip clustering: {skip_clustering}")
    print(f"  - Dry run: {dry_run}")
    
    if dry_run:
        print("\n[DRY RUN] Would execute:")
        print("  1. Load and clean raw data → flickr_cleaned.parquet")
        print("  2. Run K-Means clustering → flickr_clustered.csv")
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
            save_cache=True,
            save_log=True,
            verbose=True
        )
    
    # Optional: sample for testing
    if sample_size and len(df) > sample_size:
        print(f"\nSampling {sample_size:,} rows for testing...")
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"\n✅ Cleaning complete: {len(df):,} photos ready")
    
    # =========================================================================
    # STAGE 3: CLUSTERING
    # =========================================================================
    print_step(2, 4, "CLUSTERING (K-Means)")
    
    if skip_clustering and CLUSTERED_DATA_PATH.exists():
        print(f"Skipping clustering - loading from: {CLUSTERED_DATA_PATH}")
        import pandas as pd
        df = pd.read_csv(CLUSTERED_DATA_PATH)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        print(f"Loaded {len(df):,} rows with cluster labels")
    else:
        print(f"Running K-Means with k={n_clusters}...")
        
        # Prepare coordinates (scaled for K-Means)
        coords = prepare_coordinates(df, scale=True)
        print(f"  Prepared {len(coords):,} coordinate pairs")
        
        # Run K-Means
        labels = run_kmeans(coords, n_clusters=n_clusters, random_state=42)
        df['cluster'] = labels
        
        # Print stats
        stats = get_cluster_stats(labels)
        print(f"\n  Clusters created: {stats['n_clusters']}")
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
        min_cluster_size=10,
        show_noise=False,
        sample_per_cluster=200,
        include_heatmap=False,
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
        "--k",
        type=int,
        default=100,
        help="Number of clusters (default: 100)"
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
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            skip_cleaning=args.skip_cleaning,
            skip_clustering=args.skip_clustering,
            n_clusters=args.k,
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
