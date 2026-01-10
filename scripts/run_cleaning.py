#!/usr/bin/env python3
"""
Run the complete data cleaning pipeline.

This script provides a reproducible way to run the cleaning pipeline
from the command line with consistent parameters.

Usage:
    python scripts/run_cleaning.py [--no-bbox] [--no-cache] [--quiet]
    
Options:
    --no-bbox   Skip Lyon bounding box filtering
    --no-cache  Don't save cleaned data to cache
    --quiet     Minimal output
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_and_clean_data, get_data_stats, CLEANED_DATA_PATH, CLEANING_LOG_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Run the Grand Lyon Photo Clusters data cleaning pipeline."
    )
    parser.add_argument(
        "--no-bbox", 
        action="store_true",
        help="Skip Lyon bounding box filtering"
    )
    parser.add_argument(
        "--no-cache", 
        action="store_true",
        help="Don't save cleaned data to cache"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    # Run cleaning pipeline
    df = load_and_clean_data(
        filter_bbox=not args.no_bbox,
        save_cache=not args.no_cache,
        save_log=True,
        verbose=not args.quiet
    )
    
    if not args.quiet:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"\nOutputs:")
        print(f"  - Cleaned data: {CLEANED_DATA_PATH}")
        print(f"  - Cleaning log: {CLEANING_LOG_PATH}")
        print(f"\nReady for clustering and analysis!")
    
    return df


if __name__ == "__main__":
    main()
