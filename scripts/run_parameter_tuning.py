#!/usr/bin/env python3
"""
DBSCAN Parameter Tuning Script for Grand Lyon Photo Clusters.

Runs a grid search over eps × min_samples parameters to find optimal DBSCAN configuration.

Usage:
    python scripts/run_parameter_tuning.py [OPTIONS]

Options:
    --sample N     Sample size for faster testing (default: 20000)
    --full         Use full dataset (slower but more accurate)
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_cleaned_data
from src.clustering import run_dbscan_grid_search


def main():
    parser = argparse.ArgumentParser(
        description="Run DBSCAN parameter grid search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20000,
        help="Sample size for faster testing (default: 20000)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full dataset (ignores --sample)"
    )
    parser.add_argument(
        "--eps",
        type=str,
        default="0.002,0.003,0.004,0.005,0.006",
        help="Comma-separated eps values to test"
    )
    parser.add_argument(
        "--min-samples",
        type=str,
        default="5,10,15,20,30",
        help="Comma-separated min_samples values to test"
    )
    
    args = parser.parse_args()
    
    # Parse parameter values
    eps_values = [float(x) for x in args.eps.split(",")]
    min_samples_values = [int(x) for x in args.min_samples.split(",")]
    
    # Determine sample size
    sample_size = None if args.full else args.sample
    
    print("=" * 70)
    print("DBSCAN PARAMETER TUNING")
    print("=" * 70)
    print(f"eps values: {eps_values}")
    print(f"min_samples values: {min_samples_values}")
    print(f"Sample size: {'full dataset' if args.full else sample_size}")
    print()
    
    # Run grid search
    results = run_dbscan_grid_search(
        eps_values=eps_values,
        min_samples_values=min_samples_values,
        sample_size=sample_size,
        save_results=True
    )
    
    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)
    print("\nResults saved to: reports/dbscan_grid_search.csv")
    print("Best params saved to: reports/best_dbscan_params.json")
    
    return results


if __name__ == "__main__":
    main()
