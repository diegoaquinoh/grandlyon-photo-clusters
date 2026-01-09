"""
Data loader module for the Grand Lyon Photo Clusters project.
Provides utilities for loading, caching, and basic preprocessing of the Flickr dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "flickr_data2.csv"
CLEANED_DATA_PATH = DATA_DIR / "flickr_cleaned.parquet"

# Lyon bounding box (approximate)
LYON_BBOX = {
    "lat_min": 45.55,
    "lat_max": 45.95,
    "lon_min": 4.65,
    "lon_max": 5.10
}


def load_raw_data(nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the raw Flickr dataset.
    
    Args:
        nrows: Number of rows to load (None for all)
    
    Returns:
        DataFrame with raw photo data
    """
    df = pd.read_csv(RAW_DATA_PATH, nrows=nrows)
    
    # Clean column names (strip whitespace and remove unnamed columns)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df


def load_cleaned_data() -> pd.DataFrame:
    """
    Load the cleaned dataset from Parquet cache.
    Falls back to raw data if cache doesn't exist.
    
    Returns:
        DataFrame with cleaned photo data
    """
    if CLEANED_DATA_PATH.exists():
        return pd.read_parquet(CLEANED_DATA_PATH)
    else:
        print("Warning: Cleaned data not found, loading raw data...")
        return load_raw_data()


def create_datetime_column(df: pd.DataFrame, prefix: str = "date_taken") -> pd.Series:
    """
    Create a datetime column from component columns.
    
    Args:
        df: DataFrame with date component columns
        prefix: Column prefix ('date_taken' or 'date_upload')
    
    Returns:
        Series of datetime objects
    """
    def safe_datetime(row):
        try:
            return datetime(
                int(row[f'{prefix}_year']),
                int(row[f'{prefix}_month']),
                int(row[f'{prefix}_day']),
                int(row[f'{prefix}_hour']),
                int(row[f'{prefix}_minute'])
            )
        except (ValueError, TypeError):
            return pd.NaT
    
    return df.apply(safe_datetime, axis=1)


def filter_lyon_bbox(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to only include points within Lyon bounding box.
    
    Args:
        df: DataFrame with lat/long columns
    
    Returns:
        Filtered DataFrame
    """
    mask = (
        (df['lat'] >= LYON_BBOX['lat_min']) & 
        (df['lat'] <= LYON_BBOX['lat_max']) &
        (df['long'] >= LYON_BBOX['lon_min']) & 
        (df['long'] <= LYON_BBOX['lon_max'])
    )
    return df[mask].copy()


def remove_duplicates(df: pd.DataFrame, subset: Optional[list] = None) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates (default: 'id')
    
    Returns:
        Tuple of (cleaned DataFrame, number of duplicates removed)
    """
    if subset is None:
        subset = ['id']
    
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep='first')
    removed = initial_count - len(df_clean)
    
    return df_clean, removed


def get_data_stats(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary of statistics
    """
    return {
        'total_rows': len(df),
        'unique_photos': df['id'].nunique(),
        'unique_users': df['user'].nunique(),
        'lat_range': (df['lat'].min(), df['lat'].max()),
        'lon_range': (df['long'].min(), df['long'].max()),
        'year_range': (
            int(df['date_taken_year'].min()), 
            int(df['date_taken_year'].max())
        ),
        'null_coords': df[['lat', 'long']].isnull().any(axis=1).sum(),
        'empty_tags': (df['tags'].isnull() | (df['tags'] == '')).sum(),
        'empty_titles': (df['title'].isnull() | (df['title'] == '')).sum(),
    }


if __name__ == "__main__":
    # Quick test
    print("Loading raw data...")
    df = load_raw_data(nrows=1000)
    print(f"Loaded {len(df)} rows")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nStats: {get_data_stats(df)}")
