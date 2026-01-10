"""
Data loader module for the Grand Lyon Photo Clusters project.
Provides utilities for loading, caching, and basic preprocessing of the Flickr dataset.

Session 2: Enhanced with coherency checks, Parquet caching, and detailed logging.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List
import json

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
RAW_DATA_PATH = DATA_DIR / "flickr_data2.csv"
CLEANED_DATA_PATH = DATA_DIR / "flickr_cleaned.parquet"  # Changed to Parquet
CLEANED_CSV_PATH = DATA_DIR / "flickr_cleaned.csv"  # Keep CSV fallback
CLEANING_LOG_PATH = REPORTS_DIR / "cleaning_log.json"

# Lyon bounding box (approximate)
LYON_BBOX = {
    "lat_min": 45.55,
    "lat_max": 45.95,
    "lon_min": 4.65,
    "lon_max": 5.10
}

# Valid GPS coordinate ranges
GPS_VALID_RANGE = {
    "lat_min": -90.0,
    "lat_max": 90.0,
    "lon_min": -180.0,
    "lon_max": 180.0
}


class CleaningLog:
    """Log cleaning steps and track dropped rows with reasons."""
    
    def __init__(self):
        self.steps: List[Dict] = []
        self.initial_count: int = 0
        self.final_count: int = 0
        self.timestamp: str = datetime.now().isoformat()
        
    def set_initial(self, count: int):
        self.initial_count = count
        
    def set_final(self, count: int):
        self.final_count = count
        
    def log_step(self, step_name: str, before: int, after: int, reason: str):
        """Log a cleaning step."""
        self.steps.append({
            "step": step_name,
            "rows_before": before,
            "rows_after": after,
            "rows_removed": before - after,
            "removal_percentage": round((before - after) / before * 100, 2) if before > 0 else 0,
            "reason": reason
        })
        
    def to_dict(self) -> Dict:
        """Convert log to dictionary."""
        return {
            "timestamp": self.timestamp,
            "initial_count": self.initial_count,
            "final_count": self.final_count,
            "total_removed": self.initial_count - self.final_count,
            "retention_rate": round(self.final_count / self.initial_count * 100, 2) if self.initial_count > 0 else 0,
            "steps": self.steps
        }
    
    def save(self, path: Path = CLEANING_LOG_PATH):
        """Save log to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_summary(self):
        """Print a summary of the cleaning log."""
        print("\n" + "=" * 60)
        print("CLEANING LOG SUMMARY")
        print("=" * 60)
        print(f"Timestamp: {self.timestamp}")
        print(f"Initial rows: {self.initial_count:,}")
        print(f"Final rows: {self.final_count:,}")
        print(f"Total removed: {self.initial_count - self.final_count:,} ({100 - self.to_dict()['retention_rate']:.1f}%)")
        print("\nStep-by-step breakdown:")
        print("-" * 60)
        for step in self.steps:
            print(f"  {step['step']}:")
            print(f"    - Removed: {step['rows_removed']:,} rows ({step['removal_percentage']:.2f}%)")
            print(f"    - Reason: {step['reason']}")
        print("=" * 60)


def load_raw_data(nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the raw Flickr dataset.
    
    Args:
        nrows: Number of rows to load (None for all)
    
    Returns:
        DataFrame with raw photo data
    """
    df = pd.read_csv(RAW_DATA_PATH, nrows=nrows, low_memory=False)
    
    # Clean column names (strip whitespace and remove unnamed columns)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Clean data types for date columns (some have mixed str/int due to parsing issues)
    date_columns = [
        'date_taken_year', 'date_taken_month', 'date_taken_day', 
        'date_taken_hour', 'date_taken_minute',
        'date_upload_year', 'date_upload_month', 'date_upload_day',
        'date_upload_hour', 'date_upload_minute'
    ]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_cleaned_data() -> pd.DataFrame:
    """
    Load the cleaned dataset from Parquet cache (preferred) or CSV fallback.
    Falls back to raw data if no cache exists.
    
    Returns:
        DataFrame with cleaned photo data
    """
    if CLEANED_DATA_PATH.exists():
        return pd.read_parquet(CLEANED_DATA_PATH)
    elif CLEANED_CSV_PATH.exists():
        print("Warning: Parquet cache not found, loading from CSV...")
        return pd.read_csv(CLEANED_CSV_PATH)
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


def validate_gps_coordinates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Validate GPS coordinates: remove null, NaN, and out-of-range values.
    
    Filtering rules:
    - Latitude must be between -90 and 90
    - Longitude must be between -180 and 180
    - Neither can be null/NaN
    
    Args:
        df: DataFrame with 'lat' and 'long' columns
    
    Returns:
        Tuple of (filtered DataFrame, number of rows removed)
    """
    initial_count = len(df)
    
    # Check for null/NaN coordinates
    valid_mask = (
        df['lat'].notna() & 
        df['long'].notna() &
        (df['lat'] >= GPS_VALID_RANGE['lat_min']) &
        (df['lat'] <= GPS_VALID_RANGE['lat_max']) &
        (df['long'] >= GPS_VALID_RANGE['lon_min']) &
        (df['long'] <= GPS_VALID_RANGE['lon_max'])
    )
    
    df_clean = df[valid_mask].copy()
    removed = initial_count - len(df_clean)
    
    return df_clean, removed


def validate_date_coherency(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Validate date coherency: date_taken should be <= date_upload.
    
    Photos cannot be uploaded before they were taken.
    
    Args:
        df: DataFrame with date component columns
    
    Returns:
        Tuple of (filtered DataFrame, number of rows removed)
    """
    initial_count = len(df)
    
    # Create datetime columns for comparison
    # Construct comparable values using year, month, day
    # Using a simpler approach: compare as tuples (year, month, day)
    taken_value = (
        df['date_taken_year'] * 10000 + 
        df['date_taken_month'] * 100 + 
        df['date_taken_day']
    )
    upload_value = (
        df['date_upload_year'] * 10000 + 
        df['date_upload_month'] * 100 + 
        df['date_upload_day']
    )
    
    # date_taken should be <= date_upload (can't upload before taking the photo)
    valid_mask = taken_value <= upload_value
    
    df_clean = df[valid_mask].copy()
    removed = initial_count - len(df_clean)
    
    return df_clean, removed


def filter_lyon_bbox(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Filter data to only include points within Lyon bounding box.
    
    Args:
        df: DataFrame with lat/long columns
    
    Returns:
        Tuple of (filtered DataFrame, number of rows removed)
    """
    initial_count = len(df)
    
    mask = (
        (df['lat'] >= LYON_BBOX['lat_min']) & 
        (df['lat'] <= LYON_BBOX['lat_max']) &
        (df['long'] >= LYON_BBOX['lon_min']) & 
        (df['long'] <= LYON_BBOX['lon_max'])
    )
    
    df_clean = df[mask].copy()
    removed = initial_count - len(df_clean)
    
    return df_clean, removed


def remove_corrupted_dates(df: pd.DataFrame, min_year: int = 1990, max_year: int = 2025) -> Tuple[pd.DataFrame, int]:
    """
    Remove rows with impossible date component values.
    
    Found during exploration: some rows have month > 12, day > 31, etc.
    due to CSV parsing issues (shifted columns). Also found future dates (year 2238).
    
    Filtering rules:
    - Month: 1-12
    - Day: 1-31
    - Hour: 0-23
    - Minute: 0-59
    - Year: min_year to max_year (default: 1990-2025)
    
    Args:
        df: DataFrame with date component columns
        min_year: Minimum valid year (default: 1990, before digital cameras)
        max_year: Maximum valid year (default: 2025, current year)
    
    Returns:
        Tuple of (filtered DataFrame, number of rows removed)
    """
    initial_count = len(df)
    
    mask = (
        (df['date_taken_month'] >= 1) & (df['date_taken_month'] <= 12) & 
        (df['date_taken_day'] >= 1) & (df['date_taken_day'] <= 31) & 
        (df['date_taken_hour'] >= 0) & (df['date_taken_hour'] <= 23) &
        (df['date_taken_minute'] >= 0) & (df['date_taken_minute'] <= 59) &
        (df['date_taken_year'] >= min_year) &
        (df['date_taken_year'] <= max_year)
    )
    
    df_clean = df[mask].copy()
    removed = initial_count - len(df_clean)
    
    return df_clean, removed


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


def load_and_clean_data(
    filter_bbox: bool = True,
    save_cache: bool = True,
    save_log: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load raw data and apply all cleaning steps with detailed logging.
    
    Cleaning steps applied (in order):
    1. Validate GPS coordinates (remove null/out-of-range)
    2. Remove rows with corrupted date values
    3. Validate date coherency (date_taken <= date_upload)
    4. Remove duplicate photos (by id)
    5. Optionally filter to Lyon bounding box
    
    Args:
        filter_bbox: Whether to filter to Lyon area (default: True)
        save_cache: Whether to save cleaned data to Parquet (default: True)
        save_log: Whether to save cleaning log (default: True)
        verbose: Whether to print progress (default: True)
    
    Returns:
        Cleaned DataFrame ready for analysis
    """
    log = CleaningLog()
    
    if verbose:
        print("=" * 60)
        print("GRAND LYON PHOTO CLUSTERS - DATA CLEANING PIPELINE")
        print("=" * 60)
    
    # Load raw data
    if verbose:
        print("\n[1/6] Loading raw data...")
    df = load_raw_data()
    log.set_initial(len(df))
    if verbose:
        print(f"      Raw rows: {len(df):,}")
    
    # Step 1: Validate GPS coordinates
    if verbose:
        print("\n[2/6] Validating GPS coordinates...")
    before = len(df)
    df, removed = validate_gps_coordinates(df)
    log.log_step(
        "GPS validation", 
        before, 
        len(df), 
        "Removed rows with null, NaN, or out-of-range GPS coordinates"
    )
    if verbose:
        print(f"      Removed: {removed:,} rows (null/invalid GPS)")
    
    # Step 2: Remove corrupted dates
    if verbose:
        print("\n[3/6] Cleaning corrupted dates...")
    before = len(df)
    df, removed = remove_corrupted_dates(df)
    log.log_step(
        "Date validation", 
        before, 
        len(df), 
        "Removed rows with impossible date values (month>12, year outside 1990-2025, etc.)"
    )
    if verbose:
        print(f"      Removed: {removed:,} rows (corrupted dates)")
    
    # Step 3: Validate date coherency
    if verbose:
        print("\n[4/6] Checking date coherency (taken <= upload)...")
    before = len(df)
    df, removed = validate_date_coherency(df)
    log.log_step(
        "Date coherency", 
        before, 
        len(df), 
        "Removed rows where date_taken > date_upload (impossible: upload before capture)"
    )
    if verbose:
        print(f"      Removed: {removed:,} rows (date_taken > date_upload)")
    
    # Step 4: Remove duplicates
    if verbose:
        print("\n[5/6] Removing duplicates...")
    before = len(df)
    df, removed = remove_duplicates(df, subset=['id'])
    log.log_step(
        "Deduplication", 
        before, 
        len(df), 
        "Removed duplicate photo entries (same photo ID)"
    )
    if verbose:
        print(f"      Removed: {removed:,} rows (duplicate photo IDs)")
    
    # Step 5: Filter to Lyon bbox (optional)
    if filter_bbox:
        if verbose:
            print("\n[6/6] Filtering to Lyon bounding box...")
        before = len(df)
        df, removed = filter_lyon_bbox(df)
        log.log_step(
            "Lyon bbox filter", 
            before, 
            len(df), 
            f"Kept only photos within Lyon area (lat: {LYON_BBOX['lat_min']}-{LYON_BBOX['lat_max']}, lon: {LYON_BBOX['lon_min']}-{LYON_BBOX['lon_max']})"
        )
        if verbose:
            print(f"      Removed: {removed:,} rows (outside Lyon area)")
    
    log.set_final(len(df))
    
    # Save to Parquet cache
    if save_cache:
        if verbose:
            print("\n[SAVE] Saving cleaned data...")
        
        # Save to Parquet (primary - fast)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CLEANED_DATA_PATH, index=False)
        if verbose:
            print(f"       Parquet: {CLEANED_DATA_PATH}")
        
        # Also save to CSV (backup - portable)
        df.to_csv(CLEANED_CSV_PATH, index=False)
        if verbose:
            print(f"       CSV: {CLEANED_CSV_PATH}")
    
    # Save cleaning log
    if save_log:
        log.save()
        if verbose:
            print(f"       Log: {CLEANING_LOG_PATH}")
    
    if verbose:
        log.print_summary()
    
    return df


if __name__ == "__main__":
    # Run the full cleaning pipeline
    df = load_and_clean_data()
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)
    stats = get_data_stats(df)
    for key, value in stats.items():
        print(f"  {key}: {value}")
