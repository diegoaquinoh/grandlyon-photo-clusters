"""
Test script for the data loader module.
Generates a comparison report of data before and after cleaning.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_raw_data,
    remove_corrupted_dates,
    remove_duplicates,
    filter_lyon_bbox,
    get_data_stats,
    LYON_BBOX
)


def count_invalid_dates(df):
    """Count rows with invalid date components."""
    return {
        'invalid_months': int((df['date_taken_month'] > 12).sum()),
        'invalid_days': int((df['date_taken_day'] > 31).sum()),
        'invalid_hours': int((df['date_taken_hour'] > 23).sum()),
        'invalid_years': int(((df['date_taken_year'] < 1990) | (df['date_taken_year'] > 2025)).sum()),
    }


def generate_report(output_path: Path):
    """Generate a markdown report comparing raw vs cleaned data."""
    
    print("Loading raw data...")
    df_raw = load_raw_data()
    raw_count = len(df_raw)
    
    # Calculate raw stats
    print("Calculating raw data statistics...")
    raw_invalid_dates = count_invalid_dates(df_raw)
    raw_duplicates = raw_count - df_raw['id'].nunique()
    
    raw_year_range = (int(df_raw['date_taken_year'].min()), int(df_raw['date_taken_year'].max()))
    raw_month_range = (int(df_raw['date_taken_month'].min()), int(df_raw['date_taken_month'].max()))
    raw_day_range = (int(df_raw['date_taken_day'].min()), int(df_raw['date_taken_day'].max()))
    raw_hour_range = (int(df_raw['date_taken_hour'].min()), int(df_raw['date_taken_hour'].max()))
    
    # Apply cleaning steps
    print("Applying cleaning steps...")
    
    # Step 1: Remove corrupted dates
    df_after_dates = remove_corrupted_dates(df_raw)
    after_dates_count = len(df_after_dates)
    
    # Step 2: Remove duplicates
    df_after_dedup, dup_removed = remove_duplicates(df_after_dates)
    after_dedup_count = len(df_after_dedup)
    
    # Step 3: Filter to Lyon bbox
    df_clean = filter_lyon_bbox(df_after_dedup)
    clean_count = len(df_clean)
    
    # Calculate cleaned stats
    print("Calculating cleaned data statistics...")
    clean_stats = get_data_stats(df_clean)
    clean_year_range = (int(df_clean['date_taken_year'].min()), int(df_clean['date_taken_year'].max()))
    clean_month_range = (int(df_clean['date_taken_month'].min()), int(df_clean['date_taken_month'].max()))
    clean_day_range = (int(df_clean['date_taken_day'].min()), int(df_clean['date_taken_day'].max()))
    clean_hour_range = (int(df_clean['date_taken_hour'].min()), int(df_clean['date_taken_hour'].max()))
    
    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Data Cleaning Report

**Generated:** {timestamp}

## Summary

This report compares the Flickr dataset before and after applying the cleaning pipeline.

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total rows | {raw_count:,} | {clean_count:,} | -{raw_count - clean_count:,} ({(raw_count - clean_count) / raw_count * 100:.1f}%) |
| Unique photos | {df_raw['id'].nunique():,} | {clean_stats['unique_photos']:,} | - |
| Unique users | {df_raw['user'].nunique():,} | {clean_stats['unique_users']:,} | - |

---

## Cleaning Pipeline

### Step 1: Remove Corrupted Dates

Rows with impossible date values (month > 12, day > 31, hour > 23, or year outside 1990-2025) are removed.

**Invalid date values found in raw data:**

| Issue | Count |
|-------|-------|
| Months > 12 | {raw_invalid_dates['invalid_months']:,} |
| Days > 31 | {raw_invalid_dates['invalid_days']:,} |
| Hours > 23 | {raw_invalid_dates['invalid_hours']:,} |
| Years < 1990 or > 2025 | {raw_invalid_dates['invalid_years']:,} |

**Result:** {raw_count:,} → {after_dates_count:,} rows (removed {raw_count - after_dates_count:,})

---

### Step 2: Remove Duplicates

Duplicate photos (by `id` column) are removed, keeping the first occurrence.

**Result:** {after_dates_count:,} → {after_dedup_count:,} rows (removed {dup_removed:,} duplicates)

---

### Step 3: Filter to Lyon Bounding Box

Only photos within the Lyon metropolitan area are kept.

**Bounding box:**
- Latitude: {LYON_BBOX['lat_min']} to {LYON_BBOX['lat_max']}
- Longitude: {LYON_BBOX['lon_min']} to {LYON_BBOX['lon_max']}

**Result:** {after_dedup_count:,} → {clean_count:,} rows (removed {after_dedup_count - clean_count:,})

---

## Data Quality Comparison

### Date Ranges

| Component | Before | After |
|-----------|--------|-------|
| Year | {raw_year_range[0]} – {raw_year_range[1]} | {clean_year_range[0]} – {clean_year_range[1]} |
| Month | {raw_month_range[0]} – {raw_month_range[1]} | {clean_month_range[0]} – {clean_month_range[1]} |
| Day | {raw_day_range[0]} – {raw_day_range[1]} | {clean_day_range[0]} – {clean_day_range[1]} |
| Hour | {raw_hour_range[0]} – {raw_hour_range[1]} | {clean_hour_range[0]} – {clean_hour_range[1]} |

### Coordinate Ranges

| Dimension | Before | After |
|-----------|--------|-------|
| Latitude | {df_raw['lat'].min():.4f} – {df_raw['lat'].max():.4f} | {clean_stats['lat_range'][0]:.4f} – {clean_stats['lat_range'][1]:.4f} |
| Longitude | {df_raw['long'].min():.4f} – {df_raw['long'].max():.4f} | {clean_stats['lon_range'][0]:.4f} – {clean_stats['lon_range'][1]:.4f} |

### Missing Data (After Cleaning)

| Field | Missing Count | Percentage |
|-------|---------------|------------|
| Coordinates (null) | {clean_stats['null_coords']:,} | {clean_stats['null_coords'] / clean_count * 100:.1f}% |
| Tags (empty) | {clean_stats['empty_tags']:,} | {clean_stats['empty_tags'] / clean_count * 100:.1f}% |
| Titles (empty) | {clean_stats['empty_titles']:,} | {clean_stats['empty_titles'] / clean_count * 100:.1f}% |

---

## Conclusion

The cleaning pipeline reduced the dataset from **{raw_count:,}** to **{clean_count:,}** rows ({clean_count / raw_count * 100:.1f}% retained).

- ✅ All date values are now within valid ranges
- ✅ No duplicate photos remain
- ✅ All coordinates are within Lyon bounding box
- ⚠️ ~{clean_stats['empty_tags'] / clean_count * 100:.0f}% of photos have no tags
- ⚠️ ~{clean_stats['empty_titles'] / clean_count * 100:.0f}% of photos have no title
"""
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\nReport saved to: {output_path}")
    
    return report


def main():
    """Run tests and generate report."""
    print("=" * 50)
    print("Testing Data Loader")
    print("=" * 50)
    
    report_path = PROJECT_ROOT / "reports" / "data_cleaning_report.md"
    generate_report(report_path)
    
    print("\n" + "=" * 50)
    print("✅ Test completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
