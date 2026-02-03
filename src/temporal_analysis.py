"""
Temporal Analysis module for the Grand Lyon Photo Clusters project.
Provides utilities for analyzing temporal patterns in photo clusters.

Session 3, Task 4: Classify clusters as permanent POIs, recurring events, or one-time events.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from .data_loader import PROJECT_ROOT

# Output paths
REPORTS_DIR = PROJECT_ROOT / "reports"

# Known Lyon events for annotation
KNOWN_EVENTS = {
    "Fête des Lumières": {"months": [12], "description": "Annual light festival in December"},
    "Nuits de Fourvière": {"months": [6, 7], "description": "Summer arts festival (June-July)"},
    "Biennale de la Danse": {"months": [9], "description": "Biennial dance festival (September, odd years)"},
    "Quais du Polar": {"months": [4], "description": "Crime fiction festival (March-April)"},
    "Just4U": {"months": [6], "description": "Music festival (June)"},
    "Summer Tourism": {"months": [7, 8], "description": "Peak tourist season"},
}


# =============================================================================
# DATETIME UTILITIES
# =============================================================================

def create_date_column(df: pd.DataFrame) -> pd.Series:
    """
    Create a date column from date components.
    
    Args:
        df: DataFrame with date_taken_year, month, day columns
    
    Returns:
        Series of date objects
    """
    def safe_date(row):
        try:
            year = int(row['date_taken_year'])
            month = int(row['date_taken_month'])
            day = int(row['date_taken_day'])
            # Validate
            if month < 1 or month > 12:
                return pd.NaT
            if day < 1 or day > 31:
                return pd.NaT
            if year < 1990 or year > 2026:
                return pd.NaT
            return pd.Timestamp(year=year, month=month, day=1)  # Use first of month for aggregation
        except (ValueError, TypeError):
            return pd.NaT
    
    return df.apply(safe_date, axis=1)


def create_year_month_column(df: pd.DataFrame) -> pd.Series:
    """
    Create a year-month string column (e.g., '2015-12').
    
    Args:
        df: DataFrame with date_taken_year, month columns
    
    Returns:
        Series of year-month strings
    """
    return df['date_taken_year'].astype(int).astype(str) + '-' + \
           df['date_taken_month'].astype(int).astype(str).str.zfill(2)


# =============================================================================
# TEMPORAL AGGREGATION
# =============================================================================

def aggregate_by_period(
    df: pd.DataFrame,
    cluster_col: str = 'cluster',
    period: str = 'month'
) -> pd.DataFrame:
    """
    Aggregate photo counts by cluster and time period.
    
    Args:
        df: DataFrame with cluster and date columns
        cluster_col: Name of cluster column
        period: Aggregation period ('month', 'year', 'quarter')
    
    Returns:
        DataFrame with columns [cluster, period, count]
    """
    df = df.copy()
    
    if period == 'month':
        df['period'] = df['date_taken_month'].astype(int)
    elif period == 'year':
        df['period'] = df['date_taken_year'].astype(int)
    elif period == 'quarter':
        df['period'] = ((df['date_taken_month'].astype(int) - 1) // 3 + 1)
    elif period == 'year_month':
        df['period'] = create_year_month_column(df)
    else:
        raise ValueError(f"Unknown period type: {period}")
    
    # Group and count
    aggregated = df.groupby([cluster_col, 'period']).size().reset_index(name='count')
    
    return aggregated


def aggregate_monthly_by_year(
    df: pd.DataFrame,
    cluster_col: str = 'cluster'
) -> pd.DataFrame:
    """
    Create a pivot table: cluster × year-month with photo counts.
    
    Args:
        df: DataFrame with cluster and date columns
        cluster_col: Name of cluster column
    
    Returns:
        DataFrame with clusters as rows, year-months as columns
    """
    df = df.copy()
    df['year_month'] = create_year_month_column(df)
    
    pivot = pd.pivot_table(
        df,
        values='id',  # Count unique photo IDs
        index=cluster_col,
        columns='year_month',
        aggfunc='count',
        fill_value=0
    )
    
    # Sort columns chronologically
    sorted_cols = sorted(pivot.columns)
    pivot = pivot[sorted_cols]
    
    return pivot


# =============================================================================
# TEMPORAL STATISTICS
# =============================================================================

def compute_cluster_temporal_stats(
    df: pd.DataFrame,
    cluster_col: str = 'cluster'
) -> pd.DataFrame:
    """
    Compute temporal statistics for each cluster.
    
    Stats computed:
    - Monthly distribution (mean, std, CV per month)
    - Peak months (which months have highest activity)
    - Year range (first and last year of activity)
    - Total photos
    - December ratio (for Fête des Lumières detection)
    - Summer ratio (July-August)
    
    Args:
        df: DataFrame with cluster and date columns
        cluster_col: Name of cluster column
    
    Returns:
        DataFrame with one row per cluster and temporal statistics
    """
    df = df.copy()
    
    # Get monthly aggregation
    monthly = aggregate_by_period(df, cluster_col, period='month')
    
    stats_list = []
    
    for cluster_id in df[cluster_col].unique():
        if cluster_id == -1:  # Skip noise
            continue
        
        cluster_photos = df[df[cluster_col] == cluster_id]
        cluster_monthly = monthly[monthly[cluster_col] == cluster_id]
        
        # Basic counts
        total_photos = len(cluster_photos)
        n_years = cluster_photos['date_taken_year'].nunique()
        year_min = int(cluster_photos['date_taken_year'].min())
        year_max = int(cluster_photos['date_taken_year'].max())
        n_months_active = len(cluster_monthly)
        
        # Monthly distribution
        if not cluster_monthly.empty:
            # Create full 12-month vector
            month_counts = [0] * 12
            for _, row in cluster_monthly.iterrows():
                month = int(row['period']) - 1  # 0-indexed
                if 0 <= month < 12:
                    month_counts[month] = row['count']
            
            # Calculate stats
            month_mean = np.mean(month_counts) if month_counts else 0
            month_std = np.std(month_counts) if month_counts else 0
            month_cv = month_std / month_mean if month_mean > 0 else 0
            
            # Peak month
            peak_month = np.argmax(month_counts) + 1  # 1-indexed
            peak_count = max(month_counts)
            peak_ratio = peak_count / total_photos if total_photos > 0 else 0
            
            # Special month ratios
            december_count = month_counts[11]  # December (0-indexed = 11)
            december_ratio = december_count / total_photos if total_photos > 0 else 0
            
            summer_count = month_counts[6] + month_counts[7]  # July + August
            summer_ratio = summer_count / total_photos if total_photos > 0 else 0
            
            # Count months with significant activity (>5% of total)
            threshold = total_photos * 0.05
            months_with_activity = sum(1 for c in month_counts if c >= threshold)
        else:
            month_mean = month_std = month_cv = 0
            peak_month = peak_count = peak_ratio = 0
            december_ratio = summer_ratio = 0
            months_with_activity = 0
        
        stats_list.append({
            'cluster': cluster_id,
            'total_photos': total_photos,
            'n_years': n_years,
            'year_min': year_min,
            'year_max': year_max,
            'n_months_active': n_months_active,
            'months_with_activity': months_with_activity,
            'month_mean': month_mean,
            'month_std': month_std,
            'month_cv': month_cv,
            'peak_month': peak_month,
            'peak_count': peak_count,
            'peak_ratio': peak_ratio,
            'december_ratio': december_ratio,
            'summer_ratio': summer_ratio,
        })
    
    return pd.DataFrame(stats_list)


# =============================================================================
# PEAK DETECTION
# =============================================================================

def detect_peaks(
    series: pd.Series,
    threshold: float = 2.0
) -> List[int]:
    """
    Detect peaks in a time series using z-score method.
    
    Args:
        series: Time series of counts
        threshold: Z-score threshold (default: 2.0 = 2 standard deviations)
    
    Returns:
        List of indices where peaks occur
    """
    if series.empty or series.std() == 0:
        return []
    
    z_scores = (series - series.mean()) / series.std()
    peaks = z_scores[z_scores > threshold].index.tolist()
    
    return peaks


def detect_monthly_peaks(
    df: pd.DataFrame,
    cluster_id: int,
    threshold: float = 1.5
) -> Dict[str, Any]:
    """
    Detect peak months for a specific cluster.
    
    Args:
        df: DataFrame with cluster and date columns
        cluster_id: Cluster to analyze
        threshold: Z-score threshold for peak detection
    
    Returns:
        Dictionary with peak information
    """
    cluster_df = df[df['cluster'] == cluster_id]
    
    if cluster_df.empty:
        return {'peaks': [], 'peak_months': [], 'peak_year_months': []}
    
    # Monthly aggregation across all years
    monthly = cluster_df.groupby('date_taken_month').size()
    monthly = monthly.reindex(range(1, 13), fill_value=0)
    
    # Detect peaks
    peak_months = detect_peaks(monthly, threshold)
    
    # Year-month aggregation for spike detection
    ym = create_year_month_column(cluster_df)
    cluster_df = cluster_df.copy()
    cluster_df['year_month'] = ym
    ym_counts = cluster_df.groupby('year_month').size()
    peak_year_months = detect_peaks(ym_counts, threshold)
    
    return {
        'peaks': peak_months,
        'peak_months': [int(m) for m in peak_months],
        'peak_year_months': peak_year_months,
        'monthly_counts': monthly.to_dict(),
    }


# =============================================================================
# CLUSTER TYPE CLASSIFICATION
# =============================================================================

def classify_cluster_type(stats: Dict[str, Any]) -> str:
    """
    Classify a cluster based on its temporal statistics.
    
    Classification rules:
    - PERMANENT: Consistent activity (CV < 0.8, activity in 6+ months)
    - RECURRING: Seasonal pattern (december_ratio > 0.25 or peak_ratio > 0.35)
    - ONE_TIME: Single spike dominates (peak_ratio > 0.5, activity in < 4 months)
    
    Args:
        stats: Dictionary of temporal statistics for one cluster
    
    Returns:
        Classification string: 'permanent', 'recurring', or 'one_time'
    """
    cv = stats.get('month_cv', 0)
    peak_ratio = stats.get('peak_ratio', 0)
    december_ratio = stats.get('december_ratio', 0)
    summer_ratio = stats.get('summer_ratio', 0)
    months_active = stats.get('months_with_activity', 0)
    n_years = stats.get('n_years', 0)
    
    # Strong December pattern -> Fête des Lumières (recurring)
    if december_ratio > 0.25:
        return 'recurring'
    
    # Strong summer pattern -> tourism/festivals (recurring)
    if summer_ratio > 0.35:
        return 'recurring'
    
    # Very concentrated (>50% in one month) and limited activity -> one-time
    if peak_ratio > 0.5 and months_active <= 3:
        return 'one_time'
    
    # High variability with clear peak -> recurring
    if cv > 1.0 and peak_ratio > 0.35:
        return 'recurring'
    
    # Consistent activity across months -> permanent
    if cv < 0.8 and months_active >= 6:
        return 'permanent'
    
    # Multi-year presence with moderate variability -> permanent
    if n_years >= 5 and cv < 1.2:
        return 'permanent'
    
    # Default: if concentrated but not extremely, likely recurring
    if peak_ratio > 0.3:
        return 'recurring'
    
    return 'permanent'  # Default to permanent


def classify_all_clusters(
    df: pd.DataFrame,
    cluster_col: str = 'cluster'
) -> Dict[int, Dict[str, Any]]:
    """
    Classify all clusters by their temporal patterns.
    
    Args:
        df: DataFrame with cluster and date columns
        cluster_col: Name of cluster column
    
    Returns:
        Dictionary mapping cluster ID to classification result
    """
    print("Computing temporal statistics for all clusters...")
    stats_df = compute_cluster_temporal_stats(df, cluster_col)
    
    results = {}
    type_counts = defaultdict(int)
    
    for _, row in stats_df.iterrows():
        cluster_id = row['cluster']
        stats = row.to_dict()
        
        # Classify
        cluster_type = classify_cluster_type(stats)
        type_counts[cluster_type] += 1
        
        # Detect events
        peak_info = detect_monthly_peaks(df, cluster_id)
        
        # Get potential event matches
        matched_events = []
        for event_name, event_info in KNOWN_EVENTS.items():
            if any(m in peak_info['peak_months'] for m in event_info['months']):
                matched_events.append(event_name)
        
        results[cluster_id] = {
            'type': cluster_type,
            'stats': stats,
            'peak_info': peak_info,
            'matched_events': matched_events,
        }
    
    print(f"Classification complete:")
    print(f"  - Permanent POIs: {type_counts['permanent']}")
    print(f"  - Recurring Events: {type_counts['recurring']}")
    print(f"  - One-time Events: {type_counts['one_time']}")
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_cluster_timeseries(
    df: pd.DataFrame,
    cluster_ids: List[int],
    title: str = "Photo Activity Over Time",
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot time series of photo counts for selected clusters.
    
    Args:
        df: DataFrame with cluster and date columns
        cluster_ids: List of cluster IDs to plot
        title: Plot title
        figsize: Figure size tuple
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    df = df.copy()
    df['year_month'] = create_year_month_column(df)
    
    for cluster_id in cluster_ids:
        cluster_df = df[df['cluster'] == cluster_id]
        ym_counts = cluster_df.groupby('year_month').size().sort_index()
        
        if not ym_counts.empty:
            ax.plot(range(len(ym_counts)), ym_counts.values, 
                   label=f"Cluster {cluster_id}", linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel("Time (Year-Month)")
    ax.set_ylabel("Photo Count")
    ax.set_title(title)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_monthly_distribution(
    df: pd.DataFrame,
    cluster_ids: List[int],
    title: str = "Monthly Photo Distribution",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot bar chart of monthly distribution for selected clusters.
    
    Args:
        df: DataFrame with cluster and date columns
        cluster_ids: List of cluster IDs to plot
        title: Plot title
        figsize: Figure size tuple
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x = np.arange(12)
    width = 0.8 / len(cluster_ids)
    
    for i, cluster_id in enumerate(cluster_ids):
        cluster_df = df[df['cluster'] == cluster_id]
        monthly = cluster_df.groupby('date_taken_month').size()
        monthly = monthly.reindex(range(1, 13), fill_value=0)
        
        # Normalize to percentage
        total = monthly.sum()
        monthly_pct = monthly / total * 100 if total > 0 else monthly
        
        offset = (i - len(cluster_ids) / 2 + 0.5) * width
        ax.bar(x + offset, monthly_pct.values, width, 
               label=f"Cluster {cluster_id}", alpha=0.8)
    
    ax.set_xlabel("Month")
    ax.set_ylabel("% of Cluster Photos")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(month_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_cluster_heatmap(
    df: pd.DataFrame,
    top_n: int = 30,
    title: str = "Cluster × Month Photo Heatmap",
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create a heatmap showing photo counts by cluster and month.
    
    Args:
        df: DataFrame with cluster and date columns
        top_n: Number of top clusters to show
        title: Plot title
        figsize: Figure size tuple
    
    Returns:
        Matplotlib figure object
    """
    # Get monthly aggregation
    monthly = aggregate_by_period(df, 'cluster', period='month')
    
    # Pivot to matrix
    pivot = monthly.pivot(index='cluster', columns='period', values='count')
    pivot = pivot.fillna(0)
    
    # Reindex columns to show all months
    pivot = pivot.reindex(columns=range(1, 13), fill_value=0)
    
    # Filter to top clusters by total photos
    cluster_totals = pivot.sum(axis=1).sort_values(ascending=False)
    top_clusters = cluster_totals.head(top_n).index
    pivot = pivot.loc[top_clusters]
    
    # Normalize rows (each cluster sums to 1)
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    sns.heatmap(
        pivot_norm,
        annot=False,
        cmap='YlOrRd',
        xticklabels=month_names,
        yticklabels=[f"C{c}" for c in pivot_norm.index],
        ax=ax,
        cbar_kws={'label': 'Proportion of Cluster Photos'}
    )
    
    ax.set_xlabel("Month")
    ax.set_ylabel("Cluster")
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_classification_summary(
    classifications: Dict[int, Dict[str, Any]],
    title: str = "Cluster Classification Summary",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a summary visualization of cluster classifications.
    
    Args:
        classifications: Output from classify_all_clusters()
        title: Plot title
        figsize: Figure size tuple
    
    Returns:
        Matplotlib figure object
    """
    # Count by type
    type_counts = defaultdict(int)
    type_photos = defaultdict(int)
    
    for cluster_id, info in classifications.items():
        cluster_type = info['type']
        type_counts[cluster_type] += 1
        type_photos[cluster_type] += info['stats'].get('total_photos', 0)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Colors
    colors = {'permanent': '#2E86AB', 'recurring': '#A23B72', 'one_time': '#F18F01'}
    labels = ['Permanent POI', 'Recurring Event', 'One-time Event']
    types = ['permanent', 'recurring', 'one_time']
    
    # Plot 1: Number of clusters
    ax1 = axes[0]
    counts = [type_counts[t] for t in types]
    bars1 = ax1.bar(labels, counts, color=[colors[t] for t in types])
    ax1.set_ylabel("Number of Clusters")
    ax1.set_title("Clusters by Type")
    ax1.bar_label(bars1, padding=3)
    ax1.tick_params(axis='x', rotation=15)
    
    # Plot 2: Number of photos
    ax2 = axes[1]
    photos = [type_photos[t] for t in types]
    bars2 = ax2.bar(labels, photos, color=[colors[t] for t in types])
    ax2.set_ylabel("Number of Photos")
    ax2.set_title("Photos by Cluster Type")
    ax2.bar_label(bars2, padding=3, fmt='%d')
    ax2.tick_params(axis='x', rotation=15)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


# =============================================================================
# REPORT GENERATION
# =============================================================================

def save_temporal_visualizations(
    df: pd.DataFrame,
    classifications: Dict[int, Dict[str, Any]],
    output_dir: Path = REPORTS_DIR
) -> List[Path]:
    """
    Generate and save all temporal analysis visualizations.
    
    Args:
        df: DataFrame with cluster and date columns
        classifications: Output from classify_all_clusters()
        output_dir: Directory to save visualizations
    
    Returns:
        List of paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    
    print("Generating temporal visualizations...")
    
    # 1. Cluster x Month heatmap
    fig = plot_cluster_heatmap(df, top_n=30)
    heatmap_path = output_dir / "temporal_heatmap.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(heatmap_path)
    print(f"  Saved: {heatmap_path}")
    
    # 2. Classification summary
    fig = plot_classification_summary(classifications)
    summary_path = output_dir / "temporal_classification_summary.png"
    fig.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(summary_path)
    print(f"  Saved: {summary_path}")
    
    # 3. Top clusters time series
    top_clusters = sorted(
        [(cid, info['stats']['total_photos']) for cid, info in classifications.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    top_ids = [c[0] for c in top_clusters]
    
    fig = plot_cluster_timeseries(df, top_ids, title="Top 10 Clusters: Activity Over Time")
    ts_path = output_dir / "temporal_timeseries.png"
    fig.savefig(ts_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(ts_path)
    print(f"  Saved: {ts_path}")
    
    # 4. Monthly distribution for example clusters (one of each type)
    example_clusters = []
    for cluster_type in ['permanent', 'recurring', 'one_time']:
        for cid, info in classifications.items():
            if info['type'] == cluster_type:
                example_clusters.append(cid)
                break
    
    if example_clusters:
        fig = plot_monthly_distribution(
            df, example_clusters,
            title="Monthly Distribution: Example Clusters by Type"
        )
        dist_path = output_dir / "temporal_monthly_distribution.png"
        fig.savefig(dist_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(dist_path)
        print(f"  Saved: {dist_path}")
    
    return saved_files


def create_temporal_report(
    df: pd.DataFrame,
    classifications: Dict[int, Dict[str, Any]],
    output_path: Path = None
) -> str:
    """
    Create a markdown report summarizing temporal analysis findings.
    
    Args:
        df: DataFrame with cluster and date columns
        classifications: Output from classify_all_clusters()
        output_path: Path to save report (optional)
    
    Returns:
        Markdown report string
    """
    if output_path is None:
        output_path = REPORTS_DIR / "temporal_analysis.md"
    
    # Count by type
    type_counts = defaultdict(int)
    type_photos = defaultdict(int)
    
    for cluster_id, info in classifications.items():
        cluster_type = info['type']
        type_counts[cluster_type] += 1
        type_photos[cluster_type] += info['stats'].get('total_photos', 0)
    
    total_clusters = len(classifications)
    total_photos = sum(type_photos.values())
    
    # Find notable clusters
    december_clusters = []
    summer_clusters = []
    
    for cid, info in classifications.items():
        if info['stats']['december_ratio'] > 0.2:
            december_clusters.append((cid, info['stats']['december_ratio']))
        if info['stats']['summer_ratio'] > 0.25:
            summer_clusters.append((cid, info['stats']['summer_ratio']))
    
    december_clusters.sort(key=lambda x: x[1], reverse=True)
    summer_clusters.sort(key=lambda x: x[1], reverse=True)
    
    # Build report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Temporal Analysis Report

**Generated:** {timestamp}  
**Total Clusters Analyzed:** {total_clusters}  
**Total Photos:** {total_photos:,}

---

## Summary

This analysis classifies photo clusters based on their temporal patterns to distinguish between:

- **Permanent POIs**: Consistent year-round activity (e.g., landmarks, museums)
- **Recurring Events**: Seasonal patterns (e.g., Fête des Lumières, summer tourism)
- **One-time Events**: Single spike of activity (e.g., concerts, sports events)

### Classification Results

| Type | Clusters | Photos | % of Photos |
|------|----------|--------|-------------|
| **Permanent POI** | {type_counts['permanent']} | {type_photos['permanent']:,} | {type_photos['permanent']/total_photos*100:.1f}% |
| **Recurring Event** | {type_counts['recurring']} | {type_photos['recurring']:,} | {type_photos['recurring']/total_photos*100:.1f}% |
| **One-time Event** | {type_counts['one_time']} | {type_photos['one_time']:,} | {type_photos['one_time']/total_photos*100:.1f}% |

---

## Key Findings

### December Spikes (Fête des Lumières Candidates)

The following clusters show strong December activity:

| Cluster | December Ratio | Total Photos |
|---------|---------------|--------------|
"""
    
    for cid, ratio in december_clusters[:10]:
        info = classifications[cid]
        report += f"| {cid} | {ratio*100:.1f}% | {info['stats']['total_photos']:,} |\n"
    
    report += f"""
### Summer Peaks (Tourism/Festival Candidates)

The following clusters show strong July-August activity:

| Cluster | Summer Ratio | Total Photos |
|---------|-------------|--------------|
"""
    
    for cid, ratio in summer_clusters[:10]:
        info = classifications[cid]
        report += f"| {cid} | {ratio*100:.1f}% | {info['stats']['total_photos']:,} |\n"
    
    report += """
---

## Visualizations

### Cluster × Month Heatmap

The heatmap below shows the normalized monthly distribution for the top 30 clusters:

![Temporal Heatmap](temporal_heatmap.png)

### Classification Summary

![Classification Summary](temporal_classification_summary.png)

### Time Series: Top 10 Clusters

![Time Series](temporal_timeseries.png)

### Monthly Distribution by Type

![Monthly Distribution](temporal_monthly_distribution.png)

---

## Methodology

### Classification Rules

1. **Recurring Events** are identified when:
   - December ratio > 25% (Fête des Lumières pattern)
   - Summer ratio > 35% (tourism pattern)
   - High coefficient of variation (CV > 1.0) with clear peak

2. **One-time Events** are identified when:
   - Peak month contains > 50% of photos
   - Activity in 3 or fewer months

3. **Permanent POIs** are the default when:
   - Coefficient of variation < 0.8
   - Activity in 6+ months
   - Multi-year presence (5+ years)

### Peak Detection

Z-score based peak detection identifies months with activity more than 1.5 standard deviations above the mean.

---

## Known Lyon Events Reference

| Event | Months | Description |
|-------|--------|-------------|
| Fête des Lumières | December | Annual light festival |
| Nuits de Fourvière | June-July | Summer arts festival |
| Biennale de la Danse | September | Biennial dance festival |
| Quais du Polar | April | Crime fiction festival |
| Summer Tourism | July-August | Peak tourist season |
"""
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report saved to: {output_path}")
    
    return report


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_temporal_analysis(
    df: pd.DataFrame = None,
    output_dir: Path = REPORTS_DIR
) -> Dict[str, Any]:
    """
    Run complete temporal analysis pipeline.
    
    Args:
        df: DataFrame with cluster and date columns (loads default if None)
        output_dir: Directory for output files
    
    Returns:
        Dictionary with classifications and paths to generated files
    """
    import pandas as pd
    from pathlib import Path
    
    if df is None:
        print("Loading clustered data...")
        data_path = PROJECT_ROOT / "data" / "flickr_clustered.csv"
        df = pd.read_csv(data_path)
    
    print(f"Analyzing {len(df):,} photos in {df['cluster'].nunique()} clusters...")
    
    # Run classification
    classifications = classify_all_clusters(df)
    
    # Generate visualizations
    viz_paths = save_temporal_visualizations(df, classifications, output_dir)
    
    # Generate report
    report_path = output_dir / "temporal_analysis.md"
    create_temporal_report(df, classifications, report_path)
    
    return {
        'classifications': classifications,
        'report_path': report_path,
        'visualization_paths': viz_paths,
    }


def main():
    """Run temporal analysis from command line."""
    print("=" * 60)
    print("TEMPORAL ANALYSIS - Grand Lyon Photo Clusters")
    print("=" * 60)
    
    result = run_temporal_analysis()
    
    print("\n" + "=" * 60)
    print("✅ Temporal analysis complete!")
    print(f"   Report: {result['report_path']}")
    print(f"   Visualizations: {len(result['visualization_paths'])} files generated")
    print("=" * 60)


if __name__ == "__main__":
    main()
