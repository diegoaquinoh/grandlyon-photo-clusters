"""
Map visualization module for the Grand Lyon Photo Clusters project.
Provides utilities for creating interactive Folium maps of photo locations.
"""

import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
from pathlib import Path
from typing import Optional, List
import numpy as np
import json

from .data_loader import load_cleaned_data, load_and_clean_data, LYON_BBOX, PROJECT_ROOT

# Output paths
APP_DIR = PROJECT_ROOT / "app"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_MAP_PATH = APP_DIR / "map.html"

# Lyon center coordinates
LYON_CENTER = [45.75, 4.85]


def create_base_map(
    center: list = LYON_CENTER,
    zoom_start: int = 12,
    tiles: str = "CartoDB positron"
) -> folium.Map:
    """
    Create a base Folium map centered on Lyon.
    
    Args:
        center: [lat, lon] center coordinates
        zoom_start: Initial zoom level
        tiles: Map tile style
    
    Returns:
        Folium Map object
    """
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles=tiles
    )
    return m


def add_sample_markers(
    m: folium.Map,
    df: pd.DataFrame,
    sample_size: int = 10000,
    use_clustering: bool = True
) -> folium.Map:
    """
    Add sample photo markers to the map.
    
    Args:
        m: Folium Map object
        df: DataFrame with photo data
        sample_size: Number of points to sample
        use_clustering: Whether to use MarkerCluster for performance
    
    Returns:
        Map with markers added
    """
    # Sample data if needed
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    # Create marker cluster or feature group
    if use_clustering:
        marker_group = MarkerCluster(name="Photo Locations")
    else:
        marker_group = folium.FeatureGroup(name="Photo Locations")
    
    # Add markers with tooltips
    for _, row in df_sample.iterrows():
        # Build tooltip content
        tags = str(row.get('tags', ''))[:100] + '...' if len(str(row.get('tags', ''))) > 100 else str(row.get('tags', ''))
        title = str(row.get('title', 'No title'))
        year = int(row.get('date_taken_year', 0))
        
        tooltip = f"""
        <b>{title}</b><br>
        <b>Year:</b> {year}<br>
        <b>Tags:</b> {tags}<br>
        <b>User:</b> {row.get('user', 'Unknown')}
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=5,
            color='#3388ff',
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(tooltip, max_width=300),
            tooltip=title[:30] if title else "Photo"
        ).add_to(marker_group)
    
    marker_group.add_to(m)
    return m


def add_heatmap(
    m: folium.Map,
    df: pd.DataFrame,
    name: str = "Density Heatmap"
) -> folium.Map:
    """
    Add a heatmap layer showing photo density.
    
    Args:
        m: Folium Map object
        df: DataFrame with photo data
        name: Layer name
    
    Returns:
        Map with heatmap added
    """
    # Prepare coordinates for heatmap
    heat_data = df[['lat', 'long']].values.tolist()
    
    # Create heatmap layer
    heatmap = HeatMap(
        heat_data,
        name=name,
        min_opacity=0.3,
        radius=15,
        blur=10,
        gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}
    )
    
    heatmap.add_to(m)
    return m


def add_layer_control(m: folium.Map) -> folium.Map:
    """Add layer control to toggle layers on/off."""
    folium.LayerControl(collapsed=True).add_to(m)  # Collapsed by default to not hide map
    return m


def create_photo_map(
    df: Optional[pd.DataFrame] = None,
    sample_size: int = 2000,
    include_heatmap: bool = True,
    include_markers: bool = True,
    output_path: Optional[Path] = None
) -> folium.Map:
    """
    Create a complete photo location map.
    
    Args:
        df: DataFrame with photo data (loads cleaned data if None)
        sample_size: Number of marker points to show
        include_heatmap: Whether to add heatmap layer
        include_markers: Whether to add marker layer
        output_path: Path to save HTML (None to skip saving)
    
    Returns:
        Folium Map object
    """
    # Load data if not provided
    if df is None:
        try:
            df = load_cleaned_data()
        except:
            print("Cleaned data not found, running cleaning pipeline...")
            df = load_and_clean_data()
    
    print(f"Creating map with {len(df):,} photos...")
    
    # Create base map
    m = create_base_map()
    
    # Add layers
    if include_heatmap:
        m = add_heatmap(m, df)
        print("  Added heatmap layer")
    
    if include_markers:
        m = add_sample_markers(m, df, sample_size=sample_size)
        print(f"  Added marker layer (sample of {min(sample_size, len(df)):,})")
    
    # Add controls
    m = add_layer_control(m)
    
    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        print(f"  Saved to: {output_path}")
    
    return m


# =============================================================================
# CLUSTER VISUALIZATION
# =============================================================================

def load_cluster_descriptors() -> dict:
    """
    Load cluster descriptors from the text mining output.
    
    Returns:
        Dictionary mapping cluster ID to list of top terms
    """
    descriptors_path = REPORTS_DIR / "cluster_descriptors.json"
    
    if not descriptors_path.exists():
        print(f"  Warning: No cluster descriptors found at {descriptors_path}")
        return {}
    
    try:
        with open(descriptors_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract just the term names (top 5) for each cluster
        result = {}
        for cluster_id, terms in data.get('clusters', {}).items():
            result[int(cluster_id)] = [t['term'] for t in terms[:5]]
        
        return result
    except Exception as e:
        print(f"  Warning: Could not load cluster descriptors: {e}")
        return {}

def generate_cluster_colors(n_clusters: int) -> List[str]:
    """
    Generate a list of distinct colors for clusters.
    
    Args:
        n_clusters: Number of clusters to generate colors for
    
    Returns:
        List of hex color strings
    """
    # Use a colorful palette for up to 20 clusters, then cycle
    base_colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
    ]
    
    if n_clusters <= len(base_colors):
        return base_colors[:n_clusters]
    
    # For more clusters, generate using HSL
    colors = []
    for i in range(n_clusters):
        hue = (i * 137.5) % 360  # Golden angle for good distribution
        colors.append(f'hsl({hue}, 70%, 50%)')
    return colors


def add_cluster_markers(
    m: folium.Map,
    df: pd.DataFrame,
    min_cluster_size: int = 10,
    show_noise: bool = False,
    sample_per_cluster: int = 200,
    cluster_colors: Optional[List[str]] = None,
    cluster_descriptors: Optional[dict] = None
) -> folium.Map:
    """
    Add color-coded cluster markers to the map.
    
    Args:
        m: Folium Map object
        df: DataFrame with photo data (must have 'cluster' column)
        min_cluster_size: Minimum cluster size to display
        show_noise: Whether to show noise points (cluster=-1)
        sample_per_cluster: Max points to show per cluster (for performance)
        cluster_colors: Optional list of colors for clusters
        cluster_descriptors: Optional dict mapping cluster ID to list of top terms
    
    Returns:
        Map with cluster markers added
    """
    if 'cluster' not in df.columns:
        raise ValueError("DataFrame must have 'cluster' column")
    
    # Load descriptors if not provided
    if cluster_descriptors is None:
        cluster_descriptors = load_cluster_descriptors()
    
    # Get cluster info
    cluster_counts = df['cluster'].value_counts()
    unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
    
    # Filter clusters by size
    valid_clusters = [c for c in unique_clusters if cluster_counts.get(c, 0) >= min_cluster_size]
    
    # Generate colors
    if cluster_colors is None:
        cluster_colors = generate_cluster_colors(len(valid_clusters))
    
    color_map = {c: cluster_colors[i % len(cluster_colors)] for i, c in enumerate(valid_clusters)}
    color_map[-1] = '#888888'  # Gray for noise
    
    # Create feature groups for each cluster
    for cluster_id in valid_clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        cluster_size = len(cluster_df)
        
        # Get descriptors for this cluster
        top_terms = cluster_descriptors.get(cluster_id, [])
        descriptor_str = ", ".join(top_terms) if top_terms else "(no descriptors)"
        
        # Sample if too many points
        if len(cluster_df) > sample_per_cluster:
            cluster_df = cluster_df.sample(n=sample_per_cluster, random_state=42)
        
        # Use descriptors in group name if available
        group_name = f"Cluster {cluster_id} (n={cluster_size:,})"
        if top_terms:
            short_desc = ", ".join(top_terms[:3])
            group_name = f"Cluster {cluster_id}: {short_desc} (n={cluster_size:,})"
        
        group = folium.FeatureGroup(name=group_name)
        
        for _, row in cluster_df.iterrows():
            # Build popup content
            tags = str(row.get('tags', ''))[:80]
            if len(str(row.get('tags', ''))) > 80:
                tags += '...'
            title = str(row.get('title', 'No title'))[:50]
            year = int(row.get('date_taken_year', 0)) if pd.notna(row.get('date_taken_year')) else 'N/A'
            
            popup_html = f"""
            <div style="min-width: 220px;">
                <b style="color: {color_map[cluster_id]};">Cluster {cluster_id}</b><br>
                <b>Size:</b> {cluster_size:,} photos<br>
                <b>Top terms:</b> <i>{descriptor_str}</i><br>
                <hr style="margin: 5px 0;">
                <b>Title:</b> {title}<br>
                <b>Year:</b> {year}<br>
                <b>Tags:</b> {tags}<br>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['long']],
                radius=6,
                color=color_map[cluster_id],
                fill=True,
                fill_color=color_map[cluster_id],
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"Cluster {cluster_id}: {', '.join(top_terms[:2]) if top_terms else 'No description'}"
            ).add_to(group)
        
        group.add_to(m)
    
    # Add noise points if requested
    if show_noise and -1 in df['cluster'].values:
        noise_df = df[df['cluster'] == -1]
        noise_count = len(noise_df)
        
        if len(noise_df) > sample_per_cluster:
            noise_df = noise_df.sample(n=sample_per_cluster, random_state=42)
        
        noise_group = folium.FeatureGroup(name=f"Noise (n={noise_count:,})", show=False)
        
        for _, row in noise_df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['long']],
                radius=3,
                color='#888888',
                fill=True,
                fill_opacity=0.4,
                tooltip="Noise point"
            ).add_to(noise_group)
        
        noise_group.add_to(m)
    
    return m


def add_cluster_legend(
    m: folium.Map, 
    cluster_info: dict,
    cluster_descriptors: Optional[dict] = None
) -> folium.Map:
    """
    Add a legend showing cluster colors, sizes, and descriptors.
    
    Args:
        m: Folium Map object
        cluster_info: Dict of {cluster_id: (color, size)}
        cluster_descriptors: Optional dict mapping cluster ID to list of top terms
    
    Returns:
        Map with legend added
    """
    if cluster_descriptors is None:
        cluster_descriptors = {}
    
    # Build legend HTML
    legend_items = ""
    for cluster_id, (color, size) in sorted(cluster_info.items(), key=lambda x: -x[1][1])[:15]:  # Top 15 by size
        # Get short descriptor
        terms = cluster_descriptors.get(cluster_id, [])
        desc = ", ".join(terms[:2]) if terms else ""
        desc_html = f'<span style="color: #666; font-style: italic;"> - {desc}</span>' if desc else ""
        
        legend_items += f"""
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <span style="background-color: {color}; width: 12px; height: 12px; 
                         border-radius: 50%; display: inline-block; margin-right: 5px;"></span>
            <span style="font-size: 11px;">Cluster {cluster_id} ({size:,}){desc_html}</span>
        </div>
        """
    
    if len(cluster_info) > 15:
        legend_items += f'<div style="font-size: 10px; color: #666;">...and {len(cluster_info) - 15} more</div>'
    
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid #ccc; font-family: Arial, sans-serif; max-width: 350px;">
        <div style="font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid #ccc; padding-bottom: 3px;">
            Clusters (by size)
        </div>
        {legend_items}
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def create_cluster_map(
    df: pd.DataFrame,
    min_cluster_size: int = 10,
    show_noise: bool = False,
    sample_per_cluster: int = 200,
    include_heatmap: bool = False,
    output_path: Optional[Path] = None
) -> folium.Map:
    """
    Create a complete cluster visualization map.
    
    Args:
        df: DataFrame with photo data (must have 'cluster' column)
        min_cluster_size: Minimum cluster size to display
        show_noise: Whether to show noise points
        sample_per_cluster: Max points per cluster for performance
        include_heatmap: Whether to add density heatmap layer
        output_path: Path to save HTML (None to skip saving)
    
    Returns:
        Folium Map object with cluster visualization
    """
    if 'cluster' not in df.columns:
        raise ValueError("DataFrame must have 'cluster' column. Run clustering first.")
    
    # Get cluster statistics
    cluster_counts = df['cluster'].value_counts()
    unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
    valid_clusters = [c for c in unique_clusters if cluster_counts.get(c, 0) >= min_cluster_size]
    
    n_total = len(df)
    n_noise = (df['cluster'] == -1).sum()
    n_clustered = n_total - n_noise
    
    print("=" * 60)
    print("CLUSTER MAP GENERATION")
    print("=" * 60)
    print(f"Total points: {n_total:,}")
    print(f"Clustered: {n_clustered:,} ({n_clustered/n_total*100:.1f}%)")
    print(f"Noise: {n_noise:,} ({n_noise/n_total*100:.1f}%)")
    print(f"Total clusters: {len(unique_clusters)}")
    print(f"Clusters >= {min_cluster_size} points: {len(valid_clusters)}")
    print()
    
    # Create base map
    m = create_base_map()
    
    # Load cluster descriptors
    cluster_descriptors = load_cluster_descriptors()
    if cluster_descriptors:
        print(f"  Loaded descriptors for {len(cluster_descriptors)} clusters")
    
    # Add heatmap if requested
    if include_heatmap:
        clustered_df = df[df['cluster'] != -1]
        m = add_heatmap(m, clustered_df, name="Cluster Density")
        print("  Added heatmap layer")
    
    # Add cluster markers
    m = add_cluster_markers(
        m, df, 
        min_cluster_size=min_cluster_size,
        show_noise=show_noise,
        sample_per_cluster=sample_per_cluster,
        cluster_descriptors=cluster_descriptors
    )
    print(f"  Added {len(valid_clusters)} cluster layers")
    
    # Generate legend info
    cluster_colors = generate_cluster_colors(len(valid_clusters))
    color_map = {c: cluster_colors[i % len(cluster_colors)] for i, c in enumerate(valid_clusters)}
    cluster_info = {c: (color_map[c], cluster_counts[c]) for c in valid_clusters}
    
    # Add legend with descriptors
    m = add_cluster_legend(m, cluster_info, cluster_descriptors)
    print("  Added legend with descriptors")
    
    # Add layer control
    m = add_layer_control(m)
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        print(f"\n✅ Saved to: {output_path}")
    
    return m


def main():
    """Generate the default map and save to app directory."""
    print("=" * 50)
    print("Generating Photo Location Map")
    print("=" * 50)
    
    m = create_photo_map(
        sample_size=2000,
        include_heatmap=True,
        include_markers=True,
        output_path=DEFAULT_MAP_PATH
    )
    
    print("\n" + "=" * 50)
    print(f"✅ Map generated: {DEFAULT_MAP_PATH}")
    print("Open in browser to view")
    print("=" * 50)
    
    return m


if __name__ == "__main__":
    main()
