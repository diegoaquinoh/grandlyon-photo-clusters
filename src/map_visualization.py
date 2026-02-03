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


def load_cluster_names() -> dict:
    """
    Load cluster names from association rules output.
    
    Returns:
        Dictionary mapping cluster ID to name info dict with keys:
        - 'name': auto-generated cluster name
        - 'method': naming method used (association_rule, tfidf, frequent_itemset, fallback)
        - 'top_tfidf_terms': list of top TF-IDF terms
    """
    # Try cluster_names.json first (combined output)
    names_path = REPORTS_DIR / "cluster_names.json"
    if not names_path.exists():
        # Fall back to association_rules.json
        names_path = REPORTS_DIR / "association_rules.json"
    
    if not names_path.exists():
        print(f"  Warning: No cluster names found")
        return {}
    
    try:
        with open(names_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = {}
        clusters_data = data.get('clusters', {})
        
        for cluster_id_str, info in clusters_data.items():
            cluster_id = int(cluster_id_str)
            if isinstance(info, dict):
                result[cluster_id] = {
                    'name': info.get('name', f'Cluster {cluster_id}'),
                    'method': info.get('method', info.get('naming_method', 'unknown')),
                    'top_tfidf_terms': info.get('top_tfidf_terms', [])
                }
            else:
                result[cluster_id] = {
                    'name': f'Cluster {cluster_id}',
                    'method': 'fallback',
                    'top_tfidf_terms': []
                }
        
        return result
    except Exception as e:
        print(f"  Warning: Could not load cluster names: {e}")
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


def add_cluster_polygons(
    m: folium.Map,
    df: pd.DataFrame,
    min_cluster_size: int = 10,
    cluster_colors: Optional[List[str]] = None,
    cluster_descriptors: Optional[dict] = None,
    cluster_names: Optional[dict] = None
) -> folium.Map:
    """
    Add convex hull polygons for each cluster to the map.
    
    Much more efficient than individual markers - one polygon per cluster.
    
    Args:
        m: Folium Map object
        df: DataFrame with photo data (must have 'cluster' column)
        min_cluster_size: Minimum cluster size to display
        cluster_colors: Optional list of colors for clusters
        cluster_descriptors: Optional dict mapping cluster ID to list of top terms
        cluster_names: Optional dict mapping cluster ID to name info
    
    Returns:
        Map with cluster polygons added
    """
    from scipy.spatial import ConvexHull
    
    if 'cluster' not in df.columns:
        raise ValueError("DataFrame must have 'cluster' column")
    
    # Load descriptors and names if not provided
    if cluster_descriptors is None:
        cluster_descriptors = load_cluster_descriptors()
    
    if cluster_names is None:
        cluster_names = load_cluster_names()
    
    # Get cluster info
    cluster_counts = df['cluster'].value_counts()
    unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
    
    # Filter clusters by size
    valid_clusters = [c for c in unique_clusters if cluster_counts.get(c, 0) >= min_cluster_size]
    
    # Generate colors
    if cluster_colors is None:
        cluster_colors = generate_cluster_colors(len(valid_clusters))
    
    color_map = {c: cluster_colors[i % len(cluster_colors)] for i, c in enumerate(valid_clusters)}
    
    # Create a single feature group for all polygons
    polygon_group = folium.FeatureGroup(name="Cluster Areas")
    
    for cluster_id in valid_clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        cluster_size = len(cluster_df)
        
        # Get coordinates
        coords = cluster_df[['lat', 'long']].values
        
        # Need at least 3 points for a polygon
        if len(coords) < 3:
            continue
        
        # Compute convex hull
        try:
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]
            # Close the polygon by adding first point at end
            hull_points = np.vstack([hull_points, hull_points[0]])
            # Convert to list of [lat, lon] for folium
            polygon_coords = hull_points.tolist()
        except Exception:
            # If convex hull fails, skip this cluster
            continue
        
        # Get cluster name (loader converts to int keys)
        name_info = cluster_names.get(cluster_id, {})
        cluster_name = name_info.get('name', f'Cluster {cluster_id}')
        naming_method = name_info.get('method', 'unknown')
        
        # Get descriptors for this cluster
        top_terms = cluster_descriptors.get(cluster_id, [])
        descriptor_str = ", ".join(top_terms[:5]) if top_terms else "(no descriptors)"
        
        # Get date range
        if 'date_taken_year' in cluster_df.columns:
            years = cluster_df['date_taken_year'].dropna()
            if len(years) > 0:
                min_year = int(years.min())
                max_year = int(years.max())
                date_range = f"{min_year} - {max_year}" if min_year != max_year else str(min_year)
            else:
                date_range = "Unknown"
        else:
            date_range = "Unknown"
        
        # Calculate centroid for popup placement
        centroid_lat = np.mean(coords[:, 0])
        centroid_lon = np.mean(coords[:, 1])
        
        # Build popup content with cluster name prominently displayed
        import html
        safe_name = html.escape(cluster_name).replace('{', '&#123;').replace('}', '&#125;')
        safe_descriptor = html.escape(descriptor_str).replace('{', '&#123;').replace('}', '&#125;')
        
        popup_html = f"""
        <div style="min-width: 250px; font-family: Arial, sans-serif;">
            <div style="background-color: {color_map[cluster_id]}; color: white; padding: 8px; 
                        margin: -10px -10px 10px -10px; border-radius: 3px 3px 0 0;">
                <b style="font-size: 14px;">{safe_name}</b>
            </div>
            <div style="padding: 5px 0;">
                <b>Cluster ID:</b> {cluster_id}<br>
                <b>Photos:</b> {cluster_size:,}<br>
                <b>Date range:</b> {date_range}<br>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">
                <b>Top keywords:</b><br>
                <i style="color: #666;">{safe_descriptor}</i><br>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">
                <span style="font-size: 10px; color: #999;">Named via: {naming_method}</span>
            </div>
        </div>
        """
        
        # Get cluster name for tooltip
        tooltip_text = f"{cluster_name} ({cluster_size:,} photos)"
        
        # Create polygon
        folium.Polygon(
            locations=polygon_coords,
            color=color_map[cluster_id],
            weight=2,
            fill=True,
            fill_color=color_map[cluster_id],
            fill_opacity=0.4,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=tooltip_text
        ).add_to(polygon_group)
    
    polygon_group.add_to(m)
    return m


def add_cluster_markers(
    m: folium.Map,
    df: pd.DataFrame,
    min_cluster_size: int = 10,
    show_noise: bool = False,
    sample_per_cluster: int = 200,
    cluster_colors: Optional[List[str]] = None,
    cluster_descriptors: Optional[dict] = None,
    cluster_names: Optional[dict] = None
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
        cluster_names: Optional dict mapping cluster ID to name info
    
    Returns:
        Map with cluster markers added
    """
    if 'cluster' not in df.columns:
        raise ValueError("DataFrame must have 'cluster' column")
    
    # Load descriptors and names if not provided
    if cluster_descriptors is None:
        cluster_descriptors = load_cluster_descriptors()
    
    if cluster_names is None:
        cluster_names = load_cluster_names()
    
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
        
        # Get cluster name (loader converts to int keys)
        name_info = cluster_names.get(cluster_id, {})
        cluster_name = name_info.get('name', f'Cluster {cluster_id}')
        naming_method = name_info.get('method', 'unknown')
        
        # Get descriptors for this cluster
        top_terms = cluster_descriptors.get(cluster_id, [])
        descriptor_str = ", ".join(top_terms[:5]) if top_terms else "(no descriptors)"
        
        # Get date range for the cluster
        if 'date_taken_year' in cluster_df.columns:
            years = cluster_df['date_taken_year'].dropna()
            if len(years) > 0:
                min_year = int(years.min())
                max_year = int(years.max())
                date_range = f"{min_year} - {max_year}" if min_year != max_year else str(min_year)
            else:
                date_range = "Unknown"
        else:
            date_range = "Unknown"
        
        # Sample if too many points
        if len(cluster_df) > sample_per_cluster:
            cluster_df = cluster_df.sample(n=sample_per_cluster, random_state=42)
        
        # Use cluster name in group name
        import html
        safe_cluster_name = html.escape(cluster_name[:30])
        group_name = f"{safe_cluster_name} (n={cluster_size:,})"
        
        group = folium.FeatureGroup(name=group_name)
        
        for _, row in cluster_df.iterrows():
            # Build popup content with HTML escaping AND Jinja2-safe escaping
            raw_tags = str(row.get('tags', ''))[:80]
            if len(str(row.get('tags', ''))) > 80:
                raw_tags += '...'
            # Escape HTML and replace curly braces (Jinja2 interprets { and })
            tags = html.escape(raw_tags).replace('{', '&#123;').replace('}', '&#125;')
            title = html.escape(str(row.get('title', 'No title'))[:50]).replace('{', '&#123;').replace('}', '&#125;')
            year = int(row.get('date_taken_year', 0)) if pd.notna(row.get('date_taken_year')) else 'N/A'
            safe_descriptor = html.escape(descriptor_str).replace('{', '&#123;').replace('}', '&#125;')
            safe_name = html.escape(cluster_name).replace('{', '&#123;').replace('}', '&#125;')
            
            popup_html = f"""
            <div style="min-width: 260px; font-family: Arial, sans-serif;">
                <div style="background-color: {color_map[cluster_id]}; color: white; padding: 8px; 
                            margin: -10px -10px 10px -10px; border-radius: 3px 3px 0 0;">
                    <b style="font-size: 13px;">{safe_name}</b>
                </div>
                <b>Cluster ID:</b> {cluster_id} &nbsp;|&nbsp; <b>Size:</b> {cluster_size:,}<br>
                <b>Date range:</b> {date_range}<br>
                <hr style="margin: 6px 0; border: none; border-top: 1px solid #ddd;">
                <b>Top keywords:</b> <i style="color: #666;">{safe_descriptor}</i>
                <hr style="margin: 6px 0; border: none; border-top: 1px solid #ddd;">
                <b>This photo:</b><br>
                <b>Title:</b> {title}<br>
                <b>Year:</b> {year}<br>
                <b>Tags:</b> {tags}<br>
                <hr style="margin: 6px 0; border: none; border-top: 1px solid #ddd;">
                <span style="font-size: 10px; color: #999;">Named via: {naming_method}</span>
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
    cluster_descriptors: Optional[dict] = None,
    cluster_names: Optional[dict] = None
) -> folium.Map:
    """
    Add a legend showing cluster colors, sizes, names, and descriptors.
    
    Args:
        m: Folium Map object
        cluster_info: Dict of {cluster_id: (color, size)}
        cluster_descriptors: Optional dict mapping cluster ID to list of top terms
        cluster_names: Optional dict mapping cluster ID to auto-generated name
    
    Returns:
        Map with legend added
    """
    if cluster_descriptors is None:
        cluster_descriptors = {}
    if cluster_names is None:
        cluster_names = {}
    
    # Build legend HTML
    legend_items = ""
    for cluster_id, (color, size) in sorted(cluster_info.items(), key=lambda x: -x[1][1])[:15]:  # Top 15 by size
        # Get cluster name if available (cluster_names has integer keys)
        name_info = cluster_names.get(cluster_id, {})
        cluster_name = name_info.get('name', '') if isinstance(name_info, dict) else ''
        
        # Use cluster name if available, otherwise show Cluster ID
        if cluster_name and not cluster_name.startswith('Cluster '):
            name_html = f'<b>{cluster_name}</b>'
        else:
            name_html = f'Cluster {cluster_id}'
        
        # Get short descriptor (show if no meaningful name)
        terms = cluster_descriptors.get(cluster_id, [])
        desc = ", ".join(terms[:2]) if terms else ""
        desc_html = f'<span style="color: #666; font-style: italic;"> ({desc})</span>' if desc and cluster_name.startswith('Cluster ') else ""
        
        legend_items += f"""
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <span style="background-color: {color}; width: 12px; height: 12px; 
                         border-radius: 50%; display: inline-block; margin-right: 6px; flex-shrink: 0;"></span>
            <span style="font-size: 11px;">{name_html} ({size:,}){desc_html}</span>
        </div>
        """
    
    if len(cluster_info) > 15:
        legend_items += f'<div style="font-size: 10px; color: #666; margin-top: 5px;">...and {len(cluster_info) - 15} more clusters</div>'
    
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 12px; border-radius: 8px;
                border: 2px solid #ccc; font-family: Arial, sans-serif; max-width: 380px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
        <div style="font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid #ccc; 
                    padding-bottom: 5px; font-size: 13px;">
            📍 Top Clusters by Size
        </div>
        {legend_items}
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def add_cluster_summary_panel(
    m: folium.Map,
    df: pd.DataFrame,
    cluster_names: Optional[dict] = None,
    cluster_descriptors: Optional[dict] = None
) -> folium.Map:
    """
    Add a collapsible summary panel with search functionality.
    
    Args:
        m: Folium Map object
        df: DataFrame with cluster data
        cluster_names: Dict mapping cluster ID to name info
        cluster_descriptors: Dict mapping cluster ID to terms
    
    Returns:
        Map with summary panel added
    """
    if cluster_names is None:
        cluster_names = load_cluster_names()
    if cluster_descriptors is None:
        cluster_descriptors = load_cluster_descriptors()
    
    # Get cluster statistics
    cluster_stats = df[df['cluster'] != -1].groupby('cluster').agg({
        'lat': 'mean',
        'long': 'mean',
        'date_taken_year': ['min', 'max', 'count']
    }).reset_index()
    cluster_stats.columns = ['cluster', 'lat', 'long', 'year_min', 'year_max', 'count']
    cluster_stats = cluster_stats.sort_values('count', ascending=False)
    
    # Build cluster list HTML
    cluster_items = []
    for _, row in cluster_stats.iterrows():
        cid = int(row['cluster'])
        name_info = cluster_names.get(cid, {})  # Use integer key
        name = name_info.get('name', f'Cluster {cid}') if isinstance(name_info, dict) else f'Cluster {cid}'
        method = name_info.get('method', 'unknown') if isinstance(name_info, dict) else 'unknown'
        count = int(row['count'])
        lat, lng = row['lat'], row['long']
        year_min = int(row['year_min']) if pd.notna(row['year_min']) else '?'
        year_max = int(row['year_max']) if pd.notna(row['year_max']) else '?'
        
        # Get descriptors
        terms = cluster_descriptors.get(cid, [])[:3]
        terms_str = ', '.join(terms) if terms else ''
        
        cluster_items.append(f'''
            <div class="cluster-item" data-name="{name.lower()}" data-terms="{terms_str.lower()}" 
                 onclick="map.flyTo([{lat}, {lng}], 14)">
                <div class="cluster-name">{name}</div>
                <div class="cluster-meta">
                    <span class="cluster-size">{count:,} photos</span>
                    <span class="cluster-years">{year_min}-{year_max}</span>
                </div>
                <div class="cluster-terms">{terms_str}</div>
                <div class="cluster-method">{method}</div>
            </div>
        ''')
    
    cluster_list_html = ''.join(cluster_items)
    
    # Add panel HTML and CSS
    panel_html = f'''
    <style>
        #cluster-panel {{
            position: fixed;
            top: 10px;
            right: 10px;
            width: 320px;
            max-height: 90vh;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            font-family: Arial, sans-serif;
            z-index: 1000;
            display: flex;
            flex-direction: column;
        }}
        #panel-header {{
            padding: 12px;
            background: #2c3e50;
            color: white;
            border-radius: 8px 8px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        #panel-header h3 {{
            margin: 0;
            font-size: 14px;
        }}
        #toggle-btn {{
            background: none;
            border: none;
            color: white;
            font-size: 18px;
            cursor: pointer;
        }}
        #search-box {{
            padding: 10px;
            border-bottom: 1px solid #eee;
        }}
        #search-input {{
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 13px;
            box-sizing: border-box;
        }}
        #cluster-list {{
            overflow-y: auto;
            max-height: 60vh;
            padding: 5px;
        }}
        .cluster-item {{
            padding: 10px;
            margin: 5px;
            background: #f8f9fa;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .cluster-item:hover {{
            background: #e9ecef;
        }}
        .cluster-name {{
            font-weight: bold;
            font-size: 13px;
            color: #2c3e50;
            margin-bottom: 4px;
        }}
        .cluster-meta {{
            font-size: 11px;
            color: #666;
            display: flex;
            gap: 10px;
        }}
        .cluster-terms {{
            font-size: 11px;
            color: #888;
            font-style: italic;
            margin-top: 4px;
        }}
        .cluster-method {{
            font-size: 10px;
            color: #aaa;
            margin-top: 2px;
        }}
        #panel-stats {{
            padding: 8px 12px;
            background: #f8f9fa;
            border-top: 1px solid #eee;
            font-size: 11px;
            color: #666;
            border-radius: 0 0 8px 8px;
        }}
        .hidden {{
            display: none !important;
        }}
        #panel-content.collapsed {{
            display: none;
        }}
    </style>
    
    <div id="cluster-panel">
        <div id="panel-header">
            <h3>📍 Cluster Explorer ({len(cluster_stats)} clusters)</h3>
            <button id="toggle-btn" onclick="togglePanel()">−</button>
        </div>
        <div id="panel-content">
            <div id="search-box">
                <input type="text" id="search-input" placeholder="Search clusters by name..." 
                       oninput="filterClusters(this.value)">
            </div>
            <div id="cluster-list">
                {cluster_list_html}
            </div>
            <div id="panel-stats">
                Total: {len(df):,} photos | Clustered: {len(df[df['cluster'] != -1]):,}
            </div>
        </div>
    </div>
    
    <script>
        function togglePanel() {{
            var content = document.getElementById('panel-content');
            var btn = document.getElementById('toggle-btn');
            if (content.classList.contains('collapsed')) {{
                content.classList.remove('collapsed');
                btn.textContent = '−';
            }} else {{
                content.classList.add('collapsed');
                btn.textContent = '+';
            }}
        }}
        
        function filterClusters(query) {{
            query = query.toLowerCase();
            var items = document.querySelectorAll('.cluster-item');
            var visible = 0;
            items.forEach(function(item) {{
                var name = item.getAttribute('data-name');
                var terms = item.getAttribute('data-terms');
                if (name.includes(query) || terms.includes(query)) {{
                    item.classList.remove('hidden');
                    visible++;
                }} else {{
                    item.classList.add('hidden');
                }}
            }});
        }}
    </script>
    '''
    
    m.get_root().html.add_child(folium.Element(panel_html))
    return m


def create_cluster_map(
    df: pd.DataFrame,
    min_cluster_size: int = 10,
    show_noise: bool = False,
    sample_per_cluster: int = 200,
    include_heatmap: bool = False,
    use_polygons: bool = True,
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
        use_polygons: If True, show cluster areas as polygons; if False, show individual points
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
    print(f"Visualization mode: {'polygons' if use_polygons else 'markers'}")
    print()
    
    # Create base map
    m = create_base_map()
    
    # Load cluster descriptors
    cluster_descriptors = load_cluster_descriptors()
    if cluster_descriptors:
        print(f"  Loaded descriptors for {len(cluster_descriptors)} clusters")
    
    # Load cluster names (from association rules)
    cluster_names = load_cluster_names()
    if cluster_names:
        print(f"  Loaded auto-generated names for {len(cluster_names)} clusters")
    
    # Add heatmap if requested
    if include_heatmap:
        clustered_df = df[df['cluster'] != -1]
        m = add_heatmap(m, clustered_df, name="Cluster Density")
        print("  Added heatmap layer")
    
    # Add cluster visualization (polygons or markers)
    if use_polygons:
        m = add_cluster_polygons(
            m, df,
            min_cluster_size=min_cluster_size,
            cluster_descriptors=cluster_descriptors,
            cluster_names=cluster_names
        )
        print(f"  Added {len(valid_clusters)} cluster polygons")
    else:
        m = add_cluster_markers(
            m, df, 
            min_cluster_size=min_cluster_size,
            show_noise=show_noise,
            sample_per_cluster=sample_per_cluster,
            cluster_descriptors=cluster_descriptors,
            cluster_names=cluster_names
        )
        print(f"  Added {len(valid_clusters)} cluster layers")
    
    # Generate legend info
    cluster_colors = generate_cluster_colors(len(valid_clusters))
    color_map = {c: cluster_colors[i % len(cluster_colors)] for i, c in enumerate(valid_clusters)}
    cluster_info = {c: (color_map[c], cluster_counts[c]) for c in valid_clusters}
    
    # Add legend with descriptors and names
    m = add_cluster_legend(m, cluster_info, cluster_descriptors, cluster_names)
    print("  Added legend with cluster names")
    
    # Add cluster summary panel with search
    m = add_cluster_summary_panel(m, df, cluster_names, cluster_descriptors)
    print("  Added cluster explorer panel with search")
    
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
