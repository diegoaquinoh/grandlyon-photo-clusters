"""
Map visualization module for the Grand Lyon Photo Clusters project.
Provides utilities for creating interactive Folium maps of photo locations.
"""

import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
from pathlib import Path
from typing import Optional

from .data_loader import load_cleaned_data, load_and_clean_data, LYON_BBOX, PROJECT_ROOT

# Output paths
APP_DIR = PROJECT_ROOT / "app"
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
    folium.LayerControl(collapsed=False).add_to(m)
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
