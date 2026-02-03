#!/usr/bin/env python3
"""
Full Pipeline Script for Grand Lyon Photo Clusters - Session 3.

Runs the complete pipeline including all Session 3 deliverables:
1. Data Cleaning
2. Clustering (HDBSCAN)
3. Text Mining (TF-IDF + Association Rules)
4. Temporal Analysis
5. Enhanced Map Generation with filters

Usage:
    python scripts/run_full_pipeline.py [OPTIONS]

Options:
    --skip-if-exists    Skip steps if output files already exist
    --quick             Quick run with reduced sample size
    --map-only          Only regenerate the map (skip all processing)
"""

import sys
import argparse
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_and_clean_data, load_cleaned_data, 
    CLEANED_DATA_PATH, DATA_DIR, REPORTS_DIR
)
from src.clustering import (
    prepare_coordinates, run_hdbscan, run_dbscan, run_kmeans, 
    run_hierarchical, get_cluster_stats
)
from src.text_mining import run_text_mining, run_association_rules_mining
from src.temporal_analysis import run_temporal_analysis, classify_all_clusters
from src.map_visualization import APP_DIR

import pandas as pd

# Output paths
CLUSTERED_DATA_PATH = DATA_DIR / "flickr_clustered.csv"
CLUSTERING_CACHE_META_PATH = DATA_DIR / "clustering_cache_meta.json"
CLUSTER_MAP_PATH = APP_DIR / "cluster_map_v2.html"
TEMPORAL_CLASSIFICATIONS_PATH = REPORTS_DIR / "temporal_classifications.json"


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step_num: int, total: int, description: str):
    """Print a step indicator."""
    print(f"\n[{step_num}/{total}] {description}")
    print("-" * 50)


def create_enhanced_cluster_map(
    df: pd.DataFrame,
    temporal_classifications: dict,
    output_path: Path = CLUSTER_MAP_PATH,
    min_cluster_size: int = 1  # Show ALL clusters
):
    """
    Create an enhanced cluster map with cluster type filter and temporal filtering.
    
    Args:
        df: DataFrame with clustered photo data
        temporal_classifications: Dict mapping cluster ID to type info
        output_path: Path to save the HTML map
        min_cluster_size: Minimum cluster size to display (1 = all clusters)
    """
    import folium
    from folium.plugins import HeatMap
    import numpy as np
    from scipy.spatial import ConvexHull
    import html as html_escape
    
    from src.map_visualization import (
        create_base_map, load_cluster_descriptors, load_cluster_names,
        generate_cluster_colors, add_layer_control
    )
    
    print("Creating enhanced cluster map with filters...")
    
    # Filter noise
    df_clustered = df[df['cluster'] != -1].copy()
    
    # Get cluster info
    cluster_counts = df_clustered['cluster'].value_counts()
    unique_clusters = sorted([c for c in df_clustered['cluster'].unique()])
    valid_clusters = [c for c in unique_clusters if cluster_counts.get(c, 0) >= min_cluster_size]
    
    print(f"  Total clusters: {len(valid_clusters)}")
    
    # Load descriptors and names
    cluster_descriptors = load_cluster_descriptors()
    cluster_names = load_cluster_names()
    
    # Map clusters to types
    cluster_types = {}
    for cluster_id in valid_clusters:
        if cluster_id in temporal_classifications:
            cluster_types[cluster_id] = temporal_classifications[cluster_id]['type']
        else:
            cluster_types[cluster_id] = 'unknown'
    
    # Type colors
    type_colors = {
        'permanent': '#2E86AB',  # Blue
        'recurring': '#A23B72',  # Purple
        'one_time': '#F18F01',   # Orange
        'unknown': '#888888'     # Gray
    }
    
    # Create base map
    m = create_base_map()
    
    # Create feature groups for each type
    type_groups = {
        'permanent': folium.FeatureGroup(name="🏛️ Permanent POIs", show=True),
        'recurring': folium.FeatureGroup(name="🔄 Recurring Events", show=True),
        'one_time': folium.FeatureGroup(name="📅 One-time Events", show=True),
        'unknown': folium.FeatureGroup(name="❓ Unknown", show=False)
    }
    
    # Add polygons for each cluster
    for cluster_id in valid_clusters:
        cluster_df = df_clustered[df_clustered['cluster'] == cluster_id]
        cluster_size = len(cluster_df)
        
        # Get coordinates
        coords = cluster_df[['lat', 'long']].values
        
        if len(coords) < 3:
            continue
        
        # Compute convex hull
        try:
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])
            polygon_coords = hull_points.tolist()
        except Exception:
            continue
        
        # Get cluster metadata
        cluster_type = cluster_types.get(cluster_id, 'unknown')
        color = type_colors.get(cluster_type, '#888888')
        
        name_info = cluster_names.get(cluster_id, {})
        cluster_name = name_info.get('name', f'Cluster {cluster_id}') if isinstance(name_info, dict) else f'Cluster {cluster_id}'
        
        top_terms = cluster_descriptors.get(cluster_id, [])
        descriptor_str = ", ".join(top_terms[:5]) if top_terms else "(no descriptors)"
        
        # Get year range
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
        
        # Get temporal stats if available
        temporal_info = temporal_classifications.get(cluster_id, {})
        stats = temporal_info.get('stats', {})
        december_ratio = stats.get('december_ratio', 0) * 100
        summer_ratio = stats.get('summer_ratio', 0) * 100
        peak_month = int(stats.get('peak_month', 0))
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        peak_month_name = month_names[peak_month - 1] if 1 <= peak_month <= 12 else 'N/A'
        
        # Type badge
        type_emoji = {'permanent': '🏛️', 'recurring': '🔄', 'one_time': '📅', 'unknown': '❓'}
        type_label = {'permanent': 'Permanent POI', 'recurring': 'Recurring Event', 
                      'one_time': 'One-time Event', 'unknown': 'Unknown'}
        
        # Escape HTML
        safe_name = html_escape.escape(cluster_name)
        safe_descriptor = html_escape.escape(descriptor_str)
        
        popup_html = f"""
        <div style="min-width: 280px; font-family: Arial, sans-serif;">
            <div style="background-color: {color}; color: white; padding: 10px; 
                        margin: -10px -10px 10px -10px; border-radius: 3px 3px 0 0;">
                <b style="font-size: 14px;">{safe_name}</b>
                <div style="font-size: 11px; margin-top: 4px;">{type_emoji.get(cluster_type, '')} {type_label.get(cluster_type, 'Unknown')}</div>
            </div>
            <div style="padding: 5px 0;">
                <b>Cluster ID:</b> {cluster_id}<br>
                <b>Photos:</b> {cluster_size:,}<br>
                <b>Date range:</b> {date_range}<br>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">
                <b>Top keywords:</b><br>
                <i style="color: #666;">{safe_descriptor}</i><br>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">
                <b>Temporal Pattern:</b><br>
                Peak month: {peak_month_name}<br>
                December activity: {december_ratio:.1f}%<br>
                Summer activity: {summer_ratio:.1f}%<br>
            </div>
        </div>
        """
        
        tooltip_text = f"{cluster_name} ({cluster_size:,} photos) - {type_label.get(cluster_type, 'Unknown')}"
        
        # Create polygon and add to appropriate group
        polygon = folium.Polygon(
            locations=polygon_coords,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.4,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=tooltip_text
        )
        polygon.add_to(type_groups[cluster_type])
    
    # Add all groups to map
    for group in type_groups.values():
        group.add_to(m)
    
    # Add enhanced panel with filters
    panel_html = create_enhanced_panel(
        df_clustered, valid_clusters, cluster_types, 
        cluster_names, cluster_descriptors, temporal_classifications
    )
    m.get_root().html.add_child(folium.Element(panel_html))
    
    # Add type legend
    legend_html = create_type_legend(cluster_types, type_colors)
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    m = add_layer_control(m)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    
    print(f"  ✅ Enhanced map saved to: {output_path}")
    
    return m


def create_enhanced_panel(
    df: pd.DataFrame,
    valid_clusters: list,
    cluster_types: dict,
    cluster_names: dict,
    cluster_descriptors: dict,
    temporal_classifications: dict
) -> str:
    """Create the HTML for the enhanced cluster explorer panel."""
    
    # Get cluster statistics
    cluster_stats = df.groupby('cluster').agg({
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
        if cid not in valid_clusters:
            continue
        
        name_info = cluster_names.get(cid, {})
        name = name_info.get('name', f'Cluster {cid}') if isinstance(name_info, dict) else f'Cluster {cid}'
        count = int(row['count'])
        lat, lng = row['lat'], row['long']
        year_min = int(row['year_min']) if pd.notna(row['year_min']) else '?'
        year_max = int(row['year_max']) if pd.notna(row['year_max']) else '?'
        
        # Type info
        cluster_type = cluster_types.get(cid, 'unknown')
        type_emoji = {'permanent': '🏛️', 'recurring': '🔄', 'one_time': '📅', 'unknown': '❓'}
        
        # Get descriptors
        terms = cluster_descriptors.get(cid, [])[:3]
        terms_str = ', '.join(terms) if terms else ''
        
        cluster_items.append(f'''
            <div class="cluster-item" data-name="{name.lower()}" data-terms="{terms_str.lower()}" 
                 data-type="{cluster_type}" data-year-min="{year_min}" data-year-max="{year_max}"
                 onclick="map.flyTo([{lat}, {lng}], 14)">
                <div class="cluster-header">
                    <span class="cluster-emoji">{type_emoji.get(cluster_type, '❓')}</span>
                    <span class="cluster-name">{name}</span>
                </div>
                <div class="cluster-meta">
                    <span class="cluster-size">{count:,} photos</span>
                    <span class="cluster-years">{year_min}-{year_max}</span>
                </div>
                <div class="cluster-terms">{terms_str}</div>
            </div>
        ''')
    
    cluster_list_html = ''.join(cluster_items)
    
    # Count by type
    type_counts = {'permanent': 0, 'recurring': 0, 'one_time': 0, 'unknown': 0}
    for cid in valid_clusters:
        t = cluster_types.get(cid, 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
    
    return f'''
    <style>
        #cluster-panel {{
            position: fixed;
            top: 10px;
            right: 10px;
            width: 340px;
            max-height: 90vh;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        #panel-header {{
            padding: 14px;
            background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%);
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        #panel-header h3 {{
            margin: 0;
            font-size: 15px;
            font-weight: 600;
        }}
        #toggle-btn {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            font-size: 16px;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        #panel-content {{
            display: flex;
            flex-direction: column;
            flex: 1;
            overflow: hidden;
        }}
        #filter-section {{
            padding: 12px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        #search-input {{
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 13px;
            box-sizing: border-box;
            margin-bottom: 10px;
        }}
        #type-filters {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }}
        .type-filter {{
            padding: 6px 10px;
            border-radius: 20px;
            border: 1px solid #ddd;
            background: white;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .type-filter.active {{
            background: #2c3e50;
            color: white;
            border-color: #2c3e50;
        }}
        .type-filter:hover {{
            border-color: #2c3e50;
        }}
        #year-filter {{
            margin-top: 10px;
            font-size: 12px;
        }}
        #year-filter label {{
            display: block;
            margin-bottom: 5px;
            color: #666;
        }}
        #year-range {{
            width: 100%;
        }}
        #year-display {{
            text-align: center;
            font-weight: 500;
            color: #2c3e50;
        }}
        #cluster-list {{
            overflow-y: auto;
            flex: 1;
            padding: 8px;
        }}
        .cluster-item {{
            padding: 12px;
            margin: 6px 0;
            background: #f8f9fa;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
        }}
        .cluster-item:hover {{
            background: #e9ecef;
            border-color: #dee2e6;
            transform: translateX(2px);
        }}
        .cluster-item.hidden {{
            display: none;
        }}
        .cluster-header {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .cluster-emoji {{
            font-size: 14px;
        }}
        .cluster-name {{
            font-weight: 600;
            font-size: 13px;
            color: #2c3e50;
        }}
        .cluster-meta {{
            font-size: 11px;
            color: #666;
            display: flex;
            gap: 10px;
            margin-top: 4px;
        }}
        .cluster-terms {{
            font-size: 11px;
            color: #888;
            font-style: italic;
            margin-top: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        #panel-stats {{
            padding: 10px 14px;
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
            font-size: 12px;
            color: #666;
        }}
    </style>
    
    <div id="cluster-panel">
        <div id="panel-header">
            <h3>🗺️ Cluster Explorer</h3>
            <button id="toggle-btn" onclick="togglePanel()">−</button>
        </div>
        <div id="panel-content">
            <div id="filter-section">
                <input type="text" id="search-input" placeholder="🔍 Search clusters by name or keyword...">
                <div id="type-filters">
                    <button class="type-filter active" data-type="all">All ({len(valid_clusters)})</button>
                    <button class="type-filter" data-type="permanent">🏛️ POI ({type_counts['permanent']})</button>
                    <button class="type-filter" data-type="recurring">🔄 Recurring ({type_counts['recurring']})</button>
                    <button class="type-filter" data-type="one_time">📅 One-time ({type_counts['one_time']})</button>
                </div>
                <div id="year-filter">
                    <label>Filter by year: <span id="year-display">All years</span></label>
                    <input type="range" id="year-range" min="2010" max="2025" value="2025" step="1">
                </div>
            </div>
            <div id="cluster-list">
                {cluster_list_html}
            </div>
            <div id="panel-stats">
                Showing <span id="visible-count">{len(valid_clusters)}</span> of {len(valid_clusters)} clusters
            </div>
        </div>
    </div>
    
    <script>
        // Panel toggle
        function togglePanel() {{
            var content = document.getElementById('panel-content');
            var btn = document.getElementById('toggle-btn');
            if (content.style.display === 'none') {{
                content.style.display = 'flex';
                btn.textContent = '−';
            }} else {{
                content.style.display = 'none';
                btn.textContent = '+';
            }}
        }}
        
        // Search functionality
        document.getElementById('search-input').addEventListener('input', filterClusters);
        
        // Type filter buttons
        document.querySelectorAll('.type-filter').forEach(btn => {{
            btn.addEventListener('click', function() {{
                document.querySelectorAll('.type-filter').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                filterClusters();
            }});
        }});
        
        // Year filter
        document.getElementById('year-range').addEventListener('input', function() {{
            var year = this.value;
            document.getElementById('year-display').textContent = year == 2025 ? 'All years' : '≤ ' + year;
            filterClusters();
        }});
        
        function filterClusters() {{
            var searchTerm = document.getElementById('search-input').value.toLowerCase();
            var activeType = document.querySelector('.type-filter.active').dataset.type;
            var maxYear = parseInt(document.getElementById('year-range').value);
            
            var items = document.querySelectorAll('.cluster-item');
            var visibleCount = 0;
            
            items.forEach(function(item) {{
                var name = item.dataset.name || '';
                var terms = item.dataset.terms || '';
                var type = item.dataset.type;
                var yearMin = parseInt(item.dataset.yearMin) || 2010;
                
                var matchesSearch = name.includes(searchTerm) || terms.includes(searchTerm);
                var matchesType = (activeType === 'all') || (type === activeType);
                var matchesYear = (maxYear === 2025) || (yearMin <= maxYear);
                
                if (matchesSearch && matchesType && matchesYear) {{
                    item.classList.remove('hidden');
                    visibleCount++;
                }} else {{
                    item.classList.add('hidden');
                }}
            }});
            
            document.getElementById('visible-count').textContent = visibleCount;
        }}
    </script>
    '''


def create_type_legend(cluster_types: dict, type_colors: dict) -> str:
    """Create HTML for the type legend."""
    
    # Count by type
    type_counts = {'permanent': 0, 'recurring': 0, 'one_time': 0}
    for cid, t in cluster_types.items():
        if t in type_counts:
            type_counts[t] += 1
    
    return f'''
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 14px 18px; border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="font-weight: 600; margin-bottom: 10px; font-size: 13px; color: #2c3e50;">
            📊 Cluster Types
        </div>
        <div style="display: flex; flex-direction: column; gap: 6px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: {type_colors['permanent']}; width: 14px; height: 14px; 
                             border-radius: 4px; display: inline-block;"></span>
                <span style="font-size: 12px;">🏛️ Permanent POI ({type_counts['permanent']})</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: {type_colors['recurring']}; width: 14px; height: 14px; 
                             border-radius: 4px; display: inline-block;"></span>
                <span style="font-size: 12px;">🔄 Recurring Event ({type_counts['recurring']})</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: {type_colors['one_time']}; width: 14px; height: 14px; 
                             border-radius: 4px; display: inline-block;"></span>
                <span style="font-size: 12px;">📅 One-time Event ({type_counts['one_time']})</span>
            </div>
        </div>
    </div>
    '''


def run_full_pipeline(
    skip_if_exists: bool = False,
    quick: bool = False,
    map_only: bool = False,
    skip_rules: bool = False,
    algorithm: str = 'hdbscan',
    algo_params: dict = None
):
    """
    Run the complete Grand Lyon Photo Clusters pipeline with Session 3 enhancements.
    """
    start_time = time.time()
    
    print_header("GRAND LYON PHOTO CLUSTERS - FULL PIPELINE (Session 3)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Options: skip_if_exists={skip_if_exists}, quick={quick}, map_only={map_only}, skip_rules={skip_rules}")
    print(f"Algorithm: {algorithm} with params: {algo_params or 'defaults'}")
    
    total_steps = 1 if map_only else 5
    
    # Load data
    if map_only:
        print_step(1, 1, "MAP GENERATION (from existing data)")
        df = pd.read_csv(CLUSTERED_DATA_PATH)
    else:
        # =========================================================================
        # STAGE 1: DATA CLEANING
        # =========================================================================
        print_step(1, total_steps, "DATA CLEANING")
        
        if skip_if_exists and CLEANED_DATA_PATH.exists():
            print(f"Loading cached cleaned data: {CLEANED_DATA_PATH}")
            df = pd.read_parquet(CLEANED_DATA_PATH)
        else:
            df = load_and_clean_data(
                filter_bbox=True,
                bbox_type="large",
                save_cache=True,
                save_log=True,
                verbose=True
            )
        
        if quick:
            print(f"Quick mode: sampling 20,000 rows...")
            df = df.sample(n=min(20000, len(df)), random_state=42)
        
        print(f"✅ {len(df):,} photos ready")
        
        # =========================================================================
        # STAGE 2: CLUSTERING
        # =========================================================================
        print_step(2, total_steps, f"CLUSTERING ({algorithm.upper()})")
        
        # Build cache key from algorithm + params to detect stale cache
        params = algo_params or {}
        cache_meta = {
            'algorithm': algorithm,
            'params': {
                'min_cluster_size': params.get('min_cluster_size', 120),
                'min_samples': params.get('min_samples'),
                'eps': params.get('eps', 0.005),
                'n_clusters': params.get('n_clusters', 50),
            }
        }
        cache_key = hashlib.md5(json.dumps(cache_meta, sort_keys=True).encode()).hexdigest()[:8]
        
        # Check if cache matches current parameters
        cache_valid = False
        if skip_if_exists and CLUSTERED_DATA_PATH.exists() and CLUSTERING_CACHE_META_PATH.exists():
            try:
                with open(CLUSTERING_CACHE_META_PATH, 'r') as f:
                    saved_meta = json.load(f)
                if saved_meta.get('cache_key') == cache_key:
                    cache_valid = True
                    print(f"✓ Cache valid (key: {cache_key}, algorithm: {algorithm})")
                else:
                    print(f"⚠️  Cache stale (saved: {saved_meta.get('cache_key')}, current: {cache_key})")
                    print(f"   Saved params: {saved_meta.get('algorithm')} {saved_meta.get('params')}")
                    print(f"   Current params: {algorithm} {cache_meta['params']}")
            except (json.JSONDecodeError, KeyError):
                print("⚠️  Cache metadata corrupted, will regenerate")
        
        if cache_valid:
            print(f"Loading cached clustered data: {CLUSTERED_DATA_PATH}")
            df = pd.read_csv(CLUSTERED_DATA_PATH)
            if quick:
                df = df.sample(n=min(20000, len(df)), random_state=42)
        else:
            # Prepare coordinates (scale for kmeans/hierarchical)
            scale = algorithm in ['kmeans', 'hierarchical']
            coords = prepare_coordinates(df, scale=scale)
            
            # Run selected clustering algorithm
            if algorithm == 'hdbscan':
                labels = run_hdbscan(
                    coords,
                    min_cluster_size=params.get('min_cluster_size', 120),
                    min_samples=params.get('min_samples', None)
                )
            elif algorithm == 'dbscan':
                labels = run_dbscan(
                    coords,
                    eps=params.get('eps', 0.005),
                    min_samples=params.get('min_samples', 10)
                )
            elif algorithm == 'kmeans':
                labels = run_kmeans(
                    coords,
                    n_clusters=params.get('n_clusters', 50)
                )
            elif algorithm == 'hierarchical':
                labels = run_hierarchical(
                    coords,
                    n_clusters=params.get('n_clusters', 50),
                    linkage=params.get('linkage', 'ward')
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            df['cluster'] = labels
            
            stats = get_cluster_stats(labels)
            print(f"  Clusters: {stats['n_clusters']}")
            print(f"  Noise: {stats['n_noise']:,} ({stats['noise_percentage']:.1f}%)")
            
            # Save clustered data AND cache metadata
            df.to_csv(CLUSTERED_DATA_PATH, index=False)
            cache_meta['cache_key'] = cache_key
            cache_meta['created_at'] = datetime.now().isoformat()
            with open(CLUSTERING_CACHE_META_PATH, 'w') as f:
                json.dump(cache_meta, f, indent=2)
            print(f"  Cache saved with key: {cache_key}")
        
        print(f"✅ {df['cluster'].nunique()} clusters")
        
        # =========================================================================
        # STAGE 3: TEXT MINING
        # =========================================================================
        if skip_rules:
            print_step(3, total_steps, "TEXT MINING (TF-IDF only)")
        else:
            print_step(3, total_steps, "TEXT MINING (TF-IDF + Association Rules)")
        
        # TF-IDF
        tfidf_descriptors = run_text_mining(df=df, top_n=10, save_results=True)
        
        # Association Rules (skip if --skip-rules flag is set)
        if not skip_rules:
            run_association_rules_mining(
                df=df, 
                save_results=True,
                tfidf_descriptors=tfidf_descriptors  # Pass TF-IDF to avoid recomputation
            )
        else:
            print("⏭️  Skipping association rules mining (--skip-rules)")
        
        print("✅ Text mining complete")
        
        # =========================================================================
        # STAGE 4: TEMPORAL ANALYSIS
        # =========================================================================
        print_step(4, total_steps, "TEMPORAL ANALYSIS")
        
        result = run_temporal_analysis(df=df)
        temporal_classifications = result['classifications']
        
        # Save classifications for later use
        with open(TEMPORAL_CLASSIFICATIONS_PATH, 'w') as f:
            # Convert to JSON-serializable format
            json_data = {}
            for cid, info in temporal_classifications.items():
                json_data[str(cid)] = {
                    'type': info['type'],
                    'matched_events': info['matched_events'],
                    'stats': {
                        'total_photos': info['stats'].get('total_photos', 0),
                        'december_ratio': info['stats'].get('december_ratio', 0),
                        'summer_ratio': info['stats'].get('summer_ratio', 0),
                        'peak_month': info['stats'].get('peak_month', 0),
                    }
                }
            json.dump(json_data, f, indent=2)
        
        print("✅ Temporal analysis complete")
        
        # =========================================================================
        # STAGE 5: ENHANCED MAP
        # =========================================================================
        print_step(5, total_steps, "ENHANCED MAP GENERATION")
    
    # Load temporal classifications if not already loaded
    if map_only or 'temporal_classifications' not in dir():
        if TEMPORAL_CLASSIFICATIONS_PATH.exists():
            with open(TEMPORAL_CLASSIFICATIONS_PATH, 'r') as f:
                json_data = json.load(f)
                temporal_classifications = {}
                for cid_str, info in json_data.items():
                    temporal_classifications[int(cid_str)] = info
        else:
            print("  Running temporal classification...")
            result = run_temporal_analysis(df=df)
            temporal_classifications = result['classifications']
    
    # Generate enhanced map v2
    # Import v2 functions
    from scripts.create_enhanced_map_v2 import (
        compute_monthly_distributions, create_enhanced_map_v2
    )
    
    # Compute monthly distributions
    print("  Computing monthly distributions...")
    monthly_distributions = compute_monthly_distributions(df)
    
    # Create v2 map
    create_enhanced_map_v2(
        df=df,
        temporal_classifications=temporal_classifications,
        monthly_distributions=monthly_distributions,
        output_path=CLUSTER_MAP_PATH
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    
    print_header("✅ PIPELINE COMPLETE")
    print(f"\nTotal time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\nOutputs:")
    print(f"  📊 Cleaned data:      {CLEANED_DATA_PATH}")
    print(f"  📊 Clustered data:    {CLUSTERED_DATA_PATH}")
    print(f"  📝 TF-IDF descriptors: {REPORTS_DIR / 'cluster_descriptors.json'}")
    print(f"  📝 Association rules: {REPORTS_DIR / 'association_rules.json'}")
    print(f"  📊 Temporal analysis: {REPORTS_DIR / 'temporal_analysis.md'}")
    print(f"  🗺️  Cluster map:       {CLUSTER_MAP_PATH}")
    print(f"\n🎉 Open {CLUSTER_MAP_PATH} in your browser!")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete Grand Lyon Photo Clusters pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip steps if output files already exist"
    )
    parser.add_argument(
        "--quick",
        action="store_true", 
        help="Quick run with reduced sample size (20k photos)"
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Only regenerate the map (skip all processing)"
    )
    parser.add_argument(
        "--skip-rules",
        action="store_true",
        help="Skip association rules mining for faster execution"
    )
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="hdbscan",
        choices=["hdbscan", "dbscan", "kmeans", "hierarchical"],
        help="Clustering algorithm to use (default: hdbscan)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=120,
        help="HDBSCAN: minimum cluster size (default: 120)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.005,
        help="DBSCAN: epsilon radius (default: 0.005)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN/DBSCAN: minimum samples (default: auto)"
    )
    parser.add_argument(
        "--n-clusters", "-k",
        type=int,
        default=50,
        help="KMeans/Hierarchical: number of clusters (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Build algorithm params from CLI args
    algo_params = {
        'min_cluster_size': args.min_cluster_size,
        'min_samples': args.min_samples,
        'eps': args.eps,
        'n_clusters': args.n_clusters,
    }
    
    try:
        run_full_pipeline(
            skip_if_exists=args.skip_if_exists,
            quick=args.quick,
            map_only=args.map_only,
            skip_rules=args.skip_rules,
            algorithm=args.algorithm,
            algo_params=algo_params
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
