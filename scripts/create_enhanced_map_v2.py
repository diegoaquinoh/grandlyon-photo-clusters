#!/usr/bin/env python3
"""
Enhanced Cluster Map v2 - Interactive Filtering

Creates an interactive map with:
- Actual polygon filtering (not just sidebar)
- Month slider to explore seasonal patterns  
- Refined recurring categories (December, Summer, Seasonal)
- Peak month indicators in popups
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import html as html_escape

from src.data_loader import DATA_DIR, REPORTS_DIR
from src.map_visualization import APP_DIR, load_cluster_descriptors, load_cluster_names
from src.temporal_analysis import run_temporal_analysis

# Output paths
CLUSTERED_DATA_PATH = DATA_DIR / "flickr_clustered.csv"
CLUSTER_MAP_V2_PATH = APP_DIR / "cluster_map_v2.html"
TEMPORAL_CLASSIFICATIONS_PATH = REPORTS_DIR / "temporal_classifications.json"


def compute_monthly_distributions(df: pd.DataFrame) -> dict:
    """Compute monthly photo distribution for each cluster."""
    df_clustered = df[df['cluster'] != -1].copy()
    
    monthly_dist = {}
    for cluster_id in df_clustered['cluster'].unique():
        cluster_df = df_clustered[df_clustered['cluster'] == cluster_id]
        
        # Count by month
        month_counts = cluster_df['date_taken_month'].value_counts()
        total = len(cluster_df)
        
        # Normalize to percentages
        dist = []
        for month in range(1, 13):
            count = month_counts.get(month, 0)
            pct = (count / total * 100) if total > 0 else 0
            dist.append(round(pct, 1))
        
        monthly_dist[int(cluster_id)] = dist
    
    return monthly_dist


def classify_recurring_subtype(stats: dict) -> tuple:
    """
    Classify recurring clusters into subtypes.
    
    Returns:
        (subtype, icon, label)
    """
    december_ratio = stats.get('december_ratio', 0)
    summer_ratio = stats.get('summer_ratio', 0)
    peak_month = int(stats.get('peak_month', 0))
    
    if december_ratio > 0.4:
        return ('december', '🎄', 'Fête des Lumières Pattern')
    elif december_ratio > 0.25:
        return ('december', '🎄', 'December Event')
    elif summer_ratio > 0.5:
        return ('summer', '☀️', 'Summer Hotspot')
    elif summer_ratio > 0.35:
        return ('summer', '☀️', 'Summer Tourism')
    elif peak_month in [3, 4, 5]:
        return ('spring', '🌸', 'Spring Pattern')
    elif peak_month in [9, 10, 11]:
        return ('autumn', '🍂', 'Autumn Pattern')
    else:
        return ('seasonal', '📅', 'Seasonal Pattern')


def create_enhanced_map_v2(
    df: pd.DataFrame,
    temporal_classifications: dict,
    monthly_distributions: dict,
    output_path: Path = CLUSTER_MAP_V2_PATH
):
    """
    Create enhanced cluster map v2 with interactive JavaScript filtering.
    """
    import folium
    
    print("Creating enhanced cluster map v2...")
    
    # Filter noise
    df_clustered = df[df['cluster'] != -1].copy()
    
    # Get cluster info
    cluster_counts = df_clustered['cluster'].value_counts()
    unique_clusters = sorted([c for c in df_clustered['cluster'].unique()])
    
    print(f"  Processing {len(unique_clusters)} clusters...")
    
    # Load descriptors and names
    cluster_descriptors = load_cluster_descriptors()
    cluster_names = load_cluster_names()
    
    # Classify all clusters with refined categories
    cluster_data = []
    
    for cluster_id in unique_clusters:
        cluster_df = df_clustered[df_clustered['cluster'] == cluster_id]
        cluster_size = len(cluster_df)
        
        if cluster_size < 3:
            continue
        
        # Get coordinates
        coords = cluster_df[['lat', 'long']].values
        
        # Compute convex hull
        try:
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])
            polygon_coords = hull_points.tolist()
        except:
            # Fallback: create a small polygon around centroid
            centroid = coords.mean(axis=0)
            continue
        
        # Get cluster metadata
        temporal_info = temporal_classifications.get(cluster_id, {})
        base_type = temporal_info.get('type', 'unknown')
        stats = temporal_info.get('stats', {})
        
        # Refine recurring category
        if base_type == 'recurring':
            subtype, icon, label = classify_recurring_subtype(stats)
            refined_type = f"recurring_{subtype}"
        elif base_type == 'permanent':
            refined_type = 'permanent'
            icon = '🏛️'
            label = 'Permanent Landmark'
        elif base_type == 'one_time':
            refined_type = 'one_time'
            icon = '⚡'
            label = 'One-time Event'
        else:
            refined_type = 'unknown'
            icon = '❓'
            label = 'Unknown'
        
        # Get name and descriptors
        name_info = cluster_names.get(cluster_id, {})
        cluster_name = name_info.get('name', f'Cluster {cluster_id}') if isinstance(name_info, dict) else f'Cluster {cluster_id}'
        
        top_terms = cluster_descriptors.get(cluster_id, [])
        descriptor_str = ", ".join(top_terms[:5]) if top_terms else ""
        
        # Get year range
        years = cluster_df['date_taken_year'].dropna()
        if len(years) > 0:
            min_year = int(years.min())
            max_year = int(years.max())
        else:
            min_year = max_year = 2010
        
        # Get monthly distribution
        monthly_dist = monthly_distributions.get(cluster_id, [0]*12)
        
        # Get peak month
        peak_month = int(stats.get('peak_month', 0))
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        peak_month_name = month_names[peak_month - 1] if 1 <= peak_month <= 12 else 'N/A'
        peak_month_pct = monthly_dist[peak_month - 1] if 1 <= peak_month <= 12 else 0
        
        # Centroid for popup
        centroid = coords.mean(axis=0)
        
        cluster_data.append({
            'id': int(cluster_id),
            'name': cluster_name,
            'type': refined_type,
            'icon': icon,
            'label': label,
            'size': cluster_size,
            'polygon': polygon_coords,
            'centroid': [float(centroid[0]), float(centroid[1])],
            'year_min': min_year,
            'year_max': max_year,
            'peak_month': peak_month,
            'peak_month_name': peak_month_name,
            'peak_pct': peak_month_pct,
            'monthly_dist': monthly_dist,
            'december_ratio': round(stats.get('december_ratio', 0) * 100, 1),
            'summer_ratio': round(stats.get('summer_ratio', 0) * 100, 1),
            'terms': descriptor_str
        })
    
    print(f"  Prepared data for {len(cluster_data)} clusters")
    
    # Create the HTML map
    html_content = generate_map_html(cluster_data)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"  ✅ Enhanced map v2 saved to: {output_path}")
    
    return cluster_data


def generate_map_html(cluster_data: list) -> str:
    """Generate the complete HTML for the interactive map."""
    
    # Convert cluster data to JSON
    cluster_json = json.dumps(cluster_data)
    
    # Count by type
    type_counts = {}
    for c in cluster_data:
        t = c['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    # Type colors
    type_colors = {
        'permanent': '#2E86AB',
        'recurring_december': '#C41E3A',
        'recurring_summer': '#FF8C00',
        'recurring_spring': '#32CD32',
        'recurring_autumn': '#8B4513',
        'recurring_seasonal': '#9932CC',
        'one_time': '#FFD700',
        'unknown': '#888888'
    }
    
    colors_json = json.dumps(type_colors)
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Grand Lyon Photo Clusters - Enhanced Map v2</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        
        #map {{ 
            position: absolute; 
            top: 0; left: 0; right: 0; bottom: 0; 
            z-index: 1;
        }}
        
        /* Control Panel */
        #control-panel {{
            position: fixed;
            top: 10px;
            right: 10px;
            width: 360px;
            max-height: 90vh;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        
        .panel-header {{
            padding: 16px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
        }}
        
        .panel-header h2 {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        
        .panel-header .subtitle {{
            font-size: 12px;
            opacity: 0.8;
        }}
        
        .panel-section {{
            padding: 14px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .section-title {{
            font-size: 12px;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}
        
        /* Month Slider */
        .month-slider-container {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
        }}
        
        .month-display {{
            text-align: center;
            font-size: 24px;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 8px;
        }}
        
        .month-display .count {{
            font-size: 13px;
            font-weight: 400;
            color: #666;
        }}
        
        #month-slider {{
            width: 100%;
            height: 6px;
            -webkit-appearance: none;
            background: linear-gradient(to right, 
                #2E86AB 0%, #32CD32 25%, #FFD700 50%, 
                #FF8C00 75%, #C41E3A 100%);
            border-radius: 3px;
            outline: none;
        }}
        
        #month-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: white;
            border: 3px solid #1a1a2e;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }}
        
        .month-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #888;
            margin-top: 6px;
        }}
        
        .month-mode-toggle {{
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }}
        
        .mode-btn {{
            flex: 1;
            padding: 8px;
            border: 1px solid #ddd;
            background: white;
            border-radius: 6px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .mode-btn.active {{
            background: #1a1a2e;
            color: white;
            border-color: #1a1a2e;
        }}
        
        /* Type Filters */
        .type-filters {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        
        .type-btn {{
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 6px 10px;
            border: 2px solid transparent;
            border-radius: 20px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
            background: #f0f0f0;
        }}
        
        .type-btn.active {{
            border-color: currentColor;
            background: white;
        }}
        
        .type-btn .dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        
        /* Cluster List */
        #cluster-list {{
            flex: 1;
            overflow-y: auto;
            max-height: 40vh;
            padding: 8px;
        }}
        
        .cluster-card {{
            padding: 12px;
            margin: 6px 0;
            background: #f8f9fa;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            border-left: 4px solid transparent;
        }}
        
        .cluster-card:hover {{
            background: #e9ecef;
            transform: translateX(4px);
        }}
        
        .cluster-card.hidden {{
            display: none;
        }}
        
        .cluster-card .name {{
            font-weight: 600;
            font-size: 13px;
            color: #1a1a2e;
            margin-bottom: 4px;
        }}
        
        .cluster-card .meta {{
            font-size: 11px;
            color: #666;
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        
        .cluster-card .badge {{
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 10px;
            color: white;
        }}
        
        .cluster-card .peak-info {{
            font-size: 10px;
            color: #888;
            margin-top: 4px;
        }}
        
        /* Stats Bar */
        .stats-bar {{
            padding: 10px 14px;
            background: #f8f9fa;
            font-size: 12px;
            color: #666;
            text-align: center;
        }}
        
        /* Legend */
        #legend {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: white;
            padding: 14px 18px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            z-index: 1000;
            max-width: 220px;
        }}
        
        #legend h3 {{
            font-size: 13px;
            font-weight: 600;
            color: #1a1a2e;
            margin-bottom: 10px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            margin: 6px 0;
        }}
        
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 4px;
        }}
        
        /* Search */
        #search-input {{
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 13px;
            margin-bottom: 10px;
        }}
        
        #search-input:focus {{
            outline: none;
            border-color: #1a1a2e;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div id="control-panel">
        <div class="panel-header">
            <h2>🗺️ Grand Lyon Photo Clusters</h2>
            <div class="subtitle">Interactive Temporal Explorer</div>
        </div>
        
        <div class="panel-section">
            <div class="section-title">📅 Month Filter</div>
            <div class="month-slider-container">
                <div class="month-display">
                    <span id="current-month">All Months</span>
                    <div class="count"><span id="active-count">{len(cluster_data)}</span> clusters</div>
                </div>
                <input type="range" id="month-slider" min="0" max="12" value="0" step="1">
                <div class="month-labels">
                    <span>All</span>
                    <span>Mar</span>
                    <span>Jun</span>
                    <span>Sep</span>
                    <span>Dec</span>
                </div>
                <div class="month-mode-toggle">
                    <button class="mode-btn active" data-mode="filter">Filter Active</button>
                    <button class="mode-btn" data-mode="highlight">Highlight Only</button>
                </div>
            </div>
        </div>
        
        <div class="panel-section">
            <div class="section-title">🏷️ Cluster Types</div>
            <div class="type-filters" id="type-filters">
                <button class="type-btn active" data-type="all">
                    All ({len(cluster_data)})
                </button>
                <button class="type-btn" data-type="permanent" style="color: #2E86AB;">
                    <span class="dot" style="background: #2E86AB;"></span>
                    🏛️ Landmark ({type_counts.get('permanent', 0)})
                </button>
                <button class="type-btn" data-type="recurring_december" style="color: #C41E3A;">
                    <span class="dot" style="background: #C41E3A;"></span>
                    🎄 December ({type_counts.get('recurring_december', 0)})
                </button>
                <button class="type-btn" data-type="recurring_summer" style="color: #FF8C00;">
                    <span class="dot" style="background: #FF8C00;"></span>
                    ☀️ Summer ({type_counts.get('recurring_summer', 0)})
                </button>
                <button class="type-btn" data-type="one_time" style="color: #FFD700;">
                    <span class="dot" style="background: #FFD700;"></span>
                    ⚡ One-time ({type_counts.get('one_time', 0)})
                </button>
            </div>
        </div>
        
        <div class="panel-section">
            <input type="text" id="search-input" placeholder="🔍 Search by name or keyword...">
        </div>
        
        <div id="cluster-list"></div>
        
        <div class="stats-bar">
            Showing <span id="visible-count">{len(cluster_data)}</span> of {len(cluster_data)} clusters
        </div>
    </div>
    
    <div id="legend">
        <h3>📊 Cluster Types</h3>
        <div class="legend-item">
            <span class="legend-dot" style="background: #2E86AB;"></span>
            <span>🏛️ Permanent Landmark</span>
        </div>
        <div class="legend-item">
            <span class="legend-dot" style="background: #C41E3A;"></span>
            <span>🎄 December (Fête des Lumières)</span>
        </div>
        <div class="legend-item">
            <span class="legend-dot" style="background: #FF8C00;"></span>
            <span>☀️ Summer Tourism</span>
        </div>
        <div class="legend-item">
            <span class="legend-dot" style="background: #9932CC;"></span>
            <span>📅 Other Seasonal</span>
        </div>
        <div class="legend-item">
            <span class="legend-dot" style="background: #FFD700;"></span>
            <span>⚡ One-time Event</span>
        </div>
    </div>
    
    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <script>
        // Cluster data
        const clusters = {cluster_json};
        const typeColors = {colors_json};
        
        const monthNames = ['All', 'January', 'February', 'March', 'April', 'May', 
                           'June', 'July', 'August', 'September', 'October', 
                           'November', 'December'];
        const shortMonths = ['All', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 
                            'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        
        // Initialize map
        const map = L.map('map').setView([45.75, 4.85], 12);
        
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
            subdomains: 'abcd',
            maxZoom: 19
        }}).addTo(map);
        
        // Store polygon layers by cluster ID
        const clusterLayers = {{}};
        const allPolygonsLayer = L.layerGroup().addTo(map);
        
        // Create polygons for each cluster
        clusters.forEach(cluster => {{
            const color = typeColors[cluster.type] || '#888888';
            
            const polygon = L.polygon(cluster.polygon, {{
                color: color,
                weight: 2,
                fillColor: color,
                fillOpacity: 0.4,
                clusterId: cluster.id,
                clusterType: cluster.type
            }});
            
            // Create popup content
            const popupContent = `
                <div style="min-width: 260px; font-family: -apple-system, sans-serif;">
                    <div style="background: ${{color}}; color: white; padding: 10px; margin: -10px -10px 10px -10px; border-radius: 3px 3px 0 0;">
                        <b style="font-size: 14px;">${{cluster.name}}</b>
                        <div style="font-size: 11px; margin-top: 4px;">${{cluster.icon}} ${{cluster.label}}</div>
                    </div>
                    <div style="padding: 5px 0;">
                        <b>Photos:</b> ${{cluster.size.toLocaleString()}}<br>
                        <b>Years:</b> ${{cluster.year_min}} - ${{cluster.year_max}}<br>
                        <hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">
                        <b>📈 Peak Activity:</b> ${{cluster.peak_month_name}} (${{cluster.peak_pct}}%)<br>
                        <b>🎄 December:</b> ${{cluster.december_ratio}}%<br>
                        <b>☀️ Summer:</b> ${{cluster.summer_ratio}}%<br>
                        <hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">
                        <b>Keywords:</b> <i style="color: #666;">${{cluster.terms || 'N/A'}}</i>
                    </div>
                </div>
            `;
            
            polygon.bindPopup(popupContent, {{ maxWidth: 320 }});
            polygon.bindTooltip(`${{cluster.name}} (${{cluster.size}} photos)`, {{ sticky: true }});
            
            clusterLayers[cluster.id] = polygon;
            allPolygonsLayer.addLayer(polygon);
        }});
        
        // State
        let currentMonth = 0;
        let currentType = 'all';
        let searchTerm = '';
        let monthMode = 'filter';  // 'filter' or 'highlight'
        
        // Month slider
        const monthSlider = document.getElementById('month-slider');
        const currentMonthDisplay = document.getElementById('current-month');
        const activeCountDisplay = document.getElementById('active-count');
        
        monthSlider.addEventListener('input', function() {{
            currentMonth = parseInt(this.value);
            currentMonthDisplay.textContent = monthNames[currentMonth];
            updateDisplay();
        }});
        
        // Month mode toggle
        document.querySelectorAll('.mode-btn').forEach(btn => {{
            btn.addEventListener('click', function() {{
                document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                monthMode = this.dataset.mode;
                updateDisplay();
            }});
        }});
        
        // Type filter buttons
        document.querySelectorAll('.type-btn').forEach(btn => {{
            btn.addEventListener('click', function() {{
                document.querySelectorAll('.type-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                currentType = this.dataset.type;
                updateDisplay();
            }});
        }});
        
        // Search
        document.getElementById('search-input').addEventListener('input', function() {{
            searchTerm = this.value.toLowerCase();
            updateDisplay();
        }});
        
        function isClusterVisible(cluster) {{
            // Type filter
            if (currentType !== 'all' && cluster.type !== currentType) {{
                // Also check for partial type match (e.g., "recurring" matches "recurring_december")
                if (!cluster.type.startsWith(currentType)) {{
                    return false;
                }}
            }}
            
            // Search filter
            if (searchTerm) {{
                const nameMatch = cluster.name.toLowerCase().includes(searchTerm);
                const termsMatch = cluster.terms.toLowerCase().includes(searchTerm);
                if (!nameMatch && !termsMatch) return false;
            }}
            
            // Month filter (only in 'filter' mode)
            if (currentMonth > 0 && monthMode === 'filter') {{
                // Check if cluster has significant activity in selected month
                const monthPct = cluster.monthly_dist[currentMonth - 1];
                if (monthPct < 5) return false;  // Less than 5% activity
            }}
            
            return true;
        }}
        
        function getClusterOpacity(cluster) {{
            if (currentMonth === 0) return 0.4;
            
            const monthPct = cluster.monthly_dist[currentMonth - 1];
            
            if (monthMode === 'highlight') {{
                // In highlight mode, adjust opacity based on activity
                return Math.max(0.1, Math.min(0.8, monthPct / 30));
            }}
            
            return 0.4;
        }}
        
        function updateDisplay() {{
            let visibleCount = 0;
            let activeMonthCount = 0;
            
            clusters.forEach(cluster => {{
                const polygon = clusterLayers[cluster.id];
                const isVisible = isClusterVisible(cluster);
                
                if (isVisible) {{
                    visibleCount++;
                    if (allPolygonsLayer.hasLayer(polygon) === false) {{
                        allPolygonsLayer.addLayer(polygon);
                    }}
                    
                    // Update opacity for month highlighting
                    const opacity = getClusterOpacity(cluster);
                    polygon.setStyle({{ fillOpacity: opacity }});
                    
                    // Count clusters with significant activity in selected month
                    if (currentMonth > 0 && cluster.monthly_dist[currentMonth - 1] >= 5) {{
                        activeMonthCount++;
                    }}
                }} else {{
                    if (allPolygonsLayer.hasLayer(polygon)) {{
                        allPolygonsLayer.removeLayer(polygon);
                    }}
                }}
            }});
            
            // Update counts
            document.getElementById('visible-count').textContent = visibleCount;
            
            if (currentMonth > 0) {{
                activeCountDisplay.textContent = activeMonthCount;
            }} else {{
                activeCountDisplay.textContent = visibleCount;
            }}
            
            // Update cluster list
            updateClusterList();
        }}
        
        function updateClusterList() {{
            const listContainer = document.getElementById('cluster-list');
            listContainer.innerHTML = '';
            
            const visibleClusters = clusters.filter(isClusterVisible);
            
            // Sort by size
            visibleClusters.sort((a, b) => b.size - a.size);
            
            // Show top 50
            visibleClusters.slice(0, 50).forEach(cluster => {{
                const color = typeColors[cluster.type] || '#888888';
                const card = document.createElement('div');
                card.className = 'cluster-card';
                card.style.borderLeftColor = color;
                
                let peakInfo = '';
                if (cluster.type.includes('december')) {{
                    peakInfo = `Peak: December (${{cluster.december_ratio}}%)`;
                }} else if (cluster.type.includes('summer')) {{
                    peakInfo = `Peak: Jul-Aug (${{cluster.summer_ratio}}%)`;
                }} else {{
                    peakInfo = `Peak: ${{cluster.peak_month_name}} (${{cluster.peak_pct}}%)`;
                }}
                
                card.innerHTML = `
                    <div class="name">${{cluster.icon}} ${{cluster.name}}</div>
                    <div class="meta">
                        <span>${{cluster.size.toLocaleString()}} photos</span>
                        <span class="badge" style="background: ${{color}};">${{cluster.label}}</span>
                    </div>
                    <div class="peak-info">📈 ${{peakInfo}}</div>
                `;
                
                card.onclick = () => {{
                    map.flyTo(cluster.centroid, 15);
                    clusterLayers[cluster.id].openPopup();
                }};
                
                listContainer.appendChild(card);
            }});
            
            if (visibleClusters.length > 50) {{
                const more = document.createElement('div');
                more.style.cssText = 'text-align: center; padding: 10px; color: #666; font-size: 12px;';
                more.textContent = `...and ${{visibleClusters.length - 50}} more`;
                listContainer.appendChild(more);
            }}
        }}
        
        // Initial render
        updateDisplay();
    </script>
</body>
</html>'''
    
    return html


def main():
    print("=" * 60)
    print("  ENHANCED CLUSTER MAP v2")
    print("=" * 60)
    
    # Load data
    print("\nLoading clustered data...")
    df = pd.read_csv(CLUSTERED_DATA_PATH)
    print(f"  Loaded {len(df):,} photos")
    
    # Load or compute temporal classifications
    if TEMPORAL_CLASSIFICATIONS_PATH.exists():
        print("Loading temporal classifications...")
        with open(TEMPORAL_CLASSIFICATIONS_PATH, 'r') as f:
            json_data = json.load(f)
            temporal_classifications = {}
            for cid_str, info in json_data.items():
                temporal_classifications[int(cid_str)] = info
    else:
        print("Computing temporal classifications...")
        result = run_temporal_analysis(df=df)
        temporal_classifications = result['classifications']
    
    # Compute monthly distributions
    print("Computing monthly distributions...")
    monthly_distributions = compute_monthly_distributions(df)
    
    # Create enhanced map
    create_enhanced_map_v2(
        df=df,
        temporal_classifications=temporal_classifications,
        monthly_distributions=monthly_distributions,
        output_path=CLUSTER_MAP_V2_PATH
    )
    
    print(f"\n🎉 Open {CLUSTER_MAP_V2_PATH} in your browser!")


if __name__ == "__main__":
    main()
