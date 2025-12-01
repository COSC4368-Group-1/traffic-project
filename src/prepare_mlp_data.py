# prepare_training_data.py
"""
Complete data preparation for MLP → GNN pipeline
Generates:
1. Edge-level features for MLP
2. Node-level features for GNN
3. Adjacency matrix for GNN
4. Graph structure data
"""

import os
import xml.etree.ElementTree as ET
import geopandas as gpd
import pandas as pd
import numpy as np
import pickle
import torch
from pathlib import Path
from rtree import index
from collections import defaultdict

# ============================================
# CONFIGURATION
# ============================================
RAW_DATA_DIR = "raw_data/Houston_TX_USA"
FILTERED_DATA_DIR = "need_data"
OUTPUT_DIR = "training_data"
NUM_RUNS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# STEP 1: Load OSM Data (Nodes + Edges)
# ============================================
def load_osm_data():
    """Load both nodes and edges from OSM"""
    print("\n[1/6] Loading OSM road network (nodes + edges)...")
    
    nodes_file = os.path.join(RAW_DATA_DIR, "nodes.geojson")
    edges_file = os.path.join(RAW_DATA_DIR, "edges.geojson")
    
    if not os.path.exists(nodes_file):
        raise FileNotFoundError(f"Missing {nodes_file}")
    if not os.path.exists(edges_file):
        raise FileNotFoundError(f"Missing {edges_file}")
    
    nodes_gdf = gpd.read_file(nodes_file)
    edges_gdf = gpd.read_file(edges_file)
    
    print(f"  ✓ Loaded {len(nodes_gdf)} nodes")
    print(f"  ✓ Loaded {len(edges_gdf)} edges")
    
    # Build spatial index for edges
    print("  Building spatial index for edges...")
    osm_idx = index.Index()
    osm_edge_data = {}
    
    for i, (edge_idx, edge) in enumerate(edges_gdf.iterrows()):
        if edge['geometry'].geom_type == 'LineString':
            osm_idx.insert(i, edge['geometry'].bounds)
            
            osm_edge_data[i] = {
                'edge_tuple': (edge['u'], edge['v'], edge.get('key', 0)),
                'u': edge['u'],
                'v': edge['v'],
                'lanes': int(edge.get('lanes', 1)),
                'width': float(edge.get('width', 3.5)),
                'length': float(edge.get('length', 0)),
                'maxspeed': float(edge.get('maxspeed', 13.0)),
                'straightness': float(edge.get('straightness', 0.5)),
                'bridge': int(edge.get('bridge', 0)),
                'tunnel': int(edge.get('tunnel', 0)),
                'emergency_score': float(edge.get('emergency_score', 0.5)),
                'is_restricted': int(edge.get('is_restricted', 0)),
                'highway': str(edge.get('highway', 'residential')),
                'geometry': edge['geometry']
            }
    
    # Create node mapping
    node_to_idx = {node['osmid']: idx for idx, node in nodes_gdf.iterrows()}
    
    return nodes_gdf, edges_gdf, osm_edge_data, osm_idx, node_to_idx

# ============================================
# STEP 2: Parse SUMO Traffic Data (Same as before)
# ============================================
def parse_sumo_traffic_features(edgedata_file):
    """Extract traffic features from SUMO edge data XML"""
    print(f"  Parsing {os.path.basename(edgedata_file)}...")
    
    traffic_features = {}
    
    try:
        tree = ET.parse(edgedata_file)
        root = tree.getroot()
        intervals = root.findall('interval')
        
        if not intervals:
            return {}
        
        # Use last interval (steady-state)
        interval = intervals[-1]
        
        for edge_elem in interval.findall('edge'):
            edge_id = edge_elem.get('id')
            
            traffic_features[edge_id] = {
                'traffic_speed': float(edge_elem.get('speed', 0) or 0),
                'traffic_density': float(edge_elem.get('density', 0) or 0),
                'traffic_occupancy': float(edge_elem.get('occupancy', 0) or 0),
                'traffic_flow': float(edge_elem.get('flow', 0) or 0),
                'travel_time': float(edge_elem.get('traveltime', 0) or 0),
                'time_loss': float(edge_elem.get('timeLoss', 0) or 0),
                'waiting_time': float(edge_elem.get('waitingTime', 0) or 0),
                'departed': int(edge_elem.get('departed', 0) or 0),
                'arrived': int(edge_elem.get('arrived', 0) or 0),
                'entered': int(edge_elem.get('entered', 0) or 0),
                'left': int(edge_elem.get('left', 0) or 0),
            }
        
        print(f"    ✓ {len(traffic_features)} edges")
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return {}
    
    return traffic_features

# ============================================
# STEP 3: Spatial Matching (Same as before)
# ============================================
def load_sumo_geometries(sumo_net_file):
    """Load SUMO edge geometries"""
    print("  Loading SUMO geometries...")
    sumo_edges = {}
    
    try:
        tree = ET.parse(sumo_net_file)
        root = tree.getroot()
        
        for edge in root.findall('edge'):
            edge_id = edge.get('id')
            shape = edge.get('shape')
            if shape:
                coords = [tuple(map(float, p.split(','))) for p in shape.split()]
                sumo_edges[edge_id] = coords
        
        print(f"    ✓ {len(sumo_edges)} geometries")
        return sumo_edges
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return {}

def match_sumo_to_osm(sumo_id, sumo_edges, osm_edge_data, osm_idx, 
                      sumo_bounds, osm_bounds):
    if sumo_id not in sumo_edges or len(sumo_edges[sumo_id]) < 2:
        return None
    
    coords = sumo_edges[sumo_id]
    sumo_mid_x = (coords[0][0] + coords[-1][0]) / 2
    sumo_mid_y = (coords[0][1] + coords[-1][1]) / 2
    
    # Transform coordinates
    sumo_width = sumo_bounds[2] - sumo_bounds[0]
    sumo_height = sumo_bounds[3] - sumo_bounds[1]
    osm_width = osm_bounds[2] - osm_bounds[0]
    osm_height = osm_bounds[3] - osm_bounds[1]
    
    norm_x = (sumo_mid_x - sumo_bounds[0]) / sumo_width
    norm_y = (sumo_mid_y - sumo_bounds[1]) / sumo_height
    
    osm_mid_x = osm_bounds[0] + norm_x * osm_width
    osm_mid_y = osm_bounds[1] + norm_y * osm_height
    
    # Increased search radius for better matching
    search_radius = 0.003  # Increased from 0.001
    nearby = list(osm_idx.intersection((
        osm_mid_x - search_radius, osm_mid_y - search_radius,
        osm_mid_x + search_radius, osm_mid_y + search_radius
    )))
    
    if not nearby:
        return None
    
    # Find closest
    best_idx, min_dist = None, float('inf')
    for idx in nearby:
        osm_edge = osm_edge_data[idx]
        geom = osm_edge['geometry']
        
        if geom.geom_type != 'LineString':
            continue
        
        osm_coords = list(geom.coords)
        if len(osm_coords) < 2:
            continue
        
        osm_x = (osm_coords[0][0] + osm_coords[-1][0]) / 2
        osm_y = (osm_coords[0][1] + osm_coords[-1][1]) / 2
        
        dist = (osm_mid_x - osm_x)**2 + (osm_mid_y - osm_y)**2
        
        if dist < min_dist and dist < search_radius**2:
            min_dist, best_idx = dist, idx
    
    return best_idx

# ============================================
# STEP 4: Create Edge-Level Data (MLP)
# ============================================
def create_edge_level_data(osm_edge_data, osm_idx, edges_gdf, 
                           traffic_features_all_runs):
    """Create edge-level training data for MLP"""
    print("\n[3/6] Creating edge-level data for MLP...")
    
    # Load SUMO geometries
    sumo_net_file = os.path.join(RAW_DATA_DIR, "network.net.xml")
    sumo_edges = load_sumo_geometries(sumo_net_file)
    
    if not sumo_edges:
        return []
    
    # Calculate bounds
    all_coords = [c for coords in sumo_edges.values() for c in coords]
    sumo_bounds = (min(c[0] for c in all_coords), min(c[1] for c in all_coords),
                   max(c[0] for c in all_coords), max(c[1] for c in all_coords))
    osm_bounds = edges_gdf.total_bounds
    
    # Encode highway types
    highway_types = ['motorway', 'trunk', 'primary', 'secondary', 
                     'tertiary', 'residential', 'living_street', 'service', 'unclassified']
    highway_to_idx = {ht: i for i, ht in enumerate(highway_types)}
    
    edge_samples = []
    
    for run_num, traffic_features in traffic_features_all_runs.items():
        print(f"  Run {run_num}...")
        matched = 0
        
        for sumo_id, traffic_data in traffic_features.items():
            osm_idx_val = match_sumo_to_osm(
                sumo_id, sumo_edges, osm_edge_data, osm_idx, 
                sumo_bounds, osm_bounds
            )
            
            if osm_idx_val is None:
                continue
            
            osm_edge = osm_edge_data[osm_idx_val]
            
            sample = {
                'run': run_num,
                'osm_u': osm_edge['u'],
                'osm_v': osm_edge['v'],
                
                # OSM features
                'lanes': osm_edge['lanes'],
                'width': osm_edge['width'],
                'length': osm_edge['length'],
                'maxspeed': osm_edge['maxspeed'],
                'straightness': osm_edge['straightness'],
                'bridge': osm_edge['bridge'],
                'tunnel': osm_edge['tunnel'],
                'emergency_score': osm_edge['emergency_score'],
                'is_restricted': osm_edge['is_restricted'],
                'highway_type': highway_to_idx.get(osm_edge['highway'], 5),
                
                 # SELECTED traffic features for MLP INPUTS (different from target)
                'traffic_flow': traffic_data['traffic_flow'],      # "flow" in XML
                'travel_time': traffic_data['travel_time'],        # "traveltime" in XML  
                'time_loss': traffic_data['time_loss'],            # "timeLoss" in XML
                'waiting_time': traffic_data['waiting_time'],      # "waitingTime" in XML
                # We are NOT including: speed, density, occupancy (used in target)
                
                # Target
                'target': calculate_emergency_suitability(traffic_data, osm_edge)
            }
            
            edge_samples.append(sample)
            matched += 1
        
        print(f"    ✓ {matched} matched edges")
    
    print(f"  Total: {len(edge_samples)} edge samples")
    return edge_samples

def calculate_emergency_suitability(traffic_data, osm_edge):
    """Enhanced emergency vehicle suitability score - focused on clearance"""
    speed = traffic_data.get('traffic_speed', 0)
    density = traffic_data.get('traffic_density', 0)
    flow = traffic_data.get('traffic_flow', 0)
    
    # Emergency vehicles prioritize clear paths over speed
    if density > 0 or flow > 0:  # Has actual traffic
        clearance_score = max(0, 1.0 - (density / 40.0)) * 0.5  # Higher weight for clearance
        speed_score = min(speed / 25.0, 1.0) * 0.3  # Higher speed threshold for emergencies
        road_quality = osm_edge['emergency_score'] * 0.2
    else:
        # No traffic - perfect for emergencies
        clearance_score = 1.0 * 0.5
        speed_score = 1.0 * 0.3  
        road_quality = osm_edge['emergency_score'] * 0.2
    
    return min(1.0, max(0.0, clearance_score + speed_score + road_quality))

# ============================================
# STEP 5: Aggregate to Node-Level Features (GNN)
# ============================================
def create_node_level_data(edge_samples, nodes_gdf):
    """
    Aggregate edge-level features to node-level features for GNN
    For each node (intersection), compute:
    - Average traffic metrics of connected edges
    - Total flow through node
    - Number of high-traffic edges
    """
    print("\n[4/6] Creating node-level data for GNN...")
    
    # Group edges by run
    edges_by_run = defaultdict(list)
    for sample in edge_samples:
        edges_by_run[sample['run']].append(sample)
    
    node_samples_all_runs = []
    
    for run_num, run_edges in edges_by_run.items():
        print(f"  Run {run_num}...")
        
        # Aggregate features per node - USE THE FEATURES WE ACTUALLY HAVE
        node_features = defaultdict(lambda: {
            'flows': [],
            'travel_times': [],
            'time_losses': [],
            'waiting_times': [],
            'emergency_scores': [],
            'lanes': [],
            'connected_edges': 0
        })
        
        # Aggregate from edges - USE THE FEATURES WE ACTUALLY HAVE
        for edge in run_edges:
            u, v = edge['osm_u'], edge['osm_v']
            
            # Add to both source and target nodes
            for node_id in [u, v]:
                node_features[node_id]['flows'].append(edge['traffic_flow'])
                node_features[node_id]['travel_times'].append(edge['travel_time'])
                node_features[node_id]['time_losses'].append(edge['time_loss'])
                node_features[node_id]['waiting_times'].append(edge['waiting_time'])
                node_features[node_id]['emergency_scores'].append(edge['emergency_score'])
                node_features[node_id]['lanes'].append(edge['lanes'])
                node_features[node_id]['connected_edges'] += 1
        
        # Convert to node samples - UPDATE METRICS TO MATCH OUR FEATURES
        node_samples = []
        for node_id, features in node_features.items():
            if features['connected_edges'] == 0:
                continue
            
            sample = {
                'run': run_num,
                'node_id': node_id,
                
                # Aggregated traffic features - USE WHAT WE HAVE
                'avg_flow': np.mean(features['flows']),
                'max_flow': np.max(features['flows']),
                'min_flow': np.min(features['flows']),
                
                'avg_travel_time': np.mean(features['travel_times']),
                'max_travel_time': np.max(features['travel_times']),
                
                'avg_time_loss': np.mean(features['time_losses']),
                'max_time_loss': np.max(features['time_losses']),
                
                'avg_waiting_time': np.mean(features['waiting_times']),
                
                # Road network features
                'avg_emergency_score': np.mean(features['emergency_scores']),
                'avg_lanes': np.mean(features['lanes']),
                'degree': features['connected_edges'],
                
                # Node congestion indicator - USE TRAVEL TIME INSTEAD OF SPEED
                'is_congested': 1 if np.mean(features['travel_times']) > 30.0 else 0,
            }
            
            node_samples.append(sample)
        
        node_samples_all_runs.extend(node_samples)
        print(f"    ✓ {len(node_samples)} nodes with traffic")
    
    print(f"  Total: {len(node_samples_all_runs)} node samples")
    return node_samples_all_runs

# ============================================
# STEP 6: Create Adjacency Matrix (GNN)
# ============================================
def create_adjacency_matrix(edges_gdf, nodes_gdf, edge_samples):
    """
    ULTRA-FAST: Vectorized version using pandas
    """
    print("\n[5/6] Creating adjacency matrix for GNN (vectorized)...")
    
    # Create node ID mapping
    unique_nodes = sorted(nodes_gdf['osmid'].unique())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
    
    print(f"  Total nodes in graph: {len(unique_nodes)}")
    
    # Build traffic edge set
    traffic_edge_set = set((s['osm_u'], s['osm_v']) for s in edge_samples)
    print(f"  Traffic edges: {len(traffic_edge_set)}")
    
    # Filter edges where both nodes exist
    print("  Filtering valid edges...")
    valid_mask = edges_gdf['u'].isin(node_to_idx) & edges_gdf['v'].isin(node_to_idx)
    valid_edges = edges_gdf[valid_mask].copy()
    
    print(f"  Valid edges: {len(valid_edges)}/{len(edges_gdf)}")
    
    # Vectorized node index mapping
    print("  Mapping node indices...")
    valid_edges['u_idx'] = valid_edges['u'].map(node_to_idx)
    valid_edges['v_idx'] = valid_edges['v'].map(node_to_idx)
    
    # Vectorized traffic check
    print("  Checking traffic status...")
    valid_edges['has_traffic'] = valid_edges.apply(
        lambda row: 1 if (row['u'], row['v']) in traffic_edge_set else 0,
        axis=1
    )
    
    # Build edge index
    edge_index = valid_edges[['u_idx', 'v_idx']].values.T
    edge_has_traffic = valid_edges['has_traffic'].values
    
    print(f"  Total edges: {edge_index.shape[1]}")
    print(f"  Edges with traffic: {edge_has_traffic.sum()}")
    print(f"  Edge coverage: {edge_has_traffic.sum()/len(edge_has_traffic)*100:.1f}%")
    
    adjacency_data = {
        'edge_index': edge_index,
        'edge_has_traffic': edge_has_traffic,
        'node_to_idx': node_to_idx,
        'idx_to_node': {idx: node for node, idx in node_to_idx.items()},
        'num_nodes': len(unique_nodes),
        'num_edges': edge_index.shape[1]
    }
    
    return adjacency_data

# ============================================
# STEP 7: Save All Data (NO LEAKAGE VERSION)
# ============================================
def save_all_data(edge_samples, node_samples, adjacency_data):
    """Save all prepared data with proper edge-based splitting"""
    print("\n[6/6] Saving all data (with leakage prevention)...")
    
    # ===== SPLIT BY UNIQUE EDGES (NOT SAMPLES) =====
    print("\n  Creating train/val/test split by unique edges...")
    
    # Group samples by edge
    edges_to_samples = defaultdict(list)
    for sample in edge_samples:
        edge_key = (sample['osm_u'], sample['osm_v'])
        edges_to_samples[edge_key].append(sample)
    
    # Split EDGES (not samples)
    unique_edges = list(edges_to_samples.keys())
    np.random.seed(42)
    np.random.shuffle(unique_edges)
    
    n_edges = len(unique_edges)
    train_edge_end = int(n_edges * 0.7)
    val_edge_end = int(n_edges * 0.85)
    
    train_edges = set(unique_edges[:train_edge_end])
    val_edges = set(unique_edges[train_edge_end:val_edge_end])
    test_edges = set(unique_edges[val_edge_end:])
    
    # Assign samples based on which edge they belong to
    edge_train = []
    edge_val = []
    edge_test = []
    
    for edge_key, edge_samples_list in edges_to_samples.items():
        if edge_key in train_edges:
            edge_train.extend(edge_samples_list)
        elif edge_key in val_edges:
            edge_val.extend(edge_samples_list)
        else:
            edge_test.extend(edge_samples_list)
    
    print(f"  Edges split:")
    print(f"    Train edges: {len(train_edges)}")
    print(f"    Val edges:   {len(val_edges)}")
    print(f"    Test edges:  {len(test_edges)}")
    print(f"  Sample split:")
    print(f"    Train: {len(edge_train)} samples")
    print(f"    Val:   {len(edge_val)} samples")
    print(f"    Test:  {len(edge_test)} samples")
    
    # Verify no leakage
    train_edge_set = set((s['osm_u'], s['osm_v']) for s in edge_train)
    val_edge_set = set((s['osm_u'], s['osm_v']) for s in edge_val)
    test_edge_set = set((s['osm_u'], s['osm_v']) for s in edge_test)
    
    overlap_train_val = train_edge_set & val_edge_set
    overlap_train_test = train_edge_set & test_edge_set
    overlap_val_test = val_edge_set & test_edge_set
    
    print(f"\n  ✓ Leakage check:")
    print(f"    Train-Val overlap:  {len(overlap_train_val)} edges")
    print(f"    Train-Test overlap: {len(overlap_train_test)} edges")
    print(f"    Val-Test overlap:   {len(overlap_val_test)} edges")
    
    if len(overlap_train_val) > 0 or len(overlap_train_test) > 0 or len(overlap_val_test) > 0:
        print("    ⚠ WARNING: Data leakage detected!")
    else:
        print("    ✓ No data leakage - splits are clean!")
    
    # Save edge-level (MLP) data
    with open(os.path.join(OUTPUT_DIR, "mlp_train.pkl"), 'wb') as f:
        pickle.dump(edge_train, f)
    with open(os.path.join(OUTPUT_DIR, "mlp_val.pkl"), 'wb') as f:
        pickle.dump(edge_val, f)
    with open(os.path.join(OUTPUT_DIR, "mlp_test.pkl"), 'wb') as f:
        pickle.dump(edge_test, f)
    
    pd.DataFrame(edge_train).to_csv(os.path.join(OUTPUT_DIR, "mlp_train.csv"), index=False)
    
    print(f"\n  MLP data saved:")
    print(f"    Train: {len(edge_train)} samples")
    print(f"    Val:   {len(edge_val)} samples")
    print(f"    Test:  {len(edge_test)} samples")
    
    # Save node-level (GNN) data
    with open(os.path.join(OUTPUT_DIR, "gnn_node_features.pkl"), 'wb') as f:
        pickle.dump(node_samples, f)
    
    pd.DataFrame(node_samples).to_csv(
        os.path.join(OUTPUT_DIR, "gnn_node_features.csv"), index=False
    )
    
    print(f"  GNN node features: {len(node_samples)} samples")
    
    # Save adjacency matrix
    with open(os.path.join(OUTPUT_DIR, "adjacency_matrix.pkl"), 'wb') as f:
        pickle.dump(adjacency_data, f)
    
    print(f"  Adjacency matrix: {adjacency_data['num_nodes']} nodes, "
          f"{adjacency_data['num_edges']} edges")
    
    # Save as PyTorch tensors for GNN
    torch.save({
        'edge_index': torch.LongTensor(adjacency_data['edge_index']),
        'edge_mask': torch.FloatTensor(adjacency_data['edge_has_traffic']),
        'node_to_idx': adjacency_data['node_to_idx'],
        'num_nodes': adjacency_data['num_nodes']
    }, os.path.join(OUTPUT_DIR, "graph_structure.pt"))
    
    print(f"  ✓ Saved PyTorch graph structure")

def check_unique_edge_coverage(edge_samples):
    """Check how many unique edges we have data for"""
    print("\n[DIAGNOSTIC] Unique Edge Analysis:")
    
    unique_edges = set()
    for sample in edge_samples:
        edge_key = (sample['osm_u'], sample['osm_v'])
        unique_edges.add(edge_key)
    
    print(f"  Total samples: {len(edge_samples)}")
    print(f"  Unique edges: {len(unique_edges)}")
    print(f"  Average samples per edge: {len(edge_samples) / len(unique_edges):.1f}")
    
    # Check distribution across runs
    samples_by_run = {}
    for sample in edge_samples:
        run = sample['run']
        samples_by_run[run] = samples_by_run.get(run, 0) + 1
    
    print("  Samples per run:")
    for run, count in sorted(samples_by_run.items()):
        print(f"    Run {run}: {count} samples")


def filter_meaningful_traffic_data(edge_samples):
    """Filter to keep only samples with meaningful traffic data"""
    print("\n[FILTERING] Removing zero-traffic samples...")
    
    meaningful_samples = []
    traffic_stats = {
        'has_traffic': 0,
        'no_traffic': 0,
        'total': len(edge_samples)
    }
    
    for sample in edge_samples:
        # Use the traffic features we actually have: traffic_flow, travel_time, time_loss
        has_meaningful_traffic = (
            sample['traffic_flow'] > 1.0 or
            sample['travel_time'] > 0.1 or
            sample['time_loss'] > 0.1
        )
        
        if has_meaningful_traffic:
            meaningful_samples.append(sample)
            traffic_stats['has_traffic'] += 1
        else:
            traffic_stats['no_traffic'] += 1
    
    print(f"  Before filtering: {traffic_stats['total']} samples")
    print(f"  With meaningful traffic: {traffic_stats['has_traffic']} samples")
    print(f"  Without meaningful traffic: {traffic_stats['no_traffic']} samples")
    print(f"  Kept {len(meaningful_samples)}/{len(edge_samples)} samples ({len(meaningful_samples)/len(edge_samples)*100:.1f}%)")
    
    return meaningful_samples

# ============================================
# MAIN
# ============================================
def main():
    print("="*70)
    print("COMPLETE DATA PREPARATION (MLP + GNN)")
    print("="*70)
    
    # Step 1: Load OSM data
    nodes_gdf, edges_gdf, osm_edge_data, osm_idx, node_to_idx = load_osm_data()
    
    # Step 2: Load SUMO traffic
    print("\n[2/6] Loading SUMO traffic data...")
    traffic_features_all_runs = {}
    
    for run_num in range(1, NUM_RUNS + 1):
        edgedata_file = os.path.join(RAW_DATA_DIR, f"edgedata_run{run_num}.xml")
        if os.path.exists(edgedata_file):
            traffic_features_all_runs[run_num] = parse_sumo_traffic_features(edgedata_file)
    
    if not traffic_features_all_runs:
        print("✗ No traffic data found!")
        return
    
    # Step 3: Create edge-level data (MLP)
    edge_samples = create_edge_level_data(
        osm_edge_data, osm_idx, edges_gdf, traffic_features_all_runs
    )

    print("\n[3.5/6] Filtering for meaningful traffic data...")
    edge_samples = filter_meaningful_traffic_data(edge_samples)
    
    if not edge_samples:
        print("✗ No edge samples created!")
        return
    
    #-------diagnostic check-------#
    check_unique_edge_coverage(edge_samples)
    
    # Step 4: Create node-level data (GNN)
    node_samples = create_node_level_data(edge_samples, nodes_gdf)
    
    # Step 5: Create adjacency matrix (GNN)
    adjacency_data = create_adjacency_matrix(edges_gdf, nodes_gdf, edge_samples)
    
    # Step 6: Save everything
    save_all_data(edge_samples, node_samples, adjacency_data)
    
    print("\n" + "="*70)
    print("✓ DATA PREPARATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  MLP (edge-level):")
    print("    - mlp_train.pkl, mlp_val.pkl, mlp_test.pkl")
    print("    - mlp_train.csv (for inspection)")
    print("  GNN (node-level + graph):")
    print("    - gnn_node_features.pkl")
    print("    - adjacency_matrix.pkl")
    print("    - graph_structure.pt (PyTorch)")
    print("="*70)

if __name__ == "__main__":
    main()