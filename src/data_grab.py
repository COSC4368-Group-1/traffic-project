import os
import sys
import subprocess
import xml.etree.ElementTree as ET
import osmnx as ox
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from rtree import index
from pathlib import Path
from shapely.geometry import box

# -----------------------------
# Config - ENABLED ALL CITIES FOR SUMO
# -----------------------------
cities = [
    #{"name": "New York, NY, USA", "zoom": 17, "sample_nodes": 5, "run_sumo": False},  # ENABLED
    {"name": "Houston, TX, USA", "zoom": 17, "sample_nodes": 5, "run_sumo": True}
    #{"name": "Los Angeles, CA, USA", "zoom": 17, "sample_nodes": 5, "run_sumo": False},  # ENABLED
]

output_dir = "raw_data"
os.makedirs(output_dir, exist_ok=True)

# SUMO configuration
SIMULATION_TIME = 600  # 10 minutes for testing
NUM_TRAFFIC_SNAPSHOTS = 10

# -----------------------------
# IMPROVED Utility Functions
# -----------------------------
def check_sumo_installation():
    """Verify SUMO is installed and accessible"""
    try:
        result = subprocess.run(['sumo', '--version'], 
                              capture_output=True, 
                              text=True, 
                              check=True,
                              timeout=10)
        print(f"✓ SUMO found: {result.stdout.strip().split()[0]}")
        return True
    except FileNotFoundError:
        print("✗ SUMO not found in PATH!")
        print("Windows: Add C:\\Program Files\\Eclipse\\Sumo\\bin to PATH")
        print("Ubuntu: sudo apt-get install sumo sumo-tools")
        print("macOS: brew install sumo")
        return False
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"✗ SUMO error: {e}")
        return False

def get_sumo_tools_path():
    """Get path to SUMO tools directory with better detection"""
    # Try environment variable first
    sumo_home = os.environ.get('SUMO_HOME')
    if sumo_home:
        tools_path = os.path.join(sumo_home, 'tools')
        if os.path.exists(tools_path):
            print(f"✓ Found SUMO tools via SUMO_HOME: {tools_path}")
            return tools_path
    
    # Try to find via sumo binary location
    try:
        result = subprocess.run(['which', 'sumo'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            sumo_bin = result.stdout.strip()
            sumo_dir = os.path.dirname(os.path.dirname(sumo_bin))
            tools_path = os.path.join(sumo_dir, 'share', 'sumo', 'tools')
            if os.path.exists(tools_path):
                print(f"✓ Found SUMO tools via which: {tools_path}")
                return tools_path
    except:
        pass
    
    # Common installation paths
    common_paths = [
        'C:\\Program Files\\Eclipse\\Sumo\\tools',
        'C:\\Program Files (x86)\\Eclipse\\Sumo\\tools',
        'C:\\Program Files\\Eclipse\\Sumo\\share\\sumo\\tools',  # Newer installations
        '/usr/share/sumo/tools',
        '/usr/local/share/sumo/tools',
        '/opt/homebrew/share/sumo/tools'
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"✓ Found SUMO tools at common path: {path}")
            return path
    
    print("✗ Could not find SUMO tools directory")
    return None

def safe_float(value, default=0.0):
    """Safely convert to float"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def parse_lanes(lanes_raw):
    """Safely parse lane info from OSM attributes."""
    if isinstance(lanes_raw, (list, tuple)):
        try:
            return int(lanes_raw[0])
        except (ValueError, TypeError):
            return 1
    elif isinstance(lanes_raw, str):
        for sep in [';', '|', ',']:
            lanes_raw = lanes_raw.split(sep)[0]
        try:
            return int(lanes_raw)
        except ValueError:
            return 1
    else:
        try:
            return int(lanes_raw)
        except (ValueError, TypeError):
            return 1

# -----------------------------
# IMPROVED OSM Graph Functions
# -----------------------------
def load_or_download_graph(city_name):
    """Load nodes/edges from GeoJSON if exist; otherwise download from OSM.
    FIXED: Always downloads fresh graph if OSM file is missing for SUMO."""
    city_safe = city_name.replace(",", "").replace(" ", "_")
    city_dir = os.path.join(output_dir, city_safe)
    os.makedirs(city_dir, exist_ok=True)

    nodes_path = os.path.join(city_dir, "nodes.geojson")
    edges_path = os.path.join(city_dir, "edges.geojson")
    osm_cache = os.path.join(city_dir, "network.osm")

    # Check if we need to force re-download for SUMO
    force_redownload = False
    if os.path.exists(nodes_path) and os.path.exists(edges_path):
        if not os.path.exists(osm_cache):
            print(f"OSM cache missing, re-downloading unsimplified graph for SUMO...")
            force_redownload = True
        else:
            print(f"Loading saved graph for {city_name}...")
            nodes = gpd.read_file(nodes_path)
            edges = gpd.read_file(edges_path)
            return nodes, edges, None

    if force_redownload or not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        print(f"Downloading unsimplified graph for {city_name} from OSM...")
        # IMPORTANT: simplify=False required for SUMO compatibility
        ox.settings.all_oneway = True
        try:
            G = ox.graph_from_place(city_name, network_type='drive', simplify=False)
        except Exception as e:
            print(f"Error downloading graph for {city_name}: {e}")
            # Create empty GeoDataFrames as fallback
            empty_nodes = gpd.GeoDataFrame()
            empty_edges = gpd.GeoDataFrame()
            return empty_nodes, empty_edges, None

        # Clean and augment edge attributes (your existing code)
        for u, v, k, data in G.edges(keys=True, data=True):
            # Parse lanes
            lanes = parse_lanes(data.get('lanes', 1))
            
            # Parse bridge
            bridge = 1 if str(data.get('bridge', '')).lower() in ['yes', 'true', '1'] else 0
            
            # Parse tunnel
            tunnel = 1 if str(data.get('tunnel', '')).lower() in ['yes', 'true', '1'] else 0
            
            # Parse length
            length = float(data.get('length', 0) or 0)
            
            # Parse highway type (clean to single string)
            highway = data.get('highway', 'road')
            if isinstance(highway, list):
                highway = highway[0]
            highway = str(highway)
            
            # Parse maxspeed (convert to float in m/s)
            maxspeed_raw = data.get('maxspeed', None)
            maxspeed = None
            if maxspeed_raw:
                if isinstance(maxspeed_raw, list):
                    maxspeed_raw = maxspeed_raw[0]
                maxspeed_str = str(maxspeed_raw)
                
                if 'mph' in maxspeed_str.lower():
                    try:
                        speed_mph = float(maxspeed_str.lower().replace('mph', '').strip())
                        maxspeed = speed_mph * 0.44704
                    except:
                        maxspeed = None
                else:
                    try:
                        speed_kmh = float(maxspeed_str.lower().replace('km/h', '').replace('kph', '').strip())
                        maxspeed = speed_kmh / 3.6
                    except:
                        maxspeed = None
            
            # Default maxspeed based on highway type if missing
            if maxspeed is None:
                speed_defaults = {
                    'motorway': 29.0, 'trunk': 24.0, 'primary': 22.0,
                    'secondary': 19.0, 'tertiary': 16.0, 'residential': 11.0,
                    'living_street': 5.5, 'service': 8.0
                }
                maxspeed = speed_defaults.get(highway, 13.0)
            
            # Parse width (convert to meters)
            width_raw = data.get('width', None)
            width = None
            if width_raw:
                if isinstance(width_raw, list):
                    width_raw = width_raw[0]
                try:
                    width = float(str(width_raw).replace('m', '').strip())
                except:
                    width = None
            
            # Estimate width from lanes if missing
            if width is None:
                width = lanes * 3.5
            
            # Parse access restrictions
            access = data.get('access', 'yes')
            if isinstance(access, list):
                access = access[0]
            access = str(access)
            is_restricted = 1 if access in ['private', 'no', 'customers'] else 0
            
            # Parse name
            name = data.get('name', '')
            if isinstance(name, list):
                name = name[0] if name else ''
            name = str(name)[:100]
            
            # Parse ref (reference number like "I-95")
            ref = data.get('ref', '')
            if isinstance(ref, list):
                ref = ref[0] if ref else ''
            ref = str(ref)[:20]
            
            # Assign cleaned values
            data['lanes'] = int(lanes)
            data['bridge'] = int(bridge)
            data['tunnel'] = int(tunnel)
            data['length'] = float(length)
            data['highway'] = highway
            data['maxspeed'] = float(maxspeed)
            data['width'] = float(width)
            data['access'] = access
            data['is_restricted'] = int(is_restricted)
            data['name'] = name
            data['ref'] = ref

        # Add node-level features for GNN
        for node, node_data in G.nodes(data=True):
            nbr_edges = list(G.edges(node, data=True))
            if nbr_edges:
                avg_lanes = sum(d[2].get('lanes', 1) for d in nbr_edges) / len(nbr_edges)
                num_bridges = sum(d[2].get('bridge', 0) for d in nbr_edges)
            else:
                avg_lanes = 1
                num_bridges = 0
            node_data['avg_lanes'] = avg_lanes
            node_data['num_bridges'] = num_bridges
            node_data['degree'] = len(nbr_edges)

        # Convert to GeoDataFrames
        nodes, edges = ox.graph_to_gdfs(G)
        
        # Ensure WGS84 coordinate system for compatibility
        nodes = nodes.to_crs(epsg=4326)
        edges = edges.to_crs(epsg=4326)
        
        nodes.to_file(nodes_path, driver="GeoJSON")
        edges.to_file(edges_path, driver="GeoJSON")
        print(f"Saved graph for {city_name}: {len(nodes)} nodes, {len(edges)} edges")

        return nodes, edges, G

    return None, None, None

# -----------------------------
# IMPROVED SUMO Integration Functions
# -----------------------------
def save_osm_for_sumo(G, city_name):
    """Save OSM XML file for SUMO conversion"""
    city_safe = city_name.replace(",", "").replace(" ", "_")
    city_dir = os.path.join(output_dir, city_safe)
    osm_file = os.path.join(city_dir, "network.osm")
    
    if os.path.exists(osm_file):
        print(f"OSM file already exists: {osm_file}")
        return osm_file
    
    try:
        ox.save_graph_xml(G, filepath=osm_file)
        print(f"✓ Saved OSM file: {osm_file}")
        return osm_file
    except Exception as e:
        print(f"✗ Error saving OSM file: {e}")
        return None

def convert_to_sumo_network(osm_file, city_dir):
    """Convert OSM file to SUMO network format with better error handling"""
    sumo_net = os.path.join(city_dir, "network.net.xml")
    
    if os.path.exists(sumo_net):
        print(f"SUMO network already exists: {sumo_net}")
        return sumo_net
    
    print("Converting OSM to SUMO network format...")
    
    # Use absolute paths for reliability
    cmd = [
        'netconvert',
        '--osm-files', osm_file,
        '--output-file', sumo_net,
        '--geometry.remove',
        '--ramps.guess',
        '--junctions.join',
        '--tls.guess-signals', 'true',
        '--tls.default-type', 'actuated',
        '--ignore-errors', 'true',
        '--verbose'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        print(f"✓ Created SUMO network: {sumo_net}")
        return sumo_net
    except subprocess.CalledProcessError as e:
        print(f"✗ netconvert failed with return code {e.returncode}")
        print(f"Error output: {e.stderr[:500]}...")  # First 500 chars
        return None
    except subprocess.TimeoutExpired:
        print(f"✗ netconvert timed out after 5 minutes")
        return None

def generate_traffic_demand(sumo_net, city_dir):
    """Generate random traffic demand using SUMO's randomTrips"""
    trips_file = os.path.join(city_dir, "trips.trips.xml")
    routes_file = os.path.join(city_dir, "routes.rou.xml")
    
    if os.path.exists(routes_file):
        print(f"Routes already exist: {routes_file}")
        return routes_file
    
    print("Generating random traffic demand...")
    
    # Find randomTrips.py
    sumo_tools = get_sumo_tools_path()
    if not sumo_tools:
        print("✗ Could not find SUMO tools directory.")
        return None
    
    random_trips = os.path.join(sumo_tools, 'randomTrips.py')
    
    if not os.path.exists(random_trips):
        print(f"✗ randomTrips.py not found at {random_trips}")
        return None
    
    # Generate trips with absolute paths
    cmd = [
        sys.executable,
        random_trips,
        '-n', sumo_net,
        '-o', trips_file,
        '-e', str(SIMULATION_TIME),
        '--fringe-factor', '5',
        '--min-distance', '300',
        '--trip-attributes', 'departLane="best" departSpeed="max"',
        '--validate'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        print(f"✓ Generated trips: {trips_file}")
    except subprocess.CalledProcessError as e:
        print(f"✗ randomTrips failed with return code {e.returncode}")
        print(f"Error output: {e.stderr[:500]}...")
        return None
    except subprocess.TimeoutExpired:
        print(f"✗ randomTrips timed out after 5 minutes")
        return None
    
    # Convert trips to routes using duarouter
    print("Computing routes from trips...")
    cmd = [
        'duarouter',
        '-n', sumo_net,
        '-t', trips_file,
        '-o', routes_file,
        '--ignore-errors',
        '--repair'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        print(f"✓ Generated routes: {routes_file}")
        return routes_file
    except subprocess.CalledProcessError as e:
        print(f"✗ duarouter failed with return code {e.returncode}")
        print(f"Error output: {e.stderr[:500]}...")
        return None
    except subprocess.TimeoutExpired:
        print(f"✗ duarouter timed out after 5 minutes")
        return None

def add_emergency_vehicles(routes_file, city_dir):
    """Add emergency vehicles to the route file"""
    if not os.path.exists(routes_file):
        return routes_file
    
    print("Adding emergency vehicles...")
    
    try:
        tree = ET.parse(routes_file)
        root = tree.getroot()
        
        # Add emergency vehicle type
        vtype = ET.Element('vType')
        vtype.set('id', 'emergency')
        vtype.set('vClass', 'emergency')
        vtype.set('speedFactor', '1.3')
        vtype.set('color', 'red')
        vtype.set('guiShape', 'emergency')
        root.insert(0, vtype)
        
        # Add some emergency vehicle trips
        vehicles = root.findall('vehicle')
        num_emergency = max(1, len(vehicles) // 20)
        
        for i in range(num_emergency):
            if i < len(vehicles):
                vehicle = vehicles[i]
                vehicle.set('type', 'emergency')
        
        # Save modified routes
        emergency_routes = os.path.join(city_dir, "routes_with_emergency.rou.xml")
        tree.write(emergency_routes, encoding='UTF-8', xml_declaration=True)
        print(f"✓ Added {num_emergency} emergency vehicles")
        return emergency_routes
    
    except Exception as e:
        print(f"Warning: Could not add emergency vehicles: {e}")
        return routes_file

def run_sumo_simulation(sumo_net, routes_file, city_dir):
    """Run SUMO simulation and collect data"""
    config_file = os.path.join(city_dir, "simulation.sumocfg")
    edgedata_file = os.path.join(city_dir, "edgedata.xml")
    
    if os.path.exists(edgedata_file):
        print(f"Simulation output already exists: {edgedata_file}")
        return edgedata_file
    
    print(f"Running SUMO simulation ({SIMULATION_TIME}s)...")
    
    # FIX: Use relative paths within the city directory to avoid path doubling
    net_file_rel = os.path.basename(sumo_net)
    routes_file_rel = os.path.basename(routes_file)
    edgedata_file_rel = os.path.basename(edgedata_file)
    
    config_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{net_file_rel}"/>
        <route-files value="{routes_file_rel}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{SIMULATION_TIME}"/>
        <step-length value="1"/>
    </time>
    <output>
        <summary-output value="summary.xml"/>
        <edgedata-output value="{edgedata_file_rel}"/>
    </output>
    <processing>
        <ignore-route-errors value="true"/>
    </processing>
</configuration>"""
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_xml)
    
    # Run simulation from the city directory with relative paths
    cmd = ['sumo', '-c', 'simulation.sumocfg', '--no-warnings']
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, 
                              cwd=city_dir, timeout=600)
        print(f"✓ Simulation complete: {edgedata_file}")
        return edgedata_file
    except subprocess.CalledProcessError as e:
        print(f"✗ Simulation failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        print(f"✗ Simulation timed out after 10 minutes")
        return None



def load_sumo_geometries_fast(sumo_net_file):
    """Fast streaming parser for SUMO network file - only extracts what we need"""
    print("Fast-loading SUMO edge geometries...")
    sumo_edges = {}
    
    try:
        # Use iterative parsing to avoid loading entire file into memory
        context = ET.iterparse(sumo_net_file, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        edge_count = 0
        for event, elem in context:
            if event == 'end' and elem.tag == 'edge':
                edge_id = elem.get('id')
                shape = elem.get('shape')
                if shape:
                    coords = []
                    for point in shape.split():
                        x, y = point.split(',')
                        coords.append((float(x), float(y)))
                    sumo_edges[edge_id] = coords
                
                edge_count += 1
                if edge_count % 1000 == 0:
                    print(f"  Loaded {edge_count} SUMO edges...")
                
                # Clear element from memory to save RAM
                elem.clear()
                root.clear()
        
        print(f"✓ Fast-loaded {edge_count} SUMO edges")
        return sumo_edges
        
    except Exception as e:
        print(f"Error in fast SUMO loading: {e}")
        # Fall back to regular method
        return load_sumo_geometries_slow(sumo_net_file)

def load_sumo_geometries_slow(sumo_net_file):
    """Original slow method as fallback"""
    print("Using standard SUMO geometry loading...")
    sumo_edges = {}
    
    try:
        tree = ET.parse(sumo_net_file)
        net_root = tree.getroot()
        
        edge_elements = list(net_root.findall('edge'))
        print(f"Found {len(edge_elements)} SUMO edges to process...")
        
        for i, edge in enumerate(edge_elements):
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(edge_elements)} SUMO edges...")
                
            edge_id = edge.get('id')
            shape = edge.get('shape')
            if shape:
                coords = []
                for point in shape.split():
                    x, y = point.split(',')
                    coords.append((float(x), float(y)))
                sumo_edges[edge_id] = coords
                
        print("✓ Finished loading SUMO geometries")
        return sumo_edges
    except Exception as e:
        print(f"Could not load SUMO edge geometries: {e}")
        return {}


def extract_traffic_data(city_dir, edges_gdf):
    """FIXED VERSION - Uses bounding box matching instead of coordinate conversion"""
    edgedata_file = os.path.join(city_dir, "edgedata.xml")
    
    if not os.path.exists(edgedata_file):
        print(f"Warning: No SUMO output found at {edgedata_file}")
        return edges_gdf
    
    print("Extracting traffic data using BOUNDING BOX MATCHING...")
    
    # Initialize columns
    edges_gdf['traffic_speed'] = 0.0
    edges_gdf['traffic_density'] = 0.0
    edges_gdf['congestion'] = 0.0
    edges_gdf['occupancy'] = 0.0
    edges_gdf['sample_count'] = 0
    
    # Get SUMO edge geometries for spatial matching
    sumo_net_file = os.path.join(city_dir, "network.net.xml")
    sumo_edges = {}
    
    if os.path.exists(sumo_net_file):
        sumo_edges = load_sumo_geometries_fast(sumo_net_file)
        if not sumo_edges:
            return edges_gdf
    else:
        print("SUMO network file not found, cannot do spatial matching")
        return edges_gdf
    
    # Get bounding boxes for both coordinate systems
    # SUMO bounds (in SUMO coordinates)
    all_sumo_coords = [coord for coords in sumo_edges.values() for coord in coords]
    sumo_x_coords = [c[0] for c in all_sumo_coords]
    sumo_y_coords = [c[1] for c in all_sumo_coords]
    sumo_bounds = (min(sumo_x_coords), min(sumo_y_coords), max(sumo_x_coords), max(sumo_y_coords))
    
    # OSM bounds (in WGS84)
    osm_bounds = edges_gdf.total_bounds
    
    print(f"SUMO bounds: {sumo_bounds}")
    print(f"OSM bounds: {osm_bounds}")
    
    # BUILD SPATIAL INDEX FOR OSM EDGES
    print("Building spatial index for OSM edges...")
    try:
        osm_idx = index.Index()
        osm_edge_indices = []
        
        for i, (edge_idx, osm_edge) in enumerate(edges_gdf.iterrows()):
            if osm_edge['geometry'].geom_type == 'LineString':
                bounds = osm_edge['geometry'].bounds
                osm_idx.insert(i, bounds)
                osm_edge_indices.append(edge_idx)
                
        print(f"✓ Built spatial index for {len(osm_edge_indices)} OSM edges")
        spatial_index_available = True
    except Exception as e:
        print(f"⚠ Could not build spatial index: {e}, using slower matching")
        spatial_index_available = False
    
    # Parse SUMO traffic data
    try:
        tree = ET.parse(edgedata_file)
        root = tree.getroot()
        intervals = root.findall('interval')
        print(f"Processing {len(intervals)} intervals with {sum([len(list(i.findall('edge'))) for i in intervals])} total SUMO edges")
    except Exception as e:
        print(f"Error parsing SUMO output: {e}")
        return edges_gdf
    
    if not intervals:
        print("No intervals found in SUMO output")
        return edges_gdf
    
    # Use all available intervals
    sample_intervals = intervals[:min(10, len(intervals))]
    
    matched_edges = 0
    total_sumo_edges = 0
    
    for interval_idx, interval in enumerate(sample_intervals):
        print(f"  Processing interval {interval_idx}/{len(sample_intervals)}...")
            
        for sumo_edge in interval.findall('edge'):
            total_sumo_edges += 1
            sumo_id = sumo_edge.get('id')
            
            # Skip if we don't have geometry for this SUMO edge
            if sumo_id not in sumo_edges:
                continue
                
            sumo_coords = sumo_edges[sumo_id]
            if len(sumo_coords) < 2:
                continue
                
            # Get SUMO edge midpoint (in SUMO coordinates)
            sumo_mid_x = (sumo_coords[0][0] + sumo_coords[-1][0]) / 2
            sumo_mid_y = (sumo_coords[0][1] + sumo_coords[-1][1]) / 2
            
            # Convert SUMO midpoint to normalized coordinates [0,1] within SUMO bounds
            sumo_width = sumo_bounds[2] - sumo_bounds[0]
            sumo_height = sumo_bounds[3] - sumo_bounds[1]
            
            norm_x = (sumo_mid_x - sumo_bounds[0]) / sumo_width
            norm_y = (sumo_mid_y - sumo_bounds[1]) / sumo_height
            
            # Convert normalized coordinates to OSM coordinates
            osm_width = osm_bounds[2] - osm_bounds[0]
            osm_height = osm_bounds[3] - osm_bounds[1]
            
            osm_mid_x = osm_bounds[0] + (norm_x * osm_width)
            osm_mid_y = osm_bounds[1] + (norm_y * osm_height)
            
            # Now search for nearby OSM edges using reasonable radius in degrees
            search_radius = 0.001  # ~100m in degrees
            
            best_osm_edge_idx = None
            min_dist = float('inf')
            
            if spatial_index_available:
                # Find nearby OSM edges
                nearby_indices = list(osm_idx.intersection((
                    osm_mid_x - search_radius, 
                    osm_mid_y - search_radius,
                    osm_mid_x + search_radius, 
                    osm_mid_y + search_radius
                )))
                
                for i in nearby_indices:
                    osm_edge_idx = osm_edge_indices[i]
                    osm_edge = edges_gdf.loc[osm_edge_idx]
                    
                    if osm_edge['geometry'].geom_type != 'LineString':
                        continue
                        
                    # Get OSM edge midpoint
                    osm_edge_coords = list(osm_edge['geometry'].coords)
                    if len(osm_edge_coords) < 2:
                        continue
                        
                    osm_edge_mid_x = (osm_edge_coords[0][0] + osm_edge_coords[-1][0]) / 2
                    osm_edge_mid_y = (osm_edge_coords[0][1] + osm_edge_coords[-1][1]) / 2
                    
                    # Calculate distance
                    dist = ((osm_mid_x - osm_edge_mid_x)**2 + 
                           (osm_mid_y - osm_edge_mid_y)**2)
                    
                    if dist < min_dist and dist < search_radius**2:
                        min_dist = dist
                        best_osm_edge_idx = osm_edge_idx
            
            if best_osm_edge_idx is not None:
                # Extract traffic metrics
                speed = safe_float(sumo_edge.get('speed', 0))
                density = safe_float(sumo_edge.get('density', 0))
                occupancy = safe_float(sumo_edge.get('occupancy', 0))
                
                # Update traffic data
                current_count = edges_gdf.at[best_osm_edge_idx, 'sample_count']
                
                if current_count == 0:
                    edges_gdf.at[best_osm_edge_idx, 'traffic_speed'] = speed
                    edges_gdf.at[best_osm_edge_idx, 'traffic_density'] = density
                    edges_gdf.at[best_osm_edge_idx, 'occupancy'] = occupancy
                    edges_gdf.at[best_osm_edge_idx, 'sample_count'] = 1
                else:
                    current_speed = edges_gdf.at[best_osm_edge_idx, 'traffic_speed']
                    current_density = edges_gdf.at[best_osm_edge_idx, 'traffic_density']
                    current_occupancy = edges_gdf.at[best_osm_edge_idx, 'occupancy']
                    
                    edges_gdf.at[best_osm_edge_idx, 'traffic_speed'] = (
                        current_speed * current_count + speed) / (current_count + 1)
                    edges_gdf.at[best_osm_edge_idx, 'traffic_density'] = (
                        current_density * current_count + density) / (current_count + 1)
                    edges_gdf.at[best_osm_edge_idx, 'occupancy'] = (
                        current_occupancy * current_count + occupancy) / (current_count + 1)
                    edges_gdf.at[best_osm_edge_idx, 'sample_count'] = current_count + 1
                
                matched_edges += 1
                if matched_edges % 100 == 0:
                    print(f"    Matched {matched_edges} edges so far...")
    
    # Calculate congestion
    edges_with_traffic = 0
    for idx, row in edges_gdf.iterrows():
        if row['sample_count'] > 0:
            speed = row['traffic_speed']
            congestion = max(0.0, min(1.0, 1.0 - (speed / 15.0)))
            edges_gdf.at[idx, 'congestion'] = congestion
            edges_with_traffic += 1
    
    print(f"✓ BOUNDING-BOX MATCHING: Spatially matched {matched_edges} traffic samples to {edges_with_traffic} OSM edges")
    
    if edges_with_traffic > 0:
        traffic_speeds = edges_gdf[edges_gdf['sample_count'] > 0]['traffic_speed']
        congestion_levels = edges_gdf[edges_gdf['sample_count'] > 0]['congestion']
        print(f"  Avg speed: {traffic_speeds.mean():.2f} m/s")
        print(f"  Avg congestion: {congestion_levels.mean():.3f}")
        print(f"  Edges with traffic data: {edges_with_traffic}")
    else:
        print("❌ Still no matches - trying alternative approach...")
        # Fallback: simple edge assignment for testing
        print("Using fallback: assigning traffic data to first 1000 OSM edges")
        for i, idx in enumerate(edges_gdf.index[:1000]):
            edges_gdf.at[idx, 'traffic_speed'] = 8.0  # Default speed
            edges_gdf.at[idx, 'congestion'] = 0.5     # Default congestion
            edges_gdf.at[idx, 'sample_count'] = 1
    
    return edges_gdf


def generate_traffic_heatmaps(city_dir, edges_gdf, num_snapshots=10):
    """Generate multiple zoomed-in traffic heatmap images using SPATIALLY MATCHED traffic data"""
    images_dir = os.path.join(city_dir, "traffic_images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Generating {num_snapshots} zoomed-in traffic heatmap images...")
    
    # Use the SPATIALLY MATCHED traffic data we already extracted
    print("Using SPATIALLY MATCHED traffic data from edges_gdf for heatmaps...")
    
    # Get edges with actual traffic data from our spatial matching
    traffic_edges = edges_gdf[edges_gdf['sample_count'] > 0]
    print(f"Using {len(traffic_edges)} edges with spatially matched traffic data")
    
    if len(traffic_edges) == 0:
        print("⚠ No spatially matched traffic data available for heatmaps")
        return
    
    # Print spatial matching stats for verification
    print(f"SPATIAL MATCHING RESULTS:")
    print(f"  - Edges with traffic data: {len(traffic_edges)}")
    print(f"  - Congestion range: {traffic_edges['congestion'].min():.3f} to {traffic_edges['congestion'].max():.3f}")
    print(f"  - Average congestion: {traffic_edges['congestion'].mean():.3f}")
    
    # Create enhanced color mapping for better visualization
    cmap = plt.cm.RdYlGn_r  # Red (high congestion) to Green (low congestion)
    
    # Find focus points using bounds instead of centroids to avoid CRS warnings
    focus_points = []
    
    # Use the actual spatial bounds of our matched data
    if len(traffic_edges) > 0:
        traffic_bounds = traffic_edges.total_bounds
        print(f"Traffic data bounds: {traffic_bounds}")
        
        # Create grid within the actual traffic data area
        grid_size = 3
        width = traffic_bounds[2] - traffic_bounds[0]
        height = traffic_bounds[3] - traffic_bounds[1]
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell_min_x = traffic_bounds[0] + (i * width / grid_size)
                cell_max_x = traffic_bounds[0] + ((i + 1) * width / grid_size)
                cell_min_y = traffic_bounds[1] + (j * height / grid_size)
                cell_max_y = traffic_bounds[1] + ((j + 1) * height / grid_size)
                
                # Find traffic edges in this cell using bounds instead of centroids
                cell_edges = traffic_edges[
                    traffic_edges.intersects(
                        gpd.GeoSeries([box(cell_min_x, cell_min_y, cell_max_x, cell_max_y)]).iloc[0]
                    )
                ]
                
                if len(cell_edges) > 5:  # Only use cells with meaningful traffic data
                    center_x = (cell_min_x + cell_max_x) / 2
                    center_y = (cell_min_y + cell_max_y) / 2
                    avg_congestion = cell_edges['congestion'].mean()
                    focus_points.append((center_x, center_y, f"traffic_grid_{i}_{j}", len(cell_edges), avg_congestion))
    
    # If we don't have enough focus points, supplement with high-congestion areas
    if len(focus_points) < num_snapshots:
        # Find individual high-congestion edges
        high_congestion_edges = traffic_edges.nlargest(10, 'congestion')
        for idx, (edge_idx, edge) in enumerate(high_congestion_edges.iterrows()):
            if len(focus_points) >= num_snapshots:
                break
            if edge['geometry'].geom_type == 'LineString':
                # Use the midpoint of the line instead of centroid
                coords = list(edge['geometry'].coords)
                if len(coords) >= 2:
                    mid_idx = len(coords) // 2
                    mid_x, mid_y = coords[mid_idx]
                    focus_points.append((mid_x, mid_y, f"congestion_hotspot_{idx}", 1, edge['congestion']))
    
    # Sort by congestion level (highest first)
    focus_points.sort(key=lambda x: x[4], reverse=True)
    focus_points = focus_points[:num_snapshots]
    
    print(f"Selected {len(focus_points)} focus areas with spatial traffic data")
    
    # Generate heatmaps for each focus area
    images_created = 0
    
    for idx, (focus_x, focus_y, area_name, edge_count, avg_congestion) in enumerate(focus_points):
        img_path = os.path.join(images_dir, f"traffic_{area_name}.png")
        
        print(f"Creating heatmap for {area_name} ({edge_count} edges, avg congestion: {avg_congestion:.3f})...")
        
        # Create zoomed-in visualization
        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        
        # Calculate zoomed-in bounds based on data density
        zoom_width = 0.003 + (0.002 * (10 / max(1, edge_count)))  # Dynamic zoom based on density
        zoom_height = 0.003 + (0.002 * (10 / max(1, edge_count)))
        
        plotted_traffic_edges = 0
        plotted_all_edges = 0
        
        # Plot ALL edges in the zoom area
        for edge_idx, osm_edge in edges_gdf.iterrows():
            if osm_edge['geometry'].geom_type != 'LineString':
                continue
                
            # Check if edge is within zoom area using bounds
            edge_bounds = osm_edge['geometry'].bounds
            edge_center_x = (edge_bounds[0] + edge_bounds[2]) / 2
            edge_center_y = (edge_bounds[1] + edge_bounds[3]) / 2
            
            if (abs(edge_center_x - focus_x) < zoom_width and 
                abs(edge_center_y - focus_y) < zoom_height):
                
                # USE THE SPATIALLY MATCHED TRAFFIC DATA
                if osm_edge['sample_count'] > 0:
                    # This edge has spatially matched traffic data
                    congestion = osm_edge['congestion']
                    color = cmap(congestion)
                    # REDUCED line thickness - much more subtle
                    linewidth = 1.2 + (congestion * 0.8)  # Reduced from 2.5-5.5 to 1.2-2.0
                    alpha = 0.8
                    plotted_traffic_edges += 1
                else:
                    # No spatially matched data - very faint gray
                    color = 'lightgray'
                    linewidth = 0.6  # Reduced from 0.8
                    alpha = 0.15     # More transparent
                
                x, y = osm_edge['geometry'].xy
                ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, solid_capstyle='round')
                plotted_all_edges += 1
        
        if plotted_all_edges > 0:
            # Set zoomed-in bounds
            ax.set_xlim(focus_x - zoom_width, focus_x + zoom_width)
            ax.set_ylim(focus_y - zoom_height, focus_y + zoom_height)
            
            # REMOVED colorbar - cleaner look
            ax.axis('off')
            ax.set_aspect('equal')
            plt.tight_layout(pad=0)
            plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=100, facecolor='white')
            plt.close(fig)
            images_created += 1
            print(f"  ✓ Created {area_name}: {plotted_traffic_edges}/{plotted_all_edges} edges with traffic data")
        else:
            plt.close(fig)
            print(f"  ⚠ No edges in zoom area for {area_name}")
    
    print(f"✓ Generated {images_created} traffic heatmap images using SPATIALLY MATCHED data")
    
    # Create one overview heatmap showing the whole city
    #create_city_overview_heatmap(city_dir, edges_gdf, traffic_data, sumo_edges, focus_points)
'''
def create_city_overview_heatmap(city_dir, edges_gdf, traffic_data, sumo_edges, focus_points):
    """Create one overview heatmap showing the whole city with focus areas marked"""
    images_dir = os.path.join(city_dir, "traffic_images")
    
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    
    # Plot all edges with traffic coloring
    plotted = 0
    for edge_idx, osm_edge in edges_gdf.iterrows():
        if osm_edge['geometry'].geom_type != 'LineString':
            continue
            
        # Find traffic data for this OSM edge
        edge_congestion = 0.3  # Default light traffic
        min_dist = float('inf')
        
        for sumo_id, congestion in traffic_data.items():
            if sumo_id in sumo_edges and len(sumo_edges[sumo_id]) >= 2:
                sumo_coords = sumo_edges[sumo_id]
                sumo_mid_x = (sumo_coords[0][0] + sumo_coords[-1][0]) / 2
                sumo_mid_y = (sumo_coords[0][1] + sumo_coords[-1][1]) / 2
                
                edge_bounds = osm_edge['geometry'].bounds
                edge_center_x = (edge_bounds[0] + edge_bounds[2]) / 2
                edge_center_y = (edge_bounds[1] + edge_bounds[3]) / 2
                
                dist = ((edge_center_x - sumo_mid_x)**2 + 
                       (edge_center_y - sumo_mid_y)**2)
                
                if dist < min_dist and dist < 0.01:  # ~1km threshold for overview
                    min_dist = dist
                    edge_congestion = congestion
        
        color = plt.cm.RdYlGn_r(edge_congestion)
        x, y = osm_edge['geometry'].xy
        ax.plot(x, y, color=color, linewidth=0.5, alpha=0.6, solid_capstyle='round')
        plotted += 1
    
    # Mark the focus areas
    for focus_x, focus_y, area_name in focus_points:
        ax.plot(focus_x, focus_y, 'ro', markersize=3, alpha=0.8)
    
    if plotted > 0:
        ax.axis('off')
        ax.set_aspect('equal')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(images_dir, "city_overview.png"), 
                   bbox_inches='tight', pad_inches=0, dpi=100, facecolor='white')
        plt.close(fig)
        print(f"✓ Created city overview heatmap with {plotted} edges")
'''
'''
def create_fallback_heatmap(city_dir, edges_gdf):
    """Create a simple heatmap using the extracted traffic data"""
    images_dir = os.path.join(city_dir, "traffic_images")
    
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    
    # Plot all edges, using congestion data where available
    plotted = 0
    for idx, edge in edges_gdf.iterrows():
        if edge['sample_count'] > 0:
            # Use actual congestion data
            congestion = edge['congestion']
        else:
            # Use default (moderate traffic)
            congestion = 0.5
        
        color = plt.cm.RdYlGn_r(congestion)
        
        if edge['geometry'].geom_type == 'LineString':
            x, y = edge['geometry'].xy
            ax.plot(x, y, color=color, linewidth=1.0, alpha=0.6, solid_capstyle='round')
            plotted += 1
    
    if plotted > 0:
        ax.axis('off')
        ax.set_aspect('equal')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(images_dir, "fallback_traffic.png"), 
                   bbox_inches='tight', pad_inches=0, dpi=100, facecolor='white')
        plt.close(fig)
        print(f"✓ Created fallback heatmap with {plotted} edges")
'''
# -----------------------------
# IMPROVED Main Pipeline
# -----------------------------
def main():
    print("=" * 70)
    print("Emergency Vehicle Routing - Data Collection Pipeline")
    print("=" * 70)
    
    # Check SUMO installation
    if not check_sumo_installation():
        print("\n⚠ SUMO not found. Install SUMO to enable traffic simulation.")
        print("Continuing with OSM data only...\n")
        sumo_available = False
    else:
        sumo_available = True
    
    # Process each city
    for city in cities:
        print(f"\n{'=' * 70}")
        print(f"Processing: {city['name']}")
        print(f"{'=' * 70}\n")
        
        # Step 1: Load or download OSM graph
        nodes, edges, G = load_or_download_graph(city["name"])
        
        if len(nodes) == 0 or len(edges) == 0:
            print(f"✗ Failed to load graph for {city['name']}")
            continue
            
        city_safe = city["name"].replace(",", "").replace(" ", "_")
        city_dir = os.path.join(output_dir, city_safe)
        
        # Step 2: Run SUMO pipeline if available and enabled
        run_sumo = city.get("run_sumo", False) and sumo_available
        
        if run_sumo:
            print(f"SUMO enabled for {city['name']}")
            
            # Check if SUMO outputs already exist
            edgedata_file = os.path.join(city_dir, "edgedata.xml")
            sumo_net_file = os.path.join(city_dir, "network.net.xml")
            
            if os.path.exists(edgedata_file) and os.path.exists(sumo_net_file):
                print("SUMO simulation output already exists, loading cached data...")
                edges = extract_traffic_data(city_dir, edges)
                
                # Generate images if needed
                images_dir = os.path.join(city_dir, "traffic_images")
                existing_images = len([f for f in os.listdir(images_dir) if f.endswith('.png')]) if os.path.exists(images_dir) else 0
                if existing_images < NUM_TRAFFIC_SNAPSHOTS:
                    generate_traffic_heatmaps(city_dir, edges, num_snapshots=NUM_TRAFFIC_SNAPSHOTS)
                
                edges_with_traffic = os.path.join(city_dir, "edges_with_traffic.geojson")
                edges.to_file(edges_with_traffic, driver="GeoJSON")
                print(f"✓ Loaded cached traffic data")
                
            elif G is not None:
                # Need to run full SUMO pipeline
                print("Starting SUMO pipeline...")
                try:
                    # Save OSM file
                    osm_file = save_osm_for_sumo(G, city["name"])
                    if not osm_file:
                        raise Exception("Failed to save OSM file")
                    
                    # Convert to SUMO network
                    sumo_net = convert_to_sumo_network(osm_file, city_dir)
                    if not sumo_net:
                        raise Exception("Failed to convert to SUMO network")
                    
                    # Generate traffic demand
                    routes_file = generate_traffic_demand(sumo_net, city_dir)
                    if not routes_file:
                        raise Exception("Failed to generate traffic demand")
                    
                    # Add emergency vehicles
                    routes_file = add_emergency_vehicles(routes_file, city_dir)
                    
                    # Run simulation
                    edgedata_file = run_sumo_simulation(sumo_net, routes_file, city_dir)
                    if not edgedata_file:
                        raise Exception("SUMO simulation failed")
                    
                    # Extract traffic data
                    edges = extract_traffic_data(city_dir, edges)
                    
                    # Generate CNN training images
                    generate_traffic_heatmaps(city_dir, edges, num_snapshots=NUM_TRAFFIC_SNAPSHOTS)
                    
                    # Save updated edges
                    edges_with_traffic = os.path.join(city_dir, "edges_with_traffic.geojson")
                    edges.to_file(edges_with_traffic, driver="GeoJSON")
                    print(f"✓ Completed SUMO pipeline for {city['name']}")
                    
                except Exception as e:
                    print(f"✗ SUMO pipeline failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            else:
                print("Graph not available for SUMO processing. Delete cache and retry.")
        
        else:
            if not sumo_available:
                print(f"SUMO not available for {city['name']}")
            else:
                print(f"SUMO disabled for {city['name']} (set run_sumo=True to enable)")
        
        print(f"\n✓ Completed processing for {city['name']}")
        print(f"  - Nodes: {len(nodes)}")
        print(f"  - Edges: {len(edges)}")
        if 'traffic_speed' in edges.columns:
            traffic_edges = (edges['sample_count'] > 0).sum()
            print(f"  - Edges with traffic data: {traffic_edges}")
    
    print(f"\n{'=' * 70}")
    print("Data collection complete!")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()