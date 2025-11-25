"""
Inspect and validate the generated traffic data
Useful for understanding your dataset before training
"""

import os
import glob
import geopandas as gpd
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json


def inspect_images(images_dir):
    """Inspect traffic heatmap images"""
    print("\n" + "="*70)
    print("IMAGE INSPECTION")
    print("="*70)
    
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    
    if not image_files:
        print("❌ No images found!")
        return
    
    print(f"\nTotal images: {len(image_files)}")
    
    # Check image properties
    for i, img_path in enumerate(image_files[:3]):  # Show first 3
        img = Image.open(img_path)
        print(f"\n  Image {i+1}: {os.path.basename(img_path)}")
        print(f"    - Size: {img.size}")
        print(f"    - Mode: {img.mode}")
        print(f"    - Format: {img.format}")
        
        # Get pixel statistics
        img_array = np.array(img)
        print(f"    - Min pixel: {img_array.min()}, Max pixel: {img_array.max()}")
        print(f"    - Mean pixel: {img_array.mean():.2f}")
    
    print(f"\n✓ Found {len(image_files)} traffic heatmap images")
    return image_files


def inspect_edges(edges_file):
    """Inspect edge data with traffic attributes"""
    print("\n" + "="*70)
    print("EDGE DATA INSPECTION")
    print("="*70)
    
    edges_gdf = gpd.read_file(edges_file)
    
    print(f"\nTotal edges: {len(edges_gdf)}")
    print(f"CRS: {edges_gdf.crs}")
    print(f"\nColumns: {list(edges_gdf.columns)}")
    
    # Check traffic data columns
    traffic_cols = ['traffic_speed', 'traffic_density', 'occupancy', 'congestion', 'sample_count']
    
    print(f"\n--- Traffic Data Statistics ---")
    for col in traffic_cols:
        if col in edges_gdf.columns:
            data = edges_gdf[col]
            non_zero = (data > 0).sum()
            print(f"\n  {col}:")
            print(f"    - Non-zero samples: {non_zero}/{len(data)}")
            print(f"    - Min: {data.min():.4f}, Max: {data.max():.4f}")
            print(f"    - Mean: {data.mean():.4f}, Std: {data.std():.4f}")
    
    # Check geometric attributes
    geometry_cols = ['lanes', 'bridge', 'tunnel', 'width', 'maxspeed']
    print(f"\n--- Geometric Attributes ---")
    for col in geometry_cols:
        if col in edges_gdf.columns:
            data = edges_gdf[col]
            print(f"\n  {col}: Min={data.min():.2f}, Max={data.max():.2f}, Mean={data.mean():.2f}")
    
    return edges_gdf


def inspect_nodes(nodes_file):
    """Inspect node data"""
    print("\n" + "="*70)
    print("NODE DATA INSPECTION")
    print("="*70)
    
    nodes_gdf = gpd.read_file(nodes_file)
    
    print(f"\nTotal nodes: {len(nodes_gdf)}")
    print(f"Columns: {list(nodes_gdf.columns)}")
    
    # Check node attributes
    attr_cols = [col for col in nodes_gdf.columns if col != 'geometry']
    for col in attr_cols[:5]:  # Show first 5
        if nodes_gdf[col].dtype in ['int64', 'float64']:
            print(f"\n  {col}: Min={nodes_gdf[col].min()}, Max={nodes_gdf[col].max()}")
    
    return nodes_gdf


def validate_dataset(images_dir, edges_file, nodes_file=None):
    """Run complete validation"""
    print("\n" + "="*70)
    print("DATASET VALIDATION")
    print("="*70)
    
    issues = []
    
    # Check images
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not image_files:
        issues.append("❌ No images found in traffic_images/")
    else:
        print(f"✓ {len(image_files)} images found")
    
    # Check edges
    if not os.path.exists(edges_file):
        issues.append(f"❌ Edges file not found: {edges_file}")
    else:
        try:
            edges_gdf = gpd.read_file(edges_file)
            
            # Check for traffic data
            if 'traffic_speed' not in edges_gdf.columns:
                issues.append("⚠ 'traffic_speed' column missing")
            else:
                traffic_edges = (edges_gdf['sample_count'] > 0).sum() if 'sample_count' in edges_gdf.columns else 0
                print(f"✓ Edges file loaded ({len(edges_gdf)} edges, {traffic_edges} with traffic data)")
            
            # Check for required graph features
            required_cols = ['lanes', 'bridge', 'tunnel', 'width']
            missing_cols = [col for col in required_cols if col not in edges_gdf.columns]
            if missing_cols:
                issues.append(f"⚠ Missing columns: {missing_cols}")
            else:
                print(f"✓ All required edge attributes present")
            
        except Exception as e:
            issues.append(f"❌ Error reading edges file: {e}")
    
    # Check nodes
    if nodes_file:
        if not os.path.exists(nodes_file):
            issues.append(f"⚠ Nodes file not found: {nodes_file}")
        else:
            try:
                nodes_gdf = gpd.read_file(nodes_file)
                print(f"✓ Nodes file loaded ({len(nodes_gdf)} nodes)")
            except Exception as e:
                issues.append(f"❌ Error reading nodes file: {e}")
    
    # Summary
    print("\n" + "="*70)
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ ALL CHECKS PASSED - Data ready for training!")
    print("="*70)
    
    return len(issues) == 0


def create_dataset_summary(data_dir):
    """Create a summary of the dataset"""
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    summary = {
        'data_dir': data_dir,
        'num_images': len(glob.glob(os.path.join(data_dir, 'traffic_images', '*.png'))),
        'num_edges': 0,
        'num_nodes': 0,
        'edges_with_traffic': 0,
        'avg_traffic_speed': 0,
        'avg_congestion': 0
    }
    
    edges_file = os.path.join(data_dir, 'edges_with_traffic.geojson')
    if os.path.exists(edges_file):
        edges_gdf = gpd.read_file(edges_file)
        summary['num_edges'] = len(edges_gdf)
        
        if 'sample_count' in edges_gdf.columns:
            summary['edges_with_traffic'] = (edges_gdf['sample_count'] > 0).sum()
        
        if 'traffic_speed' in edges_gdf.columns:
            speeds = edges_gdf[edges_gdf['sample_count'] > 0]['traffic_speed']
            if len(speeds) > 0:
                summary['avg_traffic_speed'] = speeds.mean()
        
        if 'congestion' in edges_gdf.columns:
            congestion = edges_gdf[edges_gdf['sample_count'] > 0]['congestion']
            if len(congestion) > 0:
                summary['avg_congestion'] = congestion.mean()
    
    nodes_file = os.path.join(data_dir, 'nodes.geojson')
    if os.path.exists(nodes_file):
        nodes_gdf = gpd.read_file(nodes_file)
        summary['num_nodes'] = len(nodes_gdf)
    
    print(f"\n  Location: {summary['data_dir']}")
    print(f"  Images: {summary['num_images']}")
    print(f"  Nodes: {summary['num_nodes']}")
    print(f"  Edges: {summary['num_edges']}")
    print(f"  Edges with traffic data: {summary['edges_with_traffic']}")
    print(f"  Average traffic speed: {summary['avg_traffic_speed']:.2f} m/s")
    print(f"  Average congestion: {summary['avg_congestion']:.3f}")
    
    return summary


def main():
    data_dir = "raw_data/Houston_TX_USA"
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    print("\n" + "="*70)
    print("TRAFFIC DATA INSPECTION")
    print("="*70)
    
    # Inspect components
    images_dir = os.path.join(data_dir, "traffic_images")
    edges_file = os.path.join(data_dir, "edges_with_traffic.geojson")
    nodes_file = os.path.join(data_dir, "nodes.geojson")
    
    inspect_images(images_dir)
    inspect_edges(edges_file)
    inspect_nodes(nodes_file)
    
    # Validate
    is_valid = validate_dataset(images_dir, edges_file, nodes_file)
    
    # Summary
    create_dataset_summary(data_dir)
    
    print("\n" + "="*70)
    if is_valid:
        print("NEXT STEPS:")
        print("1. Train the model: python src/train.py --epochs 50")
        print("2. Or test data loading: python src/test_data_loading.py")
    else:
        print("Please fix the issues above before training")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

with open('raw_data/Houston_TX_USA/edges_with_traffic.geojson') as f:
    data = json.load(f)
    print(f'Total Edges: {len(data["features"])}')
    print()
    
    if len(data["features"]) > 0:
        props = data['features'][0]['properties']
        print('Properties in first edge:')
        print('-' * 50)
        for k, v in props.items():
            print(f'  {k}: {v} (type: {type(v).__name__})')
        
        print()
        print('Sample of all edges:')
        print('-' * 50)
        for i in range(min(3, len(data["features"]))):
            edge = data['features'][i]['properties']
            print(f"Edge {i}: {edge}")
