"""
Data loading and preprocessing for CNN-GNN model

Handles:
- Loading traffic images (CNN input)
- Constructing graph from edge/node data (GNN input)
- Creating PyTorch DataLoaders
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import geopandas as gpd
from PIL import Image
import glob


def custom_collate_fn(batch):
    """
    Custom collate function for CNN-GNN data.
    Batches images normally but stacks graphs into a single batch graph.
    """
    images = torch.stack([item[0] for item in batch])
    graphs = [item[1] for item in batch]
    
    # Stack graphs into a single batch graph
    # Use first graph for simplicity (could batch them if needed)
    graph_batch = graphs[0]
    
    return images, graph_batch


class TrafficImageGNNDataset(Dataset):
    """
    Combined dataset that loads:
    - Traffic heatmap images for CNN
    - Traffic network graph for GNN
    """
    
    def __init__(self, 
                 images_dir,
                 edges_geojson,
                 nodes_geojson=None,
                 image_size=(224, 224),
                 transform=None):
        """
        Args:
            images_dir: Directory containing traffic heatmap images
            edges_geojson: Path to edges GeoJSON with traffic data
            nodes_geojson: Path to nodes GeoJSON (optional)
            image_size: Size to resize images to
            transform: Optional image transforms (torchvision)
        """
        self.images_dir = images_dir
        self.image_size = image_size
        self.transform = transform
        
        # Load graph data
        self.edges_gdf = gpd.read_file(edges_geojson)
        self.nodes_gdf = gpd.read_file(nodes_geojson) if nodes_geojson else None
        
        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No PNG images found in {images_dir}")
        
        print(f"Loaded {len(self.image_files)} traffic images")
        print(f"Loaded {len(self.edges_gdf)} edges")
        if self.nodes_gdf is not None:
            print(f"Loaded {len(self.nodes_gdf)} nodes")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Traffic heatmap (3, H, W)
            graph_data: PyTorch Geometric Data object
        """
        # Load and process image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.FloatTensor(np.array(image)) / 255.0
            image = image.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        
        # Create graph data
        graph_data = self._create_graph_data()
        
        return image, graph_data
    
    def _create_graph_data(self):
        """
        Construct PyTorch Geometric Data object from GeoJSON
        
        Returns:
            Data: PyTorch Geometric graph with node features and edges
        """
        edges_df = self.edges_gdf
        
        # Build node features from edges
        # Features: [speed, density, occupancy, lanes, bridge, tunnel, restricted, width]
        node_features_list = []
        edge_index_list = []
        
        feature_cols = [
            'traffic_speed',    # 0: avg speed in m/s
            'traffic_density',  # 1: vehicle density
            'occupancy',        # 2: road occupancy
            'lanes',            # 3: number of lanes
            'bridge',           # 4: is bridge (binary)
            'tunnel',           # 5: is tunnel (binary)
            'is_restricted',    # 6: access restricted (binary)
            'width'             # 7: road width in meters
        ]
        
        # Extract node features
        for col in feature_cols:
            if col in edges_df.columns:
                values = edges_df[col].fillna(0).values.astype(np.float32)
                node_features_list.append(values)
            else:
                # Use default if column missing
                node_features_list.append(np.zeros(len(edges_df), dtype=np.float32))
        
        # Stack features: (num_nodes, num_features)
        node_features = np.stack(node_features_list, axis=1)
        
        # Normalize features
        for i in range(node_features.shape[1]):
            max_val = np.max(np.abs(node_features[:, i]))
            if max_val > 0:
                node_features[:, i] /= max_val
        
        # Build edges from OSM connectivity
        # Problem: u and v are OSM node IDs (very large, not sequential)
        # Solution: Create new edges based on spatial proximity between edge midpoints
        
        print("  Building spatial connectivity graph...")
        edge_index = self._create_spatial_edges(edges_df, k=5)
        
        # Convert to PyTorch tensors
        node_features_tensor = torch.FloatTensor(node_features)
        edge_index_tensor = torch.LongTensor(edge_index)
        
        # Create target labels (example: congestion level)
        if 'congestion' in edges_df.columns:
            y = torch.FloatTensor(edges_df['congestion'].fillna(0).values)
        else:
            y = torch.zeros(len(edges_df))
        
        # Create PyTorch Geometric Data object
        graph_data = Data(
            x=node_features_tensor,           # Node features
            edge_index=edge_index_tensor,     # Edge connectivity
            y=y,                              # Node-level targets (optional)
            num_nodes=len(edges_df)
        )
        
        return graph_data
    
    def _create_spatial_edges(self, edges_df, k=5):
        """
        Create k-NN edges based on spatial proximity
        
        Args:
            edges_df: GeoDataFrame of edges
            k: Number of nearest neighbors
        
        Returns:
            edge_index: (2, num_edges)
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Get edge centroids
        centroids = np.array([[geom.centroid.x, geom.centroid.y] 
                             for geom in edges_df.geometry])
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(centroids)
        distances, indices = nbrs.kneighbors(centroids)
        
        # Build edge list (skip self-loops)
        edge_list = []
        for i in range(len(edges_df)):
            for j in indices[i, 1:]:  # Skip first (self)
                edge_list.append([i, j])
        
        edge_index = np.array(edge_list).T
        return edge_index


class TrafficImageDataset(Dataset):
    """
    Simple dataset for CNN-only models (images only)
    """
    
    def __init__(self, images_dir, image_size=(224, 224), transform=None):
        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.image_size = image_size
        self.transform = transform
        
        if len(self.image_files) == 0:
            raise ValueError(f"No PNG images found in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(np.array(image)) / 255.0
            image = image.permute(2, 0, 1)
        
        return image


class GraphDataset(Dataset):
    """
    Simple dataset for GNN-only models (graphs only)
    """
    
    def __init__(self, edges_geojson):
        self.edges_gdf = gpd.read_file(edges_geojson)
    
    def __len__(self):
        return 1  # Single graph
    
    def __getitem__(self, idx):
        # Return the full graph
        edges_df = self.edges_gdf
        
        node_features_list = []
        feature_cols = [
            'traffic_speed', 'traffic_density', 'occupancy', 'lanes',
            'bridge', 'tunnel', 'is_restricted', 'width'
        ]
        
        for col in feature_cols:
            if col in edges_df.columns:
                values = edges_df[col].fillna(0).values.astype(np.float32)
                node_features_list.append(values)
            else:
                node_features_list.append(np.zeros(len(edges_df), dtype=np.float32))
        
        node_features = np.stack(node_features_list, axis=1)
        
        # Normalize
        for i in range(node_features.shape[1]):
            max_val = np.max(np.abs(node_features[:, i]))
            if max_val > 0:
                node_features[:, i] /= max_val
        
        # Build edges
        if 'u' in edges_df.columns and 'v' in edges_df.columns:
            edge_index = np.stack([
                edges_df['u'].to_numpy().astype(np.int64),
                edges_df['v'].to_numpy().astype(np.int64)
            ])
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        
        node_features_tensor = torch.FloatTensor(node_features)
        edge_index_tensor = torch.LongTensor(edge_index)
        
        graph_data = Data(
            x=node_features_tensor,
            edge_index=edge_index_tensor,
            num_nodes=len(edges_df)
        )
        
        return graph_data


def create_dataloaders(images_dir, edges_geojson, batch_size=4, 
                      num_workers=0, split_ratio=0.8):
    """
    Create train and validation dataloaders
    
    Args:
        images_dir: Directory with traffic images
        edges_geojson: Path to edges GeoJSON
        batch_size: Batch size
        num_workers: Number of data loading workers
        split_ratio: Train/val split ratio
    
    Returns:
        train_loader, val_loader
    """
    dataset = TrafficImageGNNDataset(
        images_dir=images_dir,
        edges_geojson=edges_geojson
    )
    
    # Split dataset
    num_samples = len(dataset)
    split_idx = int(num_samples * split_ratio)
    
    train_indices = list(range(split_idx))
    val_indices = list(range(split_idx, num_samples))
    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Data Loading Module")
    print("=" * 60)
    print("\nUsage example:")
    print("""
    from data_loader import TrafficImageGNNDataset, create_dataloaders
    
    # Create dataset
    dataset = TrafficImageGNNDataset(
        images_dir='raw_data/Houston_TX_USA/traffic_images',
        edges_geojson='raw_data/Houston_TX_USA/edges_with_traffic.geojson'
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        images_dir='raw_data/Houston_TX_USA/traffic_images',
        edges_geojson='raw_data/Houston_TX_USA/edges_with_traffic.geojson',
        batch_size=4
    )
    
    # Iterate through batches
    for images, graph_data in train_loader:
        print(f"Images: {images.shape}")
        print(f"Graph: nodes={graph_data.num_nodes}, edges={graph_data.num_edges}")
    """)
