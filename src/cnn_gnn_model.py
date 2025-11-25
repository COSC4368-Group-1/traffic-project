"""
CNN-GNN Hybrid Model for Traffic Prediction

Architecture Overview:
- CNN: Processes traffic heatmap images (spatial patterns in local areas)
- GNN: Processes network topology (relationships between edges/intersections)
- Fusion: Combines CNN features with GNN outputs for final prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torchvision import models
import numpy as np


class CNNFeatureExtractor(nn.Module):
    """
    CNN component: Extracts spatial features from traffic heatmap images
    Input: Traffic visualization images (B, 3, 224, 224)
    Output: Feature vectors (B, feature_dim)
    """
    def __init__(self, feature_dim=256):
        super(CNNFeatureExtractor, self).__init__()
        
        # Use pre-trained ResNet18 as backbone
        self.resnet = models.resnet18(pretrained=True)
        
        # Replace final layer to output custom feature dimension
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, feature_dim)
        
        self.feature_dim = feature_dim
    
    def forward(self, x):
        """
        Args:
            x: Batch of traffic images (B, 3, H, W)
        Returns:
            features: CNN features (B, feature_dim)
        """
        features = self.resnet(x)
        return features


class GNNGraphProcessor(nn.Module):
    """
    GNN component: Processes traffic network graph structure
    Input: Graph with node features and edges
    Output: Node embeddings capturing network relationships
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=256):
        super(GNNGraphProcessor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Two-layer graph convolution
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, output_dim)
        
        # Optional attention/aggregation layer
        self.attention = nn.Linear(output_dim, 1)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge connectivity (2, num_edges)
        Returns:
            node_embeddings: (num_nodes, output_dim)
            graph_embedding: (1, output_dim) - aggregated graph representation
        """
        # Graph convolution layers
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.gc2(x, edge_index)
        node_embeddings = F.relu(x)
        
        # Aggregate to graph-level representation using attention
        attention_weights = torch.softmax(self.attention(node_embeddings), dim=0)
        graph_embedding = torch.sum(node_embeddings * attention_weights, dim=0, keepdim=True)
        
        return node_embeddings, graph_embedding


class CNNGNNFusionModel(nn.Module):
    """
    Hybrid CNN-GNN model that combines spatial (CNN) and topological (GNN) features
    """
    def __init__(self, 
                 cnn_feature_dim=256,
                 gnn_input_dim=8,      # Node feature dimension (speed, density, etc)
                 gnn_hidden_dim=128,
                 gnn_output_dim=256,
                 fusion_dim=512,
                 output_dim=1):        # Task-dependent: 1 for regression, num_classes for classification
        super(CNNGNNFusionModel, self).__init__()
        
        # Components
        self.cnn = CNNFeatureExtractor(feature_dim=cnn_feature_dim)
        self.gnn = GNNGraphProcessor(
            input_dim=gnn_input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim
        )
        
        # Fusion layers - combine CNN and GNN features
        fusion_input_dim = cnn_feature_dim + gnn_output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, output_dim)
        )
    
    def forward(self, images, node_features, edge_index):
        """
        Args:
            images: Traffic heatmap images (B, 3, 224, 224)
            node_features: Graph node features (num_nodes, input_dim)
            edge_index: Graph edge connectivity (2, num_edges)
        
        Returns:
            predictions: Model output (B, output_dim) or (num_nodes, output_dim)
        """
        # CNN branch - process images
        cnn_features = self.cnn(images)  # (B, 256)
        
        # GNN branch - process graph
        node_embeddings, graph_embedding = self.gnn(node_features, edge_index)  # (num_nodes, 256), (1, 256)
        
        # Fusion - combine features
        # For per-image prediction, replicate graph embedding to match batch size
        batch_size = images.shape[0]
        gnn_features = graph_embedding.expand(batch_size, -1)  # (B, 256)
        
        fused_features = torch.cat([cnn_features, gnn_features], dim=1)  # (B, 512)
        predictions = self.fusion(fused_features)  # (B, output_dim)
        
        return predictions, node_embeddings, cnn_features, graph_embedding


class TrafficPredictionHead(nn.Module):
    """
    Task-specific head for traffic prediction
    Can predict: congestion level, speed, density, ETA, etc.
    """
    def __init__(self, input_dim=512, output_dim=1):
        super(TrafficPredictionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.head(x)


# Example usage and testing
if __name__ == "__main__":
    print("CNN-GNN Hybrid Model Architecture\n")
    
    # Create model
    model = CNNGNNFusionModel(
        cnn_feature_dim=256,
        gnn_input_dim=8,      # 8 features per node
        gnn_output_dim=256,
        fusion_dim=512,
        output_dim=1          # For traffic speed/congestion prediction
    )
    
    print(model)
    print("\n" + "="*60)
    
    # Test with dummy data
    batch_size = 4
    num_nodes = 100
    num_edges = 250
    
    # Dummy traffic images (B, 3, 224, 224)
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Dummy node features: [speed, density, occupancy, lane_count, avg_lanes, bridges, degree, congestion]
    node_features = torch.randn(num_nodes, 8)
    
    # Dummy edge connectivity (2, num_edges)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Forward pass
    predictions, node_embeddings, cnn_features, graph_embedding = model(
        images, node_features, edge_index
    )
    
    print(f"Input shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Node features: {node_features.shape}")
    print(f"  Edge index: {edge_index.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Node embeddings: {node_embeddings.shape}")
    print(f"  CNN features: {cnn_features.shape}")
    print(f"  Graph embedding: {graph_embedding.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
