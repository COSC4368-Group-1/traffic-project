import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric

# ADD THIS NEW LOSS CLASS AT THE TOP
class EmergencyRoutingLoss(nn.Module):
    def __init__(self, critical_penalty=2.0, safe_weight=0.1, variance_weight=0.001):
        super().__init__()
        self.critical_penalty = critical_penalty
        self.safe_weight = safe_weight
        self.variance_weight = variance_weight
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        
        # Asymmetric penalty - much heavier for critical errors
        critical_mask = (predictions > 0.6) & (targets < 0.4)
        critical_penalty = critical_mask.float().mean() * self.critical_penalty
        
        # Safe prediction bonus (but don't subtract from loss directly)
        safe_mask = (predictions < 0.5) & (targets < 0.4)
        safe_bonus = safe_mask.float().mean() * self.safe_weight
        
        # Gentle variance encouragement (positive weight)
        variance_encouragement = predictions.var() * self.variance_weight
        
        # Ensure loss is always positive
        total_loss = mse_loss + critical_penalty - safe_bonus - variance_encouragement
        return torch.clamp(total_loss, min=0.0)

# ADD THIS ENHANCED MODEL
class EmergencyGNNEnhanced(torch.nn.Module):
    def __init__(self, node_dim=32, hidden_dim=256, num_layers=3):
        super().__init__()
        
        # Enhanced node encoder (match MLP capacity)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Multiple GCN layers with residual connections
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim // 2 if i == 0 else hidden_dim // 4
            out_dim = hidden_dim // 4
            self.convs.append(GCNConv(in_dim, out_dim))
        
        # Enhanced edge predictor with attention
        self.edge_predictor = nn.Sequential(
            nn.Linear((hidden_dim // 4) * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_label_index=None):
        # Enhanced node encoding
        x = self.node_encoder(x)
        
        # Message passing with residual
        for i, conv in enumerate(self.convs):
            x_new = F.relu(conv(x, edge_index))
            if x.shape == x_new.shape:  # Residual if same dimension
                x = x + x_new
            else:
                x = x_new
            x = self.dropout(x)
        
        if edge_label_index is not None:
            src_nodes = x[edge_label_index[0]]
            dst_nodes = x[edge_label_index[1]]
            edge_features = torch.cat([src_nodes, dst_nodes], dim=1)
            return self.edge_predictor(edge_features).squeeze()
        
        return x

class EmergencyGNN(torch.nn.Module):
    def __init__(self, node_dim=32, hidden_dim=128, edge_dim=1):
        super().__init__()
        
        # Node feature transformation
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # Graph Convolutional Layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # Edge prediction head - expects (hidden_dim // 2) * 2 because we concatenate src + dst
        self.edge_predictor = nn.Sequential(
            nn.Linear((hidden_dim // 2) * 2, hidden_dim),  # Fixed input dimension
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, edge_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_label_index=None):
        # Encode node features
        x = self.node_encoder(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Message passing
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)  # Final node embeddings
        
        # If we're predicting specific edges
        if edge_label_index is not None:
            # edge_label_index should contain [u_indices, v_indices] pairs
            # where u_indices and v_indices are node indices, not edge indices
            
            # Get embeddings for source and target nodes of each edge
            src_nodes = x[edge_label_index[0]]  # Source node embeddings
            dst_nodes = x[edge_label_index[1]]  # Target node embeddings
            
            # Concatenate and predict edge suitability
            edge_features = torch.cat([src_nodes, dst_nodes], dim=1)
            edge_predictions = self.edge_predictor(edge_features).squeeze()
            
            return edge_predictions
        
        return x

class EmergencyGNNSimple(torch.nn.Module):
    """Simpler version for faster experimentation"""
    def __init__(self, node_dim=32, hidden_dim=64):
        super().__init__()
        
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        
        # Edge predictor - expects (hidden_dim // 2) input since we use sum
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # Fixed: input is hidden_dim//2
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_label_index=None):
        x = F.relu(self.node_encoder(x))
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)  # Final embeddings: [num_nodes, hidden_dim//2]
        
        if edge_label_index is not None:
            # Get node embeddings for the edges we want to predict
            src_nodes = x[edge_label_index[0]]  # [batch_size, hidden_dim//2]
            dst_nodes = x[edge_label_index[1]]  # [batch_size, hidden_dim//2]
            
            # Use sum of node embeddings as edge features
            edge_features = src_nodes + dst_nodes  # [batch_size, hidden_dim//2]
            return self.edge_predictor(edge_features).squeeze()
        
        return x