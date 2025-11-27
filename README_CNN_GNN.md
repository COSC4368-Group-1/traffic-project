<!-- filepath: c:\Users\Ricardo Trevizo\Documents\Code\COSC4368\traffic-project\README_CNN_GNN.md -->

# MLP-GNN Traffic Prediction Model 🎉

## 📋 Table of Contents

- [How It Works](#how-it-works)
- [Current Status](#current-status)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Training Process](#training-process)
- [Usage](#usage)

---

## 🔧 How It Works

### High-Level Overview

This project combines **Multi-Layer Perceptrons (MLP)** and **Graph Neural Networks (GNN)** to predict traffic congestion in urban road networks.

```
Road Feature Vectors    →  [MLP Feature Processing]   ⟍
                                                       [GNN Graph Learning]  →  Traffic Prediction
Road Network Graph      →  [Graph Structure]          ⟋
```

### Key Difference from CNN Approach

**CNN Approach** (Previous):

```
Input: A 64×64 image (pixels arranged in 2D grid)
┌─────────────────┐
│ 🟥🟥🟧🟧🟨🟨🟩🟩 │  Each pixel = congestion at that location
│ 🟥🟥🟧🟧🟨🟨🟩🟩 │  Model learns: "Red blob in top-left means traffic jam"
│ 🟧🟧🟨🟨🟩🟩🟩🟩 │
│ 🟧🟧🟨🟨🟩🟩🟩🟩 │  Problem: Which roads are these? How do they connect?
└─────────────────┘  Answer: ¯\_(ツ)_/¯ (lost in rasterization!)

Process:
1. Look at local pixel neighborhoods
2. Find spatial patterns ("traffic jam looks like this blob shape")
3. Classify: "This image shows heavy traffic in northwest"

What it learns: "Spatial patterns in images"
```

**MLP Approach** (Current):

```
Input: A feature vector per road segment
Road ID 1: [lanes=2, width=7m, speed=8m/s, congestion=0.7, straight=0.9, ...]
Road ID 2: [lanes=4, width=14m, speed=15m/s, congestion=0.2, straight=0.6, ...]
Road ID 3: [lanes=1, width=3.5m, speed=5m/s, congestion=0.9, straight=0.3, ...]

Process:
1. Look at ALL features for ONE road at a time
2. Find feature combinations: "2 lanes + high congestion + slow speed = BAD"
3. Classify: "This specific road is in heavy traffic state"

What it learns: "Feature patterns in structured data"
```

**Why This Change?**

- ✅ **More Data**: 200,000+ edges with 10+ features each (vs. 5 images)
- ✅ **Better Features**: Road geometry, connectivity, and traffic metrics directly available
- ✅ **Same Training Time**: Process structured data as fast as images
- ✅ **Explicit Connections**: Road network structure preserved (not lost in rasterization)
- ✅ **Easier to Build**: Direct feature engineering instead of image processing

### 1. **Data Collection & Preparation**

**Source**: SUMO (Simulation of Urban MObility) traffic simulator

- Simulates vehicle movements on Houston, TX road network
- Based on real OpenStreetMap (OSM) data
- Generates multiple traffic snapshots over time

**Outputs**:

- **Road Network Graph**: 200,000+ edges with rich features
  - Nodes (intersections) and edges (roads) with traffic attributes
  - **10+ Features per Edge**: speed, density, occupancy, lanes, width, geometry, bridge/tunnel flags, etc.
- **Traffic Attributes**: Speed, density, occupancy, congestion level per road

### 2. **MLP Path: Feature Processing**

The MLP processes individual road segment features:

```
Input: Road Feature Vector (10+ dimensions)
    • lanes (int)
    • width (meters)
    • speed (m/s)
    • maxspeed (km/h)
    • congestion (0-1)
    • bridge (0/1)
    • tunnel (0/1)
    • geometry metrics (straightness, length, etc.)
    ↓
Dense Layer 1: 10+ → 64 dimensions
    • ReLU Activation
    • Learns feature combinations
    • "2 lanes + narrow + slow = residential street"
    ↓
Dense Layer 2: 64 → 128 dimensions
    • ReLU Activation
    • Higher-level feature patterns
    • "High congestion + multiple lanes = highway jam"
    ↓
Dense Layer 3: 128 → 256 dimensions
    • ReLU Activation
    • Complex feature interactions
    ↓
Output: 256-dim feature vector per road
    • Compact representation of road state
    • Ready for graph processing
```

**What MLP learns**:

- Feature combinations that indicate congestion
- Road type classification from attributes
- Traffic state patterns per road segment
- Non-linear relationships between features

### 3. **GNN Path: Network Topology Processing**

The GNN processes the road network graph with MLP-enhanced features:

```
Road Network Graph
    • Nodes: Road segments (200,000+ edges as nodes)
    • Edges: Road connections from OSM
    • Node features: 256-dim vectors from MLP
    ↓
Graph Convolutional Network (GCN)
    Layer 1: 256-dim → 128-dim
    • Each node aggregates info from neighboring roads
    • Message passing: "What's traffic like on connected roads?"
    • MLP features + network structure

    Layer 2: 128-dim → 256-dim
    • Second-order information
    • Multi-hop traffic patterns
    • "Traffic 2 roads away affects me"
    ↓
Attention-Based Aggregation
    • Learns which neighbors are important
    • Dynamic weighting of road connections
    • "Highway entrance more important than side street"
    ↓
Graph Embedding (256-dim)
    • Network-aware road representation
    • Combines individual features + connectivity
```

**What GNN learns**:

- How roads are connected
- Which intersections influence each other
- Traffic flow propagation through network
- Contextual road importance

### 4. **Prediction Output**

Final prediction from GNN embeddings:

```
GNN Output: Traffic congestion level (0-1)
    • OR: Average traffic speed
    • OR: Overall network occupancy
```

---

## Current Status

✅ **Data Generation**: Complete

- 200,000+ road segments with traffic attributes
- 10+ features per edge: lanes, width, speed, geometry, etc.
- Network graph structure preserved from OSM

✅ **Architecture Design**: MLP → GNN Pipeline

- MLP: 3-layer feature processor (10+ → 64 → 128 → 256)
- GNN: 2-layer GCN with attention aggregation
- Direct feature engineering (no image processing needed)

🔄 **Implementation**: In Progress

- Data loader for edge features + graph structure
- MLP feature encoder
- GNN integration with MLP outputs
- Training pipeline setup

---

## 🏗️ Architecture

### Model Components

#### CNN (Convolutional Neural Network)

```
ResNet18 Architecture
├── Input Layer: 224×224×4 (RGBA images)
├── Conv Block 1: 64 filters
├── Conv Block 2: 128 filters
├── Conv Block 3: 256 filters
├── Conv Block 4: 512 filters
├── Global Average Pool
└── Output: 256-dim feature vector

Purpose: Extract spatial patterns from traffic heatmaps
```

#### GNN (Graph Neural Network)

```
GCN (Graph Convolutional Network)
├── Input: Node features (8-dim per node)
│   └── Features: lanes, maxspeed, width, bridge, tunnel, etc.
├── GCN Layer 1: 8 → 128 dimensions
│   └── Message passing across edges
├── ReLU Activation
├── GCN Layer 2: 128 → 256 dimensions
│   └── Aggregates 2-hop neighborhood information
├── Attention Mechanism
│   └── Learns importance weights for edges
└── Output: 256-dim graph embedding

Purpose: Capture network topology and road relationships
```

#### Fusion Module

```
Concatenation: [CNN_features(256) + GNN_features(256)] → 512-dim
    ↓
Dense Layer 1: 512 → 256 dims
    ↓
ReLU Activation
    ↓
Dense Layer 2: 256 → 1 dim
    ↓
Output: Traffic Prediction
```

### Why CNN + GNN?

| Component  | Learns                            | Input               |
| ---------- | --------------------------------- | ------------------- |
| **CNN**    | Spatial traffic patterns          | Image heatmaps      |
| **GNN**    | Network structure & relationships | Road graph topology |
| **Fusion** | How both representations relate   | Combined features   |

Example: CNN sees congestion in downtown area, GNN knows which roads feed into that area → Combined model predicts traffic will spread to connecting roads.

---

## 📊 Data Pipeline

### Step 1: Data Grab (`data_grab.py`)

```
Input: City coordinates (lat/lon)
    ↓
Download OpenStreetMap (OSM) data
    ↓
Create SUMO simulation
    ├── Generate road network
    ├── Create traffic routes
    └── Simulate vehicle movements (30+ mins)
    ↓
Outputs:
    ├── nodes.geojson - Intersection data
    ├── edges.geojson - Road data
    ├── edges_with_traffic.geojson - Roads + traffic metrics
    ├── network.net.xml - SUMO network
    ├── routes.rou.xml - Vehicle routes
    └── traffic_images/ - Traffic heatmap images (5 snapshots)
```

### Step 2: Data Loading (`data_loader.py`)

```
TrafficImageGNNDataset
├── Load image from disk
│   └── Normalize to [0, 1]
├── Load graph from GeoJSON
│   ├── Create node features matrix (269K × 8)
│   ├── Create edge index from OSM connections
│   └── Remap node IDs (OSM IDs → sequential indices)
└── Return (image, PyG Data object)

Processing:
    • Image: 224×224×4 (RGBA)
    • Graph nodes: 269,151 nodes
    • Graph edges: 300,972 connections
    • Node features: 8 attributes per node
```

### Step 3: Batch Creation

```
Collate Function
├── Stack images into batch tensor
│   └── Shape: [batch_size, 4, 224, 224]
├── Combine graphs
│   └── Concatenate node features
│   └── Offset edge indices for batch
└── Return batch ready for model
```

---

## 🚂 Training Process

### Training Loop

```
for epoch in range(num_epochs):

    # Training Phase
    for batch in train_loader:
        images, graphs = batch

        # Forward Pass
        predictions = model(images, graphs)

        # Calculate Loss
        loss = MSE(predictions, targets)

        # Backward Pass
        loss.backward()
        optimizer.step()

    # Validation Phase
    for batch in val_loader:
        predictions = model(images, graphs)
        val_loss = MSE(predictions, targets)

        # Save if best
        if val_loss < best_loss:
            save_checkpoint('model_best.pt')

    # Logging
    tensorboard.log(loss, val_loss, epoch)
```

### Loss Function

**Mean Squared Error (MSE)**:

```
Loss = mean((predicted_congestion - actual_congestion)²)
```

Minimizes prediction error across all edges.

### Optimization

- **Optimizer**: Adam
  - Adapts learning rates per parameter
  - Good for deep networks
- **Learning Rate**: 0.001
  - Controls step size in gradient descent
- **Weight Decay**: 1e-5
  - L2 regularization to prevent overfitting

---

## 💻 Usage

### Basic Training

```powershell
cd traffic-project
.\venv\Scripts\Activate.ps1
python src/train.py --epochs 50 --batch-size 16 --device cuda
```

### Monitor Training in Real-Time

```powershell
# Terminal 1: Run training
python src/train.py

# Terminal 2: Launch TensorBoard
tensorboard --logdir runs/

# Then open: http://localhost:6006
```

### Make Predictions

```python
import torch
from src.cnn_gnn_model import CNNGNNFusionModel
from src.data_loader import TrafficImageGNNDataset

# Load trained model
model = CNNGNNFusionModel()
model.load_state_dict(torch.load('model_best.pt'))
model.eval()

# Load data
dataset = TrafficImageGNNDataset(
    'raw_data/Houston_TX_USA/traffic_images',
    'raw_data/Houston_TX_USA/edges_with_traffic.geojson'
)
image, graph = dataset[0]

# Predict
with torch.no_grad():
    image = image.unsqueeze(0)  # Add batch dim
    prediction, node_emb, cnn_feat, graph_emb = model(
        image, graph.x, graph.edge_index
    )
    print(f"Predicted congestion: {prediction.item():.3f}")
```

### Evaluate Model

```python
# Calculate metrics
from sklearn.metrics import mean_absolute_error, r2_score

predictions = []
targets = []

for image, graph in test_dataset:
    pred = model(image.unsqueeze(0), graph.x, graph.edge_index)
    predictions.append(pred.item())
    targets.append(graph.y.item())

mae = mean_absolute_error(targets, predictions)
r2 = r2_score(targets, predictions)

print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
```

---

## 📁 File Structure

```
traffic-project/
├── src/
│   ├── cnn_gnn_model.py       # Model architecture (11.7M params)
│   ├── data_loader.py          # Dataset & DataLoader
│   ├── train.py                # Full training script
│   ├── train_simple.py         # Simplified training ⭐ CURRENTLY RUNNING
│   ├── data_grab.py            # SUMO data collection
│   ├── inspect_data.py         # Data inspection utility
│   └── test_data_loading.py    # Unit tests for data loading
├── raw_data/
│   └── Houston_TX_USA/
│       ├── traffic_images/     # 5 heatmap images (224×224)
│       ├── edges_with_traffic.geojson     # 300K roads + traffic data
│       ├── nodes.geojson       # 269K intersections
│       ├── network.net.xml     # SUMO network
│       ├── routes.rou.xml      # Vehicle routes
│       └── edgedata.xml        # SUMO traffic metrics
├── runs/
│   └── traffic_model_*/        # TensorBoard logs
├── model_best.pt               # Best performing model checkpoint
├── model_final.pt              # Final training checkpoint
├── requirements.txt            # Python dependencies
└── README_CNN_GNN.md          # This file
```

---

## System Info

- **Device**: CPU (GPU available with `--device cuda`)
- **Python Version**: 3.12
- **Virtual Environment**: venv activated
- **PyTorch**: 2.9.1+ installed
- **PyTorch Geometric**: 2.7.0+ installed
- **SUMO**: For traffic simulation

## Notes

- Small dataset (5 images) → use for development/testing only
- For production: Generate data for multiple cities and time periods
- Graph size (300K nodes) is manageable on CPU for training
- Can optimize with batch graph construction for larger datasets
