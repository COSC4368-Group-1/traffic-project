"""
Construct GNN graph from MLP embeddings and adjacency matrix
"""
import torch
import pickle
import numpy as np
from collections import defaultdict
import os

def load_embeddings_and_adjacency():
    """Load MLP embeddings and adjacency matrix"""
    print("Loading MLP embeddings and graph structure...")
    
    # Load MLP embeddings
    with open('training_data/gnn_embeddings_train_val.pkl', 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Load adjacency matrix
    with open('training_data/adjacency_matrix.pkl', 'rb') as f:
        adjacency_data = pickle.load(f)
    
    print("✓ Embeddings and adjacency loaded")
    return embeddings_data, adjacency_data

def create_gnn_node_features(embeddings_data, adjacency_data):
    """
    Create node features by aggregating MLP embeddings for each node
    For each node, average the embeddings of all edges connected to it
    """
    print("\nCreating GNN node features...")
    
    node_to_idx = adjacency_data['node_to_idx']
    num_nodes = len(node_to_idx)
    embedding_dim = 32  # From your MLP
    
    # Initialize node feature matrix
    node_features = torch.zeros(num_nodes, embedding_dim)
    node_feature_counts = torch.zeros(num_nodes)
    
    # Aggregate embeddings for each node from connected edges
    for split in ['train', 'val']:
        for edge_key, runs_data in embeddings_data[split].items():
            u, v = edge_key
            
            if u not in node_to_idx or v not in node_to_idx:
                continue
                
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            
            # Average embeddings across runs for this edge
            edge_embeddings = []
            for run_data in runs_data.values():
                edge_embeddings.append(run_data['embedding'])
            
            if edge_embeddings:
                avg_embedding = np.mean(edge_embeddings, axis=0)
                
                # Add to both nodes (undirected graph)
                node_features[u_idx] += torch.tensor(avg_embedding)
                node_features[v_idx] += torch.tensor(avg_embedding)
                node_feature_counts[u_idx] += 1
                node_feature_counts[v_idx] += 1
    
    # Normalize by count (avoid division by zero)
    nonzero_counts = node_feature_counts > 0
    node_features[nonzero_counts] /= node_feature_counts[nonzero_counts].unsqueeze(1)
    
    print(f"✓ Node features: {node_features.shape}")
    print(f"  Nodes with features: {nonzero_counts.sum().item()}/{num_nodes}")
    
    return node_features

def create_gnn_edge_labels(embeddings_data, adjacency_data):
    """
    Create edge-level labels for GNN training
    Use the MLP predictions as edge features or labels
    """
    print("\nCreating GNN edge labels...")
    
    edge_index = adjacency_data['edge_index']  # [2, num_edges]
    num_edges = edge_index.shape[1]
    
    # Map edge (u,v) to index in edge_index
    edge_to_idx = {}
    for i in range(num_edges):
        u_idx, v_idx = edge_index[0, i], edge_index[1, i]
        edge_to_idx[(u_idx, v_idx)] = i
    
    # Initialize edge labels
    edge_labels = torch.full((num_edges,), -1.0)  # -1 for no label
    edge_has_label = torch.zeros(num_edges, dtype=torch.bool)
    
    # Fill in labels from MLP data
    for split in ['train', 'val']:
        for edge_key, runs_data in embeddings_data[split].items():
            u, v = edge_key
            
            # Convert to node indices
            if u not in adjacency_data['node_to_idx'] or v not in adjacency_data['node_to_idx']:
                continue
                
            u_idx = adjacency_data['node_to_idx'][u]
            v_idx = adjacency_data['node_to_idx'][v]
            
            # Find this edge in edge_index (undirected, check both directions)
            edge_id = None
            if (u_idx, v_idx) in edge_to_idx:
                edge_id = edge_to_idx[(u_idx, v_idx)]
            elif (v_idx, u_idx) in edge_to_idx:
                edge_id = edge_to_idx[(v_idx, u_idx)]
            
            if edge_id is not None:
                # Use average prediction across runs
                predictions = [run_data['prediction'] for run_data in runs_data.values()]
                avg_prediction = np.mean(predictions)
                
                edge_labels[edge_id] = avg_prediction
                edge_has_label[edge_id] = True
    
    print(f"✓ Edge labels: {edge_has_label.sum().item()}/{num_edges} edges have labels")
    
    return edge_labels, edge_has_label

def create_train_val_masks(embeddings_data, adjacency_data, edge_has_label):
    """
    Create masks for train/val splits
    """
    print("\nCreating train/val masks...")
    
    edge_index = adjacency_data['edge_index']
    num_edges = edge_index.shape[1]
    
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    
    # Map edges to indices
    edge_to_idx = {}
    for i in range(num_edges):
        u_idx, v_idx = edge_index[0, i], edge_index[1, i]
        edge_to_idx[(u_idx, v_idx)] = i
    
    # Assign masks based on which split the edge came from
    for split_name in ['train', 'val']:
        for edge_key in embeddings_data[split_name].keys():
            u, v = edge_key
            
            if u not in adjacency_data['node_to_idx'] or v not in adjacency_data['node_to_idx']:
                continue
                
            u_idx = adjacency_data['node_to_idx'][u]
            v_idx = adjacency_data['node_to_idx'][v]
            
            # Find edge index
            edge_id = None
            if (u_idx, v_idx) in edge_to_idx:
                edge_id = edge_to_idx[(u_idx, v_idx)]
            elif (v_idx, u_idx) in edge_to_idx:
                edge_id = edge_to_idx[(v_idx, u_idx)]
            
            if edge_id is not None and edge_has_label[edge_id]:
                if split_name == 'train':
                    train_mask[edge_id] = True
                else:
                    val_mask[edge_id] = True
    
    print(f"✓ Train edges: {train_mask.sum().item()}")
    print(f"✓ Val edges: {val_mask.sum().item()}")
    
    return train_mask, val_mask

def save_gnn_data(graph_data):
    """Save the complete GNN dataset"""
    output_path = 'training_data/gnn_graph_data.pt'
    torch.save(graph_data, output_path)
    print(f"\n✓ GNN graph data saved to: {output_path}")
    
    # Also save as pickle for inspection
    with open('training_data/gnn_graph_data.pkl', 'wb') as f:
        pickle.dump({
            'node_features_shape': graph_data['node_features'].shape,
            'edge_index_shape': graph_data['edge_index'].shape,
            'edge_labels_shape': graph_data['edge_labels'].shape,
            'train_edges': graph_data['train_mask'].sum().item(),
            'val_edges': graph_data['val_mask'].sum().item(),
            'total_edges': graph_data['edge_index'].shape[1],
            'total_nodes': graph_data['node_features'].shape[0]
        }, f)
    
    print("✓ Graph summary saved")

def main():
    print("="*70)
    print("CONSTRUCTING GNN GRAPH FROM MLP EMBEDDINGS")
    print("="*70)
    
    # Load data
    embeddings_data, adjacency_data = load_embeddings_and_adjacency()
    
    # Create components
    node_features = create_gnn_node_features(embeddings_data, adjacency_data)
    edge_labels, edge_has_label = create_gnn_edge_labels(embeddings_data, adjacency_data)
    train_mask, val_mask = create_train_val_masks(embeddings_data, adjacency_data, edge_has_label)
    
    # Assemble final graph data
    graph_data = {
        'node_features': node_features,
        'edge_index': torch.LongTensor(adjacency_data['edge_index']),
        'edge_labels': edge_labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'node_to_idx': adjacency_data['node_to_idx'],
        'num_nodes': node_features.shape[0],
        'num_edges': adjacency_data['edge_index'].shape[1]
    }
    
    # Save everything
    save_gnn_data(graph_data)
    
    print("\n" + "="*70)
    print("🎉 GNN GRAPH CONSTRUCTION COMPLETE!")
    print("="*70)
    print(f"Graph Summary:")
    print(f"  Nodes: {graph_data['num_nodes']}")
    print(f"  Edges: {graph_data['num_edges']}")
    print(f"  Node features: {graph_data['node_features'].shape}")
    print(f"  Train edges: {graph_data['train_mask'].sum().item()}")
    print(f"  Val edges: {graph_data['val_mask'].sum().item()}")
    print(f"\nNext: Create GNN model and train!")
    print("="*70)

if __name__ == "__main__":
    main()