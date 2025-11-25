"""
Test the data loader to ensure it works correctly before training
"""

import torch
from data_loader import TrafficImageGNNDataset
import os


def test_data_loader():
    print("\n" + "="*70)
    print("TESTING DATA LOADER")
    print("="*70)
    
    data_dir = "raw_data/Houston_TX_USA"
    images_dir = os.path.join(data_dir, "traffic_images")
    edges_file = os.path.join(data_dir, "edges_with_traffic.geojson")
    
    # Create dataset
    print("\nCreating dataset...")
    try:
        dataset = TrafficImageGNNDataset(
            images_dir=images_dir,
            edges_geojson=edges_file,
            image_size=(224, 224)
        )
        print(f"✓ Dataset created successfully")
        print(f"  - Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loading one sample
    print("\nTesting sample loading...")
    try:
        image, graph_data = dataset[0]
        
        print(f"✓ Sample loaded successfully")
        print(f"\n  Image properties:")
        print(f"    - Shape: {image.shape}")
        print(f"    - Dtype: {image.dtype}")
        print(f"    - Min: {image.min():.4f}, Max: {image.max():.4f}")
        
        print(f"\n  Graph properties:")
        print(f"    - Nodes: {graph_data.num_nodes}")
        print(f"    - Edges: {graph_data.num_edges}")
        print(f"    - Node features: {graph_data.x.shape}")
        print(f"    - Edge index shape: {graph_data.edge_index.shape}")
        print(f"    - Node targets: {graph_data.y.shape if hasattr(graph_data, 'y') else 'None'}")
        
    except Exception as e:
        print(f"❌ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test DataLoader
    print("\nTesting DataLoader...")
    try:
        from torch.utils.data import DataLoader
        from data_loader import custom_collate_fn
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                               collate_fn=custom_collate_fn)
        
        for batch_idx, (images, graphs) in enumerate(dataloader):
            print(f"✓ Batch loaded successfully")
            print(f"\n  Batch properties:")
            print(f"    - Image batch shape: {images.shape}")
            print(f"    - Graph type: {type(graphs)}")
            print(f"    - Graph nodes: {graphs.num_nodes if hasattr(graphs, 'num_nodes') else 'N/A'}")
            print(f"    - Graph edges: {graphs.num_edges if hasattr(graphs, 'num_edges') else 'N/A'}")
            
            # Test forward pass with model
            print("\nTesting model forward pass...")
            try:
                from cnn_gnn_model import CNNGNNFusionModel
                
                model = CNNGNNFusionModel(
                    cnn_feature_dim=256,
                    gnn_input_dim=8,
                    gnn_output_dim=256,
                    fusion_dim=512,
                    output_dim=1
                )
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                images = images.to(device)
                
                # Use the graph from collate
                graph_data = graphs.to(device)
                
                predictions, node_embeddings, cnn_features, graph_embedding = model(
                    images,
                    graph_data.x,
                    graph_data.edge_index
                )
                
                print(f"✓ Forward pass successful")
                print(f"  - Predictions shape: {predictions.shape}")
                print(f"  - CNN features shape: {cnn_features.shape}")
                print(f"  - Graph embedding shape: {graph_embedding.shape}")
                
            except Exception as e:
                print(f"❌ Error in forward pass: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            break  # Only test first batch
    
    except Exception as e:
        print(f"❌ Error creating DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print("\nNext steps:")
    print("1. Train the model: python src/train.py --epochs 50 --batch-size 2")
    print("2. Monitor training with TensorBoard: tensorboard --logdir runs/")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_data_loader()
    exit(0 if success else 1)
