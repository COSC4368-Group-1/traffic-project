# check_data_structure_fixed.py
import torch
import numpy as np
import pickle
import os

def check_gnn_data_structure():
    """Check how GNN graph data is structured"""
    print("🔍 CHECKING GNN GRAPH DATA STRUCTURE")
    print("=" * 60)
    
    try:
        # Load GNN graph data
        graph_data = torch.load('training_data/gnn_graph_data.pt', map_location='cpu', weights_only=False)
        print("✅ Successfully loaded gnn_graph_data.pt")
        print(f"📊 Graph data keys: {list(graph_data.keys())}")
        print()
        
        # Check node features
        if 'node_features' in graph_data:
            node_features = graph_data['node_features']
            print("📈 NODE FEATURES:")
            print(f"   Shape: {node_features.shape}")
            print(f"   Type: {node_features.dtype}")
            print(f"   Device: {node_features.device}")
            print(f"   Min: {node_features.min().item():.4f}")
            print(f"   Max: {node_features.max().item():.4f}")
            print(f"   Mean: {node_features.mean().item():.4f}")
            print(f"   Std: {node_features.std().item():.4f}")
            
            # Check first few features in detail
            node_features_np = node_features.cpu().numpy()
            print(f"\n   First 10 features statistics:")
            print(f"   {'Idx':<4} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
            print("   " + "-" * 44)
            for i in range(min(10, node_features_np.shape[1])):
                feat = node_features_np[:, i]
                print(f"   {i:<4} {feat.mean():<10.4f} {feat.std():<10.4f} {feat.min():<10.4f} {feat.max():<10.4f}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading GNN data: {e}")
        return

def check_mlp_data_structure():
    """Check how MLP training data is structured"""
    print("\n🔍 CHECKING MLP DATA STRUCTURE")
    print("=" * 60)
    
    try:
        # Check MLP training data
        with open('training_data/mlp_train.pkl', 'rb') as f:
            mlp_train_data = pickle.load(f)
        print("✅ Successfully loaded mlp_train.pkl")
        print(f"📊 Training samples: {len(mlp_train_data)}")
        
        if len(mlp_train_data) > 0:
            sample = mlp_train_data[0]
            print(f"📋 Sample keys: {list(sample.keys())}")
            
            # Check features
            if 'features' in sample:
                features = sample['features']
                print(f"📏 Sample feature length: {len(features)}")
                print(f"🔢 Sample features: {features}")
            
            print(f"🎯 Sample target: {sample['target']}")
            print(f"🔗 Sample nodes: u={sample['osm_u']}, v={sample['osm_v']}")
            
            # Show feature ranges from first 10 samples
            print(f"\n📊 Feature ranges from first 10 samples:")
            all_features = []
            for i in range(min(10, len(mlp_train_data))):
                all_features.append(mlp_train_data[i]['features'])
            
            all_features = np.array(all_features)
            print(f"   Feature shape: {all_features.shape}")
            print(f"   Min per feature: {all_features.min(axis=0)}")
            print(f"   Max per feature: {all_features.max(axis=0)}")
            print(f"   Mean per feature: {all_features.mean(axis=0)}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading MLP training data: {e}")
    
    try:
        # Check MLP validation data
        with open('training_data/mlp_val.pkl', 'rb') as f:
            mlp_val_data = pickle.load(f)
        print("✅ Successfully loaded mlp_val.pkl")
        print(f"📊 Validation samples: {len(mlp_val_data)}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading MLP validation data: {e}")
    
    try:
        # Check MLP test data
        with open('training_data/mlp_test.pkl', 'rb') as f:
            mlp_test_data = pickle.load(f)
        print("✅ Successfully loaded mlp_test.pkl")
        print(f"📊 Test samples: {len(mlp_test_data)}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading MLP test data: {e}")

def check_mlp_model_structure():
    """Check MLP model to understand feature dimensions"""
    print("\n🔍 CHECKING MLP MODEL STRUCTURE")
    print("=" * 60)
    
    try:
        mlp_checkpoint = torch.load('training_data/best_mlp_model.pth', map_location='cpu', weights_only=False)
        print("✅ Successfully loaded best_mlp_model.pth")
        
        if 'model_state_dict' in mlp_checkpoint:
            model_state = mlp_checkpoint['model_state_dict']
            print("📋 MLP Model layers:")
            for key in model_state.keys():
                if 'weight' in key and len(model_state[key].shape) == 2:
                    print(f"   {key}: {model_state[key].shape}")
                    
                    # The first layer weight shape tells us input dimension
                    if 'input' in key or '0' in key:
                        input_dim = model_state[key].shape[1]
                        print(f"   🎯 MLP input dimension: {input_dim}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading MLP model: {e}")

def main():
    print("🚀 DATA STRUCTURE DIAGNOSTIC SCRIPT")
    print("=" * 60)
    
    # Check if files exist
    required_files = [
        'training_data/gnn_graph_data.pt',
        'training_data/mlp_train.pkl', 
        'training_data/mlp_val.pkl',
        'training_data/mlp_test.pkl',
        'training_data/best_mlp_model.pth'
    ]
    
    print("📁 Checking required files:")
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING!")
    
    print("\n" + "=" * 60)
    
    # Run diagnostics
    check_gnn_data_structure()
    check_mlp_data_structure()
    check_mlp_model_structure()
    
    print("🎉 DIAGNOSTIC COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()