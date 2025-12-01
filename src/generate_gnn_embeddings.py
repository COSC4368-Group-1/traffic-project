"""
Step 1: Generate MLP embeddings for TRAIN + VAL only (NOT test!)
Test edges are held out until final evaluation
This is the correct MLP → GNN pipeline
"""
import torch
import pickle
import numpy as np
from tqdm import tqdm
import os
import json

# Import your trained MLP
from train_mlp import ImprovedEdgeMLP

def load_trained_mlp(model_path='training_data/best_mlp_model.pth', 
                     scaler_path='training_data/feature_scaler.pkl'):
    """Load the trained MLP and scaler"""
    print("Loading trained MLP model...")
    
    # First, verify the files exist and are valid
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Scaler file size: {os.path.getsize(scaler_path)} bytes")
    print(f"Model file size: {os.path.getsize(model_path)} bytes")
    
    # Load model first to detect input dimension from saved state
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # First, let's inspect the saved model to determine input dimension
    checkpoint = torch.load(model_path, map_location=device)
    
    # Detect input dimension from the saved weights
    state_dict = checkpoint['model_state_dict']
    
    # Find the first linear layer weight to determine input dimension
    input_dim = None
    for key, weight in state_dict.items():
        if 'main_layers.0.weight' in key:
            input_dim = weight.shape[1]  # shape: [output_dim, input_dim]
            print(f"✓ Detected input dimension from saved model: {input_dim}")
            break
    
    if input_dim is None:
        raise RuntimeError("Could not detect input dimension from saved model")
    
    # Now load the scaler with the correct input dimension
    feature_scaler = None
    
    try:
        import joblib
        scaler_data = joblib.load(scaler_path)
        print("✓ Scaler loaded with joblib")
        
        if isinstance(scaler_data, dict):
            feature_scaler = scaler_data['scaler']
        else:
            feature_scaler = scaler_data
            
    except Exception as e:
        print(f"❌ Joblib load failed: {e}")
        raise RuntimeError(f"Failed to load scaler from {scaler_path}")
    
    # Use the EXACT features from prepare_tensors.py
    if input_dim == 10:
        # These are the exact 10 features used in prepare_tensors.py
        feature_columns = [
            'length', 'maxspeed', 'straightness', 
            'highway_type', 'lanes', 'width',
            'traffic_flow', 'travel_time', 'time_loss', 'waiting_time'
        ]
        print("✓ Using the exact 10 features from prepare_tensors.py")
    elif input_dim == 9:
        # If you removed one feature
        feature_columns = [
            'length', 'maxspeed', 'straightness', 
            'highway_type', 'lanes', 'width',
            'traffic_flow', 'travel_time', 'time_loss'
        ]
        print("✓ Using 9 features (no waiting_time)")
    else:
        # Fallback - we'll detect from data
        feature_columns = None
        print(f"⚠️  Unknown dimension {input_dim}, will detect features from data")
    
    print(f"✓ Final input dimension: {input_dim}")
    if feature_columns:
        print(f"✓ Feature columns: {feature_columns}")
    
    # Create model with the correct input dimension
    model = ImprovedEdgeMLP(input_dim=input_dim, hidden_dims=[256, 128, 64], embedding_dim=32)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        val_mae = checkpoint.get('val_mae', 'unknown')
        print(f"✓ Model loaded (val MAE: {val_mae})")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    return model, feature_scaler, feature_columns, device, input_dim

def verify_data_files():
    """Verify all required data files exist"""
    required_files = [
        'training_data/mlp_train.pkl',
        'training_data/mlp_val.pkl',
        'training_data/best_mlp_model.pth',
        'training_data/feature_scaler.pkl'
    ]
    
    print("Verifying required files...")
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} - MISSING!")
            return False
    return True

def detect_feature_columns_from_data(data_samples):
    """Detect which features are actually available in the data"""
    if not data_samples:
        return []
    
    # Get all possible feature keys (exclude non-feature keys)
    sample = data_samples[0]
    non_feature_keys = ['osm_u', 'osm_v', 'run', 'target']
    
    all_keys = [k for k in sample.keys() if k not in non_feature_keys]
    
    # Try to use the prepare_tensors.py feature order if possible
    preferred_order = [
        'length', 'maxspeed', 'straightness', 
        'highway_type', 'lanes', 'width',
        'traffic_flow', 'travel_time', 'time_loss', 'waiting_time'
    ]
    
    # Use preferred order for available features, then add any extras
    detected_features = []
    for feature in preferred_order:
        if feature in all_keys:
            detected_features.append(feature)
    
    # Add any remaining features not in preferred order
    for feature in all_keys:
        if feature not in detected_features and feature not in non_feature_keys:
            detected_features.append(feature)
    
    print(f"✓ Detected {len(detected_features)} features from data: {detected_features}")
    return detected_features

def load_mlp_train_val_only():
    """Load ONLY train and val data - test is held out!"""
    print("\nLoading MLP train and val data (test held out)...")
    
    try:
        with open('training_data/mlp_train.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('training_data/mlp_val.pkl', 'rb') as f:
            val_data = pickle.load(f)
        
        print(f"✓ Train: {len(train_data):,} samples")
        print(f"✓ Val:   {len(val_data):,} samples")
        print(f"⚠️  Test: NOT LOADED (held out for final evaluation)")
        
        # Get unique edges
        train_edges = set((s['osm_u'], s['osm_v']) for s in train_data)
        val_edges = set((s['osm_u'], s['osm_v']) for s in val_data)
        
        print(f"\nUnique edges:")
        print(f"  Train: {len(train_edges):,} edges")
        print(f"  Val:   {len(val_edges):,} edges")
        
        # Verify no overlap
        overlap = train_edges & val_edges
        if len(overlap) == 0:
            print(f"  ✓ No train/val overlap - clean split!")
        else:
            print(f"  ⚠️  WARNING: {len(overlap)} edges overlap between train and val!")
        
        return {
            'train': train_data,
            'val': val_data
        }
    except Exception as e:
        print(f"❌ Error loading data files: {e}")
        raise

def prepare_sample_features(sample, feature_columns):
    """Extract features from sample dict matching MLP training"""
    features = []
    missing_features = []
    
    for col in feature_columns:
        # Handle missing features gracefully
        if col in sample:
            feature_value = sample[col]
            if feature_value is None:
                feature_value = 0.0
                missing_features.append(f"{col} (was None)")
            features.append(float(feature_value))
        else:
            features.append(0.0)
            missing_features.append(col)
    
    if missing_features:
        print(f"⚠️  Missing features set to 0: {missing_features}")
    
    return np.array(features, dtype=np.float32)

def generate_embeddings_train_val_only():
    """Generate MLP embeddings for TRAIN and VAL splits only"""
    print("="*70)
    print("GENERATING MLP EMBEDDINGS - TRAIN + VAL ONLY")
    print("="*70)
    print("\n⚠️  IMPORTANT: Test data is HELD OUT until final evaluation")
    print("This ensures the test set is truly unseen by both MLP and GNN\n")
    
    # First verify all files exist
    if not verify_data_files():
        raise RuntimeError("Missing required data files!")
    
    # Load trained MLP
    model, scaler, feature_columns, device, input_dim = load_trained_mlp()
    
    # Load ONLY train and val data
    data_splits = load_mlp_train_val_only()
    
    # If we don't have feature_columns (unknown dimension case), detect them
    if feature_columns is None:
        print("Detecting feature columns from training data...")
        feature_columns = detect_feature_columns_from_data(data_splits['train'])
        
        # Verify dimension matches
        if len(feature_columns) != input_dim:
            print(f"⚠️  WARNING: Detected {len(feature_columns)} features but model expects {input_dim}")
            print("   Using detected features and hoping for the best...")
    
    # Generate embeddings for train and val
    all_embeddings = {}
    
    for split_name in ['train', 'val']:
        print(f"\n{'='*70}")
        print(f"Processing {split_name.upper()} split...")
        print(f"{'='*70}")
        
        samples = data_splits[split_name]
        
        # Group by edge and run
        edge_data = {}  # {(u, v): {run: {embedding, prediction, sample}}}
        
        for sample in tqdm(samples, desc=f"Generating {split_name} embeddings"):
            edge_key = (sample['osm_u'], sample['osm_v'])
            run = sample['run']
            
            # Prepare features
            features = prepare_sample_features(sample, feature_columns)
            
            # Verify feature dimension matches model expectation
            if len(features) != input_dim:
                print(f"⚠️  Feature dimension mismatch: {len(features)} != {input_dim}")
                print(f"   Features: {features}")
                print(f"   Feature columns: {feature_columns}")
                continue
            
            # Normalize
            features_scaled = scaler.transform(features.reshape(1, -1))
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            
            # Get MLP embedding (32-dim) and prediction
            with torch.no_grad():
                embedding = model(features_tensor, return_embedding=True)
                prediction = model(features_tensor, return_embedding=False)
            
            # Store by edge and run
            if edge_key not in edge_data:
                edge_data[edge_key] = {}
            
            edge_data[edge_key][run] = {
                'embedding': embedding.cpu().numpy(),
                'prediction': prediction.cpu().item(),
                'target': sample['target'],
                'sample': sample  # Keep full sample for reference
            }
        
        all_embeddings[split_name] = edge_data
        
        # Summary
        total_samples = sum(len(runs) for runs in edge_data.values())
        print(f"  ✓ {len(edge_data):,} unique edges")
        print(f"  ✓ {total_samples:,} total samples (edge × run combinations)")
    
    # Create output directory if it doesn't exist
    os.makedirs('training_data', exist_ok=True)
    
    # Save embeddings
    output_path = 'training_data/gnn_embeddings_train_val.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump({
            'train': all_embeddings['train'],
            'val': all_embeddings['val'],
            'feature_columns': feature_columns,
            'input_dim': input_dim,
            'model_checkpoint': 'training_data/best_mlp_model.pth',
            'scaler_path': 'training_data/feature_scaler.pkl'
        }, f)
    
    print(f"\n{'='*70}")
    print(f"✓ Embeddings saved to: {output_path}")
    print(f"{'='*70}")
    print(f"Split Summary:")
    print(f"  Train edges: {len(all_embeddings['train']):,}")
    print(f"  Val edges:   {len(all_embeddings['val']):,}")
    print(f"  Test edges:  NOT GENERATED (held out)")
    print(f"  Embedding dimension: 32")
    print(f"  Input features: {len(feature_columns)}")
    
    # Also save a JSON summary
    summary_path = 'training_data/embedding_generation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'train_edges': len(all_embeddings['train']),
            'val_edges': len(all_embeddings['val']),
            'embedding_dim': 32,
            'feature_columns': feature_columns,
            'input_dim': input_dim,
            'timestamp': str(np.datetime64('now'))
        }, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_path}")
    
    return all_embeddings

if __name__ == "__main__":
    try:
        embeddings = generate_embeddings_train_val_only()
        print("\n🎉 SUCCESS: Embeddings generated successfully!")
        print("\nNEXT: Run construct_gnn_graph.py to build the GNN input graph")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Make sure your MLP is trained (best_mlp_model.pth exists)")
        print("2. Check that prepare_tensors.py ran successfully")
        print("3. Verify feature dimensions match between data and model")