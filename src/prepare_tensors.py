import pickle
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_prepare_mlp_data():
    """
    Convert MLP PKL files to PyTorch tensors for training WITH NORMALIZATION
    """
    print("Loading and preparing MLP training data...")
    
    # Load the PKL files
    with open('training_data/mlp_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('training_data/mlp_val.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Define feature columns (from your data structure)
    feature_columns = [
    'length', 'maxspeed', 'straightness', 
    'highway_type', 'lanes', 'width',
    'traffic_flow', 'travel_time', 'time_loss', 'waiting_time'
    # 11 total features - all meaningful!
]
    
    def extract_features_targets(data):
        """Extract features and targets from data samples"""
        features = []
        targets = []
        
        for sample in data:
            # Extract features
            feature_vec = [sample[col] for col in feature_columns]
            features.append(feature_vec)
            
            # Extract target
            targets.append(sample['target'])
        
        return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)
    
    # Extract features and targets
    X_train, y_train = extract_features_targets(train_data)
    X_val, y_val = extract_features_targets(val_data)
    
    print(f"Feature matrix shape - Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Target vector shape - Train: {y_train.shape}, Val: {y_val.shape}")
    
    # ===== ADDED NORMALIZATION =====
    print("\nApplying feature normalization...")
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'training_data/feature_scaler.pkl')
    print("✓ Saved feature scaler to 'training_data/feature_scaler.pkl'")
    
    print(f"Before normalization - Range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"After normalization  - Range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
    print(f"After normalization  - Mean: {X_train_scaled.mean():.3f}, Std: {X_train_scaled.std():.3f}")
    # ===== END NORMALIZATION =====

    # After scaling, check what percentage of data is in reasonable range
    reasonable_mask = (X_train_scaled >= -5) & (X_train_scaled <= 5)
    percent_reasonable = reasonable_mask.mean() * 100
    print(f"Percentage of features in [-5, 5] range: {percent_reasonable:.2f}%")

    # Check the 95th percentile to see where most data lies
    print(f"95th percentile of absolute values: {np.percentile(np.abs(X_train_scaled), 95):.3f}")
        
    # Convert to PyTorch tensors (using SCALED features)
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Print data statistics
    print("\nData Statistics:")
    print(f"Features range: [{X_train_tensor.min():.3f}, {X_train_tensor.max():.3f}]")
    print(f"Targets range: [{y_train_tensor.min():.3f}, {y_train_tensor.max():.3f}]")
    print(f"Mean target: {y_train_tensor.mean():.3f}")
    
    # Feature names for reference
    print(f"\nFeature dimensions: {len(feature_columns)}")
    print("Features used:", feature_columns)
    
    return train_loader, val_loader, X_train_tensor.shape[1], feature_columns

def save_data_stats(train_loader, val_loader, input_dim, feature_columns):
    """Save data statistics and metadata for reference"""
    stats = {
        'input_dim': input_dim,
        'feature_columns': feature_columns,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'batch_size': train_loader.batch_size
    }
    
    with open('training_data/mlp_data_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"\nSaved data stats: {stats}")

def inspect_data_samples(train_loader, num_samples=3):
    """Inspect a few data samples to verify everything looks correct"""
    print("\nInspecting data samples...")
    
    for batch_idx, (features, targets) in enumerate(train_loader):
        if batch_idx >= 1:  # Just look at first batch
            break
            
        print(f"Batch {batch_idx + 1}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Sample features (first 5 values): {features[0][:5]}")
        print(f"  Sample target: {targets[0]:.4f}")
        
        # Print feature ranges in this batch
        print(f"  Feature ranges - Min: {features.min():.3f}, Max: {features.max():.3f}")
        print(f"  Target ranges - Min: {targets.min():.3f}, Max: {targets.max():.3f}")

def main():
    print("=" * 60)
    print("MLP TRAINING DATA PREPARATION (WITH NORMALIZATION)")
    print("=" * 60)
    
    # Load and prepare data
    train_loader, val_loader, input_dim, feature_columns = load_and_prepare_mlp_data()
    
    # Save data statistics
    save_data_stats(train_loader, val_loader, input_dim, feature_columns)
    
    # Inspect samples
    inspect_data_samples(train_loader)
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run train_mlp.py to train the MLP model")
    print("2. Use the trained MLP to create GNN node features")
    print("3. Train the GNN on the MLP-generated features")
    
    return train_loader, val_loader, input_dim

if __name__ == "__main__":
    train_loader, val_loader, input_dim = main()