import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def debug_data_issues():
    print("=" * 70)
    print("DATA QUALITY INVESTIGATION - TRAINING vs VALIDATION")
    print("=" * 70)
    
    # Load training and validation data (NOT test)
    with open('training_data/mlp_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('training_data/mlp_val.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    df_train = pd.DataFrame(train_data)
    df_val = pd.DataFrame(val_data)
    
    # Only the features ACTUALLY used in MLP (updated!)
    MLP_FEATURES = [
        'length', 'maxspeed', 'straightness', 'highway_type', 
        'lanes', 'width', 'emergency_score',
        'traffic_flow', 'travel_time', 'time_loss', 'waiting_time'
    ]
    
    print(f"Training samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")
    print(f"MLP Features: {MLP_FEATURES}")
    
    # Check target distribution
    print("\n" + "="*50)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("="*50)
    print(f"Train target - Min: {df_train['target'].min():.3f}, Max: {df_train['target'].max():.3f}, Mean: {df_train['target'].mean():.3f}")
    print(f"Val target   - Min: {df_val['target'].min():.3f}, Max: {df_val['target'].max():.3f}, Mean: {df_val['target'].mean():.3f}")
    
    # Check traffic features (updated to match what we actually have)
    traffic_features = ['traffic_flow', 'travel_time', 'time_loss', 'waiting_time']
    
    print(f"\nTRAFFIC DATA ANALYSIS:")
    for feature in traffic_features:
        if feature in df_train.columns:
            zero_count_train = (df_train[feature] == 0).sum()
            zero_count_val = (df_val[feature] == 0).sum()
            print(f"  {feature:15s}: {zero_count_train/len(df_train)*100:.1f}% zeros (train), {zero_count_val/len(df_val)*100:.1f}% zeros (val)")
        else:
            print(f"  {feature:15s}: NOT FOUND in data")
    
    # Feature variances for MLP features only
    print(f"\nFEATURE VARIANCE ANALYSIS (MLP Features Only):")
    for feature in MLP_FEATURES:
        if feature in df_train.columns:
            train_variance = df_train[feature].var()
            val_variance = df_val[feature].var()
            zero_ratio_train = (df_train[feature] == 0).sum() / len(df_train)
            zero_ratio_val = (df_val[feature] == 0).sum() / len(df_val)
            print(f"  {feature:20s}: variance={train_variance:8.2f} (train), {val_variance:8.2f} (val)")
            print(f"  {'':20s}  zeros={zero_ratio_train:5.1%} (train), {zero_ratio_val:5.1%} (val)")
        else:
            print(f"  {feature:20s}: NOT FOUND in data")
    
    # Feature correlations with target
    print(f"\nFEATURE CORRELATIONS WITH TARGET:")
    for feature in MLP_FEATURES:
        if feature in df_train.columns:
            train_corr = df_train[feature].corr(df_train['target'])
            val_corr = df_val[feature].corr(df_val['target'])
            print(f"  {feature:20s}: {train_corr:+.4f} (train), {val_corr:+.4f} (val)")
        else:
            print(f"  {feature:20s}: NOT FOUND in data")
    
    # Data split quality check
    print(f"\nDATA SPLIT QUALITY:")
    print(f"  Target mean difference: {abs(df_train['target'].mean() - df_val['target'].mean()):.4f}")
    print(f"  Target std difference:  {abs(df_train['target'].std() - df_val['target'].std()):.4f}")
    
    # Baseline performance
    print(f"\nBASELINE PERFORMANCE:")
    baseline_mae_train = np.mean(np.abs(df_train['target'] - df_train['target'].mean()))
    baseline_mae_val = np.mean(np.abs(df_val['target'] - df_train['target'].mean()))  # Use train mean for val
    print(f"  Baseline MAE (predict mean): {baseline_mae_train:.4f} (train), {baseline_mae_val:.4f} (val)")
    
    # Check for data leakage
    print(f"\nDATA LEAKAGE CHECK:")
    train_edges = set(zip(df_train['osm_u'], df_train['osm_v']))
    val_edges = set(zip(df_val['osm_u'], df_val['osm_v']))
    overlapping_edges = train_edges.intersection(val_edges)
    print(f"  Unique edges in train: {len(train_edges)}")
    print(f"  Unique edges in val: {len(val_edges)}")
    print(f"  Overlapping edges: {len(overlapping_edges)} ({len(overlapping_edges)/len(train_edges)*100:.1f}% of train)")
    
    # Check for data leakage in features vs target
    print(f"\nDATA LEAKAGE ANALYSIS:")
    print("  Checking if MLP inputs have direct relationship with target...")
    
    # The features that were REMOVED to prevent leakage
    removed_features = ['traffic_speed', 'traffic_density', 'traffic_occupancy']
    leakage_found = False
    
    for feature in removed_features:
        if feature in df_train.columns:
            print(f"  ⚠️  LEAKAGE: {feature} found in data (should be removed)")
            leakage_found = True
        else:
            print(f"  ✓ No leakage: {feature} correctly removed")
    
    if not leakage_found:
        print("  ✓ SUCCESS: All data leakage features have been removed!")
        print("  ✓ MLP will have to LEARN patterns instead of memorizing formulas")

if __name__ == "__main__":
    debug_data_issues()