# check_features.py
import torch

graph_data = torch.load('training_data/gnn_graph_data.pt', map_location='cpu', weights_only=False)
features = graph_data['node_features']

print("🔍 CURRENT GRAPH FEATURES:")
print(f"   Shape: {features.shape}")
print(f"   Range: {features.min():.3f} to {features.max():.3f}")
print(f"   Stats: mean={features.mean():.3f}, std={features.std():.3f}")

# Check if these look like embeddings or raw features
if features.shape[1] == 10:
    print("❌ These are RAW FEATURES (10-dim road attributes)")
    print("   You're missing the MLP embeddings!")
elif features.shape[1] == 32:
    print("✅ These are MLP EMBEDDINGS (32-dim learned representations)")
    print("   But they don't look properly normalized")
else:
    print(f"❓ Unknown feature type: {features.shape[1]} dimensions")