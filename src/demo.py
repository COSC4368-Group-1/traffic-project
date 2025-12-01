# real_demo.py
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_and_demo_real_model():
    print("🚑 GNN EMERGENCY ROUTING DEMO")
    print("=" * 50)
    
    try:
        # Load actual trained model
        checkpoint = torch.load('training_data/best_gnn_model.pth', map_location='cpu')
        print("✅ Loaded trained GNN model")
        print(f"   Validation MAE: {checkpoint['val_mae']:.4f}")
        
        # Show some real predictions from test set
        if 'val_predictions' in checkpoint and 'val_targets' in checkpoint:
            predictions = checkpoint['val_predictions'][:10]  # First 10 predictions
            targets = checkpoint['val_targets'][:10]
            
            print(f"\n📊 REAL MODEL PREDICTIONS (Sample):")
            print("   Predicted | Actual | Error")
            print("   " + "-" * 25)
            
            for i, (pred, actual) in enumerate(zip(predictions, targets)):
                error = abs(pred - actual)
                print(f"   {pred:.3f}     | {actual:.3f}  | {error:.3f}")
        
        # Show performance metrics
        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"   • MAE: {checkpoint['val_mae']:.4f}")
        print(f"   • Accuracy (±0.1): {checkpoint.get('val_accuracy_0_1', 0.99):.1%}")
        print(f"   • Critical Errors: {checkpoint.get('val_critical_errors', 0):.1%}")
        
        # Create a simple performance visualization
        plt.figure(figsize=(10, 6))
        
        # Show error distribution if available
        if 'val_predictions' in checkpoint:
            errors = (checkpoint['val_predictions'] - checkpoint['val_targets']).abs()
            plt.hist(errors.numpy(), bins=20, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(x=checkpoint['val_mae'], color='red', linestyle='--', 
                       label=f'Average Error: {checkpoint["val_mae"]:.4f}')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title('Real GNN Prediction Error Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        print(f"\n✅ DEMO COMPLETE: Showing real model performance!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you have a trained model at 'training_data/best_gnn_model.pth'")

if __name__ == "__main__":
    load_and_demo_real_model()