import torch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from train_mlp import ImprovedEdgeMLP  # Make sure to import your model

def test_mlp_model_standalone():
    """Test the trained MLP model on test data (standalone version)"""
    print("=" * 70)
    print("MLP MODEL TESTING - STANDALONE VALIDATION")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load test data
        print("\nLoading test data...")
        if not os.path.exists('training_data/mlp_test.pkl'):
            raise FileNotFoundError("Test data file not found!")
            
        with open('training_data/mlp_test.pkl', 'rb') as f:
            test_data = pickle.load(f)
        
        df_test = pd.DataFrame(test_data)
        print(f"Test samples: {len(df_test):,}")
        
        # Load the trained model
        print("Loading trained MLP model...")
        if not os.path.exists('training_data/best_mlp_model.pth'):
            raise FileNotFoundError("Model file not found!")
            
        checkpoint = torch.load('training_data/best_mlp_model.pth', map_location=device)
        
        # Recreate model architecture
        model = ImprovedEdgeMLP(
            input_dim=checkpoint.get('input_dim', 10),
            hidden_dims=[256, 128, 64], 
            embedding_dim=32
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Load feature scaler
        print("Loading feature scaler...")
        if not os.path.exists('training_data/feature_scaler.pkl'):
            raise FileNotFoundError("Scaler file not found!")
            
        scaler = joblib.load('training_data/feature_scaler.pkl')
        
        # Prepare test features and targets
        feature_columns = checkpoint.get('feature_columns', [
            'length', 'maxspeed', 'straightness', 'highway_type', 
            'lanes', 'width', 'traffic_flow', 'travel_time', 'time_loss', 'waiting_time'
        ])
        
        # Check if all required columns exist
        missing_cols = [col for col in feature_columns if col not in df_test.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in test data: {missing_cols}")
            
        if 'target' not in df_test.columns:
            raise ValueError("Target column 'target' not found in test data!")
        
        # Extract features and targets
        X_test = df_test[feature_columns].values.astype(np.float32)
        y_test = df_test['target'].values.astype(np.float32)
        
        # Normalize features
        X_test_scaled = scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        print(f"Test features: {X_test_tensor.shape}")
        print(f"Test targets: {y_test_tensor.shape}")
        
        # Run predictions
        print("\nRunning predictions...")
        with torch.no_grad():
            predictions = model(X_test_tensor).cpu().numpy().flatten()
            targets = y_test_tensor.cpu().numpy().flatten()
        
        # Calculate comprehensive metrics
        print("\n" + "="*50)
        print("TEST RESULTS - HOLDOUT SET")
        print("="*50)
        
        # Basic metrics
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Emergency-specific metrics
        critical_errors = np.mean((predictions > 0.7) & (targets < 0.3))
        safe_predictions = np.mean((predictions < 0.4) & (targets < 0.4))
        accuracy_within_0_1 = np.mean(np.abs(predictions - targets) <= 0.1)
        accuracy_within_0_05 = np.mean(np.abs(predictions - targets) <= 0.05)
        
        print(f"Mean Absolute Error (MAE):     {mae:.4f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²):               {r2:.4f}")
        print(f"Accuracy within 0.1:          {accuracy_within_0_1:.1%}")
        print(f"Accuracy within 0.05:         {accuracy_within_0_05:.1%}")
        print(f"Critical Errors:              {critical_errors:.4f}")
        print(f"Safe Predictions:             {safe_predictions:.4f}")
        
        # Compare with validation performance
        val_mae = checkpoint.get('val_mae', 0.0)
        val_r2 = checkpoint.get('val_r2', 0.0)
        
        print(f"\nComparison with Validation:")
        print(f"  Validation MAE: {val_mae:.4f}")
        print(f"  Test MAE:       {mae:.4f}")
        print(f"  Validation R²:  {val_r2:.4f}") 
        print(f"  Test R²:        {r2:.4f}")
        
        # Performance assessment
        print(f"\n" + "="*50)
        print("MODEL ASSESSMENT")
        print("="*50)
        
        if mae <= 0.015:
            assessment = "EXCELLENT"
            print("✅ EXCELLENT: Model generalizes perfectly to unseen data")
            print("   Ready for GNN pipeline!")
        elif mae <= 0.025:
            assessment = "GOOD"
            print("✅ GOOD: Model generalizes well to unseen data") 
            print("   Ready for GNN pipeline!")
        elif mae <= 0.035:
            assessment = "ACCEPTABLE"
            print("⚠️  ACCEPTABLE: Some performance drop on test data")
            print("   Proceed with GNN pipeline")
        else:
            assessment = "POOR"
            print("❌ POOR: Significant performance drop on test data")
            print("   Consider retraining or investigating data issues")
        
        if critical_errors == 0:
            print("✅ PERFECT SAFETY: Zero critical routing errors")
        else:
            print(f"⚠️  SAFETY CONCERN: {critical_errors:.4f} critical errors")
        
        # Plot predictions vs targets
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.scatter(targets, predictions, alpha=0.6, s=10)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
        plt.xlabel('True Emergency Suitability')
        plt.ylabel('Predicted Emergency Suitability')
        plt.title('Predictions vs True Values')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        errors = predictions - targets
        plt.hist(errors, bins=50, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        road_types = df_test['highway_type'].values
        plt.scatter(road_types, errors, alpha=0.6, s=10)
        plt.xlabel('Highway Type')
        plt.ylabel('Prediction Error')
        plt.title('Errors by Road Type')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_data/mlp_test_results.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Test results saved to: training_data/mlp_test_results.png")
        
        # Show some example predictions
        print(f"\n" + "="*50)
        print("SAMPLE PREDICTIONS")
        print("="*50)
        
        sample_indices = np.random.choice(len(predictions), min(5, len(predictions)), replace=False)
        for i, idx in enumerate(sample_indices):
            print(f"Sample {i+1}: True={targets[idx]:.3f}, Pred={predictions[idx]:.3f}, "
                  f"Error={errors[idx]:.3f}")
        
        return {
            'mae': mae,
            'r2': r2,
            'critical_errors': critical_errors,
            'predictions': predictions,
            'targets': targets,
            'assessment': assessment
        }
        
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        return None

if __name__ == "__main__":
    test_results = test_mlp_model_standalone()
    
    if test_results is not None:
        print(f"\n" + "="*70)
        print("MLP TESTING COMPLETE!")
        print("="*70)
        if test_results['mae'] <= 0.025 and test_results['critical_errors'] == 0:
            print("✅ MLP is READY for GNN pipeline!")
            print("Next: Run generate_gnn_embeddings.py")
        else:
            print("⚠️  Check model performance before proceeding to GNN")
        print("="*70)
    else:
        print("❌ Testing failed - check errors above")