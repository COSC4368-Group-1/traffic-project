
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle
from gnn_model import EmergencyGNNSimple, EmergencyGNN, EmergencyGNNEnhanced, EmergencyRoutingLoss
from sklearn.preprocessing import StandardScaler

class FixedEmbeddingGNNTrainer:
    def __init__(self, model_class='enhanced'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = model_class
        self.setup_directories()
        
    def setup_directories(self):
        """Create directories for saving outputs"""
        os.makedirs('training_data/gnn_training', exist_ok=True)
        os.makedirs('training_data/gnn_plots', exist_ok=True)

    def normalize_embeddings(self, graph_data):
        """Normalize MLP embeddings to have mean=0, std=1"""
        print("🔧 Normalizing MLP embeddings...")
        
        embeddings = graph_data['node_features'].cpu().numpy()
        
        # Calculate current stats
        original_mean = embeddings.mean()
        original_std = embeddings.std()
        
        print(f"   Before - Mean: {original_mean:.3f}, Std: {original_std:.3f}")
        
        # Standardize
        embeddings_normalized = (embeddings - original_mean) / original_std
        
        # Update graph data
        graph_data['node_features'] = torch.FloatTensor(embeddings_normalized)
        
        # Verify
        final_mean = graph_data['node_features'].mean().item()
        final_std = graph_data['node_features'].std().item()
        print(f"   After  - Mean: {final_mean:.3f}, Std: {final_std:.3f}")
        
        return graph_data

    def load_gnn_data(self):
        """Load the GNN graph data with normalized MLP embeddings"""
        print("📊 Loading GNN graph data...")
        
        try:
            graph_data = torch.load('training_data/gnn_graph_data.pt', map_location='cpu', weights_only=False)
            print("   ✓ Loaded graph with MLP embeddings")
        except Exception as e:
            print(f"   ❌ Torch load failed: {e}")
            raise RuntimeError("Could not load graph data")
        
        # NORMALIZE THE EMBEDDINGS
        graph_data = self.normalize_embeddings(graph_data)
        
        # Move to device
        graph_data['node_features'] = graph_data['node_features'].to(self.device)
        graph_data['edge_index'] = graph_data['edge_index'].to(self.device)
        graph_data['edge_labels'] = graph_data['edge_labels'].to(self.device)
        graph_data['train_mask'] = graph_data['train_mask'].to(self.device)
        graph_data['val_mask'] = graph_data['val_mask'].to(self.device)
        
        print(f"   ├── Nodes: {graph_data['num_nodes']:,}")
        print(f"   ├── Embeddings: {graph_data['node_features'].shape[1]}-dim (normalized)")
        print(f"   ├── Training edges: {graph_data['train_mask'].sum().item():,}")
        print(f"   └── Validation edges: {graph_data['val_mask'].sum().item():,}")
        
        return graph_data

    def create_data_object(self, graph_data):
        """Convert to PyTorch Geometric Data object"""
        train_edge_indices = graph_data['train_mask'].nonzero().squeeze()
        val_edge_indices = graph_data['val_mask'].nonzero().squeeze()
        
        edge_index = graph_data['edge_index']
        train_edge_pairs = edge_index[:, train_edge_indices]
        val_edge_pairs = edge_index[:, val_edge_indices]
        
        data = Data(
            x=graph_data['node_features'],
            edge_index=edge_index,
            edge_attr=None,
            y=graph_data['edge_labels'],
            train_mask=graph_data['train_mask'],
            val_mask=graph_data['val_mask'],
            train_edge_indices=train_edge_indices,
            val_edge_indices=val_edge_indices,
            train_edge_pairs=train_edge_pairs,
            val_edge_pairs=val_edge_pairs
        )
        
        return data

    def calculate_emergency_metrics(self, predictions, targets):
        """Enhanced metrics for emergency routing evaluation"""
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        # Ensure arrays for single sample case
        if predictions.ndim == 0:
            predictions = np.array([predictions])
        if targets.ndim == 0:
            targets = np.array([targets])
        
        # Standard regression metrics
        mae = float(np.mean(np.abs(predictions - targets)))
        mse = float(np.mean((predictions - targets)**2))
        rmse = float(np.sqrt(mse))
        
        # Emergency-specific metrics
        critical_errors = float(np.mean((predictions > 0.7) & (targets < 0.3)))
        safe_predictions = float(np.mean((predictions < 0.4) & (targets < 0.4)))
        high_conf_correct = float(np.mean((np.abs(predictions - targets) <= 0.05) & (targets < 0.5)))
        accuracy_within_0_1 = float(np.mean(np.abs(predictions - targets) <= 0.1))
        accuracy_within_0_15 = float(np.mean(np.abs(predictions - targets) <= 0.15))
        
        return {
            'mae': mae, 'mse': mse, 'rmse': rmse,
            'critical_error_rate': critical_errors,
            'safe_prediction_rate': safe_predictions,
            'high_confidence_accuracy': high_conf_correct,
            'accuracy_within_0_1': accuracy_within_0_1,
            'accuracy_within_0_15': accuracy_within_0_15
        }

    def create_model(self, node_dim=32):
        """Create GNN model with 32-dim embeddings"""
        if self.model_class == 'advanced':
            model = EmergencyGNN(node_dim=node_dim, hidden_dim=128)
            print("   🏗️  Using Advanced GNN architecture")
        elif self.model_class == 'enhanced':
            model = EmergencyGNNEnhanced(node_dim=node_dim, hidden_dim=256, num_layers=3)
            print("   🏗️  Using Enhanced GNN architecture")
        else:
            model = EmergencyGNNSimple(node_dim=node_dim, hidden_dim=128)
            print("   🏗️  Using Simple GNN architecture")
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   📈 Trainable parameters: {trainable_params:,}")
        
        return model.to(self.device)

    def train_epoch(self, model, data, criterion, optimizer):
        """Train for one epoch with dynamic batch sizing"""
        model.train()
        total_loss = 0
        
        num_train_edges = data.train_edge_pairs.shape[1]
        batch_size = max(64, num_train_edges // 25)
        
        # Shuffle training edges
        perm = torch.randperm(num_train_edges)
        train_pairs_shuffled = data.train_edge_pairs[:, perm]
        train_labels_shuffled = data.y[data.train_edge_indices][perm]
        
        print(f"   📦 Training: {batch_size} edges/batch ({num_train_edges//batch_size} batches)")
        
        with tqdm(total=num_train_edges, desc="🚂 Training", unit="edge") as pbar:
            for i in range(0, num_train_edges, batch_size):
                batch_pairs = train_pairs_shuffled[:, i:i + batch_size]
                batch_labels = train_labels_shuffled[i:i + batch_size]
                
                optimizer.zero_grad()
                predictions = model(data.x, data.edge_index, batch_pairs)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item() * batch_pairs.shape[1]
                pbar.update(batch_pairs.shape[1])
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_train_edges

    def validate_epoch(self, model, data, criterion):
        """Validate for one epoch with dynamic batch sizing"""
        model.eval()
        total_loss = 0
        
        num_val_edges = data.val_edge_pairs.shape[1]
        batch_size = min(2048, num_val_edges)
        
        print(f"   📦 Validation: {batch_size} edges/batch ({num_val_edges//batch_size} batches)")
        
        all_predictions, all_targets = [], []
        
        with torch.no_grad():
            for i in range(0, num_val_edges, batch_size):
                batch_pairs = data.val_edge_pairs[:, i:i + batch_size]
                batch_labels = data.y[data.val_edge_indices][i:i + batch_size]
                
                predictions = model(data.x, data.edge_index, batch_pairs)
                loss = criterion(predictions, batch_labels)
                total_loss += loss.item() * batch_pairs.shape[1]
                
                all_predictions.append(predictions)
                all_targets.append(batch_labels)
        
        all_predictions = torch.cat([p.reshape(-1) for p in all_predictions])
        all_targets = torch.cat([t.reshape(-1) for t in all_targets])
        
        avg_loss = total_loss / num_val_edges
        metrics = self.calculate_emergency_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics, all_predictions, all_targets

    def plot_training_progress(self, history, epoch):
        """Create a single comprehensive training plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Houston Emergency Routing GNN - Training Progress (Epoch {epoch})', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curves (top-left)
        ax1.plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2, alpha=0.8)
        ax1.plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2, alpha=0.8)
        ax1.set_title('Training & Validation Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # Plot 2: MAE and RMSE (top-right)
        epochs = range(1, len(history['val_mae']) + 1)
        ax2.plot(epochs, history['val_mae'], 'g-', label='MAE', linewidth=2, alpha=0.8)
        ax2.plot(epochs, history['val_rmse'], 'orange', label='RMSE', linewidth=2, alpha=0.8)
        ax2.set_title('Validation Metrics', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Emergency-specific metrics (bottom-left)
        ax3.plot(epochs, history['val_critical_errors'], 'r-', label='Critical Errors', linewidth=2, alpha=0.8)
        ax3.plot(epochs, history['val_safe_predictions'], 'g-', label='Safe Predictions', linewidth=2, alpha=0.8)
        ax3.plot(epochs, history['val_high_conf_accuracy'], 'purple', label='High Conf Acc', linewidth=2, alpha=0.8)
        ax3.set_title('Emergency Routing Metrics', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Accuracy within tolerance (bottom-right)
        ax4.plot(epochs, history['val_accuracy_0_1'], 'b-', label='±0.1 Accuracy', linewidth=2, alpha=0.8)
        ax4.plot(epochs, history['val_accuracy_0_15'], 'orange', label='±0.15 Accuracy', linewidth=2, alpha=0.8)
        ax4.set_title('Prediction Accuracy', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(f'training_data/gnn_plots/training_progress_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also create a final summary plot
        if epoch >= 10:  # Only create summary after some training
            self.create_final_summary_plot(history, epoch)

    def create_final_summary_plot(self, history, epoch):
        """Create a beautiful final summary plot"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create a beautiful gradient background
        ax.set_facecolor('#f8f9fa')
        
        # Plot main metrics
        epochs = range(1, len(history['val_mae']) + 1)
        
        # Normalize metrics for better visualization
        norm_mae = np.array(history['val_mae']) / max(history['val_mae'])
        norm_loss = np.array(history['val_loss']) / max(history['val_loss'])
        
        ax.plot(epochs, norm_mae, 'g-', label='Normalized MAE', linewidth=3, alpha=0.9)
        ax.plot(epochs, norm_loss, 'r-', label='Normalized Loss', linewidth=3, alpha=0.9)
        ax.plot(epochs, history['val_accuracy_0_1'], 'b-', label='±0.1 Accuracy', linewidth=3, alpha=0.9)
        
        ax.set_title('GNN Training Summary - Houston Emergency Routing', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add some statistics
        best_epoch = np.argmin(history['val_loss']) + 1
        best_loss = min(history['val_loss'])
        best_mae = history['val_mae'][best_epoch-1]
        best_acc = history['val_accuracy_0_1'][best_epoch-1]
        
        stats_text = f'Best Epoch: {best_epoch}\nBest Loss: {best_loss:.6f}\nBest MAE: {best_mae:.4f}\nBest Acc: {best_acc:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'training_data/gnn_plots/final_summary_epoch_{epoch}.png', 
                    dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()

    def save_training_history(self, history):
        """Save training history to JSON with proper type conversion"""
        # Convert numpy types to native Python types for JSON serialization
        history_serializable = {}
        for key, values in history.items():
            if isinstance(values, list) and len(values) > 0 and hasattr(values[0], 'item'):
                # Convert numpy arrays/scalars to native Python types
                history_serializable[key] = [float(v.item()) if hasattr(v, 'item') else float(v) for v in values]
            else:
                history_serializable[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in values]
        
        with open('training_data/gnn_training/training_history.json', 'w') as f:
            json.dump(history_serializable, f, indent=2)

    def print_epoch_summary(self, epoch, train_loss, val_loss, metrics, epoch_time, best_so_far=False):
        """Print detailed epoch summary"""
        status = "🏆 NEW BEST!" if best_so_far else ""
        
        print(f"📅 Epoch {epoch:3d} | Time: {epoch_time:5.1f}s {status}")
        print(f"   ├── Loss:    Train {train_loss:.4f} | Val {val_loss:.4f}")
        print(f"   ├── MAE:     {metrics['mae']:.4f}")
        print(f"   ├── RMSE:    {metrics['rmse']:.4f}")
        print(f"   ├── Critical Errors: {metrics['critical_error_rate']:.3f}")
        print(f"   ├── Safe Predictions: {metrics['safe_prediction_rate']:.3f}")
        print(f"   ├── High Conf Acc: {metrics['high_confidence_accuracy']:.3f}")
        print(f"   └── Acc (±0.1): {metrics['accuracy_within_0_1']:.3f}")

    def train(self):
        print("="*80)
        print("🚑 HOUSTON EMERGENCY ROUTING - GNN TRAINING (32-DIM EMBEDDINGS)")
        print("="*80)
        print(f"💻 Using device: {self.device}")
        print("🎯 CRITICAL: Using 32-dim MLP embeddings from gnn_embeddings_train_val.pkl")
        
        # Load data
        graph_data = self.load_gnn_data()
        if graph_data is None:
            print("❌ Failed to load properly embedded data. Cannot continue.")
            return
        
        data = self.create_data_object(graph_data)
        data = data.to(self.device)
        
        # Verify feature dimension
        if data.x.shape[1] != 32:
            print(f"❌ ERROR: Feature dimension is {data.x.shape[1]}, but should be 32!")
            return
        
        print("✅ SUCCESS: Using 32-dim MLP embeddings!")
        
        # Create model - WITH 32 DIMENSIONS
        model = self.create_model(node_dim=32)
        
        # FIXED: Use simpler loss function for stability
        print("🔧 Using STABLE EmergencyRoutingLoss")
        criterion = EmergencyRoutingLoss(critical_penalty=2.0, safe_weight=0.1, variance_weight=0.001)

        # FIXED: Lower learning rate and different optimizer
        optimizer = optim.Adam(
            model.parameters(), 
            lr=0.0005,  # Reduced from 0.001
            weight_decay=1e-5,
        )

        # FIXED: More patient scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            patience=15,  # Increased patience
            factor=0.5,
            min_lr=1e-7,
        )
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': [],
            'val_critical_errors': [], 'val_safe_predictions': [], 
            'val_high_conf_accuracy': [], 'val_accuracy_0_1': [], 'val_accuracy_0_15': [],
            'learning_rate': []
        }
        
        num_epochs = 200
        best_val_loss = float('inf')
        patience = 30  # Increased patience
        patience_counter = 0
        
        print("\n🎬 Starting GNN Training with MLP embeddings...")
        print("-" * 80)
        
        try:
            for epoch in range(num_epochs):
                epoch_start = time.time()
                
                # Training phase
                train_loss = self.train_epoch(model, data, criterion, optimizer)
                
                # Validation phase
                val_loss, metrics, val_predictions, val_targets = self.validate_epoch(model, data, criterion)
                
                # Update scheduler
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Store history
                history['train_loss'].append(float(train_loss))
                history['val_loss'].append(float(val_loss))
                history['val_mae'].append(float(metrics['mae']))
                history['val_rmse'].append(float(metrics['rmse']))
                history['val_critical_errors'].append(float(metrics['critical_error_rate']))
                history['val_safe_predictions'].append(float(metrics['safe_prediction_rate']))
                history['val_high_conf_accuracy'].append(float(metrics['high_confidence_accuracy']))
                history['val_accuracy_0_1'].append(float(metrics['accuracy_within_0_1']))
                history['val_accuracy_0_15'].append(float(metrics['accuracy_within_0_15']))
                history['learning_rate'].append(float(current_lr))
                
                epoch_time = time.time() - epoch_start
                
                # FIXED: Better early stopping logic
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_mae': metrics['mae'],
                        'val_rmse': metrics['rmse'],
                        'val_critical_errors': metrics['critical_error_rate'],
                        'val_safe_predictions': metrics['safe_prediction_rate'],
                        'val_predictions': val_predictions.cpu(),
                        'val_targets': val_targets.cpu(),
                        'training_history': history
                    }, 'training_data/best_gnn_model.pth')
                    
                    print("💾 Saved new best model!")
                else:
                    patience_counter += 1
                
                self.print_epoch_summary(epoch + 1, train_loss, val_loss, metrics, epoch_time, is_best)
                
                # FIXED: Save checkpoints more frequently at start
                if (epoch + 1) % 5 == 0 or (epoch + 1) <= 10 or is_best:
                    self.plot_training_progress(history, epoch + 1)
                    self.save_training_history(history)
                    print("📈 Saved training progress plot and history")
                
                # FIXED: Don't stop too early, allow for some bouncing
                if patience_counter >= patience and epoch > 50:  # Minimum 50 epochs
                    print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
                    break
                
                print("-" * 80)
        
        except KeyboardInterrupt:
            print(f"\n🛑 Training interrupted by user at epoch {epoch + 1}")
        
        # Final summary
        self.plot_training_progress(history, epoch + 1)
        self.save_training_history(history)
        
        try:
            checkpoint = torch.load('training_data/best_gnn_model.pth', weights_only=False)
            best_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['val_loss']
            best_val_mae = checkpoint['val_mae']
        except:
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_val_mae = metrics['mae']
        
        print(f"\n" + "="*80)
        print("🎉 GNN TRAINING COMPLETED!")
        print("="*80)
        print(f"🏆 Best Performance (Epoch {best_epoch}):")
        print(f"   ├── Validation Loss: {best_val_loss:.6f}")
        print(f"   ├── MAE: {best_val_mae:.6f}")
        print(f"   └── Target MLP MAE: 0.0095")
        
        improvement = ((0.0095 - best_val_mae) / 0.0095) * 100
        print(f"   📈 Improvement needed: {improvement:+.1f}% to match MLP")


def main():
    trainer = FixedEmbeddingGNNTrainer(model_class='enhanced')  
    trainer.train()

if __name__ == "__main__":
    main()