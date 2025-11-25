"""
Training script for CNN-GNN hybrid model

Handles:
- Model training loop
- Validation
- Checkpointing
- Logging
"""

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
import json

from cnn_gnn_model import CNNGNNFusionModel
from data_loader import create_dataloaders


class TrafficPredictor:
    """Training wrapper for CNN-GNN model"""
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        # Logging
        self.writer = SummaryWriter(config.get('log_dir', 'runs/traffic_model'))
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, graph_data) in enumerate(train_loader):
            # Move to device
            images = images.to(self.device)
            graph_data = graph_data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            predictions, node_embeddings, cnn_features, graph_embedding = self.model(
                images,
                graph_data.x,
                graph_data.edge_index
            )
            
            # Calculate loss - use actual graph targets
            if hasattr(graph_data, 'y') and graph_data.y is not None:
                y = graph_data.y.float()
                # Reshape predictions to match targets
                if predictions.shape[0] < y.shape[0]:
                    y = y[:predictions.shape[0]]
                elif predictions.shape[0] > y.shape[0]:
                    predictions = predictions[:y.shape[0]]
                
                loss = F.mse_loss(predictions.squeeze(), y.squeeze())
            else:
                # Fallback: predict average traffic speed
                loss = F.mse_loss(predictions, torch.ones_like(predictions) * 50)  # 50 km/h average
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f}")
        
        epoch_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
        
        return epoch_loss
    
    def validate(self, val_loader, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, graph_data in val_loader:
                images = images.to(self.device)
                graph_data = graph_data.to(self.device)
                
                predictions, _, _, _ = self.model(
                    images,
                    graph_data.x,
                    graph_data.edge_index
                )
                
                if hasattr(graph_data, 'y') and graph_data.y is not None:
                    target = graph_data.y[:predictions.shape[0]].unsqueeze(1)
                    loss = F.mse_loss(predictions, target)
                else:
                    loss = F.mse_loss(predictions, torch.randn_like(predictions))
                
                total_loss += loss.item()
                num_batches += 1
        
        val_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(f"model_best.pt")
            print(f"✓ Saved best model with val loss: {val_loss:.4f}")
        
        return val_loss
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop"""
        print(f"\nTraining for {num_epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader, epoch)
            
            # LR scheduling
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(f"model_epoch_{epoch}.pt")
        
        print("\n" + "=" * 60)
        print(f"Training complete! Best validation loss: {self.best_val_loss:.4f}")
        self.writer.close()
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': 0,  # TODO: track epoch number
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))


def main():
    parser = argparse.ArgumentParser(description='Train CNN-GNN traffic model')
    parser.add_argument('--data-dir', type=str, default='raw_data/Houston_TX_USA',
                       help='Data directory path')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    model = CNNGNNFusionModel(
        cnn_feature_dim=256,
        gnn_input_dim=8,
        gnn_output_dim=256,
        fusion_dim=512,
        output_dim=1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Prepare data
    images_dir = os.path.join(args.data_dir, 'traffic_images')
    edges_file = os.path.join(args.data_dir, 'edges_with_traffic.geojson')
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        print("Make sure to run data_grab.py first to generate traffic images")
        return
    
    if not os.path.exists(edges_file):
        print(f"Error: Edges file not found: {edges_file}")
        print("Make sure to run data_grab.py first to generate traffic data")
        return
    
    print(f"\nLoading data from: {args.data_dir}")
    
    try:
        train_loader, val_loader = create_dataloaders(
            images_dir=images_dir,
            edges_geojson=edges_file,
            batch_size=args.batch_size,
            split_ratio=0.8
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure traffic images have been generated")
        return
    
    # Training config
    config = {
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'log_dir': 'runs/traffic_model'
    }
    
    # Create trainer
    trainer = TrafficPredictor(model, device, config)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    trainer.train(train_loader, val_loader, args.epochs)
    
    # Save final model
    trainer.save_checkpoint('model_final.pt')
    print("✓ Training complete! Model saved to model_final.pt")


if __name__ == "__main__":
    main()
