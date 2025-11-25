"""
Simple training script for CNN-GNN traffic model

Optimized for small datasets (5 images)
"""

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import argparse
from pathlib import Path

from cnn_gnn_model import CNNGNNFusionModel
from data_loader import TrafficImageGNNDataset, custom_collate_fn
from torch.utils.data import DataLoader, Subset


def main():
    parser = argparse.ArgumentParser(description='Train CNN-GNN traffic model')
    parser.add_argument('--data-dir', type=str, default='raw_data/Houston_TX_USA',
                       help='Data directory path')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (use 1 for small dataset)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Prepare data
    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, "traffic_images")
    edges_file = os.path.join(data_dir, "edges_with_traffic.geojson")
    
    if not os.path.exists(images_dir) or not os.path.exists(edges_file):
        print(f"Error: Data not found in {data_dir}")
        print("Make sure to run data_grab.py first")
        return
    
    print(f"\nLoading data from: {data_dir}")
    
    # Create dataset
    dataset = TrafficImageGNNDataset(
        images_dir=images_dir,
        edges_geojson=edges_file
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # For small datasets, use all data for training and validation
    if len(dataset) <= 5:
        # Use 80/20 split
        num_train = max(1, len(dataset) * 80 // 100)
        train_indices = list(range(num_train))
        val_indices = list(range(num_train, len(dataset)))
        
        if len(val_indices) == 0:
            val_indices = train_indices  # Use same for both if only 1 sample
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices) if val_indices else Subset(dataset, train_indices)
    else:
        split_idx = int(len(dataset) * 0.8)
        train_dataset = Subset(dataset, list(range(split_idx)))
        val_dataset = Subset(dataset, list(range(split_idx, len(dataset))))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}\n")
    
    # Create model
    model = CNNGNNFusionModel(
        cnn_feature_dim=256,
        gnn_input_dim=8,
        gnn_output_dim=256,
        fusion_dim=512,
        output_dim=1
    )
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params:,} parameters\n")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Logging
    log_dir = f'runs/traffic_model_lr{args.learning_rate}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    best_val_loss = float('inf')
    
    print("="*70)
    print(f"Training for {args.epochs} epochs...")
    print("="*70 + "\n")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        for batch_idx, (images, graph_data) in enumerate(train_loader):
            images = images.to(device)
            graph_data = graph_data.to(device)
            
            optimizer.zero_grad()
            
            predictions, node_embeddings, cnn_features, graph_embedding = model(
                images,
                graph_data.x,
                graph_data.edge_index
            )
            
            # Loss: MSE between predictions and average congestion level
            if hasattr(graph_data, 'y') and graph_data.y is not None:
                # Use mean congestion as target for each image
                target = graph_data.y.mean().unsqueeze(0).unsqueeze(0)
                target = target.expand(predictions.shape[0], 1)
                loss = F.mse_loss(predictions, target)
            else:
                # Random target for demo
                loss = F.mse_loss(predictions, torch.randn_like(predictions))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            num_batches += 1
        
        train_loss = total_train_loss / max(num_batches, 1)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Validation
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for images, graph_data in val_loader:
                images = images.to(device)
                graph_data = graph_data.to(device)
                
                predictions, _, _, _ = model(
                    images,
                    graph_data.x,
                    graph_data.edge_index
                )
                
                if hasattr(graph_data, 'y') and graph_data.y is not None:
                    target = graph_data.y.mean().unsqueeze(0).unsqueeze(0)
                    target = target.expand(predictions.shape[0], 1)
                    loss = F.mse_loss(predictions, target)
                else:
                    loss = F.mse_loss(predictions, torch.randn_like(predictions))
                
                total_val_loss += loss.item()
                num_val_batches += 1
        
        val_loss = total_val_loss / max(num_val_batches, 1)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model_best.pt')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save periodic checkpoints
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch}.pt')
    
    # Final save
    torch.save(model.state_dict(), 'model_final.pt')
    writer.close()
    
    print("\n" + "="*70)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved: model_best.pt, model_final.pt")
    print(f"Logs: {log_dir}")
    print("="*70)
    
    # Print usage info
    print("\nTo view training progress:")
    print("  tensorboard --logdir runs/")
    print("\nThen visit http://localhost:6006 in your browser")


if __name__ == "__main__":
    main()
