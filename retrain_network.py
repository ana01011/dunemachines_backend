"""
Retrain with better regularization to prevent overfitting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import sys

sys.path.append('/root/openhermes_backend')

from app.neurochemistry.scalable_hormone_network import create_network, ExtendedMoodComponents

def retrain_network():
    print("="*60)
    print("RETRAINING WITH BETTER REGULARIZATION")
    print("="*60)
    
    # Load data
    data = torch.load('biological_training_data.pth')
    inputs = data['inputs']
    targets = data['targets']
    
    # Normalize inputs to 0-1 range
    inputs = inputs / 100.0
    
    print(f"Data: {len(inputs)} samples")
    print(f"Input range: [{inputs.min():.2f}, {inputs.max():.2f}]")
    print(f"Target range: [{targets.min():.2f}, {targets.max():.2f}]")
    
    # Split data
    dataset = TensorDataset(inputs, targets)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create smaller network to prevent overfitting
    print("\nCreating smaller network...")
    model = create_network(
        'small',  # Use smaller preset
        input_dim=5,
        output_dim=30,
        dropout_rate=0.3  # More dropout
    )
    
    # Use weight decay for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)
    
    # Use a combination of MSE and L1 loss for better gradients
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    def combined_loss(pred, target):
        return 0.7 * mse_loss(pred, target) + 0.3 * l1_loss(pred, target)
    
    print("\nTraining with regularization...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(150):
        # Training
        model.train()
        train_loss = 0
        
        for hormones, expected_moods in train_loader:
            optimizer.zero_grad()
            
            # Add noise to inputs for regularization
            noise = torch.randn_like(hormones) * 0.02
            hormones_noisy = torch.clamp(hormones + noise, 0, 1)
            
            predicted_moods = model(hormones_noisy)
            loss = combined_loss(predicted_moods, expected_moods)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for hormones, expected_moods in val_loader:
                predicted_moods = model(hormones)
                loss = combined_loss(predicted_moods, expected_moods)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        scheduler.step(avg_val)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'hormone_model_v2.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Train={avg_train:.4f}, Val={avg_val:.4f}")
        
        if patience_counter >= 30:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\n✅ Retraining complete! Best val loss: {best_val_loss:.4f}")
    return model

if __name__ == "__main__":
    retrain_network()
