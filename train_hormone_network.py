#!/usr/bin/env python
"""
Train the Hormone-to-Mood Network with biological data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import sys

sys.path.append('/root/openhermes_backend')

from app.neurochemistry.scalable_hormone_network import ScalableHormoneNetwork, ExtendedMoodComponents, create_network

def train_network():
    print("="*60)
    print("TRAINING HORMONE-TO-MOOD NEURAL NETWORK")
    print("="*60)
    
    # Load the biological training data
    print("\n1. Loading biological training data...")
    data = torch.load('biological_training_data.pth')
    inputs = data['inputs']
    targets = data['targets']
    print(f"   Loaded {len(inputs)} samples")
    print(f"   Input shape: {inputs.shape}")
    print(f"   Target shape: {targets.shape}")
    
    # Split into train/val/test (80/10/10)
    dataset = TensorDataset(inputs, targets)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create network using preset
    print("\n2. Creating network...")
    model = create_network('medium', input_dim=5, output_dim=30)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Remove verbose parameter - not all PyTorch versions have it
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    # Training loop
    print("\n3. Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 25
    
    for epoch in range(200):
        # Training phase
        model.train()
        train_loss = 0
        for hormones, expected_moods in train_loader:
            optimizer.zero_grad()
            predicted_moods = model(hormones)
            loss = criterion(predicted_moods, expected_moods)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for hormones, expected_moods in val_loader:
                predicted_moods = model(hormones)
                loss = criterion(predicted_moods, expected_moods)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        # Step scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Print if learning rate changed
        if new_lr != old_lr:
            print(f"   Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'best_hormone_model.pth')
            # Save config separately
            config = model.get_config()
            with open('model_config.json', 'w') as f:
                import json
                json.dump(config, f)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 20 == 0 or epoch < 10:
            print(f"   Epoch {epoch:3d}: Train={avg_train:.6f}, Val={avg_val:.6f}, Best={best_val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Test the model
    print("\n4. Testing on holdout set...")
    model.eval()
    test_loss = 0
    sample_predictions = []
    
    with torch.no_grad():
        for i, (hormones, expected_moods) in enumerate(test_loader):
            predicted_moods = model(hormones)
            loss = criterion(predicted_moods, expected_moods)
            test_loss += loss.item()
            
            # Save first batch for inspection
            if i == 0:
                sample_predictions = predicted_moods[:3]
    
    avg_test = test_loss / len(test_loader)
    
    # Show sample predictions
    print("\n5. Sample predictions from test set:")
    for i, pred in enumerate(sample_predictions):
        mood_state = ExtendedMoodComponents.describe_mood_state(pred.numpy(), threshold=0.3)
        print(f"   Sample {i+1}: {mood_state['description']}")
    
    print(f"\n✅ Training complete!")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    print(f"   Test loss: {avg_test:.6f}")
    print(f"   Model saved to best_hormone_model.pth")
    
    return model

if __name__ == "__main__":
    model = train_network()
