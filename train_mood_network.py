#!/usr/bin/env python
"""
Train the Multi-Mood Neural Network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# Add project to path
sys.path.append('/root/openhermes_backend')

from app.neurochemistry.multi_mood_network import MultiMoodEmergenceNetwork, MoodComponents
from app.neurochemistry.training_data_generator import TrainingDataGenerator


class MoodNetworkTrainer:
    """
    Trainer for the multi-mood neural network
    """
    
    def __init__(self, learning_rate=0.001, batch_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = MultiMoodEmergenceNetwork().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        
        # Track training history
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_loader, val_loader, epochs=200):
        """
        Train the neural network
        """
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("=" * 60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_batches = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            self.val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.save_model('best_mood_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {avg_val_loss:.6f} | "
                      f"Best Val: {best_val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch} (patience exceeded)")
                break
        
        print("\n" + "=" * 60)
        print(f"Training complete! Best validation loss: {best_val_loss:.6f}")
        
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        
    def plot_training_history(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("Saved training history plot to training_history.png")
        
    def test_model(self, test_samples=10):
        """Test the model with sample inputs"""
        print("\n" + "=" * 60)
        print("Testing trained model with sample emotional states:")
        print("=" * 60)
        
        self.model.eval()
        
        test_states = [
            ("Angry", [-0.8, 0.85, 0.8]),
            ("Fearful", [-0.85, 0.9, 0.15]),
            ("Joyful", [0.9, 0.75, 0.75]),
            ("Sad", [-0.75, 0.2, 0.2]),
            ("Calm", [0.1, 0.2, 0.5]),
            ("Excited", [0.7, 0.9, 0.6]),
            ("Frustrated", [-0.5, 0.6, 0.3]),
            ("Content", [0.5, 0.3, 0.6]),
            ("Anxious", [-0.3, 0.7, 0.3]),
            ("Confident", [0.3, 0.5, 0.8])
        ]
        
        with torch.no_grad():
            for name, vad in test_states:
                inputs = torch.tensor(vad, dtype=torch.float32).to(self.device)
                outputs = self.model(inputs)
                
                # Get top 5 components
                values, indices = torch.topk(outputs, 5)
                
                print(f"\n{name} {vad}:")
                for i, (idx, val) in enumerate(zip(indices, values)):
                    component = MoodComponents.get_component_name(idx.item())
                    intensity = val.item()
                    
                    if intensity > 0.7:
                        level = "very"
                    elif intensity > 0.5:
                        level = "notably"
                    elif intensity > 0.3:
                        level = "moderately"
                    else:
                        level = "slightly"
                    
                    print(f"  {level} {component} ({intensity:.2f})")


def main():
    """
    Main training pipeline
    """
    print("=" * 60)
    print("MULTI-MOOD NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # Generate dataset
    print("\n1. Generating training data...")
    generator = TrainingDataGenerator()
    inputs, targets = generator.generate_complete_dataset(samples_per_emotion=50)
    
    print(f"   Generated {len(inputs)} total samples")
    print(f"   Input shape: {inputs.shape}")
    print(f"   Target shape: {targets.shape}")
    
    # Save full dataset
    generator.save_dataset(inputs, targets, 'mood_training_dataset.pth')
    
    # Create data loaders
    dataset = TensorDataset(inputs, targets)
    
    # Split into train/val/test (70/15/15)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize trainer
    print("\n2. Initializing trainer...")
    trainer = MoodNetworkTrainer(learning_rate=0.001, batch_size=64)
    
    # Print model info
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # Train model
    print("\n3. Training model...")
    trainer.train(train_loader, val_loader, epochs=150)
    
    # Save final model
    trainer.save_model('final_mood_model.pth')
    print("\n4. Saved models:")
    print("   - best_mood_model.pth (best validation)")
    print("   - final_mood_model.pth (final state)")
    
    # Plot history
    trainer.plot_training_history()
    
    # Test model
    trainer.test_model()
    
    # Final test set evaluation
    print("\n5. Final test set evaluation:")
    trainer.model.eval()
    test_loss = 0
    test_batches = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(trainer.device)
            targets = targets.to(trainer.device)
            
            outputs = trainer.model(inputs)
            loss = trainer.criterion(outputs, targets)
            
            test_loss += loss.item()
            test_batches += 1
    
    avg_test_loss = test_loss / test_batches
    print(f"   Test set loss: {avg_test_loss:.6f}")
    
    print("\n" + "=" * 60)
    print("Training complete! Model ready for integration.")
    print("=" * 60)


if __name__ == "__main__":
    main()