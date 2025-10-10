"""
Train final version with explicit emotional differentiation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys

sys.path.append('/root/openhermes_backend')

from app.neurochemistry.scalable_hormone_network import create_network

def create_focused_training_data():
    """Create training data with clear emotional distinctions"""
    
    inputs = []
    targets = []
    
    # Happy: High warm, curious, energized
    for _ in range(50):
        d = np.random.uniform(70, 85)
        c = np.random.uniform(15, 30)
        a = np.random.uniform(25, 40)
        s = np.random.uniform(70, 85)
        o = np.random.uniform(60, 75)
        
        mood = np.zeros(30)
        mood[4] = 0.9   # warm - HIGH
        mood[7] = 0.8   # curious - HIGH
        mood[1] = 0.7   # energized
        mood[0] = 0.6   # attentive
        mood[2] = 0.1   # tense - LOW
        mood[5] = 0.1   # restless - LOW
        
        inputs.append([d, c, a, s, o])
        targets.append(mood)
    
    # Sad: Low energy, high reflective
    for _ in range(50):
        d = np.random.uniform(15, 30)
        c = np.random.uniform(40, 55)
        a = np.random.uniform(10, 25)
        s = np.random.uniform(20, 35)
        o = np.random.uniform(25, 40)
        
        mood = np.zeros(30)
        mood[10] = 0.8  # reflective - HIGH
        mood[1] = 0.2   # energized - LOW
        mood[0] = 0.3   # attentive - LOW
        mood[2] = 0.4   # tense - MEDIUM
        mood[4] = 0.3   # warm - LOW
        
        inputs.append([d, c, a, s, o])
        targets.append(mood)
    
    # Angry: High tense, assertive, restless
    for _ in range(50):
        d = np.random.uniform(55, 70)
        c = np.random.uniform(70, 85)
        a = np.random.uniform(75, 90)
        s = np.random.uniform(30, 45)
        o = np.random.uniform(10, 25)
        
        mood = np.zeros(30)
        mood[2] = 0.9   # tense - HIGH
        mood[14] = 0.8  # assertive - HIGH
        mood[5] = 0.8   # restless - HIGH
        mood[0] = 0.9   # attentive - HIGH
        mood[1] = 0.8   # energized - HIGH
        mood[4] = 0.1   # warm - LOW
        
        inputs.append([d, c, a, s, o])
        targets.append(mood)
    
    # Anxious: High cautious, tense, protective
    for _ in range(50):
        d = np.random.uniform(25, 40)
        c = np.random.uniform(65, 80)
        a = np.random.uniform(70, 85)
        s = np.random.uniform(35, 50)
        o = np.random.uniform(25, 40)
        
        mood = np.zeros(30)
        mood[9] = 0.9   # cautious - HIGH
        mood[2] = 0.8   # tense - HIGH
        mood[6] = 0.8   # protective - HIGH
        mood[0] = 0.9   # attentive - HIGH
        mood[5] = 0.7   # restless - HIGH
        mood[4] = 0.2   # warm - LOW
        
        inputs.append([d, c, a, s, o])
        targets.append(mood)
    
    # Calm: Low arousal, high methodical
    for _ in range(50):
        d = np.random.uniform(45, 55)
        c = np.random.uniform(25, 35)
        a = np.random.uniform(15, 25)
        s = np.random.uniform(65, 80)
        o = np.random.uniform(50, 65)
        
        mood = np.zeros(30)
        mood[12] = 0.7  # methodical - HIGH
        mood[10] = 0.7  # reflective - HIGH
        mood[4] = 0.7   # warm - HIGH
        mood[1] = 0.3   # energized - LOW
        mood[2] = 0.2   # tense - LOW
        mood[5] = 0.1   # restless - LOW
        
        inputs.append([d, c, a, s, o])
        targets.append(mood)
    
    return np.array(inputs), np.array(targets)

def train_final():
    print("="*60)
    print("TRAINING FINAL NETWORK WITH CLEAR DISTINCTIONS")
    print("="*60)
    
    # Create focused training data
    inputs, targets = create_focused_training_data()
    
    # Normalize inputs
    inputs = inputs / 100.0
    
    # Convert to tensors
    inputs_t = torch.tensor(inputs, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.float32)
    
    # Create dataset
    dataset = TensorDataset(inputs_t, targets_t)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create network
    model = create_network('small', input_dim=5, output_dim=30, dropout_rate=0.2)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"\nTraining with {len(dataset)} samples...")
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        
        for hormones, moods in loader:
            optimizer.zero_grad()
            predicted = model(hormones)
            loss = criterion(predicted, moods)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), 'hormone_model_final.pth')
    print("\n✅ Training complete! Saved to hormone_model_final.pth")

if __name__ == "__main__":
    train_final()
