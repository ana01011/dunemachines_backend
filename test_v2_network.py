"""
Test the retrained network
"""

import torch
import numpy as np
import sys

sys.path.append('/root/openhermes_backend')

from app.neurochemistry.scalable_hormone_network import create_network, ExtendedMoodComponents

def test_v2():
    print("="*60)
    print("TESTING RETRAINED NETWORK V2")
    print("="*60)
    
    # Load with small architecture
    model = create_network('small', input_dim=5, output_dim=30, dropout_rate=0.3)
    model.load_state_dict(torch.load('hormone_model_v2.pth'))
    model.eval()
    
    test_cases = [
        ("😊 Happy", [75, 20, 30, 80, 65]),
        ("😢 Sad", [20, 45, 15, 25, 30]),
        ("😡 Angry", [60, 75, 80, 35, 15]),
        ("😰 Anxious", [30, 70, 75, 40, 30]),
        ("😌 Calm", [50, 30, 20, 70, 55]),
        ("🤖 Balanced", [50, 50, 50, 50, 50]),
    ]
    
    with torch.no_grad():
        for name, hormones in test_cases:
            # Normalize inputs
            h_tensor = torch.tensor(hormones, dtype=torch.float32) / 100.0
            moods = model(h_tensor)
            
            if len(moods.shape) > 1:
                moods = moods[0]
            
            mood_state = ExtendedMoodComponents.describe_mood_state(moods.numpy(), threshold=0.3)
            
            print(f"\n{name}: D={hormones[0]} C={hormones[1]} A={hormones[2]}")
            print(f"  → {mood_state['description']}")

if __name__ == "__main__":
    test_v2()
