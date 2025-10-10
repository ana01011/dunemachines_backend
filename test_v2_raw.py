"""
Check raw outputs of retrained network
"""

import torch
import numpy as np
import sys

sys.path.append('/root/openhermes_backend')

from app.neurochemistry.scalable_hormone_network import create_network, ExtendedMoodComponents

def test_raw():
    print("="*60)
    print("RAW OUTPUTS - RETRAINED NETWORK V2")
    print("="*60)
    
    model = create_network('small', input_dim=5, output_dim=30, dropout_rate=0.3)
    model.load_state_dict(torch.load('hormone_model_v2.pth'))
    model.eval()
    
    test_cases = [
        ("Happy", [75, 20, 30, 80, 65]),
        ("Sad", [20, 45, 15, 25, 30]),
    ]
    
    with torch.no_grad():
        for name, hormones in test_cases:
            h_tensor = torch.tensor(hormones, dtype=torch.float32) / 100.0
            moods = model(h_tensor)
            
            if len(moods.shape) > 1:
                moods = moods[0]
            
            moods_np = moods.numpy()
            
            print(f"\n{name}: D={hormones[0]} C={hormones[1]} A={hormones[2]}")
            print(f"Range: [{moods_np.min():.3f}, {moods_np.max():.3f}]")
            print("Top 5 components:")
            
            indices = np.argsort(moods_np)[::-1][:5]
            for idx in indices:
                print(f"  {ExtendedMoodComponents.get_component_name(idx):20s}: {moods_np[idx]:.3f}")

if __name__ == "__main__":
    test_raw()
