"""
Test the final network
"""

import torch
import numpy as np
import sys

sys.path.append('/root/openhermes_backend')

from app.neurochemistry.scalable_hormone_network import create_network, ExtendedMoodComponents

def test_final():
    print("="*60)
    print("TESTING FINAL NETWORK")
    print("="*60)
    
    model = create_network('small', input_dim=5, output_dim=30, dropout_rate=0.2)
    model.load_state_dict(torch.load('hormone_model_final.pth'))
    model.eval()
    
    test_cases = [
        ("😊 Happy", [75, 20, 30, 80, 65]),
        ("😢 Sad", [20, 45, 15, 25, 30]),
        ("😡 Angry", [60, 75, 80, 35, 15]),
        ("😰 Anxious", [30, 70, 75, 40, 30]),
        ("😌 Calm", [50, 30, 20, 70, 55]),
    ]
    
    with torch.no_grad():
        for name, hormones in test_cases:
            h_tensor = torch.tensor(hormones, dtype=torch.float32) / 100.0
            moods = model(h_tensor)
            
            if len(moods.shape) > 1:
                moods = moods[0]
            
            moods_np = moods.numpy()
            
            print(f"\n{name}: D={hormones[0]} C={hormones[1]} A={hormones[2]}")
            
            # Show top 5 components with values
            indices = np.argsort(moods_np)[::-1][:5]
            components = []
            for idx in indices:
                comp_name = ExtendedMoodComponents.get_component_name(idx)
                value = moods_np[idx]
                if value > 0.3:
                    components.append(f"{comp_name}({value:.2f})")
            
            print(f"  → {' + '.join(components)}")

if __name__ == "__main__":
    test_final()
