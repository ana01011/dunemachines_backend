"""
Debug the network outputs to see raw values
"""

import torch
import numpy as np
import sys

sys.path.append('/root/openhermes_backend')

from app.neurochemistry.scalable_hormone_network import create_network, ExtendedMoodComponents

def debug_network():
    print("="*60)
    print("DEBUGGING NETWORK OUTPUT")
    print("="*60)
    
    # Load model
    model = create_network('medium', input_dim=5, output_dim=30)
    model.load_state_dict(torch.load('best_hormone_model.pth'))
    model.eval()
    
    # Test specific cases
    test_cases = [
        ("Happy (should be warm, curious)", [75, 20, 30, 80, 65]),
        ("Sad (should be low energy)", [20, 45, 15, 25, 30]),
        ("Angry (should be tense, assertive)", [60, 75, 80, 35, 15]),
        ("Calm (should be low arousal)", [50, 30, 20, 70, 55]),
    ]
    
    with torch.no_grad():
        for name, hormones in test_cases:
            h_tensor = torch.tensor(hormones, dtype=torch.float32)
            moods = model(h_tensor)
            
            if len(moods.shape) > 1:
                moods = moods[0]
            
            moods_np = moods.numpy()
            
            print(f"\n{name}")
            print(f"Hormones: D={hormones[0]} C={hormones[1]} A={hormones[2]} S={hormones[3]} O={hormones[4]}")
            print("\nTop 10 mood components (raw values):")
            
            # Sort by value
            indices = np.argsort(moods_np)[::-1][:10]
            
            for idx in indices:
                component = ExtendedMoodComponents.get_component_name(idx)
                value = moods_np[idx]
                print(f"  {component:20s}: {value:.3f}")
            
            print("\nAll values range: [{:.3f}, {:.3f}]".format(moods_np.min(), moods_np.max()))

if __name__ == "__main__":
    debug_network()
