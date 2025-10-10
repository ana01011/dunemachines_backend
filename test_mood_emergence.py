"""
Test mood emergence from random positions in 3D space
"""

import torch
import numpy as np
import sys
sys.path.append('/root/openhermes_backend')

from app.neurochemistry.multi_mood_network import MoodInterpreter

def test_random_positions():
    """Test random positions to see natural emergence"""
    
    print("=" * 60)
    print("TESTING NATURAL MOOD EMERGENCE")
    print("=" * 60)
    
    interpreter = MoodInterpreter(model_path='best_mood_model.pth', threshold=0.25)
    
    # Generate random positions
    np.random.seed(42)  # For reproducibility
    
    for i in range(10):
        # Random VAD position
        v = np.random.uniform(-1, 1)
        a = np.random.uniform(0, 1)
        d = np.random.uniform(0, 1)
        
        mood_state = interpreter.get_mood_state(v, a, d)
        
        print(f"\nRandom position {i+1}:")
        print(f"  VAD: [V={v:+.3f} A={a:.3f} D={d:.3f}]")
        print(f"  Emerged mood: {mood_state['description']}")
        
        # Show top 3 components with intensities
        for j, component in enumerate(mood_state['components'][:3]):
            print(f"    {j+1}. {component['component']}: {component['intensity']:.2f}")

if __name__ == "__main__":
    test_random_positions()
