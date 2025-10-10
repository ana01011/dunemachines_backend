"""
Final integration test with the working neural network
"""

import torch
import sys
sys.path.append('/root/openhermes_backend')

from app.neurochemistry.integrated_system import orchestrator
from app.neurochemistry.scalable_hormone_network import create_network, ExtendedMoodComponents

def test_complete_system():
    print("="*60)
    print("COMPLETE NEUROCHEMISTRY → HORMONE → MOOD PIPELINE")
    print("="*60)
    
    # Load the final trained model
    model = create_network('small', input_dim=5, output_dim=30, dropout_rate=0.2)
    model.load_state_dict(torch.load('hormone_model_final.pth'))
    model.eval()
    
    user_id = "test_user"
    
    messages = [
        ("I'm so happy! This finally works!", "Should be warm + curious"),
        ("I'm really frustrated with this bug", "Should be tense + restless"),
        ("I feel sad and alone", "Should be low energy + reflective"),
        ("Everything is peaceful", "Should be calm + warm"),
        ("URGENT HELP NEEDED!", "Should be high attentive + tense"),
    ]
    
    for message, expected in messages:
        # Get hormones from neurochemistry
        result = orchestrator.process_user_message(user_id, message)
        h = result['hormones']
        hormones = [h['dopamine'], h['cortisol'], h['adrenaline'], h['serotonin'], h['oxytocin']]
        
        # Get mood from neural network
        h_tensor = torch.tensor(hormones, dtype=torch.float32) / 100.0
        with torch.no_grad():
            moods = model(h_tensor)
            if len(moods.shape) > 1:
                moods = moods[0]
        
        # Get top moods
        moods_np = moods.numpy()
        indices = np.argsort(moods_np)[::-1][:4]
        mood_str = []
        for idx in indices:
            if moods_np[idx] > 0.3:
                comp = ExtendedMoodComponents.get_component_name(idx)
                mood_str.append(f"{comp}({moods_np[idx]:.2f})")
        
        print(f"\n📝 '{message}'")
        print(f"   Expected: {expected}")
        print(f"   Hormones: D={hormones[0]:.0f} C={hormones[1]:.0f} A={hormones[2]:.0f}")
        print(f"   Got mood: {' + '.join(mood_str[:3])}")

if __name__ == "__main__":
    import numpy as np
    test_complete_system()
