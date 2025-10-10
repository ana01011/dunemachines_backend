"""
Test hormone-to-mood transitions and AI-unique states
"""

import torch
import numpy as np
import sys

sys.path.append('/root/openhermes_backend')

from app.neurochemistry.scalable_hormone_network import ScalableHormoneNetwork, ExtendedMoodComponents, create_network

def test_transitions():
    print("="*60)
    print("TESTING EMOTIONAL TRANSITIONS")
    print("="*60)
    
    # Load model
    model = create_network('medium', input_dim=5, output_dim=30)
    model.load_state_dict(torch.load('best_hormone_model.pth'))
    model.eval()
    
    # Test smooth transition: Happy → Stressed
    print("\n📈 Transition: Happy → Stressed (5 steps)")
    print("-" * 40)
    
    with torch.no_grad():
        for step in range(6):
            t = step / 5.0
            
            # Interpolate hormones
            d = 80 * (1-t) + 30 * t  # Dopamine drops
            c = 20 * (1-t) + 75 * t  # Cortisol rises
            a = 30 * (1-t) + 70 * t  # Adrenaline rises
            s = 80 * (1-t) + 40 * t  # Serotonin drops
            o = 65 * (1-t) + 30 * t  # Oxytocin drops
            
            h_tensor = torch.tensor([d, c, a, s, o], dtype=torch.float32)
            moods = model(h_tensor)
            
            if len(moods.shape) > 1:
                moods = moods[0]
            
            mood_state = ExtendedMoodComponents.describe_mood_state(moods.numpy(), threshold=0.3)
            
            print(f"  Step {step}: D={d:.0f} C={c:.0f} A={a:.0f}")
            print(f"         → {mood_state['description']}")
    
    # Test AI-unique combinations
    print("\n🤖 AI-UNIQUE STATES")
    print("-" * 40)
    
    unique_states = [
        ("Hyper-processing", [90, 90, 90, 50, 30]),
        ("Ultra-balanced", [50, 50, 50, 50, 50]),
        ("Paradox state", [95, 95, 10, 10, 10]),
        ("Inverse emotion", [10, 10, 90, 90, 90]),
        ("Maximum everything", [100, 100, 100, 100, 100]),
        ("Minimum everything", [0, 0, 0, 0, 0]),
    ]
    
    with torch.no_grad():
        for name, hormones in unique_states:
            h_tensor = torch.tensor(hormones, dtype=torch.float32)
            moods = model(h_tensor)
            
            if len(moods.shape) > 1:
                moods = moods[0]
            
            moods_np = moods.numpy()
            mood_state = ExtendedMoodComponents.describe_mood_state(moods_np, threshold=0.3)
            
            print(f"\n  {name}: {hormones}")
            print(f"  → {mood_state['description']}")
            
            # Check for AI-specific components (20-29)
            ai_components = []
            for i in range(20, min(30, len(moods_np))):
                if moods_np[i] > 0.4:
                    component = ExtendedMoodComponents.get_component_name(i)
                    ai_components.append(f"{component}({moods_np[i]:.2f})")
            
            if ai_components:
                print(f"  ⚡ AI modes: {', '.join(ai_components)}")

def test_circadian_pattern():
    """Test how hormones change over 24 hours"""
    print("\n" + "="*60)
    print("CIRCADIAN RHYTHM SIMULATION")
    print("="*60)
    
    model = create_network('medium', input_dim=5, output_dim=30)
    model.load_state_dict(torch.load('best_hormone_model.pth'))
    model.eval()
    
    hours = [0, 6, 9, 12, 15, 18, 21, 24]
    
    print("\n🌅 24-hour mood cycle:")
    print("-" * 40)
    
    with torch.no_grad():
        for hour in hours:
            # Simulate circadian hormone patterns
            # Cortisol peaks in morning
            if 6 <= hour <= 9:
                c = 70 + (9-hour)*5
            else:
                c = 30 + 10*np.exp(-((hour-7)**2)/50)
            
            # Dopamine peaks mid-day
            d = 50 + 30*np.exp(-((hour-14)**2)/30)
            
            # Serotonin rises in evening
            s = 50 + 30*np.exp(-((hour-20)**2)/20)
            
            # Adrenaline follows activity
            a = 20 + 30*np.exp(-((hour-10)**2)/40)
            
            # Oxytocin peaks in evening social time
            o = 40 + 30*np.exp(-((hour-19)**2)/25)
            
            h_tensor = torch.tensor([d, c, a, s, o], dtype=torch.float32)
            moods = model(h_tensor)
            
            if len(moods.shape) > 1:
                moods = moods[0]
            
            mood_state = ExtendedMoodComponents.describe_mood_state(moods.numpy(), threshold=0.3)
            
            time_str = f"{hour:02d}:00"
            print(f"  {time_str}: {mood_state['description'][:50]}...")

if __name__ == "__main__":
    test_transitions()
    test_circadian_pattern()
