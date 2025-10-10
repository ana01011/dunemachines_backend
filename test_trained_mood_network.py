"""
Test the trained Multi-Mood Neural Network
"""

import torch
import sys
sys.path.append('/root/openhermes_backend')

from app.neurochemistry.multi_mood_network import MultiMoodEmergenceNetwork, MoodComponents, MoodInterpreter

def test_network():
    """Test the trained network with various emotional states"""
    
    print("=" * 60)
    print("TESTING TRAINED MOOD NETWORK")
    print("=" * 60)
    
    # Load the trained model
    interpreter = MoodInterpreter(model_path='best_mood_model.pth', threshold=0.25)
    
    # Test cases covering emotional spectrum
    test_cases = [
        ("Angry person", -0.8, 0.85, 0.8),
        ("Fearful person", -0.85, 0.9, 0.15),
        ("Happy person", 0.9, 0.75, 0.75),
        ("Sad person", -0.75, 0.2, 0.2),
        ("Calm person", 0.1, 0.2, 0.5),
        ("Frustrated user", -0.5, 0.6, 0.3),
        ("Excited child", 0.7, 0.9, 0.6),
        ("Anxious student", -0.3, 0.7, 0.3),
        ("Confident leader", 0.3, 0.5, 0.8),
        ("Neutral state", 0.0, 0.3, 0.5),
    ]
    
    for name, v, a, d in test_cases:
        mood_state = interpreter.get_mood_state(v, a, d)
        
        print(f"\n{name} [V={v:+.2f} A={a:.2f} D={d:.2f}]")
        print(f"  → {mood_state['description']}")
        print(f"  Primary: {mood_state['primary']} ({mood_state['primary_intensity']:.2f})")
        print(f"  Active components: {mood_state['num_active']}")
        
        # Show behavioral hints
        hints = interpreter.get_behavioral_hints(mood_state)
        if hints:
            print(f"  Behavioral hints: {', '.join(hints)}")
    
    # Test smooth transitions
    print("\n" + "=" * 60)
    print("TESTING SMOOTH TRANSITIONS")
    print("=" * 60)
    
    print("\nTransition from calm to angry:")
    for step in range(5):
        t = step / 4.0  # 0 to 1
        # Interpolate from calm to angry
        v = 0.1 * (1 - t) + (-0.8) * t
        a = 0.2 * (1 - t) + 0.85 * t  
        d = 0.5 * (1 - t) + 0.8 * t
        
        mood_state = interpreter.get_mood_state(v, a, d)
        print(f"  Step {step}: {mood_state['description']}")

if __name__ == "__main__":
    test_network()
