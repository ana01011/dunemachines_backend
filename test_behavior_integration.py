"""
Test: Neurochemistry → Behavioral Prompt → OMNIUS Integration
"""
 
import sys
sys.path.insert(0, '/root/openhermes_backend')
 
from app.neurochemistry.interface import NeurochemicalSystem
from app.neurochemistry.behavioral_prompt import get_behavior_prompt, hormones_to_behavior
 
def test_full_flow():
    print("=" * 60)
    print("Testing Neurochemistry → Behavior → OMNIUS Flow")
    print("=" * 60)
    
    # Create neurochemistry system
    neuro = NeurochemicalSystem(user_id="test")
    
    # Test messages with different emotional content
    test_messages = [
        "I'm so excited! This project is amazing!",
        "This stupid code won't work and I've been debugging for hours!",
        "Can you help me understand how neural networks work?",
        "I feel stressed about the deadline tomorrow",
        "Thank you so much, you're incredibly helpful!"
    ]
    
    for msg in test_messages:
        print(f"\n{'─' * 60}")
        print(f"USER: {msg}")
        print(f"{'─' * 60}")
        
        # Process through neurochemistry
        result = neuro.process_message(msg)
        
        # Get hormone values
        hormones = result['hormones']
        
        # Convert to behavioral state
        behavior = hormones_to_behavior(hormones)
        prompt = get_behavior_prompt(hormones)
        
        print(f"\n7D Hormones:")
        for name, val in hormones.items():
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"  {name:15} [{bar}] {val:.2f}")
        
        print(f"\nBehavioral State:")
        for name, val in behavior.to_dict().items():
            bar = "█" * int(val / 5) + "░" * (20 - int(val / 5))
            print(f"  {name:12} [{bar}] {val}%")
        
        print(f"\nOMNIUS Prompt Injection:")
        print(f"  {prompt}")
        
        print(f"\nMood: {result['mood']}")
 
if __name__ == "__main__":
    test_full_flow()
