"""
Test script for the 7D neurochemistry system
"""

import sys
sys.path.append('/root/openhermes_backend')

from app.neurochemistry.interface import NeurochemicalSystem
import json

def test_basic():
    """Test basic functionality"""
    print("="*50)
    print("NEUROCHEMISTRY V3 TEST")
    print("="*50)
    
    # Create system
    system = NeurochemicalSystem(user_id="test_user")
    
    # Get initial state
    print("\n1. INITIAL STATE:")
    summary = system.get_state_summary()
    print(json.dumps(summary, indent=2))
    
    # Test positive message
    print("\n2. PROCESSING POSITIVE MESSAGE:")
    response = system.process_message("This is wonderful! I'm so happy and excited!")
    print(f"Mood: {response['mood']}")
    print(f"Dopamine: {response['hormones']['dopamine']:.2f}")
    print(f"Serotonin: {response['hormones']['serotonin']:.2f}")
    print(f"Efficiency: {response['efficiency']:.2f}")
    
    # Test stress message
    print("\n3. PROCESSING STRESS MESSAGE:")
    response = system.process_message("URGENT! This is an emergency, need help NOW!")
    print(f"Mood: {response['mood']}")
    print(f"Cortisol: {response['hormones']['cortisol']:.2f}")
    print(f"Adrenaline: {response['hormones']['adrenaline']:.2f}")
    print(f"Norepinephrine: {response['hormones']['norepinephrine']:.2f}")
    
    # Test social message
    print("\n4. PROCESSING SOCIAL MESSAGE:")
    response = system.process_message("Let's work together as a team, we can do this!")
    print(f"Mood: {response['mood']}")
    print(f"Oxytocin: {response['hormones']['oxytocin']:.2f}")
    print(f"Serotonin: {response['hormones']['serotonin']:.2f}")
    
    # Show cost breakdown
    print("\n5. COST ANALYSIS:")
    print(json.dumps(response['cost'], indent=2))
    
    # Show seeking behavior
    print("\n6. SEEKING BEHAVIOR:")
    print(json.dumps(response['seeking'], indent=2))
    
    # Show suggestions
    print("\n7. OPTIMAL ACTIONS:")
    print(json.dumps(response['suggestions'], indent=2))
    
    # Simulate time passage
    print("\n8. SIMULATING 10 SECONDS OF REST:")
    system.simulate_time_passage(10, rest=True)
    summary = system.get_state_summary()
    print(json.dumps(summary, indent=2))
    
    # Check diagnostics
    print("\n9. SYSTEM DIAGNOSTICS:")
    diagnostics = system.get_diagnostics()
    print(f"Updates: {diagnostics['update_count']}")
    print(f"Time elapsed: {diagnostics['time_elapsed']:.2f}s")
    print(f"State valid: {diagnostics['state_valid']}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_basic()
